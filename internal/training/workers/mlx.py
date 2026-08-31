# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import traceback

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten


PROTOCOL_SCHEMA = 1
WORKER_REVISION = "builtin-mlx-worker-schema-1-r8"


def emit(kind, **payload):
    frame = {"kind": kind, "schema": PROTOCOL_SCHEMA}
    frame.update(payload)
    print(json.dumps(frame, separators=(",", ":")), flush=True)


# WALDO_GPU_THROTTLE=0.25 caps training to ~25% of wall-clock time; GPU alternates between
# full-speed and idle. Applies only to the MLX worker.
def read_gpu_throttle():
    try:
        value = float(os.environ.get("WALDO_GPU_THROTTLE") or 1)
    except ValueError:
        value = math.nan
    if not 0.01 <= value <= 1:
        emit("event", event={"kind": "log", "message": "WALDO_GPU_THROTTLE must be a decimal between 0.01 and 1.0; throttling disabled"})
        return 1
    return value
GPU_THROTTLE = read_gpu_throttle()


def artifact(path, logical_path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return {"path": logical_path, "sha256": digest.hexdigest(), "bytes": size}


def write_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def commit_directory(temporary, destination):
    for root, _, files in os.walk(temporary):
        for name in files:
            descriptor = os.open(os.path.join(root, name), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    descriptor = os.open(os.path.dirname(destination), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Attention(nn.Module):
    def __init__(self, hidden, heads, kv_heads):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        kv_width = self.head_dim * kv_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_width, bias=False)
        self.v_proj = nn.Linear(hidden, kv_width, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=10000)

    def __call__(self, value):
        batch, length, _ = value.shape
        query = self.q_proj(value).reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        key = self.k_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        val = self.v_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        query = self.rope(query)
        key = self.rope(key)
        attended = mx.fast.scaled_dot_product_attention(
            query, key, val, scale=self.head_dim ** -0.5, mask="causal"
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, value):
        return self.down(nn.silu(self.gate(value)) * self.up(value))


class DecoderBlock(nn.Module):
    def __init__(self, hidden, intermediate, heads, kv_heads, dropout):
        super().__init__()
        self.attention_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.attention = Attention(hidden, heads, kv_heads)
        self.ffn_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.feed_forward = FeedForward(hidden, intermediate)
        self.residual_dropout = nn.Dropout(dropout)

    def __call__(self, value):
        value = value + self.residual_dropout(self.attention(self.attention_norm(value)))
        return value + self.residual_dropout(self.feed_forward(self.ffn_norm(value)))


class DecoderLM(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        vocabulary = architecture["vocabulary_size"]
        hidden = architecture["hidden_size"]
        self.tie_embeddings = architecture["tie_embeddings"]
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.layers = [
            DecoderBlock(
                hidden,
                architecture["intermediate_size"],
                architecture["attention_heads"],
                architecture["key_value_heads"],
                architecture.get("dropout", 0.0),
            )
            for _ in range(architecture["layers"])
        ]
        self.norm = nn.RMSNorm(hidden, eps=1e-5)
        if not self.tie_embeddings:
            self.output = nn.Linear(hidden, vocabulary, bias=False)

    def __call__(self, tokens):
        value = self.embedding(tokens)
        for layer in self.layers:
            value = layer(value)
        value = self.norm(value)
        if self.tie_embeddings:
            return self.embedding.as_linear(value)
        return self.output(value)


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def encode(self, text):
        return [byte + 3 for byte in text.encode("utf-8")] + [self.eos_id]


class FramingTokenizer:
    def __init__(self, specification):
        self.name = specification["name"]
        self.revision = specification["revision"]
        self.pad_id = int(specification["pad_id"])
        self.bos_id = int(specification["bos_id"])
        self.eos_id = int(specification["eos_id"])

    def encode_record(self, record):
        if "tokens" in record:
            return [int(token) for token in record["tokens"]] + [self.eos_id]
        if self.name == "byte":
            return [byte + 3 for byte in record["text"].encode("utf-8")] + [self.eos_id]
        raise ValueError(f"record is missing pre-tokenized IDs for {self.name}")


class Trainer:
    def __init__(self, begin, artifact_directory, artifact_prefix):
        self.begin = begin
        self.architecture = begin["architecture"]
        self.parameters = begin["parameters"]
        self.artifact_directory = artifact_directory
        self.artifact_prefix = artifact_prefix.replace(os.sep, "/").strip("/")
        self.sequence_length = self.parameters["sequence_length"]
        self.batch_size = self.parameters["batch_size"]
        self.target_steps = self.parameters["steps"]
        self.step_number = 0
        self.replay_steps = 0
        self.consumed_tokens = 0
        self.token_buffer = []
        self.loss_buffer = []
        self.corpus_buffer = []
        self.batch = []
        self.consumed_by_corpus = {}
        self.checkpoints = []
        self.evaluations = []
        self.evaluation_sequences = []
        self.evaluation_record_count = 0
        self.evaluation_token_targets = 0
        self.final_loss = None
        self.started = time.perf_counter()
        self.last_report = self.started
        self.last_report_tokens = 0

        tokenizer = begin["tokenizer"]
        architecture_tokenizer = self.architecture["tokenizer"]
        if tokenizer["name"] != architecture_tokenizer["name"] or tokenizer["revision"] != architecture_tokenizer["revision"] or tokenizer["vocabulary_size"] != self.architecture["vocabulary_size"]:
            raise ValueError("MLX worker tokenizer framing does not match the architecture")
        self.tokenizer = FramingTokenizer(tokenizer)
        mx.random.seed(self.parameters["seed"])
        self.model = DecoderLM(self.architecture)
        self.model.train()
        self.initialization = begin.get("initialization")
        if self.initialization is not None:
            self.model.load_weights(self.initialization["path"])
        dtype_name = self.architecture["parameter_dtype"]
        dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype_name]
        if dtype != mx.float32:
            self.model.apply(lambda value: value.astype(dtype))
        mx.eval(self.model.parameters())
        optimizer_parameters = self.parameters["optimizer"]
        self.optimizer = optim.AdamW(
            learning_rate=self.parameters["learning_rate"],
            betas=(optimizer_parameters["beta1"], optimizer_parameters["beta2"]),
            eps=optimizer_parameters["epsilon"],
            weight_decay=optimizer_parameters["weight_decay"],
        )
        self.resume = begin.get("resume")
        if self.resume is not None:
            self.restore_checkpoint(self.resume)
        self.loss_and_grad = nn.value_and_grad(self.model, self.loss)

    def logical(self, name):
        return "/".join(part for part in (self.artifact_prefix, name) if part)

    def learning_rate(self, step):
        schedule = self.parameters["schedule"]
        base = self.parameters["learning_rate"]
        warmup = schedule["warmup_steps"]
        if warmup > 0 and step <= warmup:
            return base * step / warmup
        decay_steps = max(1, self.target_steps - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
        ratio = schedule["minimum_rate_ratio"] + (1.0 - schedule["minimum_rate_ratio"]) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return base * ratio

    def loss(self, model, inputs, targets, mask):
        logits = model(inputs)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        return (losses * mask).sum() / mask.sum()

    def add_record(self, record):
        if self.step_number >= self.target_steps:
            return
        encoded = self.tokenizer.encode_record(record)
        loss_mask = record.get("loss_mask", [True] * len(encoded))
        if len(loss_mask) != len(encoded):
            raise ValueError("record loss_mask does not match framed token count")
        self.token_buffer.extend(encoded)
        self.loss_buffer.extend(loss_mask)
        self.corpus_buffer.extend([record.get("corpus", "")] * len(encoded))
        window = self.sequence_length + 1
        while len(self.token_buffer) >= window and self.step_number < self.target_steps:
            self.add_sequence(self.token_buffer[:window], self.loss_buffer[1:window], self.corpus_buffer[1:window])
            del self.token_buffer[: self.sequence_length]
            del self.loss_buffer[: self.sequence_length]
            del self.corpus_buffer[: self.sequence_length]

    def add_evaluation_record(self, record):
        self.evaluation_record_count += 1
        tokens = self.tokenizer.encode_record(record)
        loss_mask = record.get("loss_mask", [True] * len(tokens))
        if len(loss_mask) != len(tokens):
            raise ValueError("evaluation record loss_mask does not match framed token count")
        window = self.sequence_length + 1
        while len(tokens) > 1:
            piece = tokens[:window]
            target_mask = loss_mask[1:len(piece)]
            padded = piece + [self.tokenizer.pad_id] * (window - len(piece))
            mask = [float(value) for value in target_mask] + [0.0] * (self.sequence_length - len(target_mask))
            self.evaluation_sequences.append((padded, mask))
            self.evaluation_token_targets += sum(target_mask)
            del tokens[: self.sequence_length]
            del loss_mask[: self.sequence_length]

    def add_sequence(self, tokens, target_mask, target_corpora=None):
        if not any(target_mask) or self.step_number >= self.target_steps:
            return
        window = self.sequence_length + 1
        padded = tokens + [self.tokenizer.pad_id] * (window - len(tokens))
        mask = [float(value) for value in target_mask] + [0.0] * (self.sequence_length - len(target_mask))
        corpus_counts = {}
        for corpus, supervised in zip(target_corpora or [], target_mask):
            if supervised:
                corpus_counts[corpus] = corpus_counts.get(corpus, 0) + 1
        self.batch.append((padded, mask, corpus_counts))
        if len(self.batch) >= self.batch_size:
            self.train_batch()

    def train_batch(self):
        if not self.batch or self.step_number >= self.target_steps:
            self.batch = []
            return
        if self.replay_steps > 0:
            self.replay_steps -= 1
            self.batch = []
            return
        step_started = time.perf_counter()
        tokens = mx.array([item[0] for item in self.batch], dtype=mx.int32)
        mask = mx.array([item[1] for item in self.batch], dtype=mx.float32)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        next_step = self.step_number + 1
        current_learning_rate = self.learning_rate(next_step)
        self.optimizer.learning_rate = current_learning_rate
        mx.random.seed(self.parameters["seed"] ^ next_step)
        loss, gradients = self.loss_and_grad(self.model, inputs, targets, mask)
        self.optimizer.update(self.model, gradients)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        if GPU_THROTTLE < 1:
            # mx.eval already drained the GPU, so this never pauses mid command-buffer.
            time.sleep((time.perf_counter() - step_started) * (1 / GPU_THROTTLE - 1))
        loss_value = float(loss.item())
        valid_tokens = int(mask.sum().item())
        self.step_number = next_step
        self.consumed_tokens += valid_tokens
        for item in self.batch:
            for corpus, count in item[2].items():
                self.consumed_by_corpus[corpus] = self.consumed_by_corpus.get(corpus, 0) + count
        self.final_loss = loss_value
        self.batch = []
        now = time.perf_counter()
        elapsed = max(now - self.started, 1e-9)
        throughput = self.consumed_tokens / elapsed
        eta = int(max(0.0, (self.target_steps - self.step_number) * elapsed / self.step_number))
        report_every = max(1, self.target_steps // 100)
        if self.step_number == 1 or self.step_number == self.target_steps or self.step_number % report_every == 0:
            emit(
                "event",
                event={
                    "kind": "progress",
                    "message": f"step {self.step_number}/{self.target_steps}, loss {loss_value:.4f}, {throughput:.0f} tokens/s",
                    "step": self.step_number,
                    "tokens": self.consumed_tokens,
                    "loss": loss_value,
                    "learning_rate": current_learning_rate,
                    "tokens_per_second": throughput,
                    "eta_seconds": eta,
                },
            )
        checkpoint_every = self.parameters["checkpoint_every"]
        if checkpoint_every > 0 and self.step_number % checkpoint_every == 0:
            self.save_checkpoint()
        evaluate_every = self.parameters["evaluate_every"]
        if evaluate_every > 0 and self.step_number % evaluate_every == 0:
            self.record_evaluation(loss_value)

    def save_weights(self, path, kind, step):
        weights = dict(tree_flatten(self.model.parameters()))
        mx.save_safetensors(
            path,
            weights,
            metadata={
                "format": "openwaldo",
                "kind": kind,
                "schema": "1",
                "backend": "mlx",
                "backend_revision": WORKER_REVISION,
                "architecture_sha256": self.begin["architecture_sha256"],
                "run_id": self.begin["run_id"],
                "step": str(step),
            },
        )

    def save_checkpoint(self):
        name = f"checkpoints/step-{self.step_number:08d}"
        path = os.path.join(self.artifact_directory, *name.split("/"))
        temporary = path + f".tmp-{os.getpid()}"
        if os.path.exists(temporary):
            shutil.rmtree(temporary)
        os.makedirs(temporary)
        weights_path = os.path.join(temporary, "model.safetensors")
        optimizer_path = os.path.join(temporary, "optimizer.safetensors")
        state_path = os.path.join(temporary, "state.json")
        self.save_weights(weights_path, "waldo-mlx-checkpoint", self.step_number)
        optimizer_state = dict(tree_flatten(self.optimizer.state))
        mx.save_safetensors(
            optimizer_path,
            optimizer_state,
            metadata={"format": "openwaldo", "kind": "waldo-mlx-optimizer", "schema": "1", "run_id": self.begin["run_id"], "step": str(self.step_number)},
        )
        write_json(
            state_path,
            {
                "kind": "waldo-training-checkpoint",
                "schema": 1,
                "backend": "mlx",
                "backend_revision": WORKER_REVISION,
                "run_id": self.begin["run_id"],
                "architecture_sha256": self.begin["architecture_sha256"],
                "step": self.step_number,
                "consumed_tokens": self.consumed_tokens,
                "consumption": self.consumed_by_corpus,
                "random": {"algorithm": "mlx-step-seeded-dropout-v1", "seed": self.parameters["seed"]},
            },
        )
        commit_directory(temporary, path)
        weights_path = os.path.join(path, "model.safetensors")
        optimizer_path = os.path.join(path, "optimizer.safetensors")
        state_path = os.path.join(path, "state.json")
        item = {
            "step": self.step_number,
            "tokens": self.consumed_tokens,
            "artifacts": [
                artifact(weights_path, self.logical(name + "/model.safetensors")),
                artifact(optimizer_path, self.logical(name + "/optimizer.safetensors")),
                artifact(state_path, self.logical(name + "/state.json")),
            ],
        }
        self.checkpoints.append(item)
        emit(
            "event",
            event={
                "kind": "checkpoint",
                "message": f"checkpoint step {self.step_number} persisted",
                "step": self.step_number,
                "tokens": self.consumed_tokens,
                "checkpoint": item,
            },
        )

    def restore_checkpoint(self, resume):
        if resume["step"] <= 0 or resume["step"] > self.target_steps:
            raise ValueError(f"resume step {resume['step']} must be in 1..{self.target_steps}")
        paths = {os.path.basename(path): path for path in resume["paths"]}
        required = {"model.safetensors", "optimizer.safetensors", "state.json"}
        if set(paths) != required:
            raise ValueError(f"MLX checkpoint requires {sorted(required)}, found {sorted(paths)}")
        with open(paths["state.json"], "r", encoding="utf-8") as stream:
            state = json.load(stream)
        if (
            state.get("kind") != "waldo-training-checkpoint"
            or state.get("schema") != 1
            or state.get("backend") != "mlx"
            or state.get("backend_revision") != WORKER_REVISION
            or state.get("run_id") != self.begin["run_id"]
            or state.get("architecture_sha256") != self.begin["architecture_sha256"]
            or state.get("step") != resume["step"]
            or state.get("consumed_tokens") != resume["tokens"]
        ):
            raise ValueError("MLX checkpoint state does not match the requested run and resume point")
        self.model.load_weights(paths["model.safetensors"])
        optimizer_state = mx.load(paths["optimizer.safetensors"])
        self.optimizer.state = tree_unflatten(list(optimizer_state.items()))
        mx.eval(self.model.parameters(), self.optimizer.state)
        mx.random.seed(state["random"]["seed"])
        self.step_number = resume["step"]
        self.consumed_tokens = resume["tokens"]
        self.consumed_by_corpus = state.get("consumption", {})
        self.replay_steps = resume["step"]
        self.checkpoints = [resume["checkpoint"]]

    def record_evaluation(self, _training_loss):
        if not self.evaluation_sequences:
            return
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0.0
        for offset in range(0, len(self.evaluation_sequences), self.batch_size):
            batch = self.evaluation_sequences[offset : offset + self.batch_size]
            tokens = mx.array([item[0] for item in batch], dtype=mx.int32)
            mask = mx.array([item[1] for item in batch], dtype=mx.float32)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits = self.model(inputs)
            losses = nn.losses.cross_entropy(logits, targets, reduction="none")
            loss_sum = (losses * mask).sum()
            token_count = mask.sum()
            mx.eval(loss_sum, token_count)
            total_loss += float(loss_sum.item())
            total_tokens += float(token_count.item())
        loss_value = total_loss / total_tokens
        self.model.train()
        item = {
            "step": self.step_number,
            "tokens": self.consumed_tokens,
            "metrics": {"heldout_loss": loss_value, "heldout_perplexity": math.exp(min(loss_value, 80.0))},
        }
        self.evaluations.append(item)
        emit(
            "event",
            event={
                "kind": "evaluation",
                "message": f"step {self.step_number} held-out loss {loss_value:.4f}",
                "step": self.step_number,
                "tokens": self.consumed_tokens,
                "evaluation": item,
            },
        )

    def finish(self):
        evaluation_set = self.begin["evaluation_set"]
        if self.evaluation_record_count != evaluation_set["records"] or self.evaluation_token_targets != evaluation_set["token_targets"]:
            raise ValueError(
                f"evaluation stream has {self.evaluation_record_count} records and {self.evaluation_token_targets} targets; "
                f"run BOM pins {evaluation_set['records']} records and {evaluation_set['token_targets']} targets"
            )
        if self.step_number < self.target_steps and len(self.token_buffer) > 1:
            target_mask = self.loss_buffer[1 : self.sequence_length + 1]
            self.add_sequence(self.token_buffer[: self.sequence_length + 1], target_mask, self.corpus_buffer[1 : self.sequence_length + 1])
        if self.step_number < self.target_steps and self.batch:
            self.train_batch()
        if self.step_number != self.target_steps:
            raise ValueError(
                f"canonical stream produced only {self.step_number} training steps; profile requires {self.target_steps}"
            )
        if self.parameters["checkpoint_every"] > 0 and (
            not self.checkpoints or self.checkpoints[-1]["step"] != self.step_number
        ):
            self.save_checkpoint()
        if self.parameters["evaluate_every"] > 0 and (
            not self.evaluations or self.evaluations[-1]["step"] != self.step_number
        ):
            self.record_evaluation(self.final_loss)

        weights_name = "model.safetensors"
        weights_path = os.path.join(self.artifact_directory, weights_name)
        self.save_weights(weights_path, "waldo-mlx-model", self.step_number)
        config_name = "config.json"
        config_path = os.path.join(self.artifact_directory, config_name)
        write_json(
            config_path,
            {
                "kind": "waldo-mlx-model-config",
                "schema": 1,
                "architecture_sha256": self.begin["architecture_sha256"],
                "architecture": self.architecture,
                "training_profile": self.parameters,
                "initialization": None if self.initialization is None else {
                    "source_type": self.initialization.get("source_type", "run"),
                    "source_id": self.initialization.get("source_id", self.initialization.get("source_run_id")),
                    "source_run_id": self.initialization.get("source_run_id"),
                    "artifact": self.initialization["artifact"],
                },
                "backend": {"name": "mlx", "revision": WORKER_REVISION, "version": importlib.metadata.version("mlx")},
            },
        )
        tokenizer_name = "tokenizer.json"
        tokenizer_path = os.path.join(self.artifact_directory, tokenizer_name)
        write_json(
            tokenizer_path,
            {
                "kind": "waldo-tokenizer",
                "schema": 1,
                **self.begin["tokenizer"],
            },
        )
        outputs = [
            artifact(weights_path, self.logical(weights_name)),
            artifact(config_path, self.logical(config_name)),
            artifact(tokenizer_path, self.logical(tokenizer_name)),
        ]
        emit(
            "complete",
            observation={
                "simulated": False,
                "steps": self.step_number,
                "consumed_tokens": self.consumed_tokens,
                "final_loss": self.final_loss,
                "checkpoints": self.checkpoints,
                "evaluations": self.evaluations,
                "artifacts": outputs,
                "consumption": [
                    {"corpus": corpus, "token_targets": targets}
                    for corpus, targets in sorted(self.consumed_by_corpus.items())
                ],
            },
        )


def run():
    if len(sys.argv) != 3:
        raise ValueError("worker requires artifact directory and artifact prefix")
    artifact_directory = os.path.abspath(sys.argv[1])
    artifact_prefix = sys.argv[2]
    os.makedirs(artifact_directory, exist_ok=True)
    trainer = None
    ended = False
    for line in sys.stdin:
        frame = json.loads(line)
        if frame.get("schema") != PROTOCOL_SCHEMA:
            raise ValueError(f"unsupported worker input schema {frame.get('schema')}")
        kind = frame.get("kind")
        if kind == "begin":
            if trainer is not None:
                raise ValueError("worker received duplicate begin frame")
            trainer = Trainer(frame["begin"], artifact_directory, artifact_prefix)
        elif kind == "record":
            if trainer is None or ended:
                raise ValueError("worker received record outside stream")
            trainer.add_record(frame["record"])
        elif kind == "evaluation_record":
            if trainer is None or ended:
                raise ValueError("worker received evaluation record outside stream")
            trainer.add_evaluation_record(frame["record"])
        elif kind == "end":
            if trainer is None or ended:
                raise ValueError("worker received invalid end frame")
            ended = True
        else:
            raise ValueError(f"unsupported worker input kind {kind!r}")
    if trainer is None or not ended:
        raise ValueError("worker input ended without begin/end framing")
    trainer.finish()


Attention = WALDO_SHARED_ATTENTION
FeedForward = WALDO_SHARED_FEED_FORWARD
DecoderBlock = WALDO_SHARED_DECODER_BLOCK
DecoderLM = WALDO_SHARED_DECODER_LM
ByteTokenizer = WALDO_SHARED_BYTE_TOKENIZER

try:
    run()
except Exception as error:
    traceback.print_exc(file=sys.stderr)
    emit("error", error=str(error))
    sys.exit(1)
