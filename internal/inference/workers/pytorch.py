# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import math
import os
import random
import struct
import sys
import time
import traceback

import torch
from torch import nn
from torch.nn import functional


PROTOCOL_SCHEMA = 1
SAFE_TORCH_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def emit(kind, **payload):
    frame = {"kind": kind, "schema": PROTOCOL_SCHEMA}
    frame.update(payload)
    print(json.dumps(frame, separators=(",", ":")), flush=True)


def load_safetensors(path):
    with open(path, "rb") as stream:
        encoded_length = stream.read(8)
        if len(encoded_length) != 8:
            raise ValueError("model Safetensors header is truncated")
        header_length = struct.unpack("<Q", encoded_length)[0]
        if header_length == 0 or header_length > 1024 * 1024 * 1024:
            raise ValueError(f"invalid model Safetensors header length {header_length}")
        header = json.loads(stream.read(header_length))
        payload = bytearray(stream.read())
    tensors = {}
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        dtype = SAFE_TORCH_DTYPES.get(descriptor["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported model Safetensors dtype {descriptor['dtype']}")
        start, end = descriptor["data_offsets"]
        if start < 0 or end < start or end > len(payload):
            raise ValueError(f"invalid model Safetensors offsets for {name}")
        value = torch.frombuffer(payload[start:end], dtype=dtype).clone()
        tensors[name] = value.reshape(descriptor["shape"])
    return tensors


class RMSNorm(nn.Module):
    def __init__(self, hidden, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.epsilon = epsilon

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.epsilon)
        return normalized.to(value.dtype) * self.weight


def rotate_half(value):
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


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

    def rope(self, value):
        length = value.shape[2]
        positions = torch.arange(length, device=value.device, dtype=torch.float32)
        frequencies = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=value.device, dtype=torch.float32) / self.head_dim))
        angles = torch.outer(positions, frequencies)
        angles = torch.cat((angles, angles), dim=-1).to(value.dtype)[None, None, :, :]
        return value * angles.cos() + rotate_half(value) * angles.sin()

    def forward(self, value):
        batch, length, _ = value.shape
        query = self.q_proj(value).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        val = self.v_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        query = self.rope(query)
        key = self.rope(key)
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            val = val.repeat_interleave(repeats, dim=1)
        attended = functional.scaled_dot_product_attention(query, key, val, is_causal=True)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class FeedForward(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, value):
        return self.down(functional.silu(self.gate(value)) * self.up(value))


class DecoderBlock(nn.Module):
    def __init__(self, hidden, intermediate, heads, kv_heads):
        super().__init__()
        self.attention_norm = RMSNorm(hidden)
        self.attention = Attention(hidden, heads, kv_heads)
        self.ffn_norm = RMSNorm(hidden)
        self.feed_forward = FeedForward(hidden, intermediate)

    def forward(self, value):
        value = value + self.attention(self.attention_norm(value))
        return value + self.feed_forward(self.ffn_norm(value))


class DecoderLM(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        vocabulary = architecture["vocabulary_size"]
        hidden = architecture["hidden_size"]
        self.tie_embeddings = architecture["tie_embeddings"]
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.layers = nn.ModuleList([
            DecoderBlock(hidden, architecture["intermediate_size"], architecture["attention_heads"], architecture["key_value_heads"])
            for _ in range(architecture["layers"])
        ])
        self.norm = RMSNorm(hidden)
        if not self.tie_embeddings:
            self.output = nn.Linear(hidden, vocabulary, bias=False)

    def forward(self, tokens):
        value = self.embedding(tokens)
        for layer in self.layers:
            value = layer(value)
        value = self.norm(value)
        if self.tie_embeddings:
            return functional.linear(value, self.embedding.weight)
        return self.output(value)


class ByteTokenizer:
    bos_id = 1
    eos_id = 2
    byte_offset = 3

    def encode(self, text):
        return [byte + self.byte_offset for byte in text.encode("utf-8")]

    def decode_token(self, token):
        if self.byte_offset <= token < self.byte_offset + 256:
            return bytes([token - self.byte_offset])
        return b""


def sample(logits, temperature, top_p, generator):
    values = [float(value) for value in logits.float().cpu().tolist()]
    if temperature == 0:
        return max(range(len(values)), key=values.__getitem__)
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    probabilities = [math.exp(value - maximum) for value in scaled]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    order = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)
    kept = []
    cumulative = 0.0
    for index in order:
        if kept and cumulative >= top_p:
            break
        probability = probabilities[index]
        kept.append((index, probability))
        cumulative += probability
    draw = generator.random() * cumulative
    cumulative = 0.0
    for index, probability in kept:
        cumulative += probability
        if draw <= cumulative:
            return index
    return kept[-1][0]


class Generator:
    def __init__(self, weights_path, config_path, tokenizer_path, device_name):
        with open(config_path, "r", encoding="utf-8") as stream:
            config = json.load(stream)
        with open(tokenizer_path, "r", encoding="utf-8") as stream:
            tokenizer = json.load(stream)
        if config.get("kind") not in ("waldo-pytorch-model-config", "waldo-torchtitan-model-config") or config.get("schema") != 1:
            raise ValueError("unsupported WALDO PyTorch model configuration")
        if tokenizer.get("kind") not in ("waldo-tokenizer", "waldo-byte-tokenizer") or tokenizer.get("schema") != 1:
            raise ValueError("unsupported WALDO tokenizer artifact")
        architecture = config["architecture"]
        architecture_tokenizer = architecture["tokenizer"]
        if tokenizer.get("name") != architecture_tokenizer.get("name") or tokenizer.get("revision") != architecture_tokenizer.get("revision") or tokenizer.get("vocabulary_size") != architecture.get("vocabulary_size"):
            raise ValueError("PyTorch chat tokenizer artifact does not match the architecture")
        self.context_tokens = int(architecture["context_tokens"])
        self.byte_tokenizer = tokenizer.get("name") == "byte"
        self.tokenizer = ByteTokenizer() if self.byte_tokenizer else None
        self.bos_id = int(tokenizer["bos_id"])
        self.eos_id = int(tokenizer["eos_id"])
        self.pad_id = int(tokenizer["pad_id"])
        self.device = torch.device(device_name)
        self.model = DecoderLM(architecture)
        missing, unexpected = self.model.load_state_dict(load_safetensors(weights_path), strict=False)
        if missing or unexpected:
            raise ValueError(f"model weights mismatch: missing={missing}, unexpected={unexpected}")
        self.model.to(self.device)
        self.model.eval()

    def generate(self, request):
        prompt = request.get("prompt", "")
        maximum = int(request["max_tokens"])
        temperature = float(request["temperature"])
        top_p = float(request["top_p"])
        seed = request.get("seed")
        stop_sequences = [[int(token) for token in sequence] for sequence in request.get("stop_token_ids", []) if sequence]
        generator = random.Random(seed if seed is not None else int.from_bytes(os.urandom(16), "big"))
        tokens = [int(token) for token in request.get("token_ids", [])]
        if not tokens and self.byte_tokenizer:
            tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [self.bos_id]
        tokens = tokens[-self.context_tokens :]
        started = time.perf_counter()
        produced = 0
        reason = "max_tokens"
        with torch.inference_mode():
            for _ in range(maximum):
                inputs = torch.tensor([tokens[-self.context_tokens :]], dtype=torch.long, device=self.device)
                logits = self.model(inputs)
                token = sample(logits[0, -1], temperature, top_p, generator)
                tokens.append(token)
                if token == self.eos_id:
                    reason = "eos"
                    break
                if token == self.pad_id or token == self.bos_id:
                    continue
                produced += 1
                if self.byte_tokenizer:
                    data = self.tokenizer.decode_token(token)
                    if data:
                        emit("token", data=base64.b64encode(data).decode("ascii"))
                else:
                    emit("token", token_id=token)
                if any(len(tokens) >= len(sequence) and tokens[-len(sequence):] == sequence for sequence in stop_sequences):
                    reason = "stop"
                    break
        emit("complete", tokens=produced, finish_reason=reason, duration_ms=int((time.perf_counter() - started) * 1000))


def run():
    if len(sys.argv) != 5:
        raise ValueError("PyTorch chat worker requires weights, config, tokenizer, and device")
    generator = Generator(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    emit("ready", context_tokens=generator.context_tokens)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("schema") != PROTOCOL_SCHEMA or request.get("kind") != "generate":
            raise ValueError("unsupported PyTorch chat request")
        generator.generate(request)


RMSNorm = WALDO_SHARED_RMS_NORM
Attention = WALDO_SHARED_ATTENTION
FeedForward = WALDO_SHARED_FEED_FORWARD
DecoderBlock = WALDO_SHARED_DECODER_BLOCK
DecoderLM = WALDO_SHARED_DECODER_LM

try:
    run()
except Exception as error:
    traceback.print_exc(file=sys.stderr)
    emit("error", error=str(error))
    sys.exit(1)
