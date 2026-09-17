# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

import base64
import contextlib
import json
import os
from pathlib import Path
import sys
import time

emit = support["emit"]


class Output:
    """Hold potential stop prefixes across streamed text boundaries."""
    def __init__(self, stops):
        self.stops = stops
        self.pending = ""
        self.stopped = False

    def write(self, text, final=False):
        if self.stopped:
            return
        self.pending += text
        matches = [self.pending.find(stop) for stop in self.stops if stop in self.pending]
        if matches:
            end = min(matches)
            self.stopped = True
        else:
            hold = 0
            if not final:
                for stop in self.stops:
                    for size in range(1, min(len(stop), len(self.pending) + 1)):
                        if self.pending.endswith(stop[:size]):
                            hold = max(hold, size)
            end = len(self.pending) - hold
        if end:
            emit("token", data=base64.b64encode(self.pending[:end].encode("utf-8")).decode("ascii"))
        self.pending = self.pending[end:]


def run():
    payload = json.loads(sys.argv[1])
    architecture, paths = payload["architecture"], payload["paths"]
    spec = architecture["transformers"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    transformers = support["verify_package"](payload["wheel"], spec["package"])
    import torch
    from safetensors.torch import load_model
    import codecs

    dtype = spec["config"].get("dtype", "float32") if payload["device"]["device"] == "cuda" else "float32"
    device = support["validate_device"](torch, payload["device"], dtype=dtype)

    support["configuration"](transformers, spec)  # Validate the allowed class pair.
    config = getattr(transformers, spec["config_class"])(**json.loads(Path(paths["config.json"]).read_text()))
    descriptor = json.loads(Path(paths.get("waldo_tokenizer.json", paths["tokenizer.json"])).read_text())
    identity = architecture["tokenizer"]
    if descriptor["name"] != identity["name"] or descriptor["revision"] != identity["revision"] or descriptor["vocabulary_size"] != config.vocab_size:
        raise ValueError("saved tokenizer differs from model identity")
    codec = None
    if descriptor["name"] == "huggingface":
        if descriptor.get("huggingface") != identity.get("huggingface"):
            raise ValueError("tokenizer pins differ from architecture")
        pin = descriptor["huggingface"]
        # Require every pinned file to be a verified artifact in one directory.
        directory = Path(paths["tokenizer.json"]).parent
        for name in pin["files"]:
            if name not in paths or Path(paths[name]).parent != directory:
                raise ValueError("missing pinned tokenizer run artifact")
        with support["pinned_tokenizer"](transformers, descriptor, directory) as (loaded, _):
            codec = loaded
        vocabulary = set(codec.get_vocab().values())
    elif (descriptor["name"], descriptor["revision"], config.vocab_size) == ("byte", "builtin-byte-schema-1", 259):
        vocabulary = set(range(259))
    else:
        raise ValueError("Transformers chat supports byte and pinned fast tokenizers only")
    for name in ("pad", "bos", "eos"):
        expected = descriptor[name + "_id"]
        if getattr(config, name + "_token_id", None) != (None if expected == -1 else expected):
            raise ValueError("model special token IDs differ from tokenizer")
    model = getattr(transformers, spec["model_class"])(config)
    load_model(model, paths["model.safetensors"], strict=True)
    model.to(device=device, dtype=getattr(torch, dtype)).eval()
    context = int(config.max_position_embeddings)
    if context != architecture["context_tokens"] or context < 1:
        raise ValueError("saved model context differs from architecture")
    allowed = torch.zeros(config.vocab_size, dtype=torch.bool, device=device)
    allowed[list(vocabulary)] = True
    for token in (descriptor["pad_id"], descriptor["bos_id"]):
        if token >= 0 and token != descriptor["eos_id"]:
            allowed[token] = False
    emit("ready", context_tokens=context)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("kind") != "generate" or request.get("schema") != 1:
            raise ValueError("unsupported inference request")
        options, prompt = request["options"], request["prompt"]
        tokens = codec.encode(prompt, add_special_tokens=False) if codec else [b + 3 for b in prompt.encode("utf-8")]
        if not tokens:
            if descriptor["bos_id"] < 0:
                raise ValueError("tokenizer has no BOS; provide a nonempty prompt")
            tokens = [descriptor["bos_id"]]
        tokens = tokens[-context:]
        generator = torch.Generator(device=device)
        generator.manual_seed(options["seed"]) if options.get("seed") is not None else generator.seed()
        output = Output(options.get("stop", []))
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        if codec:
            class Streamer(transformers.TextStreamer):
                def on_finalized_text(self, text, stream_end=False):
                    output.write(text, final=stream_end)
            streamer = Streamer(codec, skip_prompt=False, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        started, count, reason = time.perf_counter(), 0, "max_tokens"
        with torch.inference_mode():
            for _ in range(options["max_tokens"]):
                logits = model(input_ids=torch.tensor([tokens[-context:]], device=device), use_cache=False).logits[0, -1].float()
                if not torch.isfinite(logits).all():
                    raise ValueError("non-finite generation logits")
                logits[~allowed] = -float("inf")
                if options["temperature"] == 0:
                    token = int(logits.argmax())
                else:
                    probabilities = torch.softmax(logits / options["temperature"], dim=-1)
                    ordered, indices = probabilities.sort(descending=True)
                    ordered[ordered.cumsum(0) - ordered > options["top_p"]] = 0
                    token = int(indices[torch.multinomial(ordered, 1, generator=generator)])
                if token == descriptor["eos_id"]:
                    reason = "eos"
                    break
                tokens.append(token)
                tokens = tokens[-context:]
                count += 1
                if codec:
                    streamer.put(torch.tensor([token]))
                else:
                    output.write(decoder.decode(bytes([token - 3])))
                if output.stopped:
                    reason = "stop"
                    break
        if codec:
            streamer.end()
        else:
            output.write(decoder.decode(b"", final=True), final=True)
        if output.stopped:
            reason = "stop"
        emit("complete", tokens=count, finish_reason=reason, duration_ms=int((time.perf_counter()-started)*1000))


try:
    with contextlib.redirect_stdout(sys.stderr):
        run()
except Exception as error:
    emit("error", error=str(error))
    sys.exit(1)
