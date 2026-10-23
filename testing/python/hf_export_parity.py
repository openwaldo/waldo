# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent offline export oracle; deliberately does not import WALDO workers."""
import json
import os
from pathlib import Path
import sys


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    request = json.loads(sys.argv[1])
    source, exported = Path(request["source"]), Path(request["export"])
    descriptor_name = "waldo_tokenizer.json" if (source / "waldo_tokenizer.json").exists() else "tokenizer.json"
    descriptor = json.loads((source / descriptor_name).read_text())
    assert descriptor == json.loads((exported / descriptor_name).read_text())
    codec = None
    if descriptor["name"] == "huggingface":
        codec = AutoTokenizer.from_pretrained(exported, local_files_only=True, trust_remote_code=False)
        original_codec = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        for text in [request["prompt"], "", "hello\nworld", "café 世界 😀", "<|im_end|>"]:
            assert codec.encode(text, add_special_tokens=False) == original_codec.encode(text, add_special_tokens=False)
        for name, digest in descriptor["huggingface"]["files"].items():
            import hashlib
            assert hashlib.sha256((exported / name).read_bytes()).hexdigest() == digest
        tokens = codec.encode(request["prompt"], add_special_tokens=False)
        vocabulary = set(codec.get_vocab().values())
    else:
        # WALDO byte tokenizer.json is a descriptor, not an AutoTokenizer file.
        assert descriptor["name"] == "byte"
        tokens = [byte + 3 for byte in request["prompt"].encode("utf-8")]
        vocabulary = set(range(259))
    device = "cuda" if os.environ.get("WALDO_TRANSFORMERS_DEVICE") == "cuda" else "cpu"
    model = AutoModelForCausalLM.from_pretrained(exported, local_files_only=True, trust_remote_code=False).to(device).eval()
    original = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=False).to(device).eval()
    saved_config = model.config.to_dict()
    original_config = original.config.to_dict()
    saved_config.pop("_name_or_path", None)
    original_config.pop("_name_or_path", None)
    assert saved_config == original_config, "configuration changed"
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name]), name
    ids = torch.tensor([tokens], device=device)
    with torch.inference_mode():
        torch.testing.assert_close(model(ids, use_cache=False).logits, original(ids, use_cache=False).logits, rtol=0, atol=0)
        allowed = torch.zeros(model.config.vocab_size, dtype=torch.bool, device=device)
        allowed[list(vocabulary)] = True
        for special in [descriptor["pad_id"], descriptor["bos_id"]]:
            if special >= 0 and special != descriptor["eos_id"]:
                allowed[special] = False
        generated, reason = [], "max_tokens"
        for _ in range(8):
            logits = model(torch.tensor([tokens[-model.config.max_position_embeddings:]], device=device), use_cache=False).logits[0, -1].float()
            logits[~allowed] = -float("inf")
            token = int(logits.argmax())
            if token == descriptor["eos_id"]:
                reason = "eos"
                break
            generated.append(token)
            tokens.append(token)
    text = codec.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False) if codec else bytes(token - 3 for token in generated).decode("utf-8", errors="replace")
    assert (text, len(generated), reason) == (request["text"], request["tokens"], request["reason"]), (text, request)
    print("Independent Transformers reload: tokenizer, exact tensors/logits, greedy text/count/finish parity passed")


if __name__ == "__main__":
    main()
