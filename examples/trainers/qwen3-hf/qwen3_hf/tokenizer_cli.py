# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
from tokenizers import Tokenizer
from .schemas import TOKENIZER, validate
from .wire import read, emit

ASSETS = Path(__file__).resolve().parent.parent / "tokenizer"
VOCABULARY_SIZE = 151936  # Qwen3 padded embedding size, not tokenizer entry count.
SPECIAL_TOKENS = dict(pad=151643, bos=151643, eos=151645,
                      im_start=151644, im_end=151645)


def codec():
    # Removing the special added-token entries leaves the underlying BPE intact.
    # add_special_tokens=False alone still recognizes literal special spellings.
    config = json.loads((ASSETS / "tokenizer.json").read_text())
    config["added_tokens"] = []
    config["post_processor"] = None
    config["padding"] = None
    config["truncation"] = None
    return Tokenizer.from_str(json.dumps(config))


def main():
    initial = read()
    if set(initial) != {"configuration"}:
        raise ValueError("expected configuration")
    validate(initial["configuration"], TOKENIZER)
    if sys.argv[1] == "probe":
        emit(dict(name="qwen3", vocabulary_size=VOCABULARY_SIZE,
                  special_tokens=SPECIAL_TOKENS, deterministic=True,
                  schemas=dict(configuration=TOKENIZER)))
    elif sys.argv[1] == "serve":
        tokenizer = codec()
        while (request := read(allow_eof=True)) is not None:
            if set(request) != {"id", "text"} or not isinstance(request["text"], str):
                raise ValueError("expected id and text")
            emit(dict(id=request["id"], tokens=tokenizer.encode(request["text"], add_special_tokens=False).ids))
    else:
        raise ValueError("expected probe or serve")


if __name__ == "__main__":
    main()
