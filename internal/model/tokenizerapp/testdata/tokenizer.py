# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
import json
import sys

configuration = json.loads(sys.stdin.readline())["configuration"]
mode = configuration.get("mode", "normal")
if sys.argv[1] == "probe":
    print(json.dumps({"name": "fixture", "vocabulary_size": 259,
                      "special_tokens": {"pad": 0, "bos": 1, "eos": 2},
                      "deterministic": True,
                      "schemas": {"configuration": {"type": "object", "properties": {
                          "mode": {"type": "string"}}, "additionalProperties": False}}}))
else:
    pending = []
    for line in sys.stdin:
        request = json.loads(line)
        if mode == "crash":
            print("fixture crashed", file=sys.stderr)
            sys.exit(7)
        if mode == "malformed":
            print("not json", flush=True)
            continue
        if mode == "hang":
            import time
            time.sleep(60)
        tokens = [byte + 3 for byte in request["text"].encode()]
        if mode == "range":
            tokens = [259]
        if mode == "null-token":
            tokens = [None]
        if mode == "negative":
            tokens = [-1]
        if mode == "oversize":
            tokens = [3] * (6 << 20)
        response = {"id": request["id"], "tokens": tokens}
        if mode == "error":
            response = {"id": request["id"], "error": "fixture refused"}
        if mode == "missing-tokens":
            del response["tokens"]
        if mode == "unknown-field":
            response["surprise"] = 1
        if mode == "wrong-id":
            response["id"] += 1
        if mode == "reverse":
            pending.append(response)
            if len(pending) == 2:
                for item in reversed(pending):
                    print(json.dumps(item), flush=True)
                pending = []
        else:
            print(json.dumps(response), flush=True)
