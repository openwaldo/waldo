#!/usr/bin/env python3
"""Materialize the pinned full and formatting-only rung-0007 corpora."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REVISION = "8d2454f2625a85d911cd7810816e489f0ebb1448"
PREFIX = "data/sft/interaction-contract-v1/train/"


# The published interaction-contract-v1 shard holds the two-turn prompt/response
# projection that openwaldo/post-training-data provides for dialogue-pair
# ingestion. Six categories also carry typed multi-turn messages, including
# assistant tool calls with empty text content, which chat-messages rejects.
PAIR_INPUT: dict[str, object] = {
    "format": "jsonl",
    "type": "dialogue-pair",
    "fields": {"id": "id", "text": ["prompt"], "response": "response"},
}

# Every formatting row is already a plain user/assistant exchange.
MESSAGES_INPUT: dict[str, object] = {
    "format": "jsonl",
    "type": "chat-messages",
    "fields": {"id": "id", "meta": {"category": "category"}},
    "messages": {
        "role": "messages[].role",
        "content": "messages[].content",
    },
}


def pair_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "id": row["id"],
        "category": row["category"],
        "prompt": row["prompt"],
        "response": row["response"],
    }


def messages_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "id": row["id"],
        "category": row["category"],
        "messages": row["messages"],
    }


def write_source(output: Path, corpus_id: str, title: str,
                 description: str, records: list[dict[str, object]],
                 input_mapping: dict[str, object]) -> dict[str, object]:
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    data = raw / "training.jsonl"
    with data.open("w", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")

    payload = data.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    tree = hashlib.sha256(
        f"{digest}\t{len(payload)}\ttraining.jsonl\n".encode()
    ).hexdigest()
    manifest = {
        "kind": "waldo-source-directory",
        "schema": 1,
        "retrieved_at": "2026-10-25T00:00:00Z",
        "corpus": {
            "id": corpus_id,
            "title": title,
            "description": description,
        },
        "sources": [{
            "id": corpus_id,
            "path": "",
            "license": "Apache-2.0",
            "source": {
                "name": "openwaldo/post-training-data",
                "version": REVISION,
                "url": "https://github.com/openwaldo/post-training-data",
                "category": "public-dataset",
                "content": {"languages": ["en"]},
                "license_evidence": {"declaration": "Apache-2.0"},
            },
            "input": input_mapping,
            "artifacts": [],
        }],
        "fetcher": {"name": "rung-0007-pinned-interaction-contract"},
        "raw": {
            "path": "raw",
            "file_count": 1,
            "byte_count": len(payload),
            "tree_sha256": tree,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"records": len(records), "sha256": digest, "bytes": len(payload)}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} POST_TRAINING_REPO OUTPUT_ROOT")
    repository = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()

    names = subprocess.check_output(
        ["git", "-C", str(repository), "ls-tree", "-r", "--name-only", REVISION, PREFIX],
        text=True,
    ).splitlines()
    names = sorted(name for name in names if name.endswith(".jsonl"))
    if len(names) != 18:
        raise SystemExit(f"expected 18 pinned training shards, found {len(names)}")

    records: list[dict[str, object]] = []
    for name in names:
        payload = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{REVISION}:{name}"],
            text=True,
        )
        for line in payload.splitlines():
            row = json.loads(line)
            records.append(row)

    full = write_source(
        output / "interaction-contract-v1",
        "interaction-contract-v1",
        "OpenWALDO Interaction Contract v1",
        "Pinned supervised interaction-contract training split for rung 0007.",
        [pair_row(row) for row in records],
        PAIR_INPUT,
    )
    formatting = write_source(
        output / "interaction-formatting-v1",
        "interaction-formatting-v1",
        "OpenWALDO Interaction Formatting v1",
        "Formatting-only subset of the pinned interaction-contract training split.",
        [messages_row(row) for row in records if row["category"] == "formatting"],
        MESSAGES_INPUT,
    )
    print(json.dumps({"full": full, "formatting": formatting}, indent=2))


if __name__ == "__main__":
    main()
