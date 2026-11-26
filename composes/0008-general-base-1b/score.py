#!/usr/bin/env python3
"""Score objective generation diagnostics for promoted rung 0008."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load(path: Path) -> list[dict[str, object]]:
    payload = path.read_text()
    decoder = json.JSONDecoder()
    rows = []
    offset = 0
    while offset < len(payload):
        while offset < len(payload) and payload[offset].isspace():
            offset += 1
        if offset < len(payload):
            row, offset = decoder.raw_decode(payload, offset)
            rows.append(row)
    return rows


def loops(text: str) -> bool:
    words = re.findall(r"[a-z0-9']+", text.lower())
    for width in range(2, 9):
        for start in range(len(words) - width * 3 + 1):
            phrase = words[start : start + width]
            if words[start + width : start + 2 * width] == phrase == words[start + 2 * width : start + 3 * width]:
                return True
    return False


def complete_sentences(text: str) -> int:
    return sum(
        len(re.findall(r"\w+", sentence)) >= 3
        for sentence in re.findall(r"[^.!?]+[.!?]+", text)
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} SAMPLED.jsonl GREEDY.jsonl")
    sampled, greedy = (load(Path(path)) for path in sys.argv[1:])
    if len(sampled) != 20 or len(greedy) != 20:
        raise SystemExit("expected 20 sampled and 20 greedy responses")

    two_sentences = sum(complete_sentences(row["result"]["text"]) >= 2 for row in sampled)
    sampled_eos = sum(row["result"]["finish_reason"] == "eos" for row in sampled)
    sampled_loops = sum(loops(row["result"]["text"]) for row in sampled)
    greedy_loops = sum(loops(row["result"]["text"]) for row in greedy)
    gates = {
        "two_complete_sentences": two_sentences >= 18,
        "sampled_loops": sampled_loops <= 1,
        "greedy_loops": greedy_loops <= 2,
    }
    print(json.dumps({
        "responses": len(sampled),
        "two_complete_sentences": two_sentences,
        "sampled_eos_diagnostic": sampled_eos,
        "sampled_loops": sampled_loops,
        "greedy_loops": greedy_loops,
        "gates": gates,
        "passed": all(gates.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
