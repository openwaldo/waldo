#!/usr/bin/env python3
# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0
"""
Evaluator for a join gate over prompt -> Python program data (python-basics style): asks the model
each evaluation prompt at temperature 0, runs the program it answers with, and counts it passed when
its output matches the reference program's and it has the reference's structure. Writes waldo model gate's results document:

  evaluate_python.py --records <evaluation.jsonl> [--limit N] <model> <evaluation-bom.json> <results.json>

The metric is pass_rate, the share of programs whose output matched; the evaluation BOM sets its
threshold. Records are JSON lines with "id", "prompt", and "response" (the reference program, fenced).
Output is stdout and stderr, so logged warnings count. Structure is what the reference has of a
def main(), an if __name__ == "__main__" guard calling it, and logging calls; the answer needs the same.

This runs code a model wrote. Each program runs with python3 -I in an empty scratch directory, with
a timeout, no input, and the evaluator's environment stripped to PATH; that limits accidents, not
malice. Use it for runs whose models you would run anyway, as join's gates do on contributors'
machines, and keep it away from untrusted models.
WALDO (the binary) and WALDO_CONFIG come from join, which runs the evaluator for its closers.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


# json.Marshal escapes these in strings; waldo writes its JSON files without escaping them.
GO_ESCAPES = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026", "\u2028": "\\u2028", "\u2029": "\\u2029"}


def compact(text):
    """Go's json.Marshal bytes of a JSON file waldo wrote indented: the same tokens without layout."""
    out, inside, escaped = [], False, False
    for char in text:
        if inside:
            out.append(char if escaped else GO_ESCAPES.get(char, char))
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                inside = False
        elif char == '"':
            inside = True
            out.append(char)
        elif char not in " \t\r\n":
            out.append(char)
    return "".join(out)


def program(text):
    """The program in an answer: its fenced block, or the whole answer."""
    match = FENCE.search(text)
    return (match.group(1) if match else text).strip() + "\n"


def structure(source):
    """Which of a main function, a main guard, and logging calls a program has."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            found.add("main")
        if isinstance(node, ast.If) and ast.unparse(node.test) in ("__name__ == '__main__'", "'__main__' == __name__") \
                and any(isinstance(call, ast.Call) and ast.unparse(call.func) == "main" for call in ast.walk(node)):
            found.add("guard")
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
           and node.func.value.id == "logging" for node in ast.walk(tree)):
        found.add("logging")
    return found


def run(source, timeout=10):
    """A program's stdout and stderr, or None if it failed or ran too long."""
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "program.py")
        with open(path, "w") as out:
            out.write(source)
        try:
            # Python 3.11+ refuses to print integers over 4,300 digits, which large factorials are.
            result = subprocess.run([sys.executable, "-I", "-X", "int_max_str_digits=0", path], cwd=scratch, stdin=subprocess.DEVNULL, capture_output=True,
                                    text=True, timeout=timeout, env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return None
        return (result.stdout, result.stderr) if result.returncode == 0 else None


def answer(model, prompt):
    waldo = os.environ.get("WALDO", "waldo")
    result = subprocess.run([waldo, "--json", "model", "chat", model, prompt, "--temperature", "0", "--max-tokens", "256"],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)["result"]["text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True)
    parser.add_argument("--limit", type=int, default=0, help="score N records evenly spaced over the id-sorted set (0: all)")
    parser.add_argument("--show", action="store_true", help="print each prompt and the program the model wrote")
    parser.add_argument("model")
    parser.add_argument("evaluation_bom")
    parser.add_argument("results")
    args = parser.parse_args()
    with open(args.evaluation_bom) as bom_file:
        bom_text = bom_file.read()
    bom = json.loads(bom_text)
    metrics = [metric["name"] for metric in bom.get("metrics", [])]
    if metrics != ["pass_rate"]:
        sys.exit(f"{args.evaluation_bom} must define exactly the pass_rate metric, not {metrics}")
    records = sorted((json.loads(line) for line in open(args.records) if line.strip()), key=lambda record: record["id"])
    if args.limit and args.limit < len(records):
        # Evenly spaced over the id-sorted records, so every task family is sampled.
        records = [records[index * len(records) // args.limit] for index in range(args.limit)]
    passed, scored = 0, 0
    for record in records:
        reference = program(record["response"])
        expected = run(reference)
        if expected is None:
            # A reference that fails or runs too long can't be matched, so it isn't scored.
            print(f"skip  {record['id']} (reference program failed or timed out)", flush=True)
            continue
        written = program(answer(args.model, record["prompt"]))
        ok = run(written) == expected and structure(reference) <= structure(written)
        passed, scored = passed + ok, scored + 1
        print(f"{'pass' if ok else 'FAIL'}  {record['id']}", flush=True)
        if args.show:
            print(f"  > {record['prompt']}\n" + "".join(f"    {line}\n" for line in written.splitlines()), flush=True)
    rate = passed / scored if scored else 0.0
    print(f"{passed}/{scored} programs matched ({len(records) - scored} unscorable): pass_rate {rate:.4f}")
    document = {"kind": "openwaldo-evaluation-results", "schema": 1,
                "evaluation_bom_sha256": hashlib.sha256(compact(bom_text).encode()).hexdigest(),
                "results": [{"metric": "pass_rate", "value": rate}]}
    with open(args.results, "w") as out:
        json.dump(document, out, indent=2)


if __name__ == "__main__":
    main()
