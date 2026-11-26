#!/usr/bin/env python3
# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Merge two platform pylocks without resolving or changing package versions."""
import datetime
import json
from pathlib import Path
import sys
import tomllib


def literal(value):
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {literal(v)}" for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    if isinstance(value, datetime.datetime):
        return value.isoformat().replace("+00:00", "Z")
    return json.dumps(value)


def merge(first, second):
    a, b = (tomllib.loads(Path(p).read_text()) for p in (first, second))
    ap, bp = (v.pop("packages") for v in (a, b))
    if a != b:
        raise ValueError("lock headers differ")
    packages = {p["name"]: p for p in ap}
    if set(packages) != {p["name"] for p in bp}:
        raise ValueError("package sets differ; explicit platform markers required")
    for package in bp:
        existing = packages[package["name"]]
        if {k: v for k, v in existing.items() if k != "wheels"} != {k: v for k, v in package.items() if k != "wheels"}:
            raise ValueError(f"package identity/markers differ: {package['name']}")
        wheels = {w["url"]: w for w in existing["wheels"]}
        for wheel in package["wheels"]:
            if wheel["url"] in wheels and wheels[wheel["url"]] != wheel:
                raise ValueError("wheel evidence differs")
            wheels[wheel["url"]] = wheel
        existing["wheels"] = [wheels[url] for url in sorted(wheels)]
    lines = [f"{k} = {literal(v)}" for k, v in a.items()]
    for name in sorted(packages):
        lines += ["", "[[packages]]"]
        for key, value in packages[name].items():
            if key == "wheels":
                lines += ["wheels = [", *(f"    {literal(w)}," for w in value), "]"]
            else:
                lines.append(f"{key} = {literal(value)}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(merge(*sys.argv[1:]), end="")
