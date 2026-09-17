# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

import ast
import base64
from pathlib import Path
import unittest


class StreamingTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / "internal/inference/workers/transformers.py"
        module = ast.parse(path.read_text())
        # Load declarations without starting the actual model/runtime.
        module.body = [node for node in module.body if not isinstance(node, ast.Try)]
        self.output = []
        scope = {"support": {"emit": lambda kind, **data: self.output.append(base64.b64decode(data["data"]).decode())}}
        exec(compile(module, str(path), "exec"), scope)
        self.Output = scope["Output"]

    def test_stop_split_across_frames(self):
        output = self.Output(["<STOP>"])
        for text in ["日本 hello <S", "TO", "P>secret"]:
            output.write(text)
        output.write("more", final=True)
        self.assertEqual("".join(self.output), "日本 hello ")
        self.assertTrue(output.stopped)

    def test_incomplete_stop_is_flushed(self):
        output = self.Output(["<STOP>"])
        output.write("hello <ST")
        output.write("", final=True)
        self.assertEqual("".join(self.output), "hello <ST")

    def test_first_stop_wins(self):
        output = self.Output(["stop", "end"])
        output.write("hello end stop")
        self.assertEqual("".join(self.output), "hello ")


if __name__ == "__main__":
    unittest.main()
