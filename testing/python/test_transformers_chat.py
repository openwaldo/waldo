# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

import ast
import base64
import codecs
import json
from pathlib import Path
import unittest


class StreamingTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / "internal/inference/workers/transformers.py"
        module = ast.parse(path.read_text())
        # Load declarations without starting the actual model/runtime.
        module.body = [node for node in module.body if not isinstance(node, ast.Try)]
        self.output = []
        self.frames = []
        def emit(kind, **data):
            self.frames.append(json.dumps(dict(kind=kind, schema=1, **data)).encode())
            self.output.append(base64.b64decode(data["data"]).decode("utf-8", errors="strict"))
        scope = {"support": {"emit": emit}}
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

    def test_large_valid_text_is_bounded_and_lossless(self):
        text = ("é世界😀" * 100000) + "done"
        self.Output([]).write(text, final=True)
        self.assertEqual("".join(self.output), text)
        self.assertGreater(len(self.frames), 1)
        self.assertTrue(all(len(frame) < 100000 for frame in self.frames))

    def test_chunking_preserves_unicode_stop_and_prefix(self):
        text = "😀" * 40000
        output = self.Output(["終わり"])
        output.write(text + "終")
        output.write("わりhidden", final=True)
        self.assertEqual("".join(self.output), text)
        self.assertTrue(output.stopped)
        self.assertTrue(all(len(frame) < 100000 for frame in self.frames))

    def test_byte_decoder_boundaries_and_incomplete_flush(self):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        output = self.Output([])
        data = "café 世界 😀".encode() + b"\xff\xe2\x82"
        for byte in data:
            output.write(decoder.decode(bytes([byte])))
        output.write(decoder.decode(b"", final=True), final=True)
        self.assertEqual("".join(self.output), data.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    unittest.main()
