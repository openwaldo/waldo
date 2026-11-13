# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CODE = '''
from qwen3_hf.distributed import LocalStream
stream = LocalStream()
try:
    assert stream.read() == {"kind": "end", "protocol": 1}
finally:
    stream.close()
'''


@unittest.skipUnless(sys.platform == 'linux', 'Linux Unix socket transport')
class ExternalLocalStream(unittest.TestCase):
    def run_stream(self, home):
        return subprocess.run([sys.executable, '-c', CODE], cwd=ROOT/'examples/trainers/qwen3-hf',
                              env=dict(os.environ, HOME=str(home), LOCAL_RANK='0', LOCAL_WORLD_SIZE='1'),
                              input='{"kind":"end","protocol":1}\n', text=True,
                              capture_output=True, timeout=5)

    def test_long_scratch_path_and_interrupted_socket(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)/('x'*100)
            home.mkdir()
            fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
            try:
                stale = socket.socket(socket.AF_UNIX)
                stale.bind(f'/proc/self/fd/{fd}/trainer-stream.sock')
                stale.close()
            finally:
                os.close(fd)
            result = self.run_stream(home)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((home/'trainer-stream.sock').exists())

    def test_existing_regular_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'trainer-stream.sock'
            path.write_text('preserve')
            result = self.run_stream(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('not a socket', result.stderr)
            self.assertEqual(path.read_text(), 'preserve')
