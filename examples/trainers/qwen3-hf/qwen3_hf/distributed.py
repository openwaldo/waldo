# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Node-local protocol fanout. Only local rank zero reads coordinator stdin."""
import json
import os
from pathlib import Path
import socket
import stat
import time

from . import wire


class LocalStream:
    def __init__(self):
        self.rank = int(os.environ['LOCAL_RANK'])
        self.size = int(os.environ['LOCAL_WORLD_SIZE'])
        self.peers = []
        self.listener = None
        self.source = None
        self.directory_fd = os.open(os.environ['HOME'], os.O_RDONLY | os.O_DIRECTORY)
        self.path = Path(f'/proc/self/fd/{self.directory_fd}') / 'trainer-stream.sock'
        if self.rank == 0:
            if self.path.exists():
                if not stat.S_ISSOCK(self.path.lstat().st_mode):
                    raise ValueError("local stream path is not a socket")
                self.path.unlink()
            self.listener = socket.socket(socket.AF_UNIX)
            self.listener.bind(str(self.path))
            self.listener.listen(self.size)
            self.listener.settimeout(60)
            for _ in range(self.size - 1):
                peer, _ = self.listener.accept()
                self.peers.append(peer)
        else:
            peer = socket.socket(socket.AF_UNIX)
            deadline = time.monotonic() + 60
            while True:
                try:
                    peer.connect(str(self.path))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('local stream leader unavailable')
                    time.sleep(0.05)
            self.peers.append(peer)
            self.source = peer.makefile('r', encoding='utf-8')

    def read(self):
        value = wire.read(self.source)
        if self.rank == 0:
            if value.get("kind") == "end" and wire.read(allow_eof=True) is not None:
                raise ValueError("frames after end")
            data = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
            for peer in self.peers:
                peer.sendall(data)
        return value

    def close(self):
        if self.source:
            self.source.close()
        for peer in self.peers:
            peer.close()
        if self.listener:
            self.listener.close()
            self.path.unlink(missing_ok=True)
        os.close(self.directory_fd)
