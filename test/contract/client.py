#!/usr/bin/env python3
"""契約測試共用 HTTP 客戶端（stdlib only，任何 python ≥3.9 可跑）。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path


class SidecarClient:
    def __init__(self, url: str, token: str, timeout: float = 300.0):
        self.url = url.rstrip('/')
        self.token = token
        self.timeout = timeout

    @classmethod
    def from_handshake(cls, path, timeout: float = 300.0) -> 'SidecarClient':
        data = json.loads(Path(path).read_text())
        return cls(f"http://127.0.0.1:{data['port']}", data['token'], timeout=timeout)

    def call(self, method: str, path: str, body=None, token: str | None = None,
             raw_body: bytes | None = None, timeout: float | None = None):
        """返回 (status, json_or_None)。非 2xx 且 body 是 JSON 時解析 error 結構。"""
        url = self.url + path
        data = raw_body
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header('Content-Type', 'application/json')
        req.add_header('X-Embed-Token', self.token if token is None else token)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                payload = resp.read()
                return resp.status, (json.loads(payload) if payload else None)
        except urllib.error.HTTPError as err:
            payload = err.read()
            try:
                return err.code, (json.loads(payload) if payload else None)
            except ValueError:
                return err.code, {'raw': payload.decode('utf-8', 'replace')}

    # -- 便捷封裝 -------------------------------------------------
    def health(self, **kw):
        return self.call('GET', '/health', **kw)

    def backends(self, **kw):
        return self.call('GET', '/backends', **kw)

    def embed_texts(self, texts, dim=None, instruct=None, backend=None, **kw):
        body = {'texts': texts}
        if dim is not None:
            body['dim'] = dim
        if instruct is not None:
            body['instruct'] = instruct
        if backend is not None:
            body['backend'] = backend
        return self.call('POST', '/embed/texts', body, **kw)

    def embed_image(self, path, dim=None, backend=None, **kw):
        body = {'path': str(path)}
        if dim is not None:
            body['dim'] = dim
        if backend is not None:
            body['backend'] = backend
        return self.call('POST', '/embed/image', body, **kw)


def wait_for_handshake(path, timeout: float = 180.0, expect_pid_alive: bool = True):
    """輪詢握手文件出現且進程存活；返回 payload dict。"""
    p = Path(path)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data.get('port'), int) and data.get('token'):
                    return data
            except ValueError:
                pass
        time.sleep(0.5)
    raise TimeoutError(f'handshake file not ready within {timeout}s: {path}')
