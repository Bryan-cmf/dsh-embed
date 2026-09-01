#!/usr/bin/env python3
"""dsh-embed sidecar 公共框架（SPEC §4 HTTP 契約）。

兩個 sidecar 進程（mlx_serve.py / tf_serve.py）共用本模組：
  - POST /embed/texts  {texts, dim?, instruct?, backend?} → {vectors, fingerprint, dim, ms}
  - POST /embed/image  {path, dim?, backend?}             → {vector, fingerprint, dim, ms}
  - GET  /backends     → BackendInfo[]
  - GET  /health       → {ok, uptime_s, backend, ready, loaded}
  - 全端點 X-Embed-Token 鑑權（hmac 常時比較）；僅綁定 127.0.0.1
  - 握手文件：{runtime_dir}/{name}.json ← {port, token, pid, ...}（原子寫，退出清理）
  - 空閒自退出（安全網；進程監管/重啟屬 host 插件職責，見 Plan Phase 1 T1）

僅依賴 stdlib + numpy（兩個目標 venv 均含 numpy）。
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

# ── 契約常數（SPEC §4）───────────────────────────────────────
MAX_TEXTS = 64                       # texts ≤ 64 條/次
MAX_IMAGE_BYTES = 30 * 1024 * 1024   # image path 必須存在且 ≤30MB
MAX_BODY_BYTES = 32 * 1024 * 1024    # 請求體上限
DEFAULT_DIM = 512                    # MRL 默認維度（Plan §2 鎖定 MRL-512）
MAX_INSTRUCT_CHARS = 512


class SidecarError(Exception):
    """契約內錯誤：錯誤響應一律帶 code；嵌入類錯誤帶 fingerprint 上下文（SPEC §9/Boundaries）。"""

    def __init__(self, code: str, message: str, fingerprint: str | None = None, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.fingerprint = fingerprint
        self.status = status


def log(*parts) -> None:
    print(f'[{time.strftime("%H:%M:%S")}]', *parts, file=sys.stderr, flush=True)


def mrl(vec, dim: int):
    """MRL 截斷 + 重歸一（輸入/輸出 1-D float32；與 dsh-wemm-poc 驗證配方一致）。"""
    v = np.asarray(vec, dtype=np.float32)[:dim]
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        raise SidecarError('zero_norm', f'embedding vector has zero/non-finite norm at dim={dim}')
    return v / n


def resolve_hf_snapshot(repo_id: str):
    """HF 緩存快照鎖定：返回本地緩存中最新的 commit sha（無緩存返回 None）。

    社區轉換權重無官方背書 → BackendInfo.model 帶 @sha 鎖版本（Plan §4 風險 2）。
    """
    cache = Path(os.environ.get('HF_HUB_CACHE', str(Path.home() / '.cache/huggingface/hub')))
    base = cache / f"models--{repo_id.replace('/', '--')}" / 'snapshots'
    try:
        snaps = sorted((d for d in base.iterdir() if d.is_dir()), key=lambda p: p.stat().st_mtime)
    except OSError:
        return None
    return snaps[-1].name if snaps else None


# ── 後端抽象 ────────────────────────────────────────────────
class Backend:
    """一個可嵌入後端。子類實現 _load / _embed_texts_raw / _embed_image_raw。"""

    name = ''
    repo = ''            # HF repo id（@sha 由 model 屬性拼接）
    full_dim = 0
    dims = []            # 支持的 MRL 維度（含 full_dim，升冪）
    modalities = []      # ['text'] / ['text','image']
    weights_available = True

    def __init__(self):
        self._lock = threading.RLock()   # load + 推理共用（模型非線程安全）
        self._loaded = False
        self._sha = None

    # -- 元信息 -------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def model(self) -> str:
        base = self.repo or self.name
        return f'{base}@{self._sha[:12]}' if self._sha else base

    def fingerprint(self, dim=None) -> str:
        return f'{self.name}@{dim if dim is not None else DEFAULT_DIM}'

    def info(self) -> dict:
        return {
            'name': self.name,
            'model': self.model,
            'dims': list(self.dims),
            'modalities': list(self.modalities),
            'fingerprint': self.fingerprint(DEFAULT_DIM),
            'alive': bool(self.weights_available),
            'loaded': self._loaded,
        }

    # -- 生命週期 -------------------------------------------------
    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if not self.weights_available:
                raise SidecarError('backend_unavailable',
                                   f"backend '{self.name}' weights not found",
                                   fingerprint=self.fingerprint(), status=503)
            t0 = time.perf_counter()
            self._load()
            self._loaded = True
            log(f'backend {self.name} loaded in {time.perf_counter() - t0:.1f}s')

    def _load(self) -> None:  # pragma: no cover - 子類實現
        raise NotImplementedError

    # -- 推理（子類實現；返回 full_dim 已 L2 歸一化的向量）--------
    def _embed_texts_raw(self, texts, instruct):
        raise NotImplementedError

    def _embed_image_raw(self, path: str):
        raise NotImplementedError

    # -- 對外入口（MRL 截斷+重歸一；錯誤帶 fingerprint）-----------
    def embed_texts(self, texts, dim: int, instruct) -> list:
        self.ensure_loaded()
        with self._lock:
            raw = self._embed_texts_raw(texts, instruct)
        return [mrl(v, dim).tolist() for v in raw]

    def embed_image(self, path: str, dim: int) -> list:
        self.ensure_loaded()
        with self._lock:
            raw = self._embed_image_raw(path)
        return mrl(raw, dim).tolist()


class FakeBackend(Backend):
    """確定性假後端（--fake 模式）：無權重環境下驗證 HTTP 契約用，不用於質量結論。"""

    def __init__(self, spec: 'Backend'):
        Backend.__init__(self)
        self.name, self.repo, self.full_dim = spec.name, spec.repo, spec.full_dim
        self.dims, self.modalities = list(spec.dims), list(spec.modalities)

    def _load(self) -> None:
        pass

    def _vec_for(self, key: str):
        digest = __import__('hashlib').sha256(f'{self.name}:{key}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        v = rng.standard_normal(self.full_dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def _embed_texts_raw(self, texts, instruct):
        # instruct 鍵入向量空間，使「instruct 生效」在 fake 模式亦可觀測
        return [self._vec_for(f'{instruct or ""}|{t}') for t in texts]

    def _embed_image_raw(self, path: str):
        p = Path(path).expanduser()
        try:
            with open(p, 'rb') as f:
                head = f.read(64)
        except OSError:
            head = b''
        return self._vec_for(f'img:{p.name}:{head!r}')


# ── 應用與 HTTP 服務 ─────────────────────────────────────────
class SidecarApp:
    def __init__(self, name: str, backends: list, default_backend: str,
                 runtime_dir, port: int = 0, idle_timeout_sec: int = 900):
        self.name = name
        self.backends = {b.name: b for b in backends}
        if default_backend not in self.backends:
            raise ValueError(f'default backend {default_backend!r} not in {sorted(self.backends)}')
        self.default_backend = default_backend
        self.runtime_dir = Path(runtime_dir)
        self.port = int(port)
        self.idle_timeout_sec = int(idle_timeout_sec)
        self.token = secrets.token_hex(32)   # 32B → 64 hex chars（SPEC §4）
        self.started = time.time()
        self.last_activity = time.time()     # 僅 embed 調用刷新（health 輪詢不續命）
        self.handshake_path = self.runtime_dir / f'{self.name}.json'
        self.eager_names: list = []          # 由 serve() 填入

    def touch(self) -> None:
        self.last_activity = time.time()

    def ready(self) -> bool:
        return all(self.backends[n].loaded for n in self.eager_names if n in self.backends)

    def handshake_payload(self) -> dict:
        return {
            'name': self.name,
            'port': self.port,
            'token': self.token,
            'pid': os.getpid(),
            'startedAt': int(self.started * 1000),
            'backends': [b.name for b in self.backends.values() if b.loaded],
        }

    def write_handshake(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.handshake_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.handshake_payload(), ensure_ascii=False, indent=1))
        os.replace(tmp, self.handshake_path)   # 原子替換
        log(f'handshake written: {self.handshake_path} port={self.port}')

    def remove_handshake(self) -> None:
        try:
            data = json.loads(self.handshake_path.read_text())
            if data.get('pid') == os.getpid():   # 只清自己的文件（防誤刪新進程的）
                self.handshake_path.unlink()
                log(f'handshake removed: {self.handshake_path}')
        except (OSError, ValueError):
            pass

    # -- 請求處理 -------------------------------------------------
    def _pick_backend(self, body: dict) -> 'Backend':
        name = body.get('backend') or self.default_backend
        if not isinstance(name, str) or name not in self.backends:
            raise SidecarError('unknown_backend',
                               f'unknown backend {name!r}; known: {sorted(self.backends)}')
        return self.backends[name]

    def _pick_dim(self, body: dict, backend: 'Backend') -> int:
        dim = body.get('dim', DEFAULT_DIM)
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise SidecarError('invalid_dim', f'dim must be int, got {type(dim).__name__}',
                               fingerprint=backend.fingerprint())
        if dim not in backend.dims:
            raise SidecarError('invalid_dim',
                               f'dim {dim} not supported by {backend.name}; supported: {backend.dims}',
                               fingerprint=backend.fingerprint(dim))
        return dim

    def handle_health(self) -> dict:
        return {
            'ok': True,
            'uptime_s': round(time.time() - self.started, 1),
            'backend': self.default_backend,
            'ready': self.ready(),
            'loaded': [b.name for b in self.backends.values() if b.loaded],
        }

    def handle_backends(self) -> list:
        return [b.info() for b in self.backends.values()]

    def handle_embed_texts(self, body: dict) -> dict:
        backend = self._pick_backend(body)
        if 'text' not in backend.modalities:
            raise SidecarError('unsupported_modality',
                               f"backend '{backend.name}' does not support text embedding",
                               fingerprint=backend.fingerprint())
        texts = body.get('texts')
        if not isinstance(texts, list):
            raise SidecarError('invalid_texts', 'texts must be a list of strings',
                               fingerprint=backend.fingerprint())
        if len(texts) == 0:
            raise SidecarError('empty_texts', 'texts must not be empty',
                               fingerprint=backend.fingerprint())
        if len(texts) > MAX_TEXTS:
            raise SidecarError('too_many_texts',
                               f'texts exceeds limit {MAX_TEXTS} (got {len(texts)})',
                               fingerprint=backend.fingerprint())
        if any(not isinstance(t, str) for t in texts):
            raise SidecarError('invalid_texts', 'every text item must be a string',
                               fingerprint=backend.fingerprint())
        instruct = body.get('instruct')
        if instruct is not None:
            if not isinstance(instruct, str):
                raise SidecarError('invalid_instruct', 'instruct must be a string',
                                   fingerprint=backend.fingerprint())
            if len(instruct) > MAX_INSTRUCT_CHARS:
                raise SidecarError('invalid_instruct', f'instruct exceeds {MAX_INSTRUCT_CHARS} chars',
                                   fingerprint=backend.fingerprint())
            if backend.name.startswith('wemm'):
                instruct = None   # instruct 僅 Qwen3 家族生效（SPEC §3）；WeMM 忽略
        dim = self._pick_dim(body, backend)
        t0 = time.perf_counter()
        vectors = backend.embed_texts(texts, dim, instruct)
        self.touch()
        return {
            'vectors': vectors,
            'fingerprint': backend.fingerprint(dim),
            'dim': dim,
            'ms': round((time.perf_counter() - t0) * 1000, 1),
        }

    def handle_embed_image(self, body: dict) -> dict:
        backend = self._pick_backend(body)
        if 'image' not in backend.modalities:
            raise SidecarError('unsupported_modality',
                               f"backend '{backend.name}' does not support image embedding",
                               fingerprint=backend.fingerprint())
        path = body.get('path')
        if not isinstance(path, str) or not path:
            raise SidecarError('invalid_path', 'path must be a non-empty string',
                               fingerprint=backend.fingerprint())
        p = Path(path).expanduser()
        if not p.is_file():
            raise SidecarError('image_not_found', f'image file not found: {path}',
                               fingerprint=backend.fingerprint())
        size = p.stat().st_size
        if size > MAX_IMAGE_BYTES:
            raise SidecarError('image_too_large',
                               f'image exceeds {MAX_IMAGE_BYTES} bytes (got {size})',
                               fingerprint=backend.fingerprint())
        dim = self._pick_dim(body, backend)
        t0 = time.perf_counter()
        vector = backend.embed_image(str(p), dim)
        self.touch()
        return {
            'vector': vector,
            'fingerprint': backend.fingerprint(dim),
            'dim': dim,
            'ms': round((time.perf_counter() - t0) * 1000, 1),
        }


def _make_handler(app: SidecarApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'dsh-embed-sidecar/1.0'
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):   # 精簡訪問日誌 → stderr
            log(f'{self.address_string()} {fmt % args}')

        # -- 基礎設施 -------------------------------------------
        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_sidecar_error(self, err: SidecarError) -> None:
            e = {'code': err.code, 'message': err.message}
            if err.fingerprint:
                e['fingerprint'] = err.fingerprint
            self._send_json(err.status, {'error': e})

        def _auth(self) -> None:
            got = self.headers.get('X-Embed-Token', '')
            if not hmac.compare_digest(got, app.token):
                raise SidecarError('unauthorized', 'missing or invalid X-Embed-Token', status=401)

        def _read_json(self) -> dict:
            try:
                length = int(self.headers.get('Content-Length') or 0)
            except ValueError:
                raise SidecarError('invalid_request', 'bad Content-Length')
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                raise SidecarError('body_too_large',
                                   f'request body exceeds {MAX_BODY_BYTES} bytes', status=413)
            raw = self.rfile.read(length) if length > 0 else b''
            self._last_body = raw
            if not raw:
                raise SidecarError('invalid_json', 'request body must be a JSON object')
            try:
                body = json.loads(raw.decode('utf-8'))
            except (ValueError, UnicodeDecodeError) as exc:
                raise SidecarError('invalid_json', f'request body is not valid JSON: {exc}')
            if not isinstance(body, dict):
                raise SidecarError('invalid_json', 'request body must be a JSON object')
            return body

        def _drain(self, n: int, cap: int = 128 * 1024 * 1024) -> None:
            """413 後排空未讀請求體：避免客戶端 sendall 被 RST 中斷、丟失錯誤響應。"""
            remaining = min(max(n, 0), cap)
            try:
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except OSError:
                pass

        def _error_fingerprint(self):
            """500 響應的 fingerprint 上下文：盡力從最近的請求體取 backend 名。"""
            try:
                body = json.loads((self._last_body or b'').decode('utf-8'))
                name = body.get('backend') or app.default_backend if isinstance(body, dict) \
                    else app.default_backend
                b = app.backends.get(name)
                return b.fingerprint() if b else None
            except Exception:
                return None

        # -- 路由 -------------------------------------------------
        def do_GET(self):
            self._route('GET')

        def do_POST(self):
            self._route('POST')

        def _route(self, method: str) -> None:
            self._last_body = None
            path = urlparse(self.path).path.rstrip('/') or '/'
            known = {'/health', '/backends', '/embed/texts', '/embed/image'}
            try:
                self._auth()
                if path not in known:
                    raise SidecarError('unknown_path', f'no such endpoint: {path}', status=404)
                if method == 'GET' and path == '/health':
                    self._send_json(200, app.handle_health())
                elif method == 'GET' and path == '/backends':
                    self._send_json(200, app.handle_backends())
                elif path == '/embed/texts' and method == 'POST':
                    self._send_json(200, app.handle_embed_texts(self._read_json()))
                elif path == '/embed/image' and method == 'POST':
                    self._send_json(200, app.handle_embed_image(self._read_json()))
                else:
                    raise SidecarError('method_not_allowed',
                                       f'{method} not allowed on {path}', status=405)
            except SidecarError as err:
                self._send_sidecar_error(err)
                if err.status == 413:
                    try:
                        self._drain(int(self.headers.get('Content-Length') or 0))
                    except ValueError:
                        pass
                    self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:   # 推理內部錯誤：500 + fingerprint 上下文 + 堆疊日誌
                log('INTERNAL ERROR:', repr(exc))
                traceback.print_exc()
                self._send_json(500, {'error': {'code': 'internal', 'message': str(exc),
                                                'fingerprint': self._error_fingerprint()}})

        _last_body = None

    return Handler


def serve(app: SidecarApp, eager: list) -> None:
    """綁定 127.0.0.1（port=0 隨機）→ eager 加載 → 寫握手文件 → 服務 + 空閒看門狗。"""
    server = ThreadingHTTPServer(('127.0.0.1', app.port), _make_handler(app))
    server.daemon_threads = True
    app.port = server.server_address[1]     # port=0 → 內核分配
    app.eager_names = [n for n in eager if n in app.backends]

    for name in app.eager_names:            # 握手文件在 eager 加載完成後才寫（發現即就緒）
        backend = app.backends[name]
        try:
            backend.ensure_loaded()
        except Exception as exc:
            log(f'FATAL: eager load {name} failed: {exc}')
            raise SystemExit(3)

    app.write_handshake()
    print(f'SIDECAR_READY name={app.name} port={app.port} pid={os.getpid()} '
          f'backends={",".join(app.handshake_payload()["backends"])}', flush=True)

    # -- 信號：清握手文件後退出 ----------------------------------
    def _shutdown(signum, _frame):
        log(f'signal {signum} → shutdown')
        app.remove_handshake()
        os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # -- 空閒看門狗：最後一次 embed 調用後 idle_timeout_sec 退出 ---
    def _watchdog():
        while True:
            step = min(5, app.idle_timeout_sec) if app.idle_timeout_sec > 0 else 5
            time.sleep(step)
            if app.idle_timeout_sec <= 0:
                continue
            idle = time.time() - app.last_activity
            if idle >= app.idle_timeout_sec:
                log(f'idle {idle:.0f}s >= {app.idle_timeout_sec}s → idle exit')
                app.remove_handshake()
                os._exit(0)

    threading.Thread(target=_watchdog, daemon=True, name='idle-watchdog').start()

    log(f'serving on http://127.0.0.1:{app.port} (idle_timeout={app.idle_timeout_sec}s)')
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        app.remove_handshake()
        server.server_close()


def base_arg_parser(desc: str, default_name: str):
    import argparse
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('--runtime-dir', default=str(Path.home() / '.dsh/run/dsh-embed'),
                   help='握手文件目錄（默認 ~/.dsh/run/dsh-embed）')
    p.add_argument('--name', default=default_name, help='sidecar 名（握手文件 {name}.json）')
    p.add_argument('--port', type=int, default=0, help='監聽端口（默認 0=隨機；僅綁 127.0.0.1）')
    p.add_argument('--idle-timeout-sec', type=int, default=900,
                   help='最後一次 embed 調用後空閒退出秒數（默認 900；0=禁用）')
    p.add_argument('--eager', default='', help='逗號分隔的啟動即加載後端（默認各 sidecar 自帶）')
    p.add_argument('--lazy', action='store_true', help='全部惰性加載（覆蓋默認 eager）')
    p.add_argument('--fake', action='store_true', help='確定性假後端（契約測試用，無需權重）')
    p.add_argument('--online', action='store_true', help='允許 HF 聯網（默認離線 HF_HUB_OFFLINE=1）')
    return p


def offline_env() -> None:
    """默認離線：權重已預下載，出網流量為零（SPEC §12.5）。--online 可解除。"""
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
