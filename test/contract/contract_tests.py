#!/usr/bin/env python3
"""dsh-embed 雙 sidecar 共用契約測試（SPEC §4 / §10 契約層）。

兩個 sidecar（mlx / tf）跑同一組用例：
  schema、X-Embed-Token 鑑權、僅 127.0.0.1 監聽、texts≤64 批量上限、
  image 路徑校驗（存在 + ≤30MB）、MRL 截斷+重歸一（norm=1）、
  fingerprint 格式 {backend}@{dim}、錯誤響應帶 fingerprint。

stdlib only。用法：
  python3 contract_tests.py --url http://127.0.0.1:PORT --token TOK --name mlx [--fake]
  python3 contract_tests.py --url http://127.0.0.1:PORT --token TOK --name tf  [--fake] [--test-wemm]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import SidecarClient  # noqa: E402

FAILURES = []
COUNT = 0


def check(cond: bool, name: str, detail: str = '') -> bool:
    global COUNT
    COUNT += 1
    if cond:
        print(f'  PASS {name}')
    else:
        print(f'  FAIL {name}  {detail}')
        FAILURES.append(f'{name}: {detail}')
    return cond


def norm(vec) -> float:
    return math.sqrt(sum(x * x for x in vec))


def err_code(payload) -> str:
    return (payload or {}).get('error', {}).get('code', '')


def err_fp(payload):
    return (payload or {}).get('error', {}).get('fingerprint')


def make_png(w: int = 224, h: int = 224) -> bytes:
    """純 stdlib 生成一張有結構的 RGB PNG（真實視覺後端可接受）。"""
    rows = bytearray()
    for y in range(h):
        row = bytearray(b'\x00')
        for x in range(w):
            row += bytes(((x * 3) % 256, (y * 5 + x) % 256, (x + y) % 256))
        rows += row

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr)
            + chunk(b'IDAT', zlib.compress(bytes(rows))) + chunk(b'IEND', b''))


def check_bind_loopback(port: int):
    """契約：僅監聽 127.0.0.1。用 lsof 驗證 LISTEN 地址（無 lsof 則跳過）。"""
    try:
        out = subprocess.run(['lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN'],
                             capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        check(True, 'bind_loopback(lsof unavailable, skipped)')
        return
    lines = [ln for ln in out.splitlines() if 'LISTEN' in ln]
    ok = bool(lines) and all(ln.split()[8].startswith('127.0.0.1:') for ln in lines)
    check(ok, 'bind_loopback_127.0.0.1', f'lsof: {lines!r}')


def run_suite(c: SidecarClient, name: str, fake: bool, test_wemm: bool) -> None:
    visual_backend = 'wemm2b-mlx4b' if name == 'mlx' else 'wemm2b-fp16'
    default_backend = 'wemm2b-mlx4b' if name == 'mlx' else 'qwen3-4b-fp16'
    image_backend_available = (name == 'mlx') or test_wemm or fake

    print(f'== [{name}] health & auth ==')
    st, body = c.health(token='wrong-token')
    check(st == 401 and err_code(body) == 'unauthorized', 'health_wrong_token_401', f'{st} {body}')
    st, body = c.health()
    check(st == 200 and body.get('ok') is True and isinstance(body.get('uptime_s'), (int, float))
          and body.get('backend') == default_backend,
          'health_schema', f'{st} {body}')

    print(f'== [{name}] /backends schema ==')
    st, body = c.backends()
    ok = st == 200 and isinstance(body, list) and len(body) >= 1
    check(ok, 'backends_list', f'{st} {body}')
    for b in body if ok else []:
        tag = b.get('name', '?')
        check(isinstance(b.get('name'), str) and b['name'], f'backends[{tag}].name')
        check(isinstance(b.get('model'), str) and b['model'], f'backends[{tag}].model_nonempty')
        check(isinstance(b.get('dims'), list) and all(isinstance(d, int) for d in b['dims'])
              and b['dims'] == sorted(b['dims']) and 512 in b['dims'],
              f'backends[{tag}].dims_sorted_contains_512', str(b.get('dims')))
        check(set(b.get('modalities', [])) <= {'text', 'image'} and b.get('modalities'),
              f'backends[{tag}].modalities')
        fp = b.get('fingerprint', '')
        check(isinstance(fp, str) and fp == f"{b.get('name')}@512",
              f'backends[{tag}].fingerprint_format', fp)
        check(b.get('alive') is True, f'backends[{tag}].alive')
    names = {b['name'] for b in body} if ok else set()
    check(default_backend in names, 'backends_contains_default', str(names))

    print(f'== [{name}] /embed/texts ==')
    st, body = c.embed_texts(['契約測試：hello world 你好世界'])
    v = (body or {}).get('vectors') or [[]]
    check(st == 200 and len(v) == 1, 'texts_basic_200', f'{st} {str(body)[:200]}')
    check(abs(norm(v[0]) - 1.0) < 2e-3, 'texts_default_dim_norm1', f'norm={norm(v[0]) if v[0] else "?"}')
    check(body.get('dim') == 512 and body.get('fingerprint') == f'{default_backend}@512'
          and isinstance(body.get('ms'), (int, float)),
          'texts_default_dim_512_fp', f"{body.get('dim')} {body.get('fingerprint')}")

    st, body = c.embed_texts(['MRL 截斷測試'], dim=1024)
    v = (body or {}).get('vectors') or [[]]
    check(st == 200 and len(v[0]) == 1024 and abs(norm(v[0]) - 1.0) < 2e-3
          and body.get('fingerprint') == f'{default_backend}@1024',
          'texts_mrl_1024_truncate_renorm', f'{st} len={len(v[0]) if v[0] else 0}')

    st, body = c.embed_texts([f'批量條目 {i}' for i in range(64)])
    check(st == 200 and len(body.get('vectors', [])) == 64, 'texts_batch_64_ok', str(st))
    st, body = c.embed_texts([f'超限 {i}' for i in range(65)])
    check(st == 400 and err_code(body) == 'too_many_texts' and err_fp(body),
          'texts_batch_65_rejected_with_fp', f'{st} {body}')
    st, body = c.embed_texts([])
    check(st == 400 and err_code(body) == 'empty_texts', 'texts_empty_rejected', f'{st} {body}')
    st, body = c.call('POST', '/embed/texts', {'texts': 'not-a-list'})
    check(st == 400 and err_code(body) == 'invalid_texts', 'texts_non_list_rejected', f'{st} {body}')
    st, body = c.call('POST', '/embed/texts', {'texts': ['ok', 42]})
    check(st == 400 and err_code(body) == 'invalid_texts', 'texts_non_str_item_rejected', f'{st} {body}')
    st, body = c.embed_texts([' 維度校驗 '], dim=100)
    check(st == 400 and err_code(body) == 'invalid_dim' and err_fp(body) == f'{default_backend}@100',
          'texts_invalid_dim_rejected_with_fp', f'{st} {body}')
    st, body = c.embed_texts(['後端校驗'], backend='no-such-backend')
    check(st == 400 and err_code(body) == 'unknown_backend', 'texts_unknown_backend_rejected', f'{st} {body}')
    st, body = c.embed_texts(['鑑權'], token='bad')
    check(st == 401, 'texts_wrong_token_401', str(st))

    print(f'== [{name}] instruct 語義（僅 Qwen3 家族生效）==')
    instruct = 'Given a query, retrieve relevant memory notes'
    st1, b1 = c.embed_texts(['文件寫入遇到版本過期錯誤怎麼處理'], instruct=instruct)
    st2, b2 = c.embed_texts(['文件寫入遇到版本過期錯誤怎麼處理'])
    ok = st1 == 200 and st2 == 200
    if ok:
        va, vb = b1['vectors'][0], b2['vectors'][0]
        maxdiff = max(abs(x - y) for x, y in zip(va, vb))
        if default_backend.startswith('qwen3'):
            check(maxdiff > 1e-4, 'instruct_changes_qwen3_vector', f'maxdiff={maxdiff}')
        else:
            # WeMM 忽略 instruct；容忍 Metal 內核級浮點非確定性（語義生效時差異 >1e-2）
            check(maxdiff < 1e-6, 'instruct_ignored_on_wemm', f'maxdiff={maxdiff}')
    else:
        check(False, 'instruct_endpoint_ok', f'{st1} {st2}')

    print(f'== [{name}] /embed/image ==')
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / 'contract.png'
        png.write_bytes(make_png())

        if image_backend_available:
            st, body = c.embed_image(png, dim=512, backend=visual_backend if name == 'tf' else None)
            vec = (body or {}).get('vector') or []
            check(st == 200 and len(vec) == 512 and abs(norm(vec) - 1.0) < 2e-3
                  and body.get('fingerprint') == f'{visual_backend}@512',
                  'image_basic_200_norm1_fp', f'{st} len={len(vec)} fp={body.get("fingerprint")}')

        st, body = c.embed_image(str(Path(td) / 'nope.png'), backend=visual_backend if name == 'tf' else None)
        check(st == 400 and err_code(body) == 'image_not_found' and err_fp(body),
              'image_not_found_400_with_fp', f'{st} {body}')

        big = Path(td) / 'big.png'
        big.write_bytes(b'\x89PNG' + b'\x00' * (30 * 1024 * 1024 + 1))
        st, body = c.embed_image(str(big), backend=visual_backend if name == 'tf' else None)
        check(st == 400 and err_code(body) == 'image_too_large' and err_fp(body),
              'image_over_30MB_rejected', f'{st} {str(body)[:120]}')

        if name == 'tf':
            st, body = c.embed_image(str(png), backend='qwen3-4b-fp16')
            check(st == 400 and err_code(body) == 'unsupported_modality' and err_fp(body),
                  'image_on_text_only_backend_rejected', f'{st} {body}')

    print(f'== [{name}] HTTP 細節 ==')
    st, body = c.call('GET', '/nope')
    check(st == 404 and err_code(body) == 'unknown_path', 'unknown_path_404', f'{st} {body}')
    st, body = c.call('GET', '/embed/texts')
    check(st == 405 and err_code(body) == 'method_not_allowed', 'wrong_method_405', f'{st} {body}')
    st, body = c.call('POST', '/embed/texts', raw_body=b'{invalid json')
    check(st == 400 and err_code(body) == 'invalid_json', 'malformed_json_400', f'{st} {body}')
    huge = b'{"texts": ["' + b'x' * (33 * 1024 * 1024) + b'"]}'
    st, body = c.call('POST', '/embed/texts', raw_body=huge)
    check(st == 413 and err_code(body) == 'body_too_large', 'body_over_32MB_413', str(st))

    from urllib.parse import urlparse
    check_bind_loopback(urlparse(c.url).port)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--token', required=True)
    ap.add_argument('--name', choices=['mlx', 'tf'], required=True)
    ap.add_argument('--fake', action='store_true')
    ap.add_argument('--test-wemm', action='store_true',
                    help='tf 側同時測 wemm2b-fp16（惰性加載，真實模式較慢）')
    args = ap.parse_args()

    c = SidecarClient(args.url, args.token)
    run_suite(c, args.name, args.fake, args.test_wemm)

    print(f'CONTRACT_{"PASS" if not FAILURES else "FAIL"} name={args.name} '
          f'{COUNT - len(FAILURES)}/{COUNT}')
    if FAILURES:
        print('failures:', file=sys.stderr)
        for f in FAILURES:
            print(f'  - {f}', file=sys.stderr)
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
