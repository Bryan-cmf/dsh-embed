#!/usr/bin/env python3
"""sidecar-tf 語義健全性測試（Qwen3-4B fp16 + WeMM-2B fp16 惰性 fallback）。

必須用 venv-tf 的 python 運行（需要 numpy；HTTP 走 stdlib）：
  venv-tf/bin/python tf_sanity.py --handshake RUNTIME/tf.json [--test-wemm]

覆蓋：
  - qwen3-4b-fp16：相關/無關對區分度、instruct 前綴生效、批量 64、MRL norm=1
  - wemm2b-fp16（--test-wemm）：惰性加載可用、文本語義對區分度、圖像嵌入 norm=1
    （WeMM fp16 是官方路徑本身，無需另行 parity；質量由 t7 三數據集回歸驗收）
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import SidecarClient  # noqa: E402

T_REL_A = '工具 write 遇 FS_STALE_VERSION 時,可先 rm 目標檔再重寫'
T_REL_B = '文件寫入遇到版本過期錯誤怎麼處理'
T_UNREL = '今天天氣很好適合出海釣魚'

FAILURES = []


def check(cond, name, detail=''):
    print(f'  {"PASS" if cond else "FAIL"} {name}' + ('' if cond else f'  {detail}'))
    if not cond:
        FAILURES.append(name)


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--handshake', required=True)
    ap.add_argument('--test-wemm', action='store_true', help='同時測 wemm2b-fp16（惰性加載）')
    args = ap.parse_args()

    c = SidecarClient.from_handshake(args.handshake, timeout=600)

    print('== qwen3-4b-fp16 ==')
    st, body = c.embed_texts([T_REL_A, T_REL_B, T_UNREL])
    v = body.get('vectors', []) if st == 200 else []
    check(st == 200 and len(v) == 3, 'qwen3_triple_200', f'{st} {str(body)[:160]}')
    if len(v) == 3:
        rel, unrel = cos(v[0], v[1]), cos(v[0], v[2])
        check(rel > unrel + 0.10, 'qwen3_related_beats_unrelated', f'rel={rel:.4f} unrel={unrel:.4f}')
        check(all(abs(np.linalg.norm(x) - 1) < 2e-3 for x in v), 'qwen3_l2_normalized')

    instruct = 'Given a query, retrieve relevant memory notes'
    st1, b1 = c.embed_texts([T_REL_B], instruct=instruct)
    st2, b2 = c.embed_texts([T_REL_B])
    ok = st1 == 200 and st2 == 200
    diff = max(abs(x - y) for x, y in zip(b1['vectors'][0], b2['vectors'][0])) if ok else -1
    check(ok and diff > 1e-4, 'qwen3_instruct_prefix_changes_query_vec', f'maxdiff={diff}')

    st, body = c.embed_texts([f'批量條目 {i}' for i in range(64)])
    check(st == 200 and len(body.get('vectors', [])) == 64, 'qwen3_batch_64', str(st))

    st, body = c.embed_texts(['MRL'], dim=512)
    vec = body.get('vectors', [[]])[0] if st == 200 else []
    check(len(vec) == 512 and abs(np.linalg.norm(vec) - 1) < 2e-3
          and body.get('fingerprint') == 'qwen3-4b-fp16@512',
          'qwen3_mrl_512_norm1', f'len={len(vec)} fp={body.get("fingerprint")}')

    if args.test_wemm:
        print('== wemm2b-fp16（惰性加載；首次調用觸發加載）==')
        st, body = c.embed_texts([T_REL_A, T_REL_B, T_UNREL], backend='wemm2b-fp16')
        v = body.get('vectors', []) if st == 200 else []
        check(st == 200 and len(v) == 3, 'wemm_fp16_lazy_load_and_embed',
              f'{st} {str(body)[:200]}')
        if len(v) == 3:
            rel, unrel = cos(v[0], v[1]), cos(v[0], v[2])
            check(rel > unrel + 0.10, 'wemm_fp16_related_beats_unrelated',
                  f'rel={rel:.4f} unrel={unrel:.4f}')
            check(body.get('fingerprint') == 'wemm2b-fp16@512', 'wemm_fp16_fingerprint',
                  str(body.get('fingerprint')))

        with tempfile.TemporaryDirectory() as td:
            from contract_tests import make_png
            png = Path(td) / 'sanity.png'
            png.write_bytes(make_png())
            st, body = c.embed_image(str(png), backend='wemm2b-fp16', dim=512)
            vec = body.get('vector', []) if st == 200 else []
            check(st == 200 and len(vec) == 512
                  and abs(np.linalg.norm(vec) - 1) < 2e-3,
                  'wemm_fp16_image_norm1', f'{st} len={len(vec)}')

        st, body = c.backends()
        loaded = [b['name'] for b in body if b.get('loaded')] if st == 200 else []
        check('wemm2b-fp16' in loaded and 'qwen3-4b-fp16' in loaded,
              'backends_both_loaded_after_use', str(loaded))

    print(f'TF_SANITY_{"OK" if not FAILURES else "FAIL"} ({len(FAILURES)} failures)')
    if FAILURES:
        for f in FAILURES:
            print(f'  - {f}', file=sys.stderr)
    return 0 if not FAILURES else 1


if __name__ == '__main__':
    sys.exit(main())
