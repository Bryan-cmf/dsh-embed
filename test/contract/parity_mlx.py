#!/usr/bin/env python3
"""sidecar-mlx 移植正確性測試（t2 驗收核心）。

對固定輸入（D1 記憶文本 + D1 查詢 + D2 頁圖像），sidecar-mlx 的輸出必須與
dsh-wemm-poc 已驗證腳本 scripts/run_eval_mlx.py 的前向輸出 cos ≥ 0.999。

必須用 venv-mlx 的 python 運行（導入 mlx/mlx_vlm）：
  venv-mlx/bin/python parity_mlx.py --handshake RUNTIME/mlx.json [--poc-dir PATH]

注意：導入 run_eval_mlx 會在模組級加載 WeMM MLX-4bit 模型（~8s、1.4GB），
屬預期行為（該腳本即如此設計）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import SidecarClient  # noqa: E402

THRESHOLD = 0.999
N_MEM = 10
N_QUERY = 3


def load_poc_module(poc_dir: Path):
    spec = importlib.util.spec_from_file_location(
        'run_eval_mlx', poc_dir / 'scripts' / 'run_eval_mlx.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def poc_mrl(v: np.ndarray, dim: int) -> np.ndarray:
    """與 run_eval_mlx.py 完全一致的 MRL 截斷+重歸一。"""
    u = v[:dim]
    return u / np.linalg.norm(u)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--handshake', required=True, help='sidecar-mlx 握手文件路徑')
    ap.add_argument('--poc-dir', default=None, help='dsh-wemm-poc 目錄（默認同倉庫兄弟目錄）')
    args = ap.parse_args()

    here = Path(__file__).resolve()
    poc_dir = Path(args.poc_dir) if args.poc_dir else here.parents[3] / 'dsh-wemm-poc'
    if not (poc_dir / 'scripts' / 'run_eval_mlx.py').exists():
        print(f'ERROR: dsh-wemm-poc not found at {poc_dir}', file=sys.stderr)
        return 2
    data = poc_dir / 'data'

    print(f'== 載入 PoC 已驗證腳本（{poc_dir.name}）==', flush=True)
    poc = load_poc_module(poc_dir)
    print('PoC model loaded.', flush=True)

    c = SidecarClient.from_handshake(args.handshake, timeout=600)

    # 固定輸入：與 run_eval_mlx.py 同樣的文本格式化
    mems = [json.loads(l) for l in open(data / 'd1_memories.jsonl')]
    mem_txt = [f"{m['content']} tags: {', '.join(m['tags'])}" if m['tags'] else m['content']
               for m in mems[:N_MEM]]
    queries = [it['q'] for it in json.load(open(data / 'd1_queries.json'))[:N_QUERY]]
    texts = mem_txt + queries
    pages = [json.loads(l) for l in open(data / 'd2_pages.jsonl')]
    img_rel = pages[0]['image']
    img_abs = str(data / img_rel)

    failures = 0

    for dim in (2048, 512):
        refs = [poc.embed_text(t) for t in texts]
        st, body = c.embed_texts(texts, dim=dim)
        if st != 200 or len(body.get('vectors', [])) != len(texts):
            print(f'FAIL dim={dim}: sidecar /embed/texts status={st} {str(body)[:200]}')
            failures += 1
            continue
        sims = []
        for ref, got in zip(refs, body['vectors']):
            r = poc_mrl(np.asarray(ref, dtype=np.float32), dim)
            sims.append(cos(r, np.asarray(got, dtype=np.float32)))
        worst = min(sims)
        status = 'PASS' if worst >= THRESHOLD else 'FAIL'
        print(f'{status} text parity dim={dim}: n={len(sims)} min_cos={worst:.6f} '
              f'mean={float(np.mean(sims)):.6f}')
        if worst < THRESHOLD:
            failures += 1
        if body.get('fingerprint') != f'wemm2b-mlx4b@{dim}' or body.get('dim') != dim:
            print(f'FAIL fingerprint/dim mismatch: {body.get("fingerprint")} {body.get("dim")}')
            failures += 1

    for dim in (2048, 512):
        ref = poc.embed_image(img_abs)
        st, body = c.embed_image(img_abs, dim=dim)
        if st != 200:
            print(f'FAIL image dim={dim}: status={st} {str(body)[:200]}')
            failures += 1
            continue
        r = poc_mrl(np.asarray(ref, dtype=np.float32), dim)
        sim = cos(r, np.asarray(body['vector'], dtype=np.float32))
        status = 'PASS' if sim >= THRESHOLD else 'FAIL'
        print(f'{status} image parity dim={dim}: cos={sim:.6f} ({img_rel})')
        if sim < THRESHOLD:
            failures += 1

    print(f'PARITY_{"OK" if failures == 0 else "FAIL"} threshold={THRESHOLD} '
          f'texts={len(texts)} dims=2048,512 image=1')
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
