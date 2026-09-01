#!/usr/bin/env python3
"""sidecar-mlx 進程入口：WeMM-2B MLX-4bit（文本+圖像，視覺/技能默認後端）。

用法（由 dsh-embed host 插件 spawn；見 SPEC §2 進程監管）：
  venv-mlx/bin/python mlx_serve.py [--runtime-dir DIR] [--name mlx]
      [--port 0] [--idle-timeout-sec 900] [--model PATH] [--fake]

啟動即 eager 加載（冷啟動實測 ~8s），就緒後把 {port, token, pid} 寫入
{runtime_dir}/mlx.json 供插件輪詢發現。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SidecarApp, base_arg_parser, offline_env, serve   # noqa: E402


def main() -> None:
    parser = base_arg_parser('dsh-embed sidecar-mlx：WeMM-2B MLX-4bit 嵌入服務', 'mlx')
    parser.add_argument('--model', default=None,
                        help='MLX 權重目錄覆蓋（默認 HF 緩存 hfadam/WeMM-Embedding-2B-MLX-4bit）')
    args = parser.parse_args()

    if args.online:
        import os
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
    else:
        offline_env()   # 默認離線：權重已預下載

    from mlx_embed import WemmMlx4bBackend   # offline_env 之後再導入
    from common import FakeBackend

    if args.fake:
        backend = FakeBackend(WemmMlx4bBackend())
        eager = [backend.name]
    else:
        backend = WemmMlx4bBackend(model_override=args.model)
        eager = [backend.name]

    if args.lazy:
        eager = []
    elif args.eager:
        eager = [n.strip() for n in args.eager.split(',') if n.strip()]

    app = SidecarApp(
        name=args.name,
        backends=[backend],
        default_backend=backend.name,
        runtime_dir=Path(args.runtime_dir),
        port=args.port,
        idle_timeout_sec=args.idle_timeout_sec,
    )
    serve(app, eager=eager)


if __name__ == '__main__':
    main()
