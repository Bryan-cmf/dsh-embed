#!/usr/bin/env python3
"""sidecar-tf 進程入口：Qwen3-Embedding-4B fp16（eager）+ WeMM-2B fp16（惰性 fallback）。

用法（由 dsh-embed host 插件 spawn；見 SPEC §2/§7）：
  venv-tf/bin/python tf_serve.py [--runtime-dir DIR] [--name tf]
      [--port 0] [--idle-timeout-sec 900] [--eager qwen3-4b-fp16] [--fake]

默認 eager=['qwen3-4b-fp16']（冷啟動實測 ~30s），wemm2b-fp16 惰性加載
（SPEC §7 tfSidecar.eagerBackends）；就緒後把 {port, token, pid} 寫入
{runtime_dir}/tf.json。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SidecarApp, base_arg_parser, offline_env, serve   # noqa: E402

DEFAULT_EAGER = ['qwen3-4b-fp16']


def main() -> None:
    parser = base_arg_parser('dsh-embed sidecar-tf：Qwen3-4B fp16 + WeMM-2B fp16 嵌入服務', 'tf')
    args = parser.parse_args()

    if args.online:
        import os
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
    else:
        offline_env()   # 默認離線：權重已預下載（47G）

    from tf_embed import Qwen3Fp16Backend, Wemm2bFp16Backend   # offline_env 之後再導入
    from common import FakeBackend

    if args.fake:
        qwen = FakeBackend(Qwen3Fp16Backend())
        wemm = FakeBackend(Wemm2bFp16Backend())
    else:
        qwen = Qwen3Fp16Backend()
        wemm = Wemm2bFp16Backend()

    eager = list(DEFAULT_EAGER)
    if args.lazy:
        eager = []
    elif args.eager:
        eager = [n.strip() for n in args.eager.split(',') if n.strip()]

    app = SidecarApp(
        name=args.name,
        backends=[qwen, wemm],
        default_backend=qwen.name,
        runtime_dir=Path(args.runtime_dir),
        port=args.port,
        idle_timeout_sec=args.idle_timeout_sec,
    )
    serve(app, eager=eager)


if __name__ == '__main__':
    main()
