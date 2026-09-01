#!/usr/bin/env bash
# dsh-embed 環境裝配（SPEC §8）：雙 venv 隔離，版本鎖定為 Phase 0 已驗證組合。
#
#   venv-mlx: mlx==0.32.2 mlx-vlm==0.6.17                 （WeMM-2B MLX-4bit）
#   venv-tf:  torch==2.13.0 torchvision==0.28.0
#             transformers==5.2.0 qwen-vl-utils==0.0.14
#             numpy==2.5.2          （Qwen3-4B fp16 + WeMM fp16）
#
# 注：torchvision 是 qwen_vl_utils 的模組級硬依賴（SPEC §8 命令遺漏，實測補上）。
#
# 雙 venv 原因：transformers 必須鎖 5.2.0（WeMM 預處理一致性），與 mlx-vlm
# 依賴樹衝突 → 進程級隔離（dsh-wemm-poc/.venv 與 .venv-mlx 並存實證，SPEC §2）。
#
# 用法：
#   bash dsh-embed/scripts/bootstrap.sh [--force]
# 環境變量覆蓋：DSH_EMBED_HOME（默認 ~/.dsh/dsh-embed）、DSH_EMBED_RUNTIME_DIR
# （默認 ~/.dsh/run/dsh-embed）。模型權重不在此步驟範圍（已預下載至 HF 緩存）。

set -euo pipefail

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

DSH_EMBED_HOME="${DSH_EMBED_HOME:-$HOME/.dsh/dsh-embed}"
RUNTIME_DIR="${DSH_EMBED_RUNTIME_DIR:-$HOME/.dsh/run/dsh-embed}"
MLX_VENV="$DSH_EMBED_HOME/venv-mlx"
TF_VENV="$DSH_EMBED_HOME/venv-tf"
PYTHON_PIN="3.12"   # 與 dsh-wemm-poc 雙 venv 一致（3.12.13 實測）

MLX_PKGS=(mlx==0.32.2 mlx-vlm==0.6.17)
TF_PKGS=(torch==2.13.0 torchvision==0.28.0 transformers==5.2.0 qwen-vl-utils==0.0.14 numpy==2.5.2)

command -v uv >/dev/null || { echo "ERROR: uv not found (brew install uv)"; exit 1; }

if [ "$FORCE" = "1" ]; then
  echo "==> --force: removing $MLX_VENV $TF_VENV"
  rm -rf "$MLX_VENV" "$TF_VENV"
fi

echo "==> creating venv-mlx ($PYTHON_PIN) ..."
[ -x "$MLX_VENV/bin/python" ] || uv venv "$MLX_VENV" --python "$PYTHON_PIN"
echo "==> installing ${MLX_PKGS[*]}"
uv pip install --python "$MLX_VENV/bin/python" "${MLX_PKGS[@]}"

echo "==> creating venv-tf ($PYTHON_PIN) ..."
[ -x "$TF_VENV/bin/python" ] || uv venv "$TF_VENV" --python "$PYTHON_PIN"
echo "==> installing ${TF_PKGS[*]}"
uv pip install --python "$TF_VENV/bin/python" "${TF_PKGS[@]}"

mkdir -p "$RUNTIME_DIR"
echo "==> runtime dir ready: $RUNTIME_DIR"

# ── 導入自檢（乾淨環境重建的可運行性證據）─────────────────────
"$MLX_VENV/bin/python" - <<'PY'
import mlx.core as mx, mlx_vlm, numpy
print(f'venv-mlx OK: mlx={getattr(mx, "__version__", "?")} mlx-vlm={mlx_vlm.__version__}'
      f' numpy={numpy.__version__}')
PY
"$TF_VENV/bin/python" - <<'PY'
import torch, torchvision, transformers, qwen_vl_utils, numpy
print(f'venv-tf OK: torch={torch.__version__} torchvision={torchvision.__version__}'
      f' transformers={transformers.__version__}'
      f' qwen-vl-utils={getattr(qwen_vl_utils, "__version__", "0.0.14")}'
      f' numpy={numpy.__version__}')
PY

echo "BOOTSTRAP_OK mlx=$MLX_VENV tf=$TF_VENV"
