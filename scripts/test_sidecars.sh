#!/usr/bin/env bash
# dsh-embed 雙 sidecar 測試總編排（t2 verify 命令）。
#
#   bash dsh-embed/scripts/test_sidecars.sh [OPTIONS]
#
# 流程：
#   1. 解析雙 venv python（env 覆蓋 > ~/.dsh/dsh-embed/venv-* > dsh-wemm-poc/.venv*）
#   2. 隔離 runtime 目錄起 mlx_serve.py / tf_serve.py（port=0 隨機 + 握手文件）
#   3. 兩 sidecar 跑同一組契約測試（contract_tests.py）
#   4. 真實模式：mlx 移植正確性（parity_mlx.py，cos≥0.999 vs dsh-wemm-poc 已驗證腳本）
#               + tf 語義健全性（tf_sanity.py，含 wemm2b-fp16 惰性加載）
#   5. 生命週期：SIGTERM 清理握手文件；空閒超時自退出
#
# OPTIONS：
#   --fake          假後端快速模式（跳過 parity/sanity；無需權重）
#   --skip-parity   真實模式但跳過 mlx parity（迭代用）
#   --skip-tf       真實模式但跳過 tf sanity
#   --keep          不殘留進程模式反轉：保留 runtime 目錄與日誌便於排查
#   --runtime-dir D 覆蓋測試 runtime 目錄（默認 mktemp）

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
POC="$WS/dsh-wemm-poc"
CONTRACT_DIR="$REPO/test/contract"

MODE=real; SKIP_PARITY=0; SKIP_TF=0; KEEP=0
RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dsh-embed-sidecar-test.XXXXXX")"
while [ $# -gt 0 ]; do
  case "$1" in
    --fake) MODE=fake ;;
    --skip-parity) SKIP_PARITY=1 ;;
    --skip-tf) SKIP_TF=1 ;;
    --keep) KEEP=1 ;;
    --runtime-dir) RUNTIME_DIR="$2"; shift ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
  shift
done

# ── python 解析 ─────────────────────────────────────────────
PY3="${PYTHON3:-python3}"   # 契約測試/輔助腳本（stdlib only）
resolve_py() {  # $1=env var名 $2=dsh-embed venv $3=poc venv
  if [ -n "${!1:-}" ] && [ -x "${!1}" ]; then echo "${!1}"; return; fi
  if [ -x "$2" ]; then echo "$2"; return; fi
  if [ -x "$3" ]; then echo "$3"; return; fi
  echo ""
}
MLX_PY="$(resolve_py DSH_EMBED_MLX_PY "$HOME/.dsh/dsh-embed/venv-mlx/bin/python" "$POC/.venv-mlx/bin/python")"
TF_PY="$(resolve_py DSH_EMBED_TF_PY "$HOME/.dsh/dsh-embed/venv-tf/bin/python" "$POC/.venv/bin/python")"
[ -n "$MLX_PY" ] || { echo "ERROR: no mlx venv python (bootstrap.sh or DSH_EMBED_MLX_PY)"; exit 2; }
[ -n "$TF_PY" ]  || { echo "ERROR: no tf venv python (bootstrap.sh or DSH_EMBED_TF_PY)"; exit 2; }
echo "mlx python: $MLX_PY"
echo "tf  python: $TF_PY"
echo "runtime dir: $RUNTIME_DIR (mode=$MODE)"

mkdir -p "$RUNTIME_DIR"
MLX_LOG="$RUNTIME_DIR/mlx.log"; TF_LOG="$RUNTIME_DIR/tf.log"
MLX_PID=""; TF_PID=""
FAIL=0

cleanup() {
  for pid in "$MLX_PID" "$TF_PID"; do
    [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
  done
  sleep 1
  for pid in "$MLX_PID" "$TF_PID"; do
    kill -9 "$pid" 2>/dev/null
  done
  if [ "$KEEP" = "0" ]; then rm -rf "$RUNTIME_DIR"; fi
}
trap cleanup EXIT

wait_ready() {  # $1=handshake file $2=timeout_s $3=label
  local deadline=$(( $(date +%s) + $2 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ -s "$1" ] && "$PY3" -c '
import json, sys, os
d = json.load(open(sys.argv[1]))
assert isinstance(d.get("port"), int) and d.get("token") and d.get("pid")
os.kill(d["pid"], 0)
' "$1" 2>/dev/null; then return 0; fi
    sleep 1
  done
  echo "ERROR: $3 not ready in ${2}s (see log)"; return 1
}

hs_field() { "$PY3" -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

start_sidecar() {  # $1=py $2=script $3=name $4=timeout $5=log $6=extra_args...
  local py="$1" script="$2" nm="$3" timeout_s="$4" lg="$5"; shift 5
  "$py" "$script" --runtime-dir "$RUNTIME_DIR" --name "$nm" "$@" >"$lg" 2>&1 &
  local pid=$!
  if [ "$nm" = "mlx" ]; then MLX_PID=$pid; else TF_PID=$pid; fi
  wait_ready "$RUNTIME_DIR/$nm.json" "$timeout_s" "sidecar-$nm" || { tail -5 "$lg"; FAIL=1; return 1; }
  echo "sidecar-$nm ready (pid=$pid port=$(hs_field "$RUNTIME_DIR/$nm.json" port))"
}

# ── 1. 起雙 sidecar ──────────────────────────────────────────
FAKE_FLAG=""
[ "$MODE" = "fake" ] && FAKE_FLAG="--fake"
START_TIMEOUT_MLX=240; START_TIMEOUT_TF=240
[ "$MODE" = "fake" ] && START_TIMEOUT_MLX=20 && START_TIMEOUT_TF=20

start_sidecar "$MLX_PY" "$REPO/src/sidecar/mlx_serve.py" mlx "$START_TIMEOUT_MLX" "$MLX_LOG" $FAKE_FLAG || exit 1
start_sidecar "$TF_PY" "$REPO/src/sidecar/tf_serve.py" tf "$START_TIMEOUT_TF" "$TF_LOG" $FAKE_FLAG || exit 1

# ── 2. 契約測試（兩 sidecar 同一組用例）───────────────────────
WEMM_FLAG=""
[ "$MODE" = "real" ] && [ "$SKIP_TF" = "0" ] && WEMM_FLAG="--test-wemm"

echo; echo "===== 契約測試（mlx）====="
"$PY3" "$CONTRACT_DIR/contract_tests.py" \
  --url "http://127.0.0.1:$(hs_field "$RUNTIME_DIR/mlx.json" port)" \
  --token "$(hs_field "$RUNTIME_DIR/mlx.json" token)" --name mlx $FAKE_FLAG \
  || FAIL=1

echo; echo "===== 契約測試（tf）====="
"$PY3" "$CONTRACT_DIR/contract_tests.py" \
  --url "http://127.0.0.1:$(hs_field "$RUNTIME_DIR/tf.json" port)" \
  --token "$(hs_field "$RUNTIME_DIR/tf.json" token)" --name tf $FAKE_FLAG $WEMM_FLAG \
  || FAIL=1

# ── 3. 真實模式：移植正確性 + 語義健全性 ─────────────────────
if [ "$MODE" = "real" ]; then
  if [ "$SKIP_PARITY" = "0" ]; then
    echo; echo "===== mlx 移植正確性（cos≥0.999 vs dsh-wemm-poc）====="
    "$MLX_PY" "$CONTRACT_DIR/parity_mlx.py" --handshake "$RUNTIME_DIR/mlx.json" --poc-dir "$POC" || FAIL=1
  fi
  if [ "$SKIP_TF" = "0" ]; then
    echo; echo "===== tf 語義健全性 ====="
    "$TF_PY" "$CONTRACT_DIR/tf_sanity.py" --handshake "$RUNTIME_DIR/tf.json" --test-wemm || FAIL=1
  fi
fi

# ── 4. 生命週期：SIGTERM 清握手 + 空閒自退出（fake 快速驗證）──
echo; echo "===== 生命週期（SIGTERM / 空閒退出）====="
LIFE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dsh-embed-life-test.XXXXXX")"
"$MLX_PY" "$REPO/src/sidecar/mlx_serve.py" --runtime-dir "$LIFE_DIR" --name life \
  --fake --idle-timeout-sec 5 >"$LIFE_DIR/life.log" 2>&1 &
LIFE_PID=$!
if wait_ready "$LIFE_DIR/life.json" 20 "sidecar-life"; then
  TERM_TEST_PID="$(hs_field "$LIFE_DIR/life.json" pid)"
  kill -TERM "$TERM_TEST_PID" 2>/dev/null; sleep 2
  if kill -0 "$TERM_TEST_PID" 2>/dev/null; then
    echo "  FAIL sigterm_process_exit"; FAIL=1
  else
    echo "  PASS sigterm_process_exit"
  fi
  [ -e "$LIFE_DIR/life.json" ] && { echo "  FAIL sigterm_handshake_removed"; FAIL=1; } \
    || echo "  PASS sigterm_handshake_removed"
else
  echo "  FAIL lifecycle sidecar did not start"; FAIL=1; kill -9 "$LIFE_PID" 2>/dev/null
fi

"$MLX_PY" "$REPO/src/sidecar/mlx_serve.py" --runtime-dir "$LIFE_DIR" --name idle \
  --fake --idle-timeout-sec 5 >"$LIFE_DIR/idle.log" 2>&1 &
IDLE_BG=$!
if wait_ready "$LIFE_DIR/idle.json" 20 "sidecar-idle"; then
  IDLE_PID="$(hs_field "$LIFE_DIR/idle.json" pid)"
  # 不發任何 embed 調用 → 5s 空閒 + 看門狗週期後應自退出
  sleep 14
  if kill -0 "$IDLE_PID" 2>/dev/null; then
    echo "  FAIL idle_timeout_exit"; FAIL=1; kill -9 "$IDLE_PID" 2>/dev/null
  else
    echo "  PASS idle_timeout_exit"
  fi
  [ -e "$LIFE_DIR/idle.json" ] && { echo "  FAIL idle_handshake_removed"; FAIL=1; } \
    || echo "  PASS idle_handshake_removed"
else
  echo "  FAIL idle sidecar did not start"; FAIL=1; kill -9 "$IDLE_BG" 2>/dev/null
fi
rm -rf "$LIFE_DIR"

# ── 匯總 ────────────────────────────────────────────────────
echo
if [ "$FAIL" = "0" ]; then
  echo "TEST_SIDECARS_OK (mode=$MODE runtime=$RUNTIME_DIR)"
else
  echo "TEST_SIDECARS_FAIL (mode=$MODE runtime=$RUNTIME_DIR；加 --keep 保留日誌)"
  KEEP=1
fi
exit $FAIL
