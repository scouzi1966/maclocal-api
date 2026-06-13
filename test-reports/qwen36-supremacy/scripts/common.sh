#!/bin/bash
# Shared config + helpers for the Qwen3.6-27B "supremacy" benchmark harness.
#
# Results default INTO THE REPO: test-reports/qwen36-supremacy/results/
# Override the output dir with:   BENCH_OUT_DIR=/some/path ./run-<engine>.sh
#
# All machine-specific paths below are env-overridable so the harness is portable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${BENCH_OUT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/results}"
mkdir -p "$OUTDIR"

# --- model + cache (defaults match this machine; override via env) ---
MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/Crucial4TB/models/vesta-test-cache}"
MODEL_ID="${BENCH_MODEL_ID:-mlx-community/Qwen3.6-27B-4bit}"
MODEL_PATH="${BENCH_MODEL_PATH:-$MODEL_CACHE/mlx-community/Qwen3.6-27B-4bit}"
GGUF_BLOB="${GGUF_BLOB:-$HOME/.ollama/models/blobs/sha256-5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0}"

# --- engine binaries (override via env) ---
AFM_BIN="${AFM_BIN:-/Users/syl/afm-main-swift6/afm}"
VLM_PYTHON="${VLM_PYTHON:-/tmp/bench/vlmenv/bin/python}"   # create with setup-vlmenv.sh
LMS_BIN="${LMS_BIN:-$HOME/.lmstudio/bin/lms}"
OMLX_BIN="${OMLX_BIN:-/Applications/oMLX.app/Contents/MacOS/omlx-cli}"
RAPID_BIN="${RAPID_BIN:-/tmp/bench/rmlxenv/bin/rapid-mlx}"   # pip rapid-mlx in isolated env
PROBE="$SCRIPT_DIR/probe.py"

# --- llama-benchy sweep params (override via env) ---
PP="${BENCH_PP:-512}"
TG="${BENCH_TG:-128}"
DEPTHS="${BENCH_DEPTHS:-0 4096}"
RUNS="${BENCH_RUNS:-2}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"
# output format + extra flags (e.g. timeseries). BENCH_SUFFIX appends to the result filename
# so a json/timeseries run does not clobber the default markdown table.
BENCHY_FORMAT="${BENCHY_FORMAT:-md}"
BENCHY_EXTRA="${BENCHY_EXTRA:-}"
BENCH_SUFFIX="${BENCH_SUFFIX:-}"

# --- helpers ---
free_gb(){ vm_stat | awk '/Pages free/{f=$3}/Pages inactive/{i=$3}END{gsub(/\./,"",f);gsub(/\./,"",i);printf "%.1f",(f+i)*16384/1073741824}'; }
gpu(){ ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$'; }
# returns 0 (true) when free RAM is below MIN_FREE_GB — avoids bc dependency
low_ram(){ awk -v f="$(free_gb)" -v m="$MIN_FREE_GB" 'BEGIN{exit !(f<m)}'; }
free_port(){ local p="$1"; for _ in $(seq 1 12); do lsof -ti tcp:"$p" >/dev/null 2>&1 || return 0; lsof -ti tcp:"$p" | xargs kill 2>/dev/null; sleep 1; done; }
kill_mlx_engines(){ pkill -f "afm.* mlx " 2>/dev/null; pkill -f "mlx_vlm.server" 2>/dev/null; pkill -f "llama-server" 2>/dev/null; pkill -f "omlx-cli" 2>/dev/null; pkill -f "rapid-mlx serve" 2>/dev/null; "$LMS_BIN" unload --all >/dev/null 2>&1; }
