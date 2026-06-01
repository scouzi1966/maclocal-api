#!/bin/bash
# Fair, thermally-controlled re-check of the two contested metrics.
# Cool down, then measure afm -> lmstudio -> afm -> omlx (afm re-measured to track thermal drift),
# each via probe_contested. afm uses the OPTIMIZED build (.build/release/afm).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PC="$SCRIPT_DIR/probe_contested.py"
AFM_OPT="$(cd "$SCRIPT_DIR/../../.." && pwd)/.build/release/afm"
COOL="${COOL:-180}"
echo "### cooldown ${COOL}s (let M4 Pro thermals settle)"; kill_mlx_engines 2>/dev/null; sleep "$COOL"

afm_run(){ # tag
  kill_mlx_engines; free_port 9999; sleep 1
  MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" "$AFM_OPT" mlx -m "$MODEL_ID" --port 9999 --enable-prefix-caching </dev/null >"$OUTDIR/recheck-$1-srv.log" 2>&1 &
  S=$!; for i in $(seq 1 200); do curl -s -m2 http://127.0.0.1:9999/v1/models >/dev/null 2>&1 && break; kill -0 $S 2>/dev/null||{ echo "$1 DIED";return;}; sleep 1; done
  echo ">>> $1"; python3 "$PC" http://127.0.0.1:9999/v1 "$MODEL_ID" "$1" "$OUTDIR/recheck-$1.json"
  kill $S 2>/dev/null; pkill -f "release/afm" 2>/dev/null; sleep 2
}

afm_run afm-A
# LM Studio (depth competitor)
kill_mlx_engines; "$LMS_BIN" server start --port 1234 >/dev/null 2>&1; "$LMS_BIN" unload --all >/dev/null 2>&1
"$LMS_BIN" load qwen3.6-27b --gpu max --context-length 20480 -y >/dev/null 2>&1
for i in $(seq 1 200); do curl -s -m2 http://127.0.0.1:1234/v1/models >/dev/null 2>&1 && break; sleep 1; done
echo ">>> lmstudio"; python3 "$PC" http://127.0.0.1:1234/v1 "$MODEL_ID" lmstudio "$OUTDIR/recheck-lmstudio.json"; "$LMS_BIN" unload --all >/dev/null 2>&1; sleep 2

afm_run afm-B
# oMLX (prefill competitor)
kill_mlx_engines; free_port 8000
"$OMLX_BIN" serve --model-dir "$(dirname "$MODEL_PATH")" --host 127.0.0.1 --port 8000 >/dev/null 2>&1 &
for i in $(seq 1 200); do curl -s -m2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 1; done
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3.6-27B-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' >/dev/null 2>&1
echo ">>> omlx"; python3 "$PC" http://127.0.0.1:8000/v1 Qwen3.6-27B-4bit omlx "$OUTDIR/recheck-omlx.json"
kill_mlx_engines; free_port 8000
echo "### FAIR RECHECK DONE"