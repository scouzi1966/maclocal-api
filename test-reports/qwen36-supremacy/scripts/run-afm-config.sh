#!/bin/bash
# Profile afm under a given config with probe_full. Usage: run-afm-config.sh TAG [extra afm flags...]
# Always run as `bash run-afm-config.sh ...` (BASH_SOURCE works under bash, not zsh `source`).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TAG="$1"; shift; EXTRA=("$@")
PORT=9999; FP="$SCRIPT_DIR/probe_full.py"
echo "### afm config: TAG=$TAG flags=${EXTRA[*]:-(none)}  OUTDIR=$OUTDIR"
kill_mlx_engines 2>/dev/null; pkill -f "afm.* mlx " 2>/dev/null; free_port $PORT; sleep 2
# Expand EXTRA safely under `set -u` even when empty (bash treats ${arr[@]} as unbound if len 0).
MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" "$AFM_BIN" mlx -m "$MODEL_ID" --port $PORT --enable-prefix-caching ${EXTRA[@]+"${EXTRA[@]}"} </dev/null >"$OUTDIR/afm-$TAG-srv.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; pkill -f "afm.* mlx " 2>/dev/null' EXIT INT TERM
for i in $(seq 1 200); do curl -s -m2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "ready t=${i}s"; break; }; kill -0 $SRV 2>/dev/null || { echo DIED; tail -15 "$OUTDIR/afm-$TAG-srv.log"; exit 1; }; sleep 1; done
python3 "$FP" http://127.0.0.1:$PORT/v1 "$MODEL_ID" "afm-$TAG" "$OUTDIR/afm-$TAG-full.json"
echo "### wrote $OUTDIR/afm-$TAG-full.json"