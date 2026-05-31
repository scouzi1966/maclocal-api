#!/bin/bash
# Engine: mlx-vlm (official MLX Python VLM server) on Qwen3.6-27B-4bit.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT="${PORT:-9999}"; E=mlx_vlm
LOG="$OUTDIR/$E-server.log"; BENCH_OUT="$OUTDIR/$E.json"; BENCH_LOG="$OUTDIR/$E-bench.log"; PROBE_OUT="$OUTDIR/$E-probe.json"
[ -x "$VLM_PYTHON" ] || { echo "VLM_PYTHON not found: $VLM_PYTHON — run setup-vlmenv.sh first"; exit 4; }

echo "### kill other engines + free port"; kill_mlx_engines; free_port "$PORT"; echo "  freeRAM=$(free_gb)GB"
echo "### start mlx_vlm.server (preload)"
"$VLM_PYTHON" -m mlx_vlm.server --model "$MODEL_PATH" --port "$PORT" --host 127.0.0.1 --log-level INFO >"$LOG" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; pkill -f "mlx_vlm.server" 2>/dev/null' EXIT INT TERM
ready=0
for i in $(seq 1 220); do
  kill -0 $SRV 2>/dev/null || { echo "  DIED"; tail -30 "$LOG"; exit 1; }
  low_ram && { echo "  !! LOW RAM"; exit 2; }
  curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "  ready t=${i}s gpu=$(gpu)% freeRAM=$(free_gb)GB"; ready=1; break; }
  [ $((i%10)) -eq 0 ] && echo "  loading t=${i}s freeRAM=$(free_gb)GB"; sleep 1
done
[ "$ready" = 1 ] || { echo "  never ready"; tail -30 "$LOG"; exit 3; }
echo "### warmup"; curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_PATH\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":30,\"stream\":false}" \
  >"$OUTDIR/$E-warmup.json" 2>&1
python3 -c "import json;d=json.load(open('$OUTDIR/$E-warmup.json'));print('OK usage:',d.get('usage'))" 2>&1 | head -2
echo "### probe"; python3 "$PROBE" http://127.0.0.1:$PORT/v1 "$MODEL_PATH" "mlx_vlm-4bit" | tee "$PROBE_OUT"
echo "### llama-benchy"; uvx llama-benchy --base-url http://127.0.0.1:$PORT/v1 --model "$MODEL_PATH" \
  --enable-prefix-caching --pp $PP --tg $TG --depth $DEPTHS --runs $RUNS --skip-coherence \
  --save-result "${BENCH_OUT%.json}${BENCH_SUFFIX}.json" --format "$BENCHY_FORMAT" $BENCHY_EXTRA >"${BENCH_LOG%.log}${BENCH_SUFFIX}.log" 2>&1
echo "### benchmark exit=$?"; cat "$BENCH_OUT" 2>/dev/null
echo "### DONE"
