#!/bin/bash
# Engine: afm (MLX, Swift) on Qwen3.6-27B-4bit. Probe + llama-benchy (prefix caching).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT="${PORT:-9999}"; E=afm
LOG="$OUTDIR/$E-server.log"; BENCH_OUT="$OUTDIR/$E.json"; BENCH_LOG="$OUTDIR/$E-bench.log"; PROBE_OUT="$OUTDIR/$E-probe.json"

echo "### kill other engines + free port $PORT"; kill_mlx_engines; free_port "$PORT"; echo "  freeRAM=$(free_gb)GB"
echo "### start afm"
MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" "$AFM_BIN" mlx -m "$MODEL_ID" --port "$PORT" --enable-prefix-caching </dev/null >"$LOG" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; pkill -f "afm.* mlx " 2>/dev/null' EXIT INT TERM
for i in $(seq 1 200); do
  kill -0 $SRV 2>/dev/null || { echo "  DIED"; tail -25 "$LOG"; exit 1; }
  low_ram && { echo "  !! LOW RAM"; exit 2; }
  curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "  ready t=${i}s gpu=$(gpu)% freeRAM=$(free_gb)GB"; break; }
  sleep 1
done
echo "### warmup"; curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":30,\"stream\":false}" \
  >"$OUTDIR/$E-warmup.json" 2>&1
python3 -c "import json;d=json.load(open('$OUTDIR/$E-warmup.json'));print('OK usage:',d.get('usage'))" 2>&1 | head -2
echo "### probe"; python3 "$PROBE" http://127.0.0.1:$PORT/v1 "$MODEL_ID" "afm-mlx-4bit" | tee "$PROBE_OUT"
echo "### llama-benchy"; uvx llama-benchy --base-url http://127.0.0.1:$PORT/v1 --model "$MODEL_ID" \
  --enable-prefix-caching --pp $PP --tg $TG --depth $DEPTHS --runs $RUNS --skip-coherence \
  --save-result "${BENCH_OUT%.json}${BENCH_SUFFIX}.json" --format "$BENCHY_FORMAT" $BENCHY_EXTRA >"${BENCH_LOG%.log}${BENCH_SUFFIX}.log" 2>&1
echo "### benchmark exit=$?"; cat "$BENCH_OUT" 2>/dev/null
echo "### afm prefix-cache [STATS] (cold vs HIT):"; grep -oE "\[STATS\] pp:.*stream=true" "$LOG" | tail -6
echo "### DONE"
