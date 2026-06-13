#!/bin/bash
# Engine: oMLX (omlx-cli serve) on Qwen3.6-27B-4bit. Discovers models from --model-dir subdirs.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT="${PORT:-8000}"; E=omlx
MODEL_DIR="$(dirname "$MODEL_PATH")"; ID="$(basename "$MODEL_PATH")"
LOG="$OUTDIR/$E-server.log"; BENCH_OUT="$OUTDIR/$E.json"; BENCH_LOG="$OUTDIR/$E-bench.log"; PROBE_OUT="$OUTDIR/$E-probe.json"

echo "### kill other engines + free port"; kill_mlx_engines; free_port "$PORT"; echo "  freeRAM=$(free_gb)GB"
echo "### omlx serve --model-dir $MODEL_DIR (model_id=$ID)"
"$OMLX_BIN" serve --model-dir "$MODEL_DIR" --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; pkill -f "omlx-cli" 2>/dev/null' EXIT INT TERM
ready=0
for i in $(seq 1 200); do
  kill -0 $SRV 2>/dev/null || { echo "  DIED"; tail -30 "$LOG"; exit 1; }
  low_ram && { echo "  !! LOW RAM"; exit 2; }
  curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "  ready t=${i}s gpu=$(gpu)% freeRAM=$(free_gb)GB"; ready=1; break; }
  [ $((i%10)) -eq 0 ] && echo "  loading t=${i}s freeRAM=$(free_gb)GB"; sleep 1
done
[ "$ready" = 1 ] || { echo "  never ready"; tail -30 "$LOG"; exit 3; }
echo "### warmup (lazy-loads model on first request)"
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":30,\"stream\":false}" \
  >"$OUTDIR/$E-warmup.json" 2>&1
python3 -c "import json;d=json.load(open('$OUTDIR/$E-warmup.json'));assert d.get('choices');print('OK usage:',d.get('usage'))" 2>&1 | head -2 \
  || { echo '### oMLX FAILED to run qwen3_5'; head -c 600 "$OUTDIR/$E-warmup.json"; exit 5; }
echo "### probe"; python3 "$PROBE" http://127.0.0.1:$PORT/v1 "$ID" "omlx-mlx-4bit" | tee "$PROBE_OUT"
echo "### llama-benchy (note: oMLX tg t/s is inflated in llama-benchy — use probe)"
uvx llama-benchy --base-url http://127.0.0.1:$PORT/v1 --model "$ID" \
  --enable-prefix-caching --pp $PP --tg $TG --depth $DEPTHS --runs $RUNS --skip-coherence \
  --save-result "${BENCH_OUT%.json}${BENCH_SUFFIX}.json" --format "$BENCHY_FORMAT" $BENCHY_EXTRA >"${BENCH_LOG%.log}${BENCH_SUFFIX}.log" 2>&1
echo "### benchmark exit=$?"; cat "$BENCH_OUT" 2>/dev/null
echo "### DONE"
