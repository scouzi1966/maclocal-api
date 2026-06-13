#!/bin/bash
# Engine: LM Studio (bundled MLX engine) on Qwen3.6-27B-4bit.
# Requires the model indexed in LM Studio. To use the shared cache without a re-download:
#   ln -s "$MODEL_PATH" ~/.lmstudio/models/mlx-community/Qwen3.6-27B-4bit
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT="${PORT:-1234}"; E=lmstudio; KEY="${LMS_MODEL_KEY:-qwen3.6-27b}"
LOG="$OUTDIR/$E-server.log"; BENCH_OUT="$OUTDIR/$E.json"; BENCH_LOG="$OUTDIR/$E-bench.log"; PROBE_OUT="$OUTDIR/$E-probe.json"

echo "### kill other engines"; kill_mlx_engines; sleep 2; echo "  freeRAM=$(free_gb)GB"
echo "### LM Studio server + load $KEY"
"$LMS_BIN" server start --port "$PORT" >"$LOG" 2>&1
"$LMS_BIN" unload --all >>"$LOG" 2>&1 || true
"$LMS_BIN" load "$KEY" --gpu max -y >>"$LOG" 2>&1 &
LP=$!; trap '"$LMS_BIN" unload --all >/dev/null 2>&1' EXIT INT TERM
for i in $(seq 1 220); do low_ram && { echo "  !! LOW RAM"; exit 2; }; kill -0 $LP 2>/dev/null || { echo "  load returned t=${i}s freeRAM=$(free_gb)GB gpu=$(gpu)%"; break; }; [ $((i%10)) -eq 0 ] && echo "  loading t=${i}s freeRAM=$(free_gb)GB"; sleep 1; done
"$LMS_BIN" ps 2>/dev/null | head
echo "### warmup (verify qwen3_5 actually runs)"
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":30,\"stream\":false}" \
  >"$OUTDIR/$E-warmup.json" 2>&1
python3 -c "import json;d=json.load(open('$OUTDIR/$E-warmup.json'));assert d.get('choices');print('OK usage:',d.get('usage'))" 2>&1 | head -2 \
  || { echo '### LM Studio FAILED to run qwen3_5'; head -c 600 "$OUTDIR/$E-warmup.json"; exit 5; }
echo "### probe"; python3 "$PROBE" http://127.0.0.1:$PORT/v1 "$MODEL_ID" "lmstudio-mlx-4bit" | tee "$PROBE_OUT"
echo "### llama-benchy (note: LM Studio default ctx may abort depth-4096)"
uvx llama-benchy --base-url http://127.0.0.1:$PORT/v1 --model "$MODEL_ID" \
  --enable-prefix-caching --pp $PP --tg $TG --depth $DEPTHS --runs $RUNS --skip-coherence \
  --save-result "${BENCH_OUT%.json}${BENCH_SUFFIX}.json" --format "$BENCHY_FORMAT" $BENCHY_EXTRA >"${BENCH_LOG%.log}${BENCH_SUFFIX}.log" 2>&1
echo "### benchmark exit=$?"; cat "$BENCH_OUT" 2>/dev/null
echo "### DONE"
