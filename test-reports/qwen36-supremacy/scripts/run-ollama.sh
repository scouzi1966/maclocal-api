#!/bin/bash
# Engine: ollama (OpenAI-compat :11434) on Qwen3.6-27B.
# DEFAULT MODEL = qwen3.6:27b-mlx (the MLX tag). The GGUF tag
# (hf.co/unsloth/Qwen3.6-27B-GGUF:Q4_K_M) FAILS on this machine: ollama can't load the
# qwen35 architecture ("unable to load model" / "unknown model architecture: 'qwen35'").
# Always benchmark the MLX tag. Override with OLLAMA_MODEL_ID if needed.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
BASE="${OLLAMA_BASE:-http://127.0.0.1:11434}"; E=ollama
ID="${OLLAMA_MODEL_ID:-qwen3.6:27b-mlx}"
BENCH_OUT="$OUTDIR/$E.json"; BENCH_LOG="$OUTDIR/$E-bench.log"; PROBE_OUT="$OUTDIR/$E-probe.json"

echo "### kill MLX/llama engines (ollama serve stays)"; pkill -f "afm.* mlx " 2>/dev/null; pkill -f "mlx_vlm.server" 2>/dev/null; pkill -f "llama-server" 2>/dev/null; pkill -f "omlx-cli" 2>/dev/null; sleep 2
curl -s -m 5 "$BASE/v1/models" >/dev/null 2>&1 || { echo "  starting ollama serve"; (ollama serve >"$OUTDIR/$E-serve.log" 2>&1 &); sleep 4; }
echo "### warmup (loads model)"; curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":30,\"stream\":false}" \
  >"$OUTDIR/$E-warmup.json" 2>&1
if ! python3 -c "import json;d=json.load(open('$OUTDIR/$E-warmup.json'));assert d.get('choices');print('OK usage:',d.get('usage'))" 2>/dev/null; then
  echo "### ollama CANNOT serve $ID — response:"; head -c 600 "$OUTDIR/$E-warmup.json"; echo
  echo "### (expected for GGUF qwen35 on ollama<=0.24.0)"; exit 5
fi
curl -s "$BASE/api/generate" -d "{\"model\":\"$ID\",\"keep_alive\":\"30m\",\"prompt\":\"hi\",\"stream\":false}" >/dev/null 2>&1
echo "### probe"; python3 "$PROBE" "$BASE/v1" "$ID" "ollama" | tee "$PROBE_OUT"
echo "### llama-benchy"; uvx llama-benchy --base-url "$BASE/v1" --model "$ID" \
  --enable-prefix-caching --pp $PP --tg $TG --depth $DEPTHS --runs $RUNS --skip-coherence \
  --save-result "${BENCH_OUT%.json}${BENCH_SUFFIX}.json" --format "$BENCHY_FORMAT" $BENCHY_EXTRA >"${BENCH_LOG%.log}${BENCH_SUFFIX}.log" 2>&1
echo "### benchmark exit=$?"; cat "$BENCH_OUT" 2>/dev/null
curl -s "$BASE/api/generate" -d "{\"model\":\"$ID\",\"keep_alive\":0,\"prompt\":\"\",\"stream\":false}" >/dev/null 2>&1
echo "### DONE"
