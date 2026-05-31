#!/bin/bash
# Engine: llama.cpp llama-server (Homebrew, Metal) on Qwen3.6-27B GGUF Q4_K_M.
# (ollama 0.24.0 can't load 'qwen35'; upstream llama.cpp can — this is the real GGUF engine.)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT="${PORT:-9999}"; E=llamacpp; ID="qwen3.6-gguf"
LOG="$OUTDIR/$E-server.log"; BENCH_OUT="$OUTDIR/$E.json"; BENCH_LOG="$OUTDIR/$E-bench.log"; PROBE_OUT="$OUTDIR/$E-probe.json"
[ -f "$GGUF_BLOB" ] || { echo "GGUF_BLOB not found: $GGUF_BLOB"; exit 4; }

echo "### kill other engines + free port"; kill_mlx_engines; free_port "$PORT"; echo "  freeRAM=$(free_gb)GB"
echo "### start llama-server (Metal, all layers, ctx 8192)"
llama-server -m "$GGUF_BLOB" --host 127.0.0.1 --port "$PORT" -c 8192 -ngl 999 --no-warmup >"$LOG" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; pkill -f "llama-server" 2>/dev/null' EXIT INT TERM
ready=0
for i in $(seq 1 220); do
  kill -0 $SRV 2>/dev/null || { echo "  DIED"; tail -25 "$LOG"; exit 1; }
  low_ram && { echo "  !! LOW RAM"; exit 2; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && { echo "  healthy t=${i}s gpu=$(gpu)% freeRAM=$(free_gb)GB"; ready=1; break; }
  [ $((i%10)) -eq 0 ] && echo "  loading t=${i}s freeRAM=$(free_gb)GB"; sleep 1
done
[ "$ready" = 1 ] || { echo "  never healthy"; tail -25 "$LOG"; exit 3; }
echo "### warmup"; curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":30,\"stream\":false}" \
  >"$OUTDIR/$E-warmup.json" 2>&1
python3 -c "import json;d=json.load(open('$OUTDIR/$E-warmup.json'));print('OK usage:',d.get('usage'))" 2>&1 | head -2
echo "### probe"; python3 "$PROBE" http://127.0.0.1:$PORT/v1 "$ID" "llama.cpp-gguf-q4km" | tee "$PROBE_OUT"
echo "### llama-benchy"; uvx llama-benchy --base-url http://127.0.0.1:$PORT/v1 --model "$ID" \
  --enable-prefix-caching --pp $PP --tg $TG --depth $DEPTHS --runs $RUNS --skip-coherence \
  --save-result "${BENCH_OUT%.json}${BENCH_SUFFIX}.json" --format "$BENCHY_FORMAT" $BENCHY_EXTRA >"${BENCH_LOG%.log}${BENCH_SUFFIX}.log" 2>&1
echo "### benchmark exit=$?"; cat "$BENCH_OUT" 2>/dev/null
echo "### llama.cpp server eval (authoritative decode/prefill):"; grep -iE "prompt eval time|eval time =" "$LOG" | tail -6
echo "### DONE"
