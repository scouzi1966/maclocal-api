#!/bin/bash
# Engine: Rapid-MLX (raullenchai/Rapid-MLX) on Qwen3.6-27B-4bit. Probe (canonical) + prefix cache.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT="${PORT:-9999}"; E=rapidmlx; ID=rapid-qwen3.6-27b-4bit
LOG="$OUTDIR/$E-server.log"; PROBE_OUT="$OUTDIR/$E-probe.json"

echo "### kill other engines + free port"; kill_mlx_engines; free_port "$PORT"; echo "  freeRAM=$(free_gb)GB"
echo "### start rapid-mlx serve (local path, prefix cache on)"
# pass the LOCAL model dir to avoid any re-download; served-model-name gives a clean API id
# --no-mllm: text-only path. The VLM checkpoint's hybrid DeltaNet backbone is incompatible with
# rapid-mlx's multimodal continuous-batching (ArraysCache); text-only is correct for decode benchmarking.
HF_HUB_OFFLINE=1 "$RAPID_BIN" --no-telemetry serve "$MODEL_PATH" --served-model-name "$ID" \
  --host 127.0.0.1 --port "$PORT" --enable-prefix-cache --no-mllm >"$LOG" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; pkill -f "rapid-mlx serve" 2>/dev/null' EXIT INT TERM
ready=0
for i in $(seq 1 220); do
  kill -0 $SRV 2>/dev/null || { echo "  DIED"; tail -30 "$LOG"; exit 1; }
  low_ram && { echo "  !! LOW RAM"; exit 2; }
  curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "  ready t=${i}s gpu=$(gpu)% freeRAM=$(free_gb)GB"; ready=1; break; }
  [ $((i%10)) -eq 0 ] && echo "  loading t=${i}s freeRAM=$(free_gb)GB"; sleep 1
done
[ "$ready" = 1 ] || { echo "  never ready"; tail -30 "$LOG"; exit 3; }
echo "### warmup"; curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":30,\"stream\":false}" \
  >"$OUTDIR/$E-warmup.json" 2>&1
python3 -c "import json;d=json.load(open('$OUTDIR/$E-warmup.json'));assert d.get('choices');print('OK usage:',d.get('usage'))" 2>&1 | head -2 \
  || { echo '### Rapid-MLX FAILED to run qwen3_5'; head -c 600 "$OUTDIR/$E-warmup.json"; exit 5; }
echo "### probe (canonical: server usage / decode window)"; python3 "$PROBE" http://127.0.0.1:$PORT/v1 "$ID" "rapidmlx-4bit" | tee "$PROBE_OUT"
echo "### rapid-mlx prefix-cache reload probe (run probe twice; 2nd should hit cache)"; python3 "$PROBE" http://127.0.0.1:$PORT/v1 "$ID" "rapidmlx-4bit-warm" 2>/dev/null | grep -E "prefill_tps|prefill_ttft" | sed 's/^/  warm: /'
echo "### DONE"
