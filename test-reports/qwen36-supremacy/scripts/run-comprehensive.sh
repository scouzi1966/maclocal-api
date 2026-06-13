#!/bin/bash
# Comprehensive single-stream profile across all engines (latency + prefill/cache + depth sweep).
# One engine on the GPU at a time. Writes results/<engine>-full.json. Streams per-section progress.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
FP="$SCRIPT_DIR/probe_full.py"
ENGINES="${ENGINES:-afm rapidmlx mlx_vlm omlx lmstudio ollama llamacpp}"
CTX=20480   # big enough for the 16384 depth point + generation

wait_http(){ for i in $(seq 1 240); do curl -s -m2 "$1" >/dev/null 2>&1 && return 0; low_ram && return 9; sleep 1; done; return 1; }

probe(){ # name target model
  echo ">>> PROBE $1"
  python3 "$FP" "$2/v1" "$3" "$1" "$OUTDIR/$1-full.json"
}

for e in $ENGINES; do
  echo "=================== $e ==================="
  kill_mlx_engines; free_port 9999; free_port 8000
  case "$e" in
    afm)
      MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" "$AFM_BIN" mlx -m "$MODEL_ID" --port 9999 --enable-prefix-caching </dev/null >"$OUTDIR/$e-full-srv.log" 2>&1 &
      wait_http http://127.0.0.1:9999/v1/models && probe afm http://127.0.0.1:9999 "$MODEL_ID" || echo "  afm not ready" ;;
    rapidmlx)
      HF_HUB_OFFLINE=1 "$RAPID_BIN" --no-telemetry serve "$MODEL_PATH" --served-model-name rapid-qwen --host 127.0.0.1 --port 9999 --enable-prefix-cache --no-mllm >"$OUTDIR/$e-full-srv.log" 2>&1 &
      wait_http http://127.0.0.1:9999/v1/models && probe rapidmlx http://127.0.0.1:9999 rapid-qwen || echo "  rapidmlx not ready" ;;
    mlx_vlm)
      "$VLM_PYTHON" -m mlx_vlm.server --model "$MODEL_PATH" --port 9999 --host 127.0.0.1 >"$OUTDIR/$e-full-srv.log" 2>&1 &
      wait_http http://127.0.0.1:9999/v1/models && probe mlx_vlm http://127.0.0.1:9999 "$MODEL_PATH" || echo "  mlx_vlm not ready" ;;
    omlx)
      "$OMLX_BIN" serve --model-dir "$(dirname "$MODEL_PATH")" --host 127.0.0.1 --port 8000 >"$OUTDIR/$e-full-srv.log" 2>&1 &
      wait_http http://127.0.0.1:8000/v1/models && { curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3.6-27B-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' >/dev/null 2>&1; probe omlx http://127.0.0.1:8000 Qwen3.6-27B-4bit; } || echo "  omlx not ready" ;;
    lmstudio)
      "$LMS_BIN" server start --port 1234 >"$OUTDIR/$e-full-srv.log" 2>&1; "$LMS_BIN" unload --all >/dev/null 2>&1
      "$LMS_BIN" load qwen3.6-27b --gpu max --context-length $CTX -y >>"$OUTDIR/$e-full-srv.log" 2>&1
      wait_http http://127.0.0.1:1234/v1/models && probe lmstudio http://127.0.0.1:1234 "$MODEL_ID"; "$LMS_BIN" unload --all >/dev/null 2>&1 ;;
    ollama)
      curl -s -m5 http://127.0.0.1:11434/v1/models >/dev/null 2>&1 || (ollama serve >/dev/null 2>&1 &); sleep 3
      # raise context so the depth sweep isn't truncated at the default
      curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"qwen3.6:27b-mlx\",\"keep_alive\":\"30m\",\"prompt\":\"hi\",\"options\":{\"num_ctx\":$CTX}}" >/dev/null 2>&1
      probe ollama http://127.0.0.1:11434 qwen3.6:27b-mlx ;;
    llamacpp)
      llama-server -m "$GGUF_BLOB" --host 127.0.0.1 --port 9999 -c $CTX -ngl 999 --no-warmup >"$OUTDIR/$e-full-srv.log" 2>&1 &
      for i in $(seq 1 240); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m2 http://127.0.0.1:9999/health 2>/dev/null)" = 200 ] && break; sleep 1; done
      probe llamacpp http://127.0.0.1:9999 qwen3.6-gguf ;;
  esac
done
kill_mlx_engines; free_port 9999; free_port 8000
echo "=================== COMPREHENSIVE DONE ==================="
