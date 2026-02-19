#!/bin/bash
# MLX Model Test Suite
# Tests each model by starting the server, sending a prompt, and collecting results.

set -uo pipefail

AFM="${AFM_BIN:-/tmp/afm-fresh-build/.build/release/afm}"
export MACAFM_MLX_MODEL_CACHE="/Volumes/edata/models/vesta-test-cache"
PORT=9877
PROMPT="Explain calculus concepts from limits through multivariable calculus with rigorous mathematical notation"
RESULTS_FILE="/tmp/mlx-test-results.jsonl"
MAX_TOKENS=3000
TIMEOUT_LOAD=120     # seconds to wait for server to start (2 min)
TIMEOUT_GENERATE=900 # seconds for generation

> "$RESULTS_FILE"

MODELS=(
  "hub/models--mlx-community--Qwen3-VL-4B-Instruct-4bit"
  "mlx-community/Apertus-8B-Instruct-2509-4bit"
  "mlx-community/exaone-4.0-1.2b-4bit"
  "mlx-community/gemma-3-4b-it-8bit"
  "mlx-community/gemma-3n-E2B-it-lm-4bit"
  "mlx-community/GLM-4.7-Flash-4bit"
  "mlx-community/GLM-5-4bit"
  "mlx-community/gpt-oss-20b-MXFP4-Q4"
  "mlx-community/gpt-oss-20b-MXFP4-Q8"
  "mlx-community/granite-4.0-h-tiny-4bit"
  "mlx-community/JoyAI-LLM-Flash-4bit-DWQ"
  "mlx-community/LFM2-2.6B-4bit"
  "mlx-community/LFM2-VL-3B-4bit"
  "mlx-community/lille-130m-instruct-8bit"
  "mlx-community/Ling-mini-2.0-4bit"
  "mlx-community/Llama-3.2-1B-Instruct-4bit"
  "mlx-community/MiniMax-M2.5-5bit"
  "mlx-community/MiniMax-M2.5-6bit"
  "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"
  "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
  "mlx-community/Qwen3-0.6B-4bit"
  "mlx-community/Qwen3-30B-A3B-4bit"
  "mlx-community/Qwen3-Coder-Next-4bit"
  "mlx-community/Qwen3-VL-4B-Instruct-4bit"
  "mlx-community/Qwen3-VL-4B-Instruct-8bit"
  "mlx-community/Qwen3.5-397B-A17B-4bit"
  "mlx-community/SmolLM3-3B-4bit"
  "mlx-community/Kimi-K2.5-3bit"
)

SERVER_PID=0

# Kill server and all its children
kill_server() {
  local pid=$1
  if [ "$pid" != "0" ] && kill -0 "$pid" 2>/dev/null; then
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 0.5
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

# Ctrl+C handler — clean up and exit
cleanup() {
  echo ""
  echo "  ⚠️  Interrupted — cleaning up..."
  kill_server $SERVER_PID
  echo ""
  echo "=== Test interrupted. Partial results in $RESULTS_FILE ==="
  exit 130
}
trap cleanup INT TERM

wait_for_server() {
  local deadline=$((SECONDS + TIMEOUT_LOAD))
  while [ $SECONDS -lt $deadline ]; do
    if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    # Check if server process died
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

escape_json() {
  python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$1"
}

total=${#MODELS[@]}
idx=0

for model in "${MODELS[@]}"; do
  idx=$((idx + 1))
  echo "=== [$idx/$total] Testing: $model ==="

  # Build server args
  EXTRA_ARGS=()
  SYS_PROMPT=""
  if echo "$model" | grep -qi "gpt-oss"; then
    SYS_PROMPT="Reasoning:low"
    EXTRA_ARGS=(-i "$SYS_PROMPT")
  fi

  # Start server
  SERVER_LOG="/tmp/mlx-server-${idx}.log"
  load_start=$SECONDS
  "$AFM" mlx -m "$model" -p "$PORT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  # Wait for server
  if ! wait_for_server; then
    load_time=$((SECONDS - load_start))
    # Grab error from log
    error_msg=$(grep -i "error\|fatal\|unsupported\|not supported\|failed" "$SERVER_LOG" | head -3 | tr '\n' ' ' | sed 's/"/\\"/g' | head -c 500)
    if [ -z "$error_msg" ]; then
      if ! kill -0 $SERVER_PID 2>/dev/null; then
        error_msg="Server process died (check $SERVER_LOG)"
      else
        error_msg="Server failed to start within ${TIMEOUT_LOAD}s"
      fi
    fi
    echo "  FAIL: $error_msg"
    echo "{\"model\":$(escape_json "$model"),\"status\":\"FAIL\",\"error\":$(escape_json "$error_msg"),\"load_time_s\":$load_time}" >> "$RESULTS_FILE"
    kill_server $SERVER_PID
    SERVER_PID=0
    sleep 2
    echo ""
    continue
  fi

  load_time=$((SECONDS - load_start))
  echo "  Server ready in ${load_time}s"

  # Build request body
  if [ -n "$SYS_PROMPT" ]; then
    REQUEST_BODY=$(python3 -c "
import json
body = {
  'model': '$model',
  'messages': [
    {'role': 'system', 'content': 'Reasoning:low'},
    {'role': 'user', 'content': '''$PROMPT'''}
  ],
  'max_tokens': $MAX_TOKENS,
  'temperature': 0.7,
  'stream': False
}
print(json.dumps(body))
")
  else
    REQUEST_BODY=$(python3 -c "
import json
body = {
  'model': '$model',
  'messages': [
    {'role': 'user', 'content': '''$PROMPT'''}
  ],
  'max_tokens': $MAX_TOKENS,
  'temperature': 0.7,
  'stream': False
}
print(json.dumps(body))
")
  fi

  # Send request
  gen_start=$(python3 -c "import time; print(time.time())")
  RESPONSE=$(curl -s --max-time $TIMEOUT_GENERATE \
    -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$REQUEST_BODY" 2>&1)
  gen_end=$(python3 -c "import time; print(time.time())")
  curl_exit=$?

  if [ $curl_exit -ne 0 ] || echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if 'choices' in d else 1)" 2>/dev/null; then
    if [ $curl_exit -ne 0 ]; then
      error_msg="curl failed (exit $curl_exit) or timeout"
      echo "  FAIL: $error_msg"
      echo "{\"model\":$(escape_json "$model"),\"status\":\"FAIL\",\"error\":$(escape_json "$error_msg"),\"load_time_s\":$load_time}" >> "$RESULTS_FILE"
    else
      # Success — extract metrics
      METRICS=$(echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
gen_time = $gen_end - $gen_start
c = d.get('choices', [{}])[0]
msg = c.get('message', {}).get('content', '')
usage = d.get('usage', {})
prompt_tokens = usage.get('prompt_tokens', 0)
completion_tokens = usage.get('completion_tokens', 0)
total_tokens = usage.get('total_tokens', 0)
tps = completion_tokens / gen_time if gen_time > 0 else 0
content_preview = msg[:300].replace('\\n', ' ')
result = {
  'model': '$model',
  'status': 'OK',
  'load_time_s': $load_time,
  'gen_time_s': round(gen_time, 2),
  'prompt_tokens': prompt_tokens,
  'completion_tokens': completion_tokens,
  'total_tokens': total_tokens,
  'tokens_per_sec': round(tps, 2),
  'content_preview': content_preview,
  'content': msg
}
print(json.dumps(result))
" 2>&1)
      echo "$METRICS" >> "$RESULTS_FILE"
      tps=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tokens_per_sec',0))")
      ctok=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('completion_tokens',0))")
      gtime=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('gen_time_s',0))")
      echo "  OK: ${ctok} tokens in ${gtime}s (${tps} tok/s)"
    fi
  else
    # Response doesn't have choices — it's an error
    error_msg=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('error',{}).get('message', str(d))[:500])" 2>/dev/null || echo "$RESPONSE" | head -c 500)
    echo "  FAIL: $error_msg"
    echo "{\"model\":$(escape_json "$model"),\"status\":\"FAIL\",\"error\":$(escape_json "$error_msg"),\"load_time_s\":$load_time}" >> "$RESULTS_FILE"
  fi

  # Kill server
  kill_server $SERVER_PID
  SERVER_PID=0
  sleep 2
  echo ""
done

echo "=== All tests complete. Results in $RESULTS_FILE ==="
echo ""
echo "=== Generating HTML report ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/generate-report.py"
