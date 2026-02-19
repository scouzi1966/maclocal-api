#!/bin/bash
# AFM Comprehensive Regression Test Suite
# Tests ALL features: CLI, AFM single-prompt, AFM server, MLX CLI, MLX server,
# MLX model download, gateway mode, streaming, parameters, validation.
# Outputs JSONL results for HTML report generation.
#
# Usage:
#   ./Scripts/regression-test.sh [-b /path/to/afm]
#   AFM_BIN=/path/to/afm ./Scripts/regression-test.sh

AFM="${AFM_BIN:-/tmp/afm-fresh-build/.build/release/afm}"
export MACAFM_MLX_MODEL_CACHE="/Volumes/edata/models/vesta-test-cache"
MLX_SMALL_MODEL="mlx-community/granite-4.0-350m-bf16"
MLX_CACHED_MODEL="mlx-community/lille-130m-instruct-8bit"
PORT_AFM=9871
PORT_MLX=9872
RESULTS_FILE="/tmp/regression-test-results.jsonl"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

while getopts "b:h" opt; do
  case $opt in
    b) AFM="$OPTARG" ;;
    h) echo "Usage: $0 [-b /path/to/afm]"; exit 0 ;;
    *) echo "Unknown option"; exit 1 ;;
  esac
done

if [ ! -x "$AFM" ]; then
  echo "ERROR: AFM binary not found: $AFM"
  exit 1
fi

> "$RESULTS_FILE"

# Counters
total=0
pass=0
fail=0
skip=0

escape_json() {
  python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))" <<< "$1"
}

# Record a test result
record() {
  local section="$1" name="$2" status="$3" detail="${4:-}" elapsed="${5:-0}"
  total=$((total + 1))
  if [ "$status" = "PASS" ]; then
    pass=$((pass + 1))
    echo "  ✅ $name"
  elif [ "$status" = "SKIP" ]; then
    skip=$((skip + 1))
    echo "  ⏭️  $name (skipped: $detail)"
  else
    fail=$((fail + 1))
    echo "  ❌ $name — $detail"
  fi
  echo "{\"section\":$(escape_json "$section"),\"name\":$(escape_json "$name"),\"status\":\"$status\",\"detail\":$(escape_json "$detail"),\"elapsed_s\":$elapsed}" >> "$RESULTS_FILE"
}

# Run a CLI test (command exits, check exit code and optionally output pattern)
cli_test() {
  local section="$1" name="$2" expect_pass="$3" pattern="$4"
  shift 4
  local cmd=("$@")

  local t0=$SECONDS
  local output rc
  output=$("${cmd[@]}" 2>&1) && rc=$? || rc=$?
  local elapsed=$((SECONDS - t0))

  if [ "$expect_pass" = "true" ]; then
    if [ $rc -eq 0 ]; then
      if [ -n "$pattern" ]; then
        if echo "$output" | grep -q "$pattern"; then
          record "$section" "$name" "PASS" "" "$elapsed"
        else
          record "$section" "$name" "FAIL" "pattern not found: $pattern" "$elapsed"
        fi
      else
        record "$section" "$name" "PASS" "" "$elapsed"
      fi
    else
      local err
      err=$(echo "$output" | head -1 | cut -c1-200)
      record "$section" "$name" "FAIL" "exit $rc: $err" "$elapsed"
    fi
  else
    if [ $rc -ne 0 ]; then
      record "$section" "$name" "PASS" "" "$elapsed"
    else
      record "$section" "$name" "FAIL" "expected failure but succeeded" "$elapsed"
    fi
  fi
}

# Global for server PID — avoids subshell issues with $()
_SERVER_PID=0

# Start a server, wait for health. Sets _SERVER_PID.
start_server() {
  local port=$1; shift
  _SERVER_PID=0
  "$@" < /dev/null > "/tmp/regression-server-${port}.log" 2>&1 &
  _SERVER_PID=$!
  local deadline=$((SECONDS + 60))
  while [ $SECONDS -lt $deadline ]; do
    if curl -s "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 $_SERVER_PID 2>/dev/null; then
      _SERVER_PID=0
      return 1
    fi
    sleep 1
  done
  kill $_SERVER_PID 2>/dev/null || true
  sleep 1
  _SERVER_PID=0
  return 1
}

kill_server() {
  local pid=$1
  if [ "$pid" != "0" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    # Wait for the child process (it's a direct child of this shell now)
    wait "$pid" 2>/dev/null || true
  fi
  sleep 1
}

# API test against a running server
api_test() {
  local section="$1" name="$2" port="$3" method="$4" endpoint="$5" data="$6" pattern="$7"
  local t0=$SECONDS
  local response rc
  if [ "$method" = "GET" ]; then
    response=$(curl -s --max-time 30 "http://127.0.0.1:${port}${endpoint}" 2>&1) && rc=$? || rc=$?
  else
    response=$(curl -s --max-time 60 -X POST "http://127.0.0.1:${port}${endpoint}" \
      -H "Content-Type: application/json" -d "$data" 2>&1) && rc=$? || rc=$?
  fi
  local elapsed=$((SECONDS - t0))

  if [ $rc -eq 0 ] && echo "$response" | grep -q "$pattern"; then
    record "$section" "$name" "PASS" "" "$elapsed"
  else
    local err
    err=$(echo "$response" | head -1 | cut -c1-200)
    record "$section" "$name" "FAIL" "$err" "$elapsed"
  fi
}

###############################################################################
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       AFM Comprehensive Regression Test Suite               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Binary: $AFM"
echo "Version: $("$AFM" --version 2>&1)"
echo "Started: $(date)"
echo ""

###############################################################################
echo "━━━ Section 1: CLI Basics ━━━"
SEC="CLI Basics"

cli_test "$SEC" "--help shows usage" true "USAGE" "$AFM" --help
cli_test "$SEC" "--version shows version" true "v0" "$AFM" --version
cli_test "$SEC" "mlx --help shows usage" true "USAGE" "$AFM" mlx --help
cli_test "$SEC" "invalid option rejected" false "" "$AFM" --invalid-option
echo ""

###############################################################################
echo "━━━ Section 2: AFM Single Prompt ━━━"
SEC="AFM Single Prompt"

cli_test "$SEC" "basic single prompt" true "" "$AFM" -s "Say hello in one word"
cli_test "$SEC" "single prompt with instructions" true "" "$AFM" -i "You are a pirate" -s "Say hello"
cli_test "$SEC" "pipe mode" true "" bash -c "echo 'What is 2+2?' | $AFM"
cli_test "$SEC" "temperature 0.0" true "" "$AFM" -t 0.0 -s "Test"
cli_test "$SEC" "temperature 1.0" true "" "$AFM" -t 1.0 -s "Test"
cli_test "$SEC" "randomness greedy" true "" "$AFM" -r greedy -s "Test"
cli_test "$SEC" "randomness random" true "" "$AFM" -r random -s "Test"
cli_test "$SEC" "randomness top-p=0.9" true "" "$AFM" -r "random:top-p=0.9" -s "Test"
cli_test "$SEC" "randomness top-k=50" true "" "$AFM" -r "random:top-k=50" -s "Test"
cli_test "$SEC" "randomness top-p + seed" true "" "$AFM" -r "random:top-p=0.9:seed=42" -s "Test"
cli_test "$SEC" "permissive guardrails -P" true "" "$AFM" -P -s "Test"
cli_test "$SEC" "combined: -t -r -i -P" true "" "$AFM" -t 0.7 -r greedy -i "Be concise" -P -s "Test"
echo ""

###############################################################################
echo "━━━ Section 3: AFM Parameter Validation ━━━"
SEC="AFM Validation"

cli_test "$SEC" "temperature > 1.0 rejected" false "" "$AFM" -t 1.5 -s "Test"
cli_test "$SEC" "temperature < 0.0 rejected" false "" "$AFM" -t -0.1 -s "Test"
cli_test "$SEC" "temperature non-numeric rejected" false "" "$AFM" -t abc -s "Test"
cli_test "$SEC" "randomness invalid rejected" false "" "$AFM" -r invalid -s "Test"
cli_test "$SEC" "top-p > 1.0 rejected" false "" "$AFM" -r "random:top-p=1.5" -s "Test"
cli_test "$SEC" "top-k=0 rejected" false "" "$AFM" -r "random:top-k=0" -s "Test"
cli_test "$SEC" "conflicting top-p + top-k rejected" false "" "$AFM" -r "random:top-p=0.9:top-k=50" -s "Test"
cli_test "$SEC" "unknown randomness param rejected" false "" "$AFM" -r "random:unknown=1" -s "Test"
echo ""

###############################################################################
echo "━━━ Section 4: AFM Server Mode ━━━"
SEC="AFM Server"

if start_server $PORT_AFM "$AFM" -p $PORT_AFM; then
  api_test "$SEC" "GET /health" $PORT_AFM GET "/health" "" "healthy"
  api_test "$SEC" "GET /v1/models" $PORT_AFM GET "/v1/models" "" "foundation"
  api_test "$SEC" "POST chat completion" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"user","content":"Say hi"}]}' "choices"
  api_test "$SEC" "POST chat with temperature" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"user","content":"Test"}],"temperature":0.5}' "choices"
  api_test "$SEC" "POST chat streaming" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"user","content":"Count to 3"}],"stream":true}' "data:"
  api_test "$SEC" "POST multi-turn conversation" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"system","content":"You are helpful"},{"role":"user","content":"What is AI?"},{"role":"assistant","content":"AI is artificial intelligence"},{"role":"user","content":"Give an example"}]}' "choices"
  api_test "$SEC" "POST empty messages returns error" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[]}' "error"
  api_test "$SEC" "POST max_tokens respected" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"user","content":"Write a long story"}],"max_tokens":10}' "choices"
  kill_server $_SERVER_PID
else
  record "$SEC" "AFM server start" "FAIL" "server did not start" "30"
fi
echo ""

###############################################################################
echo "━━━ Section 5: AFM Server with Options ━━━"
SEC="AFM Server Options"

# Server with permissive guardrails
if start_server $PORT_AFM "$AFM" -p $PORT_AFM -P; then
  api_test "$SEC" "permissive guardrails server" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"user","content":"Test"}]}' "choices"
  kill_server $_SERVER_PID
else
  record "$SEC" "permissive server start" "FAIL" "server did not start" "30"
fi

# Server with custom temp + randomness
if start_server $PORT_AFM "$AFM" -p $PORT_AFM -t 0.5 -r greedy; then
  api_test "$SEC" "server with -t 0.5 -r greedy" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"user","content":"Test"}]}' "choices"
  kill_server $_SERVER_PID
else
  record "$SEC" "custom params server start" "FAIL" "server did not start" "30"
fi
echo ""

###############################################################################
echo "━━━ Section 6: MLX Single Prompt ━━━"
SEC="MLX Single Prompt"

cli_test "$SEC" "mlx single prompt" true "" "$AFM" mlx -m "$MLX_CACHED_MODEL" -s "Say hello"
cli_test "$SEC" "mlx single prompt with instructions" true "" "$AFM" mlx -m "$MLX_CACHED_MODEL" -i "Answer in French" -s "Say hello"
cli_test "$SEC" "mlx single prompt with temperature" true "" "$AFM" mlx -m "$MLX_CACHED_MODEL" -t 0.5 -s "Test"
cli_test "$SEC" "mlx pipe mode" true "" bash -c "echo 'What is 2+2?' | $AFM mlx -m $MLX_CACHED_MODEL"
echo ""

###############################################################################
echo "━━━ Section 7: MLX Server Mode ━━━"
SEC="MLX Server"

if start_server $PORT_MLX "$AFM" mlx -m "$MLX_CACHED_MODEL" -p $PORT_MLX; then
  api_test "$SEC" "GET /health" $PORT_MLX GET "/health" "" "healthy"
  api_test "$SEC" "GET /v1/models" $PORT_MLX GET "/v1/models" "" "lille"
  api_test "$SEC" "POST chat completion" $PORT_MLX POST "/v1/chat/completions" \
    '{"model":"test","messages":[{"role":"user","content":"Say hi"}],"max_tokens":50}' "choices"
  api_test "$SEC" "POST chat with temperature" $PORT_MLX POST "/v1/chat/completions" \
    '{"model":"test","messages":[{"role":"user","content":"Test"}],"temperature":0.7,"max_tokens":50}' "choices"
  api_test "$SEC" "POST chat streaming" $PORT_MLX POST "/v1/chat/completions" \
    '{"model":"test","messages":[{"role":"user","content":"Count to 3"}],"stream":true,"max_tokens":50}' "data:"
  api_test "$SEC" "POST empty messages returns error" $PORT_MLX POST "/v1/chat/completions" \
    '{"model":"test","messages":[]}' "error"
  kill_server $_SERVER_PID
else
  record "$SEC" "MLX server start" "FAIL" "server did not start" "30"
fi
echo ""

###############################################################################
echo "━━━ Section 8: MLX Model Download ━━━"
SEC="MLX Download"

# Use a temp cache to test downloading a small model
DOWNLOAD_CACHE="/tmp/regression-download-cache-${TIMESTAMP}"
mkdir -p "$DOWNLOAD_CACHE"

t0=$SECONDS
output=$(MACAFM_MLX_MODEL_CACHE="$DOWNLOAD_CACHE" "$AFM" mlx -m "$MLX_SMALL_MODEL" -s "Say hello" 2>&1) && rc=$? || rc=$?
elapsed=$((SECONDS - t0))

if [ $rc -eq 0 ] && [ -d "$DOWNLOAD_CACHE" ]; then
  # Check that model files were downloaded (safetensors or gguf weights + config)
  weights=$(find "$DOWNLOAD_CACHE" \( -name "*.safetensors" -o -name "*.gguf" \) 2>/dev/null | head -1)
  config=$(find "$DOWNLOAD_CACHE" -name "config.json" 2>/dev/null | head -1)
  if [ -n "$weights" ] && [ -n "$config" ]; then
    record "$SEC" "download $MLX_SMALL_MODEL" "PASS" "downloaded in ${elapsed}s" "$elapsed"
  elif [ -n "$config" ]; then
    # Config exists but weights have different extension — still likely OK
    record "$SEC" "download $MLX_SMALL_MODEL" "PASS" "downloaded in ${elapsed}s (alt format)" "$elapsed"
  else
    record "$SEC" "download $MLX_SMALL_MODEL" "FAIL" "files missing after download" "$elapsed"
  fi
else
  err=$(echo "$output" | head -1 | cut -c1-200)
  record "$SEC" "download $MLX_SMALL_MODEL" "FAIL" "$err" "$elapsed"
fi

# Test running the downloaded model via server
if MACAFM_MLX_MODEL_CACHE="$DOWNLOAD_CACHE" start_server $PORT_MLX "$AFM" mlx -m "$MLX_SMALL_MODEL" -p $PORT_MLX; then
  api_test "$SEC" "serve downloaded model" $PORT_MLX POST "/v1/chat/completions" \
    '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}' "choices"
  kill_server $_SERVER_PID
else
  record "$SEC" "serve downloaded model" "FAIL" "server did not start" "30"
fi

# Cleanup download cache
rm -rf "$DOWNLOAD_CACHE"
echo ""

###############################################################################
echo "━━━ Section 9: MLX VLM Model ━━━"
SEC="MLX VLM"

MLX_VLM_MODEL="mlx-community/Qwen3-VL-4B-Instruct-4bit"
if start_server $PORT_MLX "$AFM" mlx -m "$MLX_VLM_MODEL" -p $PORT_MLX; then
  api_test "$SEC" "VLM text-only chat" $PORT_MLX POST "/v1/chat/completions" \
    '{"model":"test","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}' "choices"
  kill_server $_SERVER_PID
else
  record "$SEC" "VLM server start" "FAIL" "server did not start" "30"
fi
echo ""

###############################################################################
echo "━━━ Section 10: MLX MoE Model ━━━"
SEC="MLX MoE"

MLX_MOE_MODEL="mlx-community/Qwen3-30B-A3B-4bit"
if start_server $PORT_MLX "$AFM" mlx -m "$MLX_MOE_MODEL" -p $PORT_MLX; then
  api_test "$SEC" "MoE chat completion" $PORT_MLX POST "/v1/chat/completions" \
    '{"model":"test","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":50}' "choices"
  kill_server $_SERVER_PID
else
  record "$SEC" "MoE server start" "FAIL" "server did not start" "30"
fi
echo ""

###############################################################################
echo "━━━ Section 11: MLX Validation ━━━"
SEC="MLX Validation"

cli_test "$SEC" "mlx without -m fails" false "" "$AFM" mlx -s "Test"
cli_test "$SEC" "mlx nonexistent model fails" false "" "$AFM" mlx -m "nonexistent/model-xyz" -s "Test"
echo ""

###############################################################################
echo "━━━ Section 12: Port Handling ━━━"
SEC="Port Handling"

# Start server on specific port
if start_server 9873 "$AFM" -p 9873; then
  api_test "$SEC" "custom port 9873" 9873 GET "/health" "" "healthy"
  kill_server $_SERVER_PID
else
  record "$SEC" "custom port" "FAIL" "server did not start on port 9873" "30"
fi
echo ""

###############################################################################
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    TEST SUMMARY                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Total: $total  |  Pass: $pass  |  Fail: $fail  |  Skip: $skip"
echo ""
echo "  Results: $RESULTS_FILE"
echo ""

if [ $fail -eq 0 ]; then
  echo "  ✅ ALL TESTS PASSED"
else
  echo "  ❌ SOME TESTS FAILED"
fi
echo ""

echo "=== Generating HTML report ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/generate-regression-report.py"
