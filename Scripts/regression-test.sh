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
MLX_SMALL_MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MLX_CACHED_MODEL="mlx-community/lille-130m-instruct-8bit"
PORT_AFM=9871
PORT_MLX=9872
RESULTS_FILE="/tmp/regression-test-results.jsonl"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

while getopts "b:h" opt; do
  case $opt in
    b) AFM="$OPTARG" ;;
    h) cat <<HELP
Usage: $0 [-b /path/to/afm]

Options:
  -b PATH   Path to afm binary (default: /tmp/afm-fresh-build/.build/release/afm)
  -h        Show this help

Environment variables:
  AFM_BIN=PATH              Same as -b
  RUN_GATEWAY_TESTS=1       Enable Section 14: Gateway Mode (requires external backends)
  MACAFM_MLX_MODEL_CACHE    Override MLX model cache directory
HELP
       exit 0 ;;
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
# Schema validation helpers (Python-based, called from schema_api_test)
###############################################################################

validate_chat_response() {
  local response="$1"
  python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception as e:
    print(f'SCHEMA_FAIL: invalid JSON: {e}')
    sys.exit(1)
errors = []
for f in ('id', 'object', 'created', 'model', 'choices', 'usage'):
    if f not in d: errors.append(f'missing field: {f}')
if d.get('object') != 'chat.completion':
    errors.append(f'object should be chat.completion, got {d.get(\"object\")}')
if not isinstance(d.get('created'), int):
    errors.append('created should be int')
if not str(d.get('id','')).startswith('chatcmpl-'):
    errors.append(f'id should start with chatcmpl-, got {d.get(\"id\")}')
choices = d.get('choices', [])
if not isinstance(choices, list) or len(choices) < 1:
    errors.append('choices should be non-empty array')
else:
    c = choices[0]
    for f in ('index', 'message', 'finish_reason'):
        if f not in c: errors.append(f'choice missing: {f}')
    msg = c.get('message', {})
    if 'role' not in msg: errors.append('message missing role')
    if 'content' not in msg: errors.append('message missing content')
usage = d.get('usage', {})
if isinstance(usage, dict):
    for f in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        if f not in usage: errors.append(f'usage missing: {f}')
if errors:
    print('SCHEMA_FAIL: ' + '; '.join(errors))
    sys.exit(1)
print('SCHEMA_OK')
" "$response"
}

validate_stream_response() {
  local response="$1"
  python3 -c "
import json, sys
raw = sys.argv[1]
errors = []
lines = [l for l in raw.split('\n') if l.strip()]
data_lines = [l for l in lines if l.startswith('data: ')]
if not data_lines:
    print('SCHEMA_FAIL: no SSE data: lines found')
    sys.exit(1)
# Check [DONE] terminator
has_done = any(l.strip() == 'data: [DONE]' for l in data_lines)
if not has_done:
    errors.append('missing [DONE] terminator')
# Validate chunk objects
chunks = []
for dl in data_lines:
    payload = dl[len('data: '):]
    if payload.strip() == '[DONE]':
        continue
    try:
        c = json.loads(payload)
        chunks.append(c)
    except Exception as e:
        errors.append(f'invalid JSON chunk: {e}')
        break
if chunks:
    c0 = chunks[0]
    if c0.get('object') != 'chat.completion.chunk':
        errors.append(f'object should be chat.completion.chunk, got {c0.get(\"object\")}')
    if not str(c0.get('id','')).startswith('chatcmpl-'):
        errors.append(f'chunk id should start with chatcmpl-, got {c0.get(\"id\")}')
    for f in ('id', 'object', 'created', 'model', 'choices'):
        if f not in c0: errors.append(f'chunk missing: {f}')
    ch = c0.get('choices', [{}])
    if isinstance(ch, list) and len(ch) > 0:
        sc = ch[0]
        if 'delta' not in sc: errors.append('stream choice missing delta')
        if 'index' not in sc: errors.append('stream choice missing index')
if errors:
    print('SCHEMA_FAIL: ' + '; '.join(errors))
    sys.exit(1)
print('SCHEMA_OK')
" "$response"
}

validate_error_response() {
  local response="$1"
  python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception as e:
    print(f'SCHEMA_FAIL: invalid JSON: {e}')
    sys.exit(1)
errors = []
if 'error' not in d:
    errors.append('missing error object')
else:
    err = d['error']
    if not isinstance(err, dict):
        errors.append('error should be an object')
    else:
        if 'message' not in err: errors.append('error missing message')
        if 'type' not in err: errors.append('error missing type')
if errors:
    print('SCHEMA_FAIL: ' + '; '.join(errors))
    sys.exit(1)
print('SCHEMA_OK')
" "$response"
}

validate_models_response() {
  local response="$1"
  python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception as e:
    print(f'SCHEMA_FAIL: invalid JSON: {e}')
    sys.exit(1)
errors = []
if d.get('object') != 'list':
    errors.append(f'object should be list, got {d.get(\"object\")}')
data = d.get('data')
if not isinstance(data, list) or len(data) < 1:
    errors.append('data should be non-empty array')
else:
    m = data[0]
    for f in ('id', 'object', 'created', 'owned_by'):
        if f not in m: errors.append(f'model missing: {f}')
    if m.get('object') != 'model':
        errors.append(f'model object should be model, got {m.get(\"object\")}')
if errors:
    print('SCHEMA_FAIL: ' + '; '.join(errors))
    sys.exit(1)
print('SCHEMA_OK')
" "$response"
}

validate_health_response() {
  local response="$1"
  python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception as e:
    print(f'SCHEMA_FAIL: invalid JSON: {e}')
    sys.exit(1)
errors = []
for f in ('status', 'timestamp', 'version'):
    if f not in d: errors.append(f'missing field: {f}')
if errors:
    print('SCHEMA_FAIL: ' + '; '.join(errors))
    sys.exit(1)
print('SCHEMA_OK')
" "$response"
}

# Schema-validated API test — runs curl then validates response with a validator function
schema_api_test() {
  local section="$1" name="$2" port="$3" method="$4" endpoint="$5" data="$6" validator="$7"
  local expect_status="${8:-}"
  local t0=$SECONDS
  local response http_code rc
  if [ "$method" = "GET" ]; then
    response=$(curl -s -w '\n%{http_code}' --max-time 30 "http://127.0.0.1:${port}${endpoint}" 2>&1) && rc=$? || rc=$?
  else
    response=$(curl -s -w '\n%{http_code}' --max-time 60 -X POST "http://127.0.0.1:${port}${endpoint}" \
      -H "Content-Type: application/json" -d "$data" 2>&1) && rc=$? || rc=$?
  fi
  local elapsed=$((SECONDS - t0))

  if [ $rc -ne 0 ]; then
    local err
    err=$(echo "$response" | head -1 | cut -c1-200)
    record "$section" "$name" "FAIL" "curl error $rc: $err" "$elapsed"
    return
  fi

  # Split response body and HTTP status code
  http_code=$(echo "$response" | tail -1)
  local body
  body=$(echo "$response" | sed '$d')

  # Check expected HTTP status if specified
  if [ -n "$expect_status" ] && [ "$http_code" != "$expect_status" ]; then
    record "$section" "$name" "FAIL" "HTTP $http_code (expected $expect_status)" "$elapsed"
    return
  fi

  # Run the validator function
  local vresult
  vresult=$($validator "$body" 2>&1) && rc=$? || rc=$?
  if [ $rc -eq 0 ]; then
    record "$section" "$name" "PASS" "" "$elapsed"
  else
    record "$section" "$name" "FAIL" "$vresult" "$elapsed"
  fi
}

# Schema-validated streaming test — needs special curl (no -w for SSE)
schema_stream_test() {
  local section="$1" name="$2" port="$3" data="$4"
  local t0=$SECONDS
  local response rc
  response=$(curl -s --max-time 60 -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" -d "$data" 2>&1) && rc=$? || rc=$?
  local elapsed=$((SECONDS - t0))

  if [ $rc -ne 0 ]; then
    local err
    err=$(echo "$response" | head -1 | cut -c1-200)
    record "$section" "$name" "FAIL" "curl error $rc: $err" "$elapsed"
    return
  fi

  local vresult
  vresult=$(validate_stream_response "$response" 2>&1) && rc=$? || rc=$?
  if [ $rc -eq 0 ]; then
    record "$section" "$name" "PASS" "" "$elapsed"
  else
    record "$section" "$name" "FAIL" "$vresult" "$elapsed"
  fi
}

# HTTP status code test — checks only the status code, no schema validation
http_status_test() {
  local section="$1" name="$2" port="$3" method="$4" endpoint="$5" data="$6" expect_status="$7"
  local t0=$SECONDS
  local http_code rc
  if [ "$method" = "GET" ]; then
    http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "http://127.0.0.1:${port}${endpoint}" 2>&1) && rc=$? || rc=$?
  else
    http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 -X POST "http://127.0.0.1:${port}${endpoint}" \
      -H "Content-Type: application/json" -d "$data" 2>&1) && rc=$? || rc=$?
  fi
  local elapsed=$((SECONDS - t0))

  if [ $rc -ne 0 ]; then
    record "$section" "$name" "FAIL" "curl error $rc" "$elapsed"
  elif [ "$http_code" = "$expect_status" ]; then
    record "$section" "$name" "PASS" "" "$elapsed"
  else
    record "$section" "$name" "FAIL" "HTTP $http_code (expected $expect_status)" "$elapsed"
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

if [ $rc -eq 0 ] && [ -n "$output" ]; then
  record "$SEC" "download $MLX_SMALL_MODEL" "PASS" "downloaded in ${elapsed}s" "$elapsed"
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

# NOTE: "mlx nonexistent model" test removed — it hangs and can't be reliably
# killed on macOS. Test manually with:  afm mlx -m "nonexistent/model-xyz" -s "Test"
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
echo "━━━ Section 13: OpenAI Schema Validation ━━━"
SEC="Schema Validation"

# --- AFM Server schema tests ---
if start_server $PORT_AFM "$AFM" -p $PORT_AFM; then
  schema_api_test "$SEC" "AFM: chat response schema" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"user","content":"Say hi"}]}' \
    validate_chat_response "200"

  schema_api_test "$SEC" "AFM: chat with temperature schema" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"user","content":"Test"}],"temperature":0.5}' \
    validate_chat_response "200"

  schema_stream_test "$SEC" "AFM: streaming SSE + chunk schema" $PORT_AFM \
    '{"model":"foundation","messages":[{"role":"user","content":"Count to 3"}],"stream":true}'

  schema_api_test "$SEC" "AFM: error response schema" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[]}' \
    validate_error_response ""

  schema_api_test "$SEC" "AFM: models endpoint schema" $PORT_AFM GET "/v1/models" "" \
    validate_models_response "200"

  schema_api_test "$SEC" "AFM: health endpoint schema" $PORT_AFM GET "/health" "" \
    validate_health_response "200"

  http_status_test "$SEC" "AFM: 200 on valid chat" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[{"role":"user","content":"Hi"}]}' "200"

  http_status_test "$SEC" "AFM: 400 on empty messages" $PORT_AFM POST "/v1/chat/completions" \
    '{"model":"foundation","messages":[]}' "400"

  kill_server $_SERVER_PID
else
  record "$SEC" "AFM schema server start" "FAIL" "server did not start" "30"
fi

# --- MLX Server schema tests ---
if start_server $PORT_MLX "$AFM" mlx -m "$MLX_CACHED_MODEL" -p $PORT_MLX; then
  schema_api_test "$SEC" "MLX: chat response schema" $PORT_MLX POST "/v1/chat/completions" \
    '{"model":"test","messages":[{"role":"user","content":"Say hi"}],"max_tokens":50}' \
    validate_chat_response "200"

  schema_stream_test "$SEC" "MLX: streaming SSE + chunk schema" $PORT_MLX \
    '{"model":"test","messages":[{"role":"user","content":"Count to 3"}],"stream":true,"max_tokens":50}'

  schema_api_test "$SEC" "MLX: error response schema" $PORT_MLX POST "/v1/chat/completions" \
    '{"model":"test","messages":[]}' \
    validate_error_response ""

  schema_api_test "$SEC" "MLX: models endpoint schema" $PORT_MLX GET "/v1/models" "" \
    validate_models_response "200"

  kill_server $_SERVER_PID
else
  record "$SEC" "MLX schema server start" "FAIL" "server did not start" "30"
fi
echo ""

###############################################################################
# Section 14: Gateway Mode (optional — requires external backends like Ollama)
# Enable with: RUN_GATEWAY_TESTS=1 ./Scripts/regression-test.sh
###############################################################################
PORT_GW=9874

if [ "${RUN_GATEWAY_TESTS:-0}" = "1" ]; then
  echo "━━━ Section 14: Gateway Mode ━━━"
  SEC="Gateway Mode"

  if start_server $PORT_GW "$AFM" -p $PORT_GW -g; then
    # Models endpoint should include discovered backends
    api_test "$SEC" "GET /v1/models (gateway)" $PORT_GW GET "/v1/models" "" "data"
    schema_api_test "$SEC" "models schema (gateway)" $PORT_GW GET "/v1/models" "" \
      validate_models_response "200"

    # Health should still work
    schema_api_test "$SEC" "health schema (gateway)" $PORT_GW GET "/health" "" \
      validate_health_response "200"

    # Chat completion via AFM backend (always available)
    schema_api_test "$SEC" "AFM chat via gateway" $PORT_GW POST "/v1/chat/completions" \
      '{"model":"foundation","messages":[{"role":"user","content":"Say hi"}]}' \
      validate_chat_response "200"

    # Streaming via gateway
    schema_stream_test "$SEC" "AFM streaming via gateway" $PORT_GW \
      '{"model":"foundation","messages":[{"role":"user","content":"Count to 3"}],"stream":true}'

    # If Ollama is running, test proxied completion
    if curl -s --max-time 5 "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
      # Get first available Ollama model
      OLLAMA_MODEL=$(curl -s "http://127.0.0.1:11434/api/tags" | python3 -c "import json,sys; m=json.load(sys.stdin).get('models',[]); print(m[0]['name'] if m else '')" 2>/dev/null)
      if [ -n "$OLLAMA_MODEL" ]; then
        api_test "$SEC" "Ollama proxy: $OLLAMA_MODEL" $PORT_GW POST "/v1/chat/completions" \
          "{\"model\":\"$OLLAMA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":50}" "choices"
      else
        record "$SEC" "Ollama proxy" "SKIP" "Ollama running but no models loaded" "0"
      fi
    else
      record "$SEC" "Ollama proxy" "SKIP" "Ollama not running on :11434" "0"
    fi

    kill_server $_SERVER_PID
  else
    record "$SEC" "gateway server start" "FAIL" "server did not start" "30"
  fi
  echo ""
else
  echo "━━━ Section 14: Gateway Mode (skipped) ━━━"
  echo "  ⏭️  Set RUN_GATEWAY_TESTS=1 to enable"
  echo ""
fi

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
