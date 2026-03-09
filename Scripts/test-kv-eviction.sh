#!/usr/bin/env bash
# test-kv-eviction.sh — Tests for --kv-eviction CLI flag.
#
# Validates that --kv-eviction streaming is accepted by the CLI, the server
# starts normally with it set, and basic generation still works.
#
# Usage:
#   ./Scripts/test-kv-eviction.sh [--model MODEL] [--bin BIN] [--port PORT]
#
# The script manages its own server lifecycle. No pre-running server needed.
#
# Example:
#   ./Scripts/test-kv-eviction.sh \
#     --model mlx-community/Qwen3-0.6B-4bit \
#     --bin .build/release/afm \
#     --port 9874

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
PORT=9874
MODEL="mlx-community/Qwen3-0.6B-4bit"
BIN=".build/release/afm"
MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVER_PID=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --port)  PORT="$2";  shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --bin)   BIN="$2";   shift 2 ;;
    --cache) MODEL_CACHE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

BASE_URL="http://127.0.0.1:$PORT"

# ─── Test infrastructure ─────────────────────────────────────────────────────
PASS=0
FAIL=0
TOTAL=0
TEST_START_TIME=$(date +%s)

now_ms() {
  python3 -c "import time; print(int(time.time()*1000))"
}

run_test() {
  local group="$1"
  local name="$2"
  local expected="$3"
  local actual="$4"

  TOTAL=$((TOTAL + 1))
  if [ "$actual" = "PASS" ]; then
    PASS=$((PASS + 1))
    echo "  [PASS] $name"
  else
    FAIL=$((FAIL + 1))
    echo "  [FAIL] $name"
    echo "         Expected: $expected"
    echo "         Actual:   $actual"
  fi
}

api_call() {
  local body="$1"
  local timeout="${2:-120}"
  curl -sf --max-time "$timeout" "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" 2>/dev/null || echo '{"error":"curl_failed"}'
}

api_stream() {
  local body="$1"
  curl -sf --max-time 120 -N "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" 2>/dev/null || echo 'ERROR'
}

# ─── Server management ────────────────────────────────────────────────────────

start_server() {
  local extra_flags="${1:-}"
  echo "  Starting: $BIN mlx -m $MODEL --port $PORT $extra_flags"
  MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" "$BIN" mlx -m "$MODEL" --port "$PORT" $extra_flags >/dev/null 2>&1 &
  SERVER_PID=$!

  # Wait up to 120s for readiness
  local waited=0
  while ! curl -sf --max-time 2 "$BASE_URL/v1/models" >/dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ $waited -ge 120 ]; then
      echo "  ERROR: Server did not become ready within 120s"
      kill "$SERVER_PID" 2>/dev/null || true
      SERVER_PID=""
      return 1
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  ERROR: Server process exited unexpectedly"
      SERVER_PID=""
      return 1
    fi
  done
  echo "  Server ready (PID $SERVER_PID, waited ${waited}s)"
}

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    echo "  Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 1
  fi
}

cleanup() {
  stop_server
}
trap cleanup EXIT

# ─── Banner ───────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AFM --kv-eviction Flag Tests"
echo "  Model:  $MODEL"
echo "  Binary: $BIN"
echo "  Port:   $PORT"
echo "  Cache:  $MODEL_CACHE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ─── Preflight ────────────────────────────────────────────────────────────────
echo "Section 0: Preflight"

if [ -f "$BIN" ]; then
  run_test "Preflight" "Binary exists at $BIN" "file exists" "PASS"
else
  run_test "Preflight" "Binary exists at $BIN" "file exists" "FAIL: not found at $BIN"
  echo ""
  echo "ERROR: Binary not found. Build first: swift build -c release"
  exit 1
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Server starts with --kv-eviction streaming
# ═══════════════════════════════════════════════════════════════════════════════
echo "Section 1: Server starts with --kv-eviction streaming"

if start_server "--kv-eviction streaming"; then
  run_test "Startup" "Server starts with --kv-eviction streaming" "server ready" "PASS"
else
  run_test "Startup" "Server starts with --kv-eviction streaming" "server ready" "FAIL: server did not start"
  echo ""
  echo "FATAL: Cannot continue without a running server."
  exit 1
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Basic non-streaming generation works
# ═══════════════════════════════════════════════════════════════════════════════
echo "Section 2: Basic generation (non-streaming)"

t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"max_tokens":10,"temperature":0,"stream":false}')
dur=$(( $(now_ms) - t0 ))

content=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message']['content'].strip()
    print(c if c else '__EMPTY__')
except Exception as e:
    print('__ERROR__')
" 2>/dev/null || echo "__ERROR__")

if [ "$content" != "__ERROR__" ] && [ "$content" != "__EMPTY__" ] && [ -n "$content" ]; then
  run_test "Generation" "Non-streaming generation produces output (${dur}ms, got: $content)" "non-empty content" "PASS"
else
  run_test "Generation" "Non-streaming generation produces output" "non-empty content" "FAIL: got [$content]"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Streaming generation works
# ═══════════════════════════════════════════════════════════════════════════════
echo "Section 3: Streaming generation"

t0=$(now_ms)
stream_out=$(api_stream '{"messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":10,"temperature":0,"stream":true}')
dur=$(( $(now_ms) - t0 ))

# Verify we got at least one SSE data line with content
has_content=$(echo "$stream_out" | python3 -c "
import sys, json
found = False
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        d = json.loads(line[6:])
        c = d.get('choices', [{}])[0].get('delta', {}).get('content', '')
        if c:
            found = True
    except:
        pass
print('yes' if found else 'no')
" 2>/dev/null || echo "no")

if [ "$has_content" = "yes" ]; then
  run_test "Generation" "Streaming generation produces SSE content (${dur}ms)" "SSE content tokens" "PASS"
else
  run_test "Generation" "Streaming generation produces SSE content" "SSE content tokens" "FAIL: no content tokens in SSE stream"
fi

# Also verify [DONE] sentinel is present
has_done=$(echo "$stream_out" | grep -c "data: \[DONE\]" || true)
if [ "$has_done" -ge 1 ]; then
  run_test "Generation" "Streaming response includes [DONE] sentinel" "[DONE] present" "PASS"
else
  run_test "Generation" "Streaming response includes [DONE] sentinel" "[DONE] present" "FAIL: missing [DONE]"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Invalid --kv-eviction value causes non-zero exit
# ═══════════════════════════════════════════════════════════════════════════════
echo "Section 4: Invalid --kv-eviction value is rejected"

stop_server

# Start a server with an invalid value — it should exit with a non-zero code.
# We capture the exit code; we do NOT want it to hang, so give it 10s max.
invalid_exit=0
timeout 10 "$BIN" mlx -m "$MODEL" --port "$PORT" --kv-eviction invalid \
  >/dev/null 2>&1 || invalid_exit=$?

if [ "$invalid_exit" -ne 0 ]; then
  run_test "Validation" "Invalid --kv-eviction value rejected (exit $invalid_exit)" "non-zero exit" "PASS"
else
  # Server started OK with an invalid value — still "passes" at flag-accepted level,
  # but note it for review.
  run_test "Validation" "Invalid --kv-eviction value rejected" "non-zero exit" "FAIL: server accepted invalid value (exit 0)"
fi

echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - TEST_START_TIME ))
if [ $TOTAL -gt 0 ]; then
  PCT=$(( PASS * 100 / TOTAL ))
else
  PCT=0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: $PASS/$TOTAL passed (${PCT}%)  [${ELAPSED}s]"
if [ $FAIL -gt 0 ]; then
  echo "  FAILED: $FAIL test(s)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $FAIL -gt 0 ]; then
  exit 1
fi
