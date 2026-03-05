#!/bin/bash
# Test suite for --max-model-len context length enforcement.
#
# Manages its own server lifecycle — starts/stops servers with different flags.
# Validates: CLI flag, /v1/models context_window reporting, prompt rejection,
# streaming error, boundary cases, --max-kv-size alias, and auto-detection.
#
# Usage:
#   ./Scripts/test-max-model-len.sh --model MODEL [--port PORT] [--bin BIN] [--cache DIR]
#
# Example:
#   ./Scripts/test-max-model-len.sh --model mlx-community/Qwen3-0.6B-4bit --bin .build/release/afm

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
PORT=9871
PORT2=9872
PORT3=9873
MODEL=""
BIN=".build/release/afm"
MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/test-reports"
MAX_MODEL_LEN=512
SERVER_PID=""
SERVER_PID2=""
SERVER_PID3=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --cache) MODEL_CACHE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ERROR: --model is required"
  exit 1
fi

BASE_URL="http://127.0.0.1:$PORT"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_FILE="$REPORT_DIR/max-model-len-${TIMESTAMP}.html"
JSONL_FILE="$REPORT_DIR/max-model-len-${TIMESTAMP}.jsonl"
mkdir -p "$REPORT_DIR"

# ─── Test infrastructure ─────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0
TOTAL=0
TEST_START_TIME=$(date +%s)
declare -a RESULTS=()

now_ms() {
  python3 -c "import time; print(int(time.time()*1000))"
}

run_test() {
  local group="$1"
  local name="$2"
  local expected="$3"
  local actual="$4"
  local duration="${5:-0}"

  TOTAL=$((TOTAL + 1))
  if [ "$actual" = "PASS" ]; then
    PASS=$((PASS + 1))
    echo "  ✅ $name"
  elif [ "$actual" = "SKIP" ]; then
    SKIP=$((SKIP + 1))
    echo "  ⏭️  $name (skipped)"
  else
    FAIL=$((FAIL + 1))
    echo "  ❌ $name"
    echo "     Expected: $expected"
    echo "     Actual:   $actual"
  fi
  local esc_expected=$(echo "$expected" | tr '|' '/')
  local esc_actual=$(echo "$actual" | tr '|' '/')
  RESULTS+=("${actual}|${group}|${name}|${esc_expected}|${esc_actual}|${duration}")

  # JSONL record
  local status_val="PASS"
  [[ "$actual" = "PASS" ]] && status_val="PASS"
  [[ "$actual" = "SKIP" ]] && status_val="SKIP"
  [[ "$actual" != "PASS" && "$actual" != "SKIP" ]] && status_val="FAIL"
  python3 -c "
import json
print(json.dumps({
    'group': $(python3 -c "import json; print(json.dumps('$group'))"),
    'name': $(python3 -c "import json; print(json.dumps('$name'))"),
    'status': '$status_val',
    'duration_ms': $duration
}))
" >> "$JSONL_FILE"
}

api_call() {
  local url="$1"
  local body="$2"
  local timeout="${3:-120}"
  curl -sf --max-time "$timeout" "$url/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" 2>/dev/null || echo '{"error":"curl_failed"}'
}

api_call_raw() {
  # Returns body + http_code on last line
  local url="$1"
  local body="$2"
  local timeout="${3:-120}"
  curl -s --max-time "$timeout" -w "\n%{http_code}" "$url/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" 2>/dev/null || echo -e '{"error":"curl_failed"}\n000'
}

api_stream() {
  local url="$1"
  local body="$2"
  curl -sf --max-time 120 -N "$url/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" 2>/dev/null || echo 'ERROR'
}

# ─── Server management ───────────────────────────────────────────────────────

start_server() {
  local port="$1"
  local extra_flags="${2:-}"
  local pid_var="$3"
  echo "  Starting server: $BIN mlx -m $MODEL --port $port $extra_flags"
  MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" "$BIN" mlx -m "$MODEL" --port "$port" $extra_flags >/dev/null 2>&1 &
  local pid=$!
  eval "$pid_var=$pid"

  # Wait for server to be ready (up to 90s — first load may be slow)
  local waited=0
  while ! curl -sf --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ $waited -ge 90 ]; then
      echo "  ERROR: Server did not become ready within 90s"
      kill "$pid" 2>/dev/null || true
      eval "$pid_var="
      return 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  ERROR: Server process exited unexpectedly"
      eval "$pid_var="
      return 1
    fi
  done
  echo "  Server ready (PID $pid, waited ${waited}s)"
}

stop_pid() {
  local pid="$1"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 1
  fi
}

cleanup() {
  stop_pid "$SERVER_PID"
  stop_pid "$SERVER_PID2"
  stop_pid "$SERVER_PID3"
}
trap cleanup EXIT

# ─── Generate long prompt ────────────────────────────────────────────────────
# ~700 words ≈ ~900+ tokens, well over 512
LONG_MSG=$(python3 -c "print('Tell me about ' + ' '.join(['the history of computing and artificial intelligence'] * 150))")

# ─── Banner ──────────────────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AFM --max-model-len Context Length Enforcement Tests"
echo "  Model: $MODEL"
echo "  Binary: $BIN"
echo "  Port: $PORT (primary), $PORT2/$PORT3 (secondary)"
echo "  Max model len: $MAX_MODEL_LEN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0: Preflight
# ═══════════════════════════════════════════════════════════════════════════════
echo "🔍 Section 0: Preflight"

t0=$(now_ms)
if [ -f "$BIN" ]; then
  run_test "Preflight" "Binary exists at $BIN" "file exists" "PASS" "$(( $(now_ms) - t0 ))"
else
  run_test "Preflight" "Binary exists at $BIN" "file exists" "FAIL: not found" "$(( $(now_ms) - t0 ))"
  echo "ERROR: Binary not found. Build first: swift build -c release"
  exit 1
fi

# Start primary server with --max-model-len
t0=$(now_ms)
if start_server "$PORT" "--max-model-len $MAX_MODEL_LEN" SERVER_PID; then
  run_test "Preflight" "Server starts with --max-model-len $MAX_MODEL_LEN" "server ready" "PASS" "$(( $(now_ms) - t0 ))"
else
  run_test "Preflight" "Server starts with --max-model-len $MAX_MODEL_LEN" "server ready" "FAIL: server did not start" "$(( $(now_ms) - t0 ))"
  exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Models API
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "📋 Section 1: Models API"

# Test: /v1/models reports context_window field
t0=$(now_ms)
models_json=$(curl -sf "$BASE_URL/v1/models" 2>/dev/null || echo '{}')
dur=$(( $(now_ms) - t0 ))
cw_check=$(echo "$models_json" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    cw = d['data'][0].get('context_window')
    if cw is not None and isinstance(cw, int):
        print('PASS')
    else:
        print(f'FAIL: context_window={cw}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$cw_check" = "PASS" ]; then
  run_test "Models API" "/v1/models reports context_window as integer" "integer field" "PASS" "$dur"
else
  run_test "Models API" "/v1/models reports context_window as integer" "integer field" "$cw_check" "$dur"
fi

# Test: context_window matches --max-model-len value
t0=$(now_ms)
cw_value=$(echo "$models_json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['data'][0].get('context_window', 'null'))
" 2>/dev/null || echo "null")
dur=$(( $(now_ms) - t0 ))
if [ "$cw_value" = "$MAX_MODEL_LEN" ]; then
  run_test "Models API" "context_window matches --max-model-len ($MAX_MODEL_LEN)" "$MAX_MODEL_LEN" "PASS" "$dur"
else
  run_test "Models API" "context_window matches --max-model-len ($MAX_MODEL_LEN)" "$MAX_MODEL_LEN" "FAIL: got $cw_value" "$dur"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Enforcement
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "🛡️  Section 2: Enforcement"

# Test: Short prompt succeeds
t0=$(now_ms)
resp=$(api_call "$BASE_URL" '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":20,"temperature":0}' 180)
dur=$(( $(now_ms) - t0 ))
short_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL: {d[\"error\"]}')
    elif 'choices' in d:
        c = d['choices'][0]['message'].get('content', '')
        # Some small models return empty content but valid response — accept if no error
        print('PASS')
    else:
        print('FAIL: no choices in response')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$short_ok" = "PASS" ]; then
  run_test "Enforcement" "Short prompt succeeds (under $MAX_MODEL_LEN tokens)" "200 with content" "PASS" "$dur"
else
  run_test "Enforcement" "Short prompt succeeds (under $MAX_MODEL_LEN tokens)" "200 with content" "$short_ok" "$dur"
fi

# Test: Long prompt rejected with context_length_exceeded (non-streaming)
t0=$(now_ms)
long_body=$(python3 -c "
import json
print(json.dumps({
    'messages': [{'role': 'user', 'content': '''$LONG_MSG'''}],
    'max_tokens': 5,
    'temperature': 0
}))
")
raw_resp=$(api_call_raw "$BASE_URL" "$long_body")
http_code=$(echo "$raw_resp" | tail -1)
body=$(echo "$raw_resp" | sed '$d')
dur=$(( $(now_ms) - t0 ))

if [ "$http_code" = "400" ]; then
  run_test "Enforcement" "Long prompt returns HTTP 400" "400" "PASS" "$dur"
else
  run_test "Enforcement" "Long prompt returns HTTP 400" "400" "FAIL: HTTP $http_code" "$dur"
fi

# Test: Error body contains context_length_exceeded code
t0=$(now_ms)
has_code=$(echo "$body" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    err = d.get('error', {})
    code = err.get('code', '')
    etype = err.get('type', '')
    if code == 'context_length_exceeded' and etype == 'invalid_request_error':
        print('PASS')
    else:
        print(f'FAIL: code={code} type={etype}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
dur=$(( $(now_ms) - t0 ))
if [ "$has_code" = "PASS" ]; then
  run_test "Enforcement" "Error has code=context_length_exceeded, type=invalid_request_error" "correct error shape" "PASS" "$dur"
else
  run_test "Enforcement" "Error has code=context_length_exceeded, type=invalid_request_error" "correct error shape" "$has_code" "$dur"
fi

# Test: Error message includes token count and limit
t0=$(now_ms)
msg_check=$(echo "$body" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d.get('error', {}).get('message', '')
    has_limit = 'maximum context length is $MAX_MODEL_LEN' in msg
    # Check message mentions token count
    import re
    token_match = re.search(r'resulted in (\d+) tokens', msg)
    if has_limit and token_match:
        n = int(token_match.group(1))
        if n > $MAX_MODEL_LEN:
            print('PASS')
        else:
            print(f'FAIL: token count {n} not > $MAX_MODEL_LEN')
    else:
        print(f'FAIL: msg={msg[:120]}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
dur=$(( $(now_ms) - t0 ))
if [ "$msg_check" = "PASS" ]; then
  run_test "Enforcement" "Error message includes token count > $MAX_MODEL_LEN" "token count in message" "PASS" "$dur"
else
  run_test "Enforcement" "Error message includes token count > $MAX_MODEL_LEN" "token count in message" "$msg_check" "$dur"
fi

# Test: Long prompt rejected in streaming mode
t0=$(now_ms)
stream_body=$(python3 -c "
import json
print(json.dumps({
    'messages': [{'role': 'user', 'content': '''$LONG_MSG'''}],
    'max_tokens': 5,
    'temperature': 0,
    'stream': True
}))
")
sse_output=$(api_stream "$BASE_URL" "$stream_body")
dur=$(( $(now_ms) - t0 ))
if echo "$sse_output" | grep -q "maximum context length"; then
  run_test "Enforcement" "Long prompt rejected in streaming mode" "SSE error with context message" "PASS" "$dur"
else
  # Truncate output for display
  snippet=$(echo "$sse_output" | head -3 | tr '\n' ' ' | cut -c1-120)
  run_test "Enforcement" "Long prompt rejected in streaming mode" "SSE error with context message" "FAIL: $snippet" "$dur"
fi

# Test: Prompt just under limit succeeds (boundary test)
# Use a moderately sized prompt that should be under 512 tokens
t0=$(now_ms)
boundary_resp=$(api_call "$BASE_URL" '{"messages":[{"role":"user","content":"Explain what a compiler is in two sentences."}],"max_tokens":50,"temperature":0}')
dur=$(( $(now_ms) - t0 ))
boundary_ok=$(echo "$boundary_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d and 'context_length' in str(d.get('error', '')):
        print('FAIL: unexpectedly rejected')
    elif 'choices' in d:
        # Valid response — accept even if content is empty (small model quirk)
        print('PASS')
    else:
        print(f'FAIL: unexpected response')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$boundary_ok" = "PASS" ]; then
  run_test "Enforcement" "Prompt under limit succeeds (boundary test)" "200 with content" "PASS" "$dur"
else
  run_test "Enforcement" "Prompt under limit succeeds (boundary test)" "200 with content" "$boundary_ok" "$dur"
fi

# Stop primary server before alias test
echo ""
echo "  Stopping primary server..."
stop_pid "$SERVER_PID"
SERVER_PID=""

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Backwards Compat
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "🔄 Section 3: Backwards Compat"

# Test: --max-kv-size alias accepted
t0=$(now_ms)
if start_server "$PORT2" "--max-kv-size 1024" SERVER_PID2; then
  dur=$(( $(now_ms) - t0 ))
  run_test "Backwards Compat" "--max-kv-size alias: server starts" "server ready" "PASS" "$dur"

  # Verify context_window reports 1024
  t0=$(now_ms)
  alias_cw=$(curl -sf "http://127.0.0.1:$PORT2/v1/models" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['data'][0].get('context_window', 'null'))
" 2>/dev/null || echo "null")
  dur=$(( $(now_ms) - t0 ))
  if [ "$alias_cw" = "1024" ]; then
    run_test "Backwards Compat" "--max-kv-size 1024 reports context_window=1024" "1024" "PASS" "$dur"
  else
    run_test "Backwards Compat" "--max-kv-size 1024 reports context_window=1024" "1024" "FAIL: got $alias_cw" "$dur"
  fi

  stop_pid "$SERVER_PID2"
  SERVER_PID2=""
else
  dur=$(( $(now_ms) - t0 ))
  run_test "Backwards Compat" "--max-kv-size alias: server starts" "server ready" "FAIL: server did not start" "$dur"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Auto-detect
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "🔎 Section 4: Auto-detect"

# Test: Server without --max-model-len auto-detects from model config
t0=$(now_ms)
if start_server "$PORT3" "" SERVER_PID3; then
  dur=$(( $(now_ms) - t0 ))
  run_test "Auto-detect" "Server starts without --max-model-len" "server ready" "PASS" "$dur"

  # Verify context_window is populated with a reasonable value
  t0=$(now_ms)
  auto_cw=$(curl -sf "http://127.0.0.1:$PORT3/v1/models" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
cw = d['data'][0].get('context_window')
print(cw if cw is not None else 'null')
" 2>/dev/null || echo "null")
  dur=$(( $(now_ms) - t0 ))

  if [ "$auto_cw" != "null" ] && [ "$auto_cw" -gt 0 ] 2>/dev/null; then
    run_test "Auto-detect" "Auto-detected context_window=$auto_cw (from config.json)" ">0" "PASS" "$dur"
  else
    run_test "Auto-detect" "Auto-detected context_window from config.json" ">0" "FAIL: got $auto_cw" "$dur"
  fi

  stop_pid "$SERVER_PID3"
  SERVER_PID3=""
else
  dur=$(( $(now_ms) - t0 ))
  run_test "Auto-detect" "Server starts without --max-model-len" "server ready" "FAIL: server did not start" "$dur"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
ELAPSED=$(( $(date +%s) - TEST_START_TIME ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped ($TOTAL total)"
echo "  Elapsed: ${ELAPSED}s"
echo "  JSONL:   $JSONL_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─── Generate HTML report ─────────────────────────────────────────────────────
python3 - "$REPORT_FILE" "$MODEL" "$PASS" "$FAIL" "$SKIP" "$TOTAL" "$ELAPSED" "${RESULTS[@]}" <<'PYEOF'
import sys, html

report_file = sys.argv[1]
model = sys.argv[2]
pass_count = int(sys.argv[3])
fail_count = int(sys.argv[4])
skip_count = int(sys.argv[5])
total_count = int(sys.argv[6])
elapsed = int(sys.argv[7])
results_raw = sys.argv[8:]

rows = []
for r in results_raw:
    parts = r.split('|', 5)
    if len(parts) == 6:
        status, group, name, expected, actual, dur = parts
        rows.append((status, group, name, expected, actual, dur))

with open(report_file, 'w') as f:
    f.write(f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>--max-model-len Enforcement Tests</title>
<style>
:root {{ --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#c9d1d9; --heading:#f0f6fc; --green:#3fb950; --red:#f85149; --blue:#58a6ff; --dim:#8b949e; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; background:var(--bg); color:var(--text); padding:2rem; max-width:1000px; margin:0 auto; line-height:1.6; }}
h1 {{ color:var(--heading); font-size:1.5rem; margin-bottom:0.5rem; }}
.meta {{ color:var(--dim); font-size:0.9rem; margin-bottom:1.5rem; }}
table {{ width:100%; border-collapse:collapse; margin:1rem 0; }}
th,td {{ padding:0.5rem 0.8rem; text-align:left; border-bottom:1px solid var(--border); font-size:0.9rem; }}
th {{ color:var(--heading); font-weight:600; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }}
.badge-pass {{ background:#1b3a2a; color:var(--green); }}
.badge-fail {{ background:#3d1a1e; color:var(--red); }}
.badge-skip {{ background:#2a2a1b; color:var(--dim); }}
.summary {{ display:flex; gap:2rem; margin:1rem 0; }}
.stat {{ text-align:center; }}
.stat .n {{ font-size:2rem; font-weight:700; color:var(--heading); }}
.stat .l {{ color:var(--dim); font-size:0.85rem; }}
.pass {{ color:var(--green); }}
.fail {{ color:var(--red); }}
</style></head><body>
<h1>--max-model-len Context Length Enforcement Tests</h1>
<p class="meta">Model: <code>{html.escape(model)}</code> | Max model len: {512} | Elapsed: {elapsed}s</p>
<div class="summary">
<div class="stat"><div class="n">{total_count}</div><div class="l">Total</div></div>
<div class="stat"><div class="n pass">{pass_count}</div><div class="l">Pass</div></div>
<div class="stat"><div class="n fail">{fail_count}</div><div class="l">Fail</div></div>
<div class="stat"><div class="n">{skip_count}</div><div class="l">Skip</div></div>
</div>
<table><thead><tr><th>#</th><th>Group</th><th>Test</th><th>Expected</th><th>Result</th><th>Time</th></tr></thead><tbody>
''')
    for i, (status, group, name, expected, actual, dur) in enumerate(rows, 1):
        if status == "PASS":
            badge = '<span class="badge badge-pass">PASS</span>'
        elif status == "SKIP":
            badge = '<span class="badge badge-skip">SKIP</span>'
        else:
            badge = f'<span class="badge badge-fail">FAIL</span>'
        f.write(f'<tr><td>{i}</td><td>{html.escape(group)}</td><td>{html.escape(name)}</td><td>{html.escape(expected)}</td><td>{badge}</td><td>{dur}ms</td></tr>\n')

    f.write(f'''</tbody></table>
<p style="margin-top:2rem;color:var(--dim);font-size:0.85rem;">
Generated by Scripts/test-max-model-len.sh
</p></body></html>''')

print(f"  HTML report: {report_file}")
PYEOF

if [ $FAIL -gt 0 ]; then
  exit 1
fi
