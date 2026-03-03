#!/bin/bash
# Regression test for chat_template_kwargs / --no-think support (Issue #34).
#
# Tests that enable_thinking can be controlled via:
#   1. API request-level chat_template_kwargs
#   2. CLI --no-think server flag
#   3. Precedence: request-level overrides server-level
#
# Requires a thinking-capable model (e.g. Qwen3.5). Non-thinking models
# are detected and tests are skipped gracefully.
#
# Usage:
#   ./Scripts/test-chat-template-kwargs.sh --port PORT [--bin BIN]
#
# The script manages its own server lifecycle — it starts/stops the server
# as needed (default mode, then --no-think mode). No pre-running server needed.
#
# Example:
#   ./Scripts/test-chat-template-kwargs.sh --port 9998 --model mlx-community/Qwen3.5-35B-A3B-4bit
#   ./Scripts/test-chat-template-kwargs.sh --port 9998 --model mlx-community/Qwen3.5-35B-A3B-4bit --bin .build/release/afm

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
PORT=9998
MODEL=""
BIN=".build/arm64-apple-macosx/release/afm"
MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/test-reports"
SERVER_PID=""
MAX_TOKENS_LOW=50
MAX_TOKENS_HIGH=2000
PROMPT='What is 2+2? Answer in one word.'

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
  echo "ERROR: --model is required (must be a thinking-capable model, e.g. Qwen3.5)"
  exit 1
fi

BASE_URL="http://127.0.0.1:$PORT"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_FILE="$REPORT_DIR/chat-template-kwargs-${TIMESTAMP}.html"
JSONL_FILE="$REPORT_DIR/chat-template-kwargs-${TIMESTAMP}.jsonl"
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

# ─── Server management ───────────────────────────────────────────────────────

start_server() {
  local extra_flags="${1:-}"
  echo "  Starting server: $BIN mlx -m $MODEL --port $PORT $extra_flags"
  MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" "$BIN" mlx -m "$MODEL" --port "$PORT" $extra_flags >/dev/null 2>&1 &
  SERVER_PID=$!

  # Wait for server to be ready (up to 60s)
  local waited=0
  while ! curl -sf --max-time 2 "$BASE_URL/v1/models" >/dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ $waited -ge 60 ]; then
      echo "  ERROR: Server did not become ready within 60s"
      kill "$SERVER_PID" 2>/dev/null || true
      SERVER_PID=""
      return 1
    fi
    # Check server is still alive
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

# ─── Assertion helpers ────────────────────────────────────────────────────────

# Check non-streaming response for thinking state.
# Returns: "no_think" if content present + no reasoning_content
#          "thinking" if reasoning_content present
#          "empty"    if content is empty (bug: thinking consumed all tokens)
#          "error"    on parse failure
check_response() {
  local resp="$1"
  echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print('error')
        sys.exit()
    msg = d['choices'][0]['message']
    content = msg.get('content', '')
    reasoning = msg.get('reasoning_content')
    if reasoning and len(reasoning) > 0:
        if content and len(content.strip()) > 0:
            print('thinking')
        else:
            print('empty')
    else:
        if content and len(content.strip()) > 0:
            print('no_think')
        else:
            print('empty')
except Exception as e:
    print('error')
" 2>/dev/null || echo "error"
}

# Check streaming response for thinking state.
# Returns same values as check_response.
check_stream() {
  local resp="$1"
  echo "$resp" | python3 -c "
import sys, json
found_reasoning = False
found_content = False
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        d = json.loads(line[6:])
        delta = d.get('choices', [{}])[0].get('delta', {})
        rc = delta.get('reasoning_content', '')
        c = delta.get('content', '')
        if rc and len(rc.strip()) > 0:
            found_reasoning = True
        if c and len(c.strip()) > 0:
            found_content = True
    except:
        pass
if found_reasoning and found_content:
    print('thinking')
elif found_reasoning and not found_content:
    print('empty')
elif found_content and not found_reasoning:
    print('no_think')
else:
    print('error')
" 2>/dev/null || echo "error"
}

# ─── Banner ───────────────────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AFM chat_template_kwargs Regression Tests (Issue #34)"
echo "  Model: $MODEL"
echo "  Binary: $BIN"
echo "  Port: $PORT"
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

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Default mode (thinking ON, no --no-think)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "🧠 Section 1: Default Server Mode (thinking ON)"

start_server ""

# Probe: does this model actually support thinking?
# Use longer timeout for first request (cold prefill)
probe_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0}" 180)
probe_state=$(check_response "$probe_resp")

if [ "$probe_state" != "thinking" ]; then
  echo "  Model does not produce reasoning_content — skipping all thinking tests"
  run_test "Preflight" "Model supports thinking" "thinking model" "SKIP" "0"
  stop_server
  # Print summary
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped ($TOTAL total)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
fi

echo "  Model supports thinking — running all tests"
echo ""

# --- Test 1: Default (no kwargs) — thinking should be ON ---
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
state=$(check_response "$resp")
if [ "$state" = "thinking" ]; then
  run_test "Default" "Default non-streaming: thinking ON" "thinking" "PASS" "$dur"
else
  run_test "Default" "Default non-streaming: thinking ON" "thinking" "FAIL: got $state" "$dur"
fi

# --- Test 2: API enable_thinking:false, non-streaming, max_tokens=50 ---
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":$MAX_TOKENS_LOW,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
state=$(check_response "$resp")
if [ "$state" = "no_think" ]; then
  run_test "API kwargs" "API enable_thinking:false non-stream (50 tok)" "no_think" "PASS" "$dur"
else
  run_test "API kwargs" "API enable_thinking:false non-stream (50 tok)" "no_think" "FAIL: got $state" "$dur"
fi

# --- Test 3: API enable_thinking:false, streaming, max_tokens=50 ---
t0=$(now_ms)
resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":$MAX_TOKENS_LOW,\"temperature\":0,\"stream\":true}")
dur=$(( $(now_ms) - t0 ))
state=$(check_stream "$resp")
if [ "$state" = "no_think" ]; then
  run_test "API kwargs" "API enable_thinking:false streaming (50 tok)" "no_think" "PASS" "$dur"
else
  run_test "API kwargs" "API enable_thinking:false streaming (50 tok)" "no_think" "FAIL: got $state" "$dur"
fi

# --- Test 4: API enable_thinking:false, non-streaming, max_tokens=2000 ---
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
state=$(check_response "$resp")
if [ "$state" = "no_think" ]; then
  run_test "API kwargs" "API enable_thinking:false non-stream (2k tok)" "no_think" "PASS" "$dur"
else
  run_test "API kwargs" "API enable_thinking:false non-stream (2k tok)" "no_think" "FAIL: got $state" "$dur"
fi

# --- Test 5: API enable_thinking:false, streaming, max_tokens=2000 ---
t0=$(now_ms)
resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0,\"stream\":true}")
dur=$(( $(now_ms) - t0 ))
state=$(check_stream "$resp")
if [ "$state" = "no_think" ]; then
  run_test "API kwargs" "API enable_thinking:false streaming (2k tok)" "no_think" "PASS" "$dur"
else
  run_test "API kwargs" "API enable_thinking:false streaming (2k tok)" "no_think" "FAIL: got $state" "$dur"
fi

# --- Test 6: Default (no kwargs), max_tokens=2000 — confirm thinking still ON ---
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
state=$(check_response "$resp")
if [ "$state" = "thinking" ]; then
  run_test "Default" "Default non-streaming 2k: thinking ON (regression)" "thinking" "PASS" "$dur"
else
  run_test "Default" "Default non-streaming 2k: thinking ON (regression)" "thinking" "FAIL: got $state" "$dur"
fi

stop_server

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: --no-think server mode
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "🔇 Section 2: Server --no-think Mode"

start_server "--no-think"

# --- Test 7: --no-think, no API kwargs, non-streaming, 50 tok ---
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS_LOW,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
state=$(check_response "$resp")
if [ "$state" = "no_think" ]; then
  run_test "CLI --no-think" "--no-think non-stream (50 tok)" "no_think" "PASS" "$dur"
else
  run_test "CLI --no-think" "--no-think non-stream (50 tok)" "no_think" "FAIL: got $state" "$dur"
fi

# --- Test 8: --no-think, no API kwargs, streaming, 50 tok ---
t0=$(now_ms)
resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS_LOW,\"temperature\":0,\"stream\":true}")
dur=$(( $(now_ms) - t0 ))
state=$(check_stream "$resp")
if [ "$state" = "no_think" ]; then
  run_test "CLI --no-think" "--no-think streaming (50 tok)" "no_think" "PASS" "$dur"
else
  run_test "CLI --no-think" "--no-think streaming (50 tok)" "no_think" "FAIL: got $state" "$dur"
fi

# --- Test 9: --no-think, no API kwargs, non-streaming, 2k tok ---
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
state=$(check_response "$resp")
if [ "$state" = "no_think" ]; then
  run_test "CLI --no-think" "--no-think non-stream (2k tok)" "no_think" "PASS" "$dur"
else
  run_test "CLI --no-think" "--no-think non-stream (2k tok)" "no_think" "FAIL: got $state" "$dur"
fi

# --- Test 10: --no-think, no API kwargs, streaming, 2k tok ---
t0=$(now_ms)
resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0,\"stream\":true}")
dur=$(( $(now_ms) - t0 ))
state=$(check_stream "$resp")
if [ "$state" = "no_think" ]; then
  run_test "CLI --no-think" "--no-think streaming (2k tok)" "no_think" "PASS" "$dur"
else
  run_test "CLI --no-think" "--no-think streaming (2k tok)" "no_think" "FAIL: got $state" "$dur"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Precedence — CLI + API combined
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "⚖️  Section 3: Precedence (CLI --no-think + API kwargs)"

# --- Test 11: --no-think + API false (both agree), non-streaming ---
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
state=$(check_response "$resp")
if [ "$state" = "no_think" ]; then
  run_test "Precedence" "--no-think + API false: both agree, non-stream" "no_think" "PASS" "$dur"
else
  run_test "Precedence" "--no-think + API false: both agree, non-stream" "no_think" "FAIL: got $state" "$dur"
fi

# --- Test 12: --no-think + API false (both agree), streaming ---
t0=$(now_ms)
resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0,\"stream\":true}")
dur=$(( $(now_ms) - t0 ))
state=$(check_stream "$resp")
if [ "$state" = "no_think" ]; then
  run_test "Precedence" "--no-think + API false: both agree, streaming" "no_think" "PASS" "$dur"
else
  run_test "Precedence" "--no-think + API false: both agree, streaming" "no_think" "FAIL: got $state" "$dur"
fi

# --- Test 13: --no-think + API true (request overrides), non-streaming ---
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"chat_template_kwargs\":{\"enable_thinking\":true},\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
state=$(check_response "$resp")
if [ "$state" = "thinking" ]; then
  run_test "Precedence" "--no-think + API true: request overrides, non-stream" "thinking" "PASS" "$dur"
else
  run_test "Precedence" "--no-think + API true: request overrides, non-stream" "thinking" "FAIL: got $state" "$dur"
fi

# --- Test 14: --no-think + API true (request overrides), streaming ---
t0=$(now_ms)
resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"chat_template_kwargs\":{\"enable_thinking\":true},\"max_tokens\":$MAX_TOKENS_HIGH,\"temperature\":0,\"stream\":true}")
dur=$(( $(now_ms) - t0 ))
state=$(check_stream "$resp")
if [ "$state" = "thinking" ]; then
  run_test "Precedence" "--no-think + API true: request overrides, streaming" "thinking" "PASS" "$dur"
else
  run_test "Precedence" "--no-think + API true: request overrides, streaming" "thinking" "FAIL: got $state" "$dur"
fi

stop_server

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

status_class = "pass" if fail_count == 0 else "fail"

with open(report_file, 'w') as f:
    f.write(f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>chat_template_kwargs Regression Tests</title>
<style>
:root {{ --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#c9d1d9; --heading:#f0f6fc; --green:#3fb950; --red:#f85149; --blue:#58a6ff; --dim:#8b949e; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; background:var(--bg); color:var(--text); padding:2rem; max-width:1000px; margin:0 auto; line-height:1.6; }}
h1 {{ color:var(--heading); font-size:1.5rem; margin-bottom:0.5rem; }}
.meta {{ color:var(--dim); font-size:0.9rem; margin-bottom:1.5rem; }}
table {{ width:100%; border-collapse:collapse; margin:1rem 0; }}
th,td {{ padding:0.5rem 0.8rem; text-align:left; border-bottom:1px solid var(--border); font-size:0.9rem; }}
th {{ color:var(--heading); font-weight:600; }}
.pass {{ color:var(--green); }}
.fail {{ color:var(--red); }}
.skip {{ color:var(--dim); }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }}
.badge-pass {{ background:#1b3a2a; color:var(--green); }}
.badge-fail {{ background:#3d1a1e; color:var(--red); }}
.badge-skip {{ background:#2a2a1b; color:var(--dim); }}
.summary {{ display:flex; gap:2rem; margin:1rem 0; }}
.stat {{ text-align:center; }}
.stat .n {{ font-size:2rem; font-weight:700; color:var(--heading); }}
.stat .l {{ color:var(--dim); font-size:0.85rem; }}
</style></head><body>
<h1>chat_template_kwargs Regression Tests (Issue #34)</h1>
<p class="meta">Model: <code>{html.escape(model)}</code> | Elapsed: {elapsed}s</p>
<div class="summary">
<div class="stat"><div class="n">{total_count}</div><div class="l">Total</div></div>
<div class="stat"><div class="n pass">{pass_count}</div><div class="l">Pass</div></div>
<div class="stat"><div class="n fail">{fail_count}</div><div class="l">Fail</div></div>
<div class="stat"><div class="n skip">{skip_count}</div><div class="l">Skip</div></div>
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
Generated by Scripts/test-chat-template-kwargs.sh for <a href="https://github.com/AnomalyCo/maclocal-api/issues/34" style="color:var(--blue);">Issue #34</a>.
</p></body></html>''')

print(f"  HTML report: {report_file}")
PYEOF

if [ $FAIL -gt 0 ]; then
  exit 1
fi
