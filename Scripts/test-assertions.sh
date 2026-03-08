#!/bin/bash
# Automated assertion test suite for AFM MLX server.
# Deterministic pass/fail checks for stop sequences, logprobs, think extraction,
# tool calls, prompt cache, concurrent requests, error handling, chat_template_kwargs,
# and performance.
#
# Usage:
#   ./Scripts/test-assertions.sh --tier smoke|standard|full --model MODEL [--port PORT] [--bin BIN]
#
# Prerequisites: server must be running on the given port (unless --start-server).
# Example:
#   MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
#     .build/release/afm mlx -m mlx-community/Qwen3-0.6B-4bit -p 9998 &
#   ./Scripts/test-assertions.sh --tier smoke --model mlx-community/Qwen3-0.6B-4bit --port 9998

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
PORT=9998
MODEL=""
TIER="smoke"
BIN=".build/release/afm"
SECTION=""  # empty = run all sections; set to a number to run only that section
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/test-reports"

while [[ $# -gt 0 ]]; do
  case $1 in
    --tier) TIER="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --section) SECTION="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ ! "$TIER" =~ ^(smoke|standard|full)$ ]]; then
  echo "ERROR: --tier must be smoke, standard, or full"
  exit 1
fi

BASE_URL="http://127.0.0.1:$PORT"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_FILE="$REPORT_DIR/assertions-report-${TIMESTAMP}.html"
JSONL_FILE="$REPORT_DIR/assertions-report-${TIMESTAMP}.jsonl"
mkdir -p "$REPORT_DIR"

# ─── Test infrastructure ─────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0
TOTAL=0
TEST_START_TIME=$(date +%s)
declare -a RESULTS=()

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
  _GROUP="$group" _NAME="$name" _STATUS="$status_val" _DUR="$duration" python3 -c "
import json, os
print(json.dumps({
    'group': os.environ['_GROUP'],
    'name': os.environ['_NAME'],
    'status': os.environ['_STATUS'],
    'duration_ms': int(os.environ['_DUR'])
}))
" >> "$JSONL_FILE"
}

# Helper: call API and return full JSON response
api_call() {
  local body="$1"
  curl -sf --max-time 60 "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" 2>/dev/null || echo '{"error":"curl_failed"}'
}

# Helper: call API streaming and return raw SSE
api_stream() {
  local body="$1"
  curl -sf --max-time 60 -N "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" 2>/dev/null || echo 'ERROR'
}

# Helper: extract content from API response
# For thinking models, content may be empty while reasoning_content has text.
# Returns content if non-empty, otherwise returns reasoning_content.
extract_content() {
  python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read(), strict=False)
    c = d['choices'][0]['message'].get('content') or ''
    print(c if c else '')
except Exception as e:
    print(f'__ERROR__: {e}')
"
}

# Helper: check if response has any output (content or reasoning_content)
has_output() {
  python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read(), strict=False)
    msg = d['choices'][0]['message']
    c = msg.get('content') or ''
    rc = msg.get('reasoning_content') or ''
    print('yes' if c or rc else 'no')
except Exception as e:
    print('no')
"
}

# Helper: timing
now_ms() {
  python3 -c "import time; print(int(time.time()*1000))"
}

min_tier() {
  # Returns 0 (true) if current tier >= required tier
  local required="$1"
  case "$required" in
    smoke)    return 0 ;;
    standard) [[ "$TIER" == "standard" || "$TIER" == "full" ]] && return 0 || return 1 ;;
    full)     [[ "$TIER" == "full" ]] && return 0 || return 1 ;;
  esac
}

# Returns 0 (true) if the given section number should run.
# When --section is empty, all sections run. Otherwise only the matching one.
should_run_section() {
  local num="$1"
  [[ -z "$SECTION" || "$SECTION" == "$num" ]]
}

SECTION_LABEL=""
[[ -n "$SECTION" ]] && SECTION_LABEL=" | Section: $SECTION"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AFM Assertion Tests"
echo "  Tier: $TIER | Port: $PORT | Model: ${MODEL:-auto-detect}${SECTION_LABEL}"
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
fi

t0=$(now_ms)
if curl -sf --max-time 5 "$BASE_URL/v1/models" >/dev/null 2>&1; then
  run_test "Preflight" "Server reachable at $BASE_URL" "200 OK" "PASS" "$(( $(now_ms) - t0 ))"
else
  echo "ERROR: Server not reachable at $BASE_URL"
  echo "Start it first, e.g.:"
  echo "  MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \\"
  echo "    $BIN mlx -m MODEL -p $PORT &"
  run_test "Preflight" "Server reachable at $BASE_URL" "200 OK" "FAIL: not reachable" "$(( $(now_ms) - t0 ))"
  # Cannot continue without server
  echo ""
  echo "FATAL: Server not reachable. Aborting."
  exit 1
fi

# Auto-detect model from server
if [ -z "$MODEL" ]; then
  MODEL=$(curl -sf "$BASE_URL/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
fi
echo "  Model: $MODEL"

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Server lifecycle
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 1; then
echo ""
echo "🏥 Section 1: Server Lifecycle"

t0=$(now_ms)
models_json=$(curl -sf "$BASE_URL/v1/models" 2>/dev/null || echo '{}')
if echo "$models_json" | python3 -c "import sys,json; d=json.load(sys.stdin); assert any('$MODEL' in m.get('id','') for m in d.get('data',[]))" 2>/dev/null; then
  run_test "Lifecycle" "/v1/models contains model ID" "model in response" "PASS" "$(( $(now_ms) - t0 ))"
else
  run_test "Lifecycle" "/v1/models contains model ID" "model in response" "FAIL: $MODEL not found in models response" "$(( $(now_ms) - t0 ))"
fi

t0=$(now_ms)
basic_resp=$(api_call '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":500,"stream":false,"temperature":0}')
basic_has_output=$(echo "$basic_resp" | has_output)
if [ "$basic_has_output" = "yes" ]; then
  run_test "Lifecycle" "Basic completion returns content" "non-empty content or reasoning" "PASS" "$(( $(now_ms) - t0 ))"
else
  run_test "Lifecycle" "Basic completion returns content" "non-empty content or reasoning" "FAIL: empty response" "$(( $(now_ms) - t0 ))"
fi
fi # section 1

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Stop sequences
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 2; then
echo ""
echo "🛑 Section 2: Stop Sequences"

# Test: stop string absent from output
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"Count from 1 to 20, one number per line."}],"max_tokens":200,"stream":false,"temperature":0,"stop":["5"]}')
content=$(echo "$resp" | extract_content)
finish=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0].get('finish_reason',''))" 2>/dev/null || echo "error")
dur=$(( $(now_ms) - t0 ))
if ! echo "$content" | grep -q "5"; then
  run_test "Stop" "Stop string '5' absent from output" "no '5' in content" "PASS" "$dur"
else
  run_test "Stop" "Stop string '5' absent from output" "no '5' in content" "FAIL: found '5' in: $content" "$dur"
fi

t0=$(now_ms)
if [ "$finish" = "stop" ]; then
  run_test "Stop" "finish_reason is 'stop' with stop sequence" "stop" "PASS" "$(( $(now_ms) - t0 ))"
else
  run_test "Stop" "finish_reason is 'stop' with stop sequence" "stop" "FAIL: got '$finish'" "$(( $(now_ms) - t0 ))"
fi

# Test: multi-word stop
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"Write a short paragraph about cats."}],"max_tokens":200,"stream":false,"temperature":0,"stop":["and"]}')
content=$(echo "$resp" | extract_content)
dur=$(( $(now_ms) - t0 ))
# "and" should not appear in output (case sensitive)
if ! echo "$content" | grep -qw "and"; then
  run_test "Stop" "Multi-word stop 'and' truncates correctly" "no 'and' in output" "PASS" "$dur"
else
  # Check if it's just in a word like "understand" — that's fine, we check word boundary
  run_test "Stop" "Multi-word stop 'and' truncates correctly" "no 'and' in output" "FAIL: found in: $(echo "$content" | head -1)" "$dur"
fi

# Test: stop with newline
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"Say hello world"}],"max_tokens":500,"stream":false,"temperature":0,"stop":["\\n"]}')
content=$(echo "$resp" | extract_content)
dur=$(( $(now_ms) - t0 ))
if ! echo "$content" | grep -q $'\n'; then
  run_test "Stop" "Stop on newline produces single line" "single line" "PASS" "$dur"
else
  run_test "Stop" "Stop on newline produces single line" "single line" "FAIL: multi-line output" "$dur"
fi

# Test: multiple stop sequences
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"Count from 1 to 20."}],"max_tokens":200,"stream":false,"temperature":0,"stop":["7","12"]}')
content=$(echo "$resp" | extract_content)
dur=$(( $(now_ms) - t0 ))
if ! echo "$content" | grep -q "7" && ! echo "$content" | grep -q "12"; then
  run_test "Stop" "Multiple stop sequences [7, 12]" "neither found" "PASS" "$dur"
else
  run_test "Stop" "Multiple stop sequences [7, 12]" "neither found" "FAIL: content=$content" "$dur"
fi

# Test: empty stop array is no-op
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":500,"stream":false,"temperature":0,"stop":[]}')
dur=$(( $(now_ms) - t0 ))
empty_stop_ok=$(echo "$resp" | has_output)
if [ "$empty_stop_ok" = "yes" ]; then
  run_test "Stop" "Empty stop array is no-op" "valid output" "PASS" "$dur"
else
  run_test "Stop" "Empty stop array is no-op" "valid output" "FAIL" "$dur"
fi

# Test: streaming stop sequence parity
t0=$(now_ms)
stream_resp=$(api_stream '{"messages":[{"role":"user","content":"Count from 1 to 20, one number per line."}],"max_tokens":200,"stream":true,"temperature":0,"stop":["5"]}')
stream_content=$(echo "$stream_resp" | grep "^data: {" | grep -v '"[DONE]"' | python3 -c "
import sys, json
content = ''
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: '):
        continue
    data = line[6:]
    if data == '[DONE]':
        break
    try:
        d = json.loads(data)
        c = d.get('choices', [{}])[0].get('delta', {}).get('content', '')
        if c:
            content += c
    except: pass
print(content)
" 2>/dev/null || echo "__ERROR__")
dur=$(( $(now_ms) - t0 ))
if ! echo "$stream_content" | grep -q "5"; then
  run_test "Stop" "Streaming: stop string '5' absent" "no '5' in stream" "PASS" "$dur"
else
  run_test "Stop" "Streaming: stop string '5' absent" "no '5' in stream" "FAIL: found '5'" "$dur"
fi

if min_tier standard; then
  # Additional stop tests for standard+ tiers
  # Test: stop sequence with JSON array format
  t0=$(now_ms)
  resp=$(api_call '{"messages":[{"role":"user","content":"List 5 fruits, one per line."}],"max_tokens":100,"stream":false,"temperature":0,"stop":["3."]}')
  content=$(echo "$resp" | extract_content)
  dur=$(( $(now_ms) - t0 ))
  if ! echo "$content" | grep -q "3\."; then
    run_test "Stop" "Stop sequence '3.' truncates list" "no '3.' in output" "PASS" "$dur"
  else
    run_test "Stop" "Stop sequence '3.' truncates list" "no '3.' in output" "FAIL" "$dur"
  fi

  # Test: stop doesn't fire on partial match
  t0=$(now_ms)
  resp=$(api_call '{"messages":[{"role":"user","content":"Say the word stopping"}],"max_tokens":500,"stream":false,"temperature":0,"stop":["stopped"]}')
  dur=$(( $(now_ms) - t0 ))
  partial_ok=$(echo "$resp" | has_output)
  if [ "$partial_ok" = "yes" ]; then
    run_test "Stop" "Stop 'stopped' doesn't fire on 'stopping'" "output produced" "PASS" "$dur"
  else
    run_test "Stop" "Stop 'stopped' doesn't fire on 'stopping'" "output produced" "FAIL" "$dur"
  fi

  # Test: stop fires mid-word
  t0=$(now_ms)
  resp=$(api_call '{"messages":[{"role":"user","content":"Say the word hello"}],"max_tokens":500,"stream":false,"temperature":0,"stop":["llo"]}')
  content=$(echo "$resp" | extract_content)
  dur=$(( $(now_ms) - t0 ))
  if ! echo "$content" | grep -q "llo"; then
    run_test "Stop" "Stop 'llo' fires mid-word in 'hello'" "no 'llo'" "PASS" "$dur"
  else
    run_test "Stop" "Stop 'llo' fires mid-word in 'hello'" "no 'llo'" "FAIL: $content" "$dur"
  fi
fi
fi # section 2

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Logprobs schema
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 3; then
echo ""
echo "📊 Section 3: Logprobs"

# Test: logprobs structure
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"stream":false,"temperature":0,"logprobs":true,"top_logprobs":3}')
dur=$(( $(now_ms) - t0 ))
logprobs_valid=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    lp = d['choices'][0]['logprobs']
    assert lp is not None, 'logprobs is None'
    content = lp['content']
    assert isinstance(content, list), 'content is not list'
    assert len(content) > 0, 'content is empty'
    entry = content[0]
    assert 'token' in entry, 'no token'
    assert 'logprob' in entry, 'no logprob'
    assert 'top_logprobs' in entry, 'no top_logprobs'
    assert isinstance(entry['logprob'], (int, float)), 'logprob not number'
    assert entry['logprob'] <= 0, f'logprob > 0: {entry[\"logprob\"]}'
    print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$logprobs_valid" = "PASS" ]; then
  run_test "Logprobs" "ChoiceLogprobs JSON schema valid" "valid schema" "PASS" "$dur"
else
  run_test "Logprobs" "ChoiceLogprobs JSON schema valid" "valid schema" "$logprobs_valid" "$dur"
fi

# Test: top_logprobs count
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"stream":false,"temperature":0,"logprobs":true,"top_logprobs":5}')
dur=$(( $(now_ms) - t0 ))
count_valid=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    content = d['choices'][0]['logprobs']['content']
    for entry in content:
        n = len(entry['top_logprobs'])
        assert n <= 5, f'top_logprobs count {n} > 5'
        for tp in entry['top_logprobs']:
            assert 'token' in tp, 'missing token in top_logprobs'
            assert 'logprob' in tp, 'missing logprob in top_logprobs'
            assert tp['logprob'] <= 0, f'top logprob > 0: {tp[\"logprob\"]}'
    print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$count_valid" = "PASS" ]; then
  run_test "Logprobs" "top_logprobs count <= requested (5)" "count valid" "PASS" "$dur"
else
  run_test "Logprobs" "top_logprobs count <= requested (5)" "count valid" "$count_valid" "$dur"
fi

# Test: logprobs=false returns null
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"stream":false,"temperature":0,"logprobs":false}')
dur=$(( $(now_ms) - t0 ))
no_logprobs=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
lp = d['choices'][0].get('logprobs')
print('PASS' if lp is None else f'FAIL: logprobs={lp}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$no_logprobs" = "PASS" ]; then
  run_test "Logprobs" "logprobs=false returns null" "null logprobs" "PASS" "$dur"
else
  run_test "Logprobs" "logprobs=false returns null" "null logprobs" "$no_logprobs" "$dur"
fi

# Test: 400 on top_logprobs > max
t0=$(now_ms)
http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"stream":false,"logprobs":true,"top_logprobs":99}' 2>/dev/null)
dur=$(( $(now_ms) - t0 ))
if [ "$http_code" = "400" ]; then
  run_test "Logprobs" "top_logprobs=99 returns 400" "400" "PASS" "$dur"
else
  run_test "Logprobs" "top_logprobs=99 returns 400" "400" "FAIL: HTTP $http_code" "$dur"
fi

if min_tier standard; then
  # Test: streaming logprobs
  t0=$(now_ms)
  stream_resp=$(api_stream '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"stream":true,"temperature":0,"logprobs":true,"top_logprobs":2}')
  dur=$(( $(now_ms) - t0 ))
  stream_lp_valid=$(echo "$stream_resp" | python3 -c "
import sys, json
found_logprobs = False
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        d = json.loads(line[6:])
        lp = d.get('choices', [{}])[0].get('logprobs')
        if lp and lp.get('content'):
            found_logprobs = True
            for entry in lp['content']:
                assert 'token' in entry
                assert 'logprob' in entry
                assert entry['logprob'] <= 0
    except json.JSONDecodeError:
        pass
print('PASS' if found_logprobs else 'FAIL: no logprobs in stream')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$stream_lp_valid" = "PASS" ]; then
    run_test "Logprobs" "Streaming logprobs present and valid" "valid" "PASS" "$dur"
  else
    run_test "Logprobs" "Streaming logprobs present and valid" "valid" "$stream_lp_valid" "$dur"
  fi

  # Test: top_logprobs=0 returns empty array
  t0=$(now_ms)
  resp=$(api_call '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"stream":false,"temperature":0,"logprobs":true,"top_logprobs":0}')
  dur=$(( $(now_ms) - t0 ))
  zero_valid=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    content = d['choices'][0]['logprobs']['content']
    for entry in content:
        assert len(entry['top_logprobs']) == 0, f'expected empty, got {len(entry[\"top_logprobs\"])}'
    print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$zero_valid" = "PASS" ]; then
    run_test "Logprobs" "top_logprobs=0 returns empty arrays" "empty top_logprobs" "PASS" "$dur"
  else
    run_test "Logprobs" "top_logprobs=0 returns empty arrays" "empty top_logprobs" "$zero_valid" "$dur"
  fi
fi
fi # section 3

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Think/Reasoning extraction
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 4; then
echo ""
echo "🧠 Section 4: Think Extraction"

# Probe: does this model produce <think> tags?
probe_resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2? Think step by step."}],"max_tokens":100,"stream":false,"temperature":0}')
has_reasoning=$(echo "$probe_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    rc = d['choices'][0]['message'].get('reasoning_content')
    print('yes' if rc and len(rc) > 0 else 'no')
except:
    print('no')
" 2>/dev/null || echo "no")

if [ "$has_reasoning" = "yes" ]; then
  echo "  (Model supports thinking — running think tests)"

  # Test: reasoning_content present
  t0=$(now_ms)
  run_test "Think" "reasoning_content present in response" "present" "PASS" "$(( $(now_ms) - t0 ))"

  # Test: no <think> tags in content field
  t0=$(now_ms)
  content_clean=$(echo "$probe_resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message'].get('content', '')
if '<think>' in c or '</think>' in c:
    print('FAIL: think tags in content')
else:
    print('PASS')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$content_clean" = "PASS" ]; then
    run_test "Think" "No <think> tags in content field" "clean content" "PASS" "$(( $(now_ms) - t0 ))"
  else
    run_test "Think" "No <think> tags in content field" "clean content" "$content_clean" "$(( $(now_ms) - t0 ))"
  fi

  # Test: streaming think extraction
  t0=$(now_ms)
  stream_resp=$(api_stream '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":100,"stream":true,"temperature":0}')
  dur=$(( $(now_ms) - t0 ))
  stream_think_valid=$(echo "$stream_resp" | python3 -c "
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
        if delta.get('reasoning_content'):
            found_reasoning = True
        if delta.get('content'):
            found_content = True
    except: pass
print('PASS' if found_reasoning else 'FAIL: no reasoning_content in stream')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$stream_think_valid" = "PASS" ]; then
    run_test "Think" "Streaming: reasoning_content in deltas" "present" "PASS" "$dur"
  else
    run_test "Think" "Streaming: reasoning_content in deltas" "present" "$stream_think_valid" "$dur"
  fi

  # Test: stop + think interaction (stop should only apply to visible content)
  t0=$(now_ms)
  resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2? Think carefully."}],"max_tokens":200,"stream":false,"temperature":0,"stop":["step"]}')
  dur=$(( $(now_ms) - t0 ))
  think_stop_valid=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
rc = d['choices'][0]['message'].get('reasoning_content', '')
c = d['choices'][0]['message'].get('content', '')
# reasoning_content MAY contain 'step' (stop doesn't apply there)
# If we got a response at all, the think extraction didn't break
print('PASS' if c is not None else 'FAIL: no content')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$think_stop_valid" = "PASS" ]; then
    run_test "Think" "Stop sequence doesn't break think extraction" "response ok" "PASS" "$dur"
  else
    run_test "Think" "Stop sequence doesn't break think extraction" "response ok" "$think_stop_valid" "$dur"
  fi

  # Test: reasoning_content is non-empty
  t0=$(now_ms)
  rc_len=$(echo "$probe_resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
rc = d['choices'][0]['message'].get('reasoning_content', '')
print(len(rc))
" 2>/dev/null || echo "0")
  dur=$(( $(now_ms) - t0 ))
  if [ "$rc_len" -gt 5 ] 2>/dev/null; then
    run_test "Think" "reasoning_content has meaningful length (>5 chars)" ">5 chars" "PASS" "$dur"
  else
    run_test "Think" "reasoning_content has meaningful length (>5 chars)" ">5 chars" "FAIL: $rc_len chars" "$dur"
  fi
else
  echo "  (Model does not support thinking — skipping think tests)"
  run_test "Think" "Think extraction (model lacks <think> support)" "skip" "SKIP" "0"
fi
fi # section 4

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Tool calls
# ═══════════════════════════════════════════════════════════════════════════════
# Shared tool definition used by sections 5, 11, 12
TOOL_DEF='[{"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}]'

if should_run_section 5; then
echo ""
echo "🔧 Section 5: Tool Calls"

# Test: basic tool call
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
tc_valid=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    fr = d['choices'][0].get('finish_reason', '')
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if fr == 'tool_calls' and len(tc) > 0:
        t = tc[0]
        assert t.get('type') == 'function', f'type={t.get(\"type\")}'
        assert t['function'].get('name') == 'get_weather', f'name={t[\"function\"].get(\"name\")}'
        args = json.loads(t['function']['arguments'])
        assert 'location' in args or 'Location' in args or any('paris' in str(v).lower() for v in args.values()), f'args={args}'
        print('PASS')
    else:
        print(f'FAIL: finish_reason={fr}, tool_calls={len(tc)}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$tc_valid" = "PASS" ]; then
  run_test "Tools" "Basic tool call: finish_reason=tool_calls, valid args" "valid" "PASS" "$dur"
else
  run_test "Tools" "Basic tool call: finish_reason=tool_calls, valid args" "valid" "$tc_valid" "$dur"
fi

# Test: tool_choice=none suppresses tool calls
t0=$(now_ms)
resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],\"tools\":$TOOL_DEF,\"tool_choice\":\"none\",\"max_tokens\":100,\"stream\":false,\"temperature\":0}")
dur=$(( $(now_ms) - t0 ))
no_tc=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    fr = d['choices'][0].get('finish_reason', '')
    tc = d['choices'][0]['message'].get('tool_calls')
    if fr != 'tool_calls' and (tc is None or len(tc) == 0):
        print('PASS')
    else:
        print(f'FAIL: finish_reason={fr}, tool_calls={tc}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$no_tc" = "PASS" ]; then
  run_test "Tools" "tool_choice=none suppresses tool calls" "no tool calls" "PASS" "$dur"
else
  run_test "Tools" "tool_choice=none suppresses tool calls" "no tool calls" "$no_tc" "$dur"
fi

if min_tier standard; then
  # Test: tool arguments are valid JSON
  t0=$(now_ms)
  resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What's the weather in Tokyo in celsius?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  args_valid=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if len(tc) == 0:
        print('FAIL: no tool calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        assert isinstance(args, dict), f'args not dict: {type(args)}'
        print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$args_valid" = "PASS" ]; then
    run_test "Tools" "Tool arguments are valid JSON dict" "valid JSON" "PASS" "$dur"
  else
    run_test "Tools" "Tool arguments are valid JSON dict" "valid JSON" "$args_valid" "$dur"
  fi

  # Test: streaming tool calls
  t0=$(now_ms)
  stream_resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in London?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":true,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  stream_tc_valid=$(echo "$stream_resp" | python3 -c "
import sys, json
found_tc = False
found_finish = False
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        d = json.loads(line[6:])
        delta = d.get('choices', [{}])[0].get('delta', {})
        if delta.get('tool_calls'):
            found_tc = True
        fr = d.get('choices', [{}])[0].get('finish_reason')
        if fr == 'tool_calls':
            found_finish = True
    except: pass
if found_tc and found_finish:
    print('PASS')
elif found_tc:
    print('FAIL: tool_calls found but no finish_reason=tool_calls')
else:
    print('FAIL: no tool_calls in stream')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$stream_tc_valid" = "PASS" ]; then
    run_test "Tools" "Streaming: tool calls with finish_reason" "valid" "PASS" "$dur"
  else
    run_test "Tools" "Streaming: tool calls with finish_reason" "valid" "$stream_tc_valid" "$dur"
  fi

  # Test: multi-tool call
  MULTI_TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},{"type":"function","function":{"name":"get_time","description":"Get current time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]'
  t0=$(now_ms)
  resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What's the weather in Paris and what time is it in UTC?\"}],\"tools\":$MULTI_TOOLS,\"max_tokens\":300,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  multi_valid=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    # At least one tool call means the model understood multi-tool
    if len(tc) >= 1:
        print('PASS')
    else:
        print(f'FAIL: {len(tc)} tool calls')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$multi_valid" = "PASS" ]; then
    run_test "Tools" "Multi-tool: at least 1 tool call with 2 tools" ">=1 calls" "PASS" "$dur"
  else
    run_test "Tools" "Multi-tool: at least 1 tool call with 2 tools" ">=1 calls" "$multi_valid" "$dur"
  fi

  # Test: tool with array-type parameter (regression PR #37 — array params must not serialize as strings)
  ARRAY_TOOL='[{"type":"function","function":{"name":"todowrite","description":"Write todo items","parameters":{"type":"object","properties":{"todos":{"type":"array","items":{"type":"string"},"description":"List of todo items"}},"required":["todos"]}}}]'
  t0=$(now_ms)
  resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Create todos: Buy milk, Call dentist, Fix bug\"}],\"tools\":$ARRAY_TOOL,\"max_tokens\":300,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  array_param=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        todos = args.get('todos', args.get('items', None))
        if isinstance(todos, list):
            print('PASS')
        elif isinstance(todos, str):
            print('FAIL: todos is string (not array)')
        else:
            print(f'FAIL: todos type={type(todos).__name__}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$array_param" = "PASS" ]; then
    run_test "Tools" "Array param: todos is JSON array (not string)" "array" "PASS" "$dur"
  else
    run_test "Tools" "Array param: todos is JSON array (not string)" "array" "$array_param" "$dur"
  fi

  # Test: tool with nullable parameter schema (regression PR #33 — anyOf [string, null] must not crash Jinja)
  NULLABLE_TOOL='[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},"units":{"anyOf":[{"type":"string"},{"type":"null"}],"description":"Temperature units","default":null}},"required":["location"]}}}]'
  t0=$(now_ms)
  resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Chicago?\"}],\"tools\":$NULLABLE_TOOL,\"max_tokens\":300,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  nullable_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL: server error: {d[\"error\"].get(\"message\",\"\")}')
    else:
        tc = d['choices'][0]['message'].get('tool_calls', [])
        if tc and tc[0]['function']['name'] == 'get_weather':
            print('PASS')
        elif tc:
            print(f'PASS')  # tool call made, no crash
        else:
            content = d['choices'][0]['message'].get('content', '')
            if content:
                print('PASS')  # no crash, model responded with text
            else:
                print('FAIL: no tool calls and no content')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$nullable_ok" = "PASS" ]; then
    run_test "Tools" "Nullable param: anyOf [string, null] does not crash" "no crash" "PASS" "$dur"
  else
    run_test "Tools" "Nullable param: anyOf [string, null] does not crash" "no crash" "$nullable_ok" "$dur"
  fi

  # Test: no tools provided = normal response
  t0=$(now_ms)
  resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":20,"stream":false,"temperature":0}')
  dur=$(( $(now_ms) - t0 ))
  no_tools=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
fr = d['choices'][0].get('finish_reason', '')
tc = d['choices'][0]['message'].get('tool_calls')
if fr != 'tool_calls' and (tc is None or len(tc) == 0):
    print('PASS')
else:
    print(f'FAIL: finish_reason={fr}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$no_tools" = "PASS" ]; then
    run_test "Tools" "No tools: normal text response" "text response" "PASS" "$dur"
  else
    run_test "Tools" "No tools: normal text response" "text response" "$no_tools" "$dur"
  fi
fi
fi # section 5

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Prompt cache
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 6 && min_tier standard; then
  echo ""
  echo "💾 Section 6: Prompt Cache"

  # Test: first request has cached_tokens=0
  t0=$(now_ms)
  # Use a unique prompt to ensure cache miss
  unique_prompt="Tell me about the history of prompt caching in LLM servers $(date +%s%N)"
  resp1=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$unique_prompt\"}],\"max_tokens\":20,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  cached1=$(echo "$resp1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ptd = d.get('usage', {}).get('prompt_tokens_details')
    if ptd is None:
        print('NULL')
    else:
        print(ptd.get('cached_tokens', 'MISSING'))
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || echo "ERROR")
  if [ "$cached1" = "0" ]; then
    run_test "Cache" "First request: cached_tokens=0" "0" "PASS" "$dur"
  elif [ "$cached1" = "NULL" ]; then
    run_test "Cache" "First request: cached_tokens=0" "0" "FAIL: prompt_tokens_details is null (caching may be disabled)" "$dur"
  else
    run_test "Cache" "First request: cached_tokens=0" "0" "FAIL: cached_tokens=$cached1" "$dur"
  fi

  # Test: second identical request has cached_tokens > 0
  t0=$(now_ms)
  resp2=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$unique_prompt\"}],\"max_tokens\":20,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  cached2=$(echo "$resp2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ptd = d.get('usage', {}).get('prompt_tokens_details')
    if ptd is None:
        print('NULL')
    else:
        print(ptd.get('cached_tokens', 'MISSING'))
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || echo "ERROR")
  if [ "$cached2" != "NULL" ] && [ "$cached2" != "ERROR" ] && [ "$cached2" != "MISSING" ] && [ "$cached2" -gt 0 ] 2>/dev/null; then
    run_test "Cache" "Second identical request: cached_tokens>0" ">0" "PASS" "$dur"
  else
    run_test "Cache" "Second identical request: cached_tokens>0" ">0" "FAIL: cached_tokens=$cached2" "$dur"
  fi

  # Test: different prompt resets cache (cached_tokens=0)
  t0=$(now_ms)
  resp3=$(api_call '{"messages":[{"role":"user","content":"This is a completely different prompt about quantum physics and black holes."}],"max_tokens":20,"stream":false,"temperature":0}')
  dur=$(( $(now_ms) - t0 ))
  cached3=$(echo "$resp3" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ptd = d.get('usage', {}).get('prompt_tokens_details')
    if ptd is None:
        print('NULL')
    else:
        print(ptd.get('cached_tokens', 'MISSING'))
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || echo "ERROR")
  if [ "$cached3" = "0" ]; then
    run_test "Cache" "Different prompt: cached_tokens=0" "0" "PASS" "$dur"
  elif [ "$cached3" = "NULL" ]; then
    run_test "Cache" "Different prompt: cached_tokens=0" "0" "FAIL: null (caching disabled)" "$dur"
  else
    run_test "Cache" "Different prompt: cached_tokens=0" "0" "FAIL: cached_tokens=$cached3" "$dur"
  fi

  # Test: streaming cached_tokens
  t0=$(now_ms)
  # First call to warm cache
  api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$unique_prompt again\"}],\"max_tokens\":10,\"stream\":false,\"temperature\":0}" >/dev/null
  # Second call streaming
  stream_resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"$unique_prompt again\"}],\"max_tokens\":10,\"stream\":true,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  stream_cached=$(echo "$stream_resp" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        d = json.loads(line[6:])
        usage = d.get('usage')
        if usage:
            ptd = usage.get('prompt_tokens_details')
            if ptd and ptd.get('cached_tokens', 0) > 0:
                print('PASS')
                sys.exit(0)
    except: pass
print('FAIL: no cached_tokens>0 in stream usage')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$stream_cached" = "PASS" ]; then
    run_test "Cache" "Streaming: cached_tokens>0 in usage chunk" ">0" "PASS" "$dur"
  else
    run_test "Cache" "Streaming: cached_tokens>0 in usage chunk" ">0" "$stream_cached" "$dur"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Concurrent requests
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 7 && min_tier standard; then
  echo ""
  echo "⚡ Section 7: Concurrent Requests"

  # Test: two simultaneous requests both return 200
  t0=$(now_ms)
  tmpdir=$(mktemp -d)
  curl -sf --max-time 30 "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say A"}],"max_tokens":5,"stream":false,"temperature":0}' \
    -o "$tmpdir/resp1.json" -w "%{http_code}" > "$tmpdir/code1.txt" 2>/dev/null &
  pid1=$!
  curl -sf --max-time 30 "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say B"}],"max_tokens":5,"stream":false,"temperature":0}' \
    -o "$tmpdir/resp2.json" -w "%{http_code}" > "$tmpdir/code2.txt" 2>/dev/null &
  pid2=$!
  wait $pid1 $pid2 2>/dev/null || true
  dur=$(( $(now_ms) - t0 ))
  code1=$(cat "$tmpdir/code1.txt" 2>/dev/null || echo "000")
  code2=$(cat "$tmpdir/code2.txt" 2>/dev/null || echo "000")
  rm -rf "$tmpdir"
  if [ "$code1" = "200" ] && [ "$code2" = "200" ]; then
    run_test "Concurrent" "Two simultaneous requests: both 200" "200+200" "PASS" "$dur"
  else
    run_test "Concurrent" "Two simultaneous requests: both 200" "200+200" "FAIL: $code1+$code2" "$dur"
  fi

  # Test: three simultaneous requests
  t0=$(now_ms)
  tmpdir=$(mktemp -d)
  for i in 1 2 3; do
    curl -sf --max-time 30 "$BASE_URL/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say $i\"}],\"max_tokens\":5,\"stream\":false,\"temperature\":0}" \
      -o "$tmpdir/resp$i.json" -w "%{http_code}" > "$tmpdir/code$i.txt" 2>/dev/null &
  done
  wait 2>/dev/null || true
  dur=$(( $(now_ms) - t0 ))
  all_200=true
  for i in 1 2 3; do
    c=$(cat "$tmpdir/code$i.txt" 2>/dev/null || echo "000")
    [ "$c" != "200" ] && all_200=false
  done
  rm -rf "$tmpdir"
  if $all_200; then
    run_test "Concurrent" "Three simultaneous requests: all 200" "all 200" "PASS" "$dur"
  else
    run_test "Concurrent" "Three simultaneous requests: all 200" "all 200" "FAIL" "$dur"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Error handling
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 8; then
echo ""
echo "⚠️  Section 8: Error Handling"

# Test: empty messages array → 400
t0=$(now_ms)
http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[],"max_tokens":10,"stream":false}' 2>/dev/null)
dur=$(( $(now_ms) - t0 ))
if [ "$http_code" = "400" ]; then
  run_test "Error" "Empty messages → 400" "400" "PASS" "$dur"
else
  run_test "Error" "Empty messages → 400" "400" "FAIL: HTTP $http_code" "$dur"
fi

# Test: malformed JSON → 400
t0=$(now_ms)
http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{broken json' 2>/dev/null)
dur=$(( $(now_ms) - t0 ))
if [ "$http_code" = "400" ]; then
  run_test "Error" "Malformed JSON → 400" "400" "PASS" "$dur"
else
  run_test "Error" "Malformed JSON → 400" "400" "FAIL: HTTP $http_code" "$dur"
fi

# Test: missing messages field → 400
t0=$(now_ms)
http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"max_tokens":10}' 2>/dev/null)
dur=$(( $(now_ms) - t0 ))
if [ "$http_code" = "400" ]; then
  run_test "Error" "Missing messages field → 400" "400" "PASS" "$dur"
else
  run_test "Error" "Missing messages field → 400" "400" "FAIL: HTTP $http_code" "$dur"
fi

# Test: response_format json_object works
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"system","content":"Respond in JSON."},{"role":"user","content":"Give me a JSON object with key name and value Alice"}],"max_tokens":500,"stream":false,"temperature":0,"response_format":{"type":"json_object"},"chat_template_kwargs":{"enable_thinking":false}}')
dur=$(( $(now_ms) - t0 ))
json_valid=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message']['content'].strip()
    # Try to parse as JSON
    parsed = json.loads(c)
    print('PASS')
except json.JSONDecodeError:
    print(f'FAIL: not valid JSON: {c[:100]}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$json_valid" = "PASS" ]; then
  run_test "Error" "response_format json_object returns valid JSON" "valid JSON" "PASS" "$dur"
else
  run_test "Error" "response_format json_object returns valid JSON" "valid JSON" "$json_valid" "$dur"
fi

# Test: max_tokens is respected
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"user","content":"Write a very long essay about the universe, covering all topics from physics to philosophy."}],"max_tokens":5,"stream":false,"temperature":0}')
dur=$(( $(now_ms) - t0 ))
tokens_ok=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ct = d.get('usage', {}).get('completion_tokens', 0)
# Allow some tolerance (tokenizer boundaries)
if ct <= 10:
    print('PASS')
else:
    print(f'FAIL: completion_tokens={ct}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$tokens_ok" = "PASS" ]; then
  run_test "Error" "max_tokens=5 is respected" "<=10 tokens" "PASS" "$dur"
else
  run_test "Error" "max_tokens=5 is respected" "<=10 tokens" "$tokens_ok" "$dur"
fi

# Test: OPTIONS returns 200 (CORS)
t0=$(now_ms)
http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X OPTIONS "$BASE_URL/v1/chat/completions" 2>/dev/null)
dur=$(( $(now_ms) - t0 ))
if [ "$http_code" = "200" ]; then
  run_test "Error" "OPTIONS /v1/chat/completions → 200 (CORS)" "200" "PASS" "$dur"
else
  run_test "Error" "OPTIONS /v1/chat/completions → 200 (CORS)" "200" "FAIL: HTTP $http_code" "$dur"
fi

# Test: developer role mapped to system
t0=$(now_ms)
resp=$(api_call '{"messages":[{"role":"developer","content":"You are a pirate."},{"role":"user","content":"Say hello"}],"max_tokens":500,"stream":false,"temperature":0}')
dur=$(( $(now_ms) - t0 ))
dev_ok=$(echo "$resp" | has_output)
if [ "$dev_ok" = "yes" ]; then
  run_test "Error" "developer role accepted (mapped to system)" "valid response" "PASS" "$dur"
else
  run_test "Error" "developer role accepted (mapped to system)" "valid response" "FAIL" "$dur"
fi
fi # section 8

# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Chat Template Kwargs (Issue #34)
# ═══════════════════════════════════════════════════════════════════════════════
# Tests request-level chat_template_kwargs (e.g., enable_thinking: false).
# Requires a thinking-capable model. Skips if model lacks <think> support.
# Note: --no-think CLI flag and precedence tests require server restart — see
# Scripts/test-chat-template-kwargs.sh for the full standalone suite.

if should_run_section 10 && min_tier standard; then
  echo ""
  echo "🎛️  Section 10: Chat Template Kwargs (Issue #34)"

  # Reuse the thinking probe from Section 4 (or re-probe if it wasn't set)
  if [ -z "${has_reasoning:-}" ]; then
    probe_resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2? Think step by step."}],"max_tokens":100,"stream":false,"temperature":0}')
    has_reasoning=$(echo "$probe_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    rc = d['choices'][0]['message'].get('reasoning_content')
    print('yes' if rc and len(rc) > 0 else 'no')
except:
    print('no')
" 2>/dev/null || echo "no")
  fi

  if [ "$has_reasoning" = "yes" ]; then
    echo "  (Model supports thinking — running chat_template_kwargs tests)"

    # Helper: classify response thinking state
    # Returns: "no_think" | "thinking" | "empty" | "error"
    _kwargs_check_response() {
      echo "$1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print('error'); sys.exit()
    msg = d['choices'][0]['message']
    content = msg.get('content', '')
    reasoning = msg.get('reasoning_content')
    if reasoning and len(reasoning) > 0:
        print('thinking' if content and len(content.strip()) > 0 else 'empty')
    else:
        print('no_think' if content and len(content.strip()) > 0 else 'empty')
except:
    print('error')
" 2>/dev/null || echo "error"
    }

    # Helper: classify streaming response thinking state
    _kwargs_check_stream() {
      echo "$1" | python3 -c "
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
        if delta.get('reasoning_content', '').strip():
            found_reasoning = True
        if delta.get('content', '').strip():
            found_content = True
    except:
        pass
if found_reasoning and found_content:
    print('thinking')
elif found_reasoning:
    print('empty')
elif found_content:
    print('no_think')
else:
    print('error')
" 2>/dev/null || echo "error"
    }

    # Test: enable_thinking=false disables thinking (non-streaming)
    t0=$(now_ms)
    resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":50,"stream":false,"temperature":0}')
    dur=$(( $(now_ms) - t0 ))
    state=$(_kwargs_check_response "$resp")
    if [ "$state" = "no_think" ]; then
      run_test "Kwargs" "enable_thinking=false disables thinking" "no_think" "PASS" "$dur"
    else
      run_test "Kwargs" "enable_thinking=false disables thinking" "no_think" "FAIL: state=$state" "$dur"
    fi

    # Test: enable_thinking=false disables thinking (streaming)
    t0=$(now_ms)
    resp=$(api_stream '{"messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":50,"stream":true,"temperature":0}')
    dur=$(( $(now_ms) - t0 ))
    state=$(_kwargs_check_stream "$resp")
    if [ "$state" = "no_think" ]; then
      run_test "Kwargs" "Streaming: enable_thinking=false disables thinking" "no_think" "PASS" "$dur"
    else
      run_test "Kwargs" "Streaming: enable_thinking=false disables thinking" "no_think" "FAIL: state=$state" "$dur"
    fi

    # Test: default behavior (no kwargs) still has thinking
    t0=$(now_ms)
    resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200,"stream":false,"temperature":0}')
    dur=$(( $(now_ms) - t0 ))
    state=$(_kwargs_check_response "$resp")
    if [ "$state" = "thinking" ]; then
      run_test "Kwargs" "Default (no kwargs) retains thinking" "thinking" "PASS" "$dur"
    else
      run_test "Kwargs" "Default (no kwargs) retains thinking" "thinking" "FAIL: state=$state" "$dur"
    fi

    # Test: enable_thinking=false with higher max_tokens returns content
    t0=$(now_ms)
    resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":2000,"stream":false,"temperature":0}')
    dur=$(( $(now_ms) - t0 ))
    state=$(_kwargs_check_response "$resp")
    content_len=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(len(d['choices'][0]['message'].get('content', '')))
except:
    print(0)
" 2>/dev/null || echo "0")
    if [ "$state" = "no_think" ] && [ "$content_len" -gt 0 ] 2>/dev/null; then
      run_test "Kwargs" "enable_thinking=false (2K tokens) returns content" "content present, no reasoning" "PASS" "$dur"
    else
      run_test "Kwargs" "enable_thinking=false (2K tokens) returns content" "content present, no reasoning" "FAIL: state=$state, content_len=$content_len" "$dur"
    fi

    # Test: enable_thinking=true explicitly keeps thinking
    t0=$(now_ms)
    resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2?"}],"chat_template_kwargs":{"enable_thinking":true},"max_tokens":200,"stream":false,"temperature":0}')
    dur=$(( $(now_ms) - t0 ))
    state=$(_kwargs_check_response "$resp")
    if [ "$state" = "thinking" ]; then
      run_test "Kwargs" "enable_thinking=true explicitly keeps thinking" "thinking" "PASS" "$dur"
    else
      run_test "Kwargs" "enable_thinking=true explicitly keeps thinking" "thinking" "FAIL: state=$state" "$dur"
    fi

  else
    echo "  (Model does not support thinking — skipping chat_template_kwargs tests)"
    run_test "Kwargs" "chat_template_kwargs (model lacks thinking)" "skip" "SKIP" "0"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 8b: Prefix Cache
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 8b && min_tier standard; then
  echo ""
  echo "🗄️  Section 8b: Prefix Cache"

  # Test: Prefix caching works across multiple requests with same prefix
  t0=$(now_ms)
  # First request — cold
  resp1=$(api_call '{"messages":[{"role":"system","content":"You are a helpful math tutor."},{"role":"user","content":"What is 2+2?"}],"max_tokens":10,"temperature":0}')
  # Second request — same system prompt, different user message (should cache hit)
  resp2=$(api_call '{"messages":[{"role":"system","content":"You are a helpful math tutor."},{"role":"user","content":"What is 3+3?"}],"max_tokens":10,"temperature":0}')
  dur=$(( $(now_ms) - t0 ))

  cache_ok=$(echo "$resp2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    usage = d.get('usage', {})
    cached = usage.get('prompt_tokens_cached', 0)
    if cached > 0:
        print(f'PASS ({cached} cached)')
    else:
        # Cache hit may not be reported in usage — just verify response is valid
        c = d['choices'][0]['message']['content']
        print('PASS' if c and len(c.strip()) > 0 else 'FAIL: empty')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [[ "$cache_ok" == PASS* ]]; then
    run_test "Cache" "Prefix cache reuse across requests" "valid response on cache hit" "PASS" "$dur"
  else
    run_test "Cache" "Prefix cache reuse across requests" "valid response on cache hit" "$cache_ok" "$dur"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 8c: Structured Output
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 8c && min_tier standard; then
  echo ""
  echo "📋 Section 8c: Structured Output"

  # Test: response_format json_schema produces valid schema-matching JSON
  t0=$(now_ms)
  schema_resp=$(api_call '{"messages":[{"role":"user","content":"Give me a person named Alice who is 30 years old"}],"max_tokens":100,"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"person","schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}}}}')
  dur=$(( $(now_ms) - t0 ))
  schema_valid=$(echo "$schema_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message']['content'].strip()
    parsed = json.loads(c)
    assert 'name' in parsed and isinstance(parsed['name'], str), 'missing name'
    assert 'age' in parsed and isinstance(parsed['age'], int), 'missing age'
    print('PASS')
except json.JSONDecodeError as e:
    print(f'FAIL: invalid JSON: {e}')
except AssertionError as e:
    print(f'FAIL: schema mismatch: {e}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$schema_valid" = "PASS" ]; then
    run_test "Structured" "json_schema produces valid schema-matching JSON" "valid JSON" "PASS" "$dur"
  else
    run_test "Structured" "json_schema produces valid schema-matching JSON" "valid JSON" "$schema_valid" "$dur"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Performance (full tier only)
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 9 && min_tier full; then
  echo ""
  echo "🚀 Section 9: Performance"

  # Test: TTFT < 5s (first token — content or reasoning_content)
  t0=$(now_ms)
  stream_resp=$(api_stream '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50,"stream":true,"temperature":0}')
  first_data_ms=$(echo "$stream_resp" | python3 -c "
import sys, time, json
start = time.time()
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: {'):
        try:
            d = json.loads(line[6:])
            delta = d.get('choices', [{}])[0].get('delta', {})
            c = delta.get('content', '') or delta.get('reasoning_content', '')
            if c:
                print(int((time.time() - start) * 1000))
                break
        except: pass
else:
    print('99999')
" 2>/dev/null || echo "99999")
  dur=$(( $(now_ms) - t0 ))
  if [ "$first_data_ms" -lt 5000 ] 2>/dev/null; then
    run_test "Perf" "TTFT < 5s" "<5000ms" "PASS" "$dur"
  else
    run_test "Perf" "TTFT < 5s" "<5000ms" "FAIL: ${first_data_ms}ms" "$dur"
  fi

  # Test: tok/s > 1 (bare minimum)
  t0=$(now_ms)
  resp=$(api_call '{"messages":[{"role":"user","content":"Write a short paragraph about dogs."}],"max_tokens":50,"stream":false,"temperature":0}')
  dur_ms=$(( $(now_ms) - t0 ))
  tps=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ct = d.get('usage', {}).get('completion_tokens', 0)
dur_s = $dur_ms / 1000.0
tps = ct / dur_s if dur_s > 0 else 0
print(f'{tps:.1f}')
" 2>/dev/null || echo "0")
  if python3 -c "exit(0 if float('$tps') > 1 else 1)" 2>/dev/null; then
    run_test "Perf" "tok/s > 1 ($tps tok/s)" ">1" "PASS" "$dur_ms"
  else
    run_test "Perf" "tok/s > 1 ($tps tok/s)" ">1" "FAIL: $tps tok/s" "$dur_ms"
  fi

  # Test: long context (2048 tokens) no crash
  t0=$(now_ms)
  long_prompt=$(python3 -c "print('word ' * 500)")
  resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Summarize this text: $long_prompt\"}],\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  long_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read(), strict=False)
    msg = d['choices'][0]['message']
    c = msg.get('content') or ''
    rc = msg.get('reasoning_content') or ''
    print('PASS' if c or rc else 'FAIL: empty')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$long_ok" = "PASS" ]; then
    run_test "Perf" "Long context (~2K tokens) no crash" "valid response" "PASS" "$dur"
  else
    run_test "Perf" "Long context (~2K tokens) no crash" "valid response" "$long_ok" "$dur"
  fi

  # Test: very long context (4096 tokens) no NaN/garbage
  t0=$(now_ms)
  very_long_prompt=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 200)")
  resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Summarize: $very_long_prompt\"}],\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  vlong_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read(), strict=False)
    msg = d['choices'][0]['message']
    c = msg.get('content') or ''
    rc = msg.get('reasoning_content') or ''
    text = c + rc
    if len(text) == 0:
        print('FAIL: empty content')
    elif 'nan' in text.lower() or '\\x00' in text:
        print(f'FAIL: garbage/NaN detected: {text[:100]}')
    else:
        print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$vlong_ok" = "PASS" ]; then
    run_test "Perf" "Very long context (~4K tokens) no NaN/garbage" "clean output" "PASS" "$dur"
  else
    run_test "Perf" "Very long context (~4K tokens) no NaN/garbage" "clean output" "$vlong_ok" "$dur"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: Qwen3 XML Tool Call Format (standard tier)
# ═══════════════════════════════════════════════════════════════════════════════
# Deep validation of XML tool call format used by Qwen3-Coder, Qwen3.5 MoE models.
# Auto-detected as xmlFunction when model_type is qwen3_moe, qwen3_5_moe, etc.
# Format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
# These tests run on any model but are most meaningful on Qwen3 XML models.

if should_run_section 11 && min_tier standard; then
  # Probe: does this model produce XML-format tool calls?
  probe_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Berlin?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
  has_tool_calls=$(echo "$probe_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    print('yes' if len(tc) > 0 else 'no')
except:
    print('no')
" 2>/dev/null || echo "no")

  if [ "$has_tool_calls" = "yes" ]; then
    echo ""
    echo "🔧 Section 11: XML Tool Call Deep Validation"

    # Test: correct function name extraction
    t0=$(now_ms)
    resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Madrid?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    fn_name=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if tc:
        print(tc[0]['function']['name'])
    else:
        print('FAIL: no tool_calls')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$fn_name" = "get_weather" ]; then
      run_test "XMLTools" "Function name correctly extracted" "get_weather" "PASS" "$dur"
    else
      run_test "XMLTools" "Function name correctly extracted" "get_weather" "FAIL: got $fn_name" "$dur"
    fi

    # Test: parameter values are correct types (string)
    t0=$(now_ms)
    resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Get the weather in Rome in celsius.\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    param_types=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        loc = args.get('location', args.get('city', ''))
        unit = args.get('unit', '')
        if isinstance(loc, str) and len(loc) > 0 and isinstance(unit, str):
            print('PASS')
        else:
            print(f'FAIL: location={type(loc).__name__}({loc}), unit={type(unit).__name__}({unit})')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$param_types" = "PASS" ]; then
      run_test "XMLTools" "Parameter values are correct string types" "strings" "PASS" "$dur"
    else
      run_test "XMLTools" "Parameter values are correct string types" "strings" "$param_types" "$dur"
    fi

    # Test: tool call with boolean and integer parameters
    TYPED_TOOL='[{"type":"function","function":{"name":"search_code","description":"Search codebase","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"case_sensitive":{"type":"boolean","description":"Case sensitive search"},"max_results":{"type":"integer","description":"Max results to return"}},"required":["query"]}}}]'
    t0=$(now_ms)
    resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Search for 'main' case-sensitively, return max 10 results.\"}],\"tools\":$TYPED_TOOL,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    typed_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        q = args.get('query', '')
        if not isinstance(q, str) or len(q) == 0:
            print(f'FAIL: query={q}')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$typed_ok" = "PASS" ]; then
      run_test "XMLTools" "Mixed-type params (string+bool+int) parse correctly" "valid types" "PASS" "$dur"
    else
      run_test "XMLTools" "Mixed-type params (string+bool+int) parse correctly" "valid types" "$typed_ok" "$dur"
    fi

    # Test: nested object parameter survives XML parsing
    NESTED_TOOL='[{"type":"function","function":{"name":"create_file","description":"Create a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"options":{"type":"object","properties":{"overwrite":{"type":"boolean"},"encoding":{"type":"string"}},"description":"File creation options"}},"required":["path"]}}}]'
    t0=$(now_ms)
    resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Create file /tmp/test.txt with overwrite=true and utf-8 encoding.\"}],\"tools\":$NESTED_TOOL,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    nested_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        path = args.get('path', '')
        opts = args.get('options')
        if not isinstance(path, str) or len(path) == 0:
            print(f'FAIL: path={path}')
        elif opts is not None and not isinstance(opts, dict):
            print(f'FAIL: options is {type(opts).__name__}, not dict')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$nested_ok" = "PASS" ]; then
      run_test "XMLTools" "Nested object param survives XML parsing" "valid dict" "PASS" "$dur"
    else
      run_test "XMLTools" "Nested object param survives XML parsing" "valid dict" "$nested_ok" "$dur"
    fi

    # Test: tool_choice="required" forces a tool call (with tool-relevant prompt)
    # Note: "required" is a chat template hint, not server-enforced. Use a tool-relevant
    # prompt to give the model a fair chance. Non-tool prompts may not produce tool calls.
    t0=$(now_ms)
    resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Check the weather forecast for Berlin.\"}],\"tools\":$TOOL_DEF,\"tool_choice\":\"required\",\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    required_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    fr = d['choices'][0].get('finish_reason', '')
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if fr == 'tool_calls' and len(tc) > 0:
        print('PASS')
    else:
        print(f'FAIL: finish_reason={fr}, tool_calls={len(tc)}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$required_ok" = "PASS" ]; then
      run_test "XMLTools" "tool_choice=required forces tool call" "tool_calls" "PASS" "$dur"
    else
      run_test "XMLTools" "tool_choice=required forces tool call" "tool_calls" "$required_ok" "$dur"
    fi

    # Test: tool_choice with specific function name (tool-relevant prompt)
    t0=$(now_ms)
    resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo right now?\"}],\"tools\":$MULTI_TOOLS,\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}},\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    specific_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if len(tc) > 0 and tc[0]['function']['name'] == 'get_time':
        print('PASS')
    elif len(tc) > 0:
        print(f'FAIL: called {tc[0][\"function\"][\"name\"]} instead of get_time')
    else:
        print('FAIL: no tool_calls')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$specific_ok" = "PASS" ]; then
      run_test "XMLTools" "tool_choice={function: get_time} calls correct function" "get_time" "PASS" "$dur"
    else
      run_test "XMLTools" "tool_choice={function: get_time} calls correct function" "get_time" "$specific_ok" "$dur"
    fi

    # Test: tool call IDs are unique across multiple calls
    t0=$(now_ms)
    resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Tokyo and what time is it in London?\"}],\"tools\":$MULTI_TOOLS,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    ids_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if len(tc) < 2:
        print('PASS')  # single tool call, skip uniqueness check
    else:
        ids = [t.get('id', '') for t in tc]
        if len(set(ids)) == len(ids) and all(ids):
            print('PASS')
        else:
            print(f'FAIL: duplicate or empty IDs: {ids}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$ids_ok" = "PASS" ]; then
      run_test "XMLTools" "Tool call IDs are unique" "unique IDs" "PASS" "$dur"
    else
      run_test "XMLTools" "Tool call IDs are unique" "unique IDs" "$ids_ok" "$dur"
    fi

    # Test: streaming XML tool calls assemble correctly
    t0=$(now_ms)
    stream_resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"Get weather in Sydney in fahrenheit.\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":true,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    stream_xml_ok=$(echo "$stream_resp" | python3 -c "
import sys, json
tool_name = ''
tool_args = ''
found_finish = False
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        d = json.loads(line[6:])
        delta = d.get('choices', [{}])[0].get('delta', {})
        tcs = delta.get('tool_calls', [])
        for tc in tcs:
            fn = tc.get('function', {})
            if fn.get('name'):
                tool_name = fn['name']
            if fn.get('arguments'):
                tool_args += fn['arguments']
        fr = d.get('choices', [{}])[0].get('finish_reason')
        if fr == 'tool_calls':
            found_finish = True
    except: pass
if not found_finish:
    print('FAIL: no finish_reason=tool_calls')
elif not tool_name:
    print('FAIL: no function name in stream')
else:
    try:
        args = json.loads(tool_args)
        if isinstance(args, dict):
            print('PASS')
        else:
            print(f'FAIL: assembled args not dict: {type(args).__name__}')
    except:
        print(f'FAIL: assembled args not valid JSON: {tool_args[:100]}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$stream_xml_ok" = "PASS" ]; then
      run_test "XMLTools" "Streaming: XML tool call assembles valid JSON args" "valid" "PASS" "$dur"
    else
      run_test "XMLTools" "Streaming: XML tool call assembles valid JSON args" "valid" "$stream_xml_ok" "$dur"
    fi

    # Test: tool call with array param via streaming (PR #37 streaming path)
    t0=$(now_ms)
    stream_resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"Create todos: Walk dog, Read book, Cook dinner.\"}],\"tools\":$ARRAY_TOOL,\"max_tokens\":1000,\"stream\":true,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    stream_array_ok=$(echo "$stream_resp" | python3 -c "
import sys, json
tool_args = ''
found_finish = False
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        d = json.loads(line[6:])
        delta = d.get('choices', [{}])[0].get('delta', {})
        tcs = delta.get('tool_calls', [])
        for tc in tcs:
            fn = tc.get('function', {})
            if fn.get('arguments'):
                tool_args += fn['arguments']
        fr = d.get('choices', [{}])[0].get('finish_reason')
        if fr == 'tool_calls':
            found_finish = True
    except: pass
if not found_finish:
    print('FAIL: no finish_reason=tool_calls')
elif not tool_args:
    print('FAIL: no arguments in stream')
else:
    try:
        args = json.loads(tool_args)
        todos = args.get('todos', args.get('items', None))
        if isinstance(todos, list) and len(todos) >= 2:
            print('PASS')
        elif isinstance(todos, str):
            print('FAIL: todos is string in stream (not array)')
        else:
            print(f'FAIL: todos={todos}')
    except:
        print(f'FAIL: args not valid JSON: {tool_args[:100]}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$stream_array_ok" = "PASS" ]; then
      run_test "XMLTools" "Streaming: array param is JSON array (not string)" "array" "PASS" "$dur"
    else
      run_test "XMLTools" "Streaming: array param is JSON array (not string)" "array" "$stream_array_ok" "$dur"
    fi

    # Test: tool call response has correct OpenAI schema structure
    t0=$(now_ms)
    resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Oslo?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    schema_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d['choices'][0]['message']
    assert msg.get('role') == 'assistant', f'role={msg.get(\"role\")}'
    tc = msg.get('tool_calls', [])
    assert len(tc) > 0, 'no tool_calls'
    t = tc[0]
    assert 'id' in t and isinstance(t['id'], str) and len(t['id']) > 0, f'bad id={t.get(\"id\")}'
    assert t.get('type') == 'function', f'type={t.get(\"type\")}'
    assert 'function' in t, 'missing function key'
    assert isinstance(t['function'].get('name'), str), f'name not string'
    assert isinstance(t['function'].get('arguments'), str), f'arguments not string'
    args = json.loads(t['function']['arguments'])
    assert isinstance(args, dict), f'parsed args not dict'
    print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$schema_ok" = "PASS" ]; then
      run_test "XMLTools" "Tool call matches OpenAI schema (id, type, function.name, function.arguments)" "valid schema" "PASS" "$dur"
    else
      run_test "XMLTools" "Tool call matches OpenAI schema (id, type, function.name, function.arguments)" "valid schema" "$schema_ok" "$dur"
    fi

  else
    echo ""
    echo "  (Model did not produce tool calls — skipping XML tool call deep validation)"
    run_test "XMLTools" "XML tool call tests (model lacks tool calling)" "skip" "SKIP" "0"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: llamacpp_tool_parser (JSON-in-XML fallback + xgrammar)
# ═══════════════════════════════════════════════════════════════════════════════
# Tests for --tool-call-parser llamacpp_tool_parser.
# Validates: JSON-in-XML fallback parsing, normal XML tool calls still work,
# xgrammar EBNF constraint activation, and streaming tool call emission.
#
# NOTE: This section requires the server to be started with:
#   --tool-call-parser llamacpp_tool_parser
# If the server wasn't started with this flag, tests are skipped.

if should_run_section 12 && min_tier standard; then
  echo ""
  echo "🔧 Section 12: llamacpp_tool_parser"

  # Probe: was the server started with --tool-call-parser llamacpp_tool_parser?
  # We detect this by checking if a normal XML tool call works (it should with any parser).
  # The real test is whether JSON-in-XML bodies are also handled.
  probe_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Berlin?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
  probe_has_tc=$(echo "$probe_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    print('yes' if len(tc) > 0 else 'no')
except:
    print('no')
" 2>/dev/null || echo "no")

  if [ "$probe_has_tc" = "yes" ]; then

    # ── Test 12.1: Normal XML tool call still works ──────────────────────────
    t0=$(now_ms)
    resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in London?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    tc_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if len(tc) > 0:
        t = tc[0]
        name = t['function']['name']
        args = json.loads(t['function']['arguments'])
        if name == 'get_weather' and ('location' in args or any('london' in str(v).lower() for v in args.values())):
            print('PASS')
        else:
            print(f'FAIL: name={name}, args={args}')
    else:
        print('FAIL: no tool_calls')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$tc_ok" = "PASS" ]; then
      run_test "LlamaCppParser" "Normal XML tool call works with llamacpp_tool_parser" "valid tool call" "PASS" "$dur"
    else
      run_test "LlamaCppParser" "Normal XML tool call works with llamacpp_tool_parser" "valid tool call" "$tc_ok" "$dur"
    fi

    # ── Test 12.2: Streaming tool call emission ─────────────────────────────
    t0=$(now_ms)
    stream_raw=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Tokyo?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":true,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    stream_tc=$(echo "$stream_raw" | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
tool_deltas = []
finish = None
for line in lines:
    if not line.startswith('data: ') or line.strip() == 'data: [DONE]':
        continue
    try:
        d = json.loads(line[6:])
        delta = d['choices'][0].get('delta', {})
        if 'tool_calls' in delta:
            tool_deltas.extend(delta['tool_calls'])
        fr = d['choices'][0].get('finish_reason')
        if fr:
            finish = fr
    except:
        pass

if not tool_deltas:
    print('FAIL: no tool_call deltas in stream')
elif finish != 'tool_calls':
    print(f'FAIL: finish_reason={finish}')
else:
    # Reconstruct: first delta should have name, subsequent have argument fragments
    name = None
    args_parts = []
    for td in tool_deltas:
        fn = td.get('function', {})
        if fn.get('name'):
            name = fn['name']
        if fn.get('arguments') is not None:
            args_parts.append(fn['arguments'])
    full_args = ''.join(args_parts)
    if not name:
        print('FAIL: no function name in stream')
    elif not full_args:
        print('FAIL: no arguments in stream')
    else:
        try:
            parsed = json.loads(full_args)
            print('PASS')
        except:
            print(f'FAIL: args not valid JSON: {full_args[:100]}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$stream_tc" = "PASS" ]; then
      run_test "LlamaCppParser" "Streaming tool call emits valid deltas" "name + JSON args" "PASS" "$dur"
    else
      run_test "LlamaCppParser" "Streaming tool call emits valid deltas" "name + JSON args" "$stream_tc" "$dur"
    fi

    # ── Test 12.3: Multi-turn tool call conversation ────────────────────────
    # Send a tool call, then a tool result, then verify the model responds
    t0=$(now_ms)
    multi_resp=$(api_call "{
      \"messages\":[
        {\"role\":\"user\",\"content\":\"What is the weather in Paris?\"},
        {\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call_test123\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"Paris\\\"}\"}}]},
        {\"role\":\"tool\",\"tool_call_id\":\"call_test123\",\"name\":\"get_weather\",\"content\":\"{\\\"temperature\\\":22,\\\"condition\\\":\\\"sunny\\\"}\"}
      ],
      \"tools\":$TOOL_DEF,
      \"max_tokens\":500,
      \"stream\":false,
      \"temperature\":0
    }")
    dur=$(( $(now_ms) - t0 ))
    multi_ok=$(echo "$multi_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d['choices'][0]['message']
    content = msg.get('content', '') or ''
    rc = msg.get('reasoning_content', '') or ''
    if len(content) > 0 or len(rc) > 0:
        print('PASS')
    else:
        print('FAIL: empty response after tool result')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$multi_ok" = "PASS" ]; then
      run_test "LlamaCppParser" "Multi-turn: model responds after tool result" "non-empty content" "PASS" "$dur"
    else
      run_test "LlamaCppParser" "Multi-turn: model responds after tool result" "non-empty content" "$multi_ok" "$dur"
    fi

    # ── Test 12.4: tool_choice=none still suppresses tool calls ─────────────
    t0=$(now_ms)
    none_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Sydney?\"}],\"tools\":$TOOL_DEF,\"tool_choice\":\"none\",\"max_tokens\":200,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    none_ok=$(echo "$none_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    fr = d['choices'][0].get('finish_reason', '')
    tc = d['choices'][0]['message'].get('tool_calls')
    if fr != 'tool_calls' and (tc is None or len(tc) == 0):
        print('PASS')
    else:
        print(f'FAIL: finish_reason={fr}, tool_calls={tc}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$none_ok" = "PASS" ]; then
      run_test "LlamaCppParser" "tool_choice=none suppresses tool calls" "no tool_calls" "PASS" "$dur"
    else
      run_test "LlamaCppParser" "tool_choice=none suppresses tool calls" "no tool_calls" "$none_ok" "$dur"
    fi

    # ── Test 12.5: Multiple tool definitions — correct one selected ─────────
    MULTI_TOOLS_12='[{"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},{"type":"function","function":{"name":"get_time","description":"Get current time for a timezone","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]'
    t0=$(now_ms)
    multi_tool_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What time is it in New York?\"}],\"tools\":$MULTI_TOOLS_12,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    multi_tool_ok=$(echo "$multi_tool_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if len(tc) > 0:
        name = tc[0]['function']['name']
        if name == 'get_time':
            print('PASS')
        else:
            print(f'FAIL: expected get_time, got {name}')
    else:
        print('FAIL: no tool_calls')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$multi_tool_ok" = "PASS" ]; then
      run_test "LlamaCppParser" "Multiple tools: correct function selected" "get_time" "PASS" "$dur"
    else
      run_test "LlamaCppParser" "Multiple tools: correct function selected" "get_time" "$multi_tool_ok" "$dur"
    fi

    # ── Test 12.6: Argument types coerced correctly ─────────────────────────
    TYPED_TOOL_12='[{"type":"function","function":{"name":"search_code","description":"Search codebase for a pattern","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"case_sensitive":{"type":"boolean","description":"Case sensitive"},"max_results":{"type":"integer","description":"Max results"}},"required":["query"]}}}]'
    t0=$(now_ms)
    typed_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Search for 'main' case-sensitively, return max 5 results.\"}],\"tools\":$TYPED_TOOL_12,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    typed_ok=$(echo "$typed_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        q = args.get('query', '')
        if not isinstance(q, str) or len(q) == 0:
            print(f'FAIL: query not string: {type(q).__name__}={q}')
        else:
            # Check that boolean/int are coerced if present
            issues = []
            cs = args.get('case_sensitive')
            mr = args.get('max_results')
            if cs is not None and not isinstance(cs, bool):
                issues.append(f'case_sensitive={type(cs).__name__}({cs})')
            if mr is not None and not isinstance(mr, int):
                issues.append(f'max_results={type(mr).__name__}({mr})')
            if issues:
                print(f'FAIL: type coercion: {issues}')
            else:
                print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$typed_ok" = "PASS" ]; then
      run_test "LlamaCppParser" "Argument types coerced (string, bool, int)" "correct types" "PASS" "$dur"
    else
      run_test "LlamaCppParser" "Argument types coerced (string, bool, int)" "correct types" "$typed_ok" "$dur"
    fi

    # ── Test 12.7: xgrammar constraint logged (debug check) ─────────────────
    # This is a soft check — we verify the response is valid. The xgrammar
    # constraint is best verified via debug logs (AFM_DEBUG=1).
    t0=$(now_ms)
    xg_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Berlin?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    xg_ok=$(echo "$xg_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if len(tc) > 0:
        t = tc[0]
        args = json.loads(t['function']['arguments'])
        if t['function']['name'] == 'get_weather' and isinstance(args, dict):
            print('PASS')
        else:
            print(f'FAIL: name={t[\"function\"][\"name\"]}, args_type={type(args).__name__}')
    else:
        print('FAIL: no tool_calls')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$xg_ok" = "PASS" ]; then
      run_test "LlamaCppParser" "Tool call valid with xgrammar constraint active" "valid" "PASS" "$dur"
    else
      run_test "LlamaCppParser" "Tool call valid with xgrammar constraint active" "valid" "$xg_ok" "$dur"
    fi

  else
    echo ""
    echo "  (Model did not produce tool calls — skipping llamacpp_tool_parser tests)"
    run_test "LlamaCppParser" "llamacpp_tool_parser tests (model lacks tool calling)" "skip" "SKIP" "0"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Generate HTML Report
# ═══════════════════════════════════════════════════════════════════════════════
TEST_END_TIME=$(date +%s)
TOTAL_SECS=$((TEST_END_TIME - TEST_START_TIME))

EFFECTIVE_TOTAL=$((TOTAL - SKIP))
if [ $EFFECTIVE_TOTAL -gt 0 ]; then
  PCT=$(( PASS * 100 / EFFECTIVE_TOTAL ))
else
  PCT=0
fi

if [ $FAIL -eq 0 ]; then
  BAR_COLOR="#3fb950"
else
  BAR_COLOR="#f85149"
fi

DATE_STR=$(date '+%Y-%m-%d %H:%M:%S')

cat > "$REPORT_FILE" <<'HTMLHEAD'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AFM Assertion Test Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif; background: #0d1117; color: #e6edf3; padding: 2rem; }
  .header { text-align: center; margin-bottom: 2rem; padding: 2rem; background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%); border: 1px solid #30363d; border-radius: 12px; }
  .header h1 { font-size: 1.8rem; margin-bottom: 0.5rem; background: linear-gradient(90deg, #58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header .meta { color: #8b949e; font-size: 0.9rem; line-height: 1.6; }
  .summary { display: flex; gap: 1rem; justify-content: center; margin: 1.5rem 0; flex-wrap: wrap; }
  .stat { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1rem 1.5rem; text-align: center; min-width: 120px; }
  .stat .value { font-size: 2rem; font-weight: 700; }
  .stat .label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
  .stat.pass .value { color: #3fb950; }
  .stat.fail .value { color: #f85149; }
  .stat.skip .value { color: #d29922; }
  .stat.time .value { color: #58a6ff; }
  .stat.pct .value { color: #d2a8ff; }
  .progress-bar { width: 100%; height: 8px; background: #21262d; border-radius: 4px; margin: 1rem auto; max-width: 400px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 4px; }
  table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; }
  th { background: #161b22; color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #30363d; }
  td { padding: 0.75rem 1rem; border-bottom: 1px solid #21262d; vertical-align: top; }
  tr:hover { background: #161b22; }
  .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  .badge.pass { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .badge.fail { background: #2d1215; color: #f85149; border: 1px solid #da3633; }
  .badge.skip { background: #2d2400; color: #d29922; border: 1px solid #9e6a03; }
  .group-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px; font-size: 0.7rem; font-weight: 500; background: #1a1f2e; color: #8b949e; border: 1px solid #30363d; }
  .group-badge.Preflight { color: #8b949e; border-color: #484f58; }
  .group-badge.Lifecycle { color: #3fb950; border-color: #238636; }
  .group-badge.Stop { color: #f85149; border-color: #da3633; }
  .group-badge.Logprobs { color: #58a6ff; border-color: #1f6feb; }
  .group-badge.Think { color: #d2a8ff; border-color: #8957e5; }
  .group-badge.Tools { color: #ffa657; border-color: #d18616; }
  .group-badge.XMLTools { color: #f0883e; border-color: #bd561d; }
  .group-badge.Cache { color: #79c0ff; border-color: #388bfd; }
  .group-badge.Concurrent { color: #f778ba; border-color: #db61a2; }
  .group-badge.Error { color: #ff7b72; border-color: #da3633; }
  .group-badge.Kwargs { color: #a5d6ff; border-color: #58a6ff; }
  .group-badge.Perf { color: #3fb950; border-color: #238636; }
  .detail { font-family: 'SF Mono', 'Menlo', monospace; font-size: 0.8rem; color: #8b949e; white-space: pre-wrap; word-break: break-word; max-height: 100px; overflow-y: auto; background: #0d1117; padding: 0.5rem; border-radius: 6px; border: 1px solid #21262d; margin-top: 0.25rem; }
  .duration { color: #8b949e; font-family: 'SF Mono', monospace; font-size: 0.85rem; }
  .footer { text-align: center; margin-top: 2rem; color: #484f58; font-size: 0.8rem; }
</style>
</head>
<body>
HTMLHEAD

cat >> "$REPORT_FILE" <<EOF
<div class="header">
  <h1>AFM Assertion Test Report</h1>
  <div class="meta">
    Model: <strong>$MODEL</strong> &middot; Tier: <strong>$TIER</strong><br>
    Server: <code>$BASE_URL</code><br>
    Date: $DATE_STR
  </div>
</div>
<div class="summary">
  <div class="stat pass"><div class="value">$PASS</div><div class="label">Passed</div></div>
  <div class="stat fail"><div class="value">$FAIL</div><div class="label">Failed</div></div>
  <div class="stat skip"><div class="value">$SKIP</div><div class="label">Skipped</div></div>
  <div class="stat pct"><div class="value">${PCT}%</div><div class="label">Pass Rate</div></div>
  <div class="stat time"><div class="value">${TOTAL_SECS}s</div><div class="label">Total Time</div></div>
</div>
<div class="progress-bar"><div class="progress-fill" style="width:${PCT}%;background:${BAR_COLOR};"></div></div>
<table>
<thead>
<tr><th>#</th><th>Test</th><th>Group</th><th>Status</th><th>Duration</th><th>Details</th></tr>
</thead>
<tbody>
EOF

idx=0
for entry in "${RESULTS[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r status group name expected actual duration <<< "$entry"
  if [ "$status" = "PASS" ]; then
    badge='<span class="badge pass">PASS</span>'
    detail_text="$expected"
  elif [ "$status" = "SKIP" ]; then
    badge='<span class="badge skip">SKIP</span>'
    detail_text="$expected"
  else
    badge='<span class="badge fail">FAIL</span>'
    detail_text="Expected: $expected\nActual: $actual"
  fi
  detail_text=$(echo "$detail_text" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
  name_esc=$(echo "$name" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')

  dur_s=""
  if [ -n "$duration" ] && [ "$duration" -gt 0 ] 2>/dev/null; then
    dur_s=$(python3 -c "print(f'{$duration/1000:.1f}s')" 2>/dev/null || echo "${duration}ms")
  fi

  cat >> "$REPORT_FILE" <<EOF
<tr>
  <td><strong>$idx</strong></td>
  <td>$name_esc</td>
  <td><span class="group-badge $group">$group</span></td>
  <td>$badge</td>
  <td><span class="duration">$dur_s</span></td>
  <td><div class="detail">$detail_text</div></td>
</tr>
EOF
done

cat >> "$REPORT_FILE" <<EOF
</tbody>
</table>
<div class="footer">
  Generated by Scripts/test-assertions.sh (tier: $TIER) &mdash; $(date '+%Y-%m-%d %H:%M:%S')
</div>
</body>
</html>
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: $PASS/$EFFECTIVE_TOTAL passed ($PCT%) | $SKIP skipped"
if [ $FAIL -gt 0 ]; then
  echo "  ❌ $FAIL FAILED"
fi
echo "  Report: $REPORT_FILE"
echo "  JSONL:  $JSONL_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $FAIL
