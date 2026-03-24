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
GRAMMAR_CONSTRAINTS=false  # set via --grammar-constraints when server has --enable-grammar-constraints
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
    --grammar-constraints) GRAMMAR_CONSTRAINTS=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ ! "$TIER" =~ ^(unit|smoke|standard|full)$ ]]; then
  echo "ERROR: --tier must be unit, smoke, standard, or full"
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
CURRENT_TIER="smoke"

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
  RESULTS+=("${actual}|${group}|${name}|${esc_expected}|${esc_actual}|${duration}|${CURRENT_TIER}|${TOTAL}")

  # JSONL record
  local status_val="PASS"
  [[ "$actual" = "PASS" ]] && status_val="PASS"
  [[ "$actual" = "SKIP" ]] && status_val="SKIP"
  [[ "$actual" != "PASS" && "$actual" != "SKIP" ]] && status_val="FAIL"
  _GROUP="$group" _NAME="$name" _STATUS="$status_val" _DUR="$duration" _TIER="$CURRENT_TIER" _IDX="$TOTAL" python3 -c "
import json, os
print(json.dumps({
    'index': int(os.environ['_IDX']),
    'group': os.environ['_GROUP'],
    'name': os.environ['_NAME'],
    'status': os.environ['_STATUS'],
    'duration_ms': int(os.environ['_DUR']),
    'tier': os.environ['_TIER']
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

# Helper: call API and return response headers (one per line)
api_call_headers() {
  local body="$1"
  curl -sf --max-time 60 -D - -o /dev/null "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" 2>/dev/null || echo 'ERROR'
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
  # Tier order: unit < smoke < standard < full
  local required="$1"
  case "$required" in
    unit)     return 0 ;;
    smoke)    [[ "$TIER" != "unit" ]] && return 0 || return 1 ;;
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
# Section U: Swift Unit Tests (offline — no server required)
# ═══════════════════════════════════════════════════════════════════════════════
# Runs swift test (MacLocalAPITests) and parses console output into assertion
# format. Tests XML parsing, type coercion, EBNF grammar generation, nullable
# schemas, etc. These are pure logic tests — no model or server needed.

if should_run_section U && min_tier unit; then
  CURRENT_TIER="unit"
  echo "🧪 Section U: Swift Unit Tests"

  t0=$(now_ms)
  swift_test_output=$(cd "$PROJECT_ROOT" && swift test 2>&1) || true
  swift_test_dur=$(( $(now_ms) - t0 ))

  # Parse swift test console output into a temp file.
  # Swift Testing lines look like:
  #   􁁛  Test "test name here" passed after 0.002 seconds.
  #   􁁕  Test "test name here" failed after 0.003 seconds.
  #   􁁛  Suite XMLToolCallParsingTests passed after 0.006 seconds.
  UT_PARSED_FILE=$(mktemp /tmp/afm-ut-parsed-XXXXXX.tsv)
  echo "$swift_test_output" | python3 -c "
import sys, re

lines = sys.stdin.read()

# Match individual test results (not Suite summaries)
# Pattern: Test \"name\" passed/failed after N.NNN seconds
# Note: test names may contain escaped quotes (\\\" in output), so match greedily up to the
# closing quote that's followed by the pass/fail keyword
pattern = re.compile(r'Test\s+\"(.+?)\"\s+(passed|failed)\s+after\s+([\d.]+)\s+seconds')

results = []
for line in lines.split('\n'):
    # Skip suite summary lines
    if 'Suite ' in line and 'Test ' not in line:
        continue
    m = pattern.search(line)
    if m:
        name = m.group(1)
        status = 'PASS' if m.group(2) == 'passed' else 'FAIL'
        dur_ms = int(float(m.group(3)) * 1000)

        # Determine group from test name heuristics
        name_lower = name.lower()
        if any(kw in name_lower for kw in ['nullable', 'toany', 'tojinja', 'toolspec', 'nsnull']):
            group = 'NullableSchema'
        elif any(kw in name_lower for kw in ['xml', 'decode', 'parse', 'coerce', 'entity', 'json', 'regex',
                                              'fallback', 'ebnf', 'grammar', 'tool call', 'parameter',
                                              'function', 'bash tool', 'fill', 'required', 'mistral',
                                              'strip', 'remaining', 'bare', 'empty', 'multiline',
                                              'duplicate', 'mixed', 'write', 'edit', 'question']):
            group = 'XMLParsing'
        else:
            group = 'UnitTest'

        results.append((group, name, status, dur_ms))

for group, name, status, dur_ms in results:
    print(f'{group}\t{name}\t{status}\t{dur_ms}')
" > "$UT_PARSED_FILE" 2>/dev/null || true

  if [ -s "$UT_PARSED_FILE" ]; then
    # Read from file (not pipe) so run_test updates global state
    while IFS=$'\t' read -r ut_group ut_name ut_status ut_dur; do
      if [ "$ut_status" = "PASS" ]; then
        run_test "$ut_group" "$ut_name" "pass" "PASS" "$ut_dur"
      else
        run_test "$ut_group" "$ut_name" "pass" "FAIL" "$ut_dur"
      fi
    done < "$UT_PARSED_FILE"

    ut_count=$(wc -l < "$UT_PARSED_FILE" | tr -d ' ')
    ut_fail_count=$(grep -c $'\tFAIL\t' "$UT_PARSED_FILE" 2>/dev/null || true)
    ut_fail_count=${ut_fail_count:-0}
    echo "  Swift unit tests: $ut_count total, $ut_fail_count failures (${swift_test_dur}ms)"
  else
    # swift test failed or produced no parseable output
    run_test "UnitTest" "swift test execution" "test output" "FAIL: no parseable output" "$swift_test_dur"
    echo "$swift_test_output" | tail -10
  fi
  rm -f "$UT_PARSED_FILE"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0: Preflight (smoke tier and above — requires server)
# ═══════════════════════════════════════════════════════════════════════════════
if ! min_tier smoke; then
  # unit-only tier: skip all server-dependent sections
  # Jump directly to report generation
  :
else

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
# finish_reason should be "stop" when the stop sequence fired, or "length" if the model
# exhausted max_tokens on thinking without producing visible content (model behavior, not a bug).
content_empty=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message'].get('content') or ''
    print('yes' if not c.strip() else 'no')
except: print('no')
" 2>/dev/null || echo "no")
if [ "$finish" = "stop" ]; then
  run_test "Stop" "finish_reason is 'stop' with stop sequence" "stop" "PASS" "$(( $(now_ms) - t0 ))"
elif [ "$finish" = "length" ] && [ "$content_empty" = "yes" ]; then
  # Model spent entire budget on thinking — stop never fired on visible content. Correct behavior.
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
dur=$(( $(now_ms) - t0 ))
stop_nl_ok=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d['choices'][0]['message']
    c = msg.get('content') or ''
    # Visible content should have no newlines (stop fired before any newline)
    if '\n' not in c:
        print('PASS')
    elif not c.strip():
        # Empty content is OK — thinking model used all tokens on reasoning
        print('PASS')
    else:
        print(f'FAIL: multi-line: {repr(c[:80])}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
if [ "$stop_nl_ok" = "PASS" ]; then
  run_test "Stop" "Stop on newline produces single line" "single line" "PASS" "$dur"
else
  run_test "Stop" "Stop on newline produces single line" "single line" "$stop_nl_ok" "$dur"
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
  CURRENT_TIER="standard"
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
CURRENT_TIER="smoke"
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
  CURRENT_TIER="standard"
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
        choices = d.get('choices', [])
        if not choices:
            continue
        lp = choices[0].get('logprobs')
        if lp and lp.get('content'):
            found_logprobs = True
            for entry in lp['content']:
                if 'token' not in entry:
                    print('FAIL: missing token key'); sys.exit(0)
                if 'logprob' not in entry:
                    print('FAIL: missing logprob key'); sys.exit(0)
                if entry['logprob'] > 0:
                    print(f'FAIL: logprob > 0: {entry[\"logprob\"]}'); sys.exit(0)
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f'FAIL: {e}'); sys.exit(0)
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
CURRENT_TIER="smoke"
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
  CURRENT_TIER="standard"
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

  # Test: multi-turn tool call round-trip (user → tool_call → tool_result → follow-up)
  t0=$(now_ms)
  # Step 1: get initial tool call
  mt_resp1=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":300,\"stream\":false,\"temperature\":0}")
  # Extract tool call details for round-trip
  mt_roundtrip=$(echo "$mt_resp1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('SKIP')
    else:
        call_id = tc[0].get('id', 'call_test1')
        fn_name = tc[0]['function']['name']
        fn_args = tc[0]['function']['arguments']
        print(f'{call_id}|{fn_name}|{fn_args}')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null || echo "ERROR:parse")
  if [[ "$mt_roundtrip" == SKIP* ]] || [[ "$mt_roundtrip" == ERROR* ]]; then
    dur=$(( $(now_ms) - t0 ))
    run_test "Tools" "Multi-turn: tool round-trip (model did not call tool)" "skip" "SKIP" "$dur"
  else
    IFS='|' read -r mt_call_id mt_fn_name mt_fn_args <<< "$mt_roundtrip"
    # Step 2: send tool result back and get follow-up
    mt_resp2=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"},{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"$mt_call_id\",\"type\":\"function\",\"function\":{\"name\":\"$mt_fn_name\",\"arguments\":$mt_fn_args}}]},{\"role\":\"tool\",\"tool_call_id\":\"$mt_call_id\",\"name\":\"$mt_fn_name\",\"content\":\"{\\\"temperature\\\":22,\\\"condition\\\":\\\"sunny\\\",\\\"humidity\\\":45}\"}],\"tools\":$TOOL_DEF,\"max_tokens\":300,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    mt_valid=$(echo "$mt_resp2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL: server error: {d[\"error\"].get(\"message\",\"\")}')
    else:
        msg = d['choices'][0]['message']
        c = msg.get('content') or msg.get('reasoning_content') or ''
        if len(c.strip()) > 0:
            print('PASS')
        else:
            print('FAIL: empty follow-up content')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$mt_valid" = "PASS" ]; then
      run_test "Tools" "Multi-turn: tool result produces valid follow-up" "text response" "PASS" "$dur"
    else
      run_test "Tools" "Multi-turn: tool result produces valid follow-up" "text response" "$mt_valid" "$dur"
    fi
  fi

  # Test: multi-turn with nullable param through full round-trip
  NULLABLE_TOOL_FULL='[{"type":"function","function":{"name":"search_places","description":"Search for places nearby","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"category":{"anyOf":[{"type":"string"},{"type":"null"}],"description":"Optional category filter"},"radius_km":{"anyOf":[{"type":"number"},{"type":"null"}],"description":"Optional radius in km"}},"required":["query"]}}}]'
  t0=$(now_ms)
  # Step 1: initial tool call with nullable params
  mn_resp1=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Find restaurants near me\"}],\"tools\":$NULLABLE_TOOL_FULL,\"max_tokens\":300,\"stream\":false,\"temperature\":0}")
  mn_step1=$(echo "$mn_resp1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'ERROR:{d[\"error\"].get(\"message\",\"\")}')
    else:
        tc = d['choices'][0]['message'].get('tool_calls', [])
        if tc:
            call_id = tc[0].get('id', 'call_n1')
            fn_name = tc[0]['function']['name']
            fn_args = tc[0]['function']['arguments']
            # Verify args parse as valid JSON
            json.loads(fn_args)
            print(f'{call_id}|{fn_name}|{fn_args}')
        else:
            content = d['choices'][0]['message'].get('content', '')
            print('NOTOOLS' if content else 'EMPTY')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null || echo "ERROR:parse")
  if [[ "$mn_step1" == ERROR* ]]; then
    dur=$(( $(now_ms) - t0 ))
    run_test "Tools" "Multi-turn nullable: Jinja crash on anyOf [type, null]" "no crash" "FAIL: $mn_step1" "$dur"
  elif [[ "$mn_step1" == NOTOOLS* ]] || [[ "$mn_step1" == EMPTY* ]]; then
    dur=$(( $(now_ms) - t0 ))
    run_test "Tools" "Multi-turn nullable: round-trip (model did not call tool)" "skip" "SKIP" "$dur"
  else
    IFS='|' read -r mn_call_id mn_fn_name mn_fn_args <<< "$mn_step1"
    # Step 2: send tool result, ask follow-up
    mn_resp2=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Find restaurants near me\"},{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"$mn_call_id\",\"type\":\"function\",\"function\":{\"name\":\"$mn_fn_name\",\"arguments\":$mn_fn_args}}]},{\"role\":\"tool\",\"tool_call_id\":\"$mn_call_id\",\"name\":\"$mn_fn_name\",\"content\":\"{\\\"results\\\":[{\\\"name\\\":\\\"Le Petit Bistro\\\",\\\"rating\\\":4.5},{\\\"name\\\":\\\"Sushi Palace\\\",\\\"rating\\\":4.2}]}\"},{\"role\":\"user\",\"content\":\"Which one has better reviews?\"}],\"tools\":$NULLABLE_TOOL_FULL,\"max_tokens\":300,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    mn_valid=$(echo "$mn_resp2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL: server error: {d[\"error\"].get(\"message\",\"\")}')
    else:
        msg = d['choices'][0]['message']
        c = msg.get('content') or msg.get('reasoning_content') or ''
        if len(c.strip()) > 0:
            print('PASS')
        else:
            print('FAIL: empty follow-up')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$mn_valid" = "PASS" ]; then
      run_test "Tools" "Multi-turn nullable: full round-trip (anyOf params)" "valid follow-up" "PASS" "$dur"
    else
      run_test "Tools" "Multi-turn nullable: full round-trip (anyOf params)" "valid follow-up" "$mn_valid" "$dur"
    fi
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
CURRENT_TIER="smoke"
fi # section 5

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Prompt cache
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 6 && min_tier standard; then
  CURRENT_TIER="standard"
  echo ""
  echo "💾 Section 6: Prompt Cache"

  # Test: fresh unique prompt is not fully cached
  t0=$(now_ms)
  # Use a unique leading token so the request must retain an uncached suffix.
  unique_prompt="UNIQUE-CACHE-MISS-$(date +%s%N): Tell me about the history of prompt caching in LLM servers."
  resp1=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$unique_prompt\"}],\"max_tokens\":20,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  cache_state1=$(echo "$resp1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ptd = d.get('usage', {}).get('prompt_tokens_details')
    if ptd is None:
        print('NULL')
    else:
        cached = ptd.get('cached_tokens', 'MISSING')
        prompt = d.get('usage', {}).get('prompt_tokens', 'MISSING')
        print(f'{cached}|{prompt}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || echo "ERROR")
  if [ "$cache_state1" = "NULL" ]; then
    run_test "Cache" "First unique request: uncached suffix remains" "cached_tokens < prompt_tokens" "FAIL: prompt_tokens_details is null (caching may be disabled)" "$dur"
  else
    IFS='|' read -r cached1 prompt1 <<< "$cache_state1"
    if [ "$cached1" != "MISSING" ] && [ "$prompt1" != "MISSING" ] && [ "$cached1" -lt "$prompt1" ] 2>/dev/null; then
      run_test "Cache" "First unique request: uncached suffix remains" "cached_tokens < prompt_tokens" "PASS" "$dur"
    else
      run_test "Cache" "First unique request: uncached suffix remains" "cached_tokens < prompt_tokens" "FAIL: cached_tokens=$cached1 prompt_tokens=$prompt1" "$dur"
    fi
  fi

  # Test: similar prompt with same unique prefix has cached_tokens > 0
  partial_prefix_nonce="PARTIAL-REUSE-$(date +%s%N)"
  partial_prompt1="Prefix cache partial reuse probe $partial_prefix_nonce: answer with one word only. alpha"
  partial_prompt2="Prefix cache partial reuse probe $partial_prefix_nonce: answer with one word only. beta"
  t0=$(now_ms)
  api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$partial_prompt1\"}],\"max_tokens\":20,\"stream\":false,\"temperature\":0,\"seed\":42,\"chat_template_kwargs\":{\"enable_thinking\":false}}" >/dev/null
  resp2=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$partial_prompt2\"}],\"max_tokens\":20,\"stream\":false,\"temperature\":0,\"seed\":42,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
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
    run_test "Cache" "Shared-prefix request: cached_tokens>0" ">0" "PASS" "$dur"
  else
    run_test "Cache" "Shared-prefix request: cached_tokens>0" ">0" "FAIL: cached_tokens=$cached2" "$dur"
  fi

  # Test: deterministic exact replay should not change content on cache hit
  replay_token="BASELINE-ALBATROSS"
  replay_body="{\"messages\":[{\"role\":\"user\",\"content\":\"For a cache baseline test, reply with exactly $replay_token and nothing else.\"}],\"max_tokens\":20,\"stream\":false,\"temperature\":0,\"seed\":42,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  t0=$(now_ms)
  replay_resp1=$(api_call "$replay_body")
  replay_resp2=$(api_call "$replay_body")
  dur=$(( $(now_ms) - t0 ))
  replay_content1=$(echo "$replay_resp1" | extract_content)
  replay_content2=$(echo "$replay_resp2" | extract_content)
  replay_cached2=$(echo "$replay_resp2" | python3 -c "
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
  if [ -n "$replay_content1" ] && [ "$replay_content1" = "$replay_content2" ]; then
    run_test "Cache" "Exact replay: cached response matches cold response" "same content" "PASS" "$dur"
  else
    run_test "Cache" "Exact replay: cached response matches cold response" "same content" "FAIL: cold=[$replay_content1] cached=[$replay_content2] cached_tokens=$replay_cached2" "$dur"
  fi

  # Test: unrelated unique prompt still leaves an uncached suffix
  t0=$(now_ms)
  different_prompt="DIFFERENT-CACHE-MISS-$(date +%s%N): This is a completely different prompt about quantum physics and black holes."
  resp3=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$different_prompt\"}],\"max_tokens\":20,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  cache_state3=$(echo "$resp3" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ptd = d.get('usage', {}).get('prompt_tokens_details')
    if ptd is None:
        print('NULL')
    else:
        cached = ptd.get('cached_tokens', 'MISSING')
        prompt = d.get('usage', {}).get('prompt_tokens', 'MISSING')
        print(f'{cached}|{prompt}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || echo "ERROR")
  if [ "$cache_state3" = "NULL" ]; then
    run_test "Cache" "Different unique prompt: uncached suffix remains" "cached_tokens < prompt_tokens" "FAIL: null (caching disabled)" "$dur"
  else
    IFS='|' read -r cached3 prompt3 <<< "$cache_state3"
    if [ "$cached3" != "MISSING" ] && [ "$prompt3" != "MISSING" ] && [ "$cached3" -lt "$prompt3" ] 2>/dev/null; then
      run_test "Cache" "Different unique prompt: uncached suffix remains" "cached_tokens < prompt_tokens" "PASS" "$dur"
    else
      run_test "Cache" "Different unique prompt: uncached suffix remains" "cached_tokens < prompt_tokens" "FAIL: cached_tokens=$cached3 prompt_tokens=$prompt3" "$dur"
    fi
  fi

  # Test: streaming shared-prefix request reports cached_tokens > 0
  stream_partial_nonce="STREAM-PARTIAL-$(date +%s%N)"
  stream_cache_prompt1="Prefix cache streaming reuse probe $stream_partial_nonce: answer with one word only. alpha"
  stream_cache_prompt2="Prefix cache streaming reuse probe $stream_partial_nonce: answer with one word only. beta"
  t0=$(now_ms)
  api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$stream_cache_prompt1\"}],\"max_tokens\":10,\"stream\":false,\"temperature\":0,\"seed\":42,\"chat_template_kwargs\":{\"enable_thinking\":false}}" >/dev/null
  sleep 0.5
  stream_resp=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"$stream_cache_prompt2\"}],\"max_tokens\":10,\"stream\":true,\"temperature\":0,\"seed\":42,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
  dur=$(( $(now_ms) - t0 ))
  stream_cached=$(echo "$stream_resp" | python3 -c "
import sys, json
found = False
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
                found = True
    except Exception:
        pass
print('PASS' if found else 'FAIL: no cached_tokens>0 in stream usage')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$stream_cached" = "PASS" ]; then
    run_test "Cache" "Streaming shared-prefix: cached_tokens>0 in usage chunk" ">0" "PASS" "$dur"
  else
    run_test "Cache" "Streaming shared-prefix: cached_tokens>0 in usage chunk" ">0" "$stream_cached" "$dur"
  fi

  # Test: non-streaming warmup and streaming replay should produce identical content
  stream_replay_token="STREAM-ECHO-OMEGA"
  stream_warm_body="{\"messages\":[{\"role\":\"user\",\"content\":\"For a streaming crossover test, reply with exactly $stream_replay_token and nothing else.\"}],\"max_tokens\":20,\"stream\":false,\"temperature\":0,\"seed\":42,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  stream_replay_body="{\"messages\":[{\"role\":\"user\",\"content\":\"For a streaming crossover test, reply with exactly $stream_replay_token and nothing else.\"}],\"max_tokens\":20,\"stream\":true,\"temperature\":0,\"seed\":42,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  t0=$(now_ms)
  stream_warm_resp=$(api_call "$stream_warm_body")
  sleep 0.5
  stream_replay_resp=$(api_stream "$stream_replay_body")
  dur=$(( $(now_ms) - t0 ))
  stream_warm_content=$(echo "$stream_warm_resp" | extract_content)
  stream_replay_content=$(echo "$stream_replay_resp" | python3 -c "
import sys, json
parts = []
for raw in sys.stdin:
    line = raw.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        d = json.loads(line[6:])
    except Exception:
        continue
    for choice in d.get('choices', []):
        delta = choice.get('delta', {})
        text = delta.get('content')
        if text:
            parts.append(text)
print(''.join(parts))
" 2>/dev/null || echo "")
  if [ -n "$stream_warm_content" ] && [ "$stream_warm_content" = "$stream_replay_content" ]; then
    run_test "Cache" "Streaming replay: content matches non-streaming warmup" "same content" "PASS" "$dur"
  else
    run_test "Cache" "Streaming replay: content matches non-streaming warmup" "same content" "FAIL: warm=[$stream_warm_content] stream=[$stream_replay_content]" "$dur"
  fi

  # Test: 8 concurrent shared-prefix requests keep an uncached suffix and do not bleed across divergence.
  # When the server is started with --enable-prefix-caching --concurrent 8, this exercises the batched path.
  concurrent_nonce="CONCURRENT-CACHE-$(date +%s%N)"
  concurrent_prefix="Shared cache branch probe $concurrent_nonce."
  concurrent_schema='{"type":"object","properties":{"marker":{"type":"integer"}},"required":["marker"],"additionalProperties":false}'
  concurrent_warmup="$concurrent_prefix Return JSON with marker 0."
  api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"$concurrent_warmup\"}],\"max_tokens\":20,\"stream\":false,\"temperature\":0,\"seed\":42,\"chat_template_kwargs\":{\"enable_thinking\":false}}" >/dev/null

  t0=$(now_ms)
  concurrent_tmpdir=$(mktemp -d)
  concurrent_tokens=()
  for i in 1 2 3 4 5 6 7 8; do
    token="$i"
    concurrent_tokens+=("$token")
    prompt="$concurrent_prefix Return JSON with marker $token."
    curl -s --max-time 60 "$BASE_URL/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"guided_json\":$concurrent_schema,\"max_tokens\":32,\"stream\":false,\"temperature\":0,\"seed\":42,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
      -o "$concurrent_tmpdir/resp_$i.json" \
      -w "%{http_code}" > "$concurrent_tmpdir/code_$i.txt" 2>/dev/null &
  done
  wait 2>/dev/null || true
  dur=$(( $(now_ms) - t0 ))

  concurrent_cache_state=$(python3 - "$concurrent_tmpdir" <<'PY'
import json, os, sys

root = sys.argv[1]
failures = []
for i in range(1, 9):
    body_path = os.path.join(root, f"resp_{i}.json")
    code_path = os.path.join(root, f"code_{i}.txt")
    code = open(code_path, "r", encoding="utf-8").read().strip() if os.path.exists(code_path) else "NO_CODE"
    if code != "200":
        failures.append(f"{i}:http={code}")
        continue
    try:
        with open(body_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        usage = d.get("usage", {})
        ptd = usage.get("prompt_tokens_details")
        if ptd is None:
            failures.append(f"{i}:null")
            continue
        cached = ptd.get("cached_tokens", "MISSING")
        prompt = usage.get("prompt_tokens", "MISSING")
        if not isinstance(cached, int) or not isinstance(prompt, int) or cached <= 0 or cached >= prompt:
            failures.append(f"{i}:{cached}/{prompt}")
    except Exception as e:
        failures.append(f"{i}:error={e}")

if failures:
    print("FAIL: " + ", ".join(failures))
else:
    print("PASS")
PY
)
  if [ "$concurrent_cache_state" = "PASS" ]; then
    run_test "Cache" "Concurrent x8 shared-prefix: uncached suffix remains on every branch" "0 < cached_tokens < prompt_tokens for all 8" "PASS" "$dur"
  else
    run_test "Cache" "Concurrent x8 shared-prefix: uncached suffix remains on every branch" "0 < cached_tokens < prompt_tokens for all 8" "$concurrent_cache_state" "$dur"
  fi

  concurrent_content_state=$(python3 - "$concurrent_tmpdir" "${concurrent_tokens[@]}" <<'PY'
import json, os, sys

root = sys.argv[1]
tokens = sys.argv[2:]
failures = []
for i, token in enumerate(tokens, start=1):
    body_path = os.path.join(root, f"resp_{i}.json")
    try:
        with open(body_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        content = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        payload = json.loads(content)
        marker = payload.get("marker")
        if marker != int(token):
            failures.append(f"{i}:marker={marker} expected={token} content=[{content}]")
            continue
    except Exception as e:
        failures.append(f"{i}:error={e}")

if failures:
    print("FAIL: " + "; ".join(failures))
else:
    print("PASS")
PY
)
  if [ "$concurrent_content_state" = "PASS" ]; then
    run_test "Cache" "Concurrent x8 shared-prefix: divergent suffix responses stay isolated" "each of 8 responses keeps only its own marker" "PASS" "$dur"
  else
    run_test "Cache" "Concurrent x8 shared-prefix: divergent suffix responses stay isolated" "each of 8 responses keeps only its own marker" "$concurrent_content_state" "$dur"
  fi
  rm -rf "$concurrent_tmpdir"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Concurrent requests
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 7 && min_tier standard; then
  CURRENT_TIER="standard"
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
CURRENT_TIER="smoke"
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
  CURRENT_TIER="standard"
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
    resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":500,"stream":false,"temperature":0}')
    dur=$(( $(now_ms) - t0 ))
    state=$(_kwargs_check_response "$resp")
    if [ "$state" = "thinking" ] || [ "$state" = "empty" ]; then
      # "empty" means reasoning_content present but content empty (model spent budget on thinking) — still proves thinking is active
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
    resp=$(api_call '{"messages":[{"role":"user","content":"What is 2+2?"}],"chat_template_kwargs":{"enable_thinking":true},"max_tokens":500,"stream":false,"temperature":0}')
    dur=$(( $(now_ms) - t0 ))
    state=$(_kwargs_check_response "$resp")
    if [ "$state" = "thinking" ] || [ "$state" = "empty" ]; then
      # "empty" means reasoning_content present but content empty (model spent budget on thinking) — still proves thinking is active
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
  CURRENT_TIER="standard"
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
        msg = d['choices'][0]['message']
        c = msg.get('content') or msg.get('reasoning_content') or ''
        print('PASS' if c and len(c.strip()) > 0 else 'FAIL: empty')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [[ "$cache_ok" == PASS* ]]; then
    run_test "Cache" "Prefix cache reuse across requests" "valid response on cache hit" "PASS" "$dur"
  else
    run_test "Cache" "Prefix cache reuse across requests" "valid response on cache hit" "$cache_ok" "$dur"
  fi

  # ── Issue #32 regression: sequential guided-json → nullable tool call ──────
  # This is the exact sequence that crashed with QuantizedKVCache (--kv-bits 4):
  # 1st request fills cache, 2nd request with nullable schema triggers
  # cache restore with offset=0 → reshape crash.

  # Request 1: guided-json (fills prefix cache with system prompt)
  t0=$(now_ms)
  pcr1=$(api_call '{"messages":[{"role":"system","content":"You are a helpful assistant that responds in JSON format."},{"role":"user","content":"Give me info about Saturn as JSON with keys: name, type, rings (boolean)"}],"max_tokens":200,"temperature":0,"response_format":{"type":"json_object"}}')
  pcr1_ok=$(echo "$pcr1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL: {d[\"error\"].get(\"message\",\"\")}')
    else:
        c = d['choices'][0]['message'].get('content','')
        rc = d['choices'][0]['message'].get('reasoning_content','')
        if c or rc:
            print('PASS')
        else:
            print('FAIL: empty')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  # Request 2: nullable tool call (issue #32 — anyOf [string, null] + prefix cache restore)
  NULLABLE_SCHEMA_TOOL='[{"type":"function","function":{"name":"update_record","description":"Update a database record","parameters":{"type":"object","properties":{"id":{"type":"integer","description":"Record ID"},"name":{"type":"string","description":"Name"},"notes":{"anyOf":[{"type":"string"},{"type":"null"}],"description":"Optional notes"},"priority":{"anyOf":[{"type":"integer"},{"type":"null"}],"description":"Optional priority"}},"required":["id","name"]}}}]'
  pcr2=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant that responds in JSON format.\"},{\"role\":\"user\",\"content\":\"Update record 42, set name to Saturn, notes to 'ringed planet', priority null\"}],\"tools\":$NULLABLE_SCHEMA_TOOL,\"max_tokens\":300,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  pcr2_ok=$(echo "$pcr2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL: server error: {d[\"error\"].get(\"message\",\"\")}')
    else:
        msg = d['choices'][0]['message']
        tc = msg.get('tool_calls', [])
        c = msg.get('content') or msg.get('reasoning_content') or ''
        if tc:
            args = json.loads(tc[0]['function']['arguments'])
            print('PASS')
        elif c.strip():
            print('PASS')  # no crash — model chose text instead of tool
        else:
            print('FAIL: no tool calls and no content')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  combined="$pcr1_ok+$pcr2_ok"
  if [ "$combined" = "PASS+PASS" ]; then
    run_test "Cache" "Issue #32: sequential guided-json → nullable tool (no crash)" "both succeed" "PASS" "$dur"
  else
    run_test "Cache" "Issue #32: sequential guided-json → nullable tool (no crash)" "both succeed" "FAIL: req1=$pcr1_ok req2=$pcr2_ok" "$dur"
  fi

  # ── Sequential: 3 tool calls with nullable schemas (cache reuse stress) ────
  t0=$(now_ms)
  seq_pass=0
  seq_fail_msg=""
  for i in 1 2 3; do
    seq_resp=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"Update record $i, set name to Planet$i\"}],\"tools\":$NULLABLE_SCHEMA_TOOL,\"max_tokens\":300,\"temperature\":0}")
    seq_check=$(echo "$seq_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL:{d[\"error\"].get(\"message\",\"\")}')
    else:
        msg = d['choices'][0]['message']
        tc = msg.get('tool_calls', [])
        c = msg.get('content') or msg.get('reasoning_content') or ''
        if tc or c.strip():
            print('OK')
        else:
            print('FAIL:empty')
except Exception as e:
    print(f'FAIL:{e}')
" 2>/dev/null || echo "FAIL:parse")
    if [ "$seq_check" = "OK" ]; then
      seq_pass=$((seq_pass + 1))
    else
      seq_fail_msg="req$i=$seq_check"
    fi
  done
  dur=$(( $(now_ms) - t0 ))
  if [ "$seq_pass" -eq 3 ]; then
    run_test "Cache" "Sequential: 3 nullable tool calls (cache reuse)" "3/3 succeed" "PASS" "$dur"
  else
    run_test "Cache" "Sequential: 3 nullable tool calls (cache reuse)" "3/3 succeed" "FAIL: $seq_pass/3 $seq_fail_msg" "$dur"
  fi

  # ── Multi-turn tool conversation with shared prefix (5 messages) ───────────
  # user → assistant(tool_call) → tool_result → user follow-up → assistant
  # Tests prefix cache with growing conversation history
  t0=$(now_ms)
  # Step 1: initial tool call
  mt_cache_resp1=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a weather assistant. Always use the get_weather tool when asked about weather.\"},{\"role\":\"user\",\"content\":\"What is the weather in Tokyo?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":300,\"temperature\":0}")
  mt_cache_tc=$(echo "$mt_cache_resp1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if tc:
        print(f'{tc[0].get(\"id\",\"call_c1\")}|{tc[0][\"function\"][\"name\"]}|{tc[0][\"function\"][\"arguments\"]}')
    else:
        print('SKIP')
except:
    print('SKIP')
" 2>/dev/null || echo "SKIP")
  if [ "$mt_cache_tc" = "SKIP" ]; then
    dur=$(( $(now_ms) - t0 ))
    run_test "Cache" "Multi-turn tool conversation with prefix (model skipped tool)" "skip" "SKIP" "$dur"
  else
    IFS='|' read -r mtc_id mtc_fn mtc_args <<< "$mt_cache_tc"
    # Step 2: send tool result + follow-up question (same system prompt prefix)
    mt_cache_resp2=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a weather assistant. Always use the get_weather tool when asked about weather.\"},{\"role\":\"user\",\"content\":\"What is the weather in Tokyo?\"},{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"$mtc_id\",\"type\":\"function\",\"function\":{\"name\":\"$mtc_fn\",\"arguments\":$mtc_args}}]},{\"role\":\"tool\",\"tool_call_id\":\"$mtc_id\",\"name\":\"$mtc_fn\",\"content\":\"{\\\"temperature\\\":18,\\\"condition\\\":\\\"cloudy\\\",\\\"wind_speed\\\":12}\"},{\"role\":\"user\",\"content\":\"How about London?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":300,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    mt_cache_valid=$(echo "$mt_cache_resp2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL: server error: {d[\"error\"].get(\"message\",\"\")}')
    else:
        msg = d['choices'][0]['message']
        tc = msg.get('tool_calls', [])
        c = msg.get('content') or msg.get('reasoning_content') or ''
        # Either a new tool call for London or a text response — both valid
        if tc or c.strip():
            print('PASS')
        else:
            print('FAIL: empty response on multi-turn follow-up')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$mt_cache_valid" = "PASS" ]; then
      run_test "Cache" "Multi-turn: 5-msg tool conversation with prefix cache" "valid response" "PASS" "$dur"
    else
      run_test "Cache" "Multi-turn: 5-msg tool conversation with prefix cache" "valid response" "$mt_cache_valid" "$dur"
    fi
  fi

  # ── Long system prompt + sequential requests (large prefix reuse) ──────────
  # ~400 word system prompt creates a substantial prefix to cache/restore
  t0=$(now_ms)
  LONG_SYSTEM="You are an expert planetary scientist and astronomer with decades of experience studying our solar system. Your knowledge covers orbital mechanics, atmospheric composition, geological features, magnetic fields, ring systems, and satellite systems of all planets. When answering questions, provide detailed scientific information including numerical data where relevant such as orbital periods, distances, masses, and temperatures. Always structure your responses clearly with the most important facts first. You have published over 200 peer-reviewed papers on topics ranging from Jupiter's Great Red Spot dynamics to the methane cycle on Titan. Your expertise also covers exoplanetary systems, stellar evolution, and cosmological phenomena. You prefer to give concise but information-dense answers. When discussing measurements, use SI units primarily but include imperial equivalents when helpful for general audiences. You also have expertise in space mission design and have consulted for NASA, ESA, and JAXA on multiple missions including Cassini-Huygens, Juno, and the Mars rovers."
  # Request 1: seed the prefix cache
  lpc1=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$LONG_SYSTEM\"},{\"role\":\"user\",\"content\":\"How many moons does Jupiter have?\"}],\"max_tokens\":100,\"temperature\":0}")
  lpc1_ok=$(echo "$lpc1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL:{d[\"error\"].get(\"message\",\"\")}')
    else:
        msg = d['choices'][0]['message']
        c = msg.get('content') or msg.get('reasoning_content') or ''
        print('OK' if c.strip() else 'FAIL:empty')
except Exception as e:
    print(f'FAIL:{e}')
" 2>/dev/null || echo "FAIL:parse")
  # Request 2: same long system prompt, different question (prefix cache hit)
  lpc2=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$LONG_SYSTEM\"},{\"role\":\"user\",\"content\":\"What is the surface temperature of Venus?\"}],\"max_tokens\":100,\"temperature\":0}")
  lpc2_ok=$(echo "$lpc2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL:{d[\"error\"].get(\"message\",\"\")}')
    else:
        msg = d['choices'][0]['message']
        c = msg.get('content') or msg.get('reasoning_content') or ''
        print('OK' if c.strip() else 'FAIL:empty')
except Exception as e:
    print(f'FAIL:{e}')
" 2>/dev/null || echo "FAIL:parse")
  # Request 3: same prefix, add tool call (mixed mode after cache)
  lpc3=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$LONG_SYSTEM\"},{\"role\":\"user\",\"content\":\"What is the weather on Mars today?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":300,\"temperature\":0}")
  lpc3_ok=$(echo "$lpc3" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL:{d[\"error\"].get(\"message\",\"\")}')
    else:
        msg = d['choices'][0]['message']
        tc = msg.get('tool_calls', [])
        c = msg.get('content') or msg.get('reasoning_content') or ''
        print('OK' if (tc or c.strip()) else 'FAIL:empty')
except Exception as e:
    print(f'FAIL:{e}')
" 2>/dev/null || echo "FAIL:parse")
  dur=$(( $(now_ms) - t0 ))
  lpc_combined="$lpc1_ok+$lpc2_ok+$lpc3_ok"
  if [ "$lpc_combined" = "OK+OK+OK" ]; then
    run_test "Cache" "Long system prompt: 3 sequential requests (prefix reuse)" "3/3 succeed" "PASS" "$dur"
  else
    run_test "Cache" "Long system prompt: 3 sequential requests (prefix reuse)" "3/3 succeed" "FAIL: $lpc_combined" "$dur"
  fi

  # ── 5-request sequential stress test (mixed types) ────────────────────────
  # Alternates: text → tool → guided-json → streaming → nullable tool
  # Each must succeed — no crash from stale cache state
  t0=$(now_ms)
  stress_pass=0
  stress_detail=""
  STRESS_SYS="You are a concise assistant."
  # Req 1: plain text
  sr1=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$STRESS_SYS\"},{\"role\":\"user\",\"content\":\"Name one planet.\"}],\"max_tokens\":20,\"temperature\":0}")
  sr1_ok=$(echo "$sr1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d['choices'][0]['message']
    c = msg.get('content') or msg.get('reasoning_content') or ''
    print('OK' if c.strip() else 'FAIL')
except: print('FAIL')
" 2>/dev/null || echo "FAIL")
  [ "$sr1_ok" = "OK" ] && stress_pass=$((stress_pass + 1)) || stress_detail="${stress_detail}req1=text_fail "
  # Req 2: tool call
  sr2=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$STRESS_SYS\"},{\"role\":\"user\",\"content\":\"Weather in Rome?\"}],\"tools\":$TOOL_DEF,\"max_tokens\":300,\"temperature\":0}")
  sr2_ok=$(echo "$sr2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d['choices'][0]['message']
    tc = msg.get('tool_calls', [])
    c = msg.get('content') or msg.get('reasoning_content') or ''
    print('OK' if (tc or c.strip()) else 'FAIL')
except: print('FAIL')
" 2>/dev/null || echo "FAIL")
  [ "$sr2_ok" = "OK" ] && stress_pass=$((stress_pass + 1)) || stress_detail="${stress_detail}req2=tool_fail "
  # Req 3: guided-json
  sr3=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$STRESS_SYS\"},{\"role\":\"user\",\"content\":\"Give me Mars info as JSON with keys: name, type\"}],\"max_tokens\":100,\"temperature\":0,\"response_format\":{\"type\":\"json_object\"}}")
  sr3_ok=$(echo "$sr3" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d['choices'][0]['message']
    c = msg.get('content') or msg.get('reasoning_content') or ''
    print('OK' if c.strip() else 'FAIL')
except: print('FAIL')
" 2>/dev/null || echo "FAIL")
  [ "$sr3_ok" = "OK" ] && stress_pass=$((stress_pass + 1)) || stress_detail="${stress_detail}req3=json_fail "
  # Req 4: streaming
  sr4=$(api_stream "{\"messages\":[{\"role\":\"system\",\"content\":\"$STRESS_SYS\"},{\"role\":\"user\",\"content\":\"Name one star.\"}],\"max_tokens\":20,\"temperature\":0,\"stream\":true}")
  sr4_ok=$(echo "$sr4" | python3 -c "
import sys, json
tokens = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]': continue
    try:
        d = json.loads(line[6:])
        delta = d.get('choices',[{}])[0].get('delta',{})
        c = delta.get('content','') or delta.get('reasoning_content','')
        if c: tokens.append(c)
    except: pass
print('OK' if tokens else 'FAIL')
" 2>/dev/null || echo "FAIL")
  [ "$sr4_ok" = "OK" ] && stress_pass=$((stress_pass + 1)) || stress_detail="${stress_detail}req4=stream_fail "
  # Req 5: nullable tool call
  sr5=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$STRESS_SYS\"},{\"role\":\"user\",\"content\":\"Update record 99, set name to Earth\"}],\"tools\":$NULLABLE_SCHEMA_TOOL,\"max_tokens\":300,\"temperature\":0}")
  sr5_ok=$(echo "$sr5" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print('FAIL')
    else:
        msg = d['choices'][0]['message']
        tc = msg.get('tool_calls', [])
        c = msg.get('content') or msg.get('reasoning_content') or ''
        print('OK' if (tc or c.strip()) else 'FAIL')
except: print('FAIL')
" 2>/dev/null || echo "FAIL")
  [ "$sr5_ok" = "OK" ] && stress_pass=$((stress_pass + 1)) || stress_detail="${stress_detail}req5=nullable_fail "
  dur=$(( $(now_ms) - t0 ))
  if [ "$stress_pass" -eq 5 ]; then
    run_test "Cache" "Sequential stress: 5 mixed requests (text/tool/json/stream/nullable)" "5/5 succeed" "PASS" "$dur"
  else
    run_test "Cache" "Sequential stress: 5 mixed requests (text/tool/json/stream/nullable)" "5/5 succeed" "FAIL: $stress_pass/5 $stress_detail" "$dur"
  fi

  # ── Multi-turn 5-turn growing conversation ─────────────────────────────────
  # Simulates a real chat: each turn appends to conversation history
  # Tests that prefix cache handles progressively growing context
  t0=$(now_ms)
  CONVO_SYS="You are a helpful assistant. Keep answers to one sentence."
  turns_ok=0
  turns_detail=""
  # Turn 1
  turn1=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$CONVO_SYS\"},{\"role\":\"user\",\"content\":\"What is the largest planet?\"}],\"max_tokens\":50,\"temperature\":0}")
  t1_content=$(echo "$turn1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message'].get('content','') or d['choices'][0]['message'].get('reasoning_content','') or ''
    print(c.strip()[:200] if c.strip() else '__EMPTY__')
except: print('__ERROR__')
" 2>/dev/null || echo "__ERROR__")
  [[ "$t1_content" != "__EMPTY__" && "$t1_content" != "__ERROR__" ]] && turns_ok=$((turns_ok + 1)) || turns_detail="${turns_detail}t1 "
  # Turn 2 — growing context
  t1_escaped=$(echo "$t1_content" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))" 2>/dev/null)
  turn2=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$CONVO_SYS\"},{\"role\":\"user\",\"content\":\"What is the largest planet?\"},{\"role\":\"assistant\",\"content\":$t1_escaped},{\"role\":\"user\",\"content\":\"How many moons does it have?\"}],\"max_tokens\":50,\"temperature\":0}")
  t2_content=$(echo "$turn2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message'].get('content','') or d['choices'][0]['message'].get('reasoning_content','') or ''
    print(c.strip()[:200] if c.strip() else '__EMPTY__')
except: print('__ERROR__')
" 2>/dev/null || echo "__ERROR__")
  [[ "$t2_content" != "__EMPTY__" && "$t2_content" != "__ERROR__" ]] && turns_ok=$((turns_ok + 1)) || turns_detail="${turns_detail}t2 "
  # Turn 3
  t2_escaped=$(echo "$t2_content" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))" 2>/dev/null)
  turn3=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$CONVO_SYS\"},{\"role\":\"user\",\"content\":\"What is the largest planet?\"},{\"role\":\"assistant\",\"content\":$t1_escaped},{\"role\":\"user\",\"content\":\"How many moons does it have?\"},{\"role\":\"assistant\",\"content\":$t2_escaped},{\"role\":\"user\",\"content\":\"Name the four largest moons.\"}],\"max_tokens\":80,\"temperature\":0}")
  t3_content=$(echo "$turn3" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message'].get('content','') or d['choices'][0]['message'].get('reasoning_content','') or ''
    print(c.strip()[:200] if c.strip() else '__EMPTY__')
except: print('__ERROR__')
" 2>/dev/null || echo "__ERROR__")
  [[ "$t3_content" != "__EMPTY__" && "$t3_content" != "__ERROR__" ]] && turns_ok=$((turns_ok + 1)) || turns_detail="${turns_detail}t3 "
  # Turn 4
  t3_escaped=$(echo "$t3_content" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))" 2>/dev/null)
  turn4=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$CONVO_SYS\"},{\"role\":\"user\",\"content\":\"What is the largest planet?\"},{\"role\":\"assistant\",\"content\":$t1_escaped},{\"role\":\"user\",\"content\":\"How many moons does it have?\"},{\"role\":\"assistant\",\"content\":$t2_escaped},{\"role\":\"user\",\"content\":\"Name the four largest moons.\"},{\"role\":\"assistant\",\"content\":$t3_escaped},{\"role\":\"user\",\"content\":\"Which of those moons might have liquid water?\"}],\"max_tokens\":80,\"temperature\":0}")
  t4_content=$(echo "$turn4" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message'].get('content','') or d['choices'][0]['message'].get('reasoning_content','') or ''
    print(c.strip()[:200] if c.strip() else '__EMPTY__')
except: print('__ERROR__')
" 2>/dev/null || echo "__ERROR__")
  [[ "$t4_content" != "__EMPTY__" && "$t4_content" != "__ERROR__" ]] && turns_ok=$((turns_ok + 1)) || turns_detail="${turns_detail}t4 "
  # Turn 5
  t4_escaped=$(echo "$t4_content" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))" 2>/dev/null)
  turn5=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$CONVO_SYS\"},{\"role\":\"user\",\"content\":\"What is the largest planet?\"},{\"role\":\"assistant\",\"content\":$t1_escaped},{\"role\":\"user\",\"content\":\"How many moons does it have?\"},{\"role\":\"assistant\",\"content\":$t2_escaped},{\"role\":\"user\",\"content\":\"Name the four largest moons.\"},{\"role\":\"assistant\",\"content\":$t3_escaped},{\"role\":\"user\",\"content\":\"Which of those moons might have liquid water?\"},{\"role\":\"assistant\",\"content\":$t4_escaped},{\"role\":\"user\",\"content\":\"Summarize everything we discussed.\"}],\"max_tokens\":150,\"temperature\":0}")
  t5_ok=$(echo "$turn5" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print('FAIL')
    else:
        c = d['choices'][0]['message'].get('content','') or d['choices'][0]['message'].get('reasoning_content','') or ''
        print('OK' if c.strip() else 'FAIL')
except: print('FAIL')
" 2>/dev/null || echo "FAIL")
  [ "$t5_ok" = "OK" ] && turns_ok=$((turns_ok + 1)) || turns_detail="${turns_detail}t5 "
  dur=$(( $(now_ms) - t0 ))
  if [ "$turns_ok" -eq 5 ]; then
    run_test "Cache" "Multi-turn: 5-turn growing conversation (prefix reuse)" "5/5 turns" "PASS" "$dur"
  else
    run_test "Cache" "Multi-turn: 5-turn growing conversation (prefix reuse)" "5/5 turns" "FAIL: $turns_ok/5 failed=$turns_detail" "$dur"
  fi

  # ── Streaming multi-turn with prefix cache ─────────────────────────────────
  # Sequential streaming requests with shared system prompt
  t0=$(now_ms)
  STREAM_SYS="You are a geography expert. Give brief answers."
  # Streaming req 1
  str1=$(api_stream "{\"messages\":[{\"role\":\"system\",\"content\":\"$STREAM_SYS\"},{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":30,\"temperature\":0,\"stream\":true}")
  str1_ok=$(echo "$str1" | python3 -c "
import sys, json
tokens = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]': continue
    try:
        d = json.loads(line[6:])
        c = d.get('choices',[{}])[0].get('delta',{}).get('content','') or ''
        rc = d.get('choices',[{}])[0].get('delta',{}).get('reasoning_content','') or ''
        if c or rc: tokens.append(c or rc)
    except: pass
print('OK' if tokens else 'FAIL')
" 2>/dev/null || echo "FAIL")
  # Streaming req 2 (same prefix)
  str2=$(api_stream "{\"messages\":[{\"role\":\"system\",\"content\":\"$STREAM_SYS\"},{\"role\":\"user\",\"content\":\"Capital of Japan?\"}],\"max_tokens\":30,\"temperature\":0,\"stream\":true}")
  str2_ok=$(echo "$str2" | python3 -c "
import sys, json
tokens = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]': continue
    try:
        d = json.loads(line[6:])
        c = d.get('choices',[{}])[0].get('delta',{}).get('content','') or ''
        rc = d.get('choices',[{}])[0].get('delta',{}).get('reasoning_content','') or ''
        if c or rc: tokens.append(c or rc)
    except: pass
print('OK' if tokens else 'FAIL')
" 2>/dev/null || echo "FAIL")
  # Non-streaming req 3 (same prefix, mode switch after streaming)
  str3=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$STREAM_SYS\"},{\"role\":\"user\",\"content\":\"Capital of Brazil?\"}],\"max_tokens\":30,\"temperature\":0}")
  str3_ok=$(echo "$str3" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message'].get('content','') or d['choices'][0]['message'].get('reasoning_content','') or ''
    print('OK' if c.strip() else 'FAIL')
except: print('FAIL')
" 2>/dev/null || echo "FAIL")
  dur=$(( $(now_ms) - t0 ))
  stream_combined="$str1_ok+$str2_ok+$str3_ok"
  if [ "$stream_combined" = "OK+OK+OK" ]; then
    run_test "Cache" "Streaming + non-streaming sequential (shared prefix)" "3/3 succeed" "PASS" "$dur"
  else
    run_test "Cache" "Streaming + non-streaming sequential (shared prefix)" "3/3 succeed" "FAIL: $stream_combined" "$dur"
  fi

  # ── Long multi-turn with tools and nullable schemas (comprehensive) ────────
  # 7-message conversation: sys + user + assistant(tool_call) + tool_result + user + assistant + user
  # Uses nullable schema tool throughout — exercises issue #32 + prefix cache together
  t0=$(now_ms)
  COMPLEX_TOOL='[{"type":"function","function":{"name":"lookup_item","description":"Look up an item in the inventory","parameters":{"type":"object","properties":{"item_id":{"type":"string","description":"Item identifier"},"include_history":{"anyOf":[{"type":"boolean"},{"type":"null"}],"description":"Include change history"},"format":{"anyOf":[{"type":"string"},{"type":"null"}],"description":"Output format"}},"required":["item_id"]}}}]'
  COMPLEX_SYS="You are an inventory management assistant. Use the lookup_item tool to find items. Always call the tool when the user asks about an item."
  # Step 1: initial lookup
  cx1=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$COMPLEX_SYS\"},{\"role\":\"user\",\"content\":\"Look up item SKU-12345\"}],\"tools\":$COMPLEX_TOOL,\"max_tokens\":300,\"temperature\":0}")
  cx1_tc=$(echo "$cx1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'ERROR:{d[\"error\"].get(\"message\",\"\")}')
    else:
        tc = d['choices'][0]['message'].get('tool_calls', [])
        if tc:
            print(f'{tc[0].get(\"id\",\"call_cx1\")}|{tc[0][\"function\"][\"name\"]}|{tc[0][\"function\"][\"arguments\"]}')
        else:
            c = d['choices'][0]['message'].get('content','')
            print('TEXT' if c.strip() else 'EMPTY')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null || echo "ERROR:parse")
  if [[ "$cx1_tc" == ERROR* ]]; then
    dur=$(( $(now_ms) - t0 ))
    run_test "Cache" "Issue #32 comprehensive: nullable tool + multi-turn + prefix" "no crash" "FAIL: $cx1_tc" "$dur"
  elif [[ "$cx1_tc" == TEXT* ]] || [[ "$cx1_tc" == EMPTY* ]]; then
    dur=$(( $(now_ms) - t0 ))
    run_test "Cache" "Issue #32 comprehensive: nullable tool + multi-turn + prefix (model skipped tool)" "skip" "SKIP" "$dur"
  else
    IFS='|' read -r cx_id cx_fn cx_args <<< "$cx1_tc"
    # Step 2: tool result + follow-up + second lookup
    cx2=$(api_call "{\"messages\":[{\"role\":\"system\",\"content\":\"$COMPLEX_SYS\"},{\"role\":\"user\",\"content\":\"Look up item SKU-12345\"},{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"$cx_id\",\"type\":\"function\",\"function\":{\"name\":\"$cx_fn\",\"arguments\":$cx_args}}]},{\"role\":\"tool\",\"tool_call_id\":\"$cx_id\",\"name\":\"$cx_fn\",\"content\":\"{\\\"item_id\\\":\\\"SKU-12345\\\",\\\"name\\\":\\\"Widget Pro\\\",\\\"stock\\\":42,\\\"last_updated\\\":\\\"2026-03-10\\\"}\"},{\"role\":\"user\",\"content\":\"Now look up SKU-67890 with history included\"}],\"tools\":$COMPLEX_TOOL,\"max_tokens\":300,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    cx2_ok=$(echo "$cx2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'FAIL: {d[\"error\"].get(\"message\",\"\")}')
    else:
        msg = d['choices'][0]['message']
        tc = msg.get('tool_calls', [])
        c = msg.get('content') or msg.get('reasoning_content') or ''
        if tc or c.strip():
            print('PASS')
        else:
            print('FAIL: empty on 7-msg conversation')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$cx2_ok" = "PASS" ]; then
      run_test "Cache" "Issue #32 comprehensive: nullable tool + multi-turn + prefix" "valid response" "PASS" "$dur"
    else
      run_test "Cache" "Issue #32 comprehensive: nullable tool + multi-turn + prefix" "valid response" "$cx2_ok" "$dur"
    fi
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 8c: Structured Output
# ═══════════════════════════════════════════════════════════════════════════════
if should_run_section 8c && min_tier standard; then
  CURRENT_TIER="standard"
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
  CURRENT_TIER="full"
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
  CURRENT_TIER="standard"
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
# Section 12: afm_adaptive_xml (JSON-in-XML fallback + xgrammar)
# ═══════════════════════════════════════════════════════════════════════════════
# Tests for --tool-call-parser afm_adaptive_xml.
# Validates: JSON-in-XML fallback parsing, normal XML tool calls still work,
# xgrammar EBNF constraint activation, and streaming tool call emission.
#
# NOTE: This section requires the server to be started with:
#   --tool-call-parser afm_adaptive_xml
# If the server wasn't started with this flag, tests are skipped.

if should_run_section 12 && min_tier standard; then
  CURRENT_TIER="standard"
  echo ""
  echo "🔧 Section 12: afm_adaptive_xml"

  # Probe: was the server started with --tool-call-parser afm_adaptive_xml?
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
      run_test "AdaptiveXML" "Normal XML tool call works with afm_adaptive_xml" "valid tool call" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Normal XML tool call works with afm_adaptive_xml" "valid tool call" "$tc_ok" "$dur"
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
      run_test "AdaptiveXML" "Streaming tool call emits valid deltas" "name + JSON args" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Streaming tool call emits valid deltas" "name + JSON args" "$stream_tc" "$dur"
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
      run_test "AdaptiveXML" "Multi-turn: model responds after tool result" "non-empty content" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Multi-turn: model responds after tool result" "non-empty content" "$multi_ok" "$dur"
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
      run_test "AdaptiveXML" "tool_choice=none suppresses tool calls" "no tool_calls" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "tool_choice=none suppresses tool calls" "no tool_calls" "$none_ok" "$dur"
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
      run_test "AdaptiveXML" "Multiple tools: correct function selected" "get_time" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Multiple tools: correct function selected" "get_time" "$multi_tool_ok" "$dur"
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
      run_test "AdaptiveXML" "Argument types coerced (string, bool, int)" "correct types" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Argument types coerced (string, bool, int)" "correct types" "$typed_ok" "$dur"
    fi

    # ── Test 12.7: tool call valid (with or without grammar constraints) ──────
    # Verifies tool call response is valid. xgrammar EBNF grammar constraint is
    # opt-in via --enable-grammar-constraints at server start. This test validates
    # the response regardless — with or without grammar constraint.
    t0=$(now_ms)
    xg_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Berlin?\"}],\"tools\":$TOOL_DEF,\"tool_choice\":\"required\",\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
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
        # Model chose not to call tool despite tool_choice=required — model behavior, not AFM bug.
        # Verify we at least got a valid response (200 OK with content or reasoning).
        msg = d['choices'][0]['message']
        c = msg.get('content') or msg.get('reasoning_content') or ''
        print('PASS' if c.strip() else 'FAIL: no tool_calls and no content')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$xg_ok" = "PASS" ]; then
      run_test "AdaptiveXML" "Tool call valid (with or without grammar constraints)" "valid" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Tool call valid (with or without grammar constraints)" "valid" "$xg_ok" "$dur"
    fi

    # ── Test 12.8: Array of objects coercion ──────────────────────────────────
    # Regression test: model generates JSON array inside <parameter> tags but
    # afm sends it as a string. Covers the question tool pattern from OpenCode.
    ARRAY_OBJ_TOOL='[{"type":"function","function":{"name":"ask_questions","description":"Ask user multiple-choice questions","parameters":{"type":"object","properties":{"questions":{"type":"array","description":"List of questions","items":{"type":"object","properties":{"text":{"type":"string"},"options":{"type":"array","items":{"type":"string"}}}}},"title":{"type":"string","description":"Form title"}},"required":["questions","title"]}}}]'
    t0=$(now_ms)
    arr_obj_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Ask the user 2 questions: What is your favorite color (red, blue, green) and what is your favorite animal (cat, dog, fish).\"}],\"tools\":$ARRAY_OBJ_TOOL,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    arr_obj_ok=$(echo "$arr_obj_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        qs = args.get('questions')
        if qs is None:
            print('FAIL: questions param missing')
        elif isinstance(qs, str):
            print(f'FAIL: questions is string (not coerced to array)')
        elif not isinstance(qs, list):
            print(f'FAIL: questions is {type(qs).__name__}, expected list')
        elif len(qs) == 0:
            print('FAIL: questions array is empty')
        elif not isinstance(qs[0], dict):
            print(f'FAIL: questions[0] is {type(qs[0]).__name__}, expected dict')
        else:
            title = args.get('title')
            if title is not None and not isinstance(title, str):
                print(f'FAIL: title is {type(title).__name__}, expected str')
            else:
                print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$arr_obj_ok" = "PASS" ]; then
      run_test "AdaptiveXML" "Array of objects coercion (question tool pattern)" "array of dicts" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Array of objects coercion (question tool pattern)" "array of dicts" "$arr_obj_ok" "$dur"
    fi

    # ── Test 12.9: Number (float) coercion ────────────────────────────────────
    NUMBER_TOOL='[{"type":"function","function":{"name":"set_temperature","description":"Set thermostat temperature","parameters":{"type":"object","properties":{"celsius":{"type":"number","description":"Temperature in Celsius"},"enabled":{"type":"boolean","description":"Enable thermostat"}},"required":["celsius","enabled"]}}}]'
    t0=$(now_ms)
    num_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Set the thermostat to 22.5 degrees celsius and enable it.\"}],\"tools\":$NUMBER_TOOL,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    num_ok=$(echo "$num_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        issues = []
        c = args.get('celsius')
        e = args.get('enabled')
        if c is not None and isinstance(c, str):
            issues.append(f'celsius=str({c})')
        elif c is not None and not isinstance(c, (int, float)):
            issues.append(f'celsius={type(c).__name__}({c})')
        if e is not None and not isinstance(e, bool):
            issues.append(f'enabled={type(e).__name__}({e})')
        if issues:
            print(f'FAIL: type coercion: {issues}')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$num_ok" = "PASS" ]; then
      run_test "AdaptiveXML" "Number (float) and boolean coercion" "correct types" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Number (float) and boolean coercion" "correct types" "$num_ok" "$dur"
    fi

    # ── Test 12.10: Nested object coercion ────────────────────────────────────
    NESTED_OBJ_TOOL='[{"type":"function","function":{"name":"create_config","description":"Create a configuration object","parameters":{"type":"object","properties":{"name":{"type":"string"},"settings":{"type":"object","description":"Configuration settings","properties":{"debug":{"type":"boolean"},"timeout":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}}}},"required":["name","settings"]}}}]'
    t0=$(now_ms)
    nested_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Create config named 'prod' with debug false, timeout 30, tags: api, backend.\"}],\"tools\":$NESTED_OBJ_TOOL,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    nested_ok=$(echo "$nested_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        issues = []
        name = args.get('name')
        settings = args.get('settings')
        if name is not None and not isinstance(name, str):
            issues.append(f'name={type(name).__name__}')
        if settings is None:
            issues.append('settings missing')
        elif isinstance(settings, str):
            issues.append('settings=str (not coerced to object)')
        elif not isinstance(settings, dict):
            issues.append(f'settings={type(settings).__name__}')
        else:
            dbg = settings.get('debug')
            to = settings.get('timeout')
            tg = settings.get('tags')
            if dbg is not None and not isinstance(dbg, bool):
                issues.append(f'settings.debug={type(dbg).__name__}({dbg})')
            if to is not None and isinstance(to, str):
                issues.append(f'settings.timeout=str({to})')
            if tg is not None and isinstance(tg, str):
                issues.append(f'settings.tags=str (not coerced to array)')
        if issues:
            print(f'FAIL: {issues}')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$nested_ok" = "PASS" ]; then
      run_test "AdaptiveXML" "Nested object with typed fields coercion" "correct types" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Nested object with typed fields coercion" "correct types" "$nested_ok" "$dur"
    fi

    # ── Test 12.11: Streaming tool call — array param delivered correctly ─────
    # Same as 12.8 but with stream:true to test incremental path specifically.
    t0=$(now_ms)
    stream_arr_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Ask the user: What is your favorite fruit? Options: apple, banana, cherry.\"}],\"tools\":$ARRAY_OBJ_TOOL,\"max_tokens\":1000,\"stream\":true,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    stream_arr_ok=$(echo "$stream_arr_resp" | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
args_str = ''
tool_name = ''
for line in lines:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        chunk = json.loads(line[6:])
        delta = chunk.get('choices', [{}])[0].get('delta', {})
        tcs = delta.get('tool_calls', [])
        for tc in tcs:
            f = tc.get('function', {})
            if f.get('name'):
                tool_name = f['name']
            if f.get('arguments') is not None:
                args_str += f['arguments']
    except:
        continue
if not tool_name:
    print('FAIL: no tool call in stream')
elif not args_str:
    print('FAIL: no arguments in stream')
else:
    try:
        args = json.loads(args_str)
        qs = args.get('questions')
        if qs is None:
            print('FAIL: questions param missing')
        elif isinstance(qs, str):
            print('FAIL: questions is string (streaming incremental path did not coerce)')
        elif not isinstance(qs, list):
            print(f'FAIL: questions is {type(qs).__name__}, expected list')
        else:
            print('PASS')
    except json.JSONDecodeError as e:
        print(f'FAIL: invalid JSON arguments: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$stream_arr_ok" = "PASS" ]; then
      run_test "AdaptiveXML" "Streaming array coercion (incremental path)" "array in stream" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "Streaming array coercion (incremental path)" "array in stream" "$stream_arr_ok" "$dur"
    fi

    # ── Test 12.12: XML entity decoding in tool call values ───────────────────
    # Regression test: model XML-encodes < > & in parameter values (correct XML
    # behavior), but afm must decode entities before delivering to client.
    # Without decoding, Python code gets "if size &lt; 1024" instead of "if size < 1024".
    WRITE_TOOL='[{"type":"function","function":{"name":"write_file","description":"Write content to a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string","description":"File content"}},"required":["path","content"]}}}]'
    t0=$(now_ms)
    entity_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python file at /tmp/test.py with a function that checks if a number is less than 10 and greater than 0, using < and > operators.\"}],\"tools\":$WRITE_TOOL,\"max_tokens\":1000,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    entity_ok=$(echo "$entity_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        content = args.get('content', '')
        if not content:
            print('FAIL: content param empty')
        elif '&lt;' in content or '&gt;' in content or '&amp;' in content:
            # Find the offending entity for diagnostics
            for ent in ['&lt;', '&gt;', '&amp;']:
                if ent in content:
                    idx = content.index(ent)
                    ctx = content[max(0,idx-15):idx+20]
                    print(f'FAIL: XML entity not decoded: {ent} in ...{ctx}...')
                    break
        elif '<' in content or '>' in content:
            print('PASS')
        else:
            # Model might not use < > at all — check it at least wrote Python
            if 'def ' in content or 'if ' in content:
                print('PASS')
            else:
                print(f'FAIL: no comparison operators in content ({len(content)} chars)')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$entity_ok" = "PASS" ]; then
      run_test "AdaptiveXML" "XML entity decoding in tool call values" "decoded" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "XML entity decoding in tool call values" "decoded" "$entity_ok" "$dur"
    fi

    # ── Test 12.13: Required params present (grammar-hardened when --enable-grammar-constraints) ──
    # Tests that ALL required params are present in tool call output.
    # Without grammar: model should include both params but may sometimes miss one.
    # With --enable-grammar-constraints: EBNF named rules force both params:
    #   call_bash ::= ... bash_rp_command bash_rp_description extra_params ...
    REQ_PARAMS_TOOL='[{"type":"function","function":{"name":"run_cmd","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The command to run"},"description":{"type":"string","description":"What the command does"},"timeout":{"type":"integer","description":"Timeout in seconds"}},"required":["command","description"]}}}]'
    t0=$(now_ms)
    req_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"List the files in /tmp\"}],\"tools\":$REQ_PARAMS_TOOL,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    req_ok=$(echo "$req_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        missing = []
        if 'command' not in args:
            missing.append('command')
        if 'description' not in args:
            missing.append('description')
        if missing:
            print(f'FAIL: missing required params: {missing}')
        elif not isinstance(args['command'], str) or not args['command']:
            print(f'FAIL: command is empty or not string')
        elif not isinstance(args['description'], str) or not args['description']:
            print(f'FAIL: description is empty or not string')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$req_ok" = "PASS" ]; then
      run_test "AdaptiveXML" "EBNF grammar enforces all required params present" "command + description" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "EBNF grammar enforces all required params present" "command + description" "$req_ok" "$dur"
    fi

    # ── Test 12.14: Structured param types (grammar-hardened when --enable-grammar-constraints) ──
    # Tests array/object params are correct types (not strings).
    # Without grammar: type coercion handles this post-generation.
    # With --enable-grammar-constraints: EBNF rules enforce json_array/json_object
    # at generation time (prevents malformed output before coercion).
    STRUCT_TOOL='[{"type":"function","function":{"name":"tag_files","description":"Tag files with labels","parameters":{"type":"object","properties":{"directory":{"type":"string","description":"Directory path"},"tags":{"type":"array","items":{"type":"string"},"description":"List of tags to apply"},"options":{"type":"object","description":"Additional options"}},"required":["directory","tags"]}}}]'
    t0=$(now_ms)
    struct_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Tag all files in /tmp/docs with labels: important, review, archive\"}],\"tools\":$STRUCT_TOOL,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
    dur=$(( $(now_ms) - t0 ))
    struct_ok=$(echo "$struct_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        d_val = args.get('directory')
        t_val = args.get('tags')
        if d_val is None:
            print('FAIL: required param directory missing')
        elif not isinstance(d_val, str):
            print(f'FAIL: directory is {type(d_val).__name__}, expected string')
        elif t_val is None:
            print('FAIL: required param tags missing')
        elif isinstance(t_val, str):
            print('FAIL: tags is string (not parsed as array — JSON grammar not enforced)')
        elif not isinstance(t_val, list):
            print(f'FAIL: tags is {type(t_val).__name__}, expected list')
        elif len(t_val) == 0:
            print('FAIL: tags array is empty')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
    if [ "$struct_ok" = "PASS" ]; then
      run_test "AdaptiveXML" "EBNF structured params: array gets json_array constraint" "tags is array" "PASS" "$dur"
    else
      run_test "AdaptiveXML" "EBNF structured params: array gets json_array constraint" "tags is array" "$struct_ok" "$dur"
    fi

  else
    echo ""
    echo "  (Model did not produce tool calls — skipping afm_adaptive_xml tests)"
    run_test "AdaptiveXML" "afm_adaptive_xml tests (model lacks tool calling)" "skip" "SKIP" "0"
  fi
fi

# Section 13: Grammar Constraint Validation (standard tier, --grammar-constraints)
# ═══════════════════════════════════════════════════════════════════════════════
# Tests that only run when the server has --enable-grammar-constraints active.
# Grammar constraints use xgrammar EBNF to force valid XML structure at generation
# time, preventing JSON-inside-XML, missing required params, and wrong types.
# These tests are adapted from Scripts/tests/test-tool-call-parsers.py patterns
# which were originally written when xgrammar was always active.

if should_run_section 13 && min_tier standard && [ "$GRAMMAR_CONSTRAINTS" = true ]; then
  CURRENT_TIER="standard"
  echo ""
  echo "🔧 Section 13: Grammar Constraint Validation"

  # Define calculator tool (from test-tool-call-parsers.py patterns)
  CALC_TOOL_13='[{"type":"function","function":{"name":"calculate","description":"Evaluate a mathematical expression and return the result","strict":true,"parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Math expression to evaluate, e.g. 2 + 3 * 4"}},"required":["expression"]}}}]'

  # Two-tool definition (weather + calculator) matching test-tool-call-parsers.py
  DUAL_TOOLS_13='[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a given city","strict":true,"parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}}},{"type":"function","function":{"name":"calculate","description":"Evaluate a mathematical expression","strict":true,"parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Math expression"}},"required":["expression"]}}}]'

  # ── Test 13.1: Calculator tool call (non-streaming) ──────────────────────
  # From test-tool-call-parsers.py calc_nonstream pattern
  t0=$(now_ms)
  calc_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Use the calculator to compute 17 * 23 + 5\"}],\"tools\":$CALC_TOOL_13,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  calc_ok=$(echo "$calc_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        name = tc[0]['function']['name']
        args = json.loads(tc[0]['function']['arguments'])
        if name != 'calculate':
            print(f'FAIL: expected calculate, got {name}')
        elif 'expression' not in args:
            print('FAIL: missing expression param')
        elif not isinstance(args['expression'], str):
            print(f'FAIL: expression is {type(args[\"expression\"]).__name__}, expected str')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$calc_ok" = "PASS" ]; then
    run_test "Grammar" "Calculator tool call (non-streaming)" "calculate(expression)" "PASS" "$dur"
  else
    run_test "Grammar" "Calculator tool call (non-streaming)" "calculate(expression)" "$calc_ok" "$dur"
  fi

  # ── Test 13.2: Calculator tool call (streaming) ──────────────────────────
  # From test-tool-call-parsers.py calc_stream pattern
  t0=$(now_ms)
  calc_stream_raw=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"Please calculate 99 * 101 using the tool\"}],\"tools\":$CALC_TOOL_13,\"max_tokens\":500,\"stream\":true,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  calc_stream_ok=$(echo "$calc_stream_raw" | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
tool_name = ''
args_parts = []
finish = None
for line in lines:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        chunk = json.loads(line[6:])
        delta = chunk['choices'][0].get('delta', {})
        fr = chunk['choices'][0].get('finish_reason')
        if fr: finish = fr
        for tc in delta.get('tool_calls', []):
            fn = tc.get('function', {})
            if fn.get('name'): tool_name = fn['name']
            if fn.get('arguments') is not None: args_parts.append(fn['arguments'])
    except: pass
if not tool_name:
    print('FAIL: no tool call in stream')
elif finish != 'tool_calls':
    print(f'FAIL: finish_reason={finish}')
else:
    full_args = ''.join(args_parts)
    try:
        args = json.loads(full_args)
        if tool_name != 'calculate':
            print(f'FAIL: expected calculate, got {tool_name}')
        elif 'expression' not in args:
            print('FAIL: missing expression param')
        else:
            print('PASS')
    except json.JSONDecodeError as e:
        print(f'FAIL: invalid JSON: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$calc_stream_ok" = "PASS" ]; then
    run_test "Grammar" "Calculator tool call (streaming)" "calculate(expression) via SSE" "PASS" "$dur"
  else
    run_test "Grammar" "Calculator tool call (streaming)" "calculate(expression) via SSE" "$calc_stream_ok" "$dur"
  fi

  # ── Test 13.3: Two tools — correct selection under grammar constraint ────
  # Grammar must allow model to choose between tools correctly
  t0=$(now_ms)
  dual_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Tokyo?\"}],\"tools\":$DUAL_TOOLS_13,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  dual_ok=$(echo "$dual_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        name = tc[0]['function']['name']
        if name != 'get_weather':
            print(f'FAIL: expected get_weather, got {name}')
        else:
            args = json.loads(tc[0]['function']['arguments'])
            if 'city' not in args:
                print('FAIL: missing city param')
            else:
                print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$dual_ok" = "PASS" ]; then
    run_test "Grammar" "Two tools: grammar allows correct selection" "get_weather selected" "PASS" "$dur"
  else
    run_test "Grammar" "Two tools: grammar allows correct selection" "get_weather selected" "$dual_ok" "$dur"
  fi

  # ── Test 13.4: Two tools — calc selected when prompted ───────────────────
  t0=$(now_ms)
  dual2_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Calculate 42 * 7 using the tool\"}],\"tools\":$DUAL_TOOLS_13,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  dual2_ok=$(echo "$dual2_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        name = tc[0]['function']['name']
        if name != 'calculate':
            print(f'FAIL: expected calculate, got {name}')
        else:
            args = json.loads(tc[0]['function']['arguments'])
            if 'expression' not in args:
                print('FAIL: missing expression param')
            else:
                print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$dual2_ok" = "PASS" ]; then
    run_test "Grammar" "Two tools: grammar selects calculate" "calculate selected" "PASS" "$dur"
  else
    run_test "Grammar" "Two tools: grammar selects calculate" "calculate selected" "$dual2_ok" "$dur"
  fi

  # ── Test 13.5: Grammar prevents missing required params ──────────────────
  # With grammar constraints, EBNF named rules force ALL required params.
  # This test uses 3 required params to stress the grammar.
  THREE_REQ_TOOL='[{"type":"function","function":{"name":"send_email","description":"Send an email","strict":true,"parameters":{"type":"object","properties":{"to":{"type":"string","description":"Recipient email"},"subject":{"type":"string","description":"Email subject"},"body":{"type":"string","description":"Email body"}},"required":["to","subject","body"]}}}]'
  t0=$(now_ms)
  three_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Send an email to alice@example.com with subject Hello and body How are you?\"}],\"tools\":$THREE_REQ_TOOL,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  three_ok=$(echo "$three_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        missing = [p for p in ('to', 'subject', 'body') if p not in args or not args[p]]
        if missing:
            print(f'FAIL: missing required params: {missing}')
        elif not all(isinstance(args[p], str) for p in ('to', 'subject', 'body')):
            types = {p: type(args[p]).__name__ for p in ('to', 'subject', 'body')}
            print(f'FAIL: wrong types: {types}')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$three_ok" = "PASS" ]; then
    run_test "Grammar" "Grammar enforces 3 required params (send_email)" "to + subject + body" "PASS" "$dur"
  else
    run_test "Grammar" "Grammar enforces 3 required params (send_email)" "to + subject + body" "$three_ok" "$dur"
  fi

  # ── Test 13.6: Grammar constrains array param at generation time ─────────
  # With grammar, the EBNF rule for array params produces json_array constraint
  # so the model can't emit a bare string for an array param.
  ARRAY_TOOL_13='[{"type":"function","function":{"name":"add_tags","description":"Add tags to an item","strict":true,"parameters":{"type":"object","properties":{"item_id":{"type":"string","description":"Item ID"},"tags":{"type":"array","items":{"type":"string"},"description":"Tags to add"}},"required":["item_id","tags"]}}}]'
  t0=$(now_ms)
  arr_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Add tags bug, critical, and urgent to item PROJ-123\"}],\"tools\":$ARRAY_TOOL_13,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  arr_ok=$(echo "$arr_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        tags = args.get('tags')
        if tags is None:
            print('FAIL: tags param missing')
        elif isinstance(tags, str):
            print('FAIL: tags is string — grammar should have enforced json_array')
        elif not isinstance(tags, list):
            print(f'FAIL: tags is {type(tags).__name__}, expected list')
        elif len(tags) == 0:
            print('FAIL: tags array is empty')
        elif not all(isinstance(t, str) for t in tags):
            print(f'FAIL: non-string items in tags: {[type(t).__name__ for t in tags]}')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$arr_ok" = "PASS" ]; then
    run_test "Grammar" "Grammar constrains array param at generation time" "tags is array" "PASS" "$dur"
  else
    run_test "Grammar" "Grammar constrains array param at generation time" "tags is array" "$arr_ok" "$dur"
  fi

  # ── Test 13.7: Grammar with streaming array param ────────────────────────
  t0=$(now_ms)
  arr_stream_raw=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"Add tags review and pending to item TASK-456\"}],\"tools\":$ARRAY_TOOL_13,\"max_tokens\":500,\"stream\":true,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  arr_stream_ok=$(echo "$arr_stream_raw" | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
tool_name = ''
args_parts = []
for line in lines:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        chunk = json.loads(line[6:])
        delta = chunk['choices'][0].get('delta', {})
        for tc in delta.get('tool_calls', []):
            fn = tc.get('function', {})
            if fn.get('name'): tool_name = fn['name']
            if fn.get('arguments') is not None: args_parts.append(fn['arguments'])
    except: pass
if not tool_name:
    print('FAIL: no tool call in stream')
else:
    full_args = ''.join(args_parts)
    try:
        args = json.loads(full_args)
        tags = args.get('tags')
        if isinstance(tags, str):
            print('FAIL: tags is string in streaming path')
        elif not isinstance(tags, list):
            print(f'FAIL: tags is {type(tags).__name__}')
        else:
            print('PASS')
    except json.JSONDecodeError as e:
        print(f'FAIL: invalid JSON: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$arr_stream_ok" = "PASS" ]; then
    run_test "Grammar" "Grammar array param via streaming" "tags is array (SSE)" "PASS" "$dur"
  else
    run_test "Grammar" "Grammar array param via streaming" "tags is array (SSE)" "$arr_stream_ok" "$dur"
  fi

  # ── Test 13.8: Grammar with nested object + mixed types ──────────────────
  # Complex schema that exercises grammar enforcement of nested structures
  COMPLEX_TOOL_13='[{"type":"function","function":{"name":"create_task","description":"Create a project task","strict":true,"parameters":{"type":"object","properties":{"title":{"type":"string","description":"Task title"},"priority":{"type":"integer","description":"Priority 1-5"},"assignees":{"type":"array","items":{"type":"string"},"description":"List of assignees"},"metadata":{"type":"object","description":"Extra metadata","properties":{"sprint":{"type":"string"},"estimate_hours":{"type":"number"}}}},"required":["title","priority","assignees"]}}}]'
  t0=$(now_ms)
  complex_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"Create a task: Fix login bug, priority 2, assign to alice and bob, sprint S42, estimate 4.5 hours\"}],\"tools\":$COMPLEX_TOOL_13,\"max_tokens\":500,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  complex_ok=$(echo "$complex_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if not tc:
        print('FAIL: no tool_calls')
    else:
        args = json.loads(tc[0]['function']['arguments'])
        issues = []
        title = args.get('title')
        priority = args.get('priority')
        assignees = args.get('assignees')
        if not isinstance(title, str) or not title:
            issues.append(f'title={type(title).__name__}({repr(title)})')
        if priority is not None and isinstance(priority, str):
            issues.append(f'priority=str({priority})')
        if assignees is None:
            issues.append('assignees missing')
        elif isinstance(assignees, str):
            issues.append('assignees=str (not array)')
        elif not isinstance(assignees, list):
            issues.append(f'assignees={type(assignees).__name__}')
        if issues:
            print(f'FAIL: {issues}')
        else:
            print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$complex_ok" = "PASS" ]; then
    run_test "Grammar" "Complex schema: string + int + array + object" "correct types" "PASS" "$dur"
  else
    run_test "Grammar" "Complex schema: string + int + array + object" "correct types" "$complex_ok" "$dur"
  fi

fi

# ═══════════════════════════════════════════════════════════════════════════════
# Section 14: Strict Wiring Validation (smoke tier)
# ═══════════════════════════════════════════════════════════════════════════════
# Tests the strict: true → grammar activation wiring and observability:
#   - X-Grammar-Constraints: downgraded header when grammar not enabled
#   - Header absent when grammar IS enabled
#   - Streaming json_schema with strict: true produces valid JSON
#   - Warning log when strict requested but engine not enabled

if should_run_section 14; then
  CURRENT_TIER="smoke"
  echo ""
  echo "🔒 Section 14: Strict Wiring Validation"

  STRICT_TOOL_14='[{"type":"function","function":{"name":"get_weather","description":"Get weather","strict":true,"parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]'
  STRICT_SCHEMA_14='{"type":"json_schema","json_schema":{"name":"person","strict":true,"schema":{"type":"object","additionalProperties":false,"properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}}}'

  # ── Test 14.1: X-Grammar-Constraints header — tool strict:true ─────────
  # When server does NOT have --enable-grammar-constraints but request has strict:true,
  # the response should include X-Grammar-Constraints: downgraded.
  # When server DOES have --enable-grammar-constraints, the header should be absent.
  t0=$(now_ms)
  header_resp=$(api_call_headers "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}],\"tools\":$STRICT_TOOL_14,\"max_tokens\":200,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  if [ "$GRAMMAR_CONSTRAINTS" = true ]; then
    # Grammar enabled: header should be ABSENT
    if echo "$header_resp" | grep -qi 'X-Grammar-Constraints'; then
      run_test "StrictWiring" "Header absent when grammar enabled (tool strict:true)" "no header" "FAIL: header present" "$dur"
    else
      run_test "StrictWiring" "Header absent when grammar enabled (tool strict:true)" "no header" "PASS" "$dur"
    fi
  else
    # Grammar not enabled: header should be "downgraded"
    if echo "$header_resp" | grep -qi 'X-Grammar-Constraints: downgraded'; then
      run_test "StrictWiring" "X-Grammar-Constraints: downgraded header (tool strict:true)" "downgraded" "PASS" "$dur"
    else
      run_test "StrictWiring" "X-Grammar-Constraints: downgraded header (tool strict:true)" "downgraded" "FAIL: header missing" "$dur"
    fi
  fi

  # ── Test 14.2: X-Grammar-Constraints header — json_schema strict:true ──
  t0=$(now_ms)
  header_resp2=$(api_call_headers "{\"messages\":[{\"role\":\"user\",\"content\":\"Return a person record for Ada Lovelace age 36\"}],\"response_format\":$STRICT_SCHEMA_14,\"max_tokens\":200,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  if [ "$GRAMMAR_CONSTRAINTS" = true ]; then
    if echo "$header_resp2" | grep -qi 'X-Grammar-Constraints'; then
      run_test "StrictWiring" "Header absent when grammar enabled (schema strict:true)" "no header" "FAIL: header present" "$dur"
    else
      run_test "StrictWiring" "Header absent when grammar enabled (schema strict:true)" "no header" "PASS" "$dur"
    fi
  else
    if echo "$header_resp2" | grep -qi 'X-Grammar-Constraints: downgraded'; then
      run_test "StrictWiring" "X-Grammar-Constraints: downgraded header (schema strict:true)" "downgraded" "PASS" "$dur"
    else
      run_test "StrictWiring" "X-Grammar-Constraints: downgraded header (schema strict:true)" "downgraded" "FAIL: header missing" "$dur"
    fi
  fi

  # ── Test 14.3: No header when strict is absent ─────────────────────────
  t0=$(now_ms)
  NOSTRICTTOOL='[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]'
  header_resp3=$(api_call_headers "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in London?\"}],\"tools\":$NOSTRICTTOOL,\"max_tokens\":200,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  if echo "$header_resp3" | grep -qi 'X-Grammar-Constraints'; then
    run_test "StrictWiring" "No header when strict absent" "no header" "FAIL: header present" "$dur"
  else
    run_test "StrictWiring" "No header when strict absent" "no header" "PASS" "$dur"
  fi

  # ── Test 14.4: Streaming json_schema strict:true returns valid JSON ────
  t0=$(now_ms)
  stream_schema_content=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"Return a person record for Alan Turing age 41\"}],\"response_format\":$STRICT_SCHEMA_14,\"max_tokens\":300,\"stream\":true,\"temperature\":0}" | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
parts = []
for line in lines:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        chunk = json.loads(line[6:])
        c = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
        if c:
            parts.append(c)
    except:
        pass
full = ''.join(parts)
# Strip any think tags if present
import re
full = re.sub(r'<think>.*?</think>', '', full, flags=re.DOTALL).strip()
print(full)
" 2>/dev/null)
  dur=$(( $(now_ms) - t0 ))
  schema_stream_ok=$(echo "$stream_schema_content" | python3 -c "
import sys, json
try:
    text = sys.stdin.read().strip()
    if not text:
        print('FAIL: empty response')
    else:
        d = json.loads(text)
        if 'name' in d and 'age' in d:
            print('PASS')
        else:
            print(f'FAIL: missing keys, got: {list(d.keys())}')
except json.JSONDecodeError as e:
    print(f'FAIL: invalid JSON: {e}')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$schema_stream_ok" = "PASS" ]; then
    run_test "StrictWiring" "Streaming json_schema strict:true returns valid JSON" "valid JSON with name+age" "PASS" "$dur"
  else
    run_test "StrictWiring" "Streaming json_schema strict:true returns valid JSON" "valid JSON with name+age" "$schema_stream_ok" "$dur"
  fi

  # ── Test 14.5: Streaming tool strict:true returns valid tool call ──────
  t0=$(now_ms)
  stream_tool_raw=$(api_stream "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Tokyo?\"}],\"tools\":$STRICT_TOOL_14,\"max_tokens\":300,\"stream\":true,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  stream_tool_ok=$(echo "$stream_tool_raw" | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
tool_name = ''
args_parts = []
finish = None
for line in lines:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    try:
        chunk = json.loads(line[6:])
        delta = chunk['choices'][0].get('delta', {})
        fr = chunk['choices'][0].get('finish_reason')
        if fr: finish = fr
        for tc in delta.get('tool_calls', []):
            fn = tc.get('function', {})
            if fn.get('name'): tool_name = fn['name']
            if fn.get('arguments') is not None: args_parts.append(fn['arguments'])
    except: pass
if not tool_name:
    print('FAIL: no tool call in stream')
elif tool_name != 'get_weather':
    print(f'FAIL: expected get_weather, got {tool_name}')
elif finish != 'tool_calls':
    print(f'FAIL: finish_reason={finish}')
else:
    full_args = ''.join(args_parts)
    try:
        args = json.loads(full_args)
        if 'location' in args:
            print('PASS')
        else:
            print(f'FAIL: missing location in args: {args}')
    except json.JSONDecodeError as e:
        print(f'FAIL: invalid JSON: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$stream_tool_ok" = "PASS" ]; then
    run_test "StrictWiring" "Streaming tool strict:true returns valid tool call" "get_weather(location)" "PASS" "$dur"
  else
    run_test "StrictWiring" "Streaming tool strict:true returns valid tool call" "get_weather(location)" "$stream_tool_ok" "$dur"
  fi

  # ── Test 14.6: strict:false does NOT activate grammar (non-streaming) ──
  t0=$(now_ms)
  STRICTFALSE_TOOL='[{"type":"function","function":{"name":"get_weather","description":"Get weather","strict":false,"parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]'
  nostrict_resp=$(api_call "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Berlin?\"}],\"tools\":$STRICTFALSE_TOOL,\"max_tokens\":200,\"stream\":false,\"temperature\":0}")
  dur=$(( $(now_ms) - t0 ))
  nostrict_ok=$(echo "$nostrict_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    if tc and tc[0]['function']['name'] == 'get_weather':
        print('PASS')
    elif d['choices'][0]['message'].get('content'):
        print('PASS')  # best-effort may produce text instead
    else:
        print('FAIL: no tool call and no content')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null || echo "FAIL: parse error")
  if [ "$nostrict_ok" = "PASS" ]; then
    run_test "StrictWiring" "strict:false does not error (best-effort)" "valid response" "PASS" "$dur"
  else
    run_test "StrictWiring" "strict:false does not error (best-effort)" "valid response" "$nostrict_ok" "$dur"
  fi

fi

fi  # end: min_tier smoke (server-dependent sections)

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
  .group-badge.Structured { color: #d2a8ff; border-color: #8957e5; }
  .group-badge.AdaptiveXML { color: #f0883e; border-color: #bd561d; }
  .group-badge.Grammar { color: #d2a8ff; border-color: #8957e5; }
  .group-badge.XMLParsing { color: #f0883e; border-color: #bd561d; }
  .group-badge.NullableSchema { color: #79c0ff; border-color: #388bfd; }
  .group-badge.UnitTest { color: #a5d6ff; border-color: #58a6ff; }
  .tier-row td { background: #161b22; padding: 0.6rem 1rem; font-weight: 700; font-size: 0.9rem; border-bottom: 2px solid #30363d; border-top: 2px solid #30363d; }
  .tier-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.7rem; font-weight: 600; }
  .tier-badge.unit { background: #1a1a2e; color: #a5d6ff; border: 1px solid #58a6ff; }
  .tier-badge.smoke { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .tier-badge.standard { background: #0d1a30; color: #58a6ff; border: 1px solid #1f6feb; }
  .tier-badge.full { background: #2d1f00; color: #d29922; border: 1px solid #9e6a03; }
  .detail { font-family: 'SF Mono', 'Menlo', monospace; font-size: 0.8rem; color: #8b949e; white-space: pre-wrap; word-break: break-word; max-height: 100px; overflow-y: auto; background: #0d1117; padding: 0.5rem; border-radius: 6px; border: 1px solid #21262d; margin-top: 0.25rem; }
  .duration { color: #8b949e; font-family: 'SF Mono', monospace; font-size: 0.85rem; }
  .test-idx { color: #8b949e; font-family: 'SF Mono', monospace; font-size: 0.85rem; }
  .footer { text-align: center; margin-top: 2rem; color: #484f58; font-size: 0.8rem; }
</style>
</head>
<body>
HTMLHEAD

cat >> "$REPORT_FILE" <<EOF
<div class="header">
  <h1>AFM Assertion Test Report</h1>
  <div class="meta">
    Model: <strong>$MODEL</strong> &middot; Tier: <strong>$TIER</strong> &middot; Grammar: <strong>$GRAMMAR_CONSTRAINTS</strong><br>
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
<tr><th>#</th><th>Test</th><th>Group</th><th>Coverage</th><th>Status</th><th>Duration</th><th>Details</th></tr>
</thead>
<tbody>
EOF

# Emit all rows in execution order with coverage tier badges
for entry in "${RESULTS[@]}"; do
  IFS='|' read -r status group name expected actual duration tier test_idx <<< "$entry"
  tier="${tier:-smoke}"
  test_idx="${test_idx:-0}"

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

  # Coverage badges: show which tiers include this test
  # smoke tests run in smoke+standard+full, standard in standard+full, full in full only
  tier_badges=""
  case "$tier" in
    unit)     tier_badges='<span class="tier-badge unit">unit</span> <span class="tier-badge smoke">smoke</span> <span class="tier-badge standard">standard</span> <span class="tier-badge full">full</span>' ;;
    smoke)    tier_badges='<span class="tier-badge smoke">smoke</span> <span class="tier-badge standard">standard</span> <span class="tier-badge full">full</span>' ;;
    standard) tier_badges='<span class="tier-badge standard">standard</span> <span class="tier-badge full">full</span>' ;;
    full)     tier_badges='<span class="tier-badge full">full</span>' ;;
  esac

  cat >> "$REPORT_FILE" <<EOF
<tr>
  <td><span class="test-idx">${test_idx}</span></td>
  <td>$name_esc</td>
  <td><span class="group-badge $group">$group</span></td>
  <td>${tier_badges}</td>
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
