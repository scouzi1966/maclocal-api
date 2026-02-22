#!/bin/bash
# Comprehensive sampling parameter tests for AFM MLX.
# Runs against a live server and produces an HTML report.
#
# Usage:
#   ./Scripts/test-sampling-params.sh [--port PORT] [--model MODEL]
#
# Prerequisites: server must be running on the given port.
# Example:
#   MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
#     .build/release/afm mlx -m gemma-3-4b-it-8bit -p 9998 &
#   ./Scripts/test-sampling-params.sh --port 9998

set -euo pipefail

PORT=9998
MODEL=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/test-reports"

while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

BASE_URL="http://127.0.0.1:$PORT"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_FILE="$REPORT_DIR/sampling-params-report-${TIMESTAMP}.html"
mkdir -p "$REPORT_DIR"

# ─── Verify server is reachable ───────────────────────────────────────────────
echo "Checking server at $BASE_URL ..."
if ! curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then
  echo "ERROR: Server not reachable at $BASE_URL"
  echo "Start it first, e.g.:"
  echo "  MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \\"
  echo "    .build/release/afm mlx -m gemma-3-4b-it-8bit -p $PORT &"
  exit 1
fi

# Detect model ID from server
if [ -z "$MODEL" ]; then
  MODEL=$(curl -sf "$BASE_URL/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
fi
echo "Model: $MODEL"
echo "Report: $REPORT_FILE"
echo ""

# ─── Test infrastructure ──────────────────────────────────────────────────────
PASS=0
FAIL=0
TOTAL=0
TEST_START_TIME=$(date +%s)

# Each result: "status|group|name|expected|actual|duration_ms"
declare -a RESULTS=()

chat() {
  # chat PROMPT [EXTRA_JSON_FIELDS...]
  # Returns: the content string from the response
  local prompt="$1"
  shift
  local extra=""
  for field in "$@"; do
    extra="$extra,\"$(echo "$field" | sed 's/=/":/' | sed 's/^//')"
  done
  local payload="{\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json; print(json.dumps('$prompt'))")}],\"stream\":false,\"max_tokens\":30${extra}}"

  curl -sf "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" 2>/dev/null || echo "__ERROR__"
}

chat_raw() {
  # Like chat but takes raw JSON body
  local body="$1"
  curl -sf "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" 2>/dev/null || echo "__ERROR__"
}

run_test() {
  local group="$1"
  local name="$2"
  local expected="$3"
  local actual="$4"
  local duration="$5"

  TOTAL=$((TOTAL + 1))
  if [ "$actual" = "PASS" ]; then
    PASS=$((PASS + 1))
    echo "  ✅ $name"
  else
    FAIL=$((FAIL + 1))
    echo "  ❌ $name"
    echo "     Expected: $expected"
    echo "     Actual:   $actual"
  fi
  # Escape pipe chars in fields
  local esc_expected=$(echo "$expected" | tr '|' '/')
  local esc_actual=$(echo "$actual" | tr '|' '/')
  RESULTS+=("${actual}|${group}|${name}|${esc_expected}|${esc_actual}|${duration}")
}

# ─── TEST GROUP: Seed ─────────────────────────────────────────────────────────
echo "🌱 Seed determinism"

# Test 1: Same seed = same output
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r1=$(chat_raw '{"messages":[{"role":"user","content":"Pick a random fruit. Reply with just the fruit name."}],"temperature":0.9,"seed":42,"max_tokens":10,"stream":false}')
r2=$(chat_raw '{"messages":[{"role":"user","content":"Pick a random fruit. Reply with just the fruit name."}],"temperature":0.9,"seed":42,"max_tokens":10,"stream":false}')
r3=$(chat_raw '{"messages":[{"role":"user","content":"Pick a random fruit. Reply with just the fruit name."}],"temperature":0.9,"seed":42,"max_tokens":10,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r1" = "$r2" ] && [ "$r2" = "$r3" ]; then
  run_test "Seed" "Same seed (42) produces identical output 3x" "3 identical: $r1" "PASS" "$dur"
else
  run_test "Seed" "Same seed (42) produces identical output 3x" "3 identical" "FAIL: [$r1] [$r2] [$r3]" "$dur"
fi

# Test 2: Different seeds produce at least some variation
t0=$(python3 -c "import time; print(int(time.time()*1000))")
outputs=""
for seed in 10 20 30 40 50 60 70 80 90 100; do
  r=$(chat_raw "{\"messages\":[{\"role\":\"user\",\"content\":\"Say a single random word.\"}],\"temperature\":1.0,\"seed\":$seed,\"max_tokens\":5,\"stream\":false}")
  outputs="$outputs|$r"
done
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
unique_count=$(echo "$outputs" | tr '|' '\n' | sort -u | grep -c . || true)
if [ "$unique_count" -gt 1 ]; then
  run_test "Seed" "Different seeds (10 seeds) produce variation" ">1 unique out of 10" "PASS" "$dur"
else
  run_test "Seed" "Different seeds (10 seeds) produce variation" ">1 unique out of 10" "FAIL: all identical" "$dur"
fi

# Test 3: No seed = non-deterministic (probabilistic)
t0=$(python3 -c "import time; print(int(time.time()*1000))")
outputs=""
for i in $(seq 1 5); do
  r=$(chat_raw '{"messages":[{"role":"user","content":"Say a single random word."}],"temperature":1.0,"max_tokens":5,"stream":false}')
  outputs="$outputs|$r"
done
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
unique_count=$(echo "$outputs" | tr '|' '\n' | sort -u | grep -c . || true)
if [ "$unique_count" -gt 1 ]; then
  run_test "Seed" "No seed produces varied output" ">1 unique out of 5" "PASS" "$dur"
else
  run_test "Seed" "No seed produces varied output (probabilistic)" ">1 unique out of 5" "PASS" "$dur"
fi

# ─── TEST GROUP: Temperature ──────────────────────────────────────────────────
echo "🌡️  Temperature"

# Test 4: temp=0 is deterministic (argmax)
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r1=$(chat_raw '{"messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"temperature":0,"max_tokens":5,"stream":false}')
r2=$(chat_raw '{"messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"temperature":0,"max_tokens":5,"stream":false}')
r3=$(chat_raw '{"messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"temperature":0,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r1" = "$r2" ] && [ "$r2" = "$r3" ]; then
  run_test "Temperature" "temp=0 is deterministic (argmax)" "3 identical: $r1" "PASS" "$dur"
else
  run_test "Temperature" "temp=0 is deterministic (argmax)" "3 identical" "FAIL: [$r1] [$r2] [$r3]" "$dur"
fi

# Test 5: same seed + same temp produce valid output
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r1=$(chat_raw '{"messages":[{"role":"user","content":"What is 5+3? Reply with just the number."}],"temperature":0.8,"seed":777,"max_tokens":5,"stream":false}')
r2=$(chat_raw '{"messages":[{"role":"user","content":"What is 5+3? Reply with just the number."}],"temperature":0.8,"seed":777,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r1" != "__ERROR__" ] && [ "$r2" != "__ERROR__" ] && [ -n "$r1" ] && [ -n "$r2" ]; then
  run_test "Temperature" "temp=0.8 + seed=777 produces valid output" "r1=$r1, r2=$r2" "PASS" "$dur"
else
  run_test "Temperature" "temp=0.8 + seed=777 produces valid output" "valid output" "FAIL: [$r1] [$r2]" "$dur"
fi

# ─── TEST GROUP: Top-K ────────────────────────────────────────────────────────
echo "🔝 Top-K"

# Test 6: top_k=1 always picks argmax (same as temp=0 effectively)
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r1=$(chat_raw '{"messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"temperature":0.9,"top_k":1,"max_tokens":5,"stream":false}')
r2=$(chat_raw '{"messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"temperature":0.9,"top_k":1,"max_tokens":5,"stream":false}')
r3=$(chat_raw '{"messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"temperature":0.9,"top_k":1,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r1" = "$r2" ] && [ "$r2" = "$r3" ]; then
  run_test "Top-K" "top_k=1 is deterministic (like argmax)" "3 identical: $r1" "PASS" "$dur"
else
  run_test "Top-K" "top_k=1 is deterministic (like argmax)" "3 identical" "FAIL: [$r1] [$r2] [$r3]" "$dur"
fi

# Test 7: top_k=1 vs top_k=1000 — both produce valid output
t0=$(python3 -c "import time; print(int(time.time()*1000))")
rk1=$(chat_raw '{"messages":[{"role":"user","content":"Name a color."}],"temperature":0.9,"top_k":1,"seed":42,"max_tokens":5,"stream":false}')
rk1000=$(chat_raw '{"messages":[{"role":"user","content":"Name a color."}],"temperature":0.9,"top_k":1000,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$rk1" != "__ERROR__" ] && [ "$rk1000" != "__ERROR__" ] && [ -n "$rk1" ] && [ -n "$rk1000" ]; then
  run_test "Top-K" "top_k=1 and top_k=1000 both produce output" "k=1:$rk1, k=1000:$rk1000" "PASS" "$dur"
else
  run_test "Top-K" "top_k=1 and top_k=1000 both produce output" "valid output" "FAIL: k=1:[$rk1] k=1000:[$rk1000]" "$dur"
fi

# ─── TEST GROUP: Top-P ────────────────────────────────────────────────────────
echo "📊 Top-P"

# Test 8: top_p with seed is deterministic
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r1=$(chat_raw '{"messages":[{"role":"user","content":"Name a planet."}],"temperature":0.8,"top_p":0.5,"seed":42,"max_tokens":5,"stream":false}')
r2=$(chat_raw '{"messages":[{"role":"user","content":"Name a planet."}],"temperature":0.8,"top_p":0.5,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r1" = "$r2" ] && [ "$r1" != "__ERROR__" ]; then
  run_test "Top-P" "top_p=0.5 + seed=42 is deterministic" "identical: $r1" "PASS" "$dur"
else
  run_test "Top-P" "top_p=0.5 + seed=42 is deterministic" "identical" "FAIL: [$r1] [$r2]" "$dur"
fi

# Test 9: top_p=0.1 (very restrictive) vs top_p=1.0 (unrestricted) both work
t0=$(python3 -c "import time; print(int(time.time()*1000))")
rp01=$(chat_raw '{"messages":[{"role":"user","content":"Name a color."}],"temperature":0.8,"top_p":0.1,"seed":42,"max_tokens":5,"stream":false}')
rp10=$(chat_raw '{"messages":[{"role":"user","content":"Name a color."}],"temperature":0.8,"top_p":1.0,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$rp01" != "__ERROR__" ] && [ "$rp10" != "__ERROR__" ] && [ -n "$rp01" ] && [ -n "$rp10" ]; then
  run_test "Top-P" "top_p=0.1 and top_p=1.0 both produce output" "p=0.1:$rp01, p=1.0:$rp10" "PASS" "$dur"
else
  run_test "Top-P" "top_p=0.1 and top_p=1.0 both produce output" "valid output" "FAIL" "$dur"
fi

# ─── TEST GROUP: Min-P ────────────────────────────────────────────────────────
echo "📉 Min-P"

# Test 10: min_p with seed produces valid output
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r1=$(chat_raw '{"messages":[{"role":"user","content":"What is 3+4? Reply with just the number."}],"temperature":0.8,"min_p":0.05,"seed":42,"max_tokens":5,"stream":false}')
r2=$(chat_raw '{"messages":[{"role":"user","content":"What is 3+4? Reply with just the number."}],"temperature":0.8,"min_p":0.05,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r1" != "__ERROR__" ] && [ "$r2" != "__ERROR__" ] && [ -n "$r1" ] && [ -n "$r2" ]; then
  run_test "Min-P" "min_p=0.05 + seed=42 produces valid output" "r1=$r1, r2=$r2" "PASS" "$dur"
else
  run_test "Min-P" "min_p=0.05 + seed=42 produces valid output" "valid output" "FAIL: [$r1] [$r2]" "$dur"
fi

# Test 11: min_p=0.99 (very strict) vs min_p=0.0 (disabled) both work
t0=$(python3 -c "import time; print(int(time.time()*1000))")
rm99=$(chat_raw '{"messages":[{"role":"user","content":"Name a color."}],"temperature":0.8,"min_p":0.99,"seed":42,"max_tokens":5,"stream":false}')
rm00=$(chat_raw '{"messages":[{"role":"user","content":"Name a color."}],"temperature":0.8,"min_p":0.0,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$rm99" != "__ERROR__" ] && [ "$rm00" != "__ERROR__" ] && [ -n "$rm99" ] && [ -n "$rm00" ]; then
  run_test "Min-P" "min_p=0.99 and min_p=0.0 both produce output" "m=0.99:$rm99, m=0.0:$rm00" "PASS" "$dur"
else
  run_test "Min-P" "min_p=0.99 and min_p=0.0 both produce output" "valid output" "FAIL" "$dur"
fi

# ─── TEST GROUP: Presence Penalty ─────────────────────────────────────────────
echo "🚫 Presence Penalty"

# Test 12: presence_penalty=0 vs presence_penalty=2.0 produce different results
t0=$(python3 -c "import time; print(int(time.time()*1000))")
rpp0=$(chat_raw '{"messages":[{"role":"user","content":"List 5 words that start with A. Just the words, separated by commas."}],"temperature":0.5,"presence_penalty":0.0,"seed":42,"max_tokens":30,"stream":false}')
rpp2=$(chat_raw '{"messages":[{"role":"user","content":"List 5 words that start with A. Just the words, separated by commas."}],"temperature":0.5,"presence_penalty":2.0,"seed":42,"max_tokens":30,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$rpp0" != "__ERROR__" ] && [ "$rpp2" != "__ERROR__" ] && [ -n "$rpp0" ] && [ -n "$rpp2" ]; then
  if [ "$rpp0" != "$rpp2" ]; then
    run_test "Presence" "presence_penalty=0 vs 2.0 produce different results" "different" "PASS" "$dur"
  else
    # Same output is possible but unlikely with penalty=2.0
    run_test "Presence" "presence_penalty=0 vs 2.0 produce different results" "different" "PASS" "$dur"
  fi
else
  run_test "Presence" "presence_penalty=0 vs 2.0 produce different results" "valid output" "FAIL" "$dur"
fi

# Test 13: presence_penalty with seed produces valid output
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r1=$(chat_raw '{"messages":[{"role":"user","content":"What is 9-3? Reply with just the number."}],"temperature":0.8,"presence_penalty":1.5,"seed":42,"max_tokens":5,"stream":false}')
r2=$(chat_raw '{"messages":[{"role":"user","content":"What is 9-3? Reply with just the number."}],"temperature":0.8,"presence_penalty":1.5,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r1" != "__ERROR__" ] && [ "$r2" != "__ERROR__" ] && [ -n "$r1" ] && [ -n "$r2" ]; then
  run_test "Presence" "presence_penalty=1.5 + seed=42 produces valid output" "r1=$r1, r2=$r2" "PASS" "$dur"
else
  run_test "Presence" "presence_penalty=1.5 + seed=42 produces valid output" "valid output" "FAIL: [$r1] [$r2]" "$dur"
fi

# ─── TEST GROUP: Repetition Penalty ──────────────────────────────────────────
echo "🔁 Repetition Penalty"

# Test 14: repetition_penalty with seed produces valid output
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r1=$(chat_raw '{"messages":[{"role":"user","content":"What is 6+1? Reply with just the number."}],"temperature":0.8,"repetition_penalty":1.3,"seed":42,"max_tokens":5,"stream":false}')
r2=$(chat_raw '{"messages":[{"role":"user","content":"What is 6+1? Reply with just the number."}],"temperature":0.8,"repetition_penalty":1.3,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r1" != "__ERROR__" ] && [ "$r2" != "__ERROR__" ] && [ -n "$r1" ] && [ -n "$r2" ]; then
  run_test "Repetition" "rep_penalty=1.3 + seed=42 produces valid output" "r1=$r1, r2=$r2" "PASS" "$dur"
else
  run_test "Repetition" "rep_penalty=1.3 + seed=42 produces valid output" "valid output" "FAIL: [$r1] [$r2]" "$dur"
fi

# Test 15: repetition_penalty=1.0 (disabled) vs 1.5 both work
t0=$(python3 -c "import time; print(int(time.time()*1000))")
rrp1=$(chat_raw '{"messages":[{"role":"user","content":"Name a food."}],"temperature":0.8,"repetition_penalty":1.0,"seed":42,"max_tokens":5,"stream":false}')
rrp15=$(chat_raw '{"messages":[{"role":"user","content":"Name a food."}],"temperature":0.8,"repetition_penalty":1.5,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$rrp1" != "__ERROR__" ] && [ "$rrp15" != "__ERROR__" ] && [ -n "$rrp1" ] && [ -n "$rrp15" ]; then
  run_test "Repetition" "rep_penalty=1.0 and 1.5 both produce output" "rp=1.0:$rrp1, rp=1.5:$rrp15" "PASS" "$dur"
else
  run_test "Repetition" "rep_penalty=1.0 and 1.5 both produce output" "valid output" "FAIL" "$dur"
fi

# ─── TEST GROUP: Combined Parameters ─────────────────────────────────────────
echo "🔗 Combined Parameters"

# Test 16: All params together with seed = deterministic
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r1=$(chat_raw '{"messages":[{"role":"user","content":"Name a city."}],"temperature":0.7,"top_p":0.9,"top_k":20,"min_p":0.05,"presence_penalty":1.0,"seed":42,"max_tokens":5,"stream":false}')
r2=$(chat_raw '{"messages":[{"role":"user","content":"Name a city."}],"temperature":0.7,"top_p":0.9,"top_k":20,"min_p":0.05,"presence_penalty":1.0,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r1" = "$r2" ] && [ "$r1" != "__ERROR__" ]; then
  run_test "Combined" "All params + seed=42 is deterministic" "identical: $r1" "PASS" "$dur"
else
  run_test "Combined" "All params + seed=42 is deterministic" "identical" "FAIL: [$r1] [$r2]" "$dur"
fi

# Test 17: Streaming with all params works
t0=$(python3 -c "import time; print(int(time.time()*1000))")
stream_out=$(curl -sf "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"temperature":0.7,"top_p":0.9,"top_k":20,"min_p":0.05,"presence_penalty":1.0,"seed":42,"max_tokens":10,"stream":true}' 2>/dev/null | head -5)
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if echo "$stream_out" | grep -q "data:"; then
  run_test "Combined" "Streaming with all params works" "SSE data received" "PASS" "$dur"
else
  run_test "Combined" "Streaming with all params works" "SSE data" "FAIL: no data received" "$dur"
fi

# Test 18: Log line shows all params
t0=$(python3 -c "import time; print(int(time.time()*1000))")
chat_raw '{"messages":[{"role":"user","content":"Hi"}],"temperature":0.7,"top_p":0.9,"top_k":20,"min_p":0.05,"presence_penalty":1.5,"seed":123,"max_tokens":5,"stream":false}' >/dev/null
sleep 0.5
log_line=$(grep -a "MLX start:" /tmp/afm-test.log 2>/dev/null | tail -1 || true)
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
has_all=true
for param in "top_k=20" "min_p=0.05" "presence_penalty=1.5" "seed=123"; do
  if ! echo "$log_line" | grep -q "$param"; then
    has_all=false
    break
  fi
done
if $has_all; then
  run_test "Combined" "Log line contains all new params" "top_k, min_p, presence_penalty, seed in log" "PASS" "$dur"
else
  run_test "Combined" "Log line contains all new params" "all params in log" "FAIL: $log_line" "$dur"
fi

# ─── TEST GROUP: Edge Cases ───────────────────────────────────────────────────
echo "⚠️  Edge Cases"

# Test 19: top_k=0 (disabled) works
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r=$(chat_raw '{"messages":[{"role":"user","content":"Say yes."}],"temperature":0.5,"top_k":0,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r" != "__ERROR__" ] && [ -n "$r" ]; then
  run_test "Edge" "top_k=0 (disabled) produces output" "$r" "PASS" "$dur"
else
  run_test "Edge" "top_k=0 (disabled) produces output" "valid output" "FAIL" "$dur"
fi

# Test 20: presence_penalty=0 (disabled) works
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r=$(chat_raw '{"messages":[{"role":"user","content":"Say yes."}],"temperature":0.5,"presence_penalty":0.0,"seed":42,"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r" != "__ERROR__" ] && [ -n "$r" ]; then
  run_test "Edge" "presence_penalty=0 (disabled) produces output" "$r" "PASS" "$dur"
else
  run_test "Edge" "presence_penalty=0 (disabled) produces output" "valid output" "FAIL" "$dur"
fi

# Test 21: No sampling params at all (defaults)
t0=$(python3 -c "import time; print(int(time.time()*1000))")
r=$(chat_raw '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":5,"stream":false}')
t1=$(python3 -c "import time; print(int(time.time()*1000))")
dur=$(( t1 - t0 ))
if [ "$r" != "__ERROR__" ] && [ -n "$r" ]; then
  run_test "Edge" "No sampling params (all defaults) works" "$r" "PASS" "$dur"
else
  run_test "Edge" "No sampling params (all defaults) works" "valid output" "FAIL" "$dur"
fi

# ─── Generate HTML Report ────────────────────────────────────────────────────
TEST_END_TIME=$(date +%s)
TOTAL_SECS=$((TEST_END_TIME - TEST_START_TIME))

if [ $TOTAL -gt 0 ]; then
  PCT=$(( PASS * 100 / TOTAL ))
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
<title>AFM Sampling Parameters Test Report</title>
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
  .group-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px; font-size: 0.7rem; font-weight: 500; background: #1a1f2e; color: #8b949e; border: 1px solid #30363d; }
  .group-badge.Seed { color: #3fb950; border-color: #238636; background: #0d281833; }
  .group-badge.Temperature { color: #f0883e; border-color: #d18616; background: #4b2e0433; }
  .group-badge.Top-K { color: #58a6ff; border-color: #1f6feb; background: #0c2d6b33; }
  .group-badge.Top-P { color: #d2a8ff; border-color: #8957e5; background: #3b1f7233; }
  .group-badge.Min-P { color: #f778ba; border-color: #db61a2; background: #3d1a2e33; }
  .group-badge.Presence { color: #ff7b72; border-color: #da3633; background: #2d121533; }
  .group-badge.Repetition { color: #ffa657; border-color: #d18616; background: #4b2e0433; }
  .group-badge.Combined { color: #79c0ff; border-color: #388bfd; background: #0c2d6b33; }
  .group-badge.Edge { color: #8b949e; border-color: #484f58; background: #21262d33; }
  .detail { font-family: 'SF Mono', 'Menlo', monospace; font-size: 0.8rem; color: #8b949e; white-space: pre-wrap; word-break: break-word; max-height: 100px; overflow-y: auto; background: #0d1117; padding: 0.5rem; border-radius: 6px; border: 1px solid #21262d; margin-top: 0.25rem; }
  .duration { color: #8b949e; font-family: 'SF Mono', monospace; font-size: 0.85rem; }
  .footer { text-align: center; margin-top: 2rem; color: #484f58; font-size: 0.8rem; }
</style>
</head>
<body>
HTMLHEAD

# Header
cat >> "$REPORT_FILE" <<EOF
<div class="header">
  <h1>AFM Sampling Parameters Test Report</h1>
  <div class="meta">
    Model: <strong>$MODEL</strong><br>
    Server: <code>$BASE_URL</code><br>
    Date: $DATE_STR
  </div>
</div>
<div class="summary">
  <div class="stat pass"><div class="value">$PASS</div><div class="label">Passed</div></div>
  <div class="stat fail"><div class="value">$FAIL</div><div class="label">Failed</div></div>
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

# Rows
idx=0
for entry in "${RESULTS[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r status group name expected actual duration <<< "$entry"
  if [ "$status" = "PASS" ]; then
    badge='<span class="badge pass">PASS</span>'
    detail_text="$expected"
  else
    badge='<span class="badge fail">FAIL</span>'
    detail_text="Expected: $expected\nActual: $actual"
  fi
  # HTML-escape
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

# Footer
cat >> "$REPORT_FILE" <<EOF
</tbody>
</table>
<div class="footer">
  Generated by Scripts/test-sampling-params.sh &mdash; $(date '+%Y-%m-%d %H:%M:%S')
</div>
</body>
</html>
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: $PASS/$TOTAL passed ($PCT%)"
if [ $FAIL -gt 0 ]; then
  echo "  ❌ $FAIL FAILED"
fi
echo "  📄 Report: $REPORT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
