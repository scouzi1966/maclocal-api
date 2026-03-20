#!/usr/bin/env bash
# Comprehensive functional + performance tests with GPU monitoring.
# Each request reports: TTFB, total time, tok/s, and GPU utilization.
# Servers run SEQUENTIALLY.
set -euo pipefail

MODEL="${TEST_MODEL:-mlx-community/Qwen3.5-35B-A3B-4bit}"
CACHE_DIR="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
PORT=19877
BASELINE_AFM="${BASELINE_AFM:-$(which afm)}"
NEW_AFM="${NEW_AFM:-.build/release/afm}"
RUNS=3
THRESHOLD=0.95

PASS=0; FAIL=0; WARN=0
TESTS=()
SERVER_PID=""

log()  { printf "\033[1;34m[TEST]\033[0m %s\n" "$1"; }
pass() { PASS=$((PASS+1)); TESTS+=("PASS: $1"); printf "\033[1;32m  PASS\033[0m %s\n" "$1"; }
fail() { FAIL=$((FAIL+1)); TESTS+=("FAIL: $1"); printf "\033[1;31m  FAIL\033[0m %s\n" "$1"; }
warn() { WARN=$((WARN+1)); TESTS+=("WARN: $1"); printf "\033[1;33m  WARN\033[0m %s\n" "$1"; }

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
    pkill -f "mactop --headless" 2>/dev/null || true
}
trap cleanup EXIT

start_server() {
    local binary="$1" label="$2" extra="${3:-}"
    cleanup
    log "Starting $label on port $PORT..."
    MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" "$binary" mlx -m "$MODEL" --port "$PORT" $extra >/dev/null 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 120); do curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && break; sleep 1; done
    curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 || { echo "ERROR: $label failed to start"; exit 1; }
    log "$label ready (PID $SERVER_PID)."
}

# ─── GPU-monitored request ────────────────────────────────────────────
# Runs mactop in background during the curl request, reports GPU stats.
# Returns: ttfb_ms,total_ms,prompt_tok,comp_tok,gpu_avg,gpu_max,gpu_power_avg
gpu_request() {
    local msg="$1" max_tokens="${2:-4096}" temperature="${3:-0}" extra="${4:-}"
    local gpu_file=$(mktemp /tmp/gpu_sample.XXXXXX)

    # Start GPU sampling at 100ms intervals
    mactop --headless --interval 100 --count 200 2>/dev/null > "$gpu_file" &
    local mactop_pid=$!
    sleep 0.1  # let mactop start

    local start end
    start=$(python3 -c "import time; print(int(time.time()*1000))")
    local resp
    resp=$(curl -s -w "\nTTFB:%{time_starttransfer}" "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"test\",
            \"messages\": [{\"role\": \"user\", \"content\": $(printf '%s' "$msg" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}],
            \"max_tokens\": ${max_tokens},
            \"temperature\": ${temperature}
            ${extra:+,$extra}
        }")
    end=$(python3 -c "import time; print(int(time.time()*1000))")

    # Stop mactop
    kill "$mactop_pid" 2>/dev/null; wait "$mactop_pid" 2>/dev/null || true

    local ttfb_s body ttfb_ms total_ms prompt_tok comp_tok
    ttfb_s=$(echo "$resp" | grep "^TTFB:" | sed 's/^TTFB://')
    body=$(echo "$resp" | grep -v "^TTFB:")
    ttfb_ms=$(python3 -c "print(int(float('${ttfb_s}') * 1000))")
    total_ms=$(( end - start ))
    prompt_tok=$(echo "$body" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(u.get('prompt_tokens',0))" 2>/dev/null || echo 0)
    comp_tok=$(echo "$body" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(u.get('completion_tokens',0))" 2>/dev/null || echo 0)

    # Parse GPU samples (mactop outputs a single JSON array)
    local gpu_stats
    gpu_stats=$(python3 -c "
import json
with open('${gpu_file}') as f:
    try:
        data = json.load(f)
    except:
        print('0,0,0.0'); exit()
samples = [(d.get('gpu_usage',0), d.get('soc_metrics',{}).get('gpu_power',0)) for d in data]
if samples:
    gpus = [s[0] for s in samples]
    powers = [s[1] for s in samples]
    print(f'{sum(gpus)/len(gpus):.0f},{max(gpus):.0f},{sum(powers)/len(powers):.1f}')
else:
    print('0,0,0.0')
" 2>/dev/null || echo "0,0,0.0")

    rm -f "$gpu_file"
    echo "${ttfb_ms},${total_ms},${prompt_tok},${comp_tok},${gpu_stats}"
}

# Print one result line
print_result() {
    local label="$1" result="$2"
    local ttfb total pp tg gpu_avg gpu_max gpu_power
    IFS=',' read -r ttfb total pp tg gpu_avg gpu_max gpu_power <<< "$result"
    local tps=0
    [ "$tg" -gt 0 ] && [ "$total" -gt "$ttfb" ] && tps=$(python3 -c "print(f'{${tg}/((${total}-${ttfb})/1000):.1f}')" 2>/dev/null || echo "0")
    printf "  %-40s  TTFB:%5sms  Total:%6sms  %4spp/%4stg  %5s tok/s  GPU avg:%3s%% max:%3s%% power:%5sW\n" \
        "$label" "$ttfb" "$total" "$pp" "$tg" "$tps" "$gpu_avg" "$gpu_max" "$gpu_power"
}

extract_content() {
    python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null
}

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  ╔═══════════════════════════════════════════════════════════════╗"
echo "  ║  Comprehensive Test Suite with GPU Monitoring                ║"
echo "  ║  Model: $MODEL"
echo "  ║  Baseline: $(${BASELINE_AFM} --version 2>/dev/null || echo '?') | New: .build/release/afm"
echo "  ╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Build
log "Building release binary..."
swift build -c release 2>&1 | tail -3

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: BASELINE
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  ════════════════════════════════════════════════════════════════════════════════════════════════════════"
echo "  PHASE 1: BASELINE (installed afm + --enable-prefix-caching)"
echo "  ════════════════════════════════════════════════════════════════════════════════════════════════════════"
start_server "$BASELINE_AFM" "Baseline" "--enable-prefix-caching"

# Warmup (don't measure)
log "Warmup..."
gpu_request "warmup" 32 >/dev/null

declare -A BASE_RESULTS
BENCHMARKS=(
    "P1:Short Q&A:What is the capital of Japan?:4096"
    "P2:Medium gen:Explain what a radix tree is.:4096"
    "P3:Code gen:Write a Python function that checks if a number is prime. Just the code.:4096"
    "P4:Cache hit:What is the capital of Japan?:4096"
    "P5:Long prompt:LONG:4096"
    "P6:Reasoning:If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly? Explain briefly.:4096"
    "P7:Creative:Write a haiku about the ocean.:4096"
    "P8:Summarize:The ECB raised rates by 25bp to 4.50 percent. Lagarde cited persistent inflation. Summarize in one sentence.:4096"
    "P9:Strict:List exactly 3 fruits. One per line. No numbering.:4096"
    "P10:Max tokens:Write a long essay about AI.:8"
)

# Build long prompt
LONG_PROMPT="Here is context. "
for i in $(seq 1 50); do LONG_PROMPT+="The quick brown fox jumps over the lazy dog. Sentence $i. "; done
LONG_PROMPT+="What was the last sentence number?"

echo ""
printf "  %-40s  %7s  %8s  %9s  %10s  %9s  %9s  %8s\n" "Benchmark" "TTFB" "Total" "pp/tg" "tok/s" "GPU avg" "GPU max" "Power"
echo "  ────────────────────────────────────────────────────────────────────────────────────────────────────────"

for bench in "${BENCHMARKS[@]}"; do
    IFS=':' read -r id name prompt maxtok <<< "$bench"
    [ "$prompt" = "LONG" ] && prompt="$LONG_PROMPT"

    # Best of RUNS
    best_total=999999
    best_result=""
    for run in $(seq 1 $RUNS); do
        result=$(gpu_request "$prompt" "$maxtok")
        total=$(echo "$result" | cut -d, -f2)
        if [ "$total" -lt "$best_total" ]; then
            best_total=$total
            best_result="$result"
        fi
    done
    BASE_RESULTS[$id]="$best_result"
    print_result "BASE $id: $name" "$best_result"
done

cleanup
log "Baseline done."
sleep 2

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: NEW BUILD — functional + perf
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  ════════════════════════════════════════════════════════════════════════════════════════════════════════"
echo "  PHASE 2: NEW BUILD (radix always on)"
echo "  ════════════════════════════════════════════════════════════════════════════════════════════════════════"
start_server "$NEW_AFM" "New build"

log "Warmup..."
gpu_request "warmup" 32 >/dev/null

# ─── Functional ───────────────────────────────────────────────────────
echo ""
echo "  ── Functional Tests ──"
echo ""

log "F1: Basic chat (2+2)"
RESP=$(curl -s "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":4096,"temperature":0}')
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  content: {r[\"choices\"][0][\"message\"][\"content\"][:60]}  tokens: {r[\"usage\"][\"completion_tokens\"]}')" 2>/dev/null
echo "$RESP" | extract_content | grep -q "4" && pass "F1: Basic chat" || fail "F1: Basic chat"

log "F2: Response structure"
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); assert all(k in r for k in ['id','choices','usage']); print('  structure OK')" 2>/dev/null && pass "F2: Response structure" || fail "F2: Response structure"

log "F3: Streaming"
STREAM=$(curl -sN "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Count 1 to 5."}],"max_tokens":4096,"temperature":0,"stream":true}')
CHUNKS=$(echo "$STREAM" | grep -c "^data: {" || true)
echo "$STREAM" | grep -q "data: \[DONE\]" && pass "F3: Streaming ($CHUNKS chunks)" || fail "F3: Streaming"

log "F4: Determinism (temp=0)"
R1=$(curl -s "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Capital of Japan? One word."}],"max_tokens":4096,"temperature":0}' | extract_content)
R2=$(curl -s "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Capital of Japan? One word."}],"max_tokens":4096,"temperature":0}' | extract_content)
[ "$R1" = "$R2" ] && pass "F4: Determinism" || warn "F4: Determinism — outputs differ"

log "F5: Max tokens (limit=8)"
TOK=$(curl -s "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Write a long essay."}],"max_tokens":8,"temperature":0}' | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
echo "  completion_tokens=$TOK"
[ "$TOK" -le 10 ] && pass "F5: Max tokens ($TOK)" || fail "F5: Max tokens ($TOK)"

log "F6: Stop sequence"
REASON=$(curl -s "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}],"max_tokens":4096,"temperature":0,"stop":["5"]}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null)
echo "  finish_reason=$REASON"
[ "$REASON" = "stop" ] && pass "F6: Stop sequence" || warn "F6: Stop ($REASON)"

log "F7: Multi-turn"
MULTI=$(curl -s "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"My name is Alice."},{"role":"assistant","content":"Nice to meet you, Alice!"},{"role":"user","content":"What is my name?"}],"max_tokens":4096,"temperature":0}' | extract_content)
echo "  content: ${MULTI:0:60}"
echo "$MULTI" | grep -iq "alice" && pass "F7: Multi-turn" || fail "F7: Multi-turn"

# ─── Performance with GPU monitoring ─────────────────────────────────
echo ""
echo "  ── Performance Benchmarks (with GPU monitoring) ──"
echo ""
printf "  %-40s  %7s  %8s  %9s  %10s  %9s  %9s  %8s\n" "Benchmark" "TTFB" "Total" "pp/tg" "tok/s" "GPU avg" "GPU max" "Power"
echo "  ────────────────────────────────────────────────────────────────────────────────────────────────────────"

declare -A NEW_RESULTS
for bench in "${BENCHMARKS[@]}"; do
    IFS=':' read -r id name prompt maxtok <<< "$bench"
    [ "$prompt" = "LONG" ] && prompt="$LONG_PROMPT"

    best_total=999999
    best_result=""
    for run in $(seq 1 $RUNS); do
        result=$(gpu_request "$prompt" "$maxtok")
        total=$(echo "$result" | cut -d, -f2)
        if [ "$total" -lt "$best_total" ]; then
            best_total=$total
            best_result="$result"
        fi
    done
    NEW_RESULTS[$id]="$best_result"
    print_result "NEW  $id: $name" "$best_result"
done

cleanup

# ═══════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "  ════════════════════════════════════════════════════════════════════════════════════════════════════════"
echo "  COMPARISON: BASELINE vs NEW BUILD"
echo "  ════════════════════════════════════════════════════════════════════════════════════════════════════════"
printf "  %-12s  %8s %8s %6s  %8s %8s %6s  %8s %8s\n" \
    "Benchmark" "B-TTFB" "N-TTFB" "Ratio" "B-Total" "N-Total" "Ratio" "B-GPU%" "N-GPU%"
echo "  ────────────────────────────────────────────────────────────────────────────────────────────────────────"

for bench in "${BENCHMARKS[@]}"; do
    IFS=':' read -r id name prompt maxtok <<< "$bench"
    b="${BASE_RESULTS[$id]}"
    n="${NEW_RESULTS[$id]}"
    bttfb=$(echo "$b" | cut -d, -f1); nttfb=$(echo "$n" | cut -d, -f1)
    btotal=$(echo "$b" | cut -d, -f2); ntotal=$(echo "$n" | cut -d, -f2)
    bgpu=$(echo "$b" | cut -d, -f5); ngpu=$(echo "$n" | cut -d, -f5)

    ttfb_r=$(python3 -c "print(f'{${bttfb}/${nttfb}:.2f}' if ${nttfb}>0 else '?')")
    total_r=$(python3 -c "print(f'{${btotal}/${ntotal}:.2f}' if ${ntotal}>0 else '?')")

    printf "  %-12s  %6sms %6sms %5sx  %6sms %6sms %5sx  %6s%% %6s%%\n" \
        "$id" "$bttfb" "$nttfb" "$ttfb_r" "$btotal" "$ntotal" "$total_r" "$bgpu" "$ngpu"

    max_ttfb=$(python3 -c "import math; print(math.ceil(${bttfb} / ${THRESHOLD}))")
    [ "$nttfb" -le "$max_ttfb" ] && pass "$id: TTFB ${nttfb}ms <= ${max_ttfb}ms" || fail "$id: TTFB regression ${nttfb}ms > ${max_ttfb}ms"
done

echo ""
echo "  Ratio >1.00 = new build faster. Threshold: new must be >= ${THRESHOLD}x baseline."
echo ""
echo "  ═══════════════════════════════════════════════════════════════"
echo "  FINAL: ${PASS} passed, ${FAIL} failed, ${WARN} warnings"
echo "  ═══════════════════════════════════════════════════════════════"
for t in "${TESTS[@]}"; do echo "  $t"; done
echo ""
[ "$FAIL" -gt 0 ] && { echo "  OVERALL: FAIL"; exit 1; } || echo "  OVERALL: PASS"
