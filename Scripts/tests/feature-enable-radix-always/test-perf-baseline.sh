#!/usr/bin/env bash
# Performance comparison: installed afm (baseline) vs new build.
# Measures TTFT and tok/s on identical prompts.
set -euo pipefail

MODEL="${TEST_MODEL:-mlx-community/Qwen3.5-2B-bf16}"
CACHE_DIR="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
PORT_BASE=19877
PORT_NEW=19878
BASELINE_AFM="${BASELINE_AFM:-$(which afm)}"
NEW_AFM="${NEW_AFM:-.build/release/afm}"
RUNS=3
THRESHOLD=0.95  # new build must be >= 95% of baseline

log()  { printf "\033[1;34m[PERF]\033[0m %s\n" "$1"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$1"; }

cleanup() {
    kill "$BASE_PID" 2>/dev/null || true
    kill "$NEW_PID" 2>/dev/null || true
    wait "$BASE_PID" 2>/dev/null || true
    wait "$NEW_PID" 2>/dev/null || true
}
trap cleanup EXIT

timed_request() {
    local port="$1" msg="$2" max_tokens="${3:-64}"
    local start end resp
    start=$(python3 -c "import time; print(int(time.time()*1000))")
    resp=$(curl -s -w "\n%{time_starttransfer}" "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"test\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${msg}\"}],
            \"max_tokens\": ${max_tokens},
            \"temperature\": 0
        }")
    end=$(python3 -c "import time; print(int(time.time()*1000))")

    local body ttfb_s
    body=$(echo "$resp" | head -n -1)
    ttfb_s=$(echo "$resp" | tail -1)

    local ttfb_ms total_ms prompt_tok comp_tok
    ttfb_ms=$(python3 -c "print(int(float('${ttfb_s}') * 1000))")
    total_ms=$(( end - start ))
    prompt_tok=$(echo "$body" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(u.get('prompt_tokens',0))" 2>/dev/null || echo 0)
    comp_tok=$(echo "$body" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(u.get('completion_tokens',0))" 2>/dev/null || echo 0)

    echo "${ttfb_ms},${total_ms},${prompt_tok},${comp_tok}"
}

wait_ready() {
    local port="$1" label="$2"
    for i in $(seq 1 90); do
        if curl -s "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: ${label} server did not start on port ${port}"
    exit 1
}

# ─── Validate binaries ───────────────────────────────────────────────
log "Baseline: $BASELINE_AFM ($(${BASELINE_AFM} --version 2>/dev/null || echo 'unknown'))"
log "New build: $NEW_AFM"
if [ ! -f "$NEW_AFM" ]; then
    log "Building release binary..."
    swift build -c release 2>&1 | tail -3
fi

PROMPTS=(
    "What is the capital of Japan? Answer briefly."
    "Explain what a radix tree is in two sentences."
    "Write a Python function that checks if a string is a palindrome."
)

# ─── Run baseline ────────────────────────────────────────────────────
log "Starting baseline server on port $PORT_BASE..."
MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" \
    "$BASELINE_AFM" mlx -m "$MODEL" --port "$PORT_BASE" --enable-prefix-caching >/dev/null 2>&1 &
BASE_PID=$!
wait_ready "$PORT_BASE" "Baseline"
log "Baseline server ready."

# Warmup
timed_request "$PORT_BASE" "warmup" 8 >/dev/null

declare -a BASE_TTFB BASE_TOTAL BASE_TOKS
for prompt in "${PROMPTS[@]}"; do
    log "  Baseline: '${prompt:0:40}...' (${RUNS} runs)"
    for run in $(seq 1 $RUNS); do
        result=$(timed_request "$PORT_BASE" "$prompt" 64)
        BASE_TTFB+=("$(echo "$result" | cut -d, -f1)")
        BASE_TOTAL+=("$(echo "$result" | cut -d, -f2)")
        BASE_TOKS+=("$(echo "$result" | cut -d, -f4)")
    done
done

# Also test repeated-prompt scenario (cache hit)
log "  Baseline: repeated prompt (cache hit test, ${RUNS} runs)"
for run in $(seq 1 $RUNS); do
    result=$(timed_request "$PORT_BASE" "What is the capital of Japan? Answer briefly." 64)
    BASE_TTFB+=("$(echo "$result" | cut -d, -f1)")
    BASE_TOTAL+=("$(echo "$result" | cut -d, -f2)")
    BASE_TOKS+=("$(echo "$result" | cut -d, -f4)")
done

kill "$BASE_PID" 2>/dev/null; wait "$BASE_PID" 2>/dev/null || true
log "Baseline done."

# ─── Run new build ───────────────────────────────────────────────────
log "Starting new build server on port $PORT_NEW..."
MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" \
    "$NEW_AFM" mlx -m "$MODEL" --port "$PORT_NEW" >/dev/null 2>&1 &
NEW_PID=$!
wait_ready "$PORT_NEW" "New build"
log "New build server ready."

# Warmup
timed_request "$PORT_NEW" "warmup" 8 >/dev/null

declare -a NEW_TTFB NEW_TOTAL NEW_TOKS
for prompt in "${PROMPTS[@]}"; do
    log "  New build: '${prompt:0:40}...' (${RUNS} runs)"
    for run in $(seq 1 $RUNS); do
        result=$(timed_request "$PORT_NEW" "$prompt" 64)
        NEW_TTFB+=("$(echo "$result" | cut -d, -f1)")
        NEW_TOTAL+=("$(echo "$result" | cut -d, -f2)")
        NEW_TOKS+=("$(echo "$result" | cut -d, -f4)")
    done
done

log "  New build: repeated prompt (cache hit test, ${RUNS} runs)"
for run in $(seq 1 $RUNS); do
    result=$(timed_request "$PORT_NEW" "What is the capital of Japan? Answer briefly." 64)
    NEW_TTFB+=("$(echo "$result" | cut -d, -f1)")
    NEW_TOTAL+=("$(echo "$result" | cut -d, -f2)")
    NEW_TOKS+=("$(echo "$result" | cut -d, -f4)")
done

kill "$NEW_PID" 2>/dev/null; wait "$NEW_PID" 2>/dev/null || true
log "New build done."

# ─── Compute medians and compare ─────────────────────────────────────
compute_median() {
    local -n arr=$1
    printf '%s\n' "${arr[@]}" | sort -n | awk 'NR==int((NR+1)/2){print}'
}

BASE_TTFB_MED=$(compute_median BASE_TTFB)
NEW_TTFB_MED=$(compute_median NEW_TTFB)
BASE_TOTAL_MED=$(compute_median BASE_TOTAL)
NEW_TOTAL_MED=$(compute_median NEW_TOTAL)

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  PERFORMANCE COMPARISON"
echo "═══════════════════════════════════════════════════════════"
printf "  %-20s %10s %10s %10s\n" "Metric" "Baseline" "New" "Ratio"
echo "  ────────────────────────────────────────────────────────"
if [ "$BASE_TTFB_MED" -gt 0 ]; then
    TTFB_RATIO=$(python3 -c "print(f'{${BASE_TTFB_MED}/${NEW_TTFB_MED}:.2f}')")
    printf "  %-20s %8sms %8sms %9sx\n" "TTFB (median)" "$BASE_TTFB_MED" "$NEW_TTFB_MED" "$TTFB_RATIO"
fi
if [ "$BASE_TOTAL_MED" -gt 0 ]; then
    TOTAL_RATIO=$(python3 -c "print(f'{${BASE_TOTAL_MED}/${NEW_TOTAL_MED}:.2f}')")
    printf "  %-20s %8sms %8sms %9sx\n" "Total (median)" "$BASE_TOTAL_MED" "$NEW_TOTAL_MED" "$TOTAL_RATIO"
fi
echo "═══════════════════════════════════════════════════════════"
echo ""

# ─── Pass/Fail ────────────────────────────────────────────────────────
MAX_TTFB=$(python3 -c "import math; print(math.ceil(${BASE_TTFB_MED} / ${THRESHOLD}))")
if [ "$NEW_TTFB_MED" -le "$MAX_TTFB" ]; then
    printf "\033[1;32m  PASS\033[0m TTFB: %sms <= %sms (threshold: %.0f%% of baseline)\n" "$NEW_TTFB_MED" "$MAX_TTFB" "$(python3 -c "print(${THRESHOLD}*100)")"
else
    printf "\033[1;31m  FAIL\033[0m TTFB regression: %sms > %sms (threshold: %.0f%% of baseline)\n" "$NEW_TTFB_MED" "$MAX_TTFB" "$(python3 -c "print(${THRESHOLD}*100)")"
    exit 1
fi

MAX_TOTAL=$(python3 -c "import math; print(math.ceil(${BASE_TOTAL_MED} / ${THRESHOLD}))")
if [ "$NEW_TOTAL_MED" -le "$MAX_TOTAL" ]; then
    printf "\033[1;32m  PASS\033[0m Total: %sms <= %sms (threshold: %.0f%% of baseline)\n" "$NEW_TOTAL_MED" "$MAX_TOTAL" "$(python3 -c "print(${THRESHOLD}*100)")"
else
    printf "\033[1;31m  FAIL\033[0m Total regression: %sms > %sms (threshold: %.0f%% of baseline)\n" "$NEW_TOTAL_MED" "$MAX_TOTAL" "$(python3 -c "print(${THRESHOLD}*100)")"
    exit 1
fi

echo ""
log "All performance checks passed."
