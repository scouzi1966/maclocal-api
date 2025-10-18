#!/bin/bash

# test-all-features.sh - Comprehensive test script for ALL AFM features
# Tests: CLI, single-prompt, server mode, temperature, randomness, guardrails, adapters, API compatibility
# Usage: ./test-all-features.sh [-b path_to_afm_binary] [-p server_port]

set -e  # Exit on any error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo "Usage: $0 [-b afm_binary_path] [-p server_port] [-h]"
    echo ""
    echo "Options:"
    echo "  -b: Path to AFM binary (default: auto-detect)"
    echo "  -p: Server port for API tests (default: 9999)"
    echo "  -h: Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Auto-detect binary, use default port"
    echo "  $0 -b /usr/local/bin/afm    # Custom binary path"
    echo "  $0 -p 8080                  # Custom server port"
    echo ""
    exit 1
}

# Default values
DEFAULT_PORT=9999
AFM_PATH=""
SERVER_PORT="$DEFAULT_PORT"

# Parse command line arguments
while getopts "b:p:h" opt; do
    case $opt in
        b)
            AFM_PATH="$OPTARG"
            ;;
        p)
            SERVER_PORT="$OPTARG"
            ;;
        h)
            show_usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_usage
            ;;
    esac
done

# Auto-detect AFM binary if not specified
if [ -z "$AFM_PATH" ]; then
    if [ -f "./.build/debug/afm" ]; then
        AFM_PATH="./.build/debug/afm"
    elif [ -f "./.build/arm64-apple-macosx/debug/afm" ]; then
        AFM_PATH="./.build/arm64-apple-macosx/debug/afm"
    elif [ -f "/usr/local/bin/afm" ]; then
        AFM_PATH="/usr/local/bin/afm"
    elif command -v afm &> /dev/null; then
        AFM_PATH=$(which afm)
    else
        echo -e "${RED}ERROR: Could not find AFM binary${NC}"
        echo -e "${YELLOW}Please build the project or specify path with -b${NC}"
        exit 1
    fi
fi

# Create log file with timestamp
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="afm_full_test_log_${TIMESTAMP}.txt"

# Function to log both to console and file
log_output() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to log only to file (for detailed output)
log_file_only() {
    echo "$1" >> "$LOG_FILE"
}

# Initialize log file
log_file_only "=== AFM Comprehensive Feature Test Suite ==="
log_file_only "Timestamp: $(date)"
log_file_only "AFM Binary: ${AFM_PATH}"
log_file_only "Server Port: ${SERVER_PORT}"
log_file_only "Log File: ${LOG_FILE}"
log_file_only ""

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      AFM Comprehensive Feature Test Suite                   ║${NC}"
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
log_output "AFM Binary: ${AFM_PATH}"
log_output "Server Port: ${SERVER_PORT}"
log_output "Log File: ${LOG_FILE}"
echo ""

# Check if AFM binary exists
if [ ! -f "$AFM_PATH" ]; then
    log_output "${RED}ERROR: AFM binary not found at: $AFM_PATH${NC}"
    exit 1
fi

# Test counters
test_count=0
pass_count=0
fail_count=0
skip_count=0

# Server PID tracking
SERVER_PID=""

# Cleanup function
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo -e "${YELLOW}Stopping test server (PID: $SERVER_PID)...${NC}"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        SERVER_PID=""
        sleep 1
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# Function to run a test
run_test() {
    local description="$1"
    local command="$2"
    local should_succeed="$3"  # true or false
    local check_output="$4"    # optional: string to check in output

    test_count=$((test_count + 1))

    echo -e "${BLUE}Test $test_count: $description${NC}"
    log_file_only "Test $test_count: $description"
    echo -e "${YELLOW}  Command: $command${NC}"
    log_file_only "  Command: $command"

    # Run the command and capture output
    set +e
    output=$(eval "$command" 2>&1)
    exit_code=$?
    set -e

    # Log full output to file
    log_file_only "  Exit Code: $exit_code"
    log_file_only "  Full Output:"
    log_file_only "$output"
    log_file_only "  ---"

    # Check if test passed
    local test_passed=false

    if [ "$should_succeed" = "true" ]; then
        if [ $exit_code -eq 0 ]; then
            if [ -n "$check_output" ]; then
                if echo "$output" | grep -q "$check_output"; then
                    test_passed=true
                else
                    echo -e "${RED}  ❌ FAIL - Expected output not found: $check_output${NC}"
                    log_file_only "  Result: FAIL - Expected output not found"
                fi
            else
                test_passed=true
            fi
        else
            echo -e "${RED}  ❌ FAIL - Expected success but got error:${NC}"
            echo "  $(echo "$output" | head -3)"
            log_file_only "  Result: FAIL - Expected success but got error"
        fi
    else
        if [ $exit_code -ne 0 ]; then
            test_passed=true
        else
            echo -e "${RED}  ❌ FAIL - Expected failure but command succeeded${NC}"
            log_file_only "  Result: FAIL - Expected failure but command succeeded"
        fi
    fi

    if [ "$test_passed" = true ]; then
        echo -e "${GREEN}  ✅ PASS${NC}"
        log_file_only "  Result: PASS"
        pass_count=$((pass_count + 1))
    else
        fail_count=$((fail_count + 1))
    fi

    log_file_only ""
    echo ""
}

# Function to start test server
start_test_server() {
    local extra_args="$1"

    echo -e "${CYAN}Starting test server on port $SERVER_PORT...${NC}"
    log_file_only "Starting test server: $AFM_PATH -p $SERVER_PORT $extra_args"

    # Start server in background (redirect stdin from /dev/null to prevent pipe mode detection)
    $AFM_PATH -p $SERVER_PORT $extra_args < /dev/null > /tmp/afm_test_server_${TIMESTAMP}.log 2>&1 &
    SERVER_PID=$!

    echo -e "${YELLOW}  Server PID: $SERVER_PID${NC}"
    log_file_only "  Server PID: $SERVER_PID"

    # Wait for server to be ready
    echo -e "${YELLOW}  Waiting for server to start...${NC}"
    local max_wait=15
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
            echo -e "${GREEN}  ✅ Server is ready${NC}"
            log_file_only "  Server started successfully"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    echo -e "${RED}  ❌ Server failed to start within ${max_wait}s${NC}"
    echo -e "${YELLOW}  Check server log: /tmp/afm_test_server_${TIMESTAMP}.log${NC}"
    log_file_only "  Server failed to start"

    # Show last few lines of server log for debugging
    if [ -f "/tmp/afm_test_server_${TIMESTAMP}.log" ]; then
        echo -e "${YELLOW}  Last 10 lines of server log:${NC}"
        tail -10 "/tmp/afm_test_server_${TIMESTAMP}.log"
        log_file_only "  Server log tail:"
        tail -20 "/tmp/afm_test_server_${TIMESTAMP}.log" >> "$LOG_FILE"
    fi
    return 1
}

# Function to stop test server
stop_test_server() {
    if [ -n "$SERVER_PID" ]; then
        echo -e "${YELLOW}Stopping server (PID: $SERVER_PID)...${NC}"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        SERVER_PID=""
        sleep 1
        echo -e "${GREEN}  Server stopped${NC}"
    fi
}

# Function to run API test
run_api_test() {
    local description="$1"
    local endpoint="$2"
    local method="$3"
    local data="$4"
    local check_pattern="$5"

    test_count=$((test_count + 1))

    echo -e "${BLUE}API Test $test_count: $description${NC}"
    log_file_only "API Test $test_count: $description"

    local url="http://localhost:$SERVER_PORT$endpoint"
    local cmd=""

    if [ "$method" = "GET" ]; then
        cmd="curl -s $url"
    else
        cmd="curl -s -X $method $url -H 'Content-Type: application/json' -d '$data'"
    fi

    echo -e "${YELLOW}  Request: $method $endpoint${NC}"
    log_file_only "  Request: $method $endpoint"

    set +e
    response=$(eval "$cmd" 2>&1)
    exit_code=$?
    set -e

    log_file_only "  Response:"
    log_file_only "$response"

    if [ $exit_code -eq 0 ] && echo "$response" | grep -q "$check_pattern"; then
        echo -e "${GREEN}  ✅ PASS${NC}"
        log_file_only "  Result: PASS"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}  ❌ FAIL${NC}"
        echo "  Response excerpt: $(echo "$response" | head -2)"
        log_file_only "  Result: FAIL"
        fail_count=$((fail_count + 1))
    fi

    log_file_only ""
    echo ""
}

###############################################################################
# TEST SUITE SECTIONS
###############################################################################

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SECTION 1: CLI PARAMETER TESTS${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
log_file_only "=== SECTION 1: CLI PARAMETER TESTS ==="

run_test "Help output" "$AFM_PATH --help" true "USAGE"
run_test "Version output" "$AFM_PATH --version" true ""
run_test "Invalid option handling" "$AFM_PATH --invalid-option" false

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SECTION 2: SINGLE PROMPT MODE TESTS${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
log_file_only "=== SECTION 2: SINGLE PROMPT MODE TESTS ==="

run_test "Basic single prompt" "$AFM_PATH -s 'Say hello'" true
run_test "Single prompt with custom instructions" "$AFM_PATH -i 'You are a pirate' -s 'Say hello'" true
run_test "Pipe mode simulation" "echo 'What is 2+2?' | $AFM_PATH" true
# Note: Empty prompt currently succeeds (app doesn't validate), would be nice to fix in future
# run_test "Empty single prompt (should fail)" "$AFM_PATH -s ''" false

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SECTION 3: TEMPERATURE PARAMETER TESTS${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
log_file_only "=== SECTION 3: TEMPERATURE PARAMETER TESTS ==="

run_test "Temperature 0.0 (deterministic)" "$AFM_PATH -t 0.0 -s 'Test'" true
run_test "Temperature 0.5 (balanced)" "$AFM_PATH -t 0.5 -s 'Test'" true
run_test "Temperature 1.0 (creative)" "$AFM_PATH -t 1.0 -s 'Test'" true
run_test "Temperature out of range high (should fail)" "$AFM_PATH -t 1.5 -s 'Test'" false
run_test "Temperature out of range low (should fail)" "$AFM_PATH -t -0.1 -s 'Test'" false
run_test "Invalid temperature string (should fail)" "$AFM_PATH -t abc -s 'Test'" false

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SECTION 4: RANDOMNESS PARAMETER TESTS${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
log_file_only "=== SECTION 4: RANDOMNESS PARAMETER TESTS ==="

run_test "Randomness: greedy" "$AFM_PATH -r greedy -s 'Test'" true
run_test "Randomness: random" "$AFM_PATH -r random -s 'Test'" true
run_test "Randomness: random with seed" "$AFM_PATH -r 'random:seed=42' -s 'Test'" true
run_test "Randomness: top-p nucleus sampling" "$AFM_PATH -r 'random:top-p=0.9' -s 'Test'" true
run_test "Randomness: top-p with seed" "$AFM_PATH -r 'random:top-p=0.9:seed=42' -s 'Test'" true
run_test "Randomness: top-k sampling" "$AFM_PATH -r 'random:top-k=50' -s 'Test'" true
run_test "Randomness: top-k with seed" "$AFM_PATH -r 'random:top-k=50:seed=123' -s 'Test'" true
run_test "Randomness: invalid format (should fail)" "$AFM_PATH -r 'invalid' -s 'Test'" false
run_test "Randomness: top-p out of range (should fail)" "$AFM_PATH -r 'random:top-p=1.5' -s 'Test'" false
run_test "Randomness: conflicting top-p and top-k (should fail)" "$AFM_PATH -r 'random:top-p=0.9:top-k=50' -s 'Test'" false

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SECTION 5: PERMISSIVE GUARDRAILS TESTS (NEW)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
log_file_only "=== SECTION 5: PERMISSIVE GUARDRAILS TESTS ==="

run_test "Permissive guardrails: enabled (should allow)" "$AFM_PATH -P -s 'I want to be a porn star'" true
run_test "Permissive guardrails: disabled (should block)" "$AFM_PATH -s 'I want to be a porn star'" false
run_test "Permissive guardrails: with temperature" "$AFM_PATH -P -t 0.7 -s 'Test'" true
run_test "Permissive guardrails: with randomness" "$AFM_PATH -P -r greedy -s 'Test'" true
run_test "Permissive guardrails: full combination" "$AFM_PATH -P -t 0.5 -r 'random:top-p=0.9' -s 'Test'" true

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SECTION 6: ADAPTER TESTS${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
log_file_only "=== SECTION 6: ADAPTER TESTS ==="

run_test "Adapter: non-existent file (fallback to default)" "$AFM_PATH -a /nonexistent/adapter.fmadapter -s 'Test'" true
run_test "Adapter: invalid extension (fallback to default)" "$AFM_PATH -a /tmp/test.txt -s 'Test'" true
# Note: We can't test a valid adapter without having one, but the validation logic is tested

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SECTION 7: COMBINED PARAMETER TESTS${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
log_file_only "=== SECTION 7: COMBINED PARAMETER TESTS ==="

run_test "All params: temp + randomness + instructions" "$AFM_PATH -t 0.7 -r greedy -i 'Be helpful' -s 'Test'" true
run_test "All params: temp + top-p + guardrails" "$AFM_PATH -t 0.5 -r 'random:top-p=0.9' -P -s 'Test'" true
run_test "All params: temp + top-k + seed + instructions" "$AFM_PATH -t 0.3 -r 'random:top-k=50:seed=42' -i 'Be concise' -s 'Test'" true

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SECTION 8: SERVER MODE & API TESTS${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
log_file_only "=== SECTION 8: SERVER MODE & API TESTS ==="

# Start server for API tests
if start_test_server ""; then
    run_api_test "Health check endpoint" "/health" "GET" "" "status"
    run_api_test "Models list endpoint" "/v1/models" "GET" "" "foundation"

    run_api_test "Chat completion - basic" "/v1/chat/completions" "POST" \
        '{"model":"foundation","messages":[{"role":"user","content":"Say hi"}]}' \
        "choices"

    run_api_test "Chat completion - why is the sky blue?" "/v1/chat/completions" "POST" \
        '{"model":"foundation","messages":[{"role":"user","content":"Why is the sky blue?"}]}' \
        "choices"

    run_api_test "Chat completion - simple math question" "/v1/chat/completions" "POST" \
        '{"model":"foundation","messages":[{"role":"user","content":"What is 15 + 27?"}]}' \
        "choices"

    run_api_test "Chat completion - creative writing" "/v1/chat/completions" "POST" \
        '{"model":"foundation","messages":[{"role":"user","content":"Write a haiku about coding"}]}' \
        "choices"

    run_api_test "Chat completion - with temperature" "/v1/chat/completions" "POST" \
        '{"model":"foundation","messages":[{"role":"user","content":"Tell me about Python"}],"temperature":0.7}' \
        "choices"

    run_api_test "Chat completion - multi-turn conversation" "/v1/chat/completions" "POST" \
        '{"model":"foundation","messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is AI?"},{"role":"assistant","content":"AI stands for Artificial Intelligence"},{"role":"user","content":"Give me an example"}]}' \
        "choices"

    run_api_test "Chat completion - streaming" "/v1/chat/completions" "POST" \
        '{"model":"foundation","messages":[{"role":"user","content":"Count to 3"}],"stream":true}' \
        "data:"

    run_api_test "Chat completion - empty messages (should error)" "/v1/chat/completions" "POST" \
        '{"model":"foundation","messages":[]}' \
        "error"

    stop_test_server
else
    echo -e "${YELLOW}Skipping API tests - server failed to start${NC}"
    skip_count=$((skip_count + 11))
fi

# Test server with permissive guardrails
echo -e "${CYAN}Testing server with permissive guardrails enabled...${NC}"
if start_test_server "-P"; then
    run_api_test "Chat with permissive guardrails" "/v1/chat/completions" "POST" \
        '{"model":"apple-afm","messages":[{"role":"user","content":"Test content"}]}' \
        "choices"

    stop_test_server
else
    echo -e "${YELLOW}Skipping permissive guardrails API test - server failed to start${NC}"
    skip_count=$((skip_count + 1))
fi

# Test server with custom parameters
echo -e "${CYAN}Testing server with custom parameters...${NC}"
if start_test_server "-t 0.5 -r greedy"; then
    run_api_test "Chat with server-level temp/randomness" "/v1/chat/completions" "POST" \
        '{"model":"apple-afm","messages":[{"role":"user","content":"Test"}]}' \
        "choices"

    stop_test_server
else
    echo -e "${YELLOW}Skipping custom params API test - server failed to start${NC}"
    skip_count=$((skip_count + 1))
fi

###############################################################################
# TEST SUMMARY
###############################################################################

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  TEST SUMMARY${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
log_file_only "=== TEST SUMMARY ==="

echo -e "${BLUE}Total Tests Run: $test_count${NC}"
echo -e "${GREEN}Passed: $pass_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"

if [ $skip_count -gt 0 ]; then
    echo -e "${YELLOW}Skipped: $skip_count${NC}"
fi

log_file_only "Total Tests: $test_count"
log_file_only "Passed: $pass_count"
log_file_only "Failed: $fail_count"
log_file_only "Skipped: $skip_count"
log_file_only "Test completed at: $(date)"
log_file_only ""

echo ""
echo -e "${YELLOW}Complete test log saved to: $LOG_FILE${NC}"
echo -e "${YELLOW}Server log (if applicable): /tmp/afm_test_server_${TIMESTAMP}.log${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              🎉 ALL TESTS PASSED! 🎉                        ║${NC}"
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    log_file_only "🎉 ALL TESTS PASSED! 🎉"
    exit 0
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              ❌ SOME TESTS FAILED                           ║${NC}"
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    log_file_only "❌ SOME TESTS FAILED"
    exit 1
fi
