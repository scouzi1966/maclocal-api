#!/bin/bash

# test-go.sh - Comprehensive test script for AFM randomness parameters
# Usage: ./test-go.sh [-b path_to_afm_binary] [-s test_prompt]
# Defaults: AFM binary: ./.build/arm64-apple-macosx/debug/afm
#          Test prompt: "Hello, how are you?"

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 [-b afm_binary_path] [-s test_prompt] [-h]"
    echo ""
    echo "Options:"
    echo "  -b: Path to AFM binary (default: ./.build/arm64-apple-macosx/debug/afm)"
    echo "  -s: Test prompt string (default: \"Hello, how are you?\")"
    echo "  -h: Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all defaults"
    echo "  $0 -b /usr/local/bin/afm             # Custom binary path"
    echo "  $0 -s \"Tell me a story\"              # Custom test prompt"
    echo "  $0 -b ./afm -s \"What is 2+2?\"       # Custom binary and prompt"
    echo ""
    echo "Log file will be created as: afm_test_log_YYYYMMDD_HHMMSS.txt"
    exit 1
}

# Default values
DEFAULT_AFM_PATH="./.build/arm64-apple-macosx/debug/afm"
DEFAULT_TEST_PROMPT="Hello, how are you?"

# Initialize with defaults
AFM_PATH="$DEFAULT_AFM_PATH"
TEST_PROMPT="$DEFAULT_TEST_PROMPT"

# Parse command line arguments
while getopts "b:s:h" opt; do
    case $opt in
        b)
            AFM_PATH="$OPTARG"
            ;;
        s)
            TEST_PROMPT="$OPTARG"
            ;;
        h)
            show_usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            show_usage
            ;;
    esac
done

# Create log file with timestamp
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="afm_test_log_${TIMESTAMP}.txt"

# Function to log both to console and file
log_output() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to log only to file (for detailed output)
log_file_only() {
    echo "$1" >> "$LOG_FILE"
}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Initialize log file
log_file_only "=== AFM Randomness Parameter Test Suite ==="
log_file_only "Timestamp: $(date)"
log_file_only "AFM Binary: ${AFM_PATH}"
log_file_only "Test Prompt: '${TEST_PROMPT}'"
log_file_only "Log File: ${LOG_FILE}"
log_file_only ""

echo -e "${BLUE}===AFM Randomness Parameter Test Suite ===${NC}"
log_output "AFM Binary: ${AFM_PATH}"
log_output "Test Prompt: '${TEST_PROMPT}'"
log_output "Log File: ${LOG_FILE}"
echo ""

# Check if AFM binary exists
if [ ! -f "$AFM_PATH" ]; then
    log_output "${RED}ERROR: AFM binary not found at: $AFM_PATH${NC}"
    log_output "${YELLOW}Please build the project first: swift build${NC}"
    exit 1
fi

# Test counter
test_count=0
pass_count=0
fail_count=0

# Function to run a test
run_test() {
    local description="$1"
    local randomness_param="$2"
    local should_succeed="$3"  # true or false
    local extra_args="$4"

    test_count=$((test_count + 1))

    # Log to both console and file
    echo -e "${BLUE}Test $test_count: $description${NC}"
    log_file_only "Test $test_count: $description"

    echo -e "${YELLOW}  Command: $AFM_PATH -s '$TEST_PROMPT' -r '$randomness_param' $extra_args${NC}"
    log_file_only "  Command: $AFM_PATH -s '$TEST_PROMPT' -r '$randomness_param' $extra_args"

    # Run the command and capture output
    set +e  # Don't exit on error for this test
    output=$($AFM_PATH -s "$TEST_PROMPT" -r "$randomness_param" $extra_args 2>&1)
    exit_code=$?
    set -e  # Re-enable exit on error

    # Log full output to file
    log_file_only "  Exit Code: $exit_code"
    log_file_only "  Full Output:"
    log_file_only "$output"
    log_file_only "  ---"

    # Check if test passed
    if [ "$should_succeed" = "true" ]; then
        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}  ✅ PASS - Command succeeded as expected${NC}"
            log_file_only "  Result: PASS - Command succeeded as expected"
            pass_count=$((pass_count + 1))
        else
            echo -e "${RED}  ❌ FAIL - Expected success but got error:${NC}"
            echo "  $output"
            log_file_only "  Result: FAIL - Expected success but got error"
            fail_count=$((fail_count + 1))
        fi
    else
        if [ $exit_code -ne 0 ]; then
            echo -e "${GREEN}  ✅ PASS - Command failed as expected${NC}"
            echo -e "${YELLOW}  Error: $(echo "$output" | head -1)${NC}"
            log_file_only "  Result: PASS - Command failed as expected"
            log_file_only "  Error: $(echo "$output" | head -1)"
            pass_count=$((pass_count + 1))
        else
            echo -e "${RED}  ❌ FAIL - Expected failure but command succeeded${NC}"
            log_file_only "  Result: FAIL - Expected failure but command succeeded"
            fail_count=$((fail_count + 1))
        fi
    fi

    log_file_only ""
    echo ""
}

# Function to run debug test
run_debug_test() {
    local description="$1"
    local randomness_param="$2"

    test_count=$((test_count + 1))
    echo -e "${BLUE}Debug Test $test_count: $description${NC}"
    log_file_only "Debug Test $test_count: $description"

    echo -e "${YELLOW}  Command: AFM_DEBUG=1 $AFM_PATH -s '$TEST_PROMPT' -r '$randomness_param'${NC}"
    log_file_only "  Command: AFM_DEBUG=1 $AFM_PATH -s '$TEST_PROMPT' -r '$randomness_param'"

    # Run with debug enabled - properly export the environment variable
    set +e
    output=$(env AFM_DEBUG=1 "$AFM_PATH" -s "$TEST_PROMPT" -r "$randomness_param" 2>&1)
    exit_code=$?
    set -e

    # Log full debug output to file
    log_file_only "  Exit Code: $exit_code"
    log_file_only "  Full Debug Output:"
    log_file_only "$output"
    log_file_only "  ---"

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}  ✅ PASS - Debug output captured${NC}"
        # Extract and show the parsed config line
        parsed_line=$(echo "$output" | grep "Parsed randomness config" || echo "No parse info found")
        echo -e "${YELLOW}  $parsed_line${NC}"
        log_file_only "  Result: PASS - Debug output captured"
        log_file_only "  $parsed_line"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}  ❌ FAIL - Debug test failed${NC}"
        echo "  $output"
        log_file_only "  Result: FAIL - Debug test failed"
        fail_count=$((fail_count + 1))
    fi

    log_file_only ""
    echo ""
}

echo -e "${BLUE}===BACKWARD COMPATIBILITY TESTS ===${NC}"
log_file_only "=== BACKWARD COMPATIBILITY TESTS ==="

# Test backward compatible parameters
run_test "Backward compatibility - greedy" "greedy" true
run_test "Backward compatibility - random" "random" true

echo -e "${BLUE}=== NUCLEUS SAMPLING (TOP-P) TESTS ===${NC}"
log_file_only "=== NUCLEUS SAMPLING (TOP-P) TESTS ===

# Test nucleus sampling
run_test "Nucleus sampling - basic" "random:top-p=0.9" true
run_test "Nucleus sampling - minimum value" "random:top-p=0.0" true
run_test "Nucleus sampling - maximum value" "random:top-p=1.0" true
run_test "Nucleus sampling - with seed" "random:top-p=0.9:seed=42" true
run_test "Nucleus sampling - different seed" "random:top-p=0.5:seed=12345" true

echo -e "${BLUE}=== TOP-K SAMPLING TESTS ===${NC}"
log_file_only "=== TOP-K SAMPLING TESTS ===

# Test top-k sampling
run_test "Top-k sampling - basic" "random:top-k=50" true
run_test "Top-k sampling - small value" "random:top-k=1" true
run_test "Top-k sampling - large value" "random:top-k=1000" true
run_test "Top-k sampling - with seed" "random:top-k=50:seed=42" true
run_test "Top-k sampling - different seed" "random:top-k=100:seed=67890" true

echo -e "${BLUE}=== SEED-ONLY TESTS ===${NC}"
log_file_only "=== SEED-ONLY TESTS ===

# Test seed-only random
run_test "Random with seed only" "random:seed=42" true
run_test "Random with different seed" "random:seed=999" true

echo -e "${BLUE}=== VALIDATION ERROR TESTS ===${NC}"
log_file_only "=== VALIDATION ERROR TESTS ===

# Test validation errors
run_test "Invalid format" "invalid_format" false
run_test "Top-p out of range (high)" "random:top-p=1.5" false
run_test "Top-p out of range (negative)" "random:top-p=-0.1" false
run_test "Top-k zero" "random:top-k=0" false
run_test "Top-k negative" "random:top-k=-5" false
run_test "Invalid seed" "random:seed=abc" false
run_test "Unknown parameter" "random:unknown=123" false
run_test "Wrong prefix" "greedy:top-p=0.9" false

echo -e "${BLUE}=== TEMPERATURE COMBINATION TESTS ===${NC}"
log_file_only "=== TEMPERATURE COMBINATION TESTS ===

# Test with temperature combinations
run_test "Greedy with temperature" "greedy" true "-t 0.5"
run_test "Nucleus with temperature" "random:top-p=0.9" true "-t 0.7"
run_test "Top-k with temperature" "random:top-k=50" true "-t 0.3"

echo -e "${BLUE}=== DEBUG OUTPUT TESTS ===${NC}"
log_file_only "=== DEBUG OUTPUT TESTS ===

# Test debug outputs
run_debug_test "Debug - greedy parsing" "greedy"
run_debug_test "Debug - nucleus parsing" "random:top-p=0.9:seed=42"
run_debug_test "Debug - top-k parsing" "random:top-k=50:seed=123"
run_debug_test "Debug - random with seed" "random:seed=456"

echo -e "${BLUE}=== TEST SUMMARY ===${NC}"
log_file_only "=== TEST SUMMARY ==="

echo -e "${BLUE}Total Tests: $test_count${NC}"
echo -e "${GREEN}Passed: $pass_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"

log_file_only "Total Tests: $test_count"
log_file_only "Passed: $pass_count"
log_file_only "Failed: $fail_count"
log_file_only "Test completed at: $(date)"
log_file_only ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    log_file_only "🎉 ALL TESTS PASSED! 🎉"
    echo -e "${YELLOW}Complete test log saved to: $LOG_FILE${NC}"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    log_file_only "❌ SOME TESTS FAILED"
    echo -e "${YELLOW}Complete test log with failure details saved to: $LOG_FILE${NC}"
    exit 1
fi