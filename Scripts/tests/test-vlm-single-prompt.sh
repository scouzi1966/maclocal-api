#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# VLM Single-Prompt CLI Tests
# Tests: afm mlx -s "prompt" --media <path> [--guided-json <schema>]
# ══════════════════════════════════════════════════════════════════════
#
# These test the --media flag with -s (single-prompt mode), bypassing
# the server entirely. Each test runs afm as a one-shot CLI command.
#
# RUN:
#   ./Scripts/tests/test-vlm-single-prompt.sh
#   ./Scripts/tests/test-vlm-single-prompt.sh --model org/model
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL="${MODEL:-mlx-community/Qwen3.5-35B-A3B-4bit}"
MEDIA_DIR="media"
IMAGE="$MEDIA_DIR/image.png"
VIDEO="$MEDIA_DIR/video-43D1392C.mp4"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Find afm binary
if [ -f .build/release/afm ]; then
  AFM=".build/release/afm"
elif command -v afm &>/dev/null; then
  AFM="afm"
else
  echo "Error: afm binary not found. Run 'swift build -c release' first."
  exit 1
fi

export MACAFM_MLX_MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"

PASS=0
FAIL=0
TOTAL=0

run_test() {
  local label="$1"
  shift
  TOTAL=$((TOTAL + 1))
  echo "[$TOTAL] $label"
  echo "  cmd: $AFM $*"

  local output
  local exit_code=0
  output=$("$AFM" "$@" 2>&1) || exit_code=$?

  if [ $exit_code -ne 0 ]; then
    echo "  FAIL (exit code $exit_code)"
    echo "  output: ${output:0:300}"
    FAIL=$((FAIL + 1))
    echo ""
    return
  fi

  if [ -z "$output" ]; then
    echo "  FAIL (empty output)"
    FAIL=$((FAIL + 1))
    echo ""
    return
  fi

  echo "  OK: ${output:0:200}"
  PASS=$((PASS + 1))
  echo ""
}

check_json() {
  local label="$1"
  shift
  TOTAL=$((TOTAL + 1))
  echo "[$TOTAL] $label"
  echo "  cmd: $AFM $*"

  local output
  local exit_code=0
  output=$("$AFM" "$@" 2>&1) || exit_code=$?

  if [ $exit_code -ne 0 ]; then
    echo "  FAIL (exit code $exit_code)"
    echo "  output: ${output:0:300}"
    FAIL=$((FAIL + 1))
    echo ""
    return
  fi

  # Validate JSON
  if echo "$output" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
    echo "  OK (valid JSON): ${output:0:200}"
    PASS=$((PASS + 1))
  else
    echo "  FAIL (invalid JSON): ${output:0:200}"
    FAIL=$((FAIL + 1))
  fi
  echo ""
}

echo "══════════════════════════════════════════════════════════════════════"
echo "VLM Single-Prompt CLI Tests"
echo "Model: $MODEL"
echo "Binary: $AFM"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

# Verify media files exist
if [ ! -f "$IMAGE" ]; then
  echo "Error: $IMAGE not found. Run from repo root."
  exit 1
fi

# ── Image tests ──────────────────────────────────────────────────────

run_test "Image: basic description" \
  mlx -m "$MODEL" --raw -s "Describe this image in detail." --media "$IMAGE" --max-tokens 512

run_test "Image: one-word answer" \
  mlx -m "$MODEL" --raw -s "What animal is in this image? Answer in one word." --media "$IMAGE" --max-tokens 128

run_test "Image: with system prompt" \
  mlx -m "$MODEL" --raw -i "You are a veterinarian." -s "Describe the animal in this image." --media "$IMAGE" --max-tokens 512

run_test "Image: greedy (temp=0)" \
  mlx -m "$MODEL" --raw -s "What is in this image? Reply in one sentence." --media "$IMAGE" --max-tokens 256 --temperature 0.0

# ── Guided JSON + image ──────────────────────────────────────────────

check_json "Image + guided-json: structured extraction" \
  mlx -m "$MODEL" --raw -s "Analyze this image. Return a JSON object with fields: subject, colors (array), setting (indoor/outdoor)." \
  --media "$IMAGE" --max-tokens 512 \
  --guided-json '{"type":"object","properties":{"subject":{"type":"string"},"colors":{"type":"array","items":{"type":"string"}},"setting":{"type":"string","enum":["indoor","outdoor"]}},"required":["subject","colors","setting"]}'

check_json "Image + guided-json: simple classification" \
  mlx -m "$MODEL" --raw -s "Classify this image." \
  --media "$IMAGE" --max-tokens 256 \
  --guided-json '{"type":"object","properties":{"category":{"type":"string","enum":["animal","person","landscape","object","food","vehicle"]},"confidence":{"type":"number"}},"required":["category","confidence"]}'

# ── Video test ───────────────────────────────────────────────────────

if [ -f "$VIDEO" ]; then
  run_test "Video: basic description" \
    mlx -m "$MODEL" --raw -s "What happens in this video?" --media "$VIDEO" --max-tokens 512
fi

# ── Multiple media ───────────────────────────────────────────────────

if [ -f "$VIDEO" ] && [ -f "$IMAGE" ]; then
  run_test "Multi-media: image + video" \
    mlx -m "$MODEL" --raw -s "Describe both the image and the video." --media "$IMAGE" "$VIDEO" --max-tokens 512
fi

# ── afm vision (OCR) tests ───────────────────────────────────────────

OCR_IMAGE="$MEDIA_DIR/ocr.png"
if [ -f "$OCR_IMAGE" ]; then
  run_test "Vision OCR: extract text from image" \
    vision -f "$OCR_IMAGE"

  run_test "Vision OCR: verbose with confidence" \
    vision -f "$OCR_IMAGE" --verbose
else
  echo "Skipping Vision OCR tests: $OCR_IMAGE not found"
fi

# ── Summary ──────────────────────────────────────────────────────────

echo "══════════════════════════════════════════════════════════════════════"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "══════════════════════════════════════════════════════════════════════"

[ $FAIL -eq 0 ] || exit 1
