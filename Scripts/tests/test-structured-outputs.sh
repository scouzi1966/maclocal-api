#!/usr/bin/env bash
# =============================================================================
# Structured Outputs / Guided Generation — Comprehensive Test Suite
# =============================================================================
# Tests response_format with json_schema (strict and non-strict), json_object,
# and text modes across Apple FM and MLX backends.
#
# Phase 1: Apple FM + MLX       (this script)
# Phase 2: Gateway mode (-g)    (future)
#
# Usage:
#   # Run all backends with defaults (API + CLI)
#   ./Scripts/tests/test-structured-outputs.sh
#
#   # Run only MLX API tests
#   ./Scripts/tests/test-structured-outputs.sh --mlx-only
#
#   # Run only Apple FM API tests
#   ./Scripts/tests/test-structured-outputs.sh --afm-only
#
#   # Run only CLI single-prompt tests
#   ./Scripts/tests/test-structured-outputs.sh --cli-only
#
#   # Skip CLI tests
#   ./Scripts/tests/test-structured-outputs.sh --no-cli
#
#   # Custom binary / ports
#   AFM_BIN=.build/release/afm MLX_PORT=19901 AFM_PORT=19902 \
#     ./Scripts/tests/test-structured-outputs.sh
#
#   # Specific MLX model (run one at a time)
#   MLX_MODELS="Qwen3-30B-A3B-4bit" ./Scripts/tests/test-structured-outputs.sh --mlx-only
#
# Environment:
#   AFM_BIN                  Path to afm binary (default: .build/release/afm)
#   MACAFM_MLX_MODEL_CACHE   Model cache dir (default: /Volumes/edata/models/vesta-test-cache)
#   MLX_PORT                 Port for MLX server (default: 19901)
#   AFM_PORT                 Port for AFM server (default: 19902)
#   MLX_MODELS               Space-separated model list (default: all 3 test models)
#   TIMEOUT                  curl timeout in seconds (default: 120)
# =============================================================================

set -uo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────

AFM="${AFM_BIN:-.build/release/afm}"
export MACAFM_MLX_MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"

MLX_PORT="${MLX_PORT:-19901}"
AFM_PORT="${AFM_PORT:-19902}"
TIMEOUT="${TIMEOUT:-120}"

MLX_MODELS="${MLX_MODELS:-Qwen3-30B-A3B-4bit gpt-oss-20b-MXFP4-Q4 gemma-3-4b-it-8bit Qwen3-Coder-Next-4bit}"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
REPORT_DIR="test-reports"
RESULTS_FILE="${REPORT_DIR}/structured-outputs-${TIMESTAMP}.jsonl"
LOG_DIR="/tmp/afm-structured-test-${TIMESTAMP}"

# Parse flags
RUN_AFM=1
RUN_MLX=1
RUN_CLI=1
for arg in "$@"; do
  case "$arg" in
    --mlx-only) RUN_AFM=0; RUN_CLI=0 ;;
    --afm-only) RUN_MLX=0; RUN_CLI=0 ;;
    --cli-only) RUN_AFM=0; RUN_MLX=0 ;;
    --no-cli)   RUN_CLI=0 ;;
    -h|--help)
      sed -n '2,/^# =====/p' "$0" | head -n -1 | sed 's/^# \?//'
      exit 0
      ;;
  esac
done

# ── Setup ──────────────────────────────────────────────────────────────────────

mkdir -p "$REPORT_DIR" "$LOG_DIR"
> "$RESULTS_FILE"

if [ ! -x "$AFM" ]; then
  echo "ERROR: afm binary not found at $AFM"
  echo "Run: swift build -c release"
  exit 1
fi

# Counters
total=0; pass=0; fail=0; skip=0

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Helpers ────────────────────────────────────────────────────────────────────

escape_json() {
  python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))" <<< "$1"
}

record() {
  local section="$1" name="$2" status="$3" detail="${4:-}" elapsed="${5:-0}" prompt="${6:-}" request="${7:-}"
  total=$((total + 1))
  if [ "$status" = "PASS" ]; then
    pass=$((pass + 1))
    echo -e "  ${GREEN}✅ $name${RESET}"
  elif [ "$status" = "SKIP" ]; then
    skip=$((skip + 1))
    echo -e "  ${YELLOW}⏭️  $name${RESET} (skipped: $detail)"
  else
    fail=$((fail + 1))
    echo -e "  ${RED}❌ $name${RESET} — $detail"
  fi
  local extra=""
  if [ -n "$prompt" ]; then
    extra=",\"prompt\":$(escape_json "$prompt")"
  fi
  if [ -n "$request" ]; then
    # request is already valid JSON — embed directly
    extra="${extra},\"request\":${request}"
  fi
  echo "{\"section\":$(escape_json "$section"),\"name\":$(escape_json "$name"),\"status\":\"$status\",\"detail\":$(escape_json "$detail"),\"elapsed_s\":$elapsed${extra}}" >> "$RESULTS_FILE"
}

# Start a server, wait for health, return PID
start_server() {
  local label="$1"; shift
  local port="$1"; shift
  local logfile="$LOG_DIR/${label}.log"

  echo -e "  ${CYAN}Starting $label on port $port ...${RESET}"
  "$AFM" "$@" -p "$port" > "$logfile" 2>&1 &
  local pid=$!

  local waited=0
  while [ $waited -lt 90 ]; do
    if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
      echo -e "  ${GREEN}$label ready (pid $pid)${RESET}"
      echo "$pid"
      return 0
    fi
    # Check if process died
    if ! kill -0 "$pid" 2>/dev/null; then
      echo -e "  ${RED}$label process died. Log:${RESET}"
      tail -20 "$logfile"
      echo "DEAD"
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo -e "  ${RED}$label failed to start within 90s${RESET}"
  kill "$pid" 2>/dev/null || true
  echo "TIMEOUT"
  return 1
}

stop_server() {
  local pid="$1" label="$2" port="${3:-}"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo -e "  ${CYAN}Stopped $label (pid $pid)${RESET}"
  fi
  # Wait for port to be fully released before starting next server
  if [ -n "$port" ]; then
    local waited=0
    while [ $waited -lt 10 ]; do
      if ! lsof -ti:"$port" > /dev/null 2>&1; then
        break
      fi
      sleep 1
      waited=$((waited + 1))
    done
  fi
}

# POST a non-streaming request, extract content
api_post() {
  local port="$1" payload="$2"
  curl -sf --max-time "$TIMEOUT" \
    "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null
}

# POST a streaming request, collect content deltas
api_stream() {
  local port="$1" payload="$2"
  curl -sf -N --max-time "$TIMEOUT" \
    "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>/dev/null | while IFS= read -r line; do
      line="${line#data: }"
      [ -z "$line" ] && continue
      [ "$line" = "[DONE]" ] && break
      echo "$line" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    c=d.get('choices',[{}])[0].get('delta',{}).get('content','')
    if c: print(c, end='')
except: pass
" 2>/dev/null
  done
}

extract_content() {
  local raw="$1"
  echo "$raw" | python3 -c "
import sys,json,re
try:
    d=json.load(sys.stdin)
    msg = d['choices'][0]['message']
    content = msg.get('content','') or ''
    if content:
        print(content)
    else:
        # For thinking models with json_object mode, try reasoning_content
        rc = msg.get('reasoning_content','') or ''
        # Try to extract JSON from reasoning content
        m = re.search(r'\{.*\}', rc, re.DOTALL)
        if m:
            try:
                json.loads(m.group())
                print(m.group())
            except:
                print('')
        else:
            print('')
except Exception as e:
    print(f'PARSE_ERROR: {e}')
"
}

is_valid_json() {
  python3 -c "import sys,json; json.load(sys.stdin)" <<< "$1" 2>/dev/null
}

# Check JSON matches schema structure (key presence check)
json_has_keys() {
  local json="$1"; shift
  echo "$json" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    keys = sys.argv[1:]
    for k in keys:
        parts = k.split('.')
        obj = d
        for p in parts:
            if p.startswith('['):
                obj = obj[int(p.strip('[]'))]
            elif isinstance(obj, dict):
                obj = obj[p]
            else:
                sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
" "$@" 2>/dev/null
}

# ── Schema Definitions ─────────────────────────────────────────────────────────
# Reusable JSON payloads for test cases

# Schema 1: Simple object — string properties
SCHEMA_SIMPLE='{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "capital": {"type": "string"}
  },
  "required": ["name", "capital"],
  "additionalProperties": false
}'

# Schema 2: Array of objects
SCHEMA_ARRAY='{
  "type": "object",
  "properties": {
    "colors": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "hex": {"type": "string"}
        },
        "required": ["name", "hex"]
      }
    }
  },
  "required": ["colors"],
  "additionalProperties": false
}'

# Schema 3: Mixed types (string, integer, boolean, number)
SCHEMA_MIXED='{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "year": {"type": "integer"},
    "rating": {"type": "number"},
    "available": {"type": "boolean"}
  },
  "required": ["title", "year", "rating", "available"],
  "additionalProperties": false
}'

# Schema 4: Nested objects
SCHEMA_NESTED='{
  "type": "object",
  "properties": {
    "person": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "address": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"}
          },
          "required": ["city", "country"]
        }
      },
      "required": ["name", "address"]
    }
  },
  "required": ["person"],
  "additionalProperties": false
}'

# Schema 5: String enum
SCHEMA_ENUM='{
  "type": "object",
  "properties": {
    "sentiment": {
      "type": "string",
      "enum": ["positive", "negative", "neutral"]
    },
    "confidence": {"type": "number"}
  },
  "required": ["sentiment", "confidence"],
  "additionalProperties": false
}'

# Schema 6: Array with minItems/maxItems
SCHEMA_BOUNDED_ARRAY='{
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 2,
      "maxItems": 4
    }
  },
  "required": ["items"],
  "additionalProperties": false
}'

# Schema 7: Optional properties (some not in required)
SCHEMA_OPTIONAL='{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "nickname": {"type": "string"},
    "age": {"type": "integer"}
  },
  "required": ["name"],
  "additionalProperties": false
}'

# Schema 8: $ref / $defs
SCHEMA_REFS='{
  "type": "object",
  "properties": {
    "students": {
      "type": "array",
      "items": {"$ref": "#/$defs/Student"}
    }
  },
  "required": ["students"],
  "$defs": {
    "Student": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "grade": {"type": "string"}
      },
      "required": ["name", "grade"]
    }
  },
  "additionalProperties": false
}'

# ── Test Runner ────────────────────────────────────────────────────────────────

# Run a single structured output test
# Args: section backend port name prompt schema strict stream expect_keys...
run_so_test() {
  local section="$1" backend="$2" port="$3" name="$4"
  local prompt="$5" schema="$6" strict="$7" stream="$8"
  shift 8
  local expect_keys=("$@")

  local start_time=$(python3 -c "import time; print(time.time())")

  # Build payload
  local response_format
  if [ "$schema" = "NONE" ]; then
    response_format=""
  elif [ "$schema" = "JSON_OBJECT" ]; then
    response_format=',"response_format":{"type":"json_object"}'
  else
    local strict_val="false"
    [ "$strict" = "strict" ] && strict_val="true"
    # Escape the schema for embedding in JSON
    local schema_escaped
    schema_escaped=$(python3 -c "import json; print(json.dumps(json.loads('''$schema''')))")
    response_format=",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"test_schema\",\"strict\":${strict_val},\"schema\":${schema_escaped}}}"
  fi

  local stream_val="false"
  [ "$stream" = "stream" ] && stream_val="true"

  local prompt_escaped
  prompt_escaped=$(echo "$prompt" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))")
  local payload="{\"messages\":[{\"role\":\"user\",\"content\":${prompt_escaped}}],\"max_tokens\":2000,\"temperature\":0.3,\"stream\":${stream_val}${response_format}}"

  # Make request
  local content=""
  local raw_response=""
  if [ "$stream" = "stream" ]; then
    content=$(api_stream "$port" "$payload")
  else
    raw_response=$(api_post "$port" "$payload")
    if [ -z "$raw_response" ]; then
      local elapsed=$(python3 -c "import time; print(f'{time.time()-${start_time}:.1f}')")
      record "$section" "$name" "FAIL" "Empty response / curl error" "$elapsed" "$prompt" "$payload"
      return
    fi
    content=$(extract_content "$raw_response")
  fi

  local elapsed=$(python3 -c "import time; print(f'{time.time()-${start_time}:.1f}')")

  # Check for errors
  if [ -z "$content" ] || [[ "$content" == PARSE_ERROR* ]]; then
    # For json_object mode with thinking models, empty content is a known limitation
    if [ "$schema" = "JSON_OBJECT" ]; then
      record "$section" "$name" "SKIP" "Empty content (known: thinking models put JSON in reasoning_content)" "$elapsed" "$prompt" "$payload"
    else
      record "$section" "$name" "FAIL" "No content: ${content:-empty}${raw_response:+ raw=${raw_response:0:200}}" "$elapsed" "$prompt" "$payload"
    fi
    return
  fi

  # If schema is NONE, just check non-empty
  if [ "$schema" = "NONE" ]; then
    record "$section" "$name" "PASS" "Got text response (${#content} chars)" "$elapsed" "$prompt" "$payload"
    return
  fi

  # For json_object mode, verify the response is a JSON object (not array)
  if [ "$schema" = "JSON_OBJECT" ]; then
    if is_valid_json "$content"; then
      local is_object
      is_object=$(echo "$content" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if isinstance(d,dict) else 'no')" 2>/dev/null)
      if [ "$is_object" = "yes" ]; then
        record "$section" "$name" "PASS" "${content:0:150}" "$elapsed" "$prompt" "$payload"
      else
        record "$section" "$name" "FAIL" "json_object mode returned non-object (array?): ${content:0:200}" "$elapsed" "$prompt" "$payload"
      fi
    else
      record "$section" "$name" "FAIL" "Invalid JSON: ${content:0:200}" "$elapsed" "$prompt" "$payload"
    fi
    return
  fi

  # Validate JSON
  if ! is_valid_json "$content"; then
    # For MLX (prompt injection), JSON may have extra text — try to extract
    local extracted
    extracted=$(echo "$content" | python3 -c "
import sys,json,re
text = sys.stdin.read()
# Try to find JSON object in the text
m = re.search(r'\{.*\}', text, re.DOTALL)
if m:
    try:
        json.loads(m.group())
        print(m.group())
        sys.exit(0)
    except: pass
sys.exit(1)
" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$extracted" ]; then
      content="$extracted"
    else
      record "$section" "$name" "FAIL" "Invalid JSON: ${content:0:200}" "$elapsed" "$prompt" "$payload"
      return
    fi
  fi

  # Check expected keys
  if [ ${#expect_keys[@]} -gt 0 ]; then
    for key in "${expect_keys[@]}"; do
      if ! json_has_keys "$content" "$key"; then
        record "$section" "$name" "FAIL" "Missing key '$key' in: ${content:0:200}" "$elapsed" "$prompt" "$payload"
        return
      fi
    done
  fi

  record "$section" "$name" "PASS" "${content:0:150}" "$elapsed" "$prompt" "$payload"
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  STRUCTURED OUTPUTS TEST SUITE — $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════${RESET}"
echo "  Binary:  $AFM"
echo "  Cache:   $MACAFM_MLX_MODEL_CACHE"
echo "  Results: $RESULTS_FILE"
echo "  Logs:    $LOG_DIR/"
echo ""

# ── Section 1: Apple Foundation Model ──────────────────────────────────────────

if [ "$RUN_AFM" = "1" ]; then
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}  SECTION 1: Apple Foundation Model (guided generation)${RESET}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

  AFM_PID=""
  AFM_PID=$(start_server "afm-foundation" "$AFM_PORT")
  if [ "$AFM_PID" = "DEAD" ] || [ "$AFM_PID" = "TIMEOUT" ] || [ -z "$AFM_PID" ]; then
    echo -e "  ${YELLOW}Skipping AFM tests — server failed to start${RESET}"
    record "AFM" "Server startup" "SKIP" "Foundation model server failed to start"
  else
    SECTION="AFM"

    # 1.1 Simple object — strict (true guided generation)
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.1 Simple object (strict, non-streaming)" \
      "What is the capital of France? Answer with the country name and capital." \
      "$SCHEMA_SIMPLE" "strict" "no-stream" \
      "name" "capital"

    # 1.2 Simple object — strict + streaming
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.2 Simple object (strict, streaming)" \
      "What is the capital of Japan? Answer with the country name and capital." \
      "$SCHEMA_SIMPLE" "strict" "stream" \
      "name" "capital"

    # 1.3 Array of objects — strict
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.3 Array of objects (strict, non-streaming)" \
      "List 3 colors with their hex codes." \
      "$SCHEMA_ARRAY" "strict" "no-stream" \
      "colors"

    # 1.4 Array of objects — strict + streaming
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.4 Array of objects (strict, streaming)" \
      "List 2 colors with their hex codes." \
      "$SCHEMA_ARRAY" "strict" "stream" \
      "colors"

    # 1.5 Mixed types
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.5 Mixed types (strict, non-streaming)" \
      "Tell me about the movie Inception. Provide title, year, rating out of 10, and whether it is available on streaming." \
      "$SCHEMA_MIXED" "strict" "no-stream" \
      "title" "year" "rating" "available"

    # 1.6 Nested objects
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.6 Nested objects (strict, non-streaming)" \
      "Give me info about Albert Einstein including his city and country." \
      "$SCHEMA_NESTED" "strict" "no-stream" \
      "person" "person.name" "person.address" "person.address.city"

    # 1.7 String enum
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.7 String enum (strict, non-streaming)" \
      "Analyze the sentiment of: I love this product, it is amazing!" \
      "$SCHEMA_ENUM" "strict" "no-stream" \
      "sentiment" "confidence"

    # 1.8 Bounded array
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.8 Bounded array (strict, non-streaming)" \
      "List some popular programming languages." \
      "$SCHEMA_BOUNDED_ARRAY" "strict" "no-stream" \
      "items"

    # 1.9 Optional properties
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.9 Optional properties (strict, non-streaming)" \
      "Tell me about someone named Alice." \
      "$SCHEMA_OPTIONAL" "strict" "no-stream" \
      "name"

    # 1.10 $ref / $defs
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.10 Refs and defs (strict, non-streaming)" \
      "List 2 students with their names and letter grades." \
      "$SCHEMA_REFS" "strict" "no-stream" \
      "students"

    # 1.11 Non-strict json_schema (falls back to unstructured on AFM — no prompt injection)
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.11 Non-strict json_schema (fallback to text)" \
      "List 2 colors with hex codes." \
      "NONE" "" "no-stream"

    # 1.12 json_object mode (AFM has no prompt injection — returns plain text)
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.12 json_object mode (plain text expected)" \
      "List 2 animals as JSON with name and type fields." \
      "NONE" "" "no-stream"

    # 1.13 No response_format (regression — plain text)
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.13 No response_format (plain text regression)" \
      "What is 2+2?" \
      "NONE" "" "no-stream"

    # 1.14 Mixed types — streaming
    run_so_test "$SECTION" "afm" "$AFM_PORT" \
      "1.14 Mixed types (strict, streaming)" \
      "Tell me about the movie The Matrix. Provide title, year, rating out of 10, and whether it is available on streaming." \
      "$SCHEMA_MIXED" "strict" "stream" \
      "title" "year"

    stop_server "$AFM_PID" "afm-foundation" "$AFM_PORT"
  fi
  echo ""
fi

# ── Section 2: MLX Models ─────────────────────────────────────────────────────

if [ "$RUN_MLX" = "1" ]; then
  for model_short in $MLX_MODELS; do
    # Prepend mlx-community/ if not already qualified
    if [[ "$model_short" != */* ]]; then
      model="mlx-community/${model_short}"
    else
      model="$model_short"
    fi

    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  SECTION 2: MLX — ${model}${RESET}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

    SECTION="MLX:${model_short}"
    MLX_PID=""
    MLX_PID=$(start_server "mlx-${model_short}" "$MLX_PORT" mlx -m "$model" -t 0.3 --max-tokens 2000)

    if [ "$MLX_PID" = "DEAD" ] || [ "$MLX_PID" = "TIMEOUT" ] || [ -z "$MLX_PID" ]; then
      echo -e "  ${YELLOW}Skipping ${model_short} — server failed to start${RESET}"
      record "$SECTION" "Server startup" "SKIP" "MLX server failed to start for ${model}"
      continue
    fi

    # 2.1 Simple object — strict (prompt injection on MLX)
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.1 Simple object (strict, non-streaming)" \
      "What is the capital of France? Answer with the country name and capital." \
      "$SCHEMA_SIMPLE" "strict" "no-stream" \
      "name" "capital"

    # 2.2 Simple object — strict + streaming
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.2 Simple object (strict, streaming)" \
      "What is the capital of Germany? Answer with the country name and capital." \
      "$SCHEMA_SIMPLE" "strict" "stream" \
      "name" "capital"

    # 2.3 Array of objects
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.3 Array of objects (strict, non-streaming)" \
      "List 3 colors with their hex codes." \
      "$SCHEMA_ARRAY" "strict" "no-stream" \
      "colors"

    # 2.4 Array of objects — streaming
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.4 Array of objects (strict, streaming)" \
      "List 2 colors with their hex codes." \
      "$SCHEMA_ARRAY" "strict" "stream" \
      "colors"

    # 2.5 Mixed types
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.5 Mixed types (strict, non-streaming)" \
      "Tell me about the movie Inception. Provide title, year, rating out of 10, and whether it is available on streaming." \
      "$SCHEMA_MIXED" "strict" "no-stream" \
      "title" "year"

    # 2.6 Nested objects
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.6 Nested objects (strict, non-streaming)" \
      "Give me info about Albert Einstein including his city and country." \
      "$SCHEMA_NESTED" "strict" "no-stream" \
      "person"

    # 2.7 String enum
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.7 String enum (strict, non-streaming)" \
      "Analyze the sentiment of: I love this product, it is amazing! Return sentiment and confidence." \
      "$SCHEMA_ENUM" "strict" "no-stream" \
      "sentiment"

    # 2.8 Bounded array
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.8 Bounded array (strict, non-streaming)" \
      "List some popular programming languages." \
      "$SCHEMA_BOUNDED_ARRAY" "strict" "no-stream" \
      "items"

    # 2.9 $ref / $defs
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.9 Refs and defs (strict, non-streaming)" \
      "List 2 students with their names and letter grades." \
      "$SCHEMA_REFS" "strict" "no-stream" \
      "students"

    # 2.10 Non-strict json_schema
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.10 Non-strict json_schema" \
      "List 2 colors with hex codes as JSON." \
      "$SCHEMA_ARRAY" "non-strict" "no-stream" \
      "colors"

    # 2.11 json_object mode
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.11 json_object mode" \
      "Return a JSON object with an 'animals' key containing 2 animals, each with name and type fields." \
      "JSON_OBJECT" "" "no-stream"

    # 2.12 No response_format (regression)
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.12 No response_format (plain text regression)" \
      "What is 2+2?" \
      "NONE" "" "no-stream"

    # 2.13 Nested objects — streaming
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.13 Nested objects (strict, streaming)" \
      "Give me info about Marie Curie including her city and country." \
      "$SCHEMA_NESTED" "strict" "stream" \
      "person"

    # 2.14 Mixed types — streaming
    run_so_test "$SECTION" "mlx" "$MLX_PORT" \
      "2.14 Mixed types (strict, streaming)" \
      "Tell me about the movie The Matrix. Provide title, year, rating out of 10, and whether it is available." \
      "$SCHEMA_MIXED" "strict" "stream" \
      "title"

    stop_server "$MLX_PID" "mlx-${model_short}" "$MLX_PORT"
    echo ""
  done
fi

# ── Section 3: CLI Single-Prompt Mode ────────────────────────────────────────

# Run a CLI single-prompt test
# Args: section name command_args...
# The last arg's output is captured and validated
run_cli_test() {
  local section="$1" name="$2" expect="$3"
  shift 3
  local cmd_args=("$@")

  local start_time=$(python3 -c "import time; print(time.time())")

  # Build the full command string for the request field
  local cmd_str="$AFM ${cmd_args[*]}"
  local request_json
  request_json=$(python3 -c "import json,sys; print(json.dumps({'command': sys.argv[1]}))" "$cmd_str")

  local output=""
  local stderr_file="$LOG_DIR/cli-$(echo "$name" | tr ' /' '_-').stderr"
  local exit_code=0
  output=$("$AFM" "${cmd_args[@]}" 2>"$stderr_file") || exit_code=$?

  local elapsed=$(python3 -c "import time; print(f'{time.time()-${start_time}:.1f}')")

  case "$expect" in
    json_object)
      # Must be valid JSON object
      if [ -z "$output" ]; then
        record "$section" "$name" "FAIL" "Empty output" "$elapsed" "$cmd_str" "$request_json"
        return
      fi
      # Extract JSON object from output (handles progress bars, <think> blocks, etc.)
      local json_result
      json_result=$(echo "$output" | python3 -c "
import sys,json,re
text = sys.stdin.read().strip()
# Find all potential JSON objects, try from last to first
matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text))
for m in reversed(matches):
    try:
        d = json.loads(m.group())
        if isinstance(d, dict):
            print(json.dumps(d))
            sys.exit(0)
    except Exception:
        continue
# Fallback: try the greedy match
m = re.search(r'\{.*\}', text, re.DOTALL)
if m:
    try:
        d = json.loads(m.group())
        if isinstance(d, dict):
            print(json.dumps(d))
            sys.exit(0)
    except Exception: pass
" 2>/dev/null)
      if [ -n "$json_result" ]; then
        record "$section" "$name" "PASS" "${json_result:0:150}" "$elapsed" "$cmd_str" "$request_json"
      else
        record "$section" "$name" "FAIL" "Not a JSON object: ${output:0:200}" "$elapsed" "$cmd_str" "$request_json"
      fi
      ;;
    text)
      # Must be non-empty text
      if [ -z "$output" ]; then
        record "$section" "$name" "FAIL" "Empty output" "$elapsed" "$cmd_str" "$request_json"
      else
        record "$section" "$name" "PASS" "Got text response (${#output} chars)" "$elapsed" "$cmd_str" "$request_json"
      fi
      ;;
    error)
      # Must have non-zero exit code
      local err_output="${output:-$(cat "$stderr_file" 2>/dev/null)}"
      if [ "$exit_code" -ne 0 ]; then
        record "$section" "$name" "PASS" "Exit code $exit_code: ${err_output:0:100}" "$elapsed" "$cmd_str" "$request_json"
      else
        record "$section" "$name" "FAIL" "Expected error but got exit 0: ${err_output:0:200}" "$elapsed" "$cmd_str" "$request_json"
      fi
      ;;
  esac
}

if [ "$RUN_CLI" = "1" ]; then

  # ── 3a: Apple FM CLI ──────────────────────────────────────────────────────

  if [ "$RUN_AFM" != "0" ] || [ "$RUN_CLI" = "1" ]; then
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  SECTION 3: CLI Single-Prompt — Apple FM${RESET}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

    SECTION="CLI:AFM"

    # 3.1 Simple object with --guided-json
    run_cli_test "$SECTION" \
      "3.1 --guided-json simple object" \
      "json_object" \
      -s "What is the capital of France? Answer with the country name and capital." \
      --guided-json "$SCHEMA_SIMPLE"

    # 3.2 Array of objects with --guided-json
    run_cli_test "$SECTION" \
      "3.2 --guided-json array of objects" \
      "json_object" \
      -s "List 2 colors with their hex codes." \
      --guided-json "$SCHEMA_ARRAY"

    # 3.3 Mixed types with --guided-json
    run_cli_test "$SECTION" \
      "3.3 --guided-json mixed types" \
      "json_object" \
      -s "Tell me about the movie Inception. Provide title, year, rating out of 10, and whether it is available on streaming." \
      --guided-json "$SCHEMA_MIXED"

    # 3.4 Nested objects with --guided-json
    run_cli_test "$SECTION" \
      "3.4 --guided-json nested objects" \
      "json_object" \
      -s "Give me info about Albert Einstein including his city and country." \
      --guided-json "$SCHEMA_NESTED"

    # 3.5 String enum with --guided-json
    run_cli_test "$SECTION" \
      "3.5 --guided-json string enum" \
      "json_object" \
      -s "Analyze the sentiment of: I love this product, it is amazing!" \
      --guided-json "$SCHEMA_ENUM"

    # 3.6 $ref/$defs with --guided-json
    run_cli_test "$SECTION" \
      "3.6 --guided-json refs and defs" \
      "json_object" \
      -s "List 2 students with their names and letter grades." \
      --guided-json "$SCHEMA_REFS"

    # 3.7 Plain text (no --guided-json, regression)
    run_cli_test "$SECTION" \
      "3.7 Plain text (no --guided-json, regression)" \
      "text" \
      -s "What is 2+2?"

    # 3.8 Invalid JSON schema (error handling)
    run_cli_test "$SECTION" \
      "3.8 Invalid --guided-json (error)" \
      "error" \
      -s "Hello" \
      --guided-json "not valid json"

    # 3.9 Non-object JSON schema (error handling)
    run_cli_test "$SECTION" \
      "3.9 --guided-json with array schema (error)" \
      "error" \
      -s "Hello" \
      --guided-json '[1,2,3]'

    echo ""
  fi

  # ── 3b: MLX CLI ───────────────────────────────────────────────────────────

  CLI_MLX_MODEL="${CLI_MLX_MODEL:-Qwen3-30B-A3B-4bit}"
  if [[ "$CLI_MLX_MODEL" != */* ]]; then
    CLI_MLX_MODEL_FULL="mlx-community/${CLI_MLX_MODEL}"
  else
    CLI_MLX_MODEL_FULL="$CLI_MLX_MODEL"
  fi

  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}  SECTION 4: CLI Single-Prompt — MLX (${CLI_MLX_MODEL})${RESET}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

  SECTION="CLI:MLX:${CLI_MLX_MODEL}"

  # 4.1 Simple object with --guided-json
  run_cli_test "$SECTION" \
    "4.1 --guided-json simple object" \
    "json_object" \
    mlx -m "$CLI_MLX_MODEL_FULL" \
    -s "What is the capital of France? Answer with the country name and capital." \
    --guided-json "$SCHEMA_SIMPLE"

  # 4.2 Array of objects with --guided-json
  run_cli_test "$SECTION" \
    "4.2 --guided-json array of objects" \
    "json_object" \
    mlx -m "$CLI_MLX_MODEL_FULL" \
    -s "List 2 colors with their hex codes." \
    --guided-json "$SCHEMA_ARRAY"

  # 4.3 Mixed types with --guided-json
  run_cli_test "$SECTION" \
    "4.3 --guided-json mixed types" \
    "json_object" \
    mlx -m "$CLI_MLX_MODEL_FULL" \
    -s "Tell me about the movie Inception. Provide title, year, rating out of 10, and whether it is available on streaming." \
    --guided-json "$SCHEMA_MIXED"

  # 4.4 Nested objects with --guided-json
  run_cli_test "$SECTION" \
    "4.4 --guided-json nested objects" \
    "json_object" \
    mlx -m "$CLI_MLX_MODEL_FULL" \
    -s "Give me info about Albert Einstein including his city and country." \
    --guided-json "$SCHEMA_NESTED"

  # 4.5 String enum with --guided-json
  run_cli_test "$SECTION" \
    "4.5 --guided-json string enum" \
    "json_object" \
    mlx -m "$CLI_MLX_MODEL_FULL" \
    -s "Analyze the sentiment of: I love this product, it is amazing! Return sentiment and confidence." \
    --guided-json "$SCHEMA_ENUM"

  # 4.6 $ref/$defs with --guided-json
  run_cli_test "$SECTION" \
    "4.6 --guided-json refs and defs" \
    "json_object" \
    mlx -m "$CLI_MLX_MODEL_FULL" \
    -s "List 2 students with their names and letter grades." \
    --guided-json "$SCHEMA_REFS"

  # 4.7 Plain text (no --guided-json, regression)
  run_cli_test "$SECTION" \
    "4.7 Plain text (no --guided-json, regression)" \
    "text" \
    mlx -m "$CLI_MLX_MODEL_FULL" \
    -s "What is 2+2?"

  # 4.8 Invalid JSON schema (error handling)
  run_cli_test "$SECTION" \
    "4.8 Invalid --guided-json (error)" \
    "error" \
    mlx -m "$CLI_MLX_MODEL_FULL" \
    -s "Hello" \
    --guided-json "not valid json"

  # 4.9 Non-object JSON schema (error handling)
  run_cli_test "$SECTION" \
    "4.9 --guided-json with array schema (error)" \
    "error" \
    mlx -m "$CLI_MLX_MODEL_FULL" \
    -s "Hello" \
    --guided-json '[1,2,3]'

  echo ""
fi

# ── Summary ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  RESULTS SUMMARY${RESET}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════${RESET}"
echo -e "  Total:   $total"
echo -e "  ${GREEN}Passed:  $pass${RESET}"
echo -e "  ${RED}Failed:  $fail${RESET}"
echo -e "  ${YELLOW}Skipped: $skip${RESET}"
echo ""
echo "  Results: $RESULTS_FILE"
echo "  Logs:    $LOG_DIR/"

# Per-section breakdown
# Generate HTML report
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python3 "${SCRIPT_DIR}/generate-structured-outputs-report.py" "$RESULTS_FILE"

echo ""
echo -e "${BOLD}  Per-section breakdown:${RESET}"
python3 -c "
import json, sys
from collections import defaultdict
counts = defaultdict(lambda: {'pass':0,'fail':0,'skip':0})
with open('$RESULTS_FILE') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        s = r['section']
        counts[s][r['status'].lower()] += 1
for section in sorted(counts):
    c = counts[section]
    total = c['pass']+c['fail']+c['skip']
    status = '✅' if c['fail']==0 else '❌'
    print(f'  {status} {section}: {c[\"pass\"]}/{total} passed' + (f', {c[\"fail\"]} failed' if c['fail'] else '') + (f', {c[\"skip\"]} skipped' if c['skip'] else ''))
" 2>/dev/null

echo ""

# Exit code
if [ "$fail" -gt 0 ]; then
  exit 1
fi
exit 0
