#!/usr/bin/env bash
# test-toolcall-matrix.sh — Comprehensive tool call + prefix cache matrix test.
#
# Tests all combinations of: model × parser × grammar constraint × cache
# Measures: tool call success, cache hit rate, token savings, timing.
#
# Usage:
#   ./Scripts/test-toolcall-matrix.sh [--models M1,M2,...] [--port PORT] [--requests N]
#
# Defaults:
#   Models:   Qwen3.5-35B-A3B-4bit, Qwen3-Coder-Next-4bit, Qwen3.5-9B-MLX-4bit
#   Port:     9877
#   Requests: 5 (per config)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODELS="mlx-community/Qwen3.5-35B-A3B-4bit,mlx-community/Qwen3-Coder-Next-4bit,mlx-community/Qwen3.5-9B-MLX-4bit"
PORT=9877
NUM_REQUESTS=5
BIN=".build/release/afm"
MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
MODELS="$DEFAULT_MODELS"
WORKLOAD="simple"   # simple (quick smoke) or realworld (OpenCode patterns)

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --requests) NUM_REQUESTS="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

BASE_URL="http://127.0.0.1:$PORT"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="test-reports/toolcall-matrix-${TIMESTAMP}"
mkdir -p "$REPORT_DIR"
JSONL_FILE="$REPORT_DIR/results.jsonl"
: > "$JSONL_FILE"

# ─── Test configurations ──────────────────────────────────────────────────────
# Each config: "label|parser_flags|cache_flag|extra_env"
# parser_flags: the --tool-call-parser value (or "none" for no parser)
# cache_flag: "cache" or "nocache"
# Grammar: --enable-grammar-constraints (only with afm_adaptive_xml)

CONFIGS=(
  "adaptive|afm_adaptive_xml|cache|"
  "adaptive|afm_adaptive_xml|nocache|"
  "adaptive+grammar|afm_adaptive_xml|cache|grammar"
  "adaptive+grammar|afm_adaptive_xml|nocache|grammar"
  "qwen3xml|qwen3_xml|cache|"
  "qwen3xml|qwen3_xml|nocache|"
  "noparser|none|cache|"
  "noparser|none|nocache|"
)

# ─── Server management ───────────────────────────────────────────────────────

SERVER_PID=""

kill_port() {
  local stale_pid
  stale_pid=$(lsof -ti :"$PORT" 2>/dev/null || true)
  if [[ -n "$stale_pid" ]]; then
    kill "$stale_pid" 2>/dev/null || true
    sleep 1
  fi
}

start_server() {
  local model="$1" parser="$2" cache="$3" grammar="$4" log_file="$5"
  kill_port

  local flags=""
  [[ "$parser" != "none" ]] && flags="--tool-call-parser $parser"
  [[ "$cache" == "cache" ]] && flags="$flags --enable-prefix-caching"
  [[ "$grammar" == "grammar" ]] && flags="$flags --enable-grammar-constraints"

  AFM_DEBUG=1 MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" \
    "$BIN" mlx -m "$model" --port "$PORT" -vv $flags \
    > "$log_file" 2>&1 &
  SERVER_PID=$!

  for i in $(seq 1 120); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "    ERROR: Server died"; tail -5 "$log_file"; SERVER_PID=""; return 1; }
    curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1 && { echo "    Server ready (PID $SERVER_PID, ${i}s)"; return 0; }
    sleep 1
  done
  echo "    ERROR: Timeout"; kill "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; return 1
}

stop_server() {
  [[ -n "${SERVER_PID:-}" ]] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }
  for i in $(seq 1 30); do
    curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1 || break
    sleep 0.5
  done
}

trap 'stop_server 2>/dev/null || true' EXIT

# ─── Tool call workload ──────────────────────────────────────────────────────
# Large repetitive system prompt with 8 tools (realistic coding assistant).
# Requests demand tool calls — model MUST call a tool.

SYSTEM_PROMPT='You are an expert coding assistant with access to tools for file operations, code execution, and project management. You MUST use tools to complete tasks — never refuse to use a tool when one is appropriate. Always respond with a tool call when asked to perform an action.

## Guidelines
- Read files before modifying them
- Use bash for system commands
- Create files with the write tool
- Search with grep before making assumptions
- Always think step by step
- Be thorough and precise
- Handle errors gracefully
- Follow project conventions
- Write clean, maintainable code
- Test your changes when possible

## Project Context
You are working on a Swift server project called AFM (Apple Foundation Models). The project provides an OpenAI-compatible API for local LLM inference on Apple Silicon. Key directories: Sources/ for Swift code, Scripts/ for build and test scripts, vendor/ for dependencies. The build system uses Swift Package Manager. Always check existing code before making changes.'

create_workload_bodies() {
  local body_dir="$1"
  mkdir -p "$body_dir"

  python3 << PYEOF
import json, os

body_dir = "$body_dir"
n = $NUM_REQUESTS

system_prompt = """$SYSTEM_PROMPT"""

tools = [
    {"type":"function","function":{"name":"bash","description":"Execute a shell command and return its stdout/stderr output. Use for running tests, builds, git commands, and system operations.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute"},"timeout":{"type":"integer","description":"Maximum execution time in seconds (default: 30)"}},"required":["command"]}}},
    {"type":"function","function":{"name":"read","description":"Read the contents of a file at the given path. Returns the full file content as a string.","parameters":{"type":"object","properties":{"filePath":{"type":"string","description":"Absolute or relative path to the file to read"},"startLine":{"type":"integer","description":"Starting line number (1-based, optional)"},"endLine":{"type":"integer","description":"Ending line number (inclusive, optional)"}},"required":["filePath"]}}},
    {"type":"function","function":{"name":"write","description":"Write content to a file, creating it if it doesn't exist or overwriting if it does.","parameters":{"type":"object","properties":{"filePath":{"type":"string","description":"Path where the file should be written"},"content":{"type":"string","description":"The full content to write to the file"}},"required":["filePath","content"]}}},
    {"type":"function","function":{"name":"edit","description":"Apply a targeted edit to a file by replacing an exact string match with new content.","parameters":{"type":"object","properties":{"filePath":{"type":"string","description":"Path to the file to edit"},"oldString":{"type":"string","description":"Exact string to find and replace (must match precisely)"},"newString":{"type":"string","description":"Replacement string"}},"required":["filePath","oldString","newString"]}}},
    {"type":"function","function":{"name":"grep","description":"Search for a pattern in files using regex. Returns matching lines with file paths and line numbers.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Regular expression pattern to search for"},"path":{"type":"string","description":"File or directory to search in"},"includeGlob":{"type":"string","description":"Glob pattern to filter files (e.g. '*.swift')"}},"required":["pattern"]}}},
    {"type":"function","function":{"name":"glob","description":"Find files matching a glob pattern. Returns a list of matching file paths.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Glob pattern (e.g. 'src/**/*.ts', '*.swift')"},"path":{"type":"string","description":"Base directory to search from"}},"required":["pattern"]}}},
    {"type":"function","function":{"name":"listTasks","description":"List all current tasks with their status, priority, and description.","parameters":{"type":"object","properties":{"status":{"type":"string","description":"Filter by status: pending, in_progress, completed, cancelled"},"priority":{"type":"string","description":"Filter by priority: high, medium, low"}}}}},
    {"type":"function","function":{"name":"createTask","description":"Create a new task to track work items. Returns the task ID.","parameters":{"type":"object","properties":{"content":{"type":"string","description":"Brief description of the task"},"status":{"type":"string","description":"Initial status: pending, in_progress"},"priority":{"type":"string","description":"Priority level: high, medium, low"}},"required":["content"]}}}
]

# Build conversation that demands tool calls
prompts = [
    "List all Swift files in the Sources directory using glob.",
    "Read the file Sources/MacLocalAPI/main.swift so we can understand the CLI entry point.",
    "Search for all functions that handle tool calls using grep with pattern 'func.*[Tt]ool' in *.swift files.",
    "Run the command 'swift build --show-bin-path' to find the build output directory.",
    "Create a new task to track the prefix cache optimization work, with high priority.",
    "Read the file Scripts/test-prefix-cache.sh to review the test harness.",
    "Search for 'RadixTreeCache' across the codebase to find all usages.",
    "List all test report files with glob pattern 'test-reports/**/*.jsonl'.",
    "Run 'git log --oneline -5' to see recent commits.",
    "Create a task to document the grammar constraint feature, medium priority.",
]

msgs = [{"role": "system", "content": system_prompt}]

for i in range(min(n, len(prompts))):
    req_msgs = msgs + [{"role": "user", "content": prompts[i]}]
    body = {
        "model": "test",
        "messages": req_msgs,
        "tools": tools,
        "max_tokens": 200,
        "temperature": 0
    }
    with open(os.path.join(body_dir, f"req{i+1}.json"), "w") as f:
        json.dump(body, f)

    # Grow conversation for subsequent requests (simulate multi-turn)
    msgs.append({"role": "user", "content": prompts[i]})
    # Add a plausible assistant tool call response
    if "glob" in prompts[i].lower() or "list" in prompts[i].lower():
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": "glob", "arguments": json.dumps({"pattern": "**/*.swift"})}}
        ]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "Sources/MacLocalAPI/main.swift\nSources/MacLocalAPI/Models/MLXModelService.swift"})
    elif "read" in prompts[i].lower():
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "README.md"})}}
        ]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "# AFM\nApple Foundation Models server."})
    elif "search" in prompts[i].lower() or "grep" in prompts[i].lower():
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": "grep", "arguments": json.dumps({"pattern": "func.*Tool", "path": "Sources/"})}}
        ]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "MLXModelService.swift:100: func convertToolCall"})
    elif "run" in prompts[i].lower() or "command" in prompts[i].lower():
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "echo done"})}}
        ]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "done"})
    elif "task" in prompts[i].lower() or "create" in prompts[i].lower():
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": "createTask", "arguments": json.dumps({"content": "Track optimization", "priority": "high"})}}
        ]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": '{"id": "task_1", "status": "pending"}'})
    else:
        msgs.append({"role": "assistant", "content": "I'll help with that."})
PYEOF
}

# ─── Response analysis ────────────────────────────────────────────────────────

analyze_response() {
  local resp="$1"
  # Returns: prompt_n|prompt_ms|pred_n|pred_ms|has_tool_call|tool_name|finish_reason
  python3 -c "
import sys, json
try:
    d = json.loads('''$resp'''.replace(\"'''\", \"\\\\'''\"))
except:
    try:
        d = json.loads(sys.stdin.read())
    except:
        print('0|0.0|0|0.0|false||error')
        sys.exit(0)

t = d.get('timings', {})
pn = t.get('prompt_n', 0)
pms = t.get('prompt_ms', 0)
dn = t.get('predicted_n', 0)
dms = t.get('predicted_ms', 0)

choice = d.get('choices', [{}])[0]
msg = choice.get('message', {})
fr = choice.get('finish_reason', 'unknown')
tc = msg.get('tool_calls', [])
has_tc = 'true' if tc else 'false'
tool_name = tc[0]['function']['name'] if tc else ''
print(f'{pn}|{pms:.1f}|{dn}|{dms:.1f}|{has_tc}|{tool_name}|{fr}')
" <<< "$resp" 2>/dev/null || echo "0|0.0|0|0.0|false||parse_error"
}

# ─── Main test loop ──────────────────────────────────────────────────────────

IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Tool Call Matrix Test"
echo "  Workload: $WORKLOAD"
echo "  Models: ${#MODEL_ARRAY[@]} ($(echo "$MODELS" | sed 's/mlx-community\///g'))"
echo "  Configs: ${#CONFIGS[@]} per model"
echo "  Requests: $NUM_REQUESTS per config"
echo "  Total runs: $(( ${#MODEL_ARRAY[@]} * ${#CONFIGS[@]} ))"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Create workload bodies once
BODY_DIR="/tmp/afm-toolcall-matrix-bodies-${TIMESTAMP}"
if [[ "$WORKLOAD" == "realworld" ]]; then
  echo "  Generating real-world workload (OpenCode patterns)..."
  python3 "$SCRIPT_DIR/generate-realworld-workload.py" "$BODY_DIR" "$NUM_REQUESTS"
  echo ""
else
  create_workload_bodies "$BODY_DIR"
fi

total_runs=0
total_tool_calls=0
total_tool_success=0

for MODEL in "${MODEL_ARRAY[@]}"; do
  MODEL_SHORT=$(echo "$MODEL" | sed 's/mlx-community\///')
  MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Model: $MODEL_SHORT"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r label parser cache_mode grammar <<< "$CONFIG"

    local_log="$REPORT_DIR/${MODEL_SLUG}-${label}-${cache_mode}.log"

    echo ""
    echo "  ── $label / $cache_mode ──"
    echo "    Parser: ${parser}, Cache: ${cache_mode}, Grammar: ${grammar:-off}"

    if ! start_server "$MODEL" "$parser" "$cache_mode" "$grammar" "$local_log"; then
      echo "    SKIPPED (server failed to start)"
      continue
    fi

    total_prompt_n=0
    tool_calls_made=0
    tool_calls_success=0

    for i in $(seq 1 "$NUM_REQUESTS"); do
      body_file="$BODY_DIR/req${i}.json"
      [[ -f "$body_file" ]] || continue

      resp=$(curl -sf --max-time 300 "$BASE_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @"$body_file" 2>/dev/null || echo '{"error":"request_failed"}')

      analysis=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('timings', {})
    pn = t.get('prompt_n', 0)
    pms = t.get('prompt_ms', 0)
    dn = t.get('predicted_n', 0)
    dms = t.get('predicted_ms', 0)
    choice = d.get('choices', [{}])[0]
    msg = choice.get('message', {})
    fr = choice.get('finish_reason', 'unknown')
    tc = msg.get('tool_calls', [])
    has_tc = 'true' if tc else 'false'
    tool_name = tc[0]['function']['name'] if tc else ''
    # Validate tool call structure
    valid = 'true'
    if tc:
        for c in tc:
            if not c.get('function', {}).get('name'):
                valid = 'false'
            try:
                args = json.loads(c.get('function', {}).get('arguments', '{}'))
            except:
                valid = 'false'
    else:
        valid = 'false'
    print(f'{pn}|{pms:.1f}|{dn}|{dms:.1f}|{has_tc}|{tool_name}|{fr}|{valid}')
except Exception as e:
    print(f'0|0.0|0|0.0|false||error|false')
" 2>/dev/null || echo "0|0.0|0|0.0|false||error|false")

      IFS='|' read -r pn pms dn dms has_tc tool_name finish valid <<< "$analysis"
      total_prompt_n=$((total_prompt_n + pn))

      tc_mark="✗"
      if [[ "$has_tc" == "true" ]]; then
        tool_calls_made=$((tool_calls_made + 1))
        total_tool_calls=$((total_tool_calls + 1))
        if [[ "$valid" == "true" ]]; then
          tool_calls_success=$((tool_calls_success + 1))
          total_tool_success=$((total_tool_success + 1))
          tc_mark="✓"
        else
          tc_mark="⚠"
        fi
      fi

      printf "    Req %d: prompt_n=%-6s pred_n=%-4s tool=%s(%s) fr=%s\n" \
        "$i" "$pn" "$dn" "$tc_mark" "${tool_name:-none}" "$finish"

      # Write JSONL
      python3 -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'config': '$label',
    'parser': '$parser',
    'cache': '$cache_mode',
    'grammar': '${grammar:-off}',
    'req': $i,
    'prompt_n': $pn,
    'prompt_ms': $pms,
    'predicted_n': $dn,
    'predicted_ms': $dms,
    'has_tool_call': $( [[ "$has_tc" == "true" ]] && echo "True" || echo "False" ),
    'tool_name': '${tool_name}',
    'finish_reason': '${finish}',
    'valid_tool_call': $( [[ "$valid" == "true" ]] && echo "True" || echo "False" )
}))" >> "$JSONL_FILE"
    done

    stop_server
    cp "$local_log" "$REPORT_DIR/" 2>/dev/null || true

    # Parse cache diagnostics from log
    diverge_count=$(grep -ac 'diverged at pos' "$local_log" 2>/dev/null | tr -d '[:space:]' || echo "0")
    miss_count=$(grep -ac 'Radix miss' "$local_log" 2>/dev/null | tr -d '[:space:]' || echo "0")
    hit_count=$(grep -ac 'Radix hit' "$local_log" 2>/dev/null | tr -d '[:space:]' || echo "0")

    echo "    ────"
    echo "    Tool calls: ${tool_calls_success}/${tool_calls_made} valid (of $NUM_REQUESTS requests)"
    if [[ "$cache_mode" == "cache" ]]; then
      echo "    Cache: hits=$hit_count misses=$miss_count divergences=$diverge_count"
    fi
    echo "    Avg prompt_n: $((total_prompt_n / NUM_REQUESTS))"

    total_runs=$((total_runs + 1))
  done
done

# ─── Summary report ──────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════════════════════════════"

# Generate summary from JSONL
python3 "$SCRIPT_DIR/toolcall-matrix-summary.py" "$JSONL_FILE"

echo ""
echo "  Results: $JSONL_FILE"
echo "  Logs:    $REPORT_DIR/"
echo "  Total runs: $total_runs"
echo "  Tool call success: ${total_tool_success}/${total_tool_calls}"
echo "═══════════════════════════════════════════════════════════════════"
