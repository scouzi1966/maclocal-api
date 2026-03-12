#!/bin/bash
# Prefix Cache A/B Test Harness
# Deterministic comparison of afm with prefix caching ON vs OFF.
# Runs workloads (W1: OpenCode session, W2: Identical repeats, W3: Growing conversation),
# captures prompt_n/prompt_ms timings, parses server logs for cache hits/misses,
# and generates a summary report with JSONL output.
#
# Usage:
#   ./Scripts/test-prefix-cache.sh --model MODEL [--port PORT] [--bin BIN] [--workload W1|W2|W3|all]
#
# Example:
#   ./Scripts/test-prefix-cache.sh --model mlx-community/Qwen3-Coder-Next-4bit --workload W2

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
PORT=9877
MODEL=""
BIN=".build/release/afm"
WORKLOAD="all"
MODEL_CACHE="/Volumes/edata/models/vesta-test-cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVER_PID=""
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_DIR="$PROJECT_ROOT/test-reports/prefix-cache-${TIMESTAMP}"

# ─── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ERROR: --model is required"
  echo "Usage: $0 --model MODEL [--port PORT] [--bin BIN] [--workload W1|W2|W3|all]"
  exit 1
fi
if [[ ! "$WORKLOAD" =~ ^(W1|W2|W3|all)$ ]]; then
  echo "ERROR: --workload must be W1, W2, W3, or all"; exit 1
fi

mkdir -p "$REPORT_DIR"
BASE_URL="http://127.0.0.1:$PORT"
JSONL_FILE="$REPORT_DIR/results.jsonl"
: > "$JSONL_FILE"

# ─── Server management ───────────────────────────────────────────────────────

start_server() {
  local extra_flags="$1" log_file="$2"
  # Kill any leftover process on the port
  local stale_pid
  stale_pid=$(lsof -ti :"$PORT" 2>/dev/null || true)
  if [[ -n "$stale_pid" ]]; then
    echo "  Killing stale process on port $PORT (PID $stale_pid)..."
    kill "$stale_pid" 2>/dev/null || true; sleep 1
  fi
  echo "  Starting server (flags: ${extra_flags:-none})..."
  AFM_DEBUG=1 MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" \
    "$BIN" mlx -m "$MODEL" --port "$PORT" --tool-call-parser afm_adaptive_xml -vv $extra_flags \
    > "$log_file" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    # Check BOTH: our process is alive AND port responds
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: Server died during startup"; tail -20 "$log_file" 2>/dev/null; SERVER_PID=""; return 1; }
    curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1 && { echo "  Server ready (PID $SERVER_PID, waited ${i}s)"; return 0; }
    sleep 1
  done
  echo "ERROR: Server failed to start in 120s"; kill "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; return 1
}

stop_server() {
  [[ -n "${SERVER_PID:-}" ]] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }
  # Wait for port to be released
  for i in $(seq 1 30); do
    curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1 || break
    sleep 0.5
  done
}

trap 'stop_server 2>/dev/null || true' EXIT

# ─── Helpers ──────────────────────────────────────────────────────────────────

send_request() {
  curl -sf --max-time 120 "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$1" 2>/dev/null || echo '{"error":"request_failed"}'
}

extract_timings() {
  echo "$1" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('timings', {})
    print(f'{t.get(\"prompt_n\",0)}|{t.get(\"prompt_ms\",0):.1f}|{t.get(\"predicted_n\",0)}|{t.get(\"predicted_ms\",0):.1f}')
except: print('0|0.0|0|0.0')
"
}

write_jsonl() {
  local model="$1" wl="$2" req="$3" cache="$4"
  local pn="$5" pms="$6" dn="$7" dms="$8"
  python3 -c "
import json; print(json.dumps({'model':'$model','workload':'$wl','req':$req,'cache':'$cache','prompt_n':$pn,'prompt_ms':$pms,'predicted_n':$dn,'predicted_ms':$dms}))
" >> "$JSONL_FILE"
}

# Run a set of requests against a running server, storing results in global arrays.
# Usage: run_requests <body_dir_or_file> <count> <array_prefix>
# If body_dir_or_file is a directory, sends req1.json..reqN.json; if a file, sends it N times.
# Populates: ${prefix}_prompt_n, ${prefix}_prompt_ms, ${prefix}_pred_n, ${prefix}_pred_ms
run_requests() {
  local body_source="$1" count="$2" prefix="$3"
  for i in $(seq 1 "$count"); do
    local body_file
    if [[ -d "$body_source" ]]; then
      body_file="$body_source/req${i}.json"
    else
      body_file="$body_source"
    fi
    local resp timings
    resp=$(send_request "$body_file")
    timings=$(extract_timings "$resp")
    IFS='|' read -r pn pms dn dms <<< "$timings"
    eval "${prefix}_prompt_n+=(\"\$pn\")"
    eval "${prefix}_prompt_ms+=(\"\$pms\")"
    eval "${prefix}_pred_n+=(\"\$dn\")"
    eval "${prefix}_pred_ms+=(\"\$dms\")"
    echo "    Req $i: prompt_n=$pn prompt_ms=${pms}ms"
  done
}

# Run full A/B comparison for a workload: start server with cache ON, send requests,
# stop server, start without cache, send same requests, stop server, write JSONL, print report.
run_ab_comparison() {
  local wl_id="$1" wl_name="$2" count="$3" body_source="$4"
  local log_on="/tmp/afm-prefix-test-$(echo "$wl_id" | tr 'A-Z' 'a-z')-cache-on.log"
  local log_off="/tmp/afm-prefix-test-$(echo "$wl_id" | tr 'A-Z' 'a-z')-cache-off.log"

  # Clear result arrays
  on_prompt_n=(); on_prompt_ms=(); on_pred_n=(); on_pred_ms=()
  off_prompt_n=(); off_prompt_ms=(); off_pred_n=(); off_pred_ms=()

  # Cache ON
  echo "  [Cache ON]"
  start_server "--enable-prefix-caching" "$log_on" || return 1
  run_requests "$body_source" "$count" "on"
  stop_server; echo ""

  # Cache OFF
  echo "  [Cache OFF]"
  start_server "" "$log_off" || return 1
  run_requests "$body_source" "$count" "off"
  stop_server; echo ""

  # Write JSONL
  for i in $(seq 0 $((count - 1))); do
    write_jsonl "$MODEL" "$wl_id" "$((i+1))" "on" \
      "${on_prompt_n[$i]}" "${on_prompt_ms[$i]}" "${on_pred_n[$i]}" "${on_pred_ms[$i]}"
    write_jsonl "$MODEL" "$wl_id" "$((i+1))" "off" \
      "${off_prompt_n[$i]}" "${off_prompt_ms[$i]}" "${off_pred_n[$i]}" "${off_pred_ms[$i]}"
  done

  # Copy logs
  cp "$log_on" "$REPORT_DIR/afm-$(echo "$wl_id" | tr 'A-Z' 'a-z')-cache-on.log"
  cp "$log_off" "$REPORT_DIR/afm-$(echo "$wl_id" | tr 'A-Z' 'a-z')-cache-off.log"

  # Print report
  print_report "$wl_id" "$wl_name" "$count"

  # Print divergence diagnostics
  print_divergence_info "$REPORT_DIR/afm-$(echo "$wl_id" | tr 'A-Z' 'a-z')-cache-on.log" "$wl_id Cache ON"
}

# ─── Report printing ─────────────────────────────────────────────────────────

print_report() {
  local wl_id="$1" wl_name="$2" count="$3"

  echo "  Req#  Cache ON         Cache OFF        Tokens Saved"
  echo "        prompt_n  ms     prompt_n  ms     count    %"
  echo "  ────  ────────  ────   ────────  ────   ─────    ──"

  local total_on_n=0 total_off_n=0 total_on_ms="0" total_off_ms="0" hit_count=0

  for i in $(seq 0 $((count - 1))); do
    local on_n="${on_prompt_n[$i]}" on_ms="${on_prompt_ms[$i]}"
    local off_n="${off_prompt_n[$i]}" off_ms="${off_prompt_ms[$i]}"
    total_on_n=$((total_on_n + on_n)); total_off_n=$((total_off_n + off_n))
    total_on_ms=$(python3 -c "print(${total_on_ms} + ${on_ms})")
    total_off_ms=$(python3 -c "print(${total_off_ms} + ${off_ms})")

    local saved_n=0 saved_pct="0"
    if [[ "$off_n" -gt 0 ]]; then
      saved_n=$((off_n - on_n))
      saved_pct=$(python3 -c "print(f'{($saved_n / $off_n) * 100:.0f}')")
    fi
    [[ "$saved_n" -gt 0 ]] && hit_count=$((hit_count + 1))

    printf "  %-4d  %-8s  %-5s  %-8s  %-5s  %-5s    %s%%\n" \
      "$((i+1))" "$on_n" "$on_ms" "$off_n" "$off_ms" "$saved_n" "$saved_pct"
  done

  echo ""
  echo "  Summary:"
  if [[ "$total_off_n" -gt 0 ]]; then
    local saved_total=$((total_off_n - total_on_n))
    local saved_pct_total=$(python3 -c "print(f'{($saved_total / $total_off_n) * 100:.1f}')")
    local saved_ms=$(python3 -c "print(f'{$total_off_ms - $total_on_ms:.1f}')")
    local saved_ms_pct="0.0"
    python3 -c "import sys; sys.exit(0 if $total_off_ms > 0 else 1)" && \
      saved_ms_pct=$(python3 -c "print(f'{(($total_off_ms - $total_on_ms) / $total_off_ms) * 100:.1f}')")
    echo "    Total prompt tokens (cache ON):  $total_on_n"
    echo "    Total prompt tokens (cache OFF): $total_off_n"
    echo "    Tokens saved: $saved_total ($saved_pct_total%)"
    echo "    Time saved: ${saved_ms}ms ($saved_ms_pct%)"
    echo "    Cache hit rate: $hit_count/$count"
  else
    echo "    No valid responses received."
  fi
}

print_divergence_info() {
  local log_file="$1" label="$2"
  local diverge_lines miss_lines
  diverge_lines=$(grep -ac 'diverged at pos' "$log_file" 2>/dev/null | tr -d '[:space:]' || echo "0")
  miss_lines=$(grep -ac 'Radix miss' "$log_file" 2>/dev/null | tr -d '[:space:]' || echo "0")

  if [[ "$diverge_lines" -gt 0 || "$miss_lines" -gt 0 ]]; then
    echo ""
    echo "  Cache Diagnostics ($label):"
    if [[ "$diverge_lines" -gt 0 ]]; then
      echo "    Divergence points ($diverge_lines):"
      grep -a 'diverged at pos' "$log_file" 2>/dev/null | head -5 | while IFS= read -r line; do
        echo "      $line"
      done
      [[ "$diverge_lines" -gt 5 ]] && echo "      ... ($((diverge_lines - 5)) more)"
    fi
    [[ "$miss_lines" -gt 0 ]] && echo "    Cache misses: $miss_lines"
  fi
}

# ─── W2 Workload: Identical Repeats ──────────────────────────────────────────

run_w2() {
  echo ""
  echo "── W2: Identical Repeats (10 requests) ──────────────────────────"
  echo ""

  local body_file="/tmp/afm-prefix-test-w2-body.json"
  cat > "$body_file" <<'ENDBODY'
{
  "model":"test",
  "messages":[
    {"role":"system","content":"You are a helpful coding assistant. You have access to tools for reading files, writing files, and executing commands. Always think step by step before using any tool. Be thorough and precise in your responses."},
    {"role":"user","content":"What is 2+2?"}
  ],
  "tools":[
    {"type":"function","function":{"name":"bash","description":"Execute a bash command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The command to execute"}},"required":["command"]}}},
    {"type":"function","function":{"name":"write","description":"Write content to a file","parameters":{"type":"object","properties":{"filePath":{"type":"string","description":"Path to write"},"content":{"type":"string","description":"File content"}},"required":["filePath","content"]}}},
    {"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"filePath":{"type":"string","description":"Path to read"}},"required":["filePath"]}}}
  ],
  "max_tokens":50,
  "temperature":0
}
ENDBODY

  run_ab_comparison "W2" "Identical Repeats" 10 "$body_file"
}

# ─── W1 Workload: OpenCode Session ───────────────────────────────────────────

run_w1() {
  local test_data_file="$SCRIPT_DIR/test-data/opencode-system-prompt.json"
  if [[ ! -f "$test_data_file" ]]; then
    echo ""
    echo "── W1: OpenCode Session ───────────────────────────────────────────"
    echo "  SKIPPED: $test_data_file not found"
    echo "  Create this file with {\"system\": \"...\", \"tools\": [...]} to enable W1."
    echo ""
    return 0
  fi

  echo ""
  echo "── W1: OpenCode Session (5 requests) ────────────────────────────"
  echo ""

  local body_dir="/tmp/afm-prefix-test-w1-bodies"
  mkdir -p "$body_dir"

  python3 << PYEOF
import json, os

with open("$test_data_file") as f:
    data = json.load(f)

system_msg = {"role": "system", "content": data["system"]}
tools = data["tools"]
body_dir = "$body_dir"

msgs1 = [system_msg, {"role": "user", "content": "List files in the current directory"}]
with open(os.path.join(body_dir, "req1.json"), "w") as f:
    json.dump({"model": "test", "messages": msgs1, "tools": tools, "max_tokens": 50, "temperature": 0}, f)

msgs2 = msgs1 + [
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "README.md\nPackage.swift\nSources/"},
    {"role": "user", "content": "Now read the README"}
]
with open(os.path.join(body_dir, "req2.json"), "w") as f:
    json.dump({"model": "test", "messages": msgs2, "tools": tools, "max_tokens": 50, "temperature": 0}, f)

msgs3 = msgs2 + [
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_2", "type": "function", "function": {"name": "read", "arguments": "{\"filePath\":\"README.md\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_2", "content": "# Project\nA sample project."},
    {"role": "user", "content": "Summarize the project"}
]
with open(os.path.join(body_dir, "req3.json"), "w") as f:
    json.dump({"model": "test", "messages": msgs3, "tools": tools, "max_tokens": 50, "temperature": 0}, f)

msgs4 = msgs3 + [
    {"role": "assistant", "content": "This is a sample project with a README and Swift package structure."},
    {"role": "user", "content": "Create a hello world file"}
]
with open(os.path.join(body_dir, "req4.json"), "w") as f:
    json.dump({"model": "test", "messages": msgs4, "tools": tools, "max_tokens": 50, "temperature": 0}, f)

msgs5 = msgs4 + [
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_3", "type": "function", "function": {"name": "write", "arguments": "{\"filePath\":\"hello.py\",\"content\":\"print('Hello, World!')\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_3", "content": "File written."},
    {"role": "user", "content": "Run the file"}
]
with open(os.path.join(body_dir, "req5.json"), "w") as f:
    json.dump({"model": "test", "messages": msgs5, "tools": tools, "max_tokens": 50, "temperature": 0}, f)
PYEOF

  run_ab_comparison "W1" "OpenCode Session" 5 "$body_dir"
  rm -rf "$body_dir"
}

# ─── W3 Workload: Growing Conversation ───────────────────────────────────────

run_w3() {
  echo ""
  echo "── W3: Growing Conversation (10 requests) ──────────────────────"
  echo ""

  local body_dir="/tmp/afm-prefix-test-w3-bodies"
  mkdir -p "$body_dir"

  W3_BODY_DIR="$body_dir" python3 << 'PYEOF'
import json, os

body_dir = os.environ["W3_BODY_DIR"]

system_msg = {"role": "system", "content": "You are a helpful coding assistant. You have access to tools for reading files, writing files, and executing commands. Always think step by step before using any tool. Be thorough and precise in your responses."}
tools = [
    {"type": "function", "function": {"name": "bash", "description": "Execute a bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to execute"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write content to a file", "parameters": {"type": "object", "properties": {"filePath": {"type": "string", "description": "Path to write"}, "content": {"type": "string", "description": "File content"}}, "required": ["filePath", "content"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"filePath": {"type": "string", "description": "Path to read"}}, "required": ["filePath"]}}}
]

questions = [
    "What is the capital of France?",
    "What language do they speak there?",
    "What is the population?",
    "Name three famous landmarks.",
    "What is the currency used?",
    "Tell me about the history of the Eiffel Tower.",
    "What foods is France known for?",
    "What is the climate like?",
    "Name some famous French authors.",
    "What are the neighboring countries?"
]
answers = [
    "The capital of France is Paris.",
    "The primary language spoken in France is French.",
    "France has a population of approximately 67 million people.",
    "Three famous landmarks are the Eiffel Tower, the Louvre, and Notre-Dame.",
    "The currency used in France is the Euro.",
    "The Eiffel Tower was built in 1889 for the World's Fair.",
    "France is known for croissants, baguettes, cheese, and wine.",
    "France has a temperate climate with mild winters and warm summers.",
    "Famous French authors include Victor Hugo, Albert Camus, and Marcel Proust.",
    "France is bordered by Germany, Belgium, Luxembourg, Switzerland, Italy, and Spain."
]

msgs = [system_msg]
for i in range(10):
    msgs.append({"role": "user", "content": questions[i]})
    with open(os.path.join(body_dir, f"req{i+1}.json"), "w") as f:
        json.dump({"model": "test", "messages": list(msgs), "tools": tools, "max_tokens": 50, "temperature": 0}, f)
    msgs.append({"role": "assistant", "content": answers[i]})
PYEOF

  run_ab_comparison "W3" "Growing Conversation" 10 "$body_dir"
  rm -rf "$body_dir"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Prefix Cache A/B Test"
echo "  Model: $MODEL"
echo "  Date:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Port:  $PORT"
echo "  Bin:   $BIN"
echo "═══════════════════════════════════════════════════════════════"

run_start=$(date +%s)

[[ "$WORKLOAD" == "W2" || "$WORKLOAD" == "all" ]] && run_w2
[[ "$WORKLOAD" == "W1" || "$WORKLOAD" == "all" ]] && run_w1
[[ "$WORKLOAD" == "W3" || "$WORKLOAD" == "all" ]] && run_w3

run_end=$(date +%s)
run_duration=$((run_end - run_start))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Test completed in ${run_duration}s"
echo "  Results: $JSONL_FILE"
echo "  Logs:    $REPORT_DIR/"
echo "═══════════════════════════════════════════════════════════════"
echo ""
