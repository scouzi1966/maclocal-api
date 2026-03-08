#!/opt/homebrew/bin/bash
#
# Benchmark: AFM prefix cache performance
#
# Runs identical and prefix-sharing requests with and without --enable-prefix-caching,
# measuring TTFT (prompt_time), cached_tokens, and decode speed.
# Generates JPEG comparison charts.
#
# Usage:
#   ./Scripts/benchmark-prefix-cache.sh [--model MODEL] [--runs N] [--max-tokens N]
#
# Prerequisites: .build/release/afm must exist.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="${MODEL:-mlx-community/Qwen3-0.6B-4bit}"
PORT=9987
CACHE_DIR="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
RUNS=3
MAX_TOKENS=50
TEMPERATURE=0.0
BIN="${AFM_BIN:-.build/release/afm}"
EXTRA_FLAGS=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_DIR="$PROJECT_ROOT/test-reports/prefix-cache-bench-${TIMESTAMP}"
RESULTS_FILE="$REPORT_DIR/results.jsonl"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL="$2"; shift 2 ;;
    --runs)       RUNS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --extra-flags) EXTRA_FLAGS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model MODEL] [--runs N] [--max-tokens N] [--port PORT] [--extra-flags FLAGS]"
      echo ""
      echo "Benchmarks AFM prefix cache: with vs without --enable-prefix-caching."
      echo "Generates JPEG charts in test-reports/prefix-cache-bench-TIMESTAMP/"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$REPORT_DIR"
> "$RESULTS_FILE"

# ── Prompts ───────────────────────────────────────────────────────────────────
# Simulate OpenCode-style requests: long system prompt + tools + varying user messages

SYSTEM_PROMPT='You are an expert software engineer. You have access to tools for reading files, writing files, running shell commands, and searching codebases. Always use the appropriate tool for the task. Think step by step before acting. When writing code, follow best practices and include error handling. When debugging, check logs and error messages carefully before proposing fixes.'

# Tool definitions (simulating OpenCode tool schema)
TOOLS='[
  {"type":"function","function":{"name":"read_file","description":"Read a file from disk","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path to read"},"offset":{"type":"integer","description":"Line offset"},"limit":{"type":"integer","description":"Max lines"}},"required":["path"]}}},
  {"type":"function","function":{"name":"write_file","description":"Write content to a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string","description":"File content"}},"required":["path","content"]}}},
  {"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The command to run"}},"required":["command"]}}},
  {"type":"function","function":{"name":"grep","description":"Search file contents","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Regex pattern"},"path":{"type":"string","description":"Directory to search"},"glob":{"type":"string","description":"File glob filter"}},"required":["pattern"]}}},
  {"type":"function","function":{"name":"glob","description":"Find files by pattern","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Glob pattern"},"path":{"type":"string","description":"Base directory"}},"required":["pattern"]}}}
]'

# Different user messages (same system prompt + tools = shared prefix)
USER_MESSAGES=(
  "Read the file main.py and tell me what it does."
  "Search for all Python files that import argparse."
  "Create a new file called utils.py with a function to parse command line arguments."
  "Run the tests and show me any failures."
  "Find all TODO comments in the codebase."
)

USER_LABELS=(
  "read-file"
  "search-imports"
  "create-utils"
  "run-tests"
  "find-todos"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

kill_port() {
  local port=$1
  local pid
  pid=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  fi
}

wait_for_server() {
  local port=$1
  local timeout=${2:-120}
  local deadline=$((SECONDS + timeout))
  while [ $SECONDS -lt $deadline ]; do
    if curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Send a request and return JSON with timing info
# If $7 is set, it's a path to a JSON file containing the full messages array
send_request() {
  local port=$1
  local user_msg="$2"
  local label="$3"
  local mode="$4"      # "cache" or "nocache"
  local run_num="$5"
  local stream="$6"    # "true" or "false"
  local messages_file="${7:-}"

  python3 << PYEOF
import json, time, sys

port = ${port}
max_tokens = ${MAX_TOKENS}
temperature = ${TEMPERATURE}
stream = "${stream}" == "true"
label = "${label}"
mode = "${mode}"
run_num = ${run_num}
messages_file = "${messages_file}"

tools = json.loads('''${TOOLS}''')

if messages_file:
    with open(messages_file) as f:
        messages = json.load(f)
else:
    user_msg = json.loads($(python3 -c "import json,sys; print(json.dumps(json.dumps(sys.argv[1])))" "$user_msg"))
    system_prompt = json.loads($(python3 -c "import json,sys; print(json.dumps(json.dumps(sys.argv[1])))" "$SYSTEM_PROMPT"))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]

payload = {
    "model": "any",
    "messages": messages,
    "tools": tools,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "stream": stream
}

import urllib.request

data = json.dumps(payload).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)

start = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode()
except Exception as e:
    print(json.dumps({"error": str(e), "label": label, "mode": mode, "run": run_num}))
    sys.exit(0)

elapsed = time.time() - start

if stream:
    # Parse SSE stream for usage in final chunk
    prompt_time = None
    cached_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    for line in body.split("\n"):
        line = line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            d = json.loads(line[6:])
            usage = d.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                ptd = usage.get("prompt_tokens_details")
                if ptd:
                    cached_tokens = ptd.get("cached_tokens", 0)
                prompt_time = usage.get("prompt_time")
        except:
            pass
    result = {
        "label": label,
        "mode": mode,
        "run": run_num,
        "stream": True,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "prompt_time_s": prompt_time,
        "wall_time_s": round(elapsed, 4),
        "decode_tps": round(completion_tokens / elapsed, 2) if elapsed > 0 else 0
    }
else:
    d = json.loads(body)
    usage = d.get("usage", {})
    ptd = usage.get("prompt_tokens_details", {})
    result = {
        "label": label,
        "mode": mode,
        "run": run_num,
        "stream": False,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": ptd.get("cached_tokens", 0) if ptd else 0,
        "prompt_time_s": usage.get("prompt_time"),
        "wall_time_s": round(elapsed, 4),
        "decode_tps": round(usage.get("completion_tokens_per_second", 0), 2)
    }

print(json.dumps(result))
PYEOF
}

# ── Main ──────────────────────────────────────────────────────────────────────

MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Prefix Cache Benchmark                                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:       $MODEL"
echo "║  Max tokens:  $MAX_TOKENS"
echo "║  Runs/prompt: $RUNS"
echo "║  Prompts:     ${#USER_MESSAGES[@]} (shared system + tools prefix)"
echo "║  Report:      $REPORT_DIR"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: WITHOUT prefix caching
# ══════════════════════════════════════════════════════════════════════════════

echo "━━━ Phase 1: WITHOUT --enable-prefix-caching ━━━"
kill_port $PORT

MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" "$BIN" mlx -m "$MODEL" -p "$PORT" \
  $EXTRA_FLAGS \
  > "$REPORT_DIR/afm-nocache.log" 2>&1 &
AFM_PID=$!

if ! wait_for_server $PORT 120; then
  echo "FATAL: afm server failed to start (no cache)"
  kill -9 $AFM_PID 2>/dev/null || true
  exit 1
fi
echo "  Server ready (PID $AFM_PID)"

# Scenario A: Sequential requests with SAME prompt (repeated identical)
echo ""
echo "  ── Scenario A: Identical repeated requests ──"
for ((r=1; r<=RUNS; r++)); do
  result=$(send_request $PORT "${USER_MESSAGES[0]}" "identical" "nocache" "$r" "false")
  pt=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_time_s','?'))")
  ct=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_tokens',0))")
  echo "     Run $r: prompt_time=${pt}s  cached_tokens=${ct}"
  echo "$result" >> "$RESULTS_FILE"
done

# Scenario B: Different user messages (shared prefix)
echo ""
echo "  ── Scenario B: Different user messages (shared prefix) ──"
for pi in "${!USER_MESSAGES[@]}"; do
  result=$(send_request $PORT "${USER_MESSAGES[$pi]}" "${USER_LABELS[$pi]}" "nocache" "1" "false")
  pt=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_time_s','?'))")
  ct=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_tokens',0))")
  echo "     ${USER_LABELS[$pi]}: prompt_time=${pt}s  cached_tokens=${ct}"
  echo "$result" >> "$RESULTS_FILE"
done

# Scenario C: Streaming identical repeated
echo ""
echo "  ── Scenario C: Streaming identical repeated ──"
for ((r=1; r<=RUNS; r++)); do
  result=$(send_request $PORT "${USER_MESSAGES[0]}" "stream-identical" "nocache" "$r" "true")
  pt=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_time_s','?'))")
  ct=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_tokens',0))")
  echo "     Run $r: prompt_time=${pt}s  cached_tokens=${ct}"
  echo "$result" >> "$RESULTS_FILE"
done

# Scenario D: Multi-turn conversation (growing prefix)
echo ""
echo "  ── Scenario D: Multi-turn conversation (5 turns) ──"

# Build multi-turn messages files — each turn adds user+assistant to history
python3 << MTEOF
import json, os

system_prompt = json.loads($(python3 -c "import json,sys; print(json.dumps(json.dumps(sys.argv[1])))" "$SYSTEM_PROMPT"))
report_dir = "${REPORT_DIR}"

turns = [
    {"user": "Read the file main.py and tell me what it does.", "assistant": "I'll read the file for you.\n\nThe file main.py is the entry point of the file organizer application. It imports argparse for CLI parsing and the organizer module. It defines two subcommands: 'organize' which takes a directory path and an optional --dry-run flag, and 'stats' which shows file type statistics. The main() function parses arguments and dispatches to the appropriate handler."},
    {"user": "Now read organizer.py and explain how it categorizes files.", "assistant": "Let me read that file.\n\nThe organizer.py module contains the core logic. It defines a FILE_CATEGORIES dict mapping extensions to folder names: .jpg/.png/.gif -> Images, .pdf/.docx/.txt -> Documents, .mp4/.mov/.avi -> Videos, .zip/.tar/.gz -> Archives. The organize() function iterates over files in the target directory, looks up each extension in the mapping, creates the target folder if needed, and moves the file. Unknown extensions go to an 'Other' folder."},
    {"user": "There's a bug - it crashes when a file has no extension. Can you fix it?", "assistant": "I see the issue. In organizer.py line 23, it calls `path.suffix[1:]` to get the extension without the dot, but when a file has no extension, `path.suffix` returns an empty string, and `[1:]` on an empty string returns empty string which isn't in FILE_CATEGORIES. But the crash is actually on line 28 where it tries `ext.lower()` — when ext is from a file like 'Makefile' with no dot, path.suffix returns '' and the code doesn't handle that. Let me fix it by adding a guard."},
    {"user": "Good. Now add a --verbose flag that prints each file move operation.", "assistant": "I'll add the verbose flag. In main.py, I'll add a --verbose/-v argument to the organize subparser. In organizer.py, I'll pass verbose through and add print statements showing 'Moving {filename} -> {category}/{filename}' for each file processed. When --dry-run is active, it will show 'Would move {filename} -> {category}/{filename}' instead."},
    {"user": "Run the tests to make sure nothing is broken after these changes.", "assistant": "I'll run the test suite now to verify everything still works."},
]

messages = [{"role": "system", "content": system_prompt}]

for i, turn in enumerate(turns):
    messages.append({"role": "user", "content": turn["user"]})
    # Save messages BEFORE adding assistant response (this is what we send)
    msg_file = os.path.join(report_dir, f"multiturn-{i+1}.json")
    with open(msg_file, "w") as f:
        json.dump(messages, f)
    # Add assistant response for next turn's context
    messages.append({"role": "assistant", "content": turn["assistant"]})

print("OK")
MTEOF

for turn in 1 2 3 4 5; do
  msg_file="$REPORT_DIR/multiturn-${turn}.json"
  result=$(send_request $PORT "" "turn-${turn}" "nocache" "$turn" "false" "$msg_file")
  pt=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_time_s','?'))")
  ct=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_tokens',0))")
  ptok=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_tokens',0))")
  echo "     Turn $turn: prompt_tokens=${ptok}  prompt_time=${pt}s  cached_tokens=${ct}"
  echo "$result" >> "$RESULTS_FILE"
done

kill $AFM_PID 2>/dev/null || true
wait $AFM_PID 2>/dev/null || true
kill_port $PORT
echo ""
echo "  Server stopped."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: WITH prefix caching
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "━━━ Phase 2: WITH --enable-prefix-caching ━━━"
kill_port $PORT

MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" "$BIN" mlx -m "$MODEL" -p "$PORT" \
  --enable-prefix-caching \
  $EXTRA_FLAGS \
  > "$REPORT_DIR/afm-cache.log" 2>&1 &
AFM_PID=$!

if ! wait_for_server $PORT 120; then
  echo "FATAL: afm server failed to start (with cache)"
  kill -9 $AFM_PID 2>/dev/null || true
  exit 1
fi
echo "  Server ready (PID $AFM_PID)"

# Scenario A: Identical repeated
echo ""
echo "  ── Scenario A: Identical repeated requests ──"
for ((r=1; r<=RUNS; r++)); do
  result=$(send_request $PORT "${USER_MESSAGES[0]}" "identical" "cache" "$r" "false")
  pt=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_time_s','?'))")
  ct=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_tokens',0))")
  echo "     Run $r: prompt_time=${pt}s  cached_tokens=${ct}"
  echo "$result" >> "$RESULTS_FILE"
done

# Scenario B: Different user messages (shared prefix)
echo ""
echo "  ── Scenario B: Different user messages (shared prefix) ──"
for pi in "${!USER_MESSAGES[@]}"; do
  result=$(send_request $PORT "${USER_MESSAGES[$pi]}" "${USER_LABELS[$pi]}" "cache" "1" "false")
  pt=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_time_s','?'))")
  ct=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_tokens',0))")
  echo "     ${USER_LABELS[$pi]}: prompt_time=${pt}s  cached_tokens=${ct}"
  echo "$result" >> "$RESULTS_FILE"
done

# Scenario C: Streaming identical repeated
echo ""
echo "  ── Scenario C: Streaming identical repeated ──"
for ((r=1; r<=RUNS; r++)); do
  result=$(send_request $PORT "${USER_MESSAGES[0]}" "stream-identical" "cache" "$r" "true")
  pt=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_time_s','?'))")
  ct=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_tokens',0))")
  echo "     Run $r: prompt_time=${pt}s  cached_tokens=${ct}"
  echo "$result" >> "$RESULTS_FILE"
done

# Scenario D: Multi-turn conversation (growing prefix) — reuses msg files from Phase 1
echo ""
echo "  ── Scenario D: Multi-turn conversation (5 turns) ──"
for turn in 1 2 3 4 5; do
  msg_file="$REPORT_DIR/multiturn-${turn}.json"
  result=$(send_request $PORT "" "turn-${turn}" "cache" "$turn" "false" "$msg_file")
  pt=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_time_s','?'))")
  ct=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_tokens',0))")
  ptok=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('prompt_tokens',0))")
  echo "     Turn $turn: prompt_tokens=${ptok}  prompt_time=${pt}s  cached_tokens=${ct}"
  echo "$result" >> "$RESULTS_FILE"
done

kill $AFM_PID 2>/dev/null || true
wait $AFM_PID 2>/dev/null || true
kill_port $PORT
echo ""
echo "  Server stopped."

# ══════════════════════════════════════════════════════════════════════════════
# Generate Charts
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "━━━ Generating Charts ━━━"

python3 << CHART_PYEOF
import json, sys, os
from collections import defaultdict

report_dir = "${REPORT_DIR}"
results_file = os.path.join(report_dir, "results.jsonl")

results = []
with open(results_file) as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

if not results:
    print("  No results to chart.")
    sys.exit(0)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("  matplotlib not installed — skipping charts. Install with: pip install matplotlib")
    sys.exit(0)

# ── Chart 1: TTFT (prompt_time) — Identical Repeated Requests ──────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Prefix Cache Benchmark: TTFT (prompt_time)", fontsize=14, fontweight="bold")

# 1a: Identical repeated — TTFT per run
ax = axes[0]
for mode, color, marker in [("nocache", "#e74c3c", "o"), ("cache", "#2ecc71", "s")]:
    pts = [r for r in results if r["label"] == "identical" and r["mode"] == mode and not r.get("stream")]
    if pts:
        runs = [r["run"] for r in pts]
        times = [r["prompt_time_s"] for r in pts if r["prompt_time_s"] is not None]
        if times:
            ax.plot(runs[:len(times)], times, f"-{marker}", color=color, label=mode, linewidth=2, markersize=8)

ax.set_xlabel("Run #")
ax.set_ylabel("Prompt Time (s)")
ax.set_title("Identical Repeated Requests")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# 1b: Different user messages — TTFT per prompt
ax = axes[1]
labels_order = []
nocache_times = []
cache_times = []

for r in results:
    if r.get("stream") or r["label"] in ("identical", "stream-identical"):
        continue
    if r["label"] not in labels_order:
        labels_order.append(r["label"])

for label in labels_order:
    nc = [r for r in results if r["label"] == label and r["mode"] == "nocache" and not r.get("stream")]
    ca = [r for r in results if r["label"] == label and r["mode"] == "cache" and not r.get("stream")]
    nc_t = nc[0]["prompt_time_s"] if nc and nc[0].get("prompt_time_s") is not None else 0
    ca_t = ca[0]["prompt_time_s"] if ca and ca[0].get("prompt_time_s") is not None else 0
    nocache_times.append(nc_t)
    cache_times.append(ca_t)

if labels_order:
    import numpy as np
    x = np.arange(len(labels_order))
    width = 0.35
    ax.bar(x - width/2, nocache_times, width, label="No Cache", color="#e74c3c", alpha=0.85)
    ax.bar(x + width/2, cache_times, width, label="Cache", color="#2ecc71", alpha=0.85)
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Prompt Time (s)")
    ax.set_title("Different User Messages (Shared Prefix)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_order, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
chart1 = os.path.join(report_dir, "ttft-comparison.jpg")
plt.savefig(chart1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart 1: {chart1}")

# ── Chart 2: Cached Tokens ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Prefix Cache Benchmark: Cached Tokens", fontsize=14, fontweight="bold")

# 2a: Identical repeated — cached_tokens per run
ax = axes[0]
for mode, color, marker in [("nocache", "#e74c3c", "o"), ("cache", "#2ecc71", "s")]:
    pts = [r for r in results if r["label"] == "identical" and r["mode"] == mode and not r.get("stream")]
    if pts:
        runs = [r["run"] for r in pts]
        cached = [r["cached_tokens"] for r in pts]
        ax.plot(runs, cached, f"-{marker}", color=color, label=mode, linewidth=2, markersize=8)

ax.set_xlabel("Run #")
ax.set_ylabel("Cached Tokens")
ax.set_title("Identical Repeated Requests")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# 2b: Different user messages — cached_tokens per prompt
ax = axes[1]
if labels_order:
    nc_cached = []
    ca_cached = []
    for label in labels_order:
        nc = [r for r in results if r["label"] == label and r["mode"] == "nocache" and not r.get("stream")]
        ca = [r for r in results if r["label"] == label and r["mode"] == "cache" and not r.get("stream")]
        nc_cached.append(nc[0]["cached_tokens"] if nc else 0)
        ca_cached.append(ca[0]["cached_tokens"] if ca else 0)

    x = np.arange(len(labels_order))
    ax.bar(x - width/2, nc_cached, width, label="No Cache", color="#e74c3c", alpha=0.85)
    ax.bar(x + width/2, ca_cached, width, label="Cache", color="#2ecc71", alpha=0.85)
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Cached Tokens")
    ax.set_title("Different User Messages (Shared Prefix)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_order, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
chart2 = os.path.join(report_dir, "cached-tokens.jpg")
plt.savefig(chart2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart 2: {chart2}")

# ── Chart 3: Streaming TTFT ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Prefix Cache Benchmark: Streaming TTFT", fontsize=14, fontweight="bold")

for mode, color, marker in [("nocache", "#e74c3c", "o"), ("cache", "#2ecc71", "s")]:
    pts = [r for r in results if r["label"] == "stream-identical" and r["mode"] == mode]
    if pts:
        runs = [r["run"] for r in pts]
        times = [r["prompt_time_s"] for r in pts if r["prompt_time_s"] is not None]
        if times:
            ax.plot(runs[:len(times)], times, f"-{marker}", color=color, label=mode, linewidth=2, markersize=8)

ax.set_xlabel("Run #")
ax.set_ylabel("Prompt Time (s)")
ax.set_title("Streaming Identical Repeated Requests")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
chart3 = os.path.join(report_dir, "streaming-ttft.jpg")
plt.savefig(chart3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart 3: {chart3}")

# ── Chart 4: Speedup Summary ──────────────────────────────────────────────

# Compute average prompt_time for each scenario
scenarios = ["identical", "stream-identical"] + labels_order
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Prefix Cache Speedup (prompt_time reduction)", fontsize=14, fontweight="bold")

speedups = []
scenario_names = []

for scenario in scenarios:
    nc_pts = [r for r in results if r["label"] == scenario and r["mode"] == "nocache"
              and r.get("prompt_time_s") is not None]
    ca_pts = [r for r in results if r["label"] == scenario and r["mode"] == "cache"
              and r.get("prompt_time_s") is not None]
    if nc_pts and ca_pts:
        nc_avg = sum(r["prompt_time_s"] for r in nc_pts) / len(nc_pts)
        ca_avg = sum(r["prompt_time_s"] for r in ca_pts) / len(ca_pts)
        if ca_avg > 0 and nc_avg > 0:
            speedup = nc_avg / ca_avg
            speedups.append(speedup)
            scenario_names.append(scenario)

if speedups:
    colors = ["#2ecc71" if s > 1 else "#e74c3c" for s in speedups]
    bars = ax.bar(range(len(speedups)), speedups, color=colors, alpha=0.85)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No improvement")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Speedup (x)")
    ax.set_title("Prompt Processing Speedup with Prefix Caching")
    ax.set_xticks(range(len(speedups)))
    ax.set_xticklabels(scenario_names, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f"{val:.1f}x", ha="center", va="bottom", fontweight="bold", fontsize=10)

plt.tight_layout()
chart4 = os.path.join(report_dir, "speedup-summary.jpg")
plt.savefig(chart4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart 4: {chart4}")

# ── Chart 5: Multi-turn Conversation ──────────────────────────────────────

turn_labels = [f"turn-{i}" for i in range(1, 6)]
nc_turn_pts = [r for r in results if r["label"].startswith("turn-") and r["mode"] == "nocache"]
ca_turn_pts = [r for r in results if r["label"].startswith("turn-") and r["mode"] == "cache"]

if nc_turn_pts and ca_turn_pts:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Multi-Turn Conversation: Prefix Cache Impact", fontsize=14, fontweight="bold")

    # 5a: Prompt time per turn
    ax = axes[0]
    nc_sorted = sorted(nc_turn_pts, key=lambda r: r["run"])
    ca_sorted = sorted(ca_turn_pts, key=lambda r: r["run"])

    nc_times = [r["prompt_time_s"] for r in nc_sorted if r.get("prompt_time_s") is not None]
    ca_times = [r["prompt_time_s"] for r in ca_sorted if r.get("prompt_time_s") is not None]
    nc_runs = list(range(1, len(nc_times)+1))
    ca_runs = list(range(1, len(ca_times)+1))

    if nc_times:
        ax.plot(nc_runs, nc_times, "-o", color="#e74c3c", label="No Cache", linewidth=2, markersize=8)
    if ca_times:
        ax.plot(ca_runs, ca_times, "-s", color="#2ecc71", label="Cache", linewidth=2, markersize=8)

    ax.set_xlabel("Conversation Turn")
    ax.set_ylabel("Prompt Time (s)")
    ax.set_title("TTFT per Turn (growing context)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 5b: Cached tokens + prompt tokens per turn
    ax = axes[1]
    ca_cached = [r["cached_tokens"] for r in ca_sorted]
    ca_prompt = [r["prompt_tokens"] for r in ca_sorted]
    nc_prompt = [r["prompt_tokens"] for r in nc_sorted]
    turns_x = list(range(1, len(ca_sorted)+1))

    if ca_prompt:
        ax.bar(turns_x, ca_prompt, color="#3498db", alpha=0.4, label="Total Prompt Tokens")
        ax.bar(turns_x, ca_cached, color="#2ecc71", alpha=0.85, label="Cached Tokens (reused)")

    ax.set_xlabel("Conversation Turn")
    ax.set_ylabel("Tokens")
    ax.set_title("Prefix Reuse per Turn")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    chart5 = os.path.join(report_dir, "multiturn-comparison.jpg")
    plt.savefig(chart5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart 5: {chart5}")

# ── Summary Table ──────────────────────────────────────────────────────────

# Include multi-turn in scenarios
all_scenarios = scenarios + turn_labels
print()
print("  ┌─────────────────────────┬──────────────┬──────────────┬──────────┐")
print("  │ Scenario                │ No Cache (s) │ Cache (s)    │ Speedup  │")
print("  ├─────────────────────────┼──────────────┼──────────────┼──────────┤")

for scenario in all_scenarios:
    nc_pts = [r for r in results if r["label"] == scenario and r["mode"] == "nocache"
              and r.get("prompt_time_s") is not None]
    ca_pts = [r for r in results if r["label"] == scenario and r["mode"] == "cache"
              and r.get("prompt_time_s") is not None]
    if nc_pts and ca_pts:
        nc_avg = sum(r["prompt_time_s"] for r in nc_pts) / len(nc_pts)
        ca_avg = sum(r["prompt_time_s"] for r in ca_pts) / len(ca_pts)
        speedup = nc_avg / ca_avg if ca_avg > 0 else 0
        print(f"  │ {scenario:<23} │ {nc_avg:>10.4f}   │ {ca_avg:>10.4f}   │ {speedup:>6.1f}x  │")

print("  └─────────────────────────┴──────────────┴──────────────┴──────────┘")
print()
print(f"  Raw data:  {results_file}")
print(f"  Charts:    {report_dir}/")

CHART_PYEOF

echo ""
echo "━━━ Done ━━━"
echo "  Results: $RESULTS_FILE"
echo "  Charts:  $REPORT_DIR/*.jpg"
