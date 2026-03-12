#!/opt/homebrew/bin/bash
#
# Benchmark: afm mlx vs mlx_lm.server
# Model: Qwen3.5-35B-A3B-4bit (or specified via --model)
#
# Runs identical prompts against both servers and compares tok/s.
# Both servers use the OpenAI-compatible /v1/chat/completions endpoint.
#

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="${MODEL:-mlx-community/Qwen3.5-35B-A3B-4bit}"
AFM_PORT=9990
MLX_LM_PORT=9991
CACHE_DIR="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
RUNS=3          # number of runs per prompt per server
MAX_TOKENS=512
TEMPERATURE=0.0
WARMUP=1        # warmup requests (discarded)

AFM_BIN="${AFM_BIN:-.build/arm64-apple-macosx/release/afm}"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --runs)    RUNS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --warmup)  WARMUP="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model MODEL] [--runs N] [--max-tokens N] [--temperature F] [--warmup N]"
      echo ""
      echo "Benchmarks afm mlx vs mlx_lm.server on identical prompts."
      echo "Set AFM_BIN to point to the afm binary (default: .build/arm64-apple-macosx/release/afm)"
      echo "Set MACAFM_MLX_MODEL_CACHE for model cache directory."
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Prompts ───────────────────────────────────────────────────────────────────
# Mix of short and long generation tasks
PROMPTS=(
  "Explain calculus concepts from limits through multivariable calculus with rigorous mathematical notation."
)

PROMPT_LABELS=(
  "calculus-deep"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

kill_port() {
  local port=$1
  local pid
  pid=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    kill -KILL $pid 2>/dev/null || true
    sleep 1
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
send_request() {
  local port=$1
  local prompt="$2"
  local label="$3"

  python3 << PYEOF
import json, time
from openai import OpenAI

client = OpenAI(base_url=f"http://127.0.0.1:${port}/v1", api_key="x", timeout=300)
prompt = json.loads('$(echo "$prompt" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))")')

start = time.time()
resp = client.chat.completions.create(
    model="${MODEL}",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=${MAX_TOKENS},
    temperature=${TEMPERATURE},
    stream=False,
)
elapsed = time.time() - start

choice = resp.choices[0] if resp.choices else None
usage = resp.usage
prompt_tokens = usage.prompt_tokens if usage else 0
completion_tokens = usage.completion_tokens if usage else 0
tps = completion_tokens / elapsed if elapsed > 0 else 0

result = {
    "label": "${label}",
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "elapsed_s": round(elapsed, 3),
    "tokens_per_sec": round(tps, 2),
    "finish_reason": choice.finish_reason if choice else None,
}
print(json.dumps(result))
PYEOF
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Benchmark: afm mlx vs mlx_lm.server                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:       $MODEL"
echo "║  Max tokens:  $MAX_TOKENS"
echo "║  Temperature: $TEMPERATURE"
echo "║  Runs/prompt: $RUNS (+ $WARMUP warmup)"
echo "║  Prompts:     ${#PROMPTS[@]}"
echo "║  AFM binary:  $AFM_BIN"
echo "║  mlx-lm:      $(python3 -c 'import mlx_lm; print(mlx_lm.__version__)')"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

RESULTS_FILE="/tmp/benchmark-results.jsonl"
> "$RESULTS_FILE"

# ── Start afm server ─────────────────────────────────────────────────────────
echo "=== Starting afm mlx server on port $AFM_PORT ==="
kill_port $AFM_PORT
MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" "$AFM_BIN" mlx -m "$MODEL" -p "$AFM_PORT" > /tmp/bench-afm.log 2>&1 &
AFM_PID=$!

if ! wait_for_server $AFM_PORT 120; then
  echo "FATAL: afm server failed to start"
  kill -KILL $AFM_PID 2>/dev/null || true
  exit 1
fi
echo "  afm ready (PID $AFM_PID)"

# ── Benchmark afm ────────────────────────────────────────────────────────────
echo ""
echo "=== Benchmarking afm mlx ==="

for pi in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$pi]}"
  label="${PROMPT_LABELS[$pi]}"

  # Warmup
  for ((w=0; w<WARMUP; w++)); do
    send_request $AFM_PORT "$prompt" "$label" > /dev/null 2>&1
  done

  # Timed runs
  for ((r=1; r<=RUNS; r++)); do
    result=$(send_request $AFM_PORT "$prompt" "$label")
    tps=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['tokens_per_sec'])")
    ctok=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['completion_tokens'])")
    echo "  [$label] run $r: ${ctok} tok, ${tps} tok/s"
    echo "$result" | python3 -c "import json,sys; r=json.load(sys.stdin); r['server']='afm'; r['run']=$r; print(json.dumps(r))" >> "$RESULTS_FILE"
  done
done

# Kill afm
kill -TERM $AFM_PID 2>/dev/null || true
sleep 2
kill -KILL $AFM_PID 2>/dev/null || true
kill_port $AFM_PORT
echo "  afm server stopped"

# ── Start mlx_lm server ──────────────────────────────────────────────────────
echo ""
echo "=== Starting mlx_lm.server on port $MLX_LM_PORT ==="
kill_port $MLX_LM_PORT

# Use HF model ID; patch strict=False for models with extra weights (e.g. vision tower)
python3 -c "
import mlx.nn as nn
_orig = nn.Module.load_weights
def _patched(self, *a, strict=True, **kw):
    return _orig(self, *a, strict=False, **kw)
nn.Module.load_weights = _patched
from mlx_lm.server import main
import sys
sys.argv = ['mlx_lm.server', '--model', '$MODEL', '--port', '$MLX_LM_PORT', '--temp', '$TEMPERATURE']
main()
" > /tmp/bench-mlx-lm.log 2>&1 &
MLX_LM_PID=$!

if ! wait_for_server $MLX_LM_PORT 120; then
  echo "FATAL: mlx_lm.server failed to start"
  cat /tmp/bench-mlx-lm.log | tail -10
  kill -KILL $MLX_LM_PID 2>/dev/null || true
  exit 1
fi
echo "  mlx_lm ready (PID $MLX_LM_PID)"

# ── Benchmark mlx_lm ─────────────────────────────────────────────────────────
echo ""
echo "=== Benchmarking mlx_lm.server ==="

for pi in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$pi]}"
  label="${PROMPT_LABELS[$pi]}"

  # Warmup
  for ((w=0; w<WARMUP; w++)); do
    send_request $MLX_LM_PORT "$prompt" "$label" > /dev/null 2>&1
  done

  # Timed runs
  for ((r=1; r<=RUNS; r++)); do
    result=$(send_request $MLX_LM_PORT "$prompt" "$label")
    tps=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['tokens_per_sec'])")
    ctok=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['completion_tokens'])")
    echo "  [$label] run $r: ${ctok} tok, ${tps} tok/s"
    echo "$result" | python3 -c "import json,sys; r=json.load(sys.stdin); r['server']='mlx_lm'; r['run']=$r; print(json.dumps(r))" >> "$RESULTS_FILE"
  done
done

# Kill mlx_lm
kill -TERM $MLX_LM_PID 2>/dev/null || true
sleep 2
kill -KILL $MLX_LM_PID 2>/dev/null || true
kill_port $MLX_LM_PORT
echo "  mlx_lm server stopped"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      RESULTS SUMMARY                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python3 << 'SUMMARY_PYEOF'
import json

results = []
with open("/tmp/benchmark-results.jsonl") as f:
    for line in f:
        results.append(json.loads(line.strip()))

# Group by label + server
from collections import defaultdict
grouped = defaultdict(list)
for r in results:
    key = (r["label"], r["server"])
    grouped[key].append(r)

labels = sorted(set(r["label"] for r in results))

print(f"{'Prompt':<25} {'afm tok/s':>12} {'mlx_lm tok/s':>14} {'Diff':>10} {'Winner':>8}")
print("-" * 75)

afm_total_tps = []
mlx_total_tps = []

for label in labels:
    afm_runs = grouped.get((label, "afm"), [])
    mlx_runs = grouped.get((label, "mlx_lm"), [])

    afm_avg = sum(r["tokens_per_sec"] for r in afm_runs) / len(afm_runs) if afm_runs else 0
    mlx_avg = sum(r["tokens_per_sec"] for r in mlx_runs) / len(mlx_runs) if mlx_runs else 0

    afm_total_tps.append(afm_avg)
    mlx_total_tps.append(mlx_avg)

    if afm_avg > 0 and mlx_avg > 0:
        diff_pct = ((afm_avg - mlx_avg) / mlx_avg) * 100
        winner = "afm" if afm_avg > mlx_avg else "mlx_lm"
        diff_str = f"{diff_pct:+.1f}%"
    else:
        diff_str = "N/A"
        winner = "?"

    print(f"{label:<25} {afm_avg:>10.1f}  {mlx_avg:>12.1f}  {diff_str:>10} {winner:>8}")

# Overall
afm_overall = sum(afm_total_tps) / len(afm_total_tps) if afm_total_tps else 0
mlx_overall = sum(mlx_total_tps) / len(mlx_total_tps) if mlx_total_tps else 0
if afm_overall > 0 and mlx_overall > 0:
    overall_diff = ((afm_overall - mlx_overall) / mlx_overall) * 100
    overall_winner = "afm" if afm_overall > mlx_overall else "mlx_lm"
else:
    overall_diff = 0
    overall_winner = "?"

print("-" * 75)
print(f"{'OVERALL AVERAGE':<25} {afm_overall:>10.1f}  {mlx_overall:>12.1f}  {overall_diff:+.1f}% {overall_winner:>8}")
print()
print(f"Raw data: /tmp/benchmark-results.jsonl")
SUMMARY_PYEOF
