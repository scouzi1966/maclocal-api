#!/usr/bin/env bash
#
# AgentBlaster × afm benchmark harness.
#
# Drives the AgentBlaster local agentic benchmark suite
# (https://github.com/scouzi1966/AgentBlaster) against a running afm server and
# reports pass/fail per suite plus afm's native /metrics telemetry.
#
# What it measures: tool calling, structured/grammar output, prompt-cache reuse
# (prefill/cache-replay), fan-out concurrency, cancellation, and agent-workflow
# profiles (opencode/hermes/codex/...). afm's vLLM-namespaced /metrics are
# scraped before/after every suite so decode tok/s, TTFT, e2e latency, and radix
# prefix-cache hits land in each run directory.
#
# Prerequisites:
#   - AgentBlaster installed:  pip install -e /path/to/AgentBlaster   (provides `agentblaster`)
#   - afm installed:           Homebrew `afm` (default) or pass --bin
#   - Model in the MLX cache:  export MACAFM_MLX_MODEL_CACHE=/path/to/cache
#
# Usage:
#   export MACAFM_MLX_MODEL_CACHE=/Volumes/Crucial4TB/models/vesta-test-cache
#   ./run-benchmark.sh                                    # MoE default, generated 94-case set
#   ./run-benchmark.sh --model mlx-community/Qwen3.6-27B-4bit --mode probe
#   ./run-benchmark.sh --bin /opt/homebrew/bin/afm --mode both --concurrency 4
#
# Modes:
#   probe      13 built-in capability probes (~19 cases) — fast smoke of every capability surface
#   generated  the larger generated set in suites/ (agent profiles + harness generators, ~94 cases)
#   both       probe then generated
#
# NOTE: --no-think is passed to afm because Qwen3.6 is a reasoning model and the
# exact-output benchmark cases otherwise fail when thinking eats the token budget
# (this also exercises the v0.9.13 --no-think fix). Avoid pairing --no-think with
# very high server --concurrent on short known-answer suites — see issue #140.
#
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
MODEL="mlx-community/Qwen3.6-35B-A3B-4bit"
PORT=9999
BIN="${AFM_BIN:-/opt/homebrew/bin/afm}"        # installed (Homebrew) afm by default
MODE="generated"
CONCURRENCY=1
FANOUT_CONCURRENCY=4
OUTPUT_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITES_DIR="$SCRIPT_DIR/suites"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;            # probe | generated | both
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/agentblaster-runs/$(date -u +%Y%m%dT%H%M%SZ)}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
METRICS_URL="http://127.0.0.1:${PORT}/metrics"

command -v agentblaster >/dev/null 2>&1 || { echo "FATAL: agentblaster not installed (pip install -e /path/to/AgentBlaster)"; exit 1; }
[ -x "$BIN" ] || { echo "FATAL: afm binary not found/executable at $BIN (override with --bin)"; exit 1; }
[ -n "${MACAFM_MLX_MODEL_CACHE:-}" ] || echo "WARN: MACAFM_MLX_MODEL_CACHE is unset — afm may try to download $MODEL"

echo "=== AgentBlaster × afm benchmark ==="
echo "  afm binary : $BIN ($($BIN --version 2>&1 | head -1))"
echo "  model      : $MODEL"
echo "  mode       : $MODE"
echo "  output     : $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ─── Launch afm with agentic flags; ensure cleanup on exit ───────────────────
AFM_PID=""
cleanup() { [ -n "$AFM_PID" ] && kill "$AFM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "--- launching afm on :$PORT (--no-think, prefix-cache, grammar, adaptive-xml, concurrent 4) ---"
"$BIN" mlx -m "$MODEL" --port "$PORT" --no-think \
  --enable-prefix-caching --enable-grammar-constraints \
  --tool-call-parser afm_adaptive_xml --concurrent 4 \
  > "$OUTPUT_DIR/afm-server.log" 2>&1 &
AFM_PID=$!

for i in $(seq 1 80); do
  curl -s -m 3 "$BASE_URL/models" >/dev/null 2>&1 && break
  sleep 3
  [ "$i" = 80 ] && { echo "FATAL: afm did not become ready"; tail -5 "$OUTPUT_DIR/afm-server.log"; exit 1; }
done
echo "    afm ready: $(curl -s "$BASE_URL/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"

# ─── Register the provider (idempotent) ──────────────────────────────────────
agentblaster providers add --name afm --contract openai \
  --base-url "$BASE_URL" --default-model "$MODEL" --metrics-url "$METRICS_URL" >/dev/null 2>&1 || true

# ─── Helpers ─────────────────────────────────────────────────────────────────
PASS_TOTAL=0; CASE_TOTAL=0
run_one() {  # $1=label  $2=(--suite NAME | --suite-file PATH)  $3=concurrency
  local label="$1"; shift
  local conc="${!#}"; set -- "${@:1:$(($#-1))}"
  local out p t
  out=$(agentblaster run --engine afm --model "$MODEL" "$@" \
        --concurrency "$conc" --output-dir "$OUTPUT_DIR" 2>&1) || true
  p=$(echo "$out" | grep -oE 'passed: [0-9]+' | grep -oE '[0-9]+' | head -1); p=${p:-0}
  t=$(echo "$out" | grep -oE 'total_cases: [0-9]+' | grep -oE '[0-9]+' | head -1); t=${t:-0}
  PASS_TOTAL=$((PASS_TOTAL+p)); CASE_TOTAL=$((CASE_TOTAL+t))
  printf "  %-30s conc=%s  %s/%s\n" "$label" "$conc" "$p" "$t"
}

# ─── Built-in capability probes ──────────────────────────────────────────────
run_probes() {
  echo "--- built-in capability probes ---"
  for s in smoke toolcall tool-parser-repair structured prefill cache-control \
           cancellation agentic-tool-loop; do
    run_one "$s" --suite "$s" "$CONCURRENCY"
  done
  run_one "agent-fanout" --suite agent-fanout "$FANOUT_CONCURRENCY"
}

# ─── Generated suites (agent profiles + harness generators) ──────────────────
run_generated() {
  if ! ls "$SUITES_DIR"/*.yaml >/dev/null 2>&1; then
    echo "--- no generated suites in $SUITES_DIR — run ./generate-suites.sh first ---"
    return
  fi
  echo "--- generated suites ($SUITES_DIR) ---"
  for f in "$SUITES_DIR"/*.yaml; do
    local conc="$CONCURRENCY"
    [[ "$(basename "$f")" == *concurrency* ]] && conc="$FANOUT_CONCURRENCY"
    run_one "$(basename "$f" .yaml)" --suite-file "$f" "$conc"
  done
}

case "$MODE" in
  probe)     run_probes ;;
  generated) run_generated ;;
  both)      run_probes; run_generated ;;
  *) echo "FATAL: unknown --mode $MODE (probe|generated|both)"; exit 2 ;;
esac

# ─── afm native performance summary (from /metrics) ──────────────────────────
echo ""
echo "=== afm native /metrics (cumulative over this run) ==="
curl -s -m 5 "$METRICS_URL" | python3 - "$MODEL" <<'PY'
import sys, re
model = sys.argv[1]
vals = {}
for line in sys.stdin.read().splitlines():
    if line.startswith('#') or '{' not in line: continue
    name = line.split('{')[0]
    try: vals.setdefault(name, 0.0); vals[name] += float(line.rsplit(' ', 1)[1])
    except Exception: pass
g = lambda n: vals.get(n, 0.0)
gen = g('afm:generation_tokens_total'); dec = g('afm:request_decode_time_seconds_sum')
ttft_s = g('afm:time_to_first_token_seconds_sum'); ttft_n = g('afm:time_to_first_token_seconds_count')
e2e_s = g('afm:e2e_request_latency_seconds_sum'); e2e_n = g('afm:e2e_request_latency_seconds_count')
print(f"  requests        : {int(e2e_n)}")
print(f"  decode tok/s    : {gen/dec:.1f}" if dec else "  decode tok/s    : n/a")
print(f"  avg TTFT (s)    : {ttft_s/ttft_n:.2f}" if ttft_n else "  avg TTFT (s)    : n/a")
print(f"  avg e2e (s)     : {e2e_s/e2e_n:.2f}" if e2e_n else "  avg e2e (s)     : n/a")
print(f"  radix cache hits: {int(g('afm:radix_cache_hits_total'))}")
PY

echo ""
echo "=== TOTAL: ${PASS_TOTAL}/${CASE_TOTAL} cases passed ==="
echo "Per-run artifacts (results.jsonl, raw/, metrics/prometheus-summary.json) under: $OUTPUT_DIR"
echo "HTML report for any run:  agentblaster report $OUTPUT_DIR/<run_id> --format html,json"
echo ""
echo "NOTE: AgentBlaster classifies each failure (model_quality | engine bug | feature gap |"
echo "runtime). Treat 'model_quality' as the model's tool/structured decision, not an afm bug —"
echo "verify the failure_class in results.jsonl before attributing anything to the engine."
