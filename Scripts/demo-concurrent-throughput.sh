#!/usr/bin/env bash
# Concurrent-throughput demo runner for AFM.
#
# Starts an afm MLX server with the requested model + concurrency + prefix
# cache, drives it with the concurrent_load_driver.py virtual-user ramp,
# records a per-250ms metrics trace, and renders a marketing-grade MP4
# dual-Y-axis chart (concurrent connections + aggregate tok/s over time).
#
# Usage:
#   ./Scripts/demo-concurrent-throughput.sh                        # defaults
#   ./Scripts/demo-concurrent-throughput.sh \
#       --model mlx-community/Qwen3.5-35B-A3B-4bit \
#       --concurrent 200 \
#       --ramp 45 --hold 75 \
#       --mode general
#
# Options:
#   --model MODEL              MLX model ID or local path (default: Qwen3.5-35B-A3B-4bit)
#   AFM_BIN=/path/to/afm      Env var to override the binary (default: local release build, then PATH)
#   --concurrent N             Peak virtual users AND server --concurrent slots (default: 200)
#   --ramp S                   Seconds to ramp 0 -> N (default: 45)
#   --ramp-jitter-pct PCT      Randomize each step's sleep by ±PCT percent
#                              (step mode only; default: 0 = periodic).
#                              Use to break the ramp's fixed cadence when
#                              diagnosing periodic throughput dips.
#   --ramp-jitter-seed N       RNG seed for jitter (default: -1 = random).
#   --hold S                   Seconds to hold at N after ramp (default: 75)
#   --mode general|prefix-cache|mixed  Prompt profile (default: general)
#   --max-tokens N             Per-request max_tokens (default: 192)
#   --temperature F            Sampling temperature (default: 0.7)
#   --top-p F                  Top-p (default: 1.0; set <1 to exercise TopP path)
#   --port N                   Server port (default: 9998 — avoids colliding with 9999)
#   --title STR                Video title line 1
#   --subtitle STR             Video subtitle (otherwise built from model + config)
#   --output PATH              Output mp4 (default: Scripts/demo/out/concurrent_demo.mp4)
#   --no-prefix-cache          Disable server --enable-prefix-caching
#   --skip-render              Run the load driver only; skip video render
#   --skip-load                Skip the load run; re-render an existing trace
#   --keep-server              Leave the server running after the demo (for inspection)
#   --watch                    Open a live matplotlib window (new Terminal) that
#                              tails trace.jsonl and redraws every 250ms during load
#   --cache PATH               Model cache dir (default: $MACAFM_MLX_MODEL_CACHE or
#                              /Volumes/edata/models/vesta-test-cache)
#   -h, --help                 This help

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"
CONCURRENT=200
RAMP_MODE="step"
RAMP_S=45                   # linear mode only
RAMP_STEP_USERS=2
RAMP_STEP_S=5
RAMP_DOWN_STEP_USERS=-1     # -1 = match RAMP_STEP_USERS
RAMP_DOWN_STEP_S=-1         # -1 = match RAMP_STEP_S
RAMP_JITTER_PCT=0           # step mode: randomize each step's sleep by ±pct
RAMP_JITTER_SEED=-1         # step mode: RNG seed (-1 = random)
HOLD_S=25
COOLDOWN_S=30              # zero-activity tail after ramp-down; early-exit on sustained idle
SMOOTHING_WINDOW=20        # moving-average window (samples) for the tok/sec line on the chart
MODE="general"
MAX_TOKENS=3000
MAX_TOKENS_JITTER=0.1
REQUEST_TIMEOUT=900       # per-request aiohttp client timeout; must exceed worst-case decode
TEMPERATURE=0.7
TOP_P=1.0
PORT=9998
TITLE="AFM: Concurrent Inference at Scale"
SUBTITLE=""
OUTPUT="$ROOT_DIR/Scripts/demo/out/concurrent_demo.mp4"
TRACE_PATH="$ROOT_DIR/Scripts/demo/out/trace.jsonl"
SUMMARY_PATH="$ROOT_DIR/Scripts/demo/out/trace.summary.json"
REQUESTS_PATH="$ROOT_DIR/Scripts/demo/out/requests.jsonl"
REQUESTS_HTML="$ROOT_DIR/Scripts/demo/out/requests.html"
MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
ENABLE_PREFIX_CACHE=true
SKIP_RENDER=false
SKIP_LOAD=false
SKIP_HTML=false
SKIP_VERIFY=false
KEEP_SERVER=false
WATCH_LIVE=false
INITIAL_TPS_MAX=1300
SMOOTHING_WINDOW_S=2.0
WARMUP_USERS=0          # 0 = disabled (default for step mode)
WARMUP_MAX_TOKENS=32    # small so warmup requests complete naturally; no zombies

# AFM binary — env var overrides; then local release build; then PATH
if [ -n "${AFM_BIN:-}" ] && [ -x "$AFM_BIN" ]; then
  : # use env var as-is
else
  AFM_BIN="$ROOT_DIR/.build/arm64-apple-macosx/release/afm"
  if [ ! -x "$AFM_BIN" ]; then
    AFM_BIN="$(command -v afm || true)"
  fi
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[demo]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[err]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --model)           MODEL="$2"; shift 2 ;;
    --concurrent)      CONCURRENT="$2"; shift 2 ;;
    --ramp-mode)       RAMP_MODE="$2"; shift 2 ;;
    --ramp)            RAMP_S="$2"; shift 2 ;;
    --ramp-step-users) RAMP_STEP_USERS="$2"; shift 2 ;;
    --ramp-step-s)     RAMP_STEP_S="$2"; shift 2 ;;
    --ramp-jitter-pct) RAMP_JITTER_PCT="$2"; shift 2 ;;
    --ramp-jitter-seed) RAMP_JITTER_SEED="$2"; shift 2 ;;
    --ramp-down-step-users) RAMP_DOWN_STEP_USERS="$2"; shift 2 ;;
    --ramp-down-step-s)     RAMP_DOWN_STEP_S="$2"; shift 2 ;;
    --hold)            HOLD_S="$2"; shift 2 ;;
    --cooldown)        COOLDOWN_S="$2"; shift 2 ;;
    --smoothing-window) SMOOTHING_WINDOW="$2"; shift 2 ;;
    --mode)            MODE="$2"; shift 2 ;;
    --max-tokens)      MAX_TOKENS="$2"; shift 2 ;;
    --max-tokens-jitter) MAX_TOKENS_JITTER="$2"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
    --temperature)     TEMPERATURE="$2"; shift 2 ;;
    --top-p)           TOP_P="$2"; shift 2 ;;
    --initial-tps-max) INITIAL_TPS_MAX="$2"; shift 2 ;;
    --smoothing-window-s) SMOOTHING_WINDOW_S="$2"; shift 2 ;;
    --warmup-users)    WARMUP_USERS="$2"; shift 2 ;;
    --warmup-max-tokens) WARMUP_MAX_TOKENS="$2"; shift 2 ;;
    --no-warmup)       WARMUP_USERS=0; shift ;;
    --port)            PORT="$2"; shift 2 ;;
    --title)           TITLE="$2"; shift 2 ;;
    --subtitle)        SUBTITLE="$2"; shift 2 ;;
    --output)          OUTPUT="$2"; shift 2 ;;
    --cache)           MODEL_CACHE="$2"; shift 2 ;;
    --no-prefix-cache) ENABLE_PREFIX_CACHE=false; shift ;;
    --skip-render)     SKIP_RENDER=true; shift ;;
    --skip-load)       SKIP_LOAD=true; shift ;;
    --skip-html)       SKIP_HTML=true; shift ;;
    --skip-verify)     SKIP_VERIFY=true; shift ;;
    --keep-server)     KEEP_SERVER=true; shift ;;
    --watch)           WATCH_LIVE=true; shift ;;
    -h|--help)
      sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) err "unknown flag: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ -z "${AFM_BIN:-}" ] || [ ! -x "$AFM_BIN" ]; then
  err "afm binary not found. Build with 'swift build -c release' or install via brew."
  exit 1
fi

if [ ! -d "$MODEL_CACHE" ]; then
  warn "model cache dir does not exist: $MODEL_CACHE (will download)"
fi

if ! command -v python3 >/dev/null 2>&1; then
  err "python3 not found"; exit 1
fi

mkdir -p "$(dirname "$TRACE_PATH")"
mkdir -p "$(dirname "$OUTPUT")"

SERVER_LOG="/tmp/afm-demo-server-$(date +%Y%m%d_%H%M%S)-$$.log"

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && [ "$KEEP_SERVER" = false ]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      log "stopping server (pid $SERVER_PID)"
      kill "$SERVER_PID" 2>/dev/null || true
      # Give it a second to shut down gracefully
      for _ in 1 2 3 4 5; do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
      done
      kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
  elif [ -n "$SERVER_PID" ] && [ "$KEEP_SERVER" = true ]; then
    log "server left running at http://127.0.0.1:${PORT} (pid $SERVER_PID)"
    log "stop it with: kill $SERVER_PID"
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  log "starting afm server"
  log "  binary   : $AFM_BIN"
  log "  model    : $MODEL"
  log "  port     : $PORT"
  log "  concurrent: $CONCURRENT"
  log "  prefix   : $ENABLE_PREFIX_CACHE"
  log "  log      : $SERVER_LOG"

  local args=(mlx -m "$MODEL" --concurrent "$CONCURRENT" --port "$PORT")
  if [ "$ENABLE_PREFIX_CACHE" = true ]; then
    args+=(--enable-prefix-caching)
  fi

  # Pass AFM_DEBUG through from the wrapper's environment so callers can do
  #   AFM_DEBUG=1 ./Scripts/demo-concurrent-throughput.sh
  # and get the BatchScheduler's per-step timing breakdown in the server
  # log. Default off (empty).
  MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" \
  AFM_DEBUG="${AFM_DEBUG:-}" \
    "$AFM_BIN" "${args[@]}" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  log "server pid: $SERVER_PID"
  if [ -n "${AFM_DEBUG:-}" ]; then
    log "  AFM_DEBUG=${AFM_DEBUG} — per-step BatchScheduler timing enabled"
  fi

  log "waiting for model to load..."
  local ready=false
  for i in $(seq 1 360); do
    if curl -s "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q '"loaded"'; then
      ready=true
      log "model loaded after ${i}s"
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      err "server exited before model loaded. tail of log:"
      tail -30 "$SERVER_LOG" >&2
      exit 1
    fi
    sleep 1
  done
  if [ "$ready" = false ]; then
    err "model did not become ready within 360s"
    tail -30 "$SERVER_LOG" >&2
    exit 1
  fi

  # Tiny warmup request so the first real request doesn't bear kernel compile
  log "warmup..."
  curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3,\"temperature\":0}" \
    > /dev/null || true
  log "warmup complete"
}

# ---------------------------------------------------------------------------
# Load run
# ---------------------------------------------------------------------------
run_load() {
  log "running load driver"
  log "  target users : $CONCURRENT"
  log "  ramp         : ${RAMP_S}s"
  log "  hold         : ${HOLD_S}s"
  log "  mode         : $MODE"

  python3 "$ROOT_DIR/Scripts/demo/concurrent_load_driver.py" \
    --endpoint "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --model "$MODEL" \
    --target-users "$CONCURRENT" \
    --ramp-mode "$RAMP_MODE" \
    --ramp-s "$RAMP_S" \
    --ramp-step-users "$RAMP_STEP_USERS" \
    --ramp-step-s "$RAMP_STEP_S" \
    --ramp-down-step-users "$RAMP_DOWN_STEP_USERS" \
    --ramp-down-step-s "$RAMP_DOWN_STEP_S" \
    --ramp-jitter-pct "$RAMP_JITTER_PCT" \
    --ramp-jitter-seed "$RAMP_JITTER_SEED" \
    --hold-s "$HOLD_S" \
    --cooldown-s "$COOLDOWN_S" \
    --max-tokens "$MAX_TOKENS" \
    --max-tokens-jitter "$MAX_TOKENS_JITTER" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --smoothing-window-s "$SMOOTHING_WINDOW_S" \
    --initial-tps-max "$INITIAL_TPS_MAX" \
    --warmup-users "$WARMUP_USERS" \
    --warmup-max-tokens "$WARMUP_MAX_TOKENS" \
    --mode "$MODE" \
    --prompts-dir "$ROOT_DIR/Scripts/demo/prompts" \
    --output "$TRACE_PATH" \
    --requests-output "$REQUESTS_PATH"
}

render_requests_html() {
  log "rendering per-request HTML report"
  python3 "$ROOT_DIR/Scripts/demo/render_requests_html.py" \
    --requests "$REQUESTS_PATH" \
    --summary "$SUMMARY_PATH" \
    --output "$REQUESTS_HTML"
}

verify_run() {
  log "running integrity verification against server log"
  python3 "$ROOT_DIR/Scripts/demo/verify_against_logs.py" \
    --trace "$TRACE_PATH" \
    --requests "$REQUESTS_PATH" \
    --server-log "$SERVER_LOG" \
    --csv "$ROOT_DIR/Scripts/demo/out/verify.csv" 2>&1 | tee "$ROOT_DIR/Scripts/demo/out/verify.txt" | tail -30
}

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
render_video() {
  log "rendering video"
  local subtitle_arg=()
  if [ -n "$SUBTITLE" ]; then
    subtitle_arg=(--subtitle "$SUBTITLE")
  fi

  python3 "$ROOT_DIR/Scripts/demo/render_demo_video.py" \
    --trace "$TRACE_PATH" \
    --summary "$SUMMARY_PATH" \
    --output "$OUTPUT" \
    --title "$TITLE" \
    --smoothing-window "$SMOOTHING_WINDOW" \
    "${subtitle_arg[@]}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo
log "AFM concurrent throughput demo"
echo

launch_watcher() {
  # Open a new Terminal window on macOS running the live watcher.
  # Falls back to spawning in the background on non-macOS platforms.
  local total_s
  total_s=$(python3 - <<PYEND
mode = "$RAMP_MODE"
target = $CONCURRENT
hold = float($HOLD_S)
cooldown = float($COOLDOWN_S)
if mode == "step":
    step_users = max(1, $RAMP_STEP_USERS)
    step_s = float($RAMP_STEP_S)
    dsu = $RAMP_DOWN_STEP_USERS
    dss = $RAMP_DOWN_STEP_S
    dsu = step_users if dsu < 0 else max(1, dsu)
    dss = step_s    if dss < 0 else dss
    n_up = (target + step_users - 1) // step_users
    n_dn = (target + dsu - 1) // dsu
    print(n_up * step_s + hold + n_dn * dss + cooldown)
else:
    print(float($RAMP_S) + hold + cooldown)
PYEND
)
  local run_params="--concurrent $CONCURRENT --ramp-step-users $RAMP_STEP_USERS --ramp-step-s $RAMP_STEP_S --hold $HOLD_S --cooldown $COOLDOWN_S --max-tokens $MAX_TOKENS --smoothing-window $SMOOTHING_WINDOW"
  local cmd="cd '$ROOT_DIR' && python3 Scripts/demo/watch_live.py --trace '$TRACE_PATH' --target-users $CONCURRENT --initial-tps-max $INITIAL_TPS_MAX --total-seconds $total_s --smoothing-window $SMOOTHING_WINDOW --model '$MODEL' --run-params '$run_params'"
  if [[ "$(uname)" == "Darwin" ]] && command -v osascript >/dev/null 2>&1; then
    log "opening live watcher in a new Terminal window"
    osascript <<OSA
tell application "Terminal"
    activate
    do script "${cmd//\"/\\\"}"
end tell
OSA
  else
    warn "non-macOS or no osascript — spawning watcher in background (no window)"
    (python3 "$ROOT_DIR/Scripts/demo/watch_live.py" --trace "$TRACE_PATH" &)
  fi
}

if [ "$SKIP_LOAD" = false ]; then
  start_server
  if [ "$WATCH_LIVE" = true ]; then
    # Create an empty trace file so the watcher has something to tail from t=0,
    # then launch the watcher before the driver starts writing.
    : > "$TRACE_PATH"
    launch_watcher
    sleep 1
  fi
  run_load
else
  log "--skip-load given, re-rendering existing trace at $TRACE_PATH"
  if [ ! -f "$TRACE_PATH" ]; then
    err "no existing trace at $TRACE_PATH — cannot skip load"
    exit 1
  fi
fi

if [ "$SKIP_RENDER" = false ]; then
  render_video
fi

if [ "$SKIP_HTML" = false ] && [ -f "$REQUESTS_PATH" ]; then
  render_requests_html
fi

if [ "$SKIP_VERIFY" = false ] && [ "$SKIP_LOAD" = false ] && [ -f "$SERVER_LOG" ]; then
  verify_run
fi

echo
log "done"
log "  trace   : $TRACE_PATH"
log "  requests: $REQUESTS_PATH"
log "  html    : $REQUESTS_HTML"
log "  video   : $OUTPUT"
log "  server  : $SERVER_LOG"
if [ -f "$SUMMARY_PATH" ]; then
  log "  summary :"
  sed 's/^/    /' "$SUMMARY_PATH"
fi
