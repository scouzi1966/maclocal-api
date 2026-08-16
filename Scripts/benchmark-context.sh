#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_DIR="${LLM_CONTEXT_BENCHMARKS_DIR:-$ROOT_DIR/vendor/llm-context-benchmarks}"
UV_BIN="${UV_BIN:-$(command -v uv 2>/dev/null || true)}"
AFM_BIN="${AFM_BIN:-}"
MODEL=""
MODE="mlx"
BASE_URL=""
PORT="9999"
CONTEXTS="0.5,1,2,4,8,16,32"
MAX_TOKENS="128"
RUNS="2"
TIMEOUT="3600"
TEMPERATURE="0"
OUTPUT_DIR=""
START_TIMEOUT="${AFM_CONTEXT_START_TIMEOUT:-1800}"
SYNC_DEPENDENCIES=1
COLD_PREFILL=1
RUN_BATCH=1
DRY_RUN=0
AFM_ARGS=()
BENCHMARK_ARGS=()
SERVER_PID=""

usage() {
    cat <<'EOF'
Usage:
  Scripts/benchmark-context.sh --model <repo-or-path> [options]
  Scripts/benchmark-context.sh --base-url <url> [--model <id>] [options]

Run llm_context_benchmarks against AFM's OpenAI-compatible API.

AFM startup:
  --model <repo-or-path>     Model passed to AFM and the benchmark
  --mode <mlx|foundation>    AFM runtime to start (default: mlx)
  --afm-bin <path>           AFM executable (default: release build, then PATH)
  --port <port>              Managed AFM server port (default: 9999)
  --afm-arg <value>          Additional AFM argument; repeat for multiple values
  --base-url <url>           Benchmark an existing endpoint; do not start AFM

Benchmark selection:
  --contexts <list>          Context sizes in thousands (default: 0.5,1,2,4,8,16,32)
  --max-tokens <count>       Maximum generated tokens per run (default: 128)
  --runs <count>             Runs per context; peak is retained (default: 2)
  --timeout <seconds>        Per-context timeout (default: 3600)
  --temperature <value>      Sampling temperature (default: 0)
  --warm-prefix              Allow prefix-cache reuse instead of cold prefill
  --no-batch                 Skip the concurrent batch-size sweep
  --benchmark-arg <value>    Additional upstream argument; repeat as needed

Harness behavior:
  --output-dir <path>        Persistent run directory
  --no-sync                  Reuse the upstream uv environment without syncing
  --start-timeout <seconds>  Wait limit for managed AFM startup (default: 1800)
  --dry-run                  Print resolved commands without starting anything
  -h, --help                 Show this help

Examples:
  Scripts/benchmark-context.sh \
    --model mlx-community/Qwen3.8-27B-4bit \
    --contexts 0.5,1,2,4,8,16 --max-tokens 256

  Scripts/benchmark-context.sh \
    --model mlx-community/Qwen3.8-27B-4bit \
    --afm-arg --mtp --warm-prefix

  Scripts/benchmark-context.sh \
    --base-url http://127.0.0.1:9999/v1 --model loaded-model --no-sync
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping managed AFM server (PID $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$SERVER_PID" 2>/dev/null || return 0
            sleep 0.25
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:?missing value for --model}"; shift 2 ;;
        --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
        --afm-bin) AFM_BIN="${2:?missing value for --afm-bin}"; shift 2 ;;
        --port) PORT="${2:?missing value for --port}"; shift 2 ;;
        --afm-arg) AFM_ARGS+=("${2:?missing value for --afm-arg}"); shift 2 ;;
        --base-url) BASE_URL="${2:?missing value for --base-url}"; shift 2 ;;
        --contexts) CONTEXTS="${2:?missing value for --contexts}"; shift 2 ;;
        --max-tokens) MAX_TOKENS="${2:?missing value for --max-tokens}"; shift 2 ;;
        --runs) RUNS="${2:?missing value for --runs}"; shift 2 ;;
        --timeout) TIMEOUT="${2:?missing value for --timeout}"; shift 2 ;;
        --temperature) TEMPERATURE="${2:?missing value for --temperature}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?missing value for --output-dir}"; shift 2 ;;
        --start-timeout) START_TIMEOUT="${2:?missing value for --start-timeout}"; shift 2 ;;
        --benchmark-arg) BENCHMARK_ARGS+=("${2:?missing value for --benchmark-arg}"); shift 2 ;;
        --warm-prefix) COLD_PREFILL=0; shift ;;
        --no-batch) RUN_BATCH=0; shift ;;
        --no-sync) SYNC_DEPENDENCIES=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; BENCHMARK_ARGS+=("$@"); break ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ "$MODE" == "mlx" || "$MODE" == "foundation" ]] || die "--mode must be mlx or foundation"
[[ -d "$BENCHMARK_DIR" && -f "$BENCHMARK_DIR/openai_benchmark.py" ]] || \
    die "benchmark dependency is unavailable; run: git submodule update --init vendor/llm-context-benchmarks"
[[ -n "$UV_BIN" && -x "$UV_BIN" ]] || die "uv is required (https://docs.astral.sh/uv/)"

MANAGED_SERVER=0
if [[ -z "$BASE_URL" ]]; then
    MANAGED_SERVER=1
    if [[ -z "$AFM_BIN" ]]; then
        if [[ -x "$ROOT_DIR/Scripts/find-afm-binary.sh" ]]; then
            AFM_BIN="$($ROOT_DIR/Scripts/find-afm-binary.sh release 2>/dev/null || true)"
        fi
        if [[ -z "$AFM_BIN" || ! -x "$AFM_BIN" ]]; then
            AFM_BIN="$(command -v afm 2>/dev/null || true)"
        fi
    fi
    [[ -n "$AFM_BIN" && -x "$AFM_BIN" ]] || \
        die "AFM executable not found; build release AFM or pass --afm-bin"
    if [[ "$MODE" == "mlx" && -z "$MODEL" ]]; then
        die "--model is required when starting the MLX runtime"
    fi
    BASE_URL="http://127.0.0.1:$PORT/v1"
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/test-reports/llm-context-benchmarks/$TIMESTAMP}"

SERVER_CMD=()
if [[ "$MANAGED_SERVER" -eq 1 ]]; then
    if [[ "$MODE" == "mlx" ]]; then
        SERVER_CMD=("$AFM_BIN" mlx -m "$MODEL" -p "$PORT")
    else
        SERVER_CMD=("$AFM_BIN" -p "$PORT")
    fi
    SERVER_CMD+=("${AFM_ARGS[@]}")
fi

BENCHMARK_CMD=(
    "$UV_BIN" run --project "$BENCHMARK_DIR" --frozen --no-sync
    openai-benchmark
    --base-url "$BASE_URL"
    --contexts "$CONTEXTS"
    --max-tokens "$MAX_TOKENS"
    --runs "$RUNS"
    --timeout "$TIMEOUT"
    --temperature "$TEMPERATURE"
    --save-responses
)
if [[ -n "$MODEL" ]]; then
    BENCHMARK_CMD+=(--model "$MODEL")
fi
if [[ "$COLD_PREFILL" -eq 0 ]]; then
    BENCHMARK_CMD+=(--no-cold-prefill)
fi
if [[ "$RUN_BATCH" -eq 0 ]]; then
    BENCHMARK_CMD+=(--no-batch)
fi
BENCHMARK_CMD+=("${BENCHMARK_ARGS[@]}")

echo "AFM context benchmark"
echo "  upstream:  $(git -C "$BENCHMARK_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
echo "  model:     ${MODEL:-auto-detect}"
echo "  endpoint:  $BASE_URL"
echo "  contexts:  $CONTEXTS"
echo "  output:    $OUTPUT_DIR"
if [[ ${#SERVER_CMD[@]} -gt 0 ]]; then
    echo "Server command:"
    print_command "${SERVER_CMD[@]}"
else
    echo "Server command: existing endpoint"
fi
echo "Benchmark command:"
print_command "${BENCHMARK_CMD[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
IFS=',' read -r -a CONTEXT_LIST <<< "$CONTEXTS"
for context in "${CONTEXT_LIST[@]}"; do
    context="${context//[[:space:]]/}"
    source_file="$BENCHMARK_DIR/${context}k.txt"
    [[ -f "$source_file" ]] || die "upstream context fixture not found: $source_file"
    ln -sf "$source_file" "$OUTPUT_DIR/${context}k.txt"
done

if [[ "$SYNC_DEPENDENCIES" -eq 1 ]]; then
    echo "Preparing pinned benchmark environment..."
    "$UV_BIN" sync --project "$BENCHMARK_DIR" --frozen
fi

if [[ ${#SERVER_CMD[@]} -gt 0 ]]; then
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        die "port $PORT is already in use; choose --port or use --base-url"
    fi
    echo "Starting managed AFM server..."
    "${SERVER_CMD[@]}" >"$OUTPUT_DIR/afm-server.log" 2>&1 &
    SERVER_PID=$!
    deadline=$((SECONDS + START_TIMEOUT))
    until curl --fail --silent "$BASE_URL/models" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -100 "$OUTPUT_DIR/afm-server.log" >&2 || true
            die "AFM exited before becoming ready"
        fi
        if (( SECONDS >= deadline )); then
            tail -100 "$OUTPUT_DIR/afm-server.log" >&2 || true
            die "AFM did not become ready within ${START_TIMEOUT}s"
        fi
        sleep 1
    done
    echo "AFM is ready (PID $SERVER_PID)."
fi

AFM_VERSION="external-endpoint"
if [[ -n "$AFM_BIN" ]]; then
    AFM_VERSION="$($AFM_BIN --version 2>/dev/null || echo unknown)"
fi
cat >"$OUTPUT_DIR/provenance.txt" <<EOF
timestamp_utc=$TIMESTAMP
afm_repository=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
benchmark_repository=$(git -C "$BENCHMARK_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
afm_version=$AFM_VERSION
model=${MODEL:-auto-detect}
base_url=$BASE_URL
contexts=$CONTEXTS
max_tokens=$MAX_TOKENS
runs=$RUNS
temperature=$TEMPERATURE
cold_prefill=$COLD_PREFILL
batch=$RUN_BATCH
EOF

echo "Running context benchmark..."
(
    cd "$OUTPUT_DIR"
    "${BENCHMARK_CMD[@]}"
) 2>&1 | tee "$OUTPUT_DIR/benchmark.log"

RESULT_DIR="$(find "$OUTPUT_DIR/output" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort | tail -1)"
[[ -n "$RESULT_DIR" ]] || die "benchmark completed without a result directory"
for artifact in benchmark_results.csv benchmark_chart.png table.txt hardware_info.json; do
    [[ -s "$RESULT_DIR/$artifact" ]] || die "required benchmark artifact is missing or empty: $artifact"
done

echo "Context benchmark passed artifact validation."
echo "Results: $RESULT_DIR"
