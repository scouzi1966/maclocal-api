#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$ROOT_DIR/Scripts/benchmark-context.sh"
AFM_BIN="${AFM_BIN:-$(command -v afm)}"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

[[ -x "$HARNESS" ]] || fail "harness is not executable"
[[ -x "$AFM_BIN" ]] || fail "AFM_BIN is not executable: $AFM_BIN"

help_output="$($HARNESS --help)"
grep -q -- '--warm-prefix' <<<"$help_output" || fail "help omits warm-prefix mode"
grep -q -- '--base-url' <<<"$help_output" || fail "help omits existing-endpoint mode"

managed_output="$(AFM_BIN="$AFM_BIN" "$HARNESS" \
    --model mlx-community/Qwen3-0.6B-4bit \
    --contexts 0.5 \
    --max-tokens 8 \
    --runs 1 \
    --no-batch \
    --afm-arg --no-thinking \
    --dry-run)"
grep -q 'mlx-community/Qwen3-0.6B-4bit' <<<"$managed_output" || fail "managed command omits model"
grep -q -- '--no-thinking' <<<"$managed_output" || fail "managed command omits AFM argument"
grep -q -- '--no-batch' <<<"$managed_output" || fail "benchmark command omits no-batch"

endpoint_output="$($HARNESS \
    --base-url http://127.0.0.1:8123/v1 \
    --model existing-model \
    --warm-prefix \
    --no-sync \
    --dry-run)"
grep -q 'Server command: existing endpoint' <<<"$endpoint_output" || fail "endpoint mode attempts managed startup"
grep -q -- '--no-cold-prefill' <<<"$endpoint_output" || fail "warm-prefix mode was not forwarded"

if "$HARNESS" --mode invalid --dry-run >/dev/null 2>&1; then
    fail "invalid mode was accepted"
fi

echo "PASS: llm_context_benchmarks integration contract"
