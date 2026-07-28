#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="$ROOT/.build/core-only-gate"
LOG="$ROOT/.build/core-only-gate.log"
CONSUMER_SCRATCH="$ROOT/.build/core-only-consumer-gate"
CONSUMER_LOG="$ROOT/.build/core-only-consumer-gate.log"

OPTIONAL_MODULE_PATTERN='-module-name (AFMKit|AFMKitMLX|AFMKitFoundationModels|AFMKitFoundationModels27|AFMKitServices|AFMOpenAICompat|AFMServer|AFMCLI|AFMXGrammar|Vapor|NIO[A-Za-z]*|CXGrammar|XGrammar|MLX|MLXLLM|MLXVLM|MLXLMCommon|Tokenizers|Hub|HuggingFace)([[:space:]]|$)'

export SWIFTPM_MODULECACHE_OVERRIDE="$ROOT/.build/core-only-swiftpm-module-cache"
export CLANG_MODULE_CACHE_PATH="$ROOT/.build/core-only-clang-module-cache"
export XDG_CACHE_HOME="$ROOT/.build/core-only-xdg-cache"

if grep -Eq '\.package\([^)]*branch:' "$ROOT/Package.swift"; then
    echo "Package.swift contains an unpinned branch dependency." >&2
    exit 1
fi

rm -rf "$SCRATCH"
mkdir -p "$ROOT/.build"

swift build \
    --package-path "$ROOT" \
    --scratch-path "$SCRATCH" \
    --target AFMKitCore \
    -v 2>&1 | tee "$LOG"

if grep -E -- "$OPTIONAL_MODULE_PATTERN" "$LOG"; then
    echo "AFMKitCore unexpectedly compiled an optional implementation target." >&2
    exit 1
fi

if ! grep -E -- '-module-name AFMKitCore' "$LOG" >/dev/null; then
    echo "AFMKitCore was not compiled by the graph gate." >&2
    exit 1
fi

swift build \
    --package-path "$ROOT/Examples/AFMKitCoreOnlyConsumer" \
    --scratch-path "$CONSUMER_SCRATCH" \
    -v 2>&1 | tee "$CONSUMER_LOG"

if grep -E -- "$OPTIONAL_MODULE_PATTERN" "$CONSUMER_LOG"; then
    echo "AFMKitCore-only consumer unexpectedly compiled an optional implementation target." >&2
    exit 1
fi

if ! grep -E -- '-module-name AFMKitCoreOnlyConsumer' "$CONSUMER_LOG" >/dev/null; then
    echo "AFMKitCore-only consumer was not compiled by the graph gate." >&2
    exit 1
fi

echo "AFMKitCore build graph and core-only consumer contain no optional implementation targets."
