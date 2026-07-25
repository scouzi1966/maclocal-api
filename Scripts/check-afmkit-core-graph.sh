#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="$ROOT/.build/core-only-gate"
LOG="$ROOT/.build/core-only-gate.log"

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

if grep -E -- '-module-name (AFMKitMLX|AFMKitFoundationModels|AFMKitFoundationModels27|AFMKitServices|AFMServer|Vapor|CXGrammar|MLX)' "$LOG"; then
    echo "AFMKitCore unexpectedly compiled an optional implementation target." >&2
    exit 1
fi

if ! grep -E -- '-module-name AFMKitCore' "$LOG" >/dev/null; then
    echo "AFMKitCore was not compiled by the graph gate." >&2
    exit 1
fi

echo "AFMKitCore build graph contains no optional implementation targets."
