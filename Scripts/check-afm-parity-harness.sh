#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="$ROOT/.build/afm-parity-harness-gate"
SOURCE="$ROOT/Examples/AFMParityCheck/Sources/AFMParityCheck/main.swift"
README="$ROOT/Examples/AFMParityCheck/README.md"

export SWIFTPM_MODULECACHE_OVERRIDE="$ROOT/.build/afm-parity-swiftpm-module-cache"
export CLANG_MODULE_CACHE_PATH="$ROOT/.build/afm-parity-clang-module-cache"
export XDG_CACHE_HOME="$ROOT/.build/afm-parity-xdg-cache"

for token in AFM_PARITY_REPORT AFM_PARITY_REQUIRED_CASES missingRequiredCases; do
    if ! grep -q "$token" "$SOURCE" "$README"; then
        echo "AFMParityCheck is missing report contract token: $token" >&2
        exit 1
    fi
done

swift build \
    --package-path "$ROOT/Examples/AFMParityCheck" \
    --scratch-path "$SCRATCH"

echo "AFMParityCheck report-capable harness builds and documents its required-case contract."
