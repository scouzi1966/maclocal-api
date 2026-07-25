#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/.build"
PRODUCTS_DIR="$BUILD_DIR/out/Products/Debug"
BASELINE="$ROOT/docs/api-baselines/AFMKitCore.symbols.json"
CURRENT_DIR="$BUILD_DIR/api-current"
MODULE_CACHE="$BUILD_DIR/api-module-cache"
SDK="$(xcrun --sdk macosx --show-sdk-path)"
ARCH="$(uname -m)"

export SWIFTPM_MODULECACHE_OVERRIDE="$BUILD_DIR/swiftpm-module-cache"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/clang-module-cache"

cd "$ROOT"
swift build --target AFMKitCore

rm -rf "$CURRENT_DIR"
mkdir -p "$CURRENT_DIR" "$MODULE_CACHE"

xcrun swift-symbolgraph-extract \
    -module-name AFMKitCore \
    -I "$PRODUCTS_DIR" \
    -output-dir "$CURRENT_DIR" \
    -minimum-access-level public \
    -skip-synthesized-members \
    -skip-inherited-docs \
    -pretty-print \
    -sdk "$SDK" \
    -target "${ARCH}-apple-macos26.0" \
    -module-cache-path "$MODULE_CACHE"

if ! cmp -s "$BASELINE" "$CURRENT_DIR/AFMKitCore.symbols.json"; then
    echo "AFMKitCore public API differs from $BASELINE" >&2
    echo "Review the API change, then replace the baseline intentionally." >&2
    diff -u "$BASELINE" "$CURRENT_DIR/AFMKitCore.symbols.json" || true
    exit 1
fi

echo "AFMKitCore public API matches its checked-in baseline."
