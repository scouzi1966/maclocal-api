#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/.build"
PRODUCTS_DIR="$BUILD_DIR/out/Products/Debug"
BASELINE="$ROOT/docs/api-baselines/AFMKitCore.symbols.json"
CURRENT_DIR="$BUILD_DIR/api-current"
RAW_CURRENT_DIR="$BUILD_DIR/api-current-raw"
MODULE_CACHE="$BUILD_DIR/api-module-cache"
SDK="$(xcrun --sdk macosx --show-sdk-path)"
ARCH="$(uname -m)"

export SWIFTPM_MODULECACHE_OVERRIDE="$BUILD_DIR/swiftpm-module-cache"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/clang-module-cache"

cd "$ROOT"
swift build --target AFMKitCore

rm -rf "$CURRENT_DIR" "$RAW_CURRENT_DIR"
mkdir -p "$CURRENT_DIR" "$RAW_CURRENT_DIR" "$MODULE_CACHE"

xcrun swift-symbolgraph-extract \
    -module-name AFMKitCore \
    -I "$PRODUCTS_DIR" \
    -output-dir "$RAW_CURRENT_DIR" \
    -minimum-access-level public \
    -skip-synthesized-members \
    -skip-inherited-docs \
    -pretty-print \
    -sdk "$SDK" \
    -target "${ARCH}-apple-macos26.0" \
    -module-cache-path "$MODULE_CACHE"

python3 - "$RAW_CURRENT_DIR/AFMKitCore.symbols.json" "$CURRENT_DIR/AFMKitCore.symbols.json" <<'PY'
import json
import sys

raw_path, normalized_path = sys.argv[1:3]
VOLATILE_KEYS = {"generator", "location", "uri", "range"}


def normalize(value):
    if isinstance(value, dict):
        return {
            key: normalize(value[key])
            for key in sorted(value)
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        normalized = [normalize(item) for item in value]
        if all(isinstance(item, dict) for item in normalized):
            return sorted(
                normalized,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        return normalized
    return value


with open(raw_path, "r", encoding="utf-8") as handle:
    raw = json.load(handle)

with open(normalized_path, "w", encoding="utf-8") as handle:
    json.dump(normalize(raw), handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if ! cmp -s "$BASELINE" "$CURRENT_DIR/AFMKitCore.symbols.json"; then
    echo "AFMKitCore public API differs from $BASELINE" >&2
    echo "Review the API change, then replace the baseline intentionally." >&2
    diff -u "$BASELINE" "$CURRENT_DIR/AFMKitCore.symbols.json" || true
    exit 1
fi

echo "AFMKitCore public API matches its checked-in baseline."
