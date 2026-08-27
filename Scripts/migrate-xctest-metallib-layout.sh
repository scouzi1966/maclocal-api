#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <swiftpm-scratch-path> <state-directory>" >&2
    exit 2
fi

SCRATCH_PATH="$1"
STATE_DIR="$2"

if [[ -z "$SCRATCH_PATH" || "$SCRATCH_PATH" == "/" ]]; then
    echo "Refusing unsafe SwiftPM scratch path: '$SCRATCH_PATH'" >&2
    exit 2
fi

mkdir -p "$STATE_DIR"
SCRATCH_KEY="$(printf '%s\n' "$SCRATCH_PATH" | shasum -a 256 | awk '{print $1}')"
MIGRATION_STAMP="$STATE_DIR/xctest-metallib-layout-v2-$SCRATCH_KEY"

if [[ -f "$MIGRATION_STAMP" ]]; then
    exit 0
fi

# Older wrapper versions copied mlx.metallib into already-signed native-driver
# XCTest bundles. Xcode cannot incrementally re-sign those bundles. Remove only
# the derived native-driver product tree when that stale layout is detected;
# dependency checkouts, downloaded artifacts, and sources remain untouched.
if [[ -d "$SCRATCH_PATH/out" ]] &&
   find "$SCRATCH_PATH/out" -type f -path '*.xctest/Contents/MacOS/mlx.metallib' -print -quit | grep -q .; then
    echo "[swiftpm-reliable] Removing stale signed XCTest products from the previous metallib layout." >&2
    rm -rf "$SCRATCH_PATH/out"
fi

touch "$MIGRATION_STAMP"
