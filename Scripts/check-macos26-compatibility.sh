#!/bin/bash
# Reject release artifacts that claim macOS 26 support while embedding newer
# executable or Metal deployment targets.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${1:-}"
METALLIB="${2:-}"
MAX_MAJOR=26

if [ -z "$BINARY" ]; then
  for candidate in \
    "$ROOT_DIR/.build/arm64-apple-macosx/release/afm" \
    "$ROOT_DIR/.build/release/afm"; do
    if [ -x "$candidate" ]; then
      BINARY="$candidate"
      break
    fi
  done
fi

if [ ! -x "$BINARY" ]; then
  echo "[compat] Missing Release executable: ${BINARY:-<not found>}" >&2
  exit 1
fi

if [ -z "$METALLIB" ]; then
  BUNDLE_DIR="$(dirname "$BINARY")/MacLocalAPI_AFMKitMLX.bundle"
  for candidate in \
    "$BUNDLE_DIR/default.metallib" \
    "$BUNDLE_DIR/Contents/Resources/default.metallib"; do
    if [ -f "$candidate" ]; then
      METALLIB="$candidate"
      break
    fi
  done
fi

if [ ! -f "$METALLIB" ]; then
  echo "[compat] Missing MLX metallib: ${METALLIB:-<not found>}" >&2
  exit 1
fi

MIN_OS="$(/usr/bin/vtool -show-build "$BINARY" | awk '/minos/{print $2; exit}')"
if [ -z "$MIN_OS" ]; then
  echo "[compat] Could not read executable deployment target: $BINARY" >&2
  exit 1
fi
if [ "${MIN_OS%%.*}" -gt "$MAX_MAJOR" ]; then
  echo "[compat] Executable requires macOS $MIN_OS; maximum supported target is macOS $MAX_MAJOR." >&2
  exit 1
fi

METAL_TARGETS="$(LC_ALL=C grep -aoE 'air64(_v[0-9]+)?-apple-macosx[0-9]+(\.[0-9]+){2}' "$METALLIB" | sort -u || true)"
if [ -z "$METAL_TARGETS" ]; then
  echo "[compat] Could not read a deployment target from $METALLIB" >&2
  exit 1
fi

while IFS= read -r target; do
  version="${target##*macosx}"
  if [ "${version%%.*}" -gt "$MAX_MAJOR" ]; then
    echo "[compat] MLX metallib contains $target and cannot load on macOS $MAX_MAJOR." >&2
    exit 1
  fi
done <<< "$METAL_TARGETS"

echo "[compat] macOS 26 compatible: executable minos=$MIN_OS; Metal targets=$(echo "$METAL_TARGETS" | tr '\n' ' ')"
