#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-release}"

case "$CONFIG" in
  release)
    XCODE_CONFIG="Release"
    ;;
  debug)
    XCODE_CONFIG="Debug"
    ;;
  *)
    echo "Usage: $0 <release|debug>" >&2
    exit 2
    ;;
esac

CANDIDATES=(
  "$ROOT_DIR/.build/out/Products/$XCODE_CONFIG/afm"
  "$ROOT_DIR/.build/arm64-apple-macosx/$CONFIG/afm"
  "$ROOT_DIR/.build/$CONFIG/afm"
)

for candidate in "${CANDIDATES[@]}"; do
  if [[ -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    exit 0
  fi
done

echo "afm $CONFIG binary not found in any supported SwiftPM output layout" >&2
exit 1
