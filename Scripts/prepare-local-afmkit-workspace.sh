#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_AFMKIT_PATH="${MACLOCAL_AFMKIT_PATH:-}"
WORK_ROOT="${MACLOCAL_AFMKIT_WORK_ROOT:-$ROOT_DIR/.build-local-afmkit-workspace}"
PACKAGE_ROOT="$WORK_ROOT/package"

if [[ -z "$LOCAL_AFMKIT_PATH" ]]; then
  echo "[local-afmkit] MACLOCAL_AFMKIT_PATH is required." >&2
  exit 2
fi
if [[ ! -f "$LOCAL_AFMKIT_PATH/Package.swift" ]]; then
  echo "[local-afmkit] Invalid AFMKit checkout: $LOCAL_AFMKIT_PATH" >&2
  exit 2
fi

LOCAL_AFMKIT_PATH="$(cd "$LOCAL_AFMKIT_PATH" && pwd)"
WORK_ROOT="$(mkdir -p "$WORK_ROOT" && cd "$WORK_ROOT" && pwd)"
if [[ "$LOCAL_AFMKIT_PATH" == "$WORK_ROOT"/* ]]; then
  echo "[local-afmkit] AFMKit checkout cannot be inside the generated workspace." >&2
  exit 2
fi

rm -rf "$PACKAGE_ROOT"
mkdir -p "$PACKAGE_ROOT"
cp "$ROOT_DIR/Package.swift" "$ROOT_DIR/Package.resolved" "$PACKAGE_ROOT/"
cp -R "$ROOT_DIR/Sources" "$ROOT_DIR/Tests" "$PACKAGE_ROOT/"

printf '%s\n' "$PACKAGE_ROOT"
