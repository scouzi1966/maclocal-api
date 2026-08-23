#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="$ROOT_DIR/.build/xcode27-wheel-layout-test"
SOURCE_BIN="$($ROOT_DIR/Scripts/find-afm-binary.sh release)"
SOURCE_DIR="$(dirname "$SOURCE_BIN")"

rm -rf "$WORK_ROOT"
mkdir -p \
  "$WORK_ROOT/AFMKit_AFMKitMLX.bundle/Contents/Resources"
cp "$SOURCE_BIN" "$WORK_ROOT/afm"
cp -R "$SOURCE_DIR/AFMKit_AFMKitDwarfStar.bundle" "$WORK_ROOT/"

SOURCE_METALLIB="$($ROOT_DIR/Scripts/resolve-afmkit-resource.sh --metallib "$SOURCE_DIR")"
cp "$SOURCE_METALLIB" \
  "$WORK_ROOT/AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib"

AFM_RELEASE_BIN="$WORK_ROOT/afm" "$ROOT_DIR/Scripts/build-stable-wheel.sh"
WHEEL="$(ls -t "$ROOT_DIR"/dist/macafm-*.whl | head -1)"
unzip -Z1 "$WHEEL" | grep -Fqx \
  'macafm/bin/AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib'

echo "[xcode27-wheel-test] nested bundle archive and installed launch verified"
