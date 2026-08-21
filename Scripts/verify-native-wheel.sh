#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <wheel> <package-root>" >&2
  exit 2
fi

WHEEL="$1"
PACKAGE_ROOT="$2"
EXPECTED_TAG="py3-none-macosx_26_0_arm64"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_ROOT="${AFM_WHEEL_VERIFY_ROOT:-$SCRIPT_DIR/../.build/native-wheel-verify}"

if [[ ! -f "$WHEEL" ]]; then
  echo "[wheel] missing wheel: $WHEEL" >&2
  exit 1
fi

if [[ "$(basename "$WHEEL")" != *"-${EXPECTED_TAG}.whl" ]]; then
  echo "[wheel] invalid platform tag: $(basename "$WHEEL")" >&2
  exit 1
fi

WHEEL_METADATA="$(unzip -Z1 "$WHEEL" | grep -E '/WHEEL$' | head -1)"
if [[ -z "$WHEEL_METADATA" ]]; then
  echo "[wheel] WHEEL metadata is missing" >&2
  exit 1
fi

METADATA="$(unzip -p "$WHEEL" "$WHEEL_METADATA")"
grep -Fqx "Root-Is-Purelib: false" <<<"$METADATA" || {
  echo "[wheel] native payload is incorrectly marked as pure Python" >&2
  exit 1
}
grep -Fqx "Tag: $EXPECTED_TAG" <<<"$METADATA" || {
  echo "[wheel] metadata tag does not match $EXPECTED_TAG" >&2
  exit 1
}

CONTENTS="$(unzip -Z1 "$WHEEL")"
if [[ "$PACKAGE_ROOT" == "macafm" ]] && grep -Eq '^macafm_next/' <<<"$CONTENTS"; then
  echo "[wheel] stable wheel contains stale nightly package data" >&2
  exit 1
fi
if [[ "$PACKAGE_ROOT" == "macafm_next" ]] && grep -Eq '^macafm/' <<<"$CONTENTS"; then
  echo "[wheel] nightly wheel contains stale stable package data" >&2
  exit 1
fi
for required in \
  "$PACKAGE_ROOT/bin/afm" \
  "$PACKAGE_ROOT/share/webui/index.html.gz"; do
  grep -Fqx "$required" <<<"$CONTENTS" || {
    echo "[wheel] missing required payload: $required" >&2
    exit 1
  }
done

FLAT_METALLIB="$PACKAGE_ROOT/bin/AFMKit_AFMKitMLX.bundle/default.metallib"
NESTED_METALLIB="$PACKAGE_ROOT/bin/AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib"
if grep -Fqx "$FLAT_METALLIB" <<<"$CONTENTS"; then
  WHEEL_METALLIB="$FLAT_METALLIB"
elif grep -Fqx "$NESTED_METALLIB" <<<"$CONTENTS"; then
  WHEEL_METALLIB="$NESTED_METALLIB"
else
  echo "[wheel] missing AFMKit MLX metallib in flat or Xcode 27 bundle layout" >&2
  exit 1
fi

FLAT_DWARF_METAL="$PACKAGE_ROOT/bin/AFMKit_AFMKitDwarfStar.bundle/metal/moe.metal"
NESTED_DWARF_METAL="$PACKAGE_ROOT/bin/AFMKit_AFMKitDwarfStar.bundle/Contents/Resources/metal/moe.metal"
if ! grep -Fqx "$FLAT_DWARF_METAL" <<<"$CONTENTS" && \
   ! grep -Fqx "$NESTED_DWARF_METAL" <<<"$CONTENTS"; then
  echo "[wheel] missing DwarfStar Metal source in flat or Xcode 27 bundle layout" >&2
  exit 1
fi

mkdir -p "$VERIFY_ROOT"
VERIFY_DIR="$(mktemp -d "$VERIFY_ROOT/run.XXXXXX")"
cleanup() {
  rm -rf "$VERIFY_DIR"
}
trap cleanup EXIT

unzip -q "$WHEEL" "$PACKAGE_ROOT/*" -d "$VERIFY_DIR"
EXTRACTED_BIN="$VERIFY_DIR/$PACKAGE_ROOT/bin/afm"
EXTRACTED_METALLIB="$VERIFY_DIR/$WHEEL_METALLIB"
chmod +x "$EXTRACTED_BIN"

"$SCRIPT_DIR/check-macos26-compatibility.sh" "$EXTRACTED_BIN" "$EXTRACTED_METALLIB"
"$EXTRACTED_BIN" --version >/dev/null

PYTHON="${AFM_WHEEL_PYTHON:-python3}"
"$PYTHON" -m venv "$VERIFY_DIR/venv"
PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$VERIFY_DIR/venv/bin/python" -m pip install --no-deps --force-reinstall "$WHEEL" >/dev/null
"$VERIFY_DIR/venv/bin/afm" --version >/dev/null

site_metallib="$(find "$VERIFY_DIR/venv" -path "*/site-packages/$WHEEL_METALLIB" -type f -print -quit)"
if [[ -z "$site_metallib" ]]; then
  echo "[wheel] installed wheel is missing its AFMKit MLX metallib" >&2
  exit 1
fi

echo "[wheel] verified archive payload plus installed-wheel launch: $WHEEL"
