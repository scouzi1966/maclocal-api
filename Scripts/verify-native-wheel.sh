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
for required in \
  "$PACKAGE_ROOT/bin/afm" \
  "$PACKAGE_ROOT/bin/AFMKit_AFMKitMLX.bundle/default.metallib" \
  "$PACKAGE_ROOT/bin/AFMKit_AFMKitDwarfStar.bundle/metal/moe.metal" \
  "$PACKAGE_ROOT/share/webui/index.html.gz"; do
  grep -Fqx "$required" <<<"$CONTENTS" || {
    echo "[wheel] missing required payload: $required" >&2
    exit 1
  }
done

mkdir -p "$VERIFY_ROOT"
VERIFY_DIR="$(mktemp -d "$VERIFY_ROOT/run.XXXXXX")"
cleanup() {
  rm -rf "$VERIFY_DIR"
}
trap cleanup EXIT

unzip -q "$WHEEL" "$PACKAGE_ROOT/bin/*" -d "$VERIFY_DIR"
EXTRACTED_BIN="$VERIFY_DIR/$PACKAGE_ROOT/bin/afm"
EXTRACTED_METALLIB="$VERIFY_DIR/$PACKAGE_ROOT/bin/AFMKit_AFMKitMLX.bundle/default.metallib"
chmod +x "$EXTRACTED_BIN"

"$SCRIPT_DIR/check-macos26-compatibility.sh" "$EXTRACTED_BIN" "$EXTRACTED_METALLIB"
"$EXTRACTED_BIN" --version >/dev/null

echo "[wheel] verified native AFM payload and executable: $WHEEL"
