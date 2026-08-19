#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <wheel> <package-root>" >&2
  exit 2
fi

WHEEL="$1"
PACKAGE_ROOT="$2"
EXPECTED_TAG="py3-none-macosx_26_0_arm64"

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

echo "[wheel] verified native AFM payload: $WHEEL"
