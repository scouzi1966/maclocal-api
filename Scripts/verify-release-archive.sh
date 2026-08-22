#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <release.tar.gz>" >&2
  exit 2
fi

ARCHIVE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_ROOT="${AFM_ARCHIVE_VERIFY_ROOT:-$SCRIPT_DIR/../.build/release-archive-verify}"

[[ -f "$ARCHIVE" ]] || { echo "[archive] missing archive: $ARCHIVE" >&2; exit 1; }
CONTENTS="$(tar -tzf "$ARCHIVE")"
if grep -Eq '(^/|(^|/)\.\.(/|$))' <<<"$CONTENTS"; then
  echo "[archive] unsafe absolute or parent-relative path" >&2
  exit 1
fi

BIN_ENTRY="$(grep -E '(^|/)afm$' <<<"$CONTENTS" | head -1)"
[[ -n "$BIN_ENTRY" ]] || { echo "[archive] afm executable is missing" >&2; exit 1; }
ROOT_ENTRY="${BIN_ENTRY%afm}"

for required in "${ROOT_ENTRY}Resources/webui/index.html.gz"; do
  grep -Fqx "$required" <<<"$CONTENTS" || {
    echo "[archive] missing required payload: $required" >&2
    exit 1
  }
done


FLAT_DWARF_METAL="${ROOT_ENTRY}AFMKit_AFMKitDwarfStar.bundle/metal/moe.metal"
NESTED_DWARF_METAL="${ROOT_ENTRY}AFMKit_AFMKitDwarfStar.bundle/Contents/Resources/metal/moe.metal"
if ! grep -Fqx "$FLAT_DWARF_METAL" <<<"$CONTENTS" && \
   ! grep -Fqx "$NESTED_DWARF_METAL" <<<"$CONTENTS"; then
  echo "[archive] missing DwarfStar Metal source in flat or Xcode 27 bundle layout" >&2
  exit 1
fi

FLAT_METALLIB="${ROOT_ENTRY}AFMKit_AFMKitMLX.bundle/default.metallib"
NESTED_METALLIB="${ROOT_ENTRY}AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib"
if grep -Fqx "$FLAT_METALLIB" <<<"$CONTENTS"; then
  METALLIB_ENTRY="$FLAT_METALLIB"
elif grep -Fqx "$NESTED_METALLIB" <<<"$CONTENTS"; then
  METALLIB_ENTRY="$NESTED_METALLIB"
else
  echo "[archive] missing AFMKit MLX metallib in flat or Xcode 27 layout" >&2
  exit 1
fi

mkdir -p "$VERIFY_ROOT"
VERIFY_DIR="$(mktemp -d "$VERIFY_ROOT/run.XXXXXX")"
cleanup() { rm -rf "$VERIFY_DIR"; }
trap cleanup EXIT
tar -xzf "$ARCHIVE" -C "$VERIFY_DIR"

EXTRACTED_BIN="$VERIFY_DIR/$BIN_ENTRY"
EXTRACTED_METALLIB="$VERIFY_DIR/$METALLIB_ENTRY"
chmod +x "$EXTRACTED_BIN"
"$SCRIPT_DIR/check-macos26-compatibility.sh" "$EXTRACTED_BIN" "$EXTRACTED_METALLIB"
(
  cd "$(dirname "$EXTRACTED_BIN")"
  ./afm --version >/dev/null
)

ARCHIVE_WEBUI="$VERIFY_DIR/archive-webui.html.gz"
tar -xOzf "$ARCHIVE" "${ROOT_ENTRY}Resources/webui/index.html.gz" > "$ARCHIVE_WEBUI"
"$SCRIPT_DIR/verify-webui.sh" "$ARCHIVE_WEBUI"

echo "[archive] verified payload and relocated launch: $ARCHIVE"
