#!/bin/bash
# Creates a release tarball from a completed build.
# Run build-from-scratch.sh first, then this script.
#
# Usage:
#   ./Scripts/create-tarball.sh
#   ./Scripts/create-tarball.sh --output /path/to/output.tar.gz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse only the declared semantic version. BuildInfo also contains string
# interpolation beginning with "v", which the old broad grep treated as a
# second version and embedded a newline in the archive's root directory.
VERSION=$(grep 'static let version' "$ROOT_DIR/Sources/AFMKit/BuildInfo.swift" \
  | head -1 \
  | sed -E 's/.*"(v[0-9]+(\.[0-9]+)+)".*/\1/')
if [ -z "$VERSION" ]; then
  log_error "Could not read version from BuildInfo.swift"
  exit 1
fi

ARCH=$(uname -m)
TARBALL_NAME="afm-${VERSION}-${ARCH}.tar.gz"
OUTPUT=""

for arg in "$@"; do
  case "$arg" in
    --output=*) OUTPUT="${arg#--output=}" ;;
    --output) shift; OUTPUT="$1" ;;  # handled below
    -h|--help)
      echo "Usage: $0 [--output /path/to/tarball.tar.gz]"
      exit 0
      ;;
  esac
done

# Handle --output with separate value
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [ -z "$OUTPUT" ]; then
  OUTPUT="$ROOT_DIR/$TARBALL_NAME"
fi

# Find the built binary
BIN_PATH_1="$ROOT_DIR/.build/arm64-apple-macosx/release/afm"
BIN_PATH_2="$ROOT_DIR/.build/release/afm"

if [ -x "$BIN_PATH_1" ]; then
  BIN="$BIN_PATH_1"
elif [ -x "$BIN_PATH_2" ]; then
  BIN="$BIN_PATH_2"
else
  log_error "No release binary found. Run build-from-scratch.sh first."
  exit 1
fi

"$SCRIPT_DIR/check-macos26-compatibility.sh" "$BIN"

# Verify webui
WEBUI="$ROOT_DIR/Resources/webui/index.html.gz"
if [ ! -f "$WEBUI" ]; then
  log_error "Missing webui: $WEBUI"
  exit 1
fi
"$SCRIPT_DIR/verify-webui.sh" "$WEBUI"

# Stage tarball contents
PACKAGE_WORK_ROOT="${AFM_PACKAGE_WORK_ROOT:-$ROOT_DIR/.build/package-work}"
mkdir -p "$PACKAGE_WORK_ROOT"
STAGING=""
cleanup_staging() {
  if [[ -n "$STAGING" && -d "$STAGING" ]]; then
    rm -rf -- "$STAGING"
  fi
}
trap cleanup_staging EXIT
if ! STAGING=$(mktemp -d "$PACKAGE_WORK_ROOT/afm-package.XXXXXX"); then
  log_error "Unable to create package staging directory under $PACKAGE_WORK_ROOT"
  exit 1
fi
DIRNAME="afm-${VERSION}-${ARCH}"
mkdir -p "$STAGING/$DIRNAME/Resources/webui"
cp "$BIN" "$STAGING/$DIRNAME/"
for BUNDLE_NAME in AFMKit_AFMKitMLX.bundle AFMKit_AFMKitDwarfStar.bundle; do
  BUNDLE_DIR="$(dirname "$BIN")/$BUNDLE_NAME"
  if [ ! -d "$BUNDLE_DIR" ]; then
    log_error "Required runtime bundle missing: $BUNDLE_DIR"
    exit 1
  fi
  cp -R "$BUNDLE_DIR" "$STAGING/$DIRNAME/"
done
cp "$WEBUI" "$STAGING/$DIRNAME/Resources/webui/"

# Create tarball
tar -czf "$OUTPUT" -C "$STAGING" "$DIRNAME"
ARCHIVE_WEBUI="$PACKAGE_WORK_ROOT/archive-webui.html.gz"
tar -xOzf "$OUTPUT" "$DIRNAME/Resources/webui/index.html.gz" > "$ARCHIVE_WEBUI"
"$SCRIPT_DIR/verify-webui.sh" "$ARCHIVE_WEBUI"
rm -f "$ARCHIVE_WEBUI"
cleanup_staging
trap - EXIT

SIZE=$(du -h "$OUTPUT" | cut -f1 | xargs)
BIN_SIZE=$(du -h "$BIN" | cut -f1 | xargs)

log_info "Tarball: $OUTPUT ($SIZE)"
log_info "Binary:  $BIN_SIZE (stripped $ARCH)"
log_info "Contents:"
tar -tzf "$OUTPUT" | sed 's/^/  /'
