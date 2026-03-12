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

# Parse version from BuildInfo.swift
VERSION=$(grep -o '"v[0-9.]*"' "$ROOT_DIR/Sources/MacLocalAPI/BuildInfo.swift" | tr -d '"')
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

# Verify webui
WEBUI="$ROOT_DIR/Resources/webui/index.html.gz"
if [ ! -f "$WEBUI" ]; then
  log_error "Missing webui: $WEBUI"
  exit 1
fi

# Stage tarball contents
STAGING=$(mktemp -d)
DIRNAME="afm-${VERSION}-${ARCH}"
mkdir -p "$STAGING/$DIRNAME/Resources/webui"
cp "$BIN" "$STAGING/$DIRNAME/"
cp "$WEBUI" "$STAGING/$DIRNAME/Resources/webui/"

# Create tarball
tar -czf "$OUTPUT" -C "$STAGING" "$DIRNAME"
rm -rf "$STAGING"

SIZE=$(du -h "$OUTPUT" | cut -f1 | xargs)
BIN_SIZE=$(du -h "$BIN" | cut -f1 | xargs)

log_info "Tarball: $OUTPUT ($SIZE)"
log_info "Binary:  $BIN_SIZE (stripped $ARCH)"
log_info "Contents:"
tar -tzf "$OUTPUT" | sed 's/^/  /'
