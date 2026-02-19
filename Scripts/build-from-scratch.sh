#!/bin/bash
# Build AFM from a fresh clone (maclocal-api context)
#
# Steps:
#   1) Initialize submodules
#   2) Apply MLX patch set (Vesta-style patch management)
#   3) Build llama.cpp webui assets (optional)
#   4) Clean + resolve Swift packages
#   5) Build afm (release by default)
#
# Usage:
#   ./Scripts/build-from-scratch.sh
#   ./Scripts/build-from-scratch.sh --skip-webui
#   ./Scripts/build-from-scratch.sh --debug --no-clean

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_CONFIG="release"
DO_CLEAN=true
DO_SUBMODULES=true
DO_PATCHES=true
DO_WEBUI=true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

usage() {
  cat <<USAGE
Usage: ./Scripts/build-from-scratch.sh [options]

Options:
  --debug              Build debug instead of release
  --no-clean           Skip clean step before build
  --skip-submodules    Skip git submodule init/update
  --skip-patches       Skip Scripts/apply-mlx-patches.sh
  --skip-webui         Skip llama.cpp webui build
  -h, --help           Show help

Default behavior:
  submodules + patches + webui + clean + release build
USAGE
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_error "Required command not found: $1"
    exit 1
  fi
}

for arg in "$@"; do
  case "$arg" in
    --debug) BUILD_CONFIG="debug" ;;
    --no-clean) DO_CLEAN=false ;;
    --skip-submodules) DO_SUBMODULES=false ;;
    --skip-patches) DO_PATCHES=false ;;
    --skip-webui) DO_WEBUI=false ;;
    -h|--help) usage; exit 0 ;;
    *)
      log_error "Unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

if [ ! -f "$ROOT_DIR/Package.swift" ]; then
  log_error "Package.swift not found. Run this script from a maclocal-api clone."
  exit 1
fi

require_cmd git
require_cmd swift

cd "$ROOT_DIR"

log_info "Repository: $ROOT_DIR"
log_info "Build configuration: $BUILD_CONFIG"

if $DO_SUBMODULES; then
  log_step "Initializing submodules"
  git submodule update --init --recursive
  git submodule status || true
else
  log_warn "Skipping submodule initialization"
fi

if $DO_PATCHES; then
  log_step "Applying MLX patch set"
  if [ ! -x "$SCRIPT_DIR/apply-mlx-patches.sh" ]; then
    log_error "Missing patch script: $SCRIPT_DIR/apply-mlx-patches.sh"
    exit 1
  fi
  "$SCRIPT_DIR/apply-mlx-patches.sh"
  "$SCRIPT_DIR/apply-mlx-patches.sh" --check
else
  log_warn "Skipping MLX patch application"
fi

if $DO_WEBUI; then
  log_step "Building llama.cpp webui"
  WEBUI_DIR="$ROOT_DIR/vendor/llama.cpp/tools/server/webui"
  if [ ! -d "$WEBUI_DIR" ]; then
    log_error "webui source not found: $WEBUI_DIR"
    log_error "Did submodules initialize correctly?"
    exit 1
  fi
  require_cmd npm
  (
    cd "$WEBUI_DIR"
    npm install
    npm run build
  )
  if [ -f "$ROOT_DIR/vendor/llama.cpp/tools/server/public/index.html.gz" ]; then
    mkdir -p "$ROOT_DIR/Resources/webui"
    cp "$ROOT_DIR/vendor/llama.cpp/tools/server/public/index.html.gz" "$ROOT_DIR/Resources/webui/index.html.gz"
    log_info "WebUI artifact copied to Resources/webui/index.html.gz"
  fi
else
  log_warn "Skipping webui build"
fi

log_step "Validating required resources"
if [ ! -f "$ROOT_DIR/Sources/MacLocalAPI/Resources/default.metallib" ]; then
  log_error "Missing metallib: Sources/MacLocalAPI/Resources/default.metallib"
  exit 1
fi

if $DO_CLEAN; then
  log_step "Cleaning previous Swift build artifacts"
  swift package clean
fi

log_step "Resolving Swift packages"
swift package resolve

log_step "Building afm ($BUILD_CONFIG)"
if [ "$BUILD_CONFIG" = "release" ]; then
  swift build -c release \
    --product afm \
    -Xswiftc -O \
    -Xswiftc -whole-module-optimization \
    -Xswiftc -cross-module-optimization
else
  swift build -c "$BUILD_CONFIG"
fi

BIN_PATH_1="$ROOT_DIR/.build/arm64-apple-macosx/$BUILD_CONFIG/afm"
BIN_PATH_2="$ROOT_DIR/.build/$BUILD_CONFIG/afm"

if [ -x "$BIN_PATH_1" ]; then
  FINAL_BIN="$BIN_PATH_1"
elif [ -x "$BIN_PATH_2" ]; then
  FINAL_BIN="$BIN_PATH_2"
else
  log_error "Build finished but afm binary was not found"
  exit 1
fi

if [ "$BUILD_CONFIG" = "release" ]; then
  strip "$FINAL_BIN"
  log_info "Stripped debug symbols"
fi

log_info "Build complete"
echo ""
echo "afm binary: $FINAL_BIN"
echo ""
echo "Example run:"
echo "  $FINAL_BIN mlx --help"
