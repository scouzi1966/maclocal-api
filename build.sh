#!/bin/bash
# Build AFM (Apple Foundation Models / MLX OpenAI-compatible server) from a fresh clone.
#
# This script is self-contained: a user can `git clone` the repo and run `./build.sh`
# with no prior knowledge of the project. It checks every dependency, installs what it
# safely can from the command line, and prints clear instructions for anything that
# requires manual action (e.g. the Xcode Command Line Tools GUI installer).
#
# Steps:
#   0) Verify / install toolchain dependencies (git, Swift/Xcode CLT, Node + npm)
#   1) Initialize git submodules (mlx-swift-lm, llama.cpp, ...)
#   2) Apply the MLX + xgrammar patch sets (Scripts/patches)
#   3) Build the llama.cpp webui assets and embed them
#   4) Clean + resolve Swift packages
#   4b) Rebuild the MLX Metal shader library (default.metallib) from the kernel sources
#   5) Build the `afm` binary (release by default) and verify the artifact
#
# Usage:
#   ./build.sh                     # full build from a clean clone
#   ./build.sh --debug             # debug build instead of release
#   ./build.sh --no-clean          # skip `swift package clean`
#   ./build.sh --skip-webui        # skip the npm webui build
#   ./build.sh --yes               # assume "yes" to dependency install prompts (CI)
#   ./build.sh --help              # show all options

set -euo pipefail

# ROOT_DIR is the directory containing this script (the repo root).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$ROOT_DIR/Scripts"

BUILD_CONFIG="release"
DO_CLEAN=true
DO_SUBMODULES=true
DO_PATCHES=true
DO_WEBUI=true
DO_METALLIB=true
ASSUME_YES=false
DO_INSTALL=false
INSTALL_PREFIX="/usr/local"

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
Usage: ./build.sh [options]

Options:
  --debug              Build debug instead of release
  --no-clean           Skip clean step before build
  --skip-submodules    Skip git submodule init/update
  --skip-patches       Skip MLX + xgrammar patch application
  --skip-webui         Skip llama.cpp webui build
  --skip-metallib      Skip rebuilding default.metallib (use the committed prebuilt one)
  --yes, -y            Assume "yes" for dependency-install prompts (non-interactive)
  --install            After building, install afm to $INSTALL_PREFIX/bin (uses sudo if needed)
  -h, --help           Show help

Default behavior:
  check deps + submodules + patches + webui + clean + metallib + release build
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --debug) BUILD_CONFIG="debug" ;;
    --no-clean) DO_CLEAN=false ;;
    --skip-submodules) DO_SUBMODULES=false ;;
    --skip-patches) DO_PATCHES=false ;;
    --skip-webui) DO_WEBUI=false ;;
    --skip-metallib) DO_METALLIB=false ;;
    --yes|-y) ASSUME_YES=true ;;
    --install) DO_INSTALL=true ;;
    -h|--help) usage; exit 0 ;;
    *)
      log_error "Unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

# Prompt the user for a yes/no decision. Honors --yes (always yes) and
# non-interactive stdin (defaults to no so we never hang a CI pipeline).
confirm() {
  local prompt="$1"
  if $ASSUME_YES; then
    return 0
  fi
  if [ ! -t 0 ]; then
    return 1
  fi
  local reply
  read -r -p "$prompt [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Step 0: Dependency verification + best-effort install
# ---------------------------------------------------------------------------
log_step "Checking build dependencies"

# This build only targets Apple Silicon macOS.
if [ "$(uname -s)" != "Darwin" ]; then
  log_error "AFM builds only on macOS (Apple Silicon). Detected: $(uname -s)"
  exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
  log_warn "AFM targets Apple Silicon (arm64). Detected arch: $(uname -m). Continuing anyway."
fi

# Xcode Command Line Tools provide both `git` and `swift`. The installer is a
# macOS GUI dialog and cannot be driven from a script, so we trigger it and stop.
ensure_xcode_clt() {
  if xcode-select -p >/dev/null 2>&1 && command -v swift >/dev/null 2>&1; then
    log_info "Swift toolchain found: $(swift --version 2>/dev/null | head -1)"
    return 0
  fi

  log_error "Swift toolchain / Xcode Command Line Tools not found."
  log_warn  "These cannot be installed non-interactively (Apple ships a GUI installer)."
  if confirm "Launch the Xcode Command Line Tools installer now?"; then
    xcode-select --install || true
    log_warn "A macOS dialog should appear. Finish the install, then re-run ./build.sh"
  else
    log_warn "Install manually with:  xcode-select --install"
    log_warn "Then re-run ./build.sh"
  fi
  exit 1
}

# Homebrew is how we install Node when it's missing. Its installer is an
# interactive curl|bash that may prompt for sudo, so we ask before running it.
ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi
  log_warn "Homebrew not found (needed to auto-install Node/npm)."
  if confirm "Install Homebrew now? (runs the official install script)"; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Make brew available in the current shell for both Apple Silicon and Intel.
    if [ -x /opt/homebrew/bin/brew ]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
      eval "$(/usr/local/bin/brew shellenv)"
    fi
  else
    return 1
  fi
}

# Node + npm build the webui. If missing, try Homebrew; otherwise instruct.
ensure_node() {
  if command -v npm >/dev/null 2>&1 && command -v node >/dev/null 2>&1; then
    log_info "Node found: $(node --version)  npm: $(npm --version)"
    return 0
  fi

  if ! $DO_WEBUI; then
    log_warn "Node/npm missing, but --skip-webui was set. Continuing without it."
    return 0
  fi

  log_warn "Node.js / npm not found (required to build the webui)."
  if ensure_homebrew && confirm "Install Node via 'brew install node'?"; then
    brew install node
  else
    log_error "Cannot build the webui without Node.js."
    log_warn  "Install Node manually (https://nodejs.org or 'brew install node'),"
    log_warn  "or re-run with --skip-webui to skip the webui build."
    exit 1
  fi
}

ensure_xcode_clt
ensure_node

# git ships with the Command Line Tools, so this should always pass by now.
if ! command -v git >/dev/null 2>&1; then
  log_error "git not found even after Command Line Tools check. Install Xcode CLT and retry."
  exit 1
fi

if [ ! -f "$ROOT_DIR/Package.swift" ]; then
  log_error "Package.swift not found in $ROOT_DIR — is this a maclocal-api clone?"
  exit 1
fi

cd "$ROOT_DIR"

log_info "Repository: $ROOT_DIR"
log_info "Build configuration: $BUILD_CONFIG"

# ---------------------------------------------------------------------------
# Step 1: Submodules
# ---------------------------------------------------------------------------
if $DO_SUBMODULES; then
  log_step "Initializing submodules"
  git submodule update --init --recursive
  git submodule status || true
else
  log_warn "Skipping submodule initialization"
fi

# ---------------------------------------------------------------------------
# Step 2: Patches
# ---------------------------------------------------------------------------
if $DO_PATCHES; then
  log_step "Applying MLX patch set"
  if [ ! -x "$SCRIPTS_DIR/apply-mlx-patches.sh" ]; then
    log_error "Missing patch script: $SCRIPTS_DIR/apply-mlx-patches.sh"
    exit 1
  fi
  "$SCRIPTS_DIR/apply-mlx-patches.sh"
  "$SCRIPTS_DIR/apply-mlx-patches.sh" --check
  "$SCRIPTS_DIR/patches/apply-xgrammar-patches.sh"
else
  log_warn "Skipping MLX patch application"
fi

# ---------------------------------------------------------------------------
# Step 3: WebUI
# ---------------------------------------------------------------------------
if $DO_WEBUI; then
  log_step "Building llama.cpp webui"
  WEBUI_DIR="$ROOT_DIR/vendor/llama.cpp/tools/server/webui"
  if [ ! -d "$WEBUI_DIR" ]; then
    log_error "webui source not found: $WEBUI_DIR"
    log_error "Did submodules initialize correctly?"
    exit 1
  fi
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

# ---------------------------------------------------------------------------
# Step 4: Resource validation + Swift build
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Step 4a: Apply MLX C++ / Metal-kernel patches to the resolved mlx-swift checkout
# ---------------------------------------------------------------------------
# These patch the EPHEMERAL .build/checkouts/mlx-swift C++ tree (wiped by clean/re-resolve),
# so they must run AFTER `swift package resolve` and BEFORE the metallib rebuild (which compiles
# the patched kernels) and `swift build` (which compiles the patched dispatch C++).
#   - apply-mlx-qmv-wide-backport.sh : mlx#3764 qmv_wide small-batch matvec (spec-decode verify,
#                                      batch B=2-8). FULL-FILE replacement — MUST run before
#                                      apply-mlx-cpp-patches.sh, which edits the same files.
#   - apply-mlx-cpp-patches.sh    : qmv_fast_wide quantized matvec kernels
#   - apply-mlx-sdpa-backport.sh  : 0.31.3 adaptive-block SDPA (decode@16k ~+10%, correct)
if $DO_PATCHES; then
  if [ -x "$SCRIPTS_DIR/apply-mlx-qmv-wide-backport.sh" ]; then
    log_step "Applying MLX qmv_wide backport (mlx#3764)"
    "$SCRIPTS_DIR/apply-mlx-qmv-wide-backport.sh"
  fi
  if [ -x "$SCRIPTS_DIR/apply-mlx-cpp-patches.sh" ]; then
    log_step "Applying MLX C++ kernel patches (qmv_fast_wide)"
    "$SCRIPTS_DIR/apply-mlx-cpp-patches.sh"
  fi
  if [ -x "$SCRIPTS_DIR/apply-mlx-sdpa-backport.sh" ]; then
    log_step "Applying MLX SDPA 0.31.3 adaptive-block backport"
    "$SCRIPTS_DIR/apply-mlx-sdpa-backport.sh"
  fi
else
  log_warn "Skipping MLX C++ / SDPA patches (--skip-patches)"
fi

# ---------------------------------------------------------------------------
# Step 4b: Rebuild the MLX Metal shader library (default.metallib) from source
# ---------------------------------------------------------------------------
# IMPORTANT: `swift build` does NOT compile any Metal — it only copies the committed
# Sources/MacLocalAPI/Resources/default.metallib into the app bundle. The kernel *sources*
# live in the resolved mlx-swift dependency (.build/checkouts/mlx-swift), so this step
# regenerates that binary from source — ensuring the shipped kernels actually match the
# (possibly patched) kernel tree rather than a stale prebuilt blob.
#
# Requires the Metal Toolchain, which Xcode 26 ships as a SEPARATE downloadable component.
# If it isn't installed we fall back to the committed prebuilt metallib (offering to
# download the toolchain first). Must run AFTER `swift package resolve` (needs the sources)
# and BEFORE `swift build` (which copies the result into the bundle).
if $DO_METALLIB; then
  log_step "Rebuilding MLX metallib from kernel sources"
  if "$SCRIPTS_DIR/rebuild-metallib.sh" --check >/dev/null 2>&1; then
    "$SCRIPTS_DIR/rebuild-metallib.sh"
  else
    log_warn "Metal Toolchain not installed — cannot compile the metallib from source."
    if confirm "Download the Metal Toolchain now? (~688 MB one-time: xcodebuild -downloadComponent MetalToolchain)"; then
      xcodebuild -downloadComponent MetalToolchain
      "$SCRIPTS_DIR/rebuild-metallib.sh"
    else
      log_warn "Falling back to the committed prebuilt metallib (Sources/MacLocalAPI/Resources/default.metallib)."
      log_warn "To build it from source later: xcodebuild -downloadComponent MetalToolchain && ./Scripts/rebuild-metallib.sh"
    fi
  fi
else
  log_warn "Skipping metallib rebuild (--skip-metallib): using committed prebuilt metallib"
fi

log_step "Injecting build commit into BuildInfo.swift"
BUILD_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILDINFO="$ROOT_DIR/Sources/MacLocalAPI/BuildInfo.swift"
if [ -f "$BUILDINFO" ]; then
  sed -i '' "s/static let commit: String? = nil/static let commit: String? = \"${BUILD_COMMIT}\"/" "$BUILDINFO"
  log_info "Commit: $BUILD_COMMIT"
fi

log_step "Building afm ($BUILD_CONFIG)"
# Disable MemberImportVisibility — async-kit (transitive from Vapor) is missing
# explicit imports for DequeModule/OrderedCollections, which Swift 6 enforces.
if [ "$BUILD_CONFIG" = "release" ]; then
  swift build -c release \
    --product afm \
    -Xswiftc -O \
    -Xswiftc -whole-module-optimization \
    -Xswiftc -cross-module-optimization \
    -Xswiftc -disable-upcoming-feature \
    -Xswiftc MemberImportVisibility
else
  swift build -c "$BUILD_CONFIG" \
    -Xswiftc -disable-upcoming-feature \
    -Xswiftc MemberImportVisibility
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

# Restore BuildInfo.swift to committed state (keep working tree clean)
if [ -f "$BUILDINFO" ]; then
  git checkout -- "$BUILDINFO" 2>/dev/null || true
fi

FINAL_DIR="$(dirname "$FINAL_BIN")"

# Verify the MLX metallib resource bundle is present
METALLIB_BUNDLE="$FINAL_DIR/MacLocalAPI_MacLocalAPI.bundle/default.metallib"
if [ -f "$METALLIB_BUNDLE" ]; then
  log_info "MLX metallib bundle OK ($(du -h "$METALLIB_BUNDLE" | cut -f1 | xargs))"
else
  log_error "Missing MLX metallib bundle: $METALLIB_BUNDLE"
  exit 1
fi

# Verify Info.plist is embedded in the binary's __TEXT,__info_plist section.
# Without this, macOS 26 SIGABRTs any process that requests privacy-sensitive APIs
# (Speech Recognition, microphone, camera, etc.) — the Speech transcription feature
# and any future privacy-API integration will crash on first use.
# The linker flags in Package.swift (-Xlinker -sectcreate -Xlinker __TEXT
# -Xlinker __info_plist -Xlinker Sources/MacLocalAPI/Info.plist) must be preserved.
INFO_PLIST_SECTION=$(otool -l "$FINAL_BIN" 2>/dev/null | grep -A2 '__info_plist' | head -3)
if echo "$INFO_PLIST_SECTION" | grep -q '__info_plist'; then
  # No `grep -q` here: -q closes the pipe on first hit, which SIGPIPEs `strings` (exit 141)
  # and — under `set -o pipefail` — fails the check even though the key IS present.
  if [ "$(strings "$FINAL_BIN" | grep -c 'NSSpeechRecognitionUsageDescription')" -gt 0 ]; then
    log_info "Info.plist embedded OK (NSSpeechRecognitionUsageDescription present)"
  else
    log_error "Info.plist section present but NSSpeechRecognitionUsageDescription key is missing"
    log_error "Check Sources/MacLocalAPI/Info.plist — required for Apple Speech Recognition"
    exit 1
  fi
else
  log_error "Missing __TEXT,__info_plist section in binary"
  log_error "Check Package.swift linker flags and Sources/MacLocalAPI/Info.plist exists"
  log_error "macOS 26 SIGABRTs any process that calls privacy-sensitive APIs without Info.plist"
  exit 1
fi

# Make metallib discoverable for `swift test` after a build.
# MLX framework searches CWD for "default.metallib" as its last resort.
# A symlink at the project root ensures `swift test` (which runs from project root) finds it.
# Point at the bundle that was ACTUALLY built ($METALLIB_BUNDLE is config-aware via $FINAL_DIR),
# not a hardcoded release path — a `--debug` build has no release bundle in a clean checkout.
ln -sf "$METALLIB_BUNDLE" "$ROOT_DIR/default.metallib"
# Also mirror into the OTHER config's bundle so `swift test` works regardless of which config
# the tester uses (debug ↔ release), for our own MLXMetalLibrary resolver.
if [ "$BUILD_CONFIG" = "release" ]; then
  OTHER_BUNDLE="$ROOT_DIR/.build/arm64-apple-macosx/debug/MacLocalAPI_MacLocalAPI.bundle"
else
  OTHER_BUNDLE="$ROOT_DIR/.build/arm64-apple-macosx/release/MacLocalAPI_MacLocalAPI.bundle"
fi
mkdir -p "$OTHER_BUNDLE"
cp "$METALLIB_BUNDLE" "$OTHER_BUNDLE/default.metallib"
log_info "Metallib available for swift test (symlink -> $BUILD_CONFIG bundle + mirror)"

# ---------------------------------------------------------------------------
# Step 6 (optional): Install to /usr/local
# ---------------------------------------------------------------------------
# /usr/local/bin is the first entry in macOS's /etc/paths, so it's on PATH by
# default for every shell — no profile edits needed. On Apple Silicon it does
# not collide with Homebrew (which lives in /opt/homebrew). The directory is
# root-owned, so writes escalate with sudo only when it isn't already writable.
if $DO_INSTALL; then
  log_step "Installing afm to $INSTALL_PREFIX/bin"
  BUNDLE_SRC="$FINAL_DIR/MacLocalAPI_MacLocalAPI.bundle"
  WEBUI_SRC="$ROOT_DIR/Resources/webui/index.html.gz"

  SUDO=""
  if [ ! -w "$INSTALL_PREFIX" ] || [ ! -w "$INSTALL_PREFIX/bin" ]; then
    SUDO="sudo"
    log_warn "$INSTALL_PREFIX is not writable — using sudo (you may be prompted for your password)"
  fi

  $SUDO install -d "$INSTALL_PREFIX/bin" "$INSTALL_PREFIX/libexec/afm" "$INSTALL_PREFIX/share/afm/webui"
  $SUDO install -m 755 "$FINAL_BIN" "$INSTALL_PREFIX/bin/afm"

  # afm resolves its Metal shader library as a sibling bundle of the binary.
  # Keep the bundle in libexec and symlink it next to the binary — this mirrors
  # the Homebrew formula and avoids macOS code-signing stripping a bundle placed
  # directly in bin.
  $SUDO rm -rf "$INSTALL_PREFIX/libexec/afm/MacLocalAPI_MacLocalAPI.bundle"
  $SUDO cp -R "$BUNDLE_SRC" "$INSTALL_PREFIX/libexec/afm/MacLocalAPI_MacLocalAPI.bundle"
  $SUDO ln -sfn "$INSTALL_PREFIX/libexec/afm/MacLocalAPI_MacLocalAPI.bundle" \
    "$INSTALL_PREFIX/bin/MacLocalAPI_MacLocalAPI.bundle"

  if [ -f "$WEBUI_SRC" ]; then
    $SUDO install -m 644 "$WEBUI_SRC" "$INSTALL_PREFIX/share/afm/webui/index.html.gz"
  fi

  log_info "Installed: $INSTALL_PREFIX/bin/afm"
fi

log_info "Build complete"
echo ""

# Always report the built binary's full path — and fail loudly if it isn't
# where we expect it.
if [ -x "$FINAL_BIN" ]; then
  log_info "afm binary: $FINAL_BIN"
else
  log_error "Expected built binary not found or not executable: $FINAL_BIN"
  exit 1
fi

# When installing, report the installed path too — and fail if the install
# didn't land where expected.
if $DO_INSTALL; then
  INSTALLED_BIN="$INSTALL_PREFIX/bin/afm"
  if [ -x "$INSTALLED_BIN" ]; then
    log_info "Installed:  $INSTALLED_BIN"
  else
    log_error "Install step ran but afm is not where expected: $INSTALLED_BIN"
    exit 1
  fi
fi

echo ""
echo "Example run:"
if $DO_INSTALL; then
  echo "  afm mlx --help"
else
  echo "  $FINAL_BIN mlx --help"
fi
