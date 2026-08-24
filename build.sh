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
#   1) Initialize the remaining consumer-owned submodules (llama.cpp, xgrammar, ...)
#   2) Resolve the immutable AFMKit dependency graph
#   3) Build the llama.cpp webui assets and embed them
#   4) Clean + resolve Swift packages
#   4b) Optionally rebuild AFMKit's MLX Metal library during local AFMKit development
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
INCLUDE_BUILD_COMMIT=true
DO_CLEAN=true
DO_SUBMODULES=true
DO_WEBUI=true
DO_METALLIB=false
ASSUME_YES=false
DO_INSTALL=false
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"

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
  --stable             Build a stable binary without a commit suffix
  --no-clean           Skip clean step before build
  --skip-submodules    Skip git submodule init/update
  --skip-webui         Skip llama.cpp webui build
  --rebuild-metallib   Rebuild default.metallib in MACLOCAL_AFMKIT_PATH
  --skip-metallib      Compatibility alias for the default immutable build
  --yes, -y            Assume "yes" for dependency-install prompts (non-interactive)
  --install            Install afm under INSTALL_PREFIX (default: $INSTALL_PREFIX)
  -h, --help           Show help

Default behavior:
  check deps + submodules + webui + clean + immutable AFMKit + release build
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --debug) BUILD_CONFIG="debug" ;;
    --stable) INCLUDE_BUILD_COMMIT=false ;;
    --no-clean) DO_CLEAN=false ;;
    --skip-submodules) DO_SUBMODULES=false ;;
    --skip-webui) DO_WEBUI=false ;;
    --rebuild-metallib) DO_METALLIB=true ;;
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

# Fail before dependency resolution or compilation if the consumer graph or
# release resource ownership has drifted back across the AFMKit boundary.
"$SCRIPTS_DIR/check-afmkit-consumer-boundary.sh"

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

# Fail before submodule or WebUI work when the private transition dependency is
# inaccessible. The resolver prints the exact local and CI remediation without
# placing credentials in Package.swift, Package.resolved, or logs.
"$SCRIPTS_DIR/resolve-release-dependencies.sh" --check-access

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
    npm ci
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
# Step 4: Resolve AFMKit resources + Swift build
# ---------------------------------------------------------------------------
if $DO_CLEAN; then
  log_step "Cleaning previous Swift build artifacts"
  swift package clean
fi

log_step "Resolving Swift packages"
"$SCRIPTS_DIR/resolve-release-dependencies.sh"

log_step "Validating AFMKit-owned resources"
if ! AFMKIT_SOURCE_METALLIB="$($SCRIPTS_DIR/resolve-afmkit-resource.sh --source)"; then
  log_error "The resolved AFMKit package does not contain its MLX metallib"
  exit 1
fi
log_info "AFMKit MLX metallib: $AFMKIT_SOURCE_METALLIB"

# ---------------------------------------------------------------------------
# Step 4b: Optional AFMKit metallib maintenance
# ---------------------------------------------------------------------------
# Normal builds consume AFMKit's immutable prebuilt resource. Rebuilding is an
# explicit AFMKit-maintainer action and is permitted only with a local AFMKit
# checkout, so a consumer never mutates a resolved package checkout.
if $DO_METALLIB; then
  if [ -z "${MACLOCAL_AFMKIT_PATH:-}" ]; then
    log_error "--rebuild-metallib requires MACLOCAL_AFMKIT_PATH; resolved package checkouts are immutable"
    exit 1
  fi
  AFMKIT_REBUILD_SCRIPT="$MACLOCAL_AFMKIT_PATH/Scripts/rebuild-mlx-metallib.sh"
  if [ ! -x "$AFMKIT_REBUILD_SCRIPT" ]; then
    log_error "AFMKit metallib build tool not found: $AFMKIT_REBUILD_SCRIPT"
    exit 1
  fi
  log_step "Rebuilding the local AFMKit MLX metallib"
  "$AFMKIT_REBUILD_SCRIPT"
else
  log_info "Using AFMKit's immutable prebuilt metallib"
fi

BUILDINFO="$ROOT_DIR/Sources/AFMKit/BuildInfo.swift"
BUILDINFO_BACKUP=""
restore_buildinfo() {
  if [ -n "$BUILDINFO_BACKUP" ] && [ -f "$BUILDINFO_BACKUP" ]; then
    cp "$BUILDINFO_BACKUP" "$BUILDINFO"
    rm -f "$BUILDINFO_BACKUP"
  fi
}

if $INCLUDE_BUILD_COMMIT; then
  log_step "Injecting build commit into BuildInfo.swift"
  BUILD_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  if [ -f "$BUILDINFO" ]; then
    BUILDINFO_BACKUP="$ROOT_DIR/.build/BuildInfo.swift.pre-build"
    cp "$BUILDINFO" "$BUILDINFO_BACKUP"
    trap restore_buildinfo EXIT
    sed -i '' "s/static let commit: String? = nil/static let commit: String? = \"${BUILD_COMMIT}\"/" "$BUILDINFO"
    log_info "Commit: $BUILD_COMMIT"
  fi
else
  log_step "Building stable version without commit suffix"
fi

log_step "Building afm ($BUILD_CONFIG)"
# Disable MemberImportVisibility — async-kit (transitive from Vapor) is missing
# explicit imports for DequeModule/OrderedCollections, which Swift 6 enforces.
if [ "$BUILD_CONFIG" = "release" ]; then
  "$SCRIPTS_DIR/swiftpm-reliable.sh" build -c release \
    --product afm \
    -Xswiftc -disable-upcoming-feature \
    -Xswiftc MemberImportVisibility
else
  "$SCRIPTS_DIR/swiftpm-reliable.sh" build -c "$BUILD_CONFIG" \
    -Xswiftc -disable-upcoming-feature \
    -Xswiftc MemberImportVisibility
fi

if ! FINAL_BIN="$($SCRIPTS_DIR/find-afm-binary.sh "$BUILD_CONFIG")"; then
  log_error "Build finished but afm binary was not found"
  exit 1
fi

"$SCRIPTS_DIR/check-tree-sitter-highlighting.sh" "$FINAL_BIN"

if [ "$BUILD_CONFIG" = "release" ]; then
  strip "$FINAL_BIN"
  log_info "Stripped debug symbols"
fi

# Restore the exact pre-build file, including legitimate local version edits.
restore_buildinfo
trap - EXIT

FINAL_DIR="$(dirname "$FINAL_BIN")"

# The AFM evaluation host owns the bundled, no-judge evaluation suites. Keep
# its SwiftPM resource bundle beside the executable in every install layout.
EVAL_BUNDLE_DIR="$FINAL_DIR/MacLocalAPI_AFMEvaluationHost.bundle"
if [ -f "$EVAL_BUNDLE_DIR/Evals/comprehensive.json" ]; then
  EVAL_SUITE="$EVAL_BUNDLE_DIR/Evals/comprehensive.json"
elif [ -f "$EVAL_BUNDLE_DIR/Contents/Resources/Evals/comprehensive.json" ]; then
  EVAL_SUITE="$EVAL_BUNDLE_DIR/Contents/Resources/Evals/comprehensive.json"
else
  log_error "Missing bundled evaluation suite under: $EVAL_BUNDLE_DIR"
  exit 1
fi
log_info "Bundled evaluation suite OK: $EVAL_SUITE"

# Verify the MLX metallib resource bundle is present. SwiftPM uses a flat bundle
# with some toolchains and a macOS Contents/Resources bundle with Xcode 27.
if METALLIB_BUNDLE_DIR="$($SCRIPTS_DIR/resolve-afmkit-resource.sh --bundle-dir "$FINAL_DIR")" &&
   METALLIB_BUNDLE="$($SCRIPTS_DIR/resolve-afmkit-resource.sh --metallib "$FINAL_DIR")"; then
  log_info "MLX metallib bundle OK ($(du -h "$METALLIB_BUNDLE" | cut -f1 | xargs))"
else
  log_error "Missing AFMKit MLX resource bundle beside: $FINAL_BIN"
  exit 1
fi

# The selected Xcode may be newer than the package deployment target. Verify
# both the executable and embedded MLX shaders before anything is packaged.
"$SCRIPTS_DIR/check-macos26-compatibility.sh" "$FINAL_BIN" "$METALLIB_BUNDLE"

# Verify Info.plist is embedded in the binary's __TEXT,__info_plist section.
# Without this, macOS 26 SIGABRTs any process that requests privacy-sensitive APIs
# (Speech Recognition, microphone, camera, etc.) — the Speech transcription feature
# and any future privacy-API integration will crash on first use.
# The linker flags in Package.swift (-Xlinker -sectcreate -Xlinker __TEXT
# -Xlinker __info_plist -Xlinker Sources/AFMCLI/Info.plist) must be preserved.
INFO_PLIST_SECTION=$(otool -l "$FINAL_BIN" 2>/dev/null | grep -A2 '__info_plist' | head -3)
if echo "$INFO_PLIST_SECTION" | grep -q '__info_plist'; then
  # No `grep -q` here: -q closes the pipe on first hit, which SIGPIPEs `strings` (exit 141)
  # and — under `set -o pipefail` — fails the check even though the key IS present.
  if [ "$(strings "$FINAL_BIN" | grep -c 'NSSpeechRecognitionUsageDescription')" -gt 0 ]; then
    log_info "Info.plist embedded OK (NSSpeechRecognitionUsageDescription present)"
  else
    log_error "Info.plist section present but NSSpeechRecognitionUsageDescription key is missing"
    log_error "Check Sources/AFMCLI/Info.plist — required for Apple Speech Recognition"
    exit 1
  fi
else
  log_error "Missing __TEXT,__info_plist section in binary"
  log_error "Check Package.swift linker flags and Sources/AFMCLI/Info.plist exists"
  log_error "macOS 26 SIGABRTs any process that calls privacy-sensitive APIs without Info.plist"
  exit 1
fi

# Make metallib discoverable for `swift test` after a build.
# MLX framework searches CWD for "default.metallib" as its last resort.
# A symlink at the project root ensures `swift test` (which runs from project root) finds it.
# Point at the bundle that was ACTUALLY built ($METALLIB_BUNDLE is config-aware via $FINAL_DIR),
# not a hardcoded release path — a `--debug` build has no release bundle in a clean checkout.
ln -sf "$METALLIB_BUNDLE" "$ROOT_DIR/default.metallib"
log_info "Metallib available for swift test (symlink -> $BUILD_CONFIG bundle)"

# ---------------------------------------------------------------------------
# Step 6 (optional): Install to /usr/local
# ---------------------------------------------------------------------------
# /usr/local/bin is the first entry in macOS's /etc/paths, so it's on PATH by
# default for every shell — no profile edits needed. On Apple Silicon it does
# not collide with Homebrew (which lives in /opt/homebrew). The directory is
# root-owned, so writes escalate with sudo only when it isn't already writable.
if $DO_INSTALL; then
  log_step "Installing afm to $INSTALL_PREFIX/bin"
  EVAL_BUNDLE_SRC="$EVAL_BUNDLE_DIR"
  WEBUI_SRC="$ROOT_DIR/Resources/webui/index.html.gz"

  INSTALL_PERMISSION_PROBE="$INSTALL_PREFIX/bin"
  while [ ! -e "$INSTALL_PERMISSION_PROBE" ]; do
    PARENT_DIR="$(dirname "$INSTALL_PERMISSION_PROBE")"
    if [ "$PARENT_DIR" = "$INSTALL_PERMISSION_PROBE" ]; then
      break
    fi
    INSTALL_PERMISSION_PROBE="$PARENT_DIR"
  done

  USE_SUDO=false
  if [ ! -w "$INSTALL_PERMISSION_PROBE" ]; then
    USE_SUDO=true
    log_warn "$INSTALL_PREFIX is not writable — using sudo (you may be prompted for your password)"
  fi

  run_install_command() {
    if $USE_SUDO; then
      sudo "$@"
    else
      "$@"
    fi
  }

  run_install_command install -d "$INSTALL_PREFIX/bin" "$INSTALL_PREFIX/libexec/afm" "$INSTALL_PREFIX/share/afm/webui"
  run_install_command install -m 755 "$FINAL_BIN" "$INSTALL_PREFIX/bin/afm"

  # Provider resources must remain beside the relocated executable. Keep both
  # immutable SwiftPM bundles in libexec and expose sibling symlinks, matching
  # the Homebrew, tarball, and wheel layouts.
  for BUNDLE_NAME in AFMKit_AFMKitMLX.bundle AFMKit_AFMKitDwarfStar.bundle; do
    BUNDLE_SRC="$FINAL_DIR/$BUNDLE_NAME"
    if [ ! -d "$BUNDLE_SRC" ]; then
      log_error "Required AFMKit provider bundle missing: $BUNDLE_SRC"
      exit 1
    fi
    run_install_command rm -rf "$INSTALL_PREFIX/libexec/afm/$BUNDLE_NAME"
    run_install_command cp -R "$BUNDLE_SRC" "$INSTALL_PREFIX/libexec/afm/$BUNDLE_NAME"
    run_install_command rm -rf "$INSTALL_PREFIX/bin/$BUNDLE_NAME"
    run_install_command ln -sfn "$INSTALL_PREFIX/libexec/afm/$BUNDLE_NAME" \
      "$INSTALL_PREFIX/bin/$BUNDLE_NAME"
  done

  run_install_command rm -rf "$INSTALL_PREFIX/libexec/afm/MacLocalAPI_AFMEvaluationHost.bundle"
  run_install_command cp -R "$EVAL_BUNDLE_SRC" "$INSTALL_PREFIX/libexec/afm/MacLocalAPI_AFMEvaluationHost.bundle"
  run_install_command ln -sfn "$INSTALL_PREFIX/libexec/afm/MacLocalAPI_AFMEvaluationHost.bundle" \
    "$INSTALL_PREFIX/bin/MacLocalAPI_AFMEvaluationHost.bundle"

  if [ -f "$WEBUI_SRC" ]; then
    run_install_command install -m 644 "$WEBUI_SRC" "$INSTALL_PREFIX/share/afm/webui/index.html.gz"
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
