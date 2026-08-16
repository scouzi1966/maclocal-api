#!/usr/bin/env bash
#
# Build a nightly wheel for macafm-next from an existing compiled afm binary.
#
# Usage:
#   ./Scripts/build-nightly-wheel.sh [--version BASE_VERSION]
#       [--build-version CANONICAL_VERSION] [--python-version PEP440_VERSION]
#
# The wheel is written to dist/macafm_next-VERSION-py3-none-macosx_14_0_arm64.whl
# Python metadata uses a PEP 440 dev version while the bundled command reports
# the canonical Homebrew/nightly display version.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------- parse args ----------
BASE_VERSION=""
BUILD_VERSION=""
PYTHON_VERSION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) BASE_VERSION="$2"; shift 2 ;;
        --build-version) BUILD_VERSION="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------- determine version ----------
if [ -z "$BASE_VERSION" ]; then
    BASE_VERSION=$(grep 'static let version' Sources/AFMKit/BuildInfo.swift \
        | sed 's/.*"\(.*\)".*/\1/' | sed 's/^v//')
fi
if [ -z "$BUILD_VERSION" ]; then
    BUILD_VERSION=$("$REPO_ROOT/Scripts/nightly-version.sh" \
        --base-version "$BASE_VERSION" --field canonical)
fi
if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION=$("$REPO_ROOT/Scripts/nightly-version.sh" \
        --base-version "$BASE_VERSION" --field python)
fi
echo "[INFO] Nightly display version: ${BUILD_VERSION}"
echo "[INFO] Nightly Python version: ${PYTHON_VERSION}"

METADATA_BACKUP="$REPO_ROOT/.build/afm-next-wheel-metadata"
rm -rf "$METADATA_BACKUP"
mkdir -p "$METADATA_BACKUP"
cp macafm_next/__init__.py "$METADATA_BACKUP/__init__.py"
cp pyproject-next.toml "$METADATA_BACKUP/pyproject-next.toml"
cp pyproject.toml "$METADATA_BACKUP/pyproject.toml"
cp -R macafm_next.egg-info "$METADATA_BACKUP/macafm_next.egg-info"

cleanup() {
    cp "$METADATA_BACKUP/__init__.py" macafm_next/__init__.py 2>/dev/null || true
    cp "$METADATA_BACKUP/pyproject-next.toml" pyproject-next.toml 2>/dev/null || true
    cp "$METADATA_BACKUP/pyproject.toml" pyproject.toml 2>/dev/null || true
    cp -R "$METADATA_BACKUP/macafm_next.egg-info/." macafm_next.egg-info/ 2>/dev/null || true
    rm -rf macafm_next/bin macafm_next/share "$WHEEL_SMOKE" "$METADATA_BACKUP"
}
WHEEL_SMOKE=""
trap cleanup EXIT

# ---------- locate binary ----------
BIN=".build/arm64-apple-macosx/release/afm"
[ -x "$BIN" ] || BIN=".build/release/afm"
if [ ! -x "$BIN" ]; then
    echo "[ERROR] No compiled binary found. Run ./Scripts/build-from-scratch.sh first."
    exit 1
fi
echo "[INFO] Binary: $(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

METALLIB="$(dirname "$BIN")/MacLocalAPI_AFMKitMLX.bundle/default.metallib"
if [ ! -f "$METALLIB" ]; then
    METALLIB="$(dirname "$BIN")/MacLocalAPI_AFMKitMLX.bundle/Contents/Resources/default.metallib"
fi
"$REPO_ROOT/Scripts/check-macos26-compatibility.sh" "$BIN" "$METALLIB"

# ---------- set version in package files ----------
sed -i '' "s/^__version__ = .*/__version__ = \"${PYTHON_VERSION}\"/" macafm_next/__init__.py
sed -i '' "s/^__build_version__ = .*/__build_version__ = \"${BUILD_VERSION}\"/" macafm_next/__init__.py
sed -i '' "s/^version = .*/version = \"${PYTHON_VERSION}\"/" pyproject-next.toml

# ---------- stage assets ----------
echo "[INFO] Staging assets into macafm_next/"
mkdir -p macafm_next/bin
cp "$BIN" macafm_next/bin/
for BUNDLE_NAME in MacLocalAPI_AFMKitMLX.bundle MacLocalAPI_AFMKitDwarfStar.bundle; do
    BUNDLE_DIR="$(dirname "$BIN")/$BUNDLE_NAME"
    if [ ! -d "$BUNDLE_DIR" ]; then
        echo "[ERROR] Required runtime bundle missing: $BUNDLE_DIR"
        exit 1
    fi
    cp -R "$BUNDLE_DIR" macafm_next/bin/
    echo "[INFO] Included $BUNDLE_NAME"
done
if [ -f "$METALLIB" ]; then
    cp "$METALLIB" macafm_next/bin/
    echo "[INFO] Included metallib"
fi
"$REPO_ROOT/Scripts/verify-webui.sh" Resources/webui/index.html.gz
mkdir -p macafm_next/share/webui
cp Resources/webui/index.html.gz macafm_next/share/webui/
echo "[INFO] Included webui"

# ---------- build wheel ----------
# Use pyproject-next.toml by temporarily swapping it in.
cp pyproject-next.toml pyproject.toml

echo "[INFO] Building wheel..."
rm -rf dist/macafm_next-*
uv build --wheel 2>&1

# ---------- clean staged assets ----------
rm -rf macafm_next/bin macafm_next/share
echo "[INFO] Cleaned staged assets"

# ---------- verify ----------
WHL=$(ls dist/macafm_next-*.whl 2>/dev/null | head -1)
if [ -z "$WHL" ]; then
    echo "[ERROR] No wheel found in dist/"
    exit 1
fi
WHL_SIZE=$(du -m "$WHL" | cut -f1)
echo "[INFO] Wheel: $WHL (${WHL_SIZE}MB)"
if [ "$WHL_SIZE" -lt 1 ]; then
    echo "[ERROR] Wheel is too small — assets were not staged correctly"
    exit 1
fi

WHEEL_WEBUI="$REPO_ROOT/.build/afm-next-wheel-webui.html.gz"
unzip -p "$WHL" macafm_next/share/webui/index.html.gz > "$WHEEL_WEBUI"
"$REPO_ROOT/Scripts/verify-webui.sh" "$WHEEL_WEBUI"
rm -f "$WHEEL_WEBUI"

WHEEL_SMOKE="$REPO_ROOT/.build/afm-next-wheel-smoke"
rm -rf "$WHEEL_SMOKE"
mkdir -p "$WHEEL_SMOKE"
unzip -q "$WHL" -d "$WHEEL_SMOKE"
WHEEL_METADATA=$(find "$WHEEL_SMOKE" -path '*.dist-info/METADATA' -print -quit)
ACTUAL_PYTHON_VERSION=$(awk '/^Version: / { print $2; exit }' "$WHEEL_METADATA")
if [ "$ACTUAL_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
    echo "[ERROR] Wheel metadata reports '$ACTUAL_PYTHON_VERSION'; expected '$PYTHON_VERSION'"
    exit 1
fi
ACTUAL_VERSION=$(AFM_BUILD_VERSION="v-invalid-host-override" PYTHONPATH="$WHEEL_SMOKE" \
    python3 -c 'from macafm_next.cli import main; main()' --version)
EXPECTED_VERSION="v${BUILD_VERSION#v}"
if [ "$ACTUAL_VERSION" != "$EXPECTED_VERSION" ]; then
    echo "[ERROR] Wheel reports '$ACTUAL_VERSION'; expected '$EXPECTED_VERSION'"
    exit 1
fi
rm -rf "$WHEEL_SMOKE"
echo "[INFO] Verified wheel metadata version: $ACTUAL_PYTHON_VERSION"
echo "[INFO] Verified wheel runtime version: $ACTUAL_VERSION"

echo "[INFO] Done. Wheel ready: $WHL"
