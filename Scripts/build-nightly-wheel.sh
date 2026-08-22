#!/usr/bin/env bash
#
# Build a nightly wheel for macafm-next from an existing compiled afm binary.
#
# Usage:
#   ./Scripts/build-nightly-wheel.sh [--version BASE_VERSION]
#
# The wheel is written to dist/macafm_next-VERSION-py3-none-macosx_14_0_arm64.whl
# VERSION defaults to <BuildInfo version>.dev<YYYYMMDD> (PEP 440 dev release).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Nightly wheel metadata is temporary build input, not a source change. Keep
# exact copies outside the checkout and restore them on every exit path so a
# failed build cannot leave the release branch dirty.
METADATA_BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/afm-nightly-wheel.XXXXXX")"
cp pyproject.toml "$METADATA_BACKUP_DIR/pyproject.toml"
cp pyproject-next.toml "$METADATA_BACKUP_DIR/pyproject-next.toml"
cp macafm_next/__init__.py "$METADATA_BACKUP_DIR/macafm-next-init.py"
mkdir "$METADATA_BACKUP_DIR/egg-info"
cp macafm_next.egg-info/* "$METADATA_BACKUP_DIR/egg-info/"

cleanup() {
    cp "$METADATA_BACKUP_DIR/pyproject.toml" pyproject.toml
    cp "$METADATA_BACKUP_DIR/pyproject-next.toml" pyproject-next.toml
    cp "$METADATA_BACKUP_DIR/macafm-next-init.py" macafm_next/__init__.py
    cp "$METADATA_BACKUP_DIR/egg-info/"* macafm_next.egg-info/
    rm -rf "$REPO_ROOT/macafm_next/bin" "$REPO_ROOT/macafm_next/share"
    rm -rf "$METADATA_BACKUP_DIR"
}
trap cleanup EXIT

# ---------- parse args ----------
BASE_VERSION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) BASE_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------- determine version ----------
if [ -z "$BASE_VERSION" ]; then
    BASE_VERSION=$(grep 'static let version' Sources/AFMKit/BuildInfo.swift \
        | sed 's/.*"\(.*\)".*/\1/' | sed 's/^v//')
fi
DATE=$(date -u +%Y%m%d)
DEV_VERSION="${BASE_VERSION}.dev${DATE}"
echo "[INFO] Nightly wheel version: ${DEV_VERSION}"

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
sed -i '' "s/^__version__ = .*/__version__ = \"${DEV_VERSION}\"/" macafm_next/__init__.py
sed -i '' "s/^version = .*/version = \"${DEV_VERSION}\"/" pyproject-next.toml

# ---------- stage assets ----------
echo "[INFO] Staging assets into macafm_next/"
mkdir -p macafm_next/bin
cp "$BIN" macafm_next/bin/
for BUNDLE_NAME in MacLocalAPI_AFMKit.bundle MacLocalAPI_AFMKitMLX.bundle MacLocalAPI_AFMKitDwarfStar.bundle; do
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
# Use pyproject-next.toml by temporarily swapping it in. The EXIT trap restores
# the original even when uv or a later verification command fails.
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

echo "[INFO] Done. Wheel ready: $WHL"
