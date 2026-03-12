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
    BASE_VERSION=$(grep 'static let version' Sources/MacLocalAPI/BuildInfo.swift \
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

METALLIB="$(dirname "$BIN")/MacLocalAPI_MacLocalAPI.bundle/default.metallib"

# ---------- set version in package files ----------
sed -i '' "s/^__version__ = .*/__version__ = \"${DEV_VERSION}\"/" macafm_next/__init__.py
sed -i '' "s/^version = .*/version = \"${DEV_VERSION}\"/" pyproject-next.toml

# ---------- stage assets ----------
echo "[INFO] Staging assets into macafm_next/"
mkdir -p macafm_next/bin
cp "$BIN" macafm_next/bin/
if [ -f "$METALLIB" ]; then
    cp "$METALLIB" macafm_next/bin/
    echo "[INFO] Included metallib"
fi
if [ -f Resources/webui/index.html.gz ]; then
    mkdir -p macafm_next/share/webui
    cp Resources/webui/index.html.gz macafm_next/share/webui/
    echo "[INFO] Included webui"
fi

# ---------- build wheel ----------
# Use pyproject-next.toml by temporarily swapping it in
cp pyproject.toml pyproject.toml.bak
cp pyproject-next.toml pyproject.toml

echo "[INFO] Building wheel..."
rm -rf dist/macafm_next-*
uv build --wheel 2>&1

# Restore original pyproject.toml
mv pyproject.toml.bak pyproject.toml

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

echo "[INFO] Done. Wheel ready: $WHL"
