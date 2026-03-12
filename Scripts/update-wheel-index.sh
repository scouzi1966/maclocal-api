#!/usr/bin/env bash
#
# Update the PEP 503 simple index on kruks.ai with a new nightly wheel.
#
# Usage:
#   ./Scripts/update-wheel-index.sh <wheel-file> <release-tag>
#
# Example:
#   ./Scripts/update-wheel-index.sh dist/macafm_next-0.9.7.dev20260312-py3-none-any.whl nightly-20260312-a49c207
#
# The script:
#   1. Uploads the wheel to the GitHub release as an asset
#   2. Adds a link to the PEP 503 index in the vesta-mac DEMO site
#   3. Commits the index update (deploy separately with DEMO/deploy.sh)
#
# The index is hosted at https://kruks.ai/afm/wheels/simple/
# Site source: vesta-mac repo DEMO/<latest-date>/afm/wheels/simple/
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <wheel-file> <release-tag>"
    exit 1
fi

WHL_PATH="$1"
RELEASE_TAG="$2"
WHL_NAME="$(basename "$WHL_PATH")"
REPO="scouzi1966/maclocal-api"

# Locate the vesta-mac DEMO site
VESTA_DIR="${VESTA_DIR:-$REPO_ROOT/../vesta-mac}"
if [ ! -d "$VESTA_DIR/DEMO" ]; then
    echo "[ERROR] vesta-mac repo not found at $VESTA_DIR"
    echo "[ERROR] Clone it or set VESTA_DIR=/path/to/vesta-mac"
    exit 1
fi

# Find the latest dated deployment directory
SITE_DIR=$(ls -d "$VESTA_DIR/DEMO"/20*/ 2>/dev/null | sort | tail -1)
if [ -z "$SITE_DIR" ]; then
    echo "[ERROR] No dated deployment directory found in $VESTA_DIR/DEMO/"
    exit 1
fi
SITE_DIR="${SITE_DIR%/}"
echo "[INFO] Site dir: $SITE_DIR"

if [ ! -f "$WHL_PATH" ]; then
    echo "[ERROR] Wheel not found: $WHL_PATH"
    exit 1
fi

# ---------- upload wheel to GitHub release ----------
echo "[INFO] Uploading wheel to release ${RELEASE_TAG}..."
gh release upload "$RELEASE_TAG" "$WHL_PATH" --repo "$REPO" --clobber

WHEEL_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${WHL_NAME}"
echo "[INFO] Wheel URL: ${WHEEL_URL}"

# ---------- compute sha256 ----------
SHA256=$(shasum -a 256 "$WHL_PATH" | cut -d' ' -f1)
echo "[INFO] SHA256: ${SHA256}"

# ---------- update index in vesta-mac DEMO site ----------
INDEX_DIR="$SITE_DIR/afm/wheels/simple/macafm-next"
mkdir -p "$INDEX_DIR"

# Create root index if missing
ROOT_INDEX="$SITE_DIR/afm/wheels/simple/index.html"
if [ ! -f "$ROOT_INDEX" ]; then
    cat > "$ROOT_INDEX" << 'INDEXEOF'
<!DOCTYPE html>
<html><body>
<a href="macafm-next/">macafm-next</a>
</body></html>
INDEXEOF
fi

# Create or update package index
PKG_INDEX="$INDEX_DIR/index.html"
if [ ! -f "$PKG_INDEX" ]; then
    cat > "$PKG_INDEX" << 'PKGEOF'
<!DOCTYPE html>
<html><body>
</body></html>
PKGEOF
fi

# Remove existing entry for this wheel if present
if grep -q "$WHL_NAME" "$PKG_INDEX" 2>/dev/null; then
    echo "[INFO] Wheel already in index, updating..."
    grep -v "$WHL_NAME" "$PKG_INDEX" > "$PKG_INDEX.tmp" || true
    mv "$PKG_INDEX.tmp" "$PKG_INDEX"
fi

# Insert link before </body>
sed -i '' "s|</body>|<a href=\"${WHEEL_URL}#sha256=${SHA256}\">${WHL_NAME}</a>\n</body>|" "$PKG_INDEX"

# Commit in vesta-mac repo
cd "$VESTA_DIR"
git add "DEMO/$(basename "$SITE_DIR")/afm/wheels/"
git commit -m "Add ${WHL_NAME} to nightly wheel index" 2>/dev/null || {
    echo "[INFO] No changes to commit"
}
cd "$REPO_ROOT"

echo "[INFO] Index updated in vesta-mac DEMO site."
echo "[INFO] Deploy with: cd $VESTA_DIR/DEMO && ./deploy.sh"
echo ""
echo "[INFO] After deploy, install with:"
echo "  pip install --extra-index-url https://kruks.ai/afm/wheels/simple/ macafm-next"
