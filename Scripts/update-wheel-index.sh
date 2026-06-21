#!/usr/bin/env bash
#
# Add a new nightly wheel to the PEP 503 simple index and deploy it.
#
# Usage:
#   ./Scripts/update-wheel-index.sh <wheel-file> <release-tag> [--no-deploy]
#
# Example:
#   ./Scripts/update-wheel-index.sh dist/macafm_next-0.9.13.dev20260621-py3-none-any.whl nightly-20260621-97e6683
#
# The script:
#   1. Uploads the wheel to the GitHub release as an asset
#   2. Appends a PEP 503 link (with #sha256) to the afm-web wheel index
#   3. Commits + pushes the index in the afm-web repo
#   4. Builds the afm-web site and deploys it to Cloudflare Pages (unless --no-deploy)
#
# The index is hosted at https://maclocal-ai.pages.dev/afm/wheels/simple/
# Source of truth: the afm-web repo (Astro site; `public/` is copied verbatim into `dist/`).
#   - Cloudflare Pages project: maclocal-ai (production branch: main)
#   - The OLD vesta-mac / kruks.ai path is DEPRECATED and no longer used.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DEPLOY=true
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --no-deploy) DEPLOY=false ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done
set -- "${POSITIONAL[@]:-}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <wheel-file> <release-tag> [--no-deploy]"
    exit 1
fi

WHL_PATH="$1"
RELEASE_TAG="$2"
WHL_NAME="$(basename "$WHL_PATH")"
REPO="scouzi1966/maclocal-api"
PAGES_PROJECT="maclocal-ai"
INDEX_URL="https://maclocal-ai.pages.dev/afm/wheels/simple/"

# Locate the afm-web site repo (source for maclocal-ai.pages.dev)
AFM_WEB_DIR="${AFM_WEB_DIR:-$REPO_ROOT/../afm-web}"
if [ ! -d "$AFM_WEB_DIR/public/afm/wheels/simple" ]; then
    echo "[ERROR] afm-web repo not found at $AFM_WEB_DIR (expected public/afm/wheels/simple/)"
    echo "[ERROR] Clone scouzi1966/afm-web or set AFM_WEB_DIR=/path/to/afm-web"
    exit 1
fi
echo "[INFO] afm-web site: $AFM_WEB_DIR"

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

# ---------- update the PEP 503 index in afm-web ----------
SIMPLE_DIR="$AFM_WEB_DIR/public/afm/wheels/simple"
INDEX_DIR="$SIMPLE_DIR/macafm-next"
mkdir -p "$INDEX_DIR"

# Root index (lists the package) — create if missing
ROOT_INDEX="$SIMPLE_DIR/index.html"
if [ ! -f "$ROOT_INDEX" ]; then
    cat > "$ROOT_INDEX" << 'INDEXEOF'
<!DOCTYPE html>
<html><body>
<a href="macafm-next/">macafm-next</a>
</body></html>
INDEXEOF
fi

# Package index — create if missing
PKG_INDEX="$INDEX_DIR/index.html"
if [ ! -f "$PKG_INDEX" ]; then
    cat > "$PKG_INDEX" << 'PKGEOF'
<!DOCTYPE html>
<html><body>
</body></html>
PKGEOF
fi

# Drop any existing entry for this exact wheel (idempotent re-runs)
if grep -q "$WHL_NAME" "$PKG_INDEX" 2>/dev/null; then
    echo "[INFO] Wheel already in index, replacing entry..."
    grep -v "$WHL_NAME" "$PKG_INDEX" > "$PKG_INDEX.tmp" || true
    mv "$PKG_INDEX.tmp" "$PKG_INDEX"
fi

# Append the new link before </body> (keep all prior entries so old pins resolve)
sed -i '' "s|</body>|<a href=\"${WHEEL_URL}#sha256=${SHA256}\">${WHL_NAME}</a>\n</body>|" "$PKG_INDEX"
echo "[INFO] Index updated: $PKG_INDEX"

# ---------- commit + push the index in afm-web ----------
cd "$AFM_WEB_DIR"
git add public/afm/wheels/simple/
if git commit -m "Add ${WHL_NAME} to nightly wheel index" 2>/dev/null; then
    git push
    echo "[INFO] afm-web index committed + pushed"
else
    echo "[INFO] No index changes to commit"
fi

# ---------- build + deploy to Cloudflare Pages ----------
if [ "$DEPLOY" = true ]; then
    echo "[INFO] Building afm-web site..."
    npm install --silent
    npm run build
    if [ ! -f "dist/afm/wheels/simple/macafm-next/index.html" ]; then
        echo "[ERROR] Built dist/ is missing the wheel index — aborting deploy"
        exit 1
    fi
    echo "[INFO] Deploying to Cloudflare Pages (project: ${PAGES_PROJECT}, branch: main)..."
    # --branch main is mandatory: it makes this a Production deploy. Without it
    # wrangler infers the current git branch and publishes a useless preview alias.
    if npx wrangler pages deploy dist --project-name "$PAGES_PROJECT" --branch main --commit-dirty=true; then
        echo "[INFO] Deployed to ${INDEX_URL%/}"
    else
        echo "[WARN] wrangler deploy failed (auth? run 'wrangler login'). Index is committed; deploy manually:"
        echo "  cd $AFM_WEB_DIR && npm run build && npx wrangler pages deploy dist --project-name $PAGES_PROJECT --branch main --commit-dirty=true"
        exit 1
    fi
else
    echo "[INFO] --no-deploy: index committed but NOT deployed. Deploy with:"
    echo "  cd $AFM_WEB_DIR && npm run build && npx wrangler pages deploy dist --project-name $PAGES_PROJECT --branch main --commit-dirty=true"
fi

cd "$REPO_ROOT"

echo ""
echo "[INFO] Install with:"
echo "  pip install --extra-index-url ${INDEX_URL} macafm-next"
