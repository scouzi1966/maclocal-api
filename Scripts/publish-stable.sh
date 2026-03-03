#!/bin/bash
# Publish stable afm release: package, upload GitHub release, update tap, bump pyproject
#
# Usage:
#   ./Scripts/publish-stable.sh <version>                  # full build + publish
#   ./Scripts/publish-stable.sh <version> --skip-build     # package + publish (assumes already built)
#
# Example:
#   ./Scripts/publish-stable.sh 0.9.5 --skip-build
#
# Prerequisites:
#   - gh CLI authenticated with push access to scouzi1966/maclocal-api
#   - homebrew-afm repo cloned at ../homebrew-afm (or set TAP_DIR env var)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TAP_DIR="${TAP_DIR:-$ROOT_DIR/../homebrew-afm}"
REPO="scouzi1966/maclocal-api"
DO_BUILD=true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
VERSION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) DO_BUILD=false ;;
    -h|--help)
      echo "Usage: ./Scripts/publish-stable.sh <version> [--skip-build]"
      echo ""
      echo "Arguments:"
      echo "  <version>           Version number (e.g. 0.9.5)"
      echo ""
      echo "Options:"
      echo "  --skip-build        Package + publish (assumes already built)"
      exit 0
      ;;
    *)
      if [ -z "$VERSION" ]; then
        VERSION="$1"
      else
        log_error "Unknown option: $1"
        exit 1
      fi
      ;;
  esac
  shift
done

if [ -z "$VERSION" ]; then
  log_error "Version required. Usage: ./Scripts/publish-stable.sh <version> [--skip-build]"
  exit 1
fi

TAG="v${VERSION}"

cd "$ROOT_DIR"

# Verify prerequisites
if ! command -v gh >/dev/null 2>&1; then
  log_error "gh CLI not found. Install with: brew install gh"
  exit 1
fi

if [ ! -d "$TAP_DIR/.git" ]; then
  log_error "Tap repo not found at $TAP_DIR"
  log_error "Clone it: gh repo clone scouzi1966/homebrew-afm $TAP_DIR"
  log_error "Or set TAP_DIR=/path/to/homebrew-afm"
  exit 1
fi

if [ ! -f "$TAP_DIR/afm.rb" ]; then
  log_error "afm.rb not found in $TAP_DIR"
  exit 1
fi

# Check if tag already exists
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  log_error "Release $TAG already exists. Delete it first if you want to recreate."
  exit 1
fi

SHORT_SHA=$(git rev-parse --short HEAD)

log_info "Publishing stable release"
log_info "  Version: ${VERSION}"
log_info "  Tag: ${TAG}"
log_info "  Commit: ${SHORT_SHA}"
log_info "  Tap: ${TAP_DIR}"

# Step 1: Build
if $DO_BUILD; then
  log_info "Running build-from-scratch.sh..."
  "$SCRIPT_DIR/build-from-scratch.sh"
else
  log_warn "Skipping build (--skip-build)"
fi

# Step 2: Find binary
BIN="$ROOT_DIR/.build/arm64-apple-macosx/release/afm"
if [ ! -x "$BIN" ]; then
  BIN="$ROOT_DIR/.build/release/afm"
fi
if [ ! -x "$BIN" ]; then
  log_error "afm binary not found. Run without --skip-build."
  exit 1
fi
log_info "Binary: $BIN"

# Step 3: Package
log_info "Creating release package..."
STAGING="$ROOT_DIR/.build/release-package-stable"
rm -rf "$STAGING"
mkdir -p "$STAGING"

cp "$BIN" "$STAGING/"

# Metallib resource bundle
BUNDLE_DIR="$(dirname "$BIN")/MacLocalAPI_MacLocalAPI.bundle"
if [ -d "$BUNDLE_DIR" ]; then
  cp -r "$BUNDLE_DIR" "$STAGING/"
  log_info "Included metallib bundle"
fi

# WebUI
if [ -f "$ROOT_DIR/Resources/webui/index.html.gz" ]; then
  mkdir -p "$STAGING/Resources/webui"
  cp "$ROOT_DIR/Resources/webui/index.html.gz" "$STAGING/Resources/webui/"
  log_info "Included webui"
fi

cp "$ROOT_DIR/README.md" "$STAGING/" 2>/dev/null || true
cp "$ROOT_DIR/LICENSE" "$STAGING/" 2>/dev/null || true

TARBALL="$ROOT_DIR/afm-${TAG}-arm64.tar.gz"
tar -czf "$TARBALL" -C "$STAGING" .
log_info "Tarball: $TARBALL ($(du -h "$TARBALL" | cut -f1 | xargs))"

# Step 4: Generate changelog (since last stable tag)
log_info "Generating changelog..."
PREV_TAG=$(gh release list --repo "$REPO" --limit 20 --json tagName -q '.[].tagName' 2>/dev/null \
  | grep '^v[0-9]' | head -1) || true

if [ -n "$PREV_TAG" ] && git cat-file -e "${PREV_TAG}^{commit}" 2>/dev/null; then
  CHANGELOG=$(git log --pretty=format:"- %s (\`%h\`)" "${PREV_TAG}..HEAD" -- . ':!vendor' 2>/dev/null)
  if [ -z "$CHANGELOG" ]; then
    CHANGELOG="- No changes since ${PREV_TAG}"
  fi
  log_info "Changelog since: $PREV_TAG"
else
  CHANGELOG=$(git log --pretty=format:"- %s (\`%h\`)" -20 -- . ':!vendor' 2>/dev/null)
  log_warn "No previous stable tag found, showing last 20 commits"
fi

# Step 5: Upload to GitHub release
log_info "Creating release: $TAG"
gh release create "$TAG" \
  --title "afm ${VERSION}" \
  --notes "$(cat <<EOF
## afm ${VERSION}

Apple Foundation Models + MLX local models — OpenAI-compatible API, WebUI, all Swift.

### Changes since ${PREV_TAG:-first release}
${CHANGELOG}

### Install / Upgrade via Homebrew

**Fresh install:**
\`\`\`
brew tap scouzi1966/afm
brew install scouzi1966/afm/afm
\`\`\`

**Upgrade:**
\`\`\`
brew upgrade afm
\`\`\`

### Install via PyPI

\`\`\`
pip install macafm==${VERSION}
\`\`\`
EOF
)" \
  --target main \
  --repo "$REPO" \
  "$TARBALL"

log_info "Release uploaded: $TAG"

# Step 6: Update tap formula
log_info "Updating tap formula..."
SHA256=$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)

cd "$TAP_DIR"
git pull --ff-only

DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${TAG}/afm-${TAG}-arm64.tar.gz"
sed -i '' "s|url \".*\"|url \"${DOWNLOAD_URL}\"|" afm.rb
sed -i '' "s/version \".*\"/version \"${VERSION}\"/" afm.rb
sed -i '' "s/sha256 \".*\"/sha256 \"${SHA256}\"/" afm.rb
# Update test block version
sed -i '' "s/assert_match \"v[0-9][^\"]*\"/assert_match \"v${VERSION}\"/" afm.rb
# Update caveats version references
sed -i '' "s/MLX Local Models (v[0-9][^)]*)/MLX Local Models (v${VERSION}+)/" afm.rb

git add afm.rb
git commit -m "afm ${VERSION}"
git push

log_info "Tap updated"

# Step 7: Bump pyproject.toml and __init__.py
cd "$ROOT_DIR"
log_info "Bumping Python package version to ${VERSION}..."
sed -i '' "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml
sed -i '' "s/^__version__ = \".*\"/__version__ = \"${VERSION}\"/" macafm/__init__.py
log_info "Updated pyproject.toml and macafm/__init__.py"

# Step 8: Build Python package
log_info "Building Python package..."
uv build
log_info "Python package built"

# Cleanup
rm -rf "$STAGING"
rm -f "$TARBALL"

echo ""
log_info "Done! afm ${VERSION} published to GitHub and Homebrew."
echo ""
log_info "To publish to PyPI, run:"
echo "  uv publish --token <YOUR_PYPI_TOKEN>"
echo ""
log_info "Then commit the version bump:"
echo "  git add pyproject.toml macafm/__init__.py"
echo "  git commit -m 'Bump version to ${VERSION}'"
echo ""
echo "  Homebrew: brew upgrade afm"
echo "  PyPI:     pip install macafm==${VERSION}"
