#!/bin/bash
# Publish afm-next: build, package, upload release, update tap formula
#
# Usage:
#   ./Scripts/publish-next.sh                      # full build + publish
#   ./Scripts/publish-next.sh --skip-build         # package + publish (assumes already built)
#   ./Scripts/publish-next.sh --since abc1234      # changelog from specific commit
#
# Prerequisites:
#   - gh CLI authenticated
#   - homebrew-afm repo cloned at ../homebrew-afm (relative to this repo root)
#     or set TAP_DIR environment variable

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TAP_DIR="${TAP_DIR:-$ROOT_DIR/../homebrew-afm}"
REPO="scouzi1966/maclocal-api"
DO_BUILD=true
SINCE_SHA=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) DO_BUILD=false ;;
    --since)
      shift
      SINCE_SHA="$1"
      if [ -z "$SINCE_SHA" ]; then
        log_error "--since requires a commit SHA"
        exit 1
      fi
      ;;
    -h|--help)
      echo "Usage: ./Scripts/publish-next.sh [--skip-build] [--since <commit-sha>]"
      echo ""
      echo "Options:"
      echo "  --skip-build        Package + publish (assumes already built)"
      echo "  --since <sha>       Generate changelog from this commit instead of auto-detecting"
      exit 0
      ;;
    *) log_error "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

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

if [ ! -f "$TAP_DIR/afm-next.rb" ]; then
  log_error "afm-next.rb not found in $TAP_DIR"
  exit 1
fi

SHORT_SHA=$(git rev-parse --short HEAD)
DATE=$(date -u +%Y%m%d)
VERSION="0.9.5-next.${SHORT_SHA}.${DATE}"

log_info "Building afm-next"
log_info "  Commit: ${SHORT_SHA}"
log_info "  Version: ${VERSION}"
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
STAGING="$ROOT_DIR/.build/release-package"
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

TARBALL="$ROOT_DIR/afm-next-arm64.tar.gz"
tar -czf "$TARBALL" -C "$STAGING" .
log_info "Tarball: $TARBALL ($(du -h "$TARBALL" | cut -f1 | xargs))"

# Step 4: Generate changelog
log_info "Generating changelog..."
if [ -n "$SINCE_SHA" ]; then
  PREV_SHA="$SINCE_SHA"
  log_info "Changelog since: $PREV_SHA (--since)"
else
  # Find the most recent nightly-* release to diff against
  PREV_SHA=$(gh release list --repo "$REPO" --limit 20 -q '.[].tagName' --json tagName 2>/dev/null \
    | grep '^nightly-' | head -1 | grep -oE '[a-f0-9]+$') || true
fi

if [ -n "$PREV_SHA" ] && git cat-file -e "${PREV_SHA}^{commit}" 2>/dev/null; then
  CHANGELOG=$(git log --pretty=format:"- %s (\`%h\`)" "${PREV_SHA}..HEAD" -- . ':!vendor' 2>/dev/null)
  if [ -z "$CHANGELOG" ]; then
    CHANGELOG="- No changes since previous build"
  fi
else
  CHANGELOG=$(git log --pretty=format:"- %s (\`%h\`)" -10 -- . ':!vendor' 2>/dev/null)
fi

# Step 5: Upload to GitHub release (unique tag per build, keep history)
RELEASE_TAG="nightly-${DATE}-${SHORT_SHA}"
log_info "Creating release: $RELEASE_TAG"
gh release create "$RELEASE_TAG" \
  --prerelease \
  --title "afm-next (${DATE} · ${SHORT_SHA})" \
  --notes "$(cat <<EOF
Nightly build from \`main\` branch.

- **Commit:** ${SHORT_SHA}
- **Date:** ${DATE}
- **Version:** ${VERSION}
- **Changes since:** \`${PREV_SHA:-first build}\`

> This is an unstable development build. For the latest stable release, use \`brew install scouzi1966/afm/afm\`.

### Changes since \`${PREV_SHA:-first build}\`
${CHANGELOG}

### Install / Upgrade via Homebrew

**Fresh install** (first time):
\`\`\`
brew tap scouzi1966/afm
brew install scouzi1966/afm/afm-next
\`\`\`

**Upgrade** (already installed):
\`\`\`
brew upgrade afm-next
\`\`\`

**If you have stable \`afm\` installed**, unlink it first:
\`\`\`
brew unlink afm
brew install scouzi1966/afm/afm-next
\`\`\`

**Switch back to stable**:
\`\`\`
brew unlink afm-next
brew link afm
\`\`\`

**Force reinstall** (same version, new build):
\`\`\`
brew reinstall afm-next
\`\`\`
EOF
)" \
  --target main \
  --repo "$REPO" \
  "$TARBALL"

log_info "Release uploaded: $RELEASE_TAG"

# Update the 'nightly' tag to point to this release (latest pointer)
git tag -f nightly HEAD
git push origin nightly --force 2>/dev/null || true

# Step 6: Update tap formula
log_info "Updating tap formula..."
SHA256=$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)

cd "$TAP_DIR"
git pull --ff-only

DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/afm-next-arm64.tar.gz"
sed -i '' "s|url \".*\"|url \"${DOWNLOAD_URL}\"|" afm-next.rb
sed -i '' "s/version \".*\"/version \"${VERSION}\"/" afm-next.rb
sed -i '' "s/sha256 \".*\"/sha256 \"${SHA256}\"/" afm-next.rb

git add afm-next.rb
git commit -m "afm-next ${VERSION} (${SHORT_SHA})"
git push

log_info "Tap updated"

# Cleanup
rm -rf "$STAGING"
rm -f "$TARBALL"

echo ""
log_info "Done! afm-next ${VERSION} published."
echo "  Install: brew install scouzi1966/afm/afm-next"
echo "  Upgrade: brew upgrade afm-next"
