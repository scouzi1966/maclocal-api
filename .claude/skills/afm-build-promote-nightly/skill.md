---
name: afm-build-promote-nightly
description: Use when promoting an afm nightly build to a stable release — rebuilds from the nightly commit with stable version, verifies patches, updates Homebrew stable tap (afm.rb), builds a PyPI wheel, updates README and version files, and verifies both brew install and pip install work. Repo admin only.
---

# Promote AFM Nightly to Stable Release

Promote an existing nightly build to a stable release. Checks out the exact nightly commit, updates `BuildInfo.swift` with the stable version, rebuilds with verified patches, then publishes to GitHub, Homebrew (`afm.rb`), and PyPI.

**Repo admin only (scouzi1966).**

## Usage

- `/afm-build-promote-nightly` — promote latest nightly to stable
- `/afm-build-promote-nightly 0.9.7` — promote with explicit stable version

## Instructions

### Step 1: Verify Admin Access

```bash
gh auth status
GH_USER=$(gh api user -q .login)
echo "GitHub user: $GH_USER"

gh api repos/scouzi1966/maclocal-api -q '.permissions.push'
gh api repos/scouzi1966/homebrew-afm -q '.permissions.push'

# Tap repo — auto-clone if missing
TAP_DIR="${TAP_DIR:-$(cd "$(git rev-parse --show-toplevel)/.." && pwd)/homebrew-afm}"
if [ ! -d "$TAP_DIR/.git" ]; then
  echo "Tap repo missing — cloning..."
  gh repo clone scouzi1966/homebrew-afm "$TAP_DIR"
fi
test -f "$TAP_DIR/afm.rb" && echo "Tap OK" || echo "FAILED"
```

**If `GH_USER` is not `scouzi1966` or push access is `false` for either repo, STOP immediately:**
```
This skill promotes nightly builds to stable releases for scouzi1966/maclocal-api.
Only the repository owner (scouzi1966) can run it. You are authenticated as: <username>
```

### Step 2: Identify Nightly to Promote

```bash
gh release list --repo scouzi1966/maclocal-api --limit 10 --json tagName,name,publishedAt,isPrerelease \
  -q '.[] | select(.isPrerelease) | "\(.tagName)\t\(.name)\t\(.publishedAt)"'
```

Show the list. Use **AskUserQuestion**:

**Question:** "Which nightly release should be promoted to stable?"

**Options:**
1. Latest nightly tag (default — show tag name and date)
2. "Other" — let user specify a different nightly tag

Save as `NIGHTLY_TAG` (e.g., `nightly-20260304-9e978c5`).

**Extract the commit SHA from the nightly tag:**
```bash
NIGHTLY_SHA=$(git rev-list -n 1 "$NIGHTLY_TAG" 2>/dev/null || \
  gh release view "$NIGHTLY_TAG" --repo scouzi1966/maclocal-api --json targetCommitish -q '.targetCommitish')
echo "Nightly commit: $NIGHTLY_SHA"
```

### Step 3: Determine Stable Version

If version was provided as argument, use it. Otherwise derive from the nightly:

```bash
NIGHTLY_VERSION=$(gh release view "$NIGHTLY_TAG" --repo scouzi1966/maclocal-api --json body -q '.body' \
  | grep -oE 'Version:\*\* [0-9]+\.[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+') || true

if [ -z "$NIGHTLY_VERSION" ]; then
  NIGHTLY_VERSION=$(grep 'static let version' Sources/MacLocalAPI/BuildInfo.swift \
    | sed 's/.*"\(.*\)".*/\1/' | sed 's/^v//')
fi
```

Use **AskUserQuestion**:

**Question:** "Stable version? Derived base version is `X.Y.Z`."

**Options:**
1. "`X.Y.Z` (derived)" — use as-is
2. "Custom version" — enter a different version

Save as `VERSION`. Set `TAG="v${VERSION}"`.

**Check stable tag doesn't already exist:**
```bash
if gh release view "v${VERSION}" --repo scouzi1966/maclocal-api >/dev/null 2>&1; then
  echo "ERROR: Release v${VERSION} already exists!"
fi
```
If it exists, STOP and ask the user to choose a different version.

### Step 4: Checkout Nightly Commit and Update Version

**CRITICAL:** The stable build must use the exact same commit as the nightly, with only `BuildInfo.swift` changed to carry the stable version.

```bash
cd "$(git rev-parse --show-toplevel)"

# Save current branch to return later
ORIGINAL_BRANCH=$(git branch --show-current)

# Stash any uncommitted changes
git stash --include-untracked 2>/dev/null || true

# Checkout the exact nightly commit
git checkout "$NIGHTLY_SHA"

# Update ONLY BuildInfo.swift with stable version
sed -i '' "s/static let version: String? = \".*\"/static let version: String? = \"v${VERSION}\"/" \
  Sources/MacLocalAPI/BuildInfo.swift

# Verify the change
grep 'static let version' Sources/MacLocalAPI/BuildInfo.swift
# Must show: static let version: String? = "vX.Y.Z"
```

### Step 5: Verify Patches and Rebuild

**CRITICAL FAIL-SAFE:** Before building, verify that ALL vendor patches are correctly applied. The build MUST use our patched files, not upstream originals.

```bash
cd "$(git rev-parse --show-toplevel)"

# Step 5a: Initialize submodules at the exact state for this commit
git submodule update --init --recursive

# Step 5b: Apply patches
./Scripts/apply-mlx-patches.sh

# Step 5c: FAIL-SAFE — Verify every patch file matches its vendor target
./Scripts/apply-mlx-patches.sh --check
PATCH_CHECK=$?

if [ $PATCH_CHECK -ne 0 ]; then
  echo "FATAL: Patch verification FAILED. Aborting build."
  echo "One or more vendor files do not match Scripts/patches/."
  echo "This means the build would use unpatched vendor code."
fi
```

**If `--check` exits non-zero, STOP IMMEDIATELY.** Do NOT proceed with the build. Report the failing patches to the user and investigate.

**Additional per-file verification** — confirm critical patched files byte-for-byte:

```bash
# Verify the most critical patches individually
PATCHES_DIR="$(git rev-parse --show-toplevel)/Scripts/patches"
VENDOR_DIR="$(git rev-parse --show-toplevel)/vendor/mlx-swift-lm"

CRITICAL_PATCHES=(
  "Evaluate.swift:Libraries/MLXLMCommon/Evaluate.swift"
  "LLMModelFactory.swift:Libraries/MLXLLM/LLMModelFactory.swift"
  "ToolCallFormat.swift:Libraries/MLXLMCommon/Tool/ToolCallFormat.swift"
  "Load.swift:Libraries/MLXLMCommon/Load.swift"
  "KVCache.swift:Libraries/MLXLMCommon/KVCache.swift"
)

ALL_OK=true
for entry in "${CRITICAL_PATCHES[@]}"; do
  PATCH_NAME="${entry%%:*}"
  TARGET_REL="${entry##*:}"
  if ! diff -q "$PATCHES_DIR/$PATCH_NAME" "$VENDOR_DIR/$TARGET_REL" >/dev/null 2>&1; then
    echo "MISMATCH: $PATCH_NAME != $TARGET_REL"
    ALL_OK=false
  else
    echo "OK: $PATCH_NAME"
  fi
done

if [ "$ALL_OK" = false ]; then
  echo "FATAL: Critical patch mismatch detected. Aborting."
fi
```

**If any critical patch fails diff, STOP.** Do not build.

**Step 5d: Build for release:**

```bash
# Build webui (if Node.js available)
if command -v node >/dev/null 2>&1 && [ -d "webui" ]; then
  cd webui && npm install && npm run build && cd ..
fi

# Clean and build release
swift package clean
swift build -c release
```

**Step 5e: Post-build version verification:**

```bash
BIN=".build/arm64-apple-macosx/release/afm"
[ -x "$BIN" ] || BIN=".build/release/afm"

# The binary MUST report the stable version
REPORTED_VERSION=$($BIN --version 2>&1)
echo "Binary reports: $REPORTED_VERSION"

if ! echo "$REPORTED_VERSION" | grep -qF "v${VERSION}"; then
  echo "FATAL: Binary reports '$REPORTED_VERSION' but expected 'v${VERSION}'"
  echo "BuildInfo.swift was not picked up by the build."
fi
```

**If the binary version doesn't match `v${VERSION}`, STOP.** The build did not incorporate the version change.

### Step 6: Package Stable Tarball

```bash
cd "$(git rev-parse --show-toplevel)"

BIN=".build/arm64-apple-macosx/release/afm"
[ -x "$BIN" ] || BIN=".build/release/afm"

STAGING=".build/release-package-stable"
rm -rf "$STAGING"
mkdir -p "$STAGING"

cp "$BIN" "$STAGING/"

# Metallib resource bundle
BUNDLE_DIR="$(dirname "$BIN")/MacLocalAPI_MacLocalAPI.bundle"
if [ -d "$BUNDLE_DIR" ]; then
  cp -r "$BUNDLE_DIR" "$STAGING/"
fi

# WebUI
if [ -f "Resources/webui/index.html.gz" ]; then
  mkdir -p "$STAGING/Resources/webui"
  cp "Resources/webui/index.html.gz" "$STAGING/Resources/webui/"
fi

cp README.md "$STAGING/" 2>/dev/null || true
cp LICENSE "$STAGING/" 2>/dev/null || true

STABLE_TARBALL="afm-v${VERSION}-arm64.tar.gz"
tar -czf "$STABLE_TARBALL" -C "$STAGING" .
SHA256=$(shasum -a 256 "$STABLE_TARBALL" | cut -d' ' -f1)

echo "Tarball: $STABLE_TARBALL ($(du -h "$STABLE_TARBALL" | cut -f1 | xargs))"
echo "SHA256: $SHA256"
```

### Step 7: Return to Original Branch

```bash
cd "$(git rev-parse --show-toplevel)"

# Return to original branch
git checkout "$ORIGINAL_BRANCH"
git stash pop 2>/dev/null || true
```

Now apply version changes to the working branch (Steps 8-9 operate on the current branch, not the detached HEAD).

### Step 8: Create Stable GitHub Release

```bash
cd "$(git rev-parse --show-toplevel)"

PREV_TAG=$(gh release list --repo scouzi1966/maclocal-api --limit 20 --json tagName -q '.[].tagName' \
  | grep '^v[0-9]' | head -1) || true

if [ -n "$PREV_TAG" ] && git cat-file -e "${PREV_TAG}^{commit}" 2>/dev/null; then
  CHANGELOG=$(git log --pretty=format:"- %s (\`%h\`)" "${PREV_TAG}..${NIGHTLY_SHA}" -- . ':!vendor' 2>/dev/null)
else
  CHANGELOG=$(git log --pretty=format:"- %s (\`%h\`)" -20 -- . ':!vendor' 2>/dev/null)
fi

gh release create "v${VERSION}" \
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
  --target "$NIGHTLY_SHA" \
  --repo scouzi1966/maclocal-api \
  "$STABLE_TARBALL"
```

Note: `--target "$NIGHTLY_SHA"` ensures the release points to the exact nightly commit.

### Step 9: Update Homebrew Stable Tap (`afm.rb`)

```bash
TAP_DIR="${TAP_DIR:-$(cd "$(git rev-parse --show-toplevel)/.." && pwd)/homebrew-afm}"
cd "$TAP_DIR"
git pull --ff-only

DOWNLOAD_URL="https://github.com/scouzi1966/maclocal-api/releases/download/v${VERSION}/afm-v${VERSION}-arm64.tar.gz"
sed -i '' "s|url \".*\"|url \"${DOWNLOAD_URL}\"|" afm.rb
sed -i '' "s/version \".*\"/version \"${VERSION}\"/" afm.rb
sed -i '' "s/sha256 \".*\"/sha256 \"${SHA256}\"/" afm.rb
sed -i '' "s/assert_match \"v[0-9][^\"]*\"/assert_match \"v${VERSION}\"/" afm.rb
sed -i '' "s/MLX Local Models (v[0-9][^)]*)/MLX Local Models (v${VERSION}+)/" afm.rb

git add afm.rb
git commit -m "afm ${VERSION}"
git push
```

### Step 10: Update Version Files on Working Branch

```bash
cd "$(git rev-parse --show-toplevel)"

# BuildInfo.swift
sed -i '' "s/static let version: String? = \".*\"/static let version: String? = \"v${VERSION}\"/" \
  Sources/MacLocalAPI/BuildInfo.swift

# pyproject.toml
sed -i '' "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml

# macafm/__init__.py
sed -i '' "s/^__version__ = \".*\"/__version__ = \"${VERSION}\"/" macafm/__init__.py
```

### Step 11: Update README.md

```bash
# Install table version
sed -i '' "s/Stable (v[0-9][^)]*)/Stable (v${VERSION})/" README.md

# Release notes link
sed -i '' "s|\[v[0-9][^]]*\](https://github.com/scouzi1966/maclocal-api/releases/tag/v[^)]*)|[v${VERSION}](https://github.com/scouzi1966/maclocal-api/releases/tag/v${VERSION})|" README.md

# "What's new" section
sed -i '' "s/everything in v[0-9][^ ]* plus/everything in v${VERSION} plus/" README.md
```

### Step 12: Build Python Wheel

Stage the freshly built binary (from the stable build in Step 5) into the Python package:

```bash
cd "$(git rev-parse --show-toplevel)"

BIN=".build/arm64-apple-macosx/release/afm"
[ -x "$BIN" ] || BIN=".build/release/afm"

mkdir -p macafm/bin macafm/share/webui
cp "$BIN" macafm/bin/

# Metallib
METALLIB="$(dirname "$BIN")/MacLocalAPI_MacLocalAPI.bundle/default.metallib"
if [ -f "$METALLIB" ]; then
  cp "$METALLIB" macafm/bin/
fi

# WebUI
if [ -f "Resources/webui/index.html.gz" ]; then
  cp "Resources/webui/index.html.gz" macafm/share/webui/
fi

# Build wheel
uv build

# Clean staged assets (never commit binaries)
rm -rf macafm/bin macafm/share

# Verify wheel size (must be >1MB — contains binary)
WHEEL=$(ls -1 dist/macafm-${VERSION}*.whl 2>/dev/null | head -1)
WHEEL_SIZE=$(stat -f%z "$WHEEL" 2>/dev/null || stat -c%s "$WHEEL" 2>/dev/null)
echo "Wheel: $WHEEL (${WHEEL_SIZE} bytes)"
if [ "$WHEEL_SIZE" -lt 1000000 ]; then
  echo "ERROR: Wheel is only ${WHEEL_SIZE} bytes — assets were not staged correctly!"
fi
```

**If wheel is under 1 MB, STOP.** Do not provide the publish command.

### Step 13: Provide PyPI Publish Command

Present the exact command with a placeholder token:

```
uv publish --token <YOUR_PYPI_TOKEN> dist/macafm-VERSION*
```

Use **AskUserQuestion**:

**Question:** "The wheel is ready. Please run the `uv publish` command above (replace `<YOUR_PYPI_TOKEN>` with your token). Confirm when done, or skip PyPI."

**Options:**
1. "Done — published to PyPI"
2. "Skip PyPI"

### Step 14: Verify Deployment

**Homebrew verification:**
```bash
brew update
brew tap scouzi1966/afm
brew install scouzi1966/afm/afm || brew upgrade afm
BREW_VERSION=$(afm --version 2>&1)
echo "Homebrew afm version: $BREW_VERSION"
echo "$BREW_VERSION" | grep -qF "v${VERSION}" && echo "PASS" || echo "FAIL: expected v${VERSION}"
```

**PyPI verification** (only if user published in Step 13):
```bash
VENV_DIR=$(mktemp -d)
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install macafm==${VERSION}
PIP_VERSION=$("$VENV_DIR/bin/afm" --version 2>&1)
echo "PyPI afm version: $PIP_VERSION"
echo "$PIP_VERSION" | grep -qF "v${VERSION}" && echo "PASS" || echo "FAIL: expected v${VERSION}"
rm -rf "$VENV_DIR"
```

Report PASS/FAIL for each. If either fails, investigate and report the issue before proceeding.

### Step 15: Commit and Report

```bash
cd "$(git rev-parse --show-toplevel)"
git add Sources/MacLocalAPI/BuildInfo.swift pyproject.toml macafm/__init__.py README.md
git commit -m "Release v${VERSION}: promote nightly to stable"
```

Use **AskUserQuestion** before pushing:

**Question:** "Version files committed. Push to origin/main?"

**Options:**
1. "Push" — `git push origin main`
2. "Don't push"

**Cleanup:**
```bash
rm -rf ".build/release-package-stable"
rm -f "afm-v${VERSION}-arm64.tar.gz"
```

**Final report:**
- Stable version: `VERSION`
- Built from commit: `NIGHTLY_SHA`
- GitHub release URL
- Homebrew: `brew install scouzi1966/afm/afm` / `brew upgrade afm`
- PyPI: `pip install macafm==VERSION` (if published)
- Wheel file: path and size
- Verification results (Homebrew ✓/✗, PyPI ✓/✗)

### Error Handling

- **Patch verification fails (Step 5)**: STOP. Do not build. Show which patches failed `diff -q`. Likely cause: submodule is at wrong commit or patches were not applied. Re-run `git submodule update --init --recursive` then `./Scripts/apply-mlx-patches.sh`.
- **Binary version mismatch (Step 5e)**: STOP. `BuildInfo.swift` change was not compiled. Run `swift package clean` and rebuild.
- **Stable tag already exists**: User must choose a different version or delete existing release first.
- **Tap push fails**: Check `$TAP_DIR` is on right branch with no uncommitted changes.
- **Wheel too small (<1 MB)**: Assets not staged — verify binary, metallib, and webui exist in staging dirs.
- **`uv build` fails**: Check `uv` is installed; check `pyproject.toml` syntax.
- **Homebrew install fails after publish**: Run `brew update`; check formula with `brew audit afm`.
- **PyPI install fails**: Check version was actually published; try `pip install macafm==VERSION --no-cache-dir`.
- **Detached HEAD issues**: Step 7 returns to original branch. If `git checkout` fails, check for uncommitted changes with `git status`.
