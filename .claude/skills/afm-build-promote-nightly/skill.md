---
name: afm-build-promote-nightly
description: Use when promoting afm to a stable release — builds from main HEAD or a nightly commit, verifies patches, updates Homebrew stable tap (afm.rb), builds a PyPI wheel, updates README and version files, and verifies both brew install and pip install work. Repo admin only.
---

# Promote AFM to Stable Release

Build a stable release from either the current `main` branch HEAD or a specific nightly commit. Updates `BuildInfo.swift` with the stable version, rebuilds with verified patches, then publishes to GitHub, Homebrew (`afm.rb`), and PyPI.

**Repo admin only (scouzi1966).**

## Usage

- `/afm-build-promote-nightly` — build stable release (choose source interactively)
- `/afm-build-promote-nightly 0.9.7` — build stable with explicit version

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
This skill publishes stable releases for scouzi1966/maclocal-api.
Only the repository owner (scouzi1966) can run it. You are authenticated as: <username>
```

### Step 2: Choose Build Source

List recent nightly releases for reference:

```bash
# Show current main HEAD
echo "main HEAD: $(git rev-parse --short HEAD) — $(git log -1 --format='%s')"

# List recent nightlies
gh release list --repo scouzi1966/maclocal-api --limit 10 --json tagName,name,publishedAt,isPrerelease \
  -q '.[] | select(.isPrerelease) | "\(.tagName)\t\(.name)\t\(.publishedAt)"'
```

Use **AskUserQuestion**:

**Question:** "Build stable release from which source?"

**Options:**
1. "main HEAD" — build from the current tip of main (most common)
2. "A nightly release" — build from a specific nightly tag's commit
3. "A specific commit" — enter a commit SHA

**If "main HEAD":** Set `BUILD_SHA=$(git rev-parse HEAD)`

**If "A nightly release":** Show the nightly list and ask user to pick one, then:
```bash
NIGHTLY_TAG="<selected tag>"
BUILD_SHA=$(git rev-list -n 1 "$NIGHTLY_TAG" 2>/dev/null || \
  gh release view "$NIGHTLY_TAG" --repo scouzi1966/maclocal-api --json targetCommitish -q '.targetCommitish')
```

**If "A specific commit":** User provides the SHA, set `BUILD_SHA=<user input>`

```bash
echo "Build source commit: $BUILD_SHA"
echo "Commit message: $(git log -1 --format='%s' $BUILD_SHA)"
```

### Step 3: Determine Stable Version

If version was provided as argument, use it. Otherwise derive from `BuildInfo.swift`:

```bash
BASE_VERSION=$(grep 'static let version' Sources/MacLocalAPI/BuildInfo.swift \
  | sed 's/.*"\(.*\)".*/\1/' | sed 's/^v//')
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

### Step 3b: Capture Rollback State

Before any mutations, snapshot the current state so the entire promotion can be reverted (except PyPI, which is immutable).

```bash
cd "$(git rev-parse --show-toplevel)"

# Previous stable version (for tap rollback)
PREV_STABLE_VERSION=$(grep 'version "' "$TAP_DIR/afm.rb" | head -1 | sed 's/.*"\(.*\)".*/\1/')
PREV_STABLE_URL=$(grep 'url "' "$TAP_DIR/afm.rb" | head -1 | sed 's/.*"\(.*\)".*/\1/')
PREV_STABLE_SHA256=$(grep 'sha256 "' "$TAP_DIR/afm.rb" | head -1 | sed 's/.*"\(.*\)".*/\1/')

# Previous version file values
PREV_BUILDINFO_VERSION=$(grep 'static let version' Sources/MacLocalAPI/BuildInfo.swift | sed 's/.*"\(.*\)".*/\1/')
PREV_PYPROJECT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
PREV_INIT_VERSION=$(grep '^__version__' macafm/__init__.py | sed 's/.*"\(.*\)".*/\1/')

# Previous README version
PREV_README_STABLE=$(grep -o 'Stable (v[^)]*)' README.md | head -1)

# Tap commit before our changes
PREV_TAP_COMMIT=$(cd "$TAP_DIR" && git rev-parse HEAD)

echo "=== Rollback snapshot ==="
echo "Tap version: $PREV_STABLE_VERSION"
echo "Tap commit: $PREV_TAP_COMMIT"
echo "BuildInfo: $PREV_BUILDINFO_VERSION"
echo "pyproject: $PREV_PYPROJECT_VERSION"
echo "__init__: $PREV_INIT_VERSION"
echo "README: $PREV_README_STABLE"
```

Report the rollback snapshot to the user so they have it on record.

### Step 4: Checkout Nightly Commit and Update Version

**CRITICAL:** The stable build must use the chosen commit (`BUILD_SHA`), with only `BuildInfo.swift` changed to carry the stable version.

```bash
cd "$(git rev-parse --show-toplevel)"

# Save current branch to return later
ORIGINAL_BRANCH=$(git branch --show-current)

# Stash any uncommitted changes
git stash --include-untracked 2>/dev/null || true

# Checkout the chosen commit
git checkout "$BUILD_SHA"

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

**Step 5d: Build WebUI assets (MANDATORY):**

The WebUI (`-w` flag) is a core feature. Every release MUST include `Resources/webui/index.html.gz`. This asset comes from the llama.cpp submodule and must be built or copied before packaging.

```bash
cd "$(git rev-parse --show-toplevel)"

# Build webui from llama.cpp submodule
WEBUI_DIR="vendor/llama.cpp/tools/server/webui"
WEBUI_PREBUILT="vendor/llama.cpp/tools/server/public/index.html.gz"

if [ -f "$WEBUI_PREBUILT" ]; then
  # Pre-built webui exists in submodule — copy it
  mkdir -p Resources/webui
  cp "$WEBUI_PREBUILT" Resources/webui/index.html.gz
  echo "WebUI: copied pre-built from llama.cpp submodule"
elif [ -d "$WEBUI_DIR" ] && command -v node >/dev/null 2>&1; then
  # Build from source
  cd "$WEBUI_DIR" && npm install && npm run build && cd "$(git rev-parse --show-toplevel)"
  mkdir -p Resources/webui
  cp vendor/llama.cpp/tools/server/public/index.html.gz Resources/webui/index.html.gz
  echo "WebUI: built from source"
else
  echo "FATAL: Cannot build or find WebUI assets."
  echo "Neither $WEBUI_PREBUILT nor $WEBUI_DIR with Node.js available."
fi

# MANDATORY CHECK — release MUST have webui
if [ ! -f "Resources/webui/index.html.gz" ]; then
  echo "FATAL: Resources/webui/index.html.gz is MISSING. Cannot release without WebUI."
else
  echo "WebUI OK: $(ls -lh Resources/webui/index.html.gz | awk '{print $5}')"
fi
```

**If `Resources/webui/index.html.gz` does not exist, STOP IMMEDIATELY.** Do NOT proceed with the build. The WebUI is a core feature and every release must include it.

**Step 5e: Build for release:**

```bash
# Clean and build release
swift package clean
swift build -c release
```

**Step 5f: Post-build version verification:**

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

**Step 5g: Relocated binary check (MANDATORY — blocks pip/Homebrew install crash):**

SPM auto-generates `resource_bundle_accessor.swift` with a hardcoded absolute build path. If any code calls `Bundle.module`, the binary will `fatalError` when relocated (pip install, Homebrew). This has shipped broken releases before. **This check is non-negotiable.**

```bash
# 1. Source code audit — Bundle.module must NEVER appear in non-comment source
HITS=$(grep -r 'Bundle\.module' Sources/ --include='*.swift' | grep -v '^\s*//' | grep -v '// ' | wc -l | tr -d ' ')
if [ "$HITS" -gt 0 ]; then
  echo "FATAL: Found $HITS Bundle.module call(s) — will crash on pip/Homebrew install"
  grep -rn 'Bundle\.module' Sources/ --include='*.swift' | grep -v '//'
  echo "STOP. Remove all Bundle.module calls before proceeding."
  exit 1
fi
echo "PASS: No Bundle.module calls in source"

# 2. Runtime simulation — copy binary + loose metallib to temp dir (NO SPM bundle)
TMPDIR=$(mktemp -d)
cp "$BIN" "$TMPDIR/"
cp "$(dirname "$BIN")/MacLocalAPI_MacLocalAPI.bundle/default.metallib" "$TMPDIR/"

MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  "$TMPDIR/afm" mlx -m mlx-community/Qwen3.5-35B-A3B-4bit -s "hello" --max-tokens 5 2>&1 | head -3
EXIT_CODE=${PIPESTATUS[0]}
rm -rf "$TMPDIR"

if [ "$EXIT_CODE" -ne 0 ]; then
  echo "FATAL: Relocated binary crashed (exit $EXIT_CODE)"
  echo "This means pip install and Homebrew install will crash on first run."
  echo "STOP. Fix MLXMetalLibrary.swift before proceeding."
  exit 1
fi
echo "PASS: Relocated binary runs without crash"
```

**If either check fails, STOP IMMEDIATELY. Do NOT package, publish, or release.** Every pip and Homebrew user will get a crash on first run.

**Step 5h: Info.plist embedding check (MANDATORY — blocks Speech Recognition SIGABRT):**

macOS 26 SIGABRTs any process that requests privacy-sensitive APIs (Speech Recognition, microphone, camera, etc.) without the matching `*UsageDescription` key in its embedded Info.plist. Required for `afm speech`, `POST /v1/audio/transcriptions`, and chat `input_audio` content parts. Any future privacy-API integration needs its `*UsageDescription` key added here too.

```bash
# 1. Verify __TEXT,__info_plist section is present in the binary
if ! otool -l "$BIN" | grep -q '__info_plist'; then
  echo "FATAL: Missing __TEXT,__info_plist section in binary"
  echo "Check Package.swift linker flags (-Xlinker -sectcreate ...) and Sources/MacLocalAPI/Info.plist"
  exit 1
fi

# 2. Verify NSSpeechRecognitionUsageDescription key is embedded
if ! strings "$BIN" | grep -q 'NSSpeechRecognitionUsageDescription'; then
  echo "FATAL: NSSpeechRecognitionUsageDescription missing from embedded plist"
  echo "afm speech / /v1/audio/transcriptions will SIGABRT on macOS 26"
  exit 1
fi

# 3. Verify Info.plist source file is well-formed
plutil -lint Sources/MacLocalAPI/Info.plist || { echo "FATAL: Info.plist is malformed"; exit 1; }

echo "PASS: Info.plist embedded with NSSpeechRecognitionUsageDescription"
```

**If this fails, STOP.** Shipping a stable release with broken Speech Recognition is worse than shipping a nightly — stable users have higher expectations.

**Note on CFBundleIdentifier in Info.plist:** The `CFBundleIdentifier` (`com.scouzi1966.afm`) establishes the TCC identity. Changing it later forces every user to re-grant Speech Recognition / microphone / camera permission. Treat it as a stable contract across releases.

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

# WebUI (MANDATORY — verified in Step 5d)
mkdir -p "$STAGING/Resources/webui"
cp "Resources/webui/index.html.gz" "$STAGING/Resources/webui/"

cp README.md "$STAGING/" 2>/dev/null || true
cp LICENSE "$STAGING/" 2>/dev/null || true

STABLE_TARBALL="afm-v${VERSION}-arm64.tar.gz"
tar -czf "$STABLE_TARBALL" -C "$STAGING" .
SHA256=$(shasum -a 256 "$STABLE_TARBALL" | cut -d' ' -f1)

echo "Tarball: $STABLE_TARBALL ($(du -h "$STABLE_TARBALL" | cut -f1 | xargs))"
echo "SHA256: $SHA256"

# MANDATORY: Verify tarball contains webui
if ! tar tzf "$STABLE_TARBALL" | grep -q "index.html.gz"; then
  echo "FATAL: Tarball does NOT contain WebUI! Aborting."
fi
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
  CHANGELOG=$(git log --pretty=format:"- %s (\`%h\`)" "${PREV_TAG}..${BUILD_SHA}" -- . ':!vendor' 2>/dev/null)
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
  --target "$BUILD_SHA" \
  --repo scouzi1966/maclocal-api \
  "$STABLE_TARBALL"
```

Note: `--target "$BUILD_SHA"` ensures the release points to the exact commit used for the build.

**IMPORTANT: Do NOT delete, edit, or modify any existing nightly releases.** Nightly releases and their `nightly-*` tags must remain intact on GitHub. This skill creates a NEW stable release — it does not replace any existing release.

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

# "What's new in afm-next" section — reset to show no new features since this stable release.
# All previous nightly features are now IN the stable release, so the list must be cleared.
# Future nightlies will add items back as new features land.
sed -i '' "s/everything in v[0-9][^ ]* plus/everything in v${VERSION} plus/" README.md
```

**IMPORTANT:** After updating the version reference, replace the bullet list under "What's new in afm-next" with:
```
> - No new features yet — nightly is currently in sync with the stable release
```
Remove all previous bullet points — they are now part of the stable release and no longer "new in afm-next". Future nightly builds will add new items to this section as features land after the stable cut.

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

# WebUI (MANDATORY — verified in Step 5d)
cp "Resources/webui/index.html.gz" macafm/share/webui/

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

# MANDATORY: Verify wheel contains webui
if ! unzip -l "$WHEEL" | grep -q "index.html.gz"; then
  echo "FATAL: Wheel does NOT contain WebUI!"
fi
```

**If wheel is under 1 MB or missing WebUI, STOP.** Do not provide the publish command.

### Step 12b: User Clean-Slate Installation Testing (MANDATORY)

Before publishing to PyPI, the user MUST verify both installation methods work from a completely clean slate. Each method must be tested independently with full uninstall in between to avoid conflicts.

First, uninstall ALL existing afm installations:

```bash
# Remove all traces
brew uninstall afm 2>/dev/null || true
brew uninstall afm-next 2>/dev/null || true
pip uninstall macafm -y 2>/dev/null || true
uv pip uninstall macafm 2>/dev/null || true

# Verify clean slate
echo "=== Verify clean slate ==="
brew list --formula 2>&1 | grep -E "^afm" || echo "Homebrew: clean"
pip show macafm 2>&1 | grep -i version || echo "pip: clean"
which afm 2>&1 || echo "PATH: clean"
```

Use **AskUserQuestion**:

**Question:** "Clean-slate testing required. Please test Homebrew install first, then uninstall before testing pip. Ready to start?"

Provide the user with these commands:

**Test 1 — Homebrew (test, then fully uninstall before Test 2):**
```
brew update
brew tap scouzi1966/afm
brew install scouzi1966/afm/afm
afm --version          # must show vVERSION
afm mlx -w -m <model>  # must launch WebUI in browser
brew uninstall afm      # MUST uninstall before Test 2
```

**Test 2 — pip from local wheel (after brew is fully removed):**
```
pip install FULL_PATH_TO_WHEEL
afm --version          # must show vVERSION
afm mlx -w -m <model>  # must launch WebUI in browser
pip uninstall macafm    # clean up after testing
```

**IMPORTANT:** Always provide the **full absolute path** to the wheel file. The user may run the command from any directory.

**Options:**
1. "Both passed" — continue to PyPI publish
2. "Homebrew failed" — investigate and fix before continuing
3. "pip failed" — investigate and fix before continuing

**If either test fails, STOP.** Investigate the failure, fix the issue, and re-run the failed test before proceeding. Do NOT publish to PyPI until both methods pass.

### Step 13: Provide PyPI Publish Command

Present the exact command with the **absolute path** to the wheel and a placeholder token:

```bash
ROOT="$(git rev-parse --show-toplevel)"
echo "uv publish --token <YOUR_PYPI_TOKEN> ${ROOT}/dist/macafm-${VERSION}*"
```

**IMPORTANT:** Always provide the full absolute path to the dist files — the user may run the command from a different directory.

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
# If afm-next is linked, unlink it first: brew unlink afm-next && brew link afm
BREW_VERSION=$(afm --version 2>&1)
echo "Homebrew afm version: $BREW_VERSION"
echo "$BREW_VERSION" | grep -qF "v${VERSION}" && echo "VERSION: PASS" || echo "VERSION: FAIL: expected v${VERSION}"

# Smoke test — run a simple prompt to verify the binary actually works
afm -s "why is the sky blue" 2>&1 | head -20
echo "BREW SMOKE TEST: PASS (if output above contains a response)"
```

**PyPI verification** (only if user published in Step 13):
```bash
VENV_DIR=$(mktemp -d)
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install macafm==${VERSION}
PIP_VERSION=$("$VENV_DIR/bin/afm" --version 2>&1)
echo "PyPI afm version: $PIP_VERSION"
echo "$PIP_VERSION" | grep -qF "v${VERSION}" && echo "VERSION: PASS" || echo "VERSION: FAIL: expected v${VERSION}"

# Smoke test — run a simple prompt to verify the pip-installed binary works
$VENV_DIR/bin/afm -s "why is the sky blue" 2>&1 | head -20
echo "PIP SMOKE TEST: PASS (if output above contains a response)"
rm -rf "$VENV_DIR"
```

Report PASS/FAIL for each (version check + smoke test). If either fails, investigate and report the issue before proceeding.

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
- Built from commit: `BUILD_SHA`
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

### Rollback Procedure

If something goes wrong after publishing, use the rollback snapshot from Step 3b to revert. **PyPI is immutable — a published version cannot be deleted or replaced.** All other channels can be fully reverted.

Use **AskUserQuestion** before executing rollback:

**Question:** "Are you sure you want to rollback the v${VERSION} stable release? This will delete the GitHub release, revert the Homebrew tap, and undo version file changes. PyPI cannot be reverted."

**Options:**
1. "Rollback" — proceed
2. "Cancel" — abort rollback

**1. Delete GitHub release and tag:**
```bash
gh release delete "v${VERSION}" --repo scouzi1966/maclocal-api --yes
git tag -d "v${VERSION}" 2>/dev/null || true
git push origin --delete "v${VERSION}" 2>/dev/null || true
```

**2. Revert Homebrew tap:**
```bash
cd "$TAP_DIR"
git revert HEAD --no-edit
git push
# Verify it matches the previous state
grep 'version "' afm.rb
# Should show: PREV_STABLE_VERSION
```

**3. Revert version files on working branch:**
```bash
cd "$(git rev-parse --show-toplevel)"

# Restore previous values
sed -i '' "s/static let version: String? = \".*\"/static let version: String? = \"${PREV_BUILDINFO_VERSION}\"/" \
  Sources/MacLocalAPI/BuildInfo.swift
sed -i '' "s/^version = \".*\"/version = \"${PREV_PYPROJECT_VERSION}\"/" pyproject.toml
sed -i '' "s/^__version__ = \".*\"/__version__ = \"${PREV_INIT_VERSION}\"/" macafm/__init__.py

# Restore README
sed -i '' "s/Stable (v[0-9][^)]*)/Stable (v${PREV_PYPROJECT_VERSION})/" README.md
sed -i '' "s|\[v${VERSION}\](https://github.com/scouzi1966/maclocal-api/releases/tag/v${VERSION})|[v${PREV_PYPROJECT_VERSION}](https://github.com/scouzi1966/maclocal-api/releases/tag/v${PREV_PYPROJECT_VERSION})|" README.md

git add Sources/MacLocalAPI/BuildInfo.swift pyproject.toml macafm/__init__.py README.md
git commit -m "Rollback: revert v${VERSION} promotion"
```

Use **AskUserQuestion** before pushing the rollback commit.

**4. PyPI (immutable — cannot rollback):**

If the wheel was published to PyPI, it cannot be deleted. Options:
- Publish a new patch version (e.g., `X.Y.Z+1`) with the fix
- Yank the version (marks it as not recommended but doesn't delete it):
  `pip install twine && twine yank macafm ${VERSION} --repository pypi`

**After rollback, report:**
- GitHub release: deleted ✓
- Git tag `v${VERSION}`: deleted ✓
- Homebrew tap: reverted to `PREV_STABLE_VERSION` ✓
- Version files: reverted ✓
- PyPI: immutable (not reverted)
