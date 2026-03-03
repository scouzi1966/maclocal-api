---
name: build-afm-nightly-publish
description: Build, test, and publish an afm-next nightly release — full from-scratch build, user testing pause, GitHub release, and Homebrew tap update. Use when user types /build-afm-nightly-publish or asks to publish a nightly build.
user_invocable: true
---

# Build & Publish AFM Nightly

Build `afm` from scratch (works from a fresh clone), let the user test it, then publish a GitHub pre-release and update the Homebrew tap.

## Usage

- `/build-afm-nightly-publish` — full pipeline: build + test + publish
- `/build-afm-nightly-publish --skip-build` — skip build, use existing release binary

## Prerequisites

The publish script (`Scripts/publish-next.sh`) requires:
- `gh` CLI authenticated with push access to `scouzi1966/maclocal-api`
- `homebrew-afm` repo cloned adjacent to this repo (at `../homebrew-afm`) or `TAP_DIR` env var set
- All build prerequisites from `/build-afm` (Xcode, Swift, Node.js, etc.)

## Instructions

### Step 1: Validate Environment

Run these checks and present results to the user:

```bash
# Build prerequisites
uname -m                    # must be arm64
sw_vers -productVersion     # must be 26.0+
xcode-select -p             # must point to Xcode.app
swift --version             # Swift 5.9+
git --version
node --version              # Node 18+
npm --version

# Publish prerequisites
gh auth status              # must be authenticated

# CRITICAL: Verify the user is the repo owner (scouzi1966)
# This prevents non-owners from accidentally overwriting releases or the brew tap.
GH_USER=$(gh api user -q .login)
echo "GitHub user: $GH_USER"
# Must be "scouzi1966"

# Verify push (write) access to both repos
gh api repos/scouzi1966/maclocal-api -q '.permissions.push'    # must be true
gh api repos/scouzi1966/homebrew-afm -q '.permissions.push'    # must be true

# Tap repo
TAP_DIR="${TAP_DIR:-$(cd "$(git rev-parse --show-toplevel)/.." && pwd)/homebrew-afm}"
test -f "$TAP_DIR/afm-next.rb" && echo "Tap OK: $TAP_DIR" || echo "MISSING: $TAP_DIR/afm-next.rb"
```

Present as a checklist. **If the GitHub user is not `scouzi1966` or push access is `false` for either repo, STOP immediately** and tell the user:
```
This skill publishes releases and updates the Homebrew tap for scouzi1966/maclocal-api.
Only the repository owner (scouzi1966) can run it. You are authenticated as: <username>
```

If the tap repo is missing, tell the user:
```
gh repo clone scouzi1966/homebrew-afm ../homebrew-afm
```

**Do NOT proceed unless: (1) all build checks pass, (2) GitHub user has push access to BOTH repos.**

### Step 2: Build from Scratch

Run the full build — this works even from a fresh clone:

```bash
./Scripts/build-from-scratch.sh
```

**IMPORTANT:** Never add `--skip-submodules`, `--skip-patches`, or `--skip-webui`. This is a release build — everything must be from scratch.

If the user passed `--skip-build`, skip this step and verify the release binary exists:
```bash
test -x .build/arm64-apple-macosx/release/afm || test -x .build/release/afm
```

### Step 3: Open Binary for User Testing

After build completes, get the binary path and open a Finder window to it:

```bash
BIN=".build/arm64-apple-macosx/release/afm"
[ -x "$BIN" ] || BIN=".build/release/afm"
echo "Binary: $(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
$BIN --version
open "$(dirname "$BIN")"
```

Report to the user:
- Binary path
- Version string
- That a Finder window has been opened to the binary location

Then **use AskUserQuestion** to pause and ask:

**Question:** "The binary is ready for testing. Please test it and confirm when ready to publish."

**Options:**
1. "Publish" — Continue with GitHub release and tap update
2. "Cancel" — Abort without publishing

**Do NOT proceed to Step 4 unless the user selects "Publish".**

### Step 4: Publish Release

Run the publish script with `--skip-build` (already built in Step 2):

```bash
./Scripts/publish-next.sh --skip-build
```

This script handles everything:
1. Packages the binary + metallib bundle + webui into `afm-next-arm64.tar.gz`
2. Generates changelog from commits since the last `nightly-*` release
3. Creates a GitHub pre-release tagged `nightly-YYYYMMDD-SHORTSHA`
4. Updates the `nightly` tag to point to HEAD
5. Updates `afm-next.rb` in the homebrew-afm tap (url, version, sha256)
6. Commits and pushes the tap update

### Step 5: Verify & Report

After the publish script completes, verify and report:

```bash
# Verify GitHub release exists
SHORT_SHA=$(git rev-parse --short HEAD)
DATE=$(date -u +%Y%m%d)
RELEASE_TAG="nightly-${DATE}-${SHORT_SHA}"
gh release view "$RELEASE_TAG" --repo scouzi1966/maclocal-api --json tagName,url,assets -q '.url'

# Verify tap was updated
TAP_DIR="${TAP_DIR:-$(cd "$(git rev-parse --show-toplevel)/.." && pwd)/homebrew-afm}"
grep 'version "' "$TAP_DIR/afm-next.rb"
```

Report to the user:
- Release URL (link to the GitHub release)
- Release tag name
- Changelog (what changed since last nightly)
- Homebrew install/upgrade commands:
  ```
  brew tap scouzi1966/afm
  brew install scouzi1966/afm/afm-next    # fresh install
  brew upgrade afm-next                    # upgrade
  ```

### Error Handling

- **Build failure:** Show error output, suggest running `/build-afm` first to diagnose
- **gh release create failure:** Check `gh auth status`, check if tag already exists (`gh release view <tag>`)
- **Tap push failure:** Check if `../homebrew-afm` is on the right branch and has no uncommitted changes
- **User cancels at Step 3:** Clean exit, no publish. The built binary remains available for manual use.
