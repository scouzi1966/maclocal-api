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
- `homebrew-afm` repo at `../homebrew-afm` (relative to repo root) or `TAP_DIR` env var — auto-cloned if missing
- All build prerequisites from `/build-afm` (Xcode, Swift, Node.js, etc.)
- `promptfoo` CLI (`npm install -g promptfoo` or `npx promptfoo`) — required for the promptfoo agentic eval suite

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

# Promptfoo (required for agentic eval suite)
command -v promptfoo && promptfoo --version 2>/dev/null | head -1
# Must be installed (npm install -g promptfoo)

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

# Tap repo — auto-clone if missing
TAP_DIR="${TAP_DIR:-$(cd "$(git rev-parse --show-toplevel)/.." && pwd)/homebrew-afm}"
if [ ! -f "$TAP_DIR/afm-next.rb" ]; then
  echo "Tap repo missing at $TAP_DIR — cloning..."
  gh repo clone scouzi1966/homebrew-afm "$TAP_DIR"
fi
test -f "$TAP_DIR/afm-next.rb" && echo "Tap OK: $TAP_DIR" || echo "FAILED to clone tap repo"
```

Present as a checklist. **If the GitHub user is not `scouzi1966` or push access is `false` for either repo, STOP immediately** and tell the user:
```
This skill publishes releases and updates the Homebrew tap for scouzi1966/maclocal-api.
Only the repository owner (scouzi1966) can run it. You are authenticated as: <username>
```

**Do NOT proceed unless: (1) all build checks pass, (2) GitHub user has push access to BOTH repos, (3) tap repo is available.**

### Step 2: Build from Scratch (True Clean Build)

A nightly release **must** be built from a completely clean state. `swift package clean` is NOT sufficient — it leaves behind cached modules, package resolution state, and precompiled headers that can mask stale code:

| Cached artifact | Location | What `swift package clean` does |
|-----------------|----------|-------------------------------|
| Compiled .o/.swiftmodule | `.build/arm64-apple-macosx/release/` | Removes |
| Module cache (PCM/PCH) | `.build/arm64-apple-macosx/release/ModuleCache/` | **Keeps** (~400MB) |
| Cloned SPM dependencies | `.build/repositories/` | **Keeps** (~300MB) |
| Package resolution lock | `.build/workspace-state.json` | **Keeps** |
| Xcode DerivedData | `~/Library/Developer/Xcode/DerivedData/*maclocal*` | **Keeps** (if exists) |

**Before running the build script**, nuke all cached state:

```bash
# 1. Remove entire SPM build directory (modules, cache, resolution state — everything)
rm -rf .build

# 2. Remove Xcode DerivedData for this project (if anyone opened it in Xcode)
rm -rf ~/Library/Developer/Xcode/DerivedData/*maclocal* \
       ~/Library/Developer/Xcode/DerivedData/*MacLocal* \
       ~/Library/Developer/Xcode/DerivedData/*afm* 2>/dev/null || true

# 3. Verify clean state
test -d .build && echo "FAIL: .build still exists" || echo "OK: .build removed"
```

Then run the full build:

```bash
./Scripts/build-from-scratch.sh
```

**IMPORTANT:** Never add `--skip-submodules`, `--skip-patches`, or `--skip-webui`. This is a release build — everything must be from scratch.

**Why this matters:** Stale ModuleCache can cause the compiler to use old .swiftmodule files from a previous build, meaning your patches compile but the binary links against the cached (unpatched) version. Stale `workspace-state.json` can resolve a different version of MLX Swift than what the pin specifies. Both failures are silent — the build succeeds, the binary runs, but behavior is wrong.

If the user passed `--skip-build`, skip the clean and build steps, but **still run all Step 2b verification checks** against the existing binary:
```bash
test -x .build/arm64-apple-macosx/release/afm || test -x .build/release/afm
```

### Step 2b: Post-Build Verification ("What Could Go Wrong")

The build script reports success, but **do not trust its output alone**. Independently verify every critical artifact. The build script could succeed (exit 0) while:
- Patches silently failed to apply (vendor reverted by `git submodule update`)
- xgrammar compiled but wasn't linked (missing from Package.swift targets)
- MLX Swift resolved a wrong version (pin not applied to Package.swift)
- Metallib bundle missing (Metal shaders won't load at runtime → crash)
- WebUI assets missing (llama.cpp web interface won't serve)
- BuildInfo.swift not restored (leaves dirty working tree)

Run **all** of these checks. Present results as a table. **If ANY check fails, STOP and investigate before proceeding.**

#### Check 1: Patches byte-identical to vendor targets

The patch script says "Applied" but `git submodule update` can silently revert files. Verify every patch file is byte-for-byte identical to its vendor target using the actual arrays from `Scripts/apply-mlx-patches.sh`:

```python
python3 -c "
import os
# These arrays MUST match Scripts/apply-mlx-patches.sh — if they drift, the check is wrong.
# Read them from the script itself to stay in sync.
patches = [
  ('Qwen3VL.swift','Libraries/MLXVLM/Models/Qwen3VL.swift'),
  ('Qwen3Next.swift','Libraries/MLXLLM/Models/Qwen3Next.swift'),
  # ... all 20 entries from PATCH_FILES/TARGET_PATHS arrays ...
]
ok = fail = 0
for pf, tp in patches:
    src, tgt = f'Scripts/patches/{pf}', f'vendor/mlx-swift-lm/{tp}'
    if not os.path.exists(tgt):
        print(f'MISSING:   {pf} -> {tp}'); fail += 1
    else:
        with open(src,'rb') as a, open(tgt,'rb') as b:
            if a.read() == b.read():
                print(f'MATCH:     {pf}'); ok += 1
            else:
                print(f'MISMATCH:  {pf}'); fail += 1
print(f'\n{ok}/{ok+fail} patches verified')
"
```

**Why this matters:** If even one patch is stale, the compiled binary has upstream code instead of our optimized/fixed version. This has happened when `git submodule update --init --recursive` runs AFTER `apply-mlx-patches.sh` — it silently reverts patches.

#### Check 2: MLX Swift pinned AND resolved to exact version

```bash
# Check the pin in source
grep 'mlx-swift.*exact' vendor/mlx-swift-lm/Package.swift
# Must show: exact: "0.30.3"
# 0.30.4+ has SDPA NaN regression — if this shows any other version, STOP.

# Check what SPM actually resolved (the pin could say 0.30.3 but resolution used a cached different version)
python3 -c "
import json
d = json.load(open('Package.resolved'))
for p in d.get('pins', []):
    if 'mlx' in p.get('identity','').lower():
        print(f'{p[\"identity\"]}: {p[\"state\"].get(\"version\",\"?\")}')
"
# Must show: mlx-swift: 0.30.3
# If version differs from pin, the resolution is stale — this is exactly what rm -rf .build prevents.
```

**Why this matters:** The pin in `Package.swift` is a request, but `Package.resolved` is what was actually fetched and compiled against. A stale `workspace-state.json` or `Package.resolved` from a previous build can cause SPM to use a cached resolution even after the pin changes. Nuking `.build/` in Step 2 prevents this, but verify anyway.

#### Check 3: xgrammar submodule present and at expected version

```bash
git submodule status vendor/xgrammar
# Must show a commit hash, NOT a '-' prefix (which means uninitialized)
cd vendor/xgrammar && git describe --tags --always && cd -
# Must show v0.1.32 or the expected pinned tag
```

**Why this matters:** xgrammar is a C++ library compiled from source. If the submodule is missing or at the wrong version, the EBNF grammar constraint feature either doesn't exist or has different behavior.

#### Check 4: xgrammar symbols linked into the binary

```bash
# Verify xgrammar C++ was compiled and linked (not just present as source)
strings .build/arm64-apple-macosx/release/afm | grep -c 'xgrammar/cpp/'
# Must be > 0 (typically 10+)

# Verify our Swift XGrammarService wrapper is in the binary
strings .build/arm64-apple-macosx/release/afm | grep 'XGrammarService'
# Must show: XGrammarService, _TtC11MacLocalAPI15XGrammarService, etc.

# Verify xgrammar C++ symbols are actually linked
nm -a .build/arm64-apple-macosx/release/afm 2>/dev/null | grep -c 'xgrammar'
# Must be > 0 (typically 30+)
```

**Why this matters:** xgrammar could be in the source tree but excluded from the Swift Package Manager target graph. The binary would build fine but grammar-constrained decoding would silently fail at runtime.

#### Check 5: Metallib bundle present

```bash
METALLIB=".build/arm64-apple-macosx/release/MacLocalAPI_MacLocalAPI.bundle/default.metallib"
test -f "$METALLIB" && echo "OK: metallib $(du -h "$METALLIB" | cut -f1)" || echo "FAIL: metallib missing"
# Must exist and be > 1MB (typically ~3.7MB)
```

**Why this matters:** Without the metallib, MLX GPU kernels can't load. The server starts but crashes on first inference. The build script checks this, but verify independently.

#### Check 6: WebUI assets present

```bash
test -f "Resources/webui/index.html.gz" && echo "OK: webui assets" || echo "FAIL: webui missing"
```

**Why this matters:** The llama.cpp web UI is served at `/` — without it, browser access shows nothing.

#### Check 7: BuildInfo.swift is clean (not left with injected SHA)

```bash
grep 'static let version' Sources/MacLocalAPI/BuildInfo.swift
# Must show the base version like: static let version: String? = "v0.9.7"
# Must NOT show a commit SHA like: static let version: String? = "v0.9.7-3d71b40"
git diff Sources/MacLocalAPI/BuildInfo.swift
# Must show no diff (file restored to committed state)
```

**Why this matters:** The build script injects the git SHA into BuildInfo.swift during compilation then restores it. If restore fails, the working tree is dirty and the next `git commit` could accidentally commit the injected version.

#### Check 8: Binary is stripped and reasonable size

```bash
ls -lh .build/arm64-apple-macosx/release/afm
# Size should be 30-50MB for a stripped release binary
# If > 100MB, it's likely unstripped (debug symbols included)
nm -gU .build/arm64-apple-macosx/release/afm 2>/dev/null | wc -l
# Stripped binary has minimal external symbols (< 500 typically)
# Unstripped has thousands
```

#### Check 9: Relocated binary does NOT crash (pip install simulation)

This is the most critical distribution check. SPM auto-generates `resource_bundle_accessor.swift` with a hardcoded absolute build path. If any code path calls `Bundle.module`, the binary will `fatalError` when installed via pip or Homebrew (because the build path no longer exists). This has shipped broken nightlies before.

```bash
# Simulate pip install: copy binary + loose metallib to a temp dir (NO SPM bundle directory)
TMPDIR=$(mktemp -d)
cp .build/arm64-apple-macosx/release/afm "$TMPDIR/"
cp .build/arm64-apple-macosx/release/MacLocalAPI_MacLocalAPI.bundle/default.metallib "$TMPDIR/"

# Must NOT crash with "could not load resource bundle" fatalError
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  "$TMPDIR/afm" mlx -m mlx-community/Qwen3.5-35B-A3B-4bit -s "hello" --max-tokens 5 2>&1 | head -3
EXIT_CODE=${PIPESTATUS[0]}
rm -rf "$TMPDIR"

if [ "$EXIT_CODE" -ne 0 ]; then
  echo "FAIL: Relocated binary crashed (exit $EXIT_CODE)"
  echo "This means Bundle.module fatalError is still reachable."
  echo "Check MLXMetalLibrary.swift — it must NOT call Bundle.module."
else
  echo "PASS: Relocated binary works"
fi
```

**Why this matters:** This is the exact layout pip creates: `macafm_next/bin/afm` + `macafm_next/bin/default.metallib`. If this test fails, every pip user gets a crash on first run. **This check is non-negotiable — if it fails, do NOT publish.**

**Root cause if it fails:** Someone added a `Bundle.module` call somewhere in the codebase. Search for it:
```bash
grep -r 'Bundle\.module' Sources/ --include='*.swift'
# Must return ZERO results (only comments allowed)
```

#### Check 10: No Bundle.module calls in source code

```bash
# Bundle.module uses SPM's auto-generated accessor which fatalError's on relocated binaries.
# It must NEVER be called from our code. Comments referencing it are OK.
HITS=$(grep -r 'Bundle\.module' Sources/ --include='*.swift' | grep -v '//' | grep -v '^\s*//' | wc -l)
if [ "$HITS" -gt 0 ]; then
  echo "FAIL: Found $HITS Bundle.module call(s) in source code"
  grep -rn 'Bundle\.module' Sources/ --include='*.swift' | grep -v '//'
  echo "This WILL crash when installed via pip or Homebrew."
else
  echo "PASS: No Bundle.module calls"
fi
```

**Why this matters:** Even a single `Bundle.module` call anywhere in the code path triggers the auto-generated `fatalError`. This is a regression guard — any future code that adds `Bundle.module` will be caught here before it ships.

#### Check 11: Report all vendor/submodule pin levels

Present the exact version of every submodule and SPM dependency so the user can verify the build reproduces the expected dependency tree. Also fetch the latest release tag from each upstream repo to show if we're behind.

```bash
echo "=== Git Submodules ==="
git submodule status

echo "=== SPM Resolved Versions ==="
python3 -c "
import json
d = json.load(open('Package.resolved'))
for p in sorted(d.get('pins', []), key=lambda x: x.get('identity','')):
    v = p['state'].get('version') or p['state'].get('revision','?')[:12]
    print(f'  {p[\"identity\"]}: {v}')
"

echo "=== Upstream Latest Releases ==="
for repo in ml-explore/mlx-swift ml-explore/mlx-swift-lm mlc-ai/xgrammar ggml-org/llama.cpp huggingface/swift-transformers huggingface/swift-huggingface; do
  tag=$(gh api "repos/$repo/releases/latest" -q '.tag_name' 2>/dev/null || echo "?")
  echo "  $repo: $tag"
done
```

This is informational — no pass/fail. But if a resolved version is unexpected (e.g., mlx-swift != 0.30.3), STOP.

#### Present verification results

| # | Check | What could go wrong | Result |
|---|-------|---------------------|--------|
| 1 | Patches byte-identical (N/N) | submodule update reverted patches | PASS/FAIL |
| 2 | MLX Swift pin + resolved 0.30.3 | stale resolution → SDPA NaN crashes | PASS/FAIL |
| 3 | xgrammar at expected tag | missing submodule → no grammar constraints | PASS/FAIL |
| 4 | xgrammar linked in binary | compiled but not linked → silent runtime failure | PASS/FAIL |
| 5 | Metallib bundle present | missing → crash on first inference | PASS/FAIL |
| 6 | WebUI assets present | missing → no browser UI | PASS/FAIL |
| 7 | BuildInfo.swift clean | dirty working tree → accidental commit | PASS/FAIL |
| 8 | Binary stripped, reasonable size | unstripped → bloated download | PASS/FAIL |
| 9 | Relocated binary works (pip sim) | Bundle.module fatalError → crash on pip install | PASS/FAIL |
| 10 | No Bundle.module in source | regression guard → future crash on relocated binary | PASS/FAIL |

Then present two separate tables for vendor pins:

**Git Submodules:**

| Submodule | Source | Pinned Commit | Upstream Latest | Notes |
|-----------|--------|---------------|-----------------|-------|
| `vendor/mlx-swift-lm` | Submodule | `git submodule status` hash + tag | `gh api repos/.../releases/latest` | Our patched fork |
| `vendor/xgrammar` | Submodule | `git submodule status` hash + tag | `gh api repos/.../releases/latest` | C++ grammar engine |
| `vendor/llama.cpp` | Submodule | `git submodule status` hash + tag | `gh api repos/.../releases/latest` | WebUI only |

**SPM Dependencies (from Package.resolved):**

| Package | Source | Resolved Version | Upstream Latest | Notes |
|---------|--------|-----------------|-----------------|-------|
| `mlx-swift` | SPM (exact pin) | `Package.resolved` version | `gh api repos/.../releases/latest` | 0.30.4+ has SDPA NaN — pinned to exact 0.30.3 |
| `swift-transformers` | SPM (from) | `Package.resolved` version | `gh api repos/.../releases/latest` | Tokenizer/chat templates |
| `swift-huggingface` | SPM (from) | `Package.resolved` version | `gh api repos/.../releases/latest` | HF hub downloads |
| `swift-jinja` | SPM (transitive) | `Package.resolved` version | — | Jinja2 template engine |
| `vapor` | SPM (from) | `Package.resolved` version | — | HTTP framework |

Populate the "Upstream Latest" column by querying `gh api repos/OWNER/REPO/releases/latest -q '.tag_name'`. This lets the user see at a glance if we're behind upstream on any dependency.

**If ANY check fails, STOP. Do not proceed to user testing or publishing.**

### Step 3: Present Binary and Enter Test/Fix/Rebuild Loop

After **all verification checks pass**, get the binary path and version:

```bash
BIN=".build/arm64-apple-macosx/release/afm"
[ -x "$BIN" ] || BIN=".build/release/afm"
echo "Binary: $(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
$BIN --version
```

Report to the user:
- Binary path (absolute)
- Version string
- Verification results table (from Step 2b)

Then **use AskUserQuestion** to pause and let the user decide what to do next:

**Question:** "The build is verified. What would you like to do?"

**Options:**
1. "Publish as-is" — Skip testing, go straight to GitHub release and tap update
2. "Run tests" — Run automated tests, then decide (see test scope question below)
3. "I'll test manually" — Pause here while the user tests the binary themselves
4. "Cancel" — Abort without publishing

#### If user selects "Run tests"

First, list available models in the cache and let the user pick:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache ./Scripts/list-models.sh
```

**Question:** "Which model to test with?"

Present the available models as options (show model name and size). The user picks one.

Then ask the test scope:

**Question:** "Which tests to run?"

**Options:**
1. "Assertions only (all tiers including unit)" — Run `/test-afm-assertions` with full tier (deterministic pass/fail tests, ~15 min/model)
2. "Comprehensive only" — Run `/test-macafm` smart analysis (AI-scored quality evaluation)
3. "Both" — Run assertions first, then comprehensive (most thorough, ~30 min/model)
4. "Full nightly suite" — Run assertions + comprehensive + promptfoo agentic evals (most complete, ~60 min)

Invoke the appropriate skill(s) with the selected model. Do NOT re-ask the model question — pass it through to the test skill(s).

#### If test scope includes promptfoo

Run the full promptfoo agentic eval suite after assertions/comprehensive complete:

```bash
AFM_MODEL=MODEL \
AFM_BINARY=.build/arm64-apple-macosx/release/afm \
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh all
```

This manages its own server lifecycle (starts/stops across 8 server profiles) and runs **~137 test cases across 16 configs**:
- **Structured** (6+4 tests): `json_schema` response format and stress tests
- **Tool calling** (7 tests × 3 profiles): default, adaptive-xml, adaptive-xml-grammar
- **Tool call quality** (6 tests × 3 profiles): BFCL-inspired when-to-call decisions
- **Grammar constraints** (17 tests × 8 server phases): schema/tools enforcement, concurrent, prefix-cache, mixed-strict, header assertions
- **Agentic** (4 tests × 3 profiles): Multi-turn coding workflows
- **Frameworks** (8 tests × 3 profiles): Agent framework tool schemas (OpenCode, Pi, OpenClaw, Hermes shapes)
- **OpenCode** (37 tests × 3 profiles): Primary-source OpenCode built-in tools
- **PI** (20 tests × 3 profiles): Pi coding-agent tools
- **OpenClaw** (12 tests × 3 profiles): OpenClaw tool coverage
- **Hermes** (12 tests × 3 profiles): Hermes agentic framework tools

**Output**: JSON reports in `$AFM_PROMPTFOO_OUT_DIR` (default: `/Volumes/edata/promptfoo/data/maclocal-api/current/`).

**Interpreting promptfoo results:**
- **Server-side features** (structured, toolcall, grammar schema/tools/headers): Should be 100% pass. Failures here indicate server bugs.
- **Concurrent grammar** failures: Known race condition in `--concurrent 2` grammar path — not a release blocker.
- **OpenCode/PI/agentic** failures: Model quality at task complexity — the model can't always pick the right tool or produce correct arguments for complex multi-tool scenarios. Not server bugs.
- **adaptive-xml profile** failures scoring lower than default: The adaptive-xml parser produces slightly different formatting that quality judges may score lower. Compare with default profile to confirm it's not a regression.

To extract pass/fail summary from all result files:
```bash
python3 -c "
import json, os, glob
files = sorted(glob.glob('$AFM_PROMPTFOO_OUT_DIR/*MODEL_SLUG*.json'))
total_pass = total_fail = 0
for f in files:
    name = os.path.basename(f).replace('-MODEL_SLUG.json','')
    d = json.load(open(f))
    stats = d.get('results', d).get('stats', {})
    p, fa = stats.get('successes', 0), stats.get('failures', 0)
    total_pass += p; total_fail += fa
    status = 'PASS' if fa == 0 else f'FAIL ({fa})'
    print(f'{name}: {p}/{p+fa} — {status}')
print(f'\nTOTAL: {total_pass}/{total_pass+total_fail}')
"
```

After tests complete, present results and ask:

**Question:** "Tests complete. What next?"

**Options:**
1. "Publish" — Results are acceptable, proceed to release
2. "Fix and rebuild" — There are issues to fix before releasing
3. "Cancel" — Abort

#### If user selects "I'll test manually"

Wait for the user to come back. When they do, ask:

**Question:** "Ready to proceed?"

**Options:**
1. "Publish" — Testing passed, proceed to release
2. "Fix and rebuild" — There are issues to fix before releasing
3. "Cancel" — Abort

#### If user selects "Fix and rebuild" (from any path above)

The user will make code changes (or ask you to). After changes are made:

1. **Re-run Step 2** (full clean build: `rm -rf .build` + `./Scripts/build-from-scratch.sh`)
2. **Re-run Step 2b** (all 8 verification checks)
3. **Return to Step 3** (present binary and ask again)

This loop repeats until the user selects "Publish" or "Cancel". Each iteration is a full clean rebuild — never do an incremental build for a release.

#### Version and changelog selection

Before publishing, ask the user **two questions** via AskUserQuestion:

**Question 1 — Version:** Determine the suggested version by reading `Sources/MacLocalAPI/BuildInfo.swift` and extracting the version (strip leading `v`). Present it to the user:

"Release version? The base version from BuildInfo.swift is `X.Y.Z`. The full nightly version will be `X.Y.Z-next.<sha>.<date>`."

**Options:**
1. "X.Y.Z (from BuildInfo.swift)" — Use the version from BuildInfo.swift (recommended)
2. "Custom version" — Enter a different base version

**Question 2 — Changelog since:** Show both the last nightly tag AND the last stable release tag so the user can choose the right baseline:

```bash
# Find the last nightly tag
LAST_NIGHTLY=$(git tag -l 'nightly-*' --sort=-creatordate | head -1)
if [ -n "$LAST_NIGHTLY" ]; then
  NIGHTLY_DATE=$(git log -1 --format='%ci' "$LAST_NIGHTLY" 2>/dev/null | cut -d' ' -f1)
  NIGHTLY_COUNT=$(git rev-list "${LAST_NIGHTLY}..HEAD" --count 2>/dev/null)
  echo "Last nightly: $LAST_NIGHTLY ($NIGHTLY_DATE) — $NIGHTLY_COUNT commits since"
fi

# Find the last stable release tag (v*.*.* without -next or nightly)
LAST_STABLE=$(git tag -l 'v*' --sort=-version:refname | grep -v 'nightly\|next' | head -1)
if [ -n "$LAST_STABLE" ]; then
  STABLE_DATE=$(git log -1 --format='%ci' "$LAST_STABLE" 2>/dev/null | cut -d' ' -f1)
  STABLE_COUNT=$(git rev-list "${LAST_STABLE}..HEAD" --count 2>/dev/null)
  echo "Last stable:  $LAST_STABLE ($STABLE_DATE) — $STABLE_COUNT commits since"
fi

# Show commit log from the more recent of the two
echo "--- Commits since last nightly ---"
git log --oneline "${LAST_NIGHTLY}..HEAD" 2>/dev/null
```

Present both reference points and ask:

"Generate changelog from which point?"

**Options:**
1. "Since last nightly `<tag>` (`N` commits)" — Default for routine nightlies (incremental changelog)
2. "Since last stable release `<tag>` (`N` commits)" — Use for the first nightly after a stable release, or when you want the full delta since the last official version
3. "Custom commit SHA" — Enter a specific commit SHA

**Guidance for which to pick:**
- **Routine nightly** (there have been nightlies since the last stable release): use "since last nightly" — the changelog shows only what's new since the previous nightly
- **First nightly after a stable release** (no nightlies since last `v*` tag): use "since last stable release" — the changelog shows everything new in this development cycle
- **No previous tags at all**: use "Custom commit SHA" or omit `--since` entirely (the script will include all commits)

**Changelog filtering — exclude reverted/superseded commits:** When reviewing the commit list for the changelog, omit commits whose work was later removed or fully replaced. For example, if a commit adds a Python bridge and a later commit removes it, neither should appear in the release notes — the net effect is zero. Only include commits that contribute to the final state of the codebase at HEAD. The `publish-next.sh` script includes all commits mechanically; you should review and curate the changelog before presenting it to the user.

If the user selects "since last stable release", pass `--since <stable-tag-sha>` to the publish script.
If the user provides a custom SHA, pass `--since <sha>`.
If the user selects "since last nightly", no `--since` flag is needed (the script defaults to this).

**Do NOT proceed to Step 4 unless the user selects "Publish".**

### Step 4: Publish Release

Run the publish script with `--skip-build` (already built in Step 2), the confirmed version, and optional `--since`:

```bash
# Without custom since (uses last nightly tag):
./Scripts/publish-next.sh --skip-build --version <confirmed-version>

# With custom since:
./Scripts/publish-next.sh --skip-build --version <confirmed-version> --since <commit-sha>
```

This script handles everything:
1. Packages the binary + metallib bundle + webui into `afm-next-arm64.tar.gz`
2. Generates changelog from commits since the last `nightly-*` release (with both Homebrew and pip install instructions in the release notes)
3. Creates a GitHub pre-release tagged `nightly-YYYYMMDD-SHORTSHA`
4. Updates the `nightly` tag to point to HEAD
5. Updates `afm-next.rb` in the homebrew-afm tap (url, version, sha256)
6. Commits and pushes the tap update
7. Builds a nightly wheel (`macafm-next`) via `Scripts/build-nightly-wheel.sh`
8. Uploads the wheel to the GitHub release and updates the PEP 503 index on kruks.ai via `Scripts/update-wheel-index.sh` (requires vesta-mac repo at `../vesta-mac` and `wrangler` for Cloudflare Pages deploy)

### Step 4b: Update README Release Link

After publishing, update the nightly release notes link in `README.md` to point to the new release tag:

```bash
# The README has a table row like:
# | **Release notes** | [v0.9.6](...) | [v0.9.7-next](https://github.com/scouzi1966/maclocal-api/releases/tag/nightly-YYYYMMDD-SHORTSHA) |
# Update the nightly link to the just-published tag
```

1. Read `README.md` and find the nightly release notes link in the Install table
2. Replace the old `nightly-*` tag in the URL with the new release tag (e.g., `nightly-20260312-a49c207`)
3. If the base version changed, also update the link text (e.g., `v0.9.7-next` → `v0.9.8-next`)
4. Commit with message: `Update nightly release link to YYYYMMDD-SHORTSHA`
5. Push to remote

**Do not skip this step.** The README is the main page users see — it must always point to the latest nightly.

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
- Install commands (both methods):
  ```
  # Homebrew
  brew tap scouzi1966/afm
  brew install scouzi1966/afm/afm-next    # fresh install
  brew upgrade afm-next                    # upgrade

  # pip
  pip install --extra-index-url https://kruks.ai/afm/wheels/simple/ macafm-next
  ```

### Step 5b: Archive Test Results

Copy all test reports from this nightly run to the versioned nightly archive:

```bash
DATE=$(date +%Y-%m-%d)
mkdir -p test-reports/nightly/$DATE
cp test-reports/assertions-report-*.html test-reports/assertions-report-*.jsonl \
   test-reports/multi-assertions-report-*.html test-reports/multi-assertions-report-*.jsonl \
   test-reports/nightly/$DATE/ 2>/dev/null || true

# Also copy promptfoo results if they were run
AFM_PROMPTFOO_OUT_DIR="${AFM_PROMPTFOO_OUT_DIR:-/Volumes/edata/promptfoo/data/maclocal-api/current}"
cp "$AFM_PROMPTFOO_OUT_DIR"/*.json test-reports/nightly/$DATE/ 2>/dev/null || true

# Also copy smart analysis if it was run
cp test-reports/smart-analysis-*.md test-reports/nightly/$DATE/ 2>/dev/null || true

git add test-reports/nightly/$DATE/
git commit -m "Add nightly test results for $DATE ($(git rev-parse --short HEAD))"
git push
```

This maintains the test history at `test-reports/nightly/YYYY-MM-DD/` for cross-nightly comparison.

### Error Handling

- **Build failure:** Show error output, suggest running `/build-afm` first to diagnose
- **Verification failure (Step 2b):** Do not proceed. Investigate the specific check that failed. If patches are stale, re-run `./Scripts/apply-mlx-patches.sh`. If resolution is wrong, `rm -rf .build` and rebuild.
- **Test failures (Step 3):** Present the failures and let the user decide: fix and rebuild, publish anyway, or cancel. Do NOT automatically fix test failures — that's the user's decision.
- **gh release create failure:** Check `gh auth status`, check if tag already exists (`gh release view <tag>`)
- **Tap push failure:** Check if `../homebrew-afm` is on the right branch and has no uncommitted changes
- **User cancels at any point:** Clean exit, no publish. The built binary remains available for manual use.
