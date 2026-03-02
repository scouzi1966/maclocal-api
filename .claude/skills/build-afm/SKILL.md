---
name: build-afm
description: Build AFM from scratch — submodules, patches, webui, and Swift build. Use when user types /build-afm, asks to build afm, or needs a fresh build from a clean clone.
user_invocable: true
---

# Build AFM

Build the `afm` binary from scratch, suitable for a fresh clone or a full rebuild.

## Usage

- `/build-afm` — release build (default)
- `/build-afm debug` — debug build
- `/build-afm release` — explicit release build

## Instructions

### Step 0: Validate Prerequisites

Before building, check that all required tools and environment are present. Run these checks and collect results:

```bash
# 1. Apple Silicon check (MLX requires arm64)
uname -m   # must be "arm64"

# 2. macOS version (Package.swift requires macOS 26+)
sw_vers -productVersion   # must be 26.0 or higher

# 3. Homebrew (must come before brew-installed tools)
brew --version

# 4. Xcode (full install — mlx-swift uses Metal framework SDK, not available in standalone CLI Tools)
xcode-select -p           # must point to Xcode.app, NOT /Library/Developer/CommandLineTools
swift --version           # needs Swift 5.9+ (swift-tools-version: 5.9)

# 5. Git (for submodule operations — installed via brew or Xcode)
git --version

# 6. Node.js + npm (for llama.cpp webui build — Svelte/Vite frontend)
node --version            # Node 18+ recommended
npm --version
```

**Present results as a checklist to the user** in dependency order (install top-to-bottom). For each item, show pass/fail, reason, and install command (even on pass, for copy-paste on other machines):

| # | Prerequisite | Check | Status | Reason | Install |
|---|---|---|---|---|---|
| 1 | Apple Silicon | `uname -m` = arm64 | pass/fail | MLX framework requires ARM64 GPU | N/A (hardware requirement) |
| 2 | macOS 26+ (Tahoe) | `sw_vers` >= 26.0 | pass/fail | Foundation Models backend + SDK APIs | System Settings > Software Update |
| 3 | Homebrew | `brew --version` | pass/fail | Package manager — Git, Node.js depend on it | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| 4 | Xcode (full) | `xcode-select -p` points to Xcode.app | pass/fail | mlx-swift uses Metal SDK (not in standalone CLI Tools) | Install from App Store → search "Xcode" |
| 5 | Swift 5.9+ | `swift --version` | pass/fail | swift-tools-version: 5.9 in Package.swift | Included with Xcode |
| 6 | Git | `git --version` | pass/fail | Submodule init (mlx-swift-lm, llama.cpp) | `brew install git` |
| 7 | Node.js 18+ | `node --version` | pass/fail | llama.cpp webui build (Svelte/Vite) | `brew install node` |
| 8 | npm | `npm --version` | pass/fail | `npm install` + `npm run build` for webui | Included with Node.js |

**Important Xcode notes:**
- Standalone CLI Tools (`xcode-select --install`) are NOT sufficient — mlx-swift imports the Metal framework which requires the full Xcode SDK
- If `xcode-select -p` returns `/Library/Developer/CommandLineTools`, switch to Xcode: `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`
- After installing Xcode, accept the license: `sudo xcodebuild -license accept`

**If anything is missing**, present the failing items with install commands and ask the user to confirm when ready. Alternative Node.js install:

- **Node.js + npm** (via nvm instead of Homebrew): `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && nvm install 22`

**Do NOT proceed with the build until all prerequisites pass.**

### Step 1: Build

Parse the argument: if the user passes `debug`, use `--debug`. Otherwise default to release.

Run the full build script with NO skip flags:
```bash
./Scripts/build-from-scratch.sh        # release (default)
./Scripts/build-from-scratch.sh --debug # debug
```

**IMPORTANT:** Never add `--skip-submodules`, `--skip-patches`, or `--skip-webui`. The point of this skill is a complete from-scratch build.

### Step 2: Monitor

The script already handles:
   - `git submodule update --init --recursive`
   - `Scripts/apply-mlx-patches.sh` (apply + verify)
   - llama.cpp webui build (npm install + build)
   - Swift package resolve + clean + build
   - Version injection: writes the git commit SHA into `BuildInfo.swift` (then restores it after build)
   - Strip symbols for release builds
   - Metallib bundle verification

### Step 3: Report Results

After the build succeeds, report to the user:
   - The build configuration (debug or release)
   - The **full absolute path** to the compiled `afm` binary (from the script output)
   - The version string by running: `<binary-path>/afm --version`

Do NOT add example run commands, CLI options, or environment variables. Just report the binary path and version — the user knows how to run it.

### Step 4: Handle Failures

If the build fails, show the error output and suggest checking:
   - Xcode Command Line Tools are installed (`xcode-select -p`)
   - Node.js/npm available for webui build
   - Submodules initialized properly
   - Re-run Step 0 prerequisite checks to catch environment issues
