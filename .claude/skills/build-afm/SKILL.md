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

1. Parse the argument: if the user passes `debug`, use `--debug`. Otherwise default to release.

2. Run the full build script with NO skip flags:
```bash
./Scripts/build-from-scratch.sh        # release (default)
./Scripts/build-from-scratch.sh --debug # debug
```

**IMPORTANT:** Never add `--skip-submodules`, `--skip-patches`, or `--skip-webui`. The point of this skill is a complete from-scratch build.

3. The script already handles:
   - `git submodule update --init --recursive`
   - `Scripts/apply-mlx-patches.sh` (apply + verify)
   - llama.cpp webui build (npm install + build)
   - Swift package resolve + clean + build
   - Version injection: writes the git commit SHA into `BuildInfo.swift` (then restores it after build)
   - Strip symbols for release builds
   - Metallib bundle verification

4. After the build succeeds, report to the user:
   - The build configuration (debug or release)
   - The full binary path (from `swift build --show-bin-path` or the script output)
   - The version string by running: `<binary-path>/afm --version`
   - Example run command: `<binary-path>/afm mlx --help`

5. If the build fails, show the error output and suggest checking:
   - Xcode Command Line Tools are installed (`xcode-select -p`)
   - Node.js/npm available for webui build
   - Submodules initialized properly
