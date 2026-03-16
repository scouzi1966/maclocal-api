---
name: build-afm
description: Build AFM from scratch, including submodules, patches, WebUI assets, and the Swift package build.
user_invocable: true
---

# Build AFM

Use this skill when the user wants a fresh AFM build or a clean rebuild from the current checkout.

## Usage

- `/build-afm`
- `/build-afm debug`
- `/build-afm release`

## Workflow

### 1. Validate prerequisites

Check the local environment before building:

```bash
uname -m
sw_vers -productVersion
brew --version
xcode-select -p
swift --version
git --version
node --version
npm --version
```

Expected baseline:
- Apple Silicon (`arm64`)
- macOS version compatible with `Package.swift`
- full Xcode selected, not standalone Command Line Tools
- Swift toolchain compatible with the package manifest
- Node.js and npm available for the vendored WebUI build

If prerequisites are missing, stop and tell the user exactly what failed.

### 2. Run the build

Default to release unless the user explicitly asks for debug.

```bash
./Scripts/build-from-scratch.sh
./Scripts/build-from-scratch.sh --debug
```

Do not add skip flags. This workflow is for full builds.

### 3. Report the result

After a successful build, report:
- build configuration
- full path to the `afm` binary
- version output from `afm --version`

### 4. Failure hints

If the build fails, check:
- Xcode selection and license acceptance
- Node.js/npm availability
- submodule initialization
- patch application failures
