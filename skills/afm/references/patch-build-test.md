# Patch, Build, and Test Reference

## Scope

Use this file when changing vendor integration, build steps, WebUI packaging, regression scripts, or release flow.

## Vendor Patch Workflow

Never treat `vendor/mlx-swift-lm` edits as source of truth.

Source of truth:

- `Scripts/patches/*.swift`
- mapping logic in `Scripts/apply-mlx-patches.sh`

Patch script modes:

- apply: `./Scripts/apply-mlx-patches.sh`
- check: `./Scripts/apply-mlx-patches.sh --check`
- revert: `./Scripts/apply-mlx-patches.sh --revert`

Script behavior:

- copies patch files over mapped vendor targets
- keeps `.original` backups for revert
- handles new files list and package pin replacements

## Makefile Build Path

`make build` depends on patch stamp:

- applies patches first
- builds release binary with optimization flags
- strips binary

Other targets:

- `make debug`
- `make webui`
- `make build-with-webui`
- `make clean` (also reverts patches if stamp exists)
- `make test`

## Full Bootstrap Build

`./Scripts/build-from-scratch.sh` (default behavior):

1. init/update submodules
2. apply patches and verify
3. build llama.cpp webui assets
4. clean + resolve Swift packages
5. inject commit into `BuildInfo.swift`
6. build afm
7. verify metallib bundle

## Common Test Scripts

- `./test-all-features.sh`
- `./Scripts/afm-cli-tests.sh`
- `./test-streaming.sh`
- `./test-go.sh`
- `./test-metrics.sh`
- `./Scripts/tests/test-structured-outputs.sh`
- `./Scripts/tests/test-vlm-single-prompt.sh`
- `./Scripts/tests/test-tool-call-parsers.py`

Use targeted scripts first, then broader sweeps for confidence.

## MLX Cache and Runtime Notes

Use stable cache path to avoid re-downloads during repeated tests:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache ./.build/release/afm mlx -m <model>
```

For debug logging:

```bash
AFM_DEBUG=1 MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache ./.build/release/afm mlx -m <model>
```

## Release/Distribution Anchors

- `build-release.sh`
- `build-portable.sh`
- `create-distribution.sh`
- `install.sh`
- formula files at repo root (`afm-next.rb`)

Preserve compatibility expectations for Homebrew and pip packaging workflows.
