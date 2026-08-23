# Patch, Build, and Test Reference

## Scope

Use this file when changing vendor integration, build steps, WebUI packaging, regression scripts, or release flow.

## AFMKit provider workflow

Never add a provider source or MLX patch stack to maclocal-api.

Source of truth is the exact AFMKit release. For paired development:

- make the provider change in AFMKit
- set `MACLOCAL_AFMKIT_WORKSPACE_PATH` to that checkout
- validate AFMKit and maclocal-api together
- release AFMKit and bump maclocal-api's single exact version

## Makefile Build Path

`make build` first checks the AFMKit consumer boundary:

- resolves the immutable AFMKit graph
- builds release binary with optimization flags
- strips binary

Other targets:

- `make debug`
- `make webui`
- `make build-with-webui`
- `make clean`
- `make test`

## Full Bootstrap Build

`./Scripts/build-from-scratch.sh` (default behavior):

1. init/update application submodules
2. build llama.cpp webui assets
3. clean + resolve the exact AFMKit package
4. inject commit into `BuildInfo.swift`
5. build afm
6. verify AFMKit resource bundles

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
