# Repository Guidelines

See `CLAUDE.md` for additional project-specific build, architecture, and workflow reference details.

## Project Structure & Module Organization
`Sources/AFMCLI/` contains the CLI, `Sources/AFMServer/` owns Vapor and the OpenAI HTTP boundary, and `Sources/AFMKit/` is maclocal-api's aggregate facade. The standalone provider contracts and runtimes come from the pinned AFMKit package. Tests live in `Tests/MacLocalAPITests/`; automation and regression scripts are in `Scripts/`, design notes in `docs/`, and generated artifacts in `test-reports/` and `archive/`.

`vendor/` contains pinned submodules (`mlx-swift-lm`, `llama.cpp`, `xgrammar`, and canonical `antirez/ds4`). Do not edit vendor files directly. Normal builds do not mutate or compile the legacy `vendor/mlx-swift-lm` checkout; they consume immutable AFMKit and AFM-compatible MLX dependencies. `Scripts/patches/` and `apply-mlx-patches.sh` remain only for explicit compatibility-package maintenance. DwarfStar must remain an unchanged upstream checkout; keep its integration in AFMKit-owned adapter sources.

## Build, Test, and Development Commands
Use the project `Makefile` for normal workflows. All direct SwiftPM build and
test invocations must go through `Scripts/swiftpm-reliable.sh`; do not invoke
raw `swift build` or `swift test`. The wrapper selects the reliable Xcode 27
driver, repairs stale explicit-module state once, and stages the canonical MLX
metallib beside every XCTest executable for MLX's C++ runtime. The default
package graph pins AFMKit by immutable revision and pins both AFM-compatible MLX
packages by exact version. Local dependency paths and the legacy source-patch
stack require explicit environment opt-ins.
This applies to release/coverage harness scripts and copied XCTest reruns too;
do not replace the wrapper with raw `swift test` or a one-off environment fix.

- `make build` verifies the AFMKit consumer boundary and builds the release `afm` binary without mutating vendor sources.
- `make debug` builds a debug binary at `.build/debug/afm`.
- `make run` starts the debug server on port `9999`.
- `make test` performs the basic binary and portability checks.
- `./Scripts/build-from-scratch.sh` runs the full clean consumer build, including submodules and WebUI assets, against immutable package dependencies.
- `make patch` is restricted to explicit legacy compatibility-package maintenance.
- `Scripts/swiftpm-reliable.sh build -c release --product afm` builds AFM directly.
- `Scripts/swiftpm-reliable.sh test -c release` runs the Swift unit test suite directly.
- `./Scripts/test-assertions.sh --tier smoke --model <model>` runs the broader assertion and integration harness.

## Coding Style & Naming Conventions
Follow existing Swift conventions: 4-space indentation, `UpperCamelCase` for types, `lowerCamelCase` for methods and properties, and descriptive filenames that match the primary type (`MLXModelService.swift`). Keep shell scripts executable, POSIX-friendly where practical, and named with hyphenated verbs such as `build-from-scratch.sh`.

Preserve current module boundaries and avoid broad refactors when a targeted change is enough.

## Testing Guidelines
Add or update `XCTest` coverage in `Tests/MacLocalAPITests/` for parser, request, or controller behavior changes. Name tests by behavior, for example `testXMLToolCallParsesObjectArguments`. For MLX or end-to-end changes, pair `swift test` with the relevant script in `Scripts/` and capture outputs under `test-reports/` only when generating reports intentionally.

## Publishing Release Test Artifacts
Preserve bulky release-validation output without burdening clones by attaching one curated `/tmp/afm-v<VERSION>-test-reports.tar.gz` bundle to the matching GitHub release. Include a README with test totals, known failures, baseline identity, and a file inventory; include final reports and supporting raw data, but omit caches, bytecode, secrets, and redundant intermediate runs.

Verify the archive with `shasum -a 256` and `tar -tzf`, check existing assets with `gh release view v<VERSION> --repo scouzi1966/maclocal-api --json assets`, upload with `gh release upload v<VERSION> /tmp/afm-v<VERSION>-test-reports.tar.gz --repo scouzi1966/maclocal-api`, and verify the live asset afterward. Keep reports and archives untracked. Release assets are optional downloads and do not enter clones, source archives, Homebrew installs, or pip installs. Use Actions artifacts only for temporary output; use a separate reports repository with GitHub Pages when permanent browser-rendered HTML is required.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects such as `Fix prefix cache save path` or `Add unit test tier`. Prefer `Add`, `Fix`, `Update`, or `Restore`, and keep the subject focused on user-visible behavior. PRs should describe the problem, the approach, and validation performed; link the issue when applicable and include screenshots only for WebUI or report-facing changes.

## Security & Configuration Tips
This project targets Apple Silicon and current macOS/Xcode toolchains. When running MLX locally, set `MACAFM_MLX_MODEL_CACHE` to an existing model cache path to avoid repeated downloads during development.
