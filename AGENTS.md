# Repository Guidelines

## Project Structure & Module Organization
`Sources/MacLocalAPI/` contains the Swift CLI and server code, with `Controllers/`, `Models/`, `Services/`, and `Utils/` split by responsibility. `Sources/CXGrammar/` holds the C++ grammar bridge used by the Swift package. Tests live in `Tests/MacLocalAPITests/` as `XCTest` cases such as `XMLToolCallParsingTests.swift`. Automation and regression scripts are in `Scripts/`, design notes in `docs/`, and generated artifacts in `test-reports/` and `archive/`.

`vendor/` contains pinned submodules (`mlx-swift-lm`, `llama.cpp`). Do not edit vendor files directly; patch them through `Scripts/patches/` and `Scripts/apply-mlx-patches.sh`.

## Build, Test, and Development Commands
Use the project `Makefile` for normal workflows:

- `make build` builds the release `afm` binary and applies vendor patches first.
- `make debug` builds a debug binary at `.build/debug/afm`.
- `make run` starts the debug server on port `9999`.
- `make test` performs the basic binary and portability checks.
- `./Scripts/build-from-scratch.sh` runs the full clean build flow, including submodules, patches, and web UI assets.
- `swift test` runs the Swift unit test suite directly.
- `./Scripts/test-assertions.sh --tier smoke --model <model>` runs the broader assertion and integration harness.

## Coding Style & Naming Conventions
Follow existing Swift conventions: 4-space indentation, `UpperCamelCase` for types, `lowerCamelCase` for methods and properties, and descriptive filenames that match the primary type (`MLXModelService.swift`). Keep shell scripts executable, POSIX-friendly where practical, and named with hyphenated verbs such as `build-from-scratch.sh`.

Preserve current module boundaries and avoid broad refactors when a targeted change is enough.

## Testing Guidelines
Add or update `XCTest` coverage in `Tests/MacLocalAPITests/` for parser, request, or controller behavior changes. Name tests by behavior, for example `testXMLToolCallParsesObjectArguments`. For MLX or end-to-end changes, pair `swift test` with the relevant script in `Scripts/` and capture outputs under `test-reports/` only when generating reports intentionally.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects such as `Fix prefix cache save path` or `Add unit test tier`. Prefer `Add`, `Fix`, `Update`, or `Restore`, and keep the subject focused on user-visible behavior. PRs should describe the problem, the approach, and validation performed; link the issue when applicable and include screenshots only for WebUI or report-facing changes.

## Security & Configuration Tips
This project targets Apple Silicon and current macOS/Xcode toolchains. When running MLX locally, set `MACAFM_MLX_MODEL_CACHE` to an existing model cache path to avoid repeated downloads during development.
