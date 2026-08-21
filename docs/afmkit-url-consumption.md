# Consuming AFMKit by URL (no submodules)

AFMKit is designed to be embedded by other Swift packages/apps (e.g.
[vesta-mac](./vesta-integration.md), and macOS 27's pluggable SPM AI packages) with a plain:

```swift
.package(url: "https://github.com/scouzi1966/maclocal-api.git", branch: "feature/afmlib")
```

and **no `git submodule` step**. A `git clone` (without `--recursive`) followed by `swift build`
resolves and compiles AFMKit. This document explains how that works and what maintainers must do.

## How the vendored dependencies resolve

afm has two C/Swift dependencies that were historically git submodules — which broke URL
consumption, because a consumer who clones without `--recursive` gets empty submodule
directories and the build fails. Both are now resolved without submodules:

| Dependency | Before | Now (URL-consumable) |
|------------|--------|----------------------|
| **mlx-swift-lm** (patched Swift inference lib) | `.package(path: "vendor/mlx-swift-lm")` — the vanilla upstream submodule patched at build time | `.package(path: "Dependencies/mlx-swift-lm")` — a checked-in snapshot generated from the same repository-owned patches with `mlx-swift` pinned to 0.31.6 |
| **xgrammar** (C++ grammar engine) | `Sources/CXGrammar/xgrammar` symlink → `vendor/xgrammar` submodule | the xgrammar source is **vendored in-repo** under `Sources/CXGrammar/xgrammar`, trimmed to the compile set (cpp/, include/, dlpack/include, header-only picojson) |

The lower-level MLX **C++/Metal kernel** patches (`Scripts/apply-mlx-cpp-patches.sh`,
`apply-mlx-sdpa-backport.sh`) do **not** travel through the Swift source snapshot, but they only affect the
generated `default.metallib`, which is committed (`Sources/MacLocalAPI/Resources/default.metallib`)
and copied into the build. So a URL consumer gets the correct, already-compiled kernels. Only a
maintainer *regenerating* the metallib needs those C++ patch scripts and a forked `mlx-swift`.

## Verifying URL consumption

```bash
# Simulate a plain consumer: clone WITHOUT --recursive, then build AFMKit.
git clone --branch feature/afmlib --single-branch https://github.com/scouzi1966/maclocal-api afmkit-url
cd afmkit-url
swift build --target AFMKit          # compiles the bundled MLX-LM and xgrammar sources
```

This is exactly the check run when the fork dependency was introduced: all `vendor/*` submodules
stay empty and AFMKit still builds. `Examples/AFMKitConsumer` is the equivalent proof from a
*separate* package importing `AFMKit`.

## Maintainer workflow — IMPORTANT

Development builds use the patched `vendor/mlx-swift-lm` checkout. URL consumers, whose submodule
directory is empty, use `Dependencies/mlx-swift-lm`. Both are derived from the vanilla upstream
revision recorded in `Scripts/mlx-swift-lm-upstream-revision` and the repository-owned
`Scripts/patches/` files.

> Applying patches to `vendor/mlx-swift-lm` does not update the clean-consumer snapshot. Editing
> the patch set without synchronizing the fallback means the two build paths can diverge.

When you change anything under `Scripts/patches/` (or bump the upstream mlx-swift-lm submodule):

```bash
./Scripts/apply-mlx-patches.sh
./Scripts/sync-mlx-swift-lm-fallback.sh
Scripts/swiftpm-reliable.sh build -c debug --target AFMKitMLX
```

`vendor/xgrammar` (still a submodule) is for bumping the pinned xgrammar version; after updating it,
re-sync the in-repo copy under `Sources/CXGrammar/xgrammar` (same trim) and rebuild.

## Why both dependencies have in-repo fallbacks

- **mlx-swift-lm** is patched at build time via repository-owned overwrite files. The generated
  snapshot preserves the "never edit vendor/ or generated dependencies directly" model while
  making every patch revision available to URL consumers in the same commit.
- **xgrammar** has no build-time patch step (one small `grammar_functor.cc` tweak, captured in the
  vendored copy) and its compile subset is ~1.1 MB, so committing it in-repo is simpler than
  maintaining a second fork and needs no extra remote.
