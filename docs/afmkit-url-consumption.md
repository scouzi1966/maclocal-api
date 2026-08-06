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
| **mlx-swift-lm** (patched Swift inference lib) | `.package(path: "vendor/mlx-swift-lm")` — a submodule patched at build time | `.package(url: "github.com/scouzi1966/mlx-swift-lm", revision: …)` — a **pre-patched fork** with the patch set already applied and `mlx-swift` pinned to 0.30.3 |
| **xgrammar** (C++ grammar engine) | `Sources/CXGrammar/xgrammar` symlink → `vendor/xgrammar` submodule | the xgrammar source is **vendored in-repo** under `Sources/CXGrammar/xgrammar`, trimmed to the compile set (cpp/, include/, dlpack/include, header-only picojson) |

The lower-level MLX **C++/Metal kernel** patches (`Scripts/apply-mlx-cpp-patches.sh`,
`apply-mlx-sdpa-backport.sh`) do **not** travel through the fork — but they only affect the
generated `default.metallib`, which is committed (`Sources/MacLocalAPI/Resources/default.metallib`)
and copied into the build. So a URL consumer gets the correct, already-compiled kernels. Only a
maintainer *regenerating* the metallib needs those C++ patch scripts and a forked `mlx-swift`.

## Verifying URL consumption

```bash
# Simulate a plain consumer: clone WITHOUT --recursive, then build AFMKit.
git clone --branch feature/afmlib --single-branch https://github.com/scouzi1966/maclocal-api /tmp/afmkit-url
cd /tmp/afmkit-url
swift build --target AFMKit          # resolves the fork from GitHub, compiles vendored xgrammar
```

This is exactly the check run when the fork dependency was introduced: all `vendor/*` submodules
stay empty and AFMKit still builds. `Examples/AFMKitConsumer` is the equivalent proof from a
*separate* package importing `AFMKit`.

## Maintainer workflow — IMPORTANT

The build depends on the **fork**, not on `vendor/mlx-swift-lm`. The submodule + `Scripts/patches/`
are retained only to *regenerate* the fork.

> ⚠️ Applying patches to `vendor/mlx-swift-lm` (as `Scripts/build-from-scratch.sh` does) has **no
> effect on the build** until you regenerate the fork and bump the pinned revision. Editing the
> patch set without regenerating the fork means the build silently keeps the old code.

When you change anything under `Scripts/patches/` (or bump the upstream mlx-swift-lm submodule):

```bash
./Scripts/build-mlx-swift-lm-fork.sh          # applies patches, pushes a new fork revision
# then update the revision it prints in the root Package.swift:
#   .package(url: "https://github.com/scouzi1966/mlx-swift-lm.git", revision: "<new-sha>"),
swift build                                    # verify green against the new revision
```

`vendor/xgrammar` (still a submodule) is for bumping the pinned xgrammar version; after updating it,
re-sync the in-repo copy under `Sources/CXGrammar/xgrammar` (same trim) and rebuild.

## Why a fork for mlx-swift-lm but in-repo for xgrammar?

- **mlx-swift-lm** is patched at build time via a 27-file overwrite set; keeping it a fork
  preserves the "never edit vendor/ directly, patches are the source of truth" model and keeps the
  main repo lean (the fork carries the 2.9 MB Swift tree).
- **xgrammar** has no build-time patch step (one small `grammar_functor.cc` tweak, captured in the
  vendored copy) and its compile subset is ~1.1 MB, so committing it in-repo is simpler than
  maintaining a second fork and needs no extra remote.
