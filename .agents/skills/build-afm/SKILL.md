---
name: build-afm
description: Build and test maclocal-api/AFM reliably with Xcode 27 and MLX.
---

# Build AFM

Run commands from the maclocal-api repository root.

## SwiftPM

Never invoke raw `swift build` or `swift test`. Always use:

```bash
Scripts/swiftpm-reliable.sh build -c release --product afm
Scripts/swiftpm-reliable.sh test -c release
```

The wrapper selects the known-good Xcode 27 SwiftPM driver, repairs stale
explicit-module state, exports the canonical
`Sources/AFMKitMLX/Resources/default.metallib`, and stages `mlx.metallib` for
XCTest. Do not work around metallib failures with one-off environment values.

It also fingerprints `vendor/mlx-swift-lm` and invalidates compiled products
when those sources change. This prevents Xcode 27 Beta 3 from reporting a
successful no-op build after applying an MLX source or Metal-kernel patch.
The manifest compiles this vendor directly when initialized; a submodule-free
consumer falls back to the pinned pre-patched URL fork.
Run `Scripts/check-mlx-source-selection.sh` after dependency changes.

The comprehensive assertion harness must continue to delegate its Swift tests
to this wrapper. Local copied XCTest binaries also have a source-checkout
fallback in `MLXMetalLibrary`; packaged binaries must carry their resource
bundle or loose `default.metallib` beside the executable.

## Performance

Use Release binaries for inference benchmarks. Never report Debug throughput.
