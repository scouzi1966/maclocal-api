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
explicit-module state, resolves AFMKit's committed `default.metallib`, and
stages `mlx.metallib` for XCTest. Do not work around metallib failures with
one-off environment values.

Normal builds consume the immutable AFMKit revision and exact AFM-compatible
MLX tags. They do not compile or mutate `vendor/mlx-swift-lm`. Use
`Scripts/check-afmkit-consumer-boundary.sh` after dependency or packaging
changes. The legacy patch stack requires an explicit maintenance opt-in.

The comprehensive assertion harness must continue to delegate its Swift tests
to this wrapper. Local copied XCTest binaries also have a source-checkout
fallback in `MLXMetalLibrary`; packaged binaries must carry their resource
bundle or loose `default.metallib` beside the executable.

## Performance

Use Release binaries for inference benchmarks. Never report Debug throughput.
