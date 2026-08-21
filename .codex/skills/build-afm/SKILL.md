---
name: build-afm
description: Build or test maclocal-api reliably with Xcode 27 and MLX resources.
---

# Build AFM

Run every direct SwiftPM build or test through the repository wrapper:

```bash
Scripts/swiftpm-reliable.sh build -c release --product afm
Scripts/swiftpm-reliable.sh test -c release
```

Do not invoke raw `swift build` or `swift test`. The wrapper selects the
reliable Xcode 27 build driver, detects and repairs stale explicit-module state
once, writes persistent logs, and stages the canonical metallib from the
resolved AFMKit package beside every XCTest executable where MLX's C++ runtime
expects `mlx.metallib`.

The wrapper also exports `MACAFM_MLX_METALLIB` for AFMKit's own resource
locator. Do not add ad hoc copies or paths to normal test commands. Set that
variable only when deliberately qualifying a different metallib. Use Release
builds for all performance measurements.

Run `Scripts/check-afmkit-consumer-boundary.sh` after dependency or packaging
changes. Normal builds must keep the immutable AFMKit/MLX graph and must not
apply the legacy vendor patch stack.
