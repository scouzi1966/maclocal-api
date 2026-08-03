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
once, writes persistent logs, and exports the canonical
`Sources/AFMKitMLX/Resources/default.metallib` for XCTest.

Do not add ad hoc `MACAFM_MLX_METALLIB` paths to normal test commands. Set that
variable only when deliberately qualifying a different metallib. Use Release
builds for all performance measurements.
