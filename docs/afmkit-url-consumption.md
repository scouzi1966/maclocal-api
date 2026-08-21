# Consuming AFMKit by URL

AFMKit is now a standalone Swift package. maclocal-api consumes it from:

```swift
.package(
    url: "https://github.com/scouzi1966/AFMKit.git",
    revision: "dfeab23e95ea1979432958e3f9b002beb5685191"
)
```

The revision is an immutable pre-tag checkpoint shared with Vesta. Replace it
with an exact AFMKit version after the first package tag is published. Do not use
a branch requirement for release builds.

## Published Graph

The normal graph is independent of maclocal-api's dirty vendor worktrees:

| Dependency | Requirement | Purpose |
| --- | --- | --- |
| `AFMKit` | revision `dfeab23e...` | Core, OpenAI, Apple, MLX, and DwarfStar provider products |
| `mlx-swift-afm` | exact `0.31.6-afm.1` | AFM-compatible MLX Swift/C++/Metal runtime |
| `mlx-swift-lm` | exact `0.31.6-afm.3` | AFM model architectures and generation behavior |

AFMKit owns `Sources/AFMKitMLX/Resources/default.metallib` and DwarfStar's Metal
sources. SwiftPM emits those as `AFMKit_AFMKitMLX.bundle` and
`AFMKit_AFMKitDwarfStar.bundle`. maclocal-api packages the bundles unchanged
beside `afm`; it does not rebuild or rename them.

## Local Development Overrides

Normal builds leave all overrides unset. AFMKit development may opt into a
writable checkout with `MACLOCAL_AFMKIT_PATH`. Compatibility-package maintenance
may additionally set `MACLOCAL_MLX_SWIFT_LM_PATH`,
`AFMKIT_MLX_SWIFT_PATH`, and `MACLOCAL_USE_LEGACY_MLX_PATCH_STACK=1`.
These variables are build-only maintenance controls and are not supported
release inputs.

## Verification

Run the boundary and graph checks before building:

```bash
Scripts/check-afmkit-consumer-boundary.sh
swift package show-dependencies --format json
Scripts/swiftpm-reliable.sh build -c release --product afm
```

The boundary check rejects restored shadow targets, mutable dependency drift,
normal Make targets that invoke the legacy patch stack, server access to known
provider implementation types, and stale maclocal-api-owned resource names in
release packaging.

The source package targets macOS 26. AFMKit's Apple provider APIs that require
the macOS 27 SDK remain availability-gated and additive, so MLX and core
consumers retain the macOS 26 deployment boundary.
