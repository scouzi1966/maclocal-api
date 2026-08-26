# Consuming AFMKit by URL

AFMKit is the single source of truth for provider implementations. AFM consumes
one exact release and selects every provider product from it:

```swift
.package(
    url: "https://github.com/scouzi1966/AFMKit.git",
    exact: "0.1.8"
)
```

Do not use a branch or revision requirement for release builds.

## Production Blocker

AFMKit is public and exposes `0.1.8`. AFM release publication remains
fail-closed unless AFMKit and every dependency in its root graph are
anonymously readable and the tracked lock resolves that exact tag.

Before publishing a new AFM version, rollback means restoring the previous
exact AFMKit version and its reviewed lock revision. After an AFM release, make
the same dependency rollback in a new patch release; never move an existing
version tag.

The current policy chooses the first option and fails closed. An anonymously
fetchable revision alone is not evidence of versioned SwiftPM readiness: the
exact release tag and tracked lock must agree. Local qualification and
publication do not depend on GitHub Actions; hosted workflows are optional.

Every tag, nightly, and manual publishing entry point runs
`Scripts/check-public-release-eligibility.sh` before building or uploading. The
gate first requires an exact semantic-version manifest and matching resolved
version, then removes ambient credentials and proves that the locked revision
can be fetched anonymously. Authentication can keep development CI working,
but neither authentication nor a public bare revision makes this source-package
surface publishable.

## Dependency Resolution

AFMKit 0.1.8 is public, so normal resolution requires no GitHub credential:

```bash
Scripts/resolve-release-dependencies.sh
```

The resolver retains optional `AFMKIT_READ_TOKEN` support only for explicitly
private development sources. Its `GIT_ASKPASS` helper contains no credential
material, the token is never written to the manifest or lock, and production
eligibility removes ambient credentials before proving anonymous access.

## Immutable Release Graph

The normal graph is independent of maclocal-api's dirty vendor worktrees:

| Dependency | Requirement | Purpose |
| --- | --- | --- |
| `AFMKit` | exact `0.1.8` | Core, services, evaluation, MLX, DwarfStar, Apple, Foundation Models bridges, audio, and the vendored MLX runtime |

The MLX, mlx-c, Swift bindings, and mlx-swift-lm sources are snapshots inside
AFMKit's `vendor/MLX` tree. They are not separate SwiftPM dependencies of
maclocal-api.

The tracked root `Package.resolved` is the release lock for all 40 direct and
transitive SwiftPM dependencies, not only AFMKit and MLX. The boundary gate
requires a full 40-character revision for every pin, rejects branch state and
local release dependencies, validates direct manifest requirements, verifies
the clean AFMKit checkout against its lock and origin, and checks the remaining
consumer-owned vendor gitlinks. Release resolution uses
`swift package --force-resolved-versions resolve` and fails if the lock is stale.

AFMKit owns `Packages/AFMKitMLX/Sources/AFMKitMLX/Resources/default.metallib` and DwarfStar's Metal
sources. SwiftPM emits those as `AFMKit_AFMKitMLX.bundle` and
`AFMKit_AFMKitDwarfStar.bundle`. maclocal-api packages the bundles unchanged
beside `afm`; it does not rebuild or rename them.

## Local Development Overrides

Normal builds leave all overrides unset. AFMKit development may opt into a
writable checkout with `MACLOCAL_AFMKIT_PATH` when invoking
`Scripts/swiftpm-reliable.sh`. The wrapper copies the consumer manifest and
sources into an ignored `.build-local-afmkit-workspace`, applies the local path
only there, and keeps the tracked release manifest and `Package.resolved`
byte-for-byte unchanged. Compatibility-package maintenance
may additionally set `MACLOCAL_MLX_SWIFT_LM_PATH`,
`AFMKIT_MLX_SWIFT_PATH`, and `MACLOCAL_USE_LEGACY_MLX_PATCH_STACK=1`.
These variables are build-only maintenance controls and are not supported
release inputs.

## Verification

Run the boundary and graph checks before building:

```bash
Scripts/check-afmkit-consumer-boundary.sh
Scripts/resolve-release-dependencies.sh
Scripts/swiftpm-reliable.sh build -c release --product afm
Scripts/validate-release.sh
```

The boundary check rejects restored shadow targets, provider-owned C/C++ roots
or gitlinks, provider-internal consumer tests and API baselines, mutable
dependency drift, normal Make targets that invoke the legacy patch stack,
server access to known provider implementation types, and stale
maclocal-api-owned resource names in release packaging.

The package deployment target and release artifact compatibility floor are
macOS 26. Xcode 27 is required to compile the current Apple provider adapters;
symbols that require macOS 27 remain in availability-gated products and source
blocks. The release gate checks the executable and Metal libraries for a macOS
26 minimum while exercising the macOS 27 bundle layout
`AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib`.
