# Consuming AFMKit by URL

AFMKit is a standalone but private Swift package during API development.
maclocal-api consumes this immutable checkpoint:

```swift
.package(
    url: "https://github.com/scouzi1966/AFMKit.git",
    revision: "dfeab23e95ea1979432958e3f9b002beb5685191"
)
```

The revision is an immutable pre-tag checkpoint shared with Vesta. Replace it
with an exact AFMKit version after the first package tag is published. Do not use
a branch requirement for release builds.

## Production Blocker

This transition is not publicly source-buildable while
`scouzi1966/AFMKit` is private. This PR must not be production-merged or used to
publish release artifacts until one of these policy conditions is met:

1. AFMKit is public at the pinned immutable revision or an exact release tag.
2. An approved public immutable package or artifact replaces the private URL.

Private CI access is a development transition mechanism, not evidence of public
readiness. As observed on 2026-08-20, GitHub Actions are also disabled for the
maclocal-api repository. The workflow definitions are complete, but hosted
enforcement requires a repository administrator to re-enable Actions.

Every tag, nightly, and manual publishing entry point runs
`Scripts/check-public-release-eligibility.sh` before building or uploading. The
gate removes ambient credentials and proves that the exact AFMKit revision in
the tracked lock can be fetched anonymously. Authentication can therefore keep
development CI working, but it cannot make a private dependency publishable.

## Authenticated Resolution

Local developers must authenticate a GitHub identity with read access:

```bash
gh auth login
gh auth setup-git
Scripts/resolve-release-dependencies.sh
```

CI and release jobs provide a masked `AFMKIT_READ_TOKEN` secret with repository
read access. The resolver uses an ephemeral `GIT_ASKPASS` helper containing no
credential material, never writes the token to the manifest or lock, and emits
an actionable error when access is missing. GitHub's default repository token
does not grant cross-repository access to a separate private dependency.

The Swift CodeQL job uses that same authenticated resolver on trusted branches.
Fork and Dependabot pull requests never receive the secret: their analysis job
is skipped with an explicit transition notice, and maintainers must run CodeQL
from a trusted branch until AFMKit is publicly resolvable.

## Immutable Release Graph

The normal graph is independent of maclocal-api's dirty vendor worktrees:

| Dependency | Requirement | Purpose |
| --- | --- | --- |
| `AFMKit` | revision `dfeab23e...` | Core, OpenAI, Apple, MLX, and DwarfStar provider products |
| `mlx-swift-afm` | exact `0.31.6-afm.1` | AFM-compatible MLX Swift/C++/Metal runtime |
| `mlx-swift-lm` | exact `0.31.6-afm.3` | AFM model architectures and generation behavior |

The tracked root `Package.resolved` is the release lock for all 40 direct and
transitive SwiftPM dependencies, not only AFMKit and MLX. The boundary gate
requires a full 40-character revision for every pin, rejects branch state and
local release dependencies, validates direct manifest requirements, and checks
pinned vendor gitlinks. Release resolution uses
`swift package --force-resolved-versions resolve` and fails if the lock is stale.

AFMKit owns `Sources/AFMKitMLX/Resources/default.metallib` and DwarfStar's Metal
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

The boundary check rejects restored shadow targets, mutable dependency drift,
normal Make targets that invoke the legacy patch stack, server access to known
provider implementation types, and stale maclocal-api-owned resource names in
release packaging.

The package deployment target and release artifact compatibility floor are
macOS 26. Xcode 27 is required to compile the current Apple provider adapters;
symbols that require macOS 27 remain in availability-gated products and source
blocks. The release gate checks the executable and Metal libraries for a macOS
26 minimum while exercising the macOS 27 bundle layout
`AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib`.
