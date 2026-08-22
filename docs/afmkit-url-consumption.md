# Consuming AFMKit by URL

AFMKit is the single source of truth for provider implementations. AFM consumes
one coordinated, exact release of its three dependency-scoped packages:

```swift
.package(
    url: "https://github.com/scouzi1966/AFMKit.git",
    exact: "0.1.0"
)
.package(url: "https://github.com/scouzi1966/AFMKitMLX.git", exact: "0.1.0")
.package(url: "https://github.com/scouzi1966/AFMKitDwarfStar.git", exact: "0.1.0")
```

The three versions move together. Do not use a branch or revision requirement
for release builds.

## Production Blocker

This branch deliberately stops before making repositories public or publishing
tags. AFM release publication remains fail-closed until all three repositories
exist, are anonymously readable, expose `0.1.0`, and the tracked lock resolves
those exact tags. Making a repository public is an explicit human checkpoint,
not part of the reversible code cutover.

Rollback before that checkpoint is a normal branch deletion. After merging but
before publishing a new AFM version, revert the merge commit. After an AFM
release, restore the previous AFM version and revert the dependency bump; the
old implementation remains preserved in the pre-cutover branch and tag.

The current policy chooses the first option and fails closed. Private CI access
and an anonymously fetchable revision are development mechanisms, not evidence
of versioned SwiftPM readiness. As observed on 2026-08-20, GitHub Actions are
also disabled for the maclocal-api repository. The workflow definitions are
complete, but hosted enforcement requires a repository administrator to
re-enable Actions.

Every tag, nightly, and manual publishing entry point runs
`Scripts/check-public-release-eligibility.sh` before building or uploading. The
gate first requires an exact semantic-version manifest and matching resolved
version, then removes ambient credentials and proves that the locked revision
can be fetched anonymously. Authentication can keep development CI working,
but neither authentication nor a public bare revision makes this source-package
surface publishable.

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
| `AFMKit` | exact `0.1.0` | Core, OpenAI compatibility, and Apple products |
| `AFMKitMLX` | exact `0.1.0` | MLX runtime and macOS 27 Foundation Models adapter |
| `AFMKitDwarfStar` | exact `0.1.0` | DwarfStar runtime and native bridge |
| `mlx-swift-afm` | exact `0.31.6-afm.1` | AFM-compatible MLX Swift/C++/Metal runtime |
| `mlx-swift-lm` | exact `0.31.6-afm.3` | AFM model architectures and generation behavior |

The tracked root `Package.resolved` is the release lock for all 40 direct and
transitive SwiftPM dependencies, not only AFMKit and MLX. The boundary gate
requires a full 40-character revision for every pin, rejects branch state and
local release dependencies, validates direct manifest requirements, verifies
the clean AFMKit checkout against its lock and origin, and checks the remaining
consumer-owned vendor gitlinks. Release resolution uses
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
