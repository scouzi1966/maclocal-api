// swift-tools-version: 6.0
import PackageDescription

// Minimal standalone package proving AFMKit is importable via SPM.
// In a real consumer, replace the path dependency with:
//   .package(url: "https://github.com/scouzi1966/maclocal-api.git", branch: "main")
// (clone with --recursive, or once submodules are converted to git-URL deps — see
//  docs/wwdc26-migration.md / the AFMKit remote-consumption note.)
let package = Package(
    name: "AFMKitConsumer",
    platforms: [.macOS("26.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "AFMKitConsumer",
            dependencies: [
                .product(name: "AFMKit", package: "MacLocalAPI")
            ]
        )
    ]
)
