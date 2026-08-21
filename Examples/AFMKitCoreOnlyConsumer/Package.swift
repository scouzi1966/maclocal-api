// swift-tools-version: 6.0
import PackageDescription
import Foundation

let afmKitDependency: Package.Dependency
if let localPath = ProcessInfo.processInfo.environment["AFMKIT_EXAMPLE_PATH"],
   !localPath.isEmpty {
    afmKitDependency = .package(name: "AFMKit", path: localPath)
} else {
    afmKitDependency = .package(
        url: "https://github.com/scouzi1966/AFMKit.git",
        revision: "dfeab23e95ea1979432958e3f9b002beb5685191"
    )
}

// Minimal independent package proving AFMKitCore is consumable directly from
// the AFMKit package without maclocal-api, MLX, FoundationModels, or Vapor.
let package = Package(
    name: "AFMKitCoreOnlyConsumer",
    platforms: [.macOS("26.0")],
    dependencies: [
        afmKitDependency
    ],
    targets: [
        .executableTarget(
            name: "AFMKitCoreOnlyConsumer",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit")
            ]
        )
    ]
)
