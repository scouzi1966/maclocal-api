// swift-tools-version: 6.0
import PackageDescription

// Minimal standalone package proving AFMKitCore is importable without the
// optional MLX, FoundationModels, Services, OpenAI compatibility, server, or
// CLI implementation products.
let package = Package(
    name: "AFMKitCoreOnlyConsumer",
    platforms: [.macOS("26.0")],
    dependencies: [
        .package(name: "MacLocalAPI", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "AFMKitCoreOnlyConsumer",
            dependencies: [
                .product(name: "AFMKitCore", package: "MacLocalAPI")
            ]
        )
    ]
)
