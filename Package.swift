// swift-tools-version: 5.9
import PackageDescription
import Foundation

let package = Package(
    name: "MacLocalAPI",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .executable(
            name: "afm",
            targets: ["MacLocalAPI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/vapor/vapor.git", from: "4.99.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        .package(name: "mlx-swift-lm", path: "vendor/mlx-swift-lm"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // Pin mlx-swift to 0.30.3 — 0.30.4+ has SDPA regression (PR #3023 "Faster two pass sdpa")
        // causing NaN/garbage at ~1500 tokens. Post-0.30.6 fixes (PRs #3119, #3121) don't fully
        // resolve it. Monitor for a properly fixed release.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.30.3")
    ],
    targets: [
        .executableTarget(
            name: "MacLocalAPI",
            dependencies: [
                .product(name: "Vapor", package: "vapor"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers")
            ],
            resources: [
                .copy("Resources/default.metallib")
            ],
            swiftSettings: [
                // Enable optimizations for release builds
                .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-O"], .when(configuration: .release))
            ],
            linkerSettings: [
                // Create a more portable executable
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path"], .when(configuration: .release)),
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "/usr/lib/swift"], .when(configuration: .release)),
                .unsafeFlags(["-Xlinker", "-dead_strip"], .when(configuration: .release))
            ]
        )
    ]
)
