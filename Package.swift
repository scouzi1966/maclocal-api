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
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0")
    ],
    targets: [
        .executableTarget(
            name: "MacLocalAPI",
            dependencies: [
                .product(name: "Vapor", package: "vapor"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
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