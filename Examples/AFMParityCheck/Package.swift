// swift-tools-version: 6.0
import PackageDescription

// Differential parity harness: proves AFMKit-direct inference is byte-for-byte
// identical to the `afm` HTTP server at temperature 0. Depends only on AFMKit
// (no Vapor) and Foundation's URLSession for the HTTP client side.
//
// In a real consumer, replace the path dependency with:
//   .package(url: "https://github.com/scouzi1966/maclocal-api.git", branch: "feature/afmlib")
let package = Package(
    name: "AFMParityCheck",
    platforms: [.macOS("26.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "AFMParityCheck",
            dependencies: [
                .product(name: "AFMKit", package: "maclocal-api")
            ]
        )
    ]
)
