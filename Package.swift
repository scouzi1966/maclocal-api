// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Strip absolute build paths from __FILE__ macros in C++ warnings (privacy: don't leak dev machine paths)
let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path

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
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.30.3"),
        // Jinja (transitive via swift-transformers) — exposed for test target
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0")
    ],
    targets: [
        .target(
            name: "CXGrammar",
            exclude: [
                "xgrammar/web",
                "xgrammar/tests",
                "xgrammar/python",
                "xgrammar/docs",
                "xgrammar/examples",
                "xgrammar/scripts",
                "xgrammar/site",
                "xgrammar/cmake",
                "xgrammar/3rdparty/cpptrace",
                "xgrammar/3rdparty/googletest",
                "xgrammar/3rdparty/dlpack/contrib",
                "xgrammar/3rdparty/picojson",
                "xgrammar/cpp/nanobind",
            ],
            cSettings: [
                .headerSearchPath("xgrammar/include"),
                .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
                .headerSearchPath("xgrammar/3rdparty/picojson"),
            ],
            cxxSettings: [
                .headerSearchPath("xgrammar/include"),
                .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
                .headerSearchPath("xgrammar/3rdparty/picojson"),
                // Strip local build paths from __FILE__ macros in xgrammar warnings
                .unsafeFlags(["-ffile-prefix-map=\(packageDir)/Sources/CXGrammar/="])
            ]
        ),
        .executableTarget(
            name: "MacLocalAPI",
            dependencies: [
                "CXGrammar",
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
                .unsafeFlags(["-Xlinker", "-dead_strip"], .when(configuration: .release)),
                .linkedFramework("Security"),
                .linkedLibrary("sqlite3")
            ]
        ),
        .testTarget(
            name: "MacLocalAPITests",
            dependencies: [
                "MacLocalAPI",
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "XCTVapor", package: "vapor")
            ]
        )
    ],
    cxxLanguageStandard: .gnucxx17
)
