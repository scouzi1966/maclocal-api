// swift-tools-version: 6.0
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
        // Headless, SPM-importable library: model loading + inference + OpenAI-compatible
        // services + the HTTP server. `import AFMKit` from another package/app.
        .library(
            name: "AFMKit",
            targets: ["AFMKit"]
        ),
        // Vapor HTTP layer (OpenAI-compatible server). Separate product so consumers that
        // only want headless inference can depend on AFMKit alone (no Vapor/NIO in their graph).
        .library(
            name: "AFMServer",
            targets: ["AFMServer"]
        ),
        // The `afm` CLI executable (thin wrapper over AFMKit + AFMServer).
        .executable(
            name: "afm",
            targets: ["AFMCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/vapor/vapor.git", from: "4.99.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        .package(name: "mlx-swift-lm", path: "vendor/mlx-swift-lm"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.8.1"),
        // Pin mlx-swift to 0.30.3 — 0.30.4+ has SDPA regression (PR #3023 "Faster two pass sdpa")
        // causing NaN/garbage at ~1500 tokens. Post-0.30.6 fixes (PRs #3119, #3121) don't fully
        // resolve it. RECONFIRMED 2026-05-31: 0.31.3 still produces garbage/empty at >1500 tok
        // (afm decode@16k deficit vs newer-MLX engines is the price of correct long-context output).
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.30.3"),
        // Jinja (transitive via swift-transformers) — exposed for test target
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0")
    ],
    targets: [
        .target(
            name: "CXGrammar",
            exclude: [
                // xgrammar is now vendored in-repo (Sources/CXGrammar/xgrammar) trimmed to
                // exactly the compile set — cpp/ (minus the nanobind Python binding), include/,
                // 3rdparty/dlpack/include, and the header-only 3rdparty/picojson. The web /
                // tests / python / docs / examples / cpptrace / googletest trees are no longer
                // committed, so their excludes are gone. picojson is header-only and stays on
                // the header search path, so exclude its directory from compilation.
                "xgrammar/3rdparty/picojson",
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
        // Core library — all reusable inference/service/server code. Importable via SPM.
        .target(
            name: "AFMKit",
            dependencies: [
                "CXGrammar",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface")
            ],
            resources: [
                .copy("Resources/default.metallib")
            ],
            swiftSettings: [
                // Enable optimizations for release builds
                .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-O"], .when(configuration: .release)),
                // Strip build machine prefix so errors show Sources/... not /Volumes/.../Sources/...
                .unsafeFlags(["-file-prefix-map", "\(packageDir)/="], .when(configuration: .release))
            ],
            linkerSettings: [
                .linkedFramework("Security"),
                .linkedFramework("IOKit"),
                .linkedLibrary("IOReport"),
                .linkedLibrary("sqlite3")
            ]
        ),
        // Vapor HTTP layer — the OpenAI-compatible server, controllers, backend
        // discovery/proxy, and Telegram bridge. Depends on AFMKit + Vapor.
        .target(
            name: "AFMServer",
            dependencies: [
                "AFMKit",
                .product(name: "Vapor", package: "vapor"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ],
            swiftSettings: [
                .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-O"], .when(configuration: .release)),
                .unsafeFlags(["-file-prefix-map", "\(packageDir)/="], .when(configuration: .release))
            ]
        ),
        // Thin CLI executable over AFMKit + AFMServer.
        .executableTarget(
            name: "AFMCLI",
            dependencies: [
                "AFMKit",
                "AFMServer",
                .product(name: "Vapor", package: "vapor"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ],
            exclude: [
                // Embedded into the binary's __TEXT,__info_plist section via linker flags below.
                "Info.plist"
            ],
            swiftSettings: [
                .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-O"], .when(configuration: .release)),
                .unsafeFlags(["-file-prefix-map", "\(packageDir)/="], .when(configuration: .release))
            ],
            linkerSettings: [
                // Embed Info.plist with NSSpeechRecognitionUsageDescription into the binary's
                // __TEXT,__info_plist section. macOS 26 SIGABRTs any process that requests
                // privacy-sensitive APIs (Speech Recognition, microphone, camera, etc.) without
                // a matching *UsageDescription key in its Info.plist. Required for PR #107's
                // Apple Speech feature; harmless for non-Speech code paths.
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "\(packageDir)/Sources/AFMCLI/Info.plist"
                ]),
                // Create a more portable executable
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path"], .when(configuration: .release)),
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "/usr/lib/swift"], .when(configuration: .release)),
                .unsafeFlags(["-Xlinker", "-dead_strip"], .when(configuration: .release))
            ]
        ),
        .testTarget(
            name: "MacLocalAPITests",
            dependencies: [
                "AFMKit",
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "XCTVapor", package: "vapor"),
                .product(name: "VaporTesting", package: "vapor"),
                // MTP P0 validation needs the patched Qwen3.6 VLM model (Qwen3_5MTPHead).
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                // EAGLE3 P0 validation needs the Gemma4 drafter (MLXLLM module).
                .product(name: "MLXLLM", package: "mlx-swift-lm")
            ]
        )
    ],
    cxxLanguageStandard: .gnucxx17
)
