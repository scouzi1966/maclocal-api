// swift-tools-version: 6.1
import PackageDescription
import Foundation

// Strip absolute build paths from __FILE__ macros in C++ warnings (privacy: don't leak dev machine paths)
let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let mlxSwiftDependency: Package.Dependency
let mlxSwiftPackageIdentity: String
if let localMLXSwiftPath = ProcessInfo.processInfo.environment["AFMKIT_MLX_SWIFT_PATH"],
   !localMLXSwiftPath.isEmpty {
    mlxSwiftDependency = .package(path: localMLXSwiftPath)
    mlxSwiftPackageIdentity = URL(fileURLWithPath: localMLXSwiftPath).lastPathComponent.lowercased()
} else {
    mlxSwiftDependency = .package(
        url: "https://github.com/scouzi1966/mlx-swift-afm",
        exact: "0.31.6-afm.1"
    )
    mlxSwiftPackageIdentity = "mlx-swift-afm"
}
let vendoredMLXSwiftLMPath = "\(packageDir)/vendor/mlx-swift-lm"
let mlxSwiftLMDependency: Package.Dependency = FileManager.default.fileExists(
    atPath: "\(vendoredMLXSwiftLMPath)/Package.swift"
) ? .package(path: vendoredMLXSwiftLMPath) : .package(
    url: "https://github.com/scouzi1966/mlx-swift-lm.git",
    revision: "6bab4f5ac55e81903dd74090244c25feb3233338"
)
let afmKitPath = ProcessInfo.processInfo.environment["MACLOCAL_AFMKIT_PATH"]
    ?? "../../../AFMKit"

let package = Package(
    name: "MacLocalAPI",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(
            name: "AFMKitFoundationModels",
            targets: ["AFMKitFoundationModels"]
        ),
        .library(
            name: "AFMKitFoundationModels27",
            targets: ["AFMKitFoundationModels27"]
        ),
        .library(
            name: "AFMKitFoundationModels27DwarfStar",
            targets: ["AFMKitFoundationModels27DwarfStar"]
        ),
        .library(
            name: "AFMKitServices",
            targets: ["AFMKitServices"]
        ),
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
        // During the private transition, maclocal-api consumes the standalone
        // AFMKit checkout through a configurable path. This becomes a tagged
        // URL dependency after the first AFMKit release.
        .package(path: afmKitPath),
        .package(url: "https://github.com/vapor/vapor.git", from: "4.99.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        // Development checkouts compile the patched vendor directly so local source edits
        // cannot be mistaken for successful stale builds. A plain downstream clone without
        // initialized submodules falls back to the pre-patched URL fork and remains portable.
        mlxSwiftLMDependency,
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(
            url: "https://github.com/huggingface/swift-huggingface.git",
            from: "0.8.1",
            traits: ["Xet"]
        ),
        // AFMKitDwarfStar uses the public byte-range API directly so very large
        // GGUF downloads can resume without discarding completed Xet ranges.
        .package(url: "https://github.com/huggingface/swift-xet.git", exact: "0.2.3"),
        // Share the official XGrammar product with host applications such as Vesta.
        // Compiling the vendored implementation here as well as in coreai-models
        // produces duplicate native symbols when both libraries are linked.
        .package(
            url: "https://github.com/mlc-ai/xgrammar",
            revision: "c1570cdb4f8c867a4dbd07b7ff90581f4a2a432b"
        ),
        // All AFMKit consumers share one tagged MLX fork identity. This avoids
        // duplicate MLX modules while preserving the kernels required by AFM.
        mlxSwiftDependency,
        // Jinja (transitive via swift-transformers) — exposed for test target.
        // 2.4.0 broke swift-transformers ≤1.3.3 (ObjectKey change in Hub/Config.swift);
        // 2.4.1 restored source compatibility upstream, so no cap is needed.
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0")
    ],
    targets: [
        .target(
            name: "AFMKitFoundationModels",
            dependencies: [
                .product(name: "AFMOpenAICompat", package: "AFMKit")
            ]
        ),
        .target(
            name: "AFMKitFoundationModels27",
            dependencies: [
                "AFMKit"
            ]
        ),
        .target(
            name: "AFMKitFoundationModels27DwarfStar",
            dependencies: [
                "AFMKit",
                .product(name: "AFMKitDwarfStar", package: "AFMKit"),
                "AFMKitFoundationModels27"
            ]
        ),
        .target(
            name: "AFMKitServices",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit")
            ]
        ),
        // Core library — all reusable inference/service/server code. Importable via SPM.
        .target(
            name: "AFMKit",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMOpenAICompat", package: "AFMKit"),
                .product(name: "AFMKitMLX", package: "AFMKit"),
                "AFMKitFoundationModels",
                "AFMKitServices"
            ],
            swiftSettings: [
                // Enable optimizations for release builds
                .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-O"], .when(configuration: .release)),
                // Strip build machine prefix so errors show Sources/... not /Volumes/.../Sources/...
                .unsafeFlags(["-file-prefix-map", "\(packageDir)/="], .when(configuration: .release))
            ],
            linkerSettings: []
        ),
        // Vapor HTTP layer — the OpenAI-compatible server, controllers, backend
        // discovery/proxy, and Telegram bridge. Depends on AFMKit + Vapor.
        .target(
            name: "AFMServer",
            dependencies: [
                "AFMKit",
                .product(name: "AFMKitMLX", package: "AFMKit"),
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
                .product(name: "AFMKitDwarfStar", package: "AFMKit"),
                .product(name: "AFMKitMLX", package: "AFMKit"),
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
                // Xcode 27 Beta 3 reports a false circular reference when this
                // two-file CLI target is compiled with whole-module/Cross-
                // module optimization. Runtime libraries retain their Release
                // optimization; only the thin command parser is isolated.
                .unsafeFlags(["-no-whole-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-Onone"], .when(configuration: .release)),
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
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMOpenAICompat", package: "AFMKit"),
                .product(name: "AFMKitDwarfStar", package: "AFMKit"),
                .product(name: "AFMKitMLX", package: "AFMKit"),
                "AFMKitFoundationModels",
                "AFMKitFoundationModels27",
                "AFMKitFoundationModels27DwarfStar",
                "AFMKitServices",
                "AFMServer",
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "XCTVapor", package: "vapor"),
                .product(name: "VaporTesting", package: "vapor"),
                // MTP P0 validation needs the patched Qwen3.6 VLM model (Qwen3_5MTPHead).
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                // EAGLE3 P0 validation needs the Gemma4 drafter (MLXLLM module).
                .product(name: "MLXLLM", package: "mlx-swift-lm")
            ],
            swiftSettings: [
                // Xcode 27 Beta 3 reports a false circular reference while
                // optimizing the combined release test module. Product targets
                // remain fully optimized.
                .unsafeFlags(["-no-whole-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-Onone"], .when(configuration: .release))
            ]
        ),
        .testTarget(
            name: "AFMKitDwarfStarTests",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMKitDwarfStar", package: "AFMKit"),
            ],
            swiftSettings: [
                // Match the Xcode 27 Beta 3 workaround used by the main test target.
                .unsafeFlags(["-no-whole-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-Onone"], .when(configuration: .release))
            ]
        )
    ],
    cxxLanguageStandard: .gnucxx17
)
