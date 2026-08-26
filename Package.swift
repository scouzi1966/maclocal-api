// swift-tools-version: 6.1
import PackageDescription
import Foundation

// Strip absolute build paths from __FILE__ macros in C++ warnings (privacy: don't leak dev machine paths)
let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let foundationModelsDependencies: [Target.Dependency]
let foundationModels27Dependencies: [Target.Dependency]
let dwarfStarFoundationModelsDependencies: [Target.Dependency]
#if compiler(>=6.4)
foundationModelsDependencies = [
    .product(name: "AFMKitApple", package: "AFMKit")
]
foundationModels27Dependencies = [
    .product(name: "AFMKitApple", package: "AFMKit"),
    .product(name: "AFMKitFoundationModelsMLX", package: "AFMKit")
]
dwarfStarFoundationModelsDependencies = [
    .product(name: "AFMKitFoundationModelsDwarfStar", package: "AFMKit")
]
#else
foundationModelsDependencies = []
foundationModels27Dependencies = []
dwarfStarFoundationModelsDependencies = []
#endif
let afmKitDependency: Package.Dependency
if let localAFMKitPath = ProcessInfo.processInfo.environment["MACLOCAL_AFMKIT_WORKSPACE_PATH"],
   !localAFMKitPath.isEmpty {
    // Paired development consumes every product from one AFMKit checkout. The
    // tracked release manifest remains exact-versioned.
    afmKitDependency = .package(name: "AFMKit", path: localAFMKitPath)
} else {
    // Release builds lock one AFMKit package and all of its products together.
    // Bumping AFMKit is therefore one explicit, reviewable AFM change.
    afmKitDependency = .package(
        url: "https://github.com/scouzi1966/AFMKit.git",
        exact: "0.1.5"
    )
}

let package = Package(
    name: "MacLocalAPI",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        // Transitional product forwarding. Existing consumers that selected
        // these products from maclocal-api keep resolving while the modules
        // themselves are supplied by the standalone AFMKit package.
        .library(
            name: "AFMKitCore",
            targets: ["AFMKitCoreCompatibility"]
        ),
        .library(
            name: "AFMOpenAICompat",
            targets: ["AFMOpenAICompatCompatibility"]
        ),
        .library(
            name: "AFMKitMLX",
            targets: ["AFMKitMLXCompatibility"]
        ),
        .library(
            name: "AFMKitDwarfStar",
            targets: ["AFMKitDwarfStarCompatibility"]
        ),
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
            targets: ["AFMKitServicesCompatibility"]
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
        // Native terminal chat UI used by both Foundation and MLX modes.
        .library(
            name: "AFMTerminalUI",
            targets: ["AFMTerminalUI"]
        ),
        // The `afm` CLI executable (thin wrapper over AFMKit + AFMServer).
        .executable(
            name: "afm",
            targets: ["AFMCLI"]
        )
    ],
    dependencies: [
        // Normal builds consume the immutable AFMKit checkpoint above. The
        // local-development wrapper stages a disposable manifest before using
        // the internal workspace-only path override.
        afmKitDependency,
        .package(url: "https://github.com/vapor/vapor.git", from: "4.99.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        // Parse CommonMark/GFM into a real syntax tree for the native terminal UI.
        // Rendering remains AFM-owned so ANSI output, math, diffs, and artifact
        // actions behave consistently in Terminal.app and richer terminals.
        .package(url: "https://github.com/swiftlang/swift-markdown.git", exact: "0.8.0"),
        // Tree-sitter is compiled into the native TUI for grammar-aware ANSI
        // highlighting. Every component is pinned so release builds cannot
        // silently pick up a new runtime ABI or regenerated parser.
        .package(url: "https://github.com/tree-sitter/tree-sitter.git", exact: "0.25.10"),
        .package(url: "https://github.com/tree-sitter/swift-tree-sitter.git", exact: "0.25.0"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-bash.git", revision: "a06c2e4415e9bc0346c6b86d401879ffb44058f7"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-c.git", revision: "b780e47fc780ddc8da13afa35a3f4ed5c157823d"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-cpp.git", revision: "8b5b49eb196bec7040441bee33b2c9a4838d6967"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-c-sharp.git", revision: "9150f7d56bb47f1a809fa23623f1ba1413e93fa9"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-css.git", revision: "dda5cfc5722c429eaba1c910ca32c2c0c5bb1a3f"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-go.git", revision: "2346a3ab1bb3857b48b29d779a1ef9799a248cd7"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-html.git", revision: "73a3947324f6efddf9e17c0ea58d454843590cc0"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-java.git", revision: "e10607b45ff745f5f876bfa3e94fbcc6b44bdc11"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-javascript.git", revision: "58404d8cf191d69f2674a8fd507bd5776f46cb11"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-json.git", revision: "254c42a6476413b776221e03982ac8ae159eeb72"),
        .package(url: "https://github.com/tree-sitter-grammars/tree-sitter-kotlin.git", revision: "3dea6dfa9c0129deb7c4315afbda806c85c41667"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-php.git", revision: "3fda2fb9577166c6399834917f9844f30370beea"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-python.git", revision: "26855eabccb19c6abf499fbc5b8dc7cc9ab8bc64"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-ruby.git", revision: "ad907a69da0c8a4f7a943a7fe012712208da6dee"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-rust.git", revision: "77a3747266f4d621d0757825e6b11edcbf991ca5"),
        .package(url: "https://github.com/tree-sitter/tree-sitter-typescript.git", revision: "75b3874edb2dc714fb1fd77a32013d0f8699989f"),
        .package(url: "https://github.com/alex-pinkus/tree-sitter-swift.git", revision: "31d17fe7e818a2048c808b5c6fdc2dc792f4f5b5"),
        .package(url: "https://github.com/the-mikedavis/tree-sitter-diff.git", revision: "ada384ac7bfc1307f32de474620120add29998fb"),
        .package(url: "https://github.com/DerekStride/tree-sitter-sql.git", revision: "851e9cb257ba7c66cc8c14214a31c44d2f1e954e"),
        .package(url: "https://github.com/tree-sitter-grammars/tree-sitter-toml.git", revision: "64b56832c2cffe41758f28e05c756a3a98d16f41"),
        .package(url: "https://github.com/tree-sitter-grammars/tree-sitter-yaml.git", revision: "a1c4812a73ec5e089de8e441fdea3a921e8d5079"),
        .package(url: "https://github.com/tree-sitter-grammars/tree-sitter-markdown.git", revision: "a0a00f817d02412bd92c54d316f164d827b57b5c"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(
            url: "https://github.com/huggingface/swift-huggingface.git",
            from: "0.8.1",
            traits: ["Xet"]
        ),
        // Jinja (transitive via swift-transformers) — exposed for test target.
        // 2.4.0 broke swift-transformers ≤1.3.3 (ObjectKey change in Hub/Config.swift);
        // 2.4.1 restored source compatibility upstream, so no cap is needed.
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0")
    ],
    targets: [
        .target(
            name: "AFMKitCoreCompatibility",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit")
            ]
        ),
        .target(
            name: "AFMOpenAICompatCompatibility",
            dependencies: [
                .product(name: "AFMOpenAICompat", package: "AFMKit")
            ]
        ),
        .target(
            name: "AFMKitMLXCompatibility",
            dependencies: [
                .product(name: "AFMKitMLX", package: "AFMKit")
            ]
        ),
        .target(
            name: "AFMKitDwarfStarCompatibility",
            dependencies: [
                .product(name: "AFMKitDwarfStar", package: "AFMKit")
            ]
        ),
        .target(
            name: "AFMKitFoundationModels",
            dependencies: foundationModelsDependencies
        ),
        .target(
            name: "AFMKitFoundationModels27",
            dependencies: foundationModels27Dependencies
        ),
        .target(
            name: "AFMKitFoundationModels27DwarfStar",
            dependencies: dwarfStarFoundationModelsDependencies
        ),
        .target(
            name: "AFMKitServicesCompatibility",
            dependencies: [
                .product(name: "AFMKitServices", package: "AFMKit")
            ]
        ),
        .target(
            name: "AFMEvaluationHost",
            dependencies: [
                .product(name: "AFMEvalKit", package: "AFMKit")
            ],
            resources: [
                .copy("Resources/Evals")
            ]
        ),
        // Core library — all reusable inference/service/server code. Importable via SPM.
        .target(
            name: "AFMKit",
            dependencies: [
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMOpenAICompat", package: "AFMKit"),
                .product(name: "AFMKitInference", package: "AFMKit"),
                "AFMEvaluationHost",
                .product(name: "AFMKitMLX", package: "AFMKit"),
                "AFMKitFoundationModels",
                .product(name: "AFMKitServices", package: "AFMKit")
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
                .product(name: "Vapor", package: "vapor")
            ],
            swiftSettings: [
                .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-O"], .when(configuration: .release)),
                .unsafeFlags(["-file-prefix-map", "\(packageDir)/="], .when(configuration: .release))
            ]
        ),
        // Four upstream grammar manifests conditionally omit their external
        // scanners when SwiftPM evaluates them from a parent package.  Keep
        // the pinned scanner sources in an explicit target so an incomplete
        // grammar can never survive the final executable link.
        .target(
            name: "AFMTreeSitterScanners",
            dependencies: [
                .product(name: "TreeSitter", package: "tree-sitter")
            ],
            path: "Sources/AFMTreeSitterScanners",
            exclude: ["README.md", "LICENSES.md"],
            publicHeadersPath: ".",
            cSettings: [.headerSearchPath(".")],
            linkerSettings: []
        ),
        .target(
            name: "AFMTerminalUI",
            dependencies: [
                "AFMKit",
                .product(name: "AFMOpenAICompat", package: "AFMKit"),
                "AFMTreeSitterScanners",
                .product(name: "Markdown", package: "swift-markdown"),
                .product(name: "TreeSitter", package: "tree-sitter"),
                .product(name: "SwiftTreeSitter", package: "swift-tree-sitter"),
                .product(name: "TreeSitterBash", package: "tree-sitter-bash"),
                .product(name: "TreeSitterC", package: "tree-sitter-c"),
                .product(name: "TreeSitterCPP", package: "tree-sitter-cpp"),
                .product(name: "TreeSitterCSharp", package: "tree-sitter-c-sharp"),
                .product(name: "TreeSitterCSS", package: "tree-sitter-css"),
                .product(name: "TreeSitterGo", package: "tree-sitter-go"),
                .product(name: "TreeSitterHTML", package: "tree-sitter-html"),
                .product(name: "TreeSitterJava", package: "tree-sitter-java"),
                .product(name: "TreeSitterJavaScript", package: "tree-sitter-javascript"),
                .product(name: "TreeSitterJSON", package: "tree-sitter-json"),
                .product(name: "TreeSitterKotlin", package: "tree-sitter-kotlin"),
                .product(name: "TreeSitterPHP", package: "tree-sitter-php"),
                .product(name: "TreeSitterPython", package: "tree-sitter-python"),
                .product(name: "TreeSitterRuby", package: "tree-sitter-ruby"),
                .product(name: "TreeSitterRust", package: "tree-sitter-rust"),
                .product(name: "TreeSitterTypeScript", package: "tree-sitter-typescript"),
                .product(name: "TreeSitterSwift", package: "tree-sitter-swift"),
                .product(name: "TreeSitterDiff", package: "tree-sitter-diff"),
                .product(name: "TreeSitterSql", package: "tree-sitter-sql"),
                .product(name: "TreeSitterTOML", package: "tree-sitter-toml"),
                .product(name: "TreeSitterYAML", package: "tree-sitter-yaml"),
                .product(name: "TreeSitterMarkdown", package: "tree-sitter-markdown")
            ]
        ),
        // Thin CLI executable over AFMKit + AFMServer.
        .executableTarget(
            name: "AFMCLI",
            dependencies: [
                "AFMKit",
                "AFMTerminalUI",
                .product(name: "AFMKitCore", package: "AFMKit"),
                .product(name: "AFMKitDwarfStar", package: "AFMKit"),
                .product(name: "AFMKitMLX", package: "AFMKit"),
                "AFMServer",
                .product(name: "Vapor", package: "vapor"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
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
            name: "AFMKitServicesCompatibilityTests",
            dependencies: ["AFMKitServicesCompatibility"]
        ),
        .testTarget(
            name: "AFMEvaluationHostTests",
            dependencies: ["AFMEvaluationHost"]
        ),
        .testTarget(
            name: "AFMKitFoundationModels27DwarfStarCompatibilityTests",
            dependencies: ["AFMKitFoundationModels27DwarfStar"]
        ),
        .testTarget(
            name: "MacLocalAPITests",
            dependencies: [
                "AFMKit",
                "AFMKitFoundationModels",
                "AFMKitFoundationModels27",
                "AFMKitFoundationModels27DwarfStar",
                "AFMKitServicesCompatibility",
                "AFMServer",
                "AFMTerminalUI",
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "XCTVapor", package: "vapor"),
                .product(name: "VaporTesting", package: "vapor")
            ],
            swiftSettings: [
                // Xcode 27 Beta 3 reports a false circular reference while
                // optimizing the combined release test module. Product targets
                // remain fully optimized.
                .unsafeFlags(["-no-whole-module-optimization"], .when(configuration: .release)),
                .unsafeFlags(["-Onone"], .when(configuration: .release))
            ]
        )
    ],
    cxxLanguageStandard: .gnucxx17
)
