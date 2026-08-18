// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "Issue192TelemetryClient",
    platforms: [.macOS("26.0")],
    dependencies: [
        .package(name: "MacLocalAPI", path: "../../..")
    ],
    targets: [
        .testTarget(
            name: "Issue192TelemetryClientTests",
            dependencies: [
                .product(name: "AFMKitCore", package: "MacLocalAPI"),
                .product(name: "AFMOpenAICompat", package: "MacLocalAPI"),
                .product(name: "AFMKitServices", package: "MacLocalAPI"),
            ]
        )
    ]
)
