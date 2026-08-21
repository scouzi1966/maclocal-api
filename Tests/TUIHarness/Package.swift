// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "AFMTerminalUIHarness",
    platforms: [.macOS("26.0")],
    dependencies: [
        .package(name: "MacLocalAPI", path: "../..")
    ],
    targets: [
        .testTarget(
            name: "AFMTerminalUIHarnessTests",
            dependencies: [
                .product(name: "AFMTerminalUI", package: "MacLocalAPI")
            ],
            resources: [.copy("Snapshots")]
        )
    ]
)
