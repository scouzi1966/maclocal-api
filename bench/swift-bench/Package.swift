// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "swift-bench",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.3"),
        // Use local vendor copy (same code as afm uses, WITH our patches applied)
        .package(path: "../../vendor/mlx-swift-lm"),
    ],
    targets: [
        .executableTarget(
            name: "swift-bench",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
    ]
)
