import Foundation
import XCTest
@testable import AFMKitMLX

final class AFMMLXVisionAssetQualificationTests: XCTestCase {
    private var directories: [URL] = []

    override func tearDownWithError() throws {
        for directory in directories {
            try? FileManager.default.removeItem(at: directory)
        }
        directories.removeAll()
    }

    func testPublishedQwenConfigurationWithIndexedVisionWeightsIsUsable() throws {
        let directory = try makeModelDirectory()
        let qualification = try qualify(directory)

        XCTAssertTrue(qualification.isConditionalGeneration)
        XCTAssertTrue(qualification.declaresVision)
        XCTAssertEqual(qualification.processorClass, "Qwen3VLProcessor")
        XCTAssertEqual(qualification.visionTensorCount, 33)
        XCTAssertTrue(qualification.missingAssets.isEmpty)
        XCTAssertTrue(qualification.isUsableQwenConditionalGeneration)
    }

    func testOptionalVisionFailuresDoNotChangeBaseCacheCompleteness() throws {
        let mutations: [(String, (inout [String: Any], URL) throws -> Void)] = [
            ("processor", { _, directory in
                try FileManager.default.removeItem(
                    at: directory.appendingPathComponent("preprocessor_config.json")
                )
            }),
            ("token IDs", { config, _ in
                config.removeValue(forKey: "image_token_id")
            }),
            ("vision config", { config, _ in
                config.removeValue(forKey: "vision_config")
            }),
            ("vision weights", { _, directory in
                try Self.writeJSON([
                    "weight_map": [
                        "language_model.layers.0.weight":
                            "model-00001-of-00001.safetensors"
                    ]
                ], to: directory.appendingPathComponent("model.safetensors.index.json"))
            }),
        ]

        for (label, mutate) in mutations {
            let directory = try makeModelDirectory()
            var config = Self.fixtureConfiguration()
            try mutate(&config, directory)
            try Self.writeJSON(config, to: directory.appendingPathComponent("config.json"))

            XCTAssertTrue(
                MLXCacheResolver(cacheRoot: nil).hasRequiredFiles(directory),
                "Base cache completeness changed for missing \(label)"
            )
            XCTAssertFalse(
                try qualify(directory).isAssetUsable,
                "Vision unexpectedly remained usable for missing \(label)"
            )
        }
    }

    func testMalformedIndexReportsMissingVisionWeights() throws {
        let directory = try makeModelDirectory()
        try Data("not json".utf8).write(
            to: directory.appendingPathComponent("model.safetensors.index.json")
        )

        let qualification = try qualify(directory)

        XCTAssertTrue(qualification.missingAssets.contains(.visionWeights))
    }

    func testIndexedCheckpointFailsWhenIndexReferencesMissingTensor() throws {
        let directory = try makeModelDirectory()
        var names = Self.requiredVisionTensorNames(depth: 2)
        names.remove("vision_tower.blocks.1.attn.qkv.weight")
        let weightMap = Dictionary(uniqueKeysWithValues: names.map {
            ($0, "model-00001-of-00001.safetensors")
        })
        try Self.writeJSON(
            ["weight_map": weightMap],
            to: directory.appendingPathComponent("model.safetensors.index.json")
        )
        try Self.writeSafetensorHeader(
            tensorNames: names,
            to: directory.appendingPathComponent("model-00001-of-00001.safetensors")
        )

        let qualification = try qualify(directory)

        XCTAssertEqual(qualification.visionTensorCount, 0)
        XCTAssertTrue(qualification.missingAssets.contains(.visionWeights))
    }

    func testIndexedCheckpointFailsWhenShardDoesNotContainMappedTensor() throws {
        let directory = try makeModelDirectory()
        let names = Self.requiredVisionTensorNames(depth: 2)
        let omitted = try XCTUnwrap(names.first)
        var shardNames = names
        shardNames.remove(omitted)
        var weightMap = Dictionary(uniqueKeysWithValues: names.map {
            ($0, "model-00001-of-00001.safetensors")
        })
        weightMap["language_model.layers.0.weight"] = "model-00001-of-00001.safetensors"
        try Self.writeJSON(
            ["weight_map": weightMap],
            to: directory.appendingPathComponent("model.safetensors.index.json")
        )
        try Self.writeSafetensorHeader(
            tensorNames: shardNames.union(["language_model.layers.0.weight"]),
            to: directory.appendingPathComponent("model-00001-of-00001.safetensors")
        )

        XCTAssertTrue(try qualify(directory).missingAssets.contains(.visionWeights))
    }

    func testIndexedCheckpointFailsWhenShardPayloadIsTruncated() throws {
        let directory = try makeModelDirectory()
        let shard = directory.appendingPathComponent("model-00001-of-00001.safetensors")
        var data = try Data(contentsOf: shard)
        data.removeLast()
        try data.write(to: shard)

        let qualification = try qualify(directory)

        XCTAssertTrue(qualification.missingAssets.contains(.visionWeights))
        XCTAssertFalse(qualification.isAssetUsable)
    }

    func testIndexedCheckpointFailsWhenShardContainsUnindexedTensor() throws {
        let directory = try makeModelDirectory()
        let names = Self.requiredVisionTensorNames(depth: 2)
            .union(["language_model.layers.0.weight"])
        try Self.writeSafetensorHeader(
            tensorNames: names.union(["vision_tower.unindexed.weight"]),
            to: directory.appendingPathComponent("model-00001-of-00001.safetensors")
        )

        let qualification = try qualify(directory)

        XCTAssertTrue(qualification.missingAssets.contains(.visionWeights))
        XCTAssertFalse(qualification.isAssetUsable)
    }

    func testIndexedCheckpointFailsWhenTensorPayloadRangesOverlap() throws {
        let directory = try makeModelDirectory()
        let first = "vision_tower.patch_embed.proj.bias"
        let second = "vision_tower.patch_embed.proj.weight"
        try Self.writeSafetensorHeader(
            tensorNames: Self.requiredVisionTensorNames(depth: 2)
                .union(["language_model.layers.0.weight"]),
            offsetOverrides: [
                first: [0, 2],
                second: [0, 2],
            ],
            to: directory.appendingPathComponent("model-00001-of-00001.safetensors")
        )

        let qualification = try qualify(directory)

        XCTAssertTrue(qualification.missingAssets.contains(.visionWeights))
        XCTAssertFalse(qualification.isAssetUsable)
    }

    func testMalformedQwenVisionConfigurationFailsQualification() throws {
        let directory = try makeModelDirectory()
        var config = Self.fixtureConfiguration()
        var vision = try XCTUnwrap(config["vision_config"] as? [String: Any])
        vision.removeValue(forKey: "hidden_size")
        config["vision_config"] = vision
        try Self.writeJSON(config, to: directory.appendingPathComponent("config.json"))

        let qualification = try qualify(directory)

        XCTAssertTrue(qualification.missingAssets.contains(.visionConfiguration))
        XCTAssertFalse(qualification.isAssetUsable)
    }

    func testMXFPPackedVisionWeightRequiresScaleTensor() throws {
        let directory = try makeModelDirectory()
        let packedWeight = "vision_tower.blocks.0.attn.proj.weight"
        try rewriteVisionShard(
            in: directory,
            metadata: [packedWeight: ("U32", [1])]
        )

        let qualification = try qualify(directory)

        XCTAssertTrue(qualification.missingAssets.contains(.visionWeights))
        XCTAssertFalse(qualification.isAssetUsable)
    }

    func testMXFPPackedVisionWeightWithScaleQualifiesWithoutAffineBiases() throws {
        let directory = try makeModelDirectory()
        let base = "vision_tower.blocks.0.attn.proj"
        try rewriteVisionShard(
            in: directory,
            additionalNames: ["\(base).scales"],
            metadata: [
                "\(base).weight": ("U32", [1]),
                "\(base).scales": ("U8", [1]),
            ]
        )

        let qualification = try qualify(directory)

        XCTAssertTrue(qualification.isAssetUsable)
        XCTAssertEqual(qualification.visionTensorCount, 34)
    }

    func testAffinePackedVisionWeightRequiresScalesAndBiases() throws {
        let directory = try makeModelDirectory()
        var config = Self.fixtureConfiguration()
        config["quantization"] = ["group_size": 32, "bits": 4, "mode": "affine"]
        config["quantization_config"] = [
            "group_size": 32, "bits": 4, "mode": "affine",
        ]
        try Self.writeJSON(config, to: directory.appendingPathComponent("config.json"))
        let base = "vision_tower.blocks.0.attn.proj"
        try rewriteVisionShard(
            in: directory,
            additionalNames: ["\(base).scales"],
            metadata: [
                "\(base).weight": ("U32", [1]),
                "\(base).scales": ("F16", [1]),
            ]
        )
        XCTAssertFalse(try qualify(directory).isAssetUsable)

        try rewriteVisionShard(
            in: directory,
            additionalNames: ["\(base).scales", "\(base).biases"],
            metadata: [
                "\(base).weight": ("U32", [1]),
                "\(base).scales": ("F16", [1]),
                "\(base).biases": ("F16", [1]),
            ]
        )
        XCTAssertTrue(try qualify(directory).isAssetUsable)
    }

    func testMalformedQwenProcessorMetadataIsNotUsable() throws {
        let directory = try makeModelDirectory()
        try Self.writeJSON(
            ["processor_class": "Qwen3VLProcessor"],
            to: directory.appendingPathComponent("preprocessor_config.json")
        )

        let qualification = try qualify(directory)

        XCTAssertNil(qualification.processorClass)
        XCTAssertTrue(qualification.missingAssets.contains(.processorConfiguration))
        XCTAssertFalse(qualification.isAssetUsable)
    }

    func testStandaloneSafetensorHeaderQualifiesVisionWeights() throws {
        let directory = try makeModelDirectory(indexed: false)
        try Self.writeSafetensorHeader(
            tensorNames: Self.requiredVisionTensorNames(depth: 2)
                .union(["language_model.embed_tokens.weight"]),
            to: directory.appendingPathComponent("weights.safetensors")
        )

        let qualification = try qualify(directory)

        XCTAssertEqual(qualification.visionTensorCount, 33)
        XCTAssertTrue(qualification.isAssetUsable)
    }

    func testSnapshotFingerprintInvalidatesCachedQualification() throws {
        let directory = try makeModelDirectory()
        let validator = AFMMLXVisionAssetValidator()
        let architecture = try AFMMLXModelArchitecture.preflightConfiguration(
            in: directory,
            modelID: "fixture"
        )
        let first = validator.qualify(
            modelDirectory: directory,
            architecture: architecture
        )

        try Self.writeJSON([
            "weight_map": [
                "language_model.layers.0.weight": "model-00001-of-00001.safetensors"
            ]
        ], to: directory.appendingPathComponent("model.safetensors.index.json"))
        let second = validator.qualify(
            modelDirectory: directory,
            architecture: architecture
        )

        XCTAssertNotEqual(first.snapshotIdentity, second.snapshotIdentity)
        XCTAssertTrue(first.isAssetUsable)
        XCTAssertFalse(second.isAssetUsable)
    }

    private func qualify(_ directory: URL) throws -> AFMMLXVisionAssetQualification {
        let architecture = try AFMMLXModelArchitecture.preflightConfiguration(
            in: directory,
            modelID: "fixture"
        )
        return AFMMLXVisionAssetValidator().qualify(
            modelDirectory: directory,
            architecture: architecture
        )
    }

    private func makeModelDirectory(indexed: Bool = true) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "afm-qwen-vision-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        directories.append(directory)
        try Self.writeJSON(
            Self.fixtureConfiguration(),
            to: directory.appendingPathComponent("config.json")
        )
        try Self.writeJSON(
            [
                "processor_class": "Qwen3VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessorFast",
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "merge_size": 2,
                "patch_size": 16,
                "temporal_patch_size": 2,
            ],
            to: directory.appendingPathComponent("preprocessor_config.json")
        )
        if indexed {
            let tensorNames = Self.requiredVisionTensorNames(depth: 2)
                .union(["language_model.layers.0.weight"])
            let weightMap = Dictionary(uniqueKeysWithValues: tensorNames.map {
                ($0, "model-00001-of-00001.safetensors")
            })
            try Self.writeJSON(
                ["weight_map": weightMap],
                to: directory.appendingPathComponent("model.safetensors.index.json")
            )
            try Self.writeSafetensorHeader(
                tensorNames: tensorNames,
                to: directory.appendingPathComponent("model-00001-of-00001.safetensors")
            )
        }
        return directory
    }

    private func rewriteVisionShard(
        in directory: URL,
        additionalNames: Set<String> = [],
        metadata: [String: (dtype: String, shape: [Int])]
    ) throws {
        let tensorNames = Self.requiredVisionTensorNames(depth: 2)
            .union(["language_model.layers.0.weight"])
            .union(additionalNames)
        let weightMap = Dictionary(uniqueKeysWithValues: tensorNames.map {
            ($0, "model-00001-of-00001.safetensors")
        })
        try Self.writeJSON(
            ["weight_map": weightMap],
            to: directory.appendingPathComponent("model.safetensors.index.json")
        )
        try Self.writeSafetensorHeader(
            tensorNames: tensorNames,
            metadata: metadata,
            to: directory.appendingPathComponent("model-00001-of-00001.safetensors")
        )
    }

    private static func writeJSON(_ object: Any, to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            .write(to: url)
    }

    private static func writeSafetensorHeader(
        tensorNames: Set<String>,
        metadata: [String: (dtype: String, shape: [Int])] = [:],
        offsetOverrides: [String: [Int]] = [:],
        to url: URL
    ) throws {
        var offset = 0
        let entries = Dictionary(uniqueKeysWithValues: tensorNames.sorted().map { name in
            let tensor = metadata[name] ?? ("F16", [1])
            let byteWidth: Int
            switch tensor.dtype {
            case "U8", "I8": byteWidth = 1
            case "U32", "I32": byteWidth = 4
            default: byteWidth = 2
            }
            let byteCount = tensor.shape.reduce(byteWidth, *)
            defer { offset += byteCount }
            return (
                name,
                [
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": offsetOverrides[name] ?? [offset, offset + byteCount],
                ] as [String: Any]
            )
        })
        let header = try JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        data.append(Data(repeating: 0, count: offset))
        try data.write(to: url)
    }

    private static func fixtureConfiguration() -> [String: Any] {
        var config = Qwen38PublishedConfigFixture.mxfp8
        var vision = config["vision_config"] as! [String: Any]
        vision["depth"] = 2
        vision["deepstack_visual_indexes"] = []
        config["vision_config"] = vision
        return config
    }

    private static func requiredVisionTensorNames(depth: Int) -> Set<String> {
        var names: Set<String> = [
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.patch_embed.proj.bias",
            "vision_tower.pos_embed.weight",
        ]
        let blockSuffixes = [
            "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.linear_fc1.bias", "mlp.linear_fc1.weight",
            "mlp.linear_fc2.bias", "mlp.linear_fc2.weight",
            "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight",
        ]
        for block in 0..<depth {
            for suffix in blockSuffixes {
                names.insert("vision_tower.blocks.\(block).\(suffix)")
            }
        }
        for suffix in [
            "linear_fc1.bias", "linear_fc1.weight",
            "linear_fc2.bias", "linear_fc2.weight", "norm.bias", "norm.weight",
        ] {
            names.insert("vision_tower.merger.\(suffix)")
        }
        return names
    }
}
