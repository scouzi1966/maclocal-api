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
        XCTAssertEqual(qualification.visionTensorCount, 2)
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
            var config = Qwen38PublishedConfigFixture.mxfp8
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

    func testStandaloneSafetensorHeaderQualifiesVisionWeights() throws {
        let directory = try makeModelDirectory(indexed: false)
        try Self.writeSafetensorHeader(
            tensorNames: [
                "vision_tower.patch_embed.proj.weight",
                "language_model.embed_tokens.weight",
            ],
            to: directory.appendingPathComponent("weights.safetensors")
        )

        let qualification = try qualify(directory)

        XCTAssertEqual(qualification.visionTensorCount, 1)
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
            Qwen38PublishedConfigFixture.mxfp8,
            to: directory.appendingPathComponent("config.json")
        )
        try Self.writeJSON(
            ["processor_class": "Qwen3VLProcessor"],
            to: directory.appendingPathComponent("preprocessor_config.json")
        )
        if indexed {
            try Self.writeJSON([
                "weight_map": [
                    "vision_tower.blocks.0.attn.qkv.weight":
                        "model-00001-of-00001.safetensors",
                    "model.visual.patch_embed.proj.weight":
                        "model-00001-of-00001.safetensors",
                    "language_model.layers.0.weight":
                        "model-00001-of-00001.safetensors",
                ]
            ], to: directory.appendingPathComponent("model.safetensors.index.json"))
            try Data().write(
                to: directory.appendingPathComponent("model-00001-of-00001.safetensors")
            )
        }
        return directory
    }

    private static func writeJSON(_ object: Any, to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            .write(to: url)
    }

    private static func writeSafetensorHeader(
        tensorNames: [String],
        to url: URL
    ) throws {
        let entries = Dictionary(uniqueKeysWithValues: tensorNames.map {
            ($0, ["dtype": "F16", "shape": [1], "data_offsets": [0, 0]] as [String: Any])
        })
        let header = try JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        try data.write(to: url)
    }
}
