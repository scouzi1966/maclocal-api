import Foundation
import XCTest
@testable import AFMKitDwarfStar
@testable import AFMKitMLX

final class AFMDwarfStarCheckpointTests: XCTestCase {
    func testExternalAFMCheckpointHeadersCanBeCataloged() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["AFM_DWARFSTAR_REAL_CHECKPOINT_TEST"] == "1" else {
            throw XCTSkip("Set AFM_DWARFSTAR_REAL_CHECKPOINT_TEST=1 for the external checkpoint catalog test.")
        }
        let path = environment["AFM_DWARFSTAR_REAL_CHECKPOINT"]
            ?? "/Volumes/edata/models/vesta-test-cache/deepseek-ai/DeepSeek-V4-Flash-0731-AFM-MLX"
        let catalog = try AFMDwarfStarCheckpointCatalog(
            checkpointURL: URL(fileURLWithPath: path, isDirectory: true)
        )

        XCTAssertEqual(catalog.shardPaths.count, 48)
        XCTAssertGreaterThan(catalog.tensors.count, 1_900)
        XCTAssertGreaterThan(catalog.totalTensorBytes, 100_000_000_000)
        XCTAssertNotNil(catalog.tensor(named: "model.embed_tokens.weight"))
    }

    func testCatalogResolvesFileBackedTensorLocations() throws {
        let fixture = try makeFixture(executorReady: true)
        defer { try? FileManager.default.removeItem(at: fixture) }

        let catalog = try AFMDwarfStarCheckpointCatalog(checkpointURL: fixture)
        try catalog.requireExecutorReady()

        XCTAssertEqual(catalog.layout.executorLayoutVersion, 3)
        XCTAssertEqual(catalog.shardPaths.count, 1)
        XCTAssertEqual(catalog.tensors.count, 2)
        XCTAssertEqual(catalog.totalTensorBytes, 12)

        let first = try XCTUnwrap(catalog.tensor(named: "model.embed_tokens.weight"))
        XCTAssertEqual(first.dtype, "F16")
        XCTAssertEqual(first.shape, [2, 2])
        XCTAssertEqual(first.byteCount, 8)
        let second = try XCTUnwrap(catalog.tensor(named: "model.norm.weight"))
        XCTAssertTrue(first.fileOffset.isMultiple(of: 32))
        XCTAssertTrue(second.fileOffset.isMultiple(of: 32))
        XCTAssertGreaterThan(second.fileOffset, first.fileOffset + first.byteCount)
        XCTAssertEqual(second.byteCount, 4)
    }

    func testExecutorAlignmentRewritePreservesTensorBytes() throws {
        let fixture = try makeFixture(executorReady: false)
        defer { try? FileManager.default.removeItem(at: fixture) }
        let shard = fixture.appendingPathComponent("model-00001-of-00001.safetensors")
        let before = try readRealTensorPayloads(from: shard)

        try AlignedSafetensorRewriter.rewriteCheckpoint(at: fixture)

        let after = try readRealTensorPayloads(from: shard)
        XCTAssertEqual(after, before)
        let header = try readSafetensorHeader(from: shard)
        XCTAssertTrue(header.payloadStart.isMultiple(of: 4_096))
        for (name, value) in header.object
        where name != "__metadata__" && !name.hasPrefix("__afm_padding_") {
            let metadata = try XCTUnwrap(value as? [String: Any])
            let offsets = try XCTUnwrap(metadata["data_offsets"] as? [Int])
            XCTAssertTrue((header.payloadStart + offsets[0]).isMultiple(of: 32), name)
        }
        let config = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: fixture.appendingPathComponent("config.json")))
                as? [String: Any])
        XCTAssertEqual(config["afm_dwarfstar_executor_layout_version"] as? Int, 3)
        XCTAssertEqual(config["afm_dwarfstar_tensor_alignment"] as? Int, 32)
    }

    func testCatalogRejectsCheckpointWithoutExecutorLayoutContract() throws {
        let fixture = try makeFixture(executorReady: false)
        defer { try? FileManager.default.removeItem(at: fixture) }

        let catalog = try AFMDwarfStarCheckpointCatalog(checkpointURL: fixture)
        XCTAssertFalse(catalog.layout.isExecutorReady)
        XCTAssertThrowsError(try catalog.requireExecutorReady()) { error in
            guard case AFMDwarfStarCheckpointCatalog.CatalogError.unsupportedLayout = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testCatalogRejectsIndexedTensorMissingFromShard() throws {
        let fixture = try makeFixture(executorReady: true)
        defer { try? FileManager.default.removeItem(at: fixture) }
        let indexURL = fixture.appendingPathComponent("model.safetensors.index.json")
        try writeJSON([
            "weight_map": ["model.missing.weight": "model-00001-of-00001.safetensors"]
        ], to: indexURL)

        XCTAssertThrowsError(try AFMDwarfStarCheckpointCatalog(checkpointURL: fixture)) { error in
            XCTAssertTrue(error.localizedDescription.contains("absent"))
        }
    }

    func testCatalogRejectsShardPathTraversal() throws {
        let fixture = try makeFixture(executorReady: true)
        defer { try? FileManager.default.removeItem(at: fixture) }
        let indexURL = fixture.appendingPathComponent("model.safetensors.index.json")
        try writeJSON([
            "weight_map": ["model.embed_tokens.weight": "../outside.safetensors"]
        ], to: indexURL)

        XCTAssertThrowsError(try AFMDwarfStarCheckpointCatalog(checkpointURL: fixture)) { error in
            XCTAssertTrue(error.localizedDescription.contains("Unsafe"))
        }
    }

    func testGGUFAliasProjectsOriginalTensorBytesWithoutCopyingPayload() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-dwarfstar-gguf-alias-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("source.gguf")
        let layout = try writeGGUF(to: source)
        let metadata = root.appendingPathComponent("projection.gguf")
        let projection = try AFMDwarfStarProjection.buildGGUFAlias(
            ggufURL: source,
            metadataOutputURL: metadata)

        XCTAssertEqual(projection.regions.count, 1)
        XCTAssertEqual(projection.regions[0].path, source.path)
        XCTAssertEqual(projection.regions[0].fileOffset, 0)
        XCTAssertTrue(projection.regions[0].virtualOffset.isMultiple(of: UInt64(getpagesize())))
        XCTAssertTrue(projection.virtualSize.isMultiple(of: UInt64(getpagesize())))

        let projected = try Data(contentsOf: metadata)
        XCTAssertEqual(
            projected.readUInt64(at: layout.relativeOffsetPositions[0]),
            projection.regions[0].virtualOffset)
        XCTAssertEqual(
            projected.readUInt64(at: layout.relativeOffsetPositions[1]),
            projection.regions[0].virtualOffset + 8)
    }

    func testAFMProjectionPointsAtSafetensorPayloadOffsets() throws {
        let fixture = try makeFixture(executorReady: true)
        defer { try? FileManager.default.removeItem(at: fixture) }
        let template = fixture.appendingPathComponent("template.gguf")
        let layout = try writeGGUF(to: template)
        let metadata = fixture.appendingPathComponent("projection.gguf")

        let projection = try AFMDwarfStarProjection.build(
            checkpointURL: fixture,
            templateGGUF: template,
            metadataOutputURL: metadata)
        let catalog = try AFMDwarfStarCheckpointCatalog(checkpointURL: fixture)
        let embedding = try XCTUnwrap(catalog.tensor(named: "model.embed_tokens.weight"))
        let norm = try XCTUnwrap(catalog.tensor(named: "model.norm.weight"))
        let region = try XCTUnwrap(projection.regions.first)
        let projected = try Data(contentsOf: metadata)

        XCTAssertEqual(
            projected.readUInt64(at: layout.relativeOffsetPositions[0]) + layout.tensorDataOffset,
            region.virtualOffset + embedding.fileOffset)
        XCTAssertEqual(
            projected.readUInt64(at: layout.relativeOffsetPositions[1]) + layout.tensorDataOffset,
            region.virtualOffset + norm.fileOffset)
    }

    func testAFMProjectionRejectsTemplateByteCountMismatch() throws {
        let fixture = try makeFixture(executorReady: true)
        defer { try? FileManager.default.removeItem(at: fixture) }
        let template = fixture.appendingPathComponent("mismatch.gguf")
        _ = try writeGGUF(to: template, normElements: 3)

        XCTAssertThrowsError(try AFMDwarfStarProjection.build(
            checkpointURL: fixture,
            templateGGUF: template,
            metadataOutputURL: fixture.appendingPathComponent("projection.gguf"))) { error in
            XCTAssertTrue(error.localizedDescription.contains("expects 6 bytes"))
        }
    }

    private func makeFixture(executorReady: Bool) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-dwarfstar-checkpoint-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        var config: [String: Any] = [
            "model_type": "deepseek_v4",
            "afm_native_checkpoint": true,
        ]
        if executorReady {
            config["afm_q8_0"] = true
            config["afm_dwarfstar_mxfp4_layout"] = true
            config["afm_dwarfstar_mxfp4_packed"] = true
            config["afm_dwarfstar_executor_layout_version"] = 3
        }
        try writeJSON(config, to: root.appendingPathComponent("config.json"))

        let shardName = "model-00001-of-00001.safetensors"
        let tensors: [String: Any]
        let payload: Data
        if executorReady {
            tensors = [
                "model.embed_tokens.weight": ["dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]],
                "__afm_padding_000000": ["dtype": "U8", "shape": [24], "data_offsets": [8, 32]],
                "model.norm.weight": ["dtype": "F16", "shape": [2], "data_offsets": [32, 36]],
            ]
            payload = Data(repeating: 0x2a, count: 36)
        } else {
            tensors = [
                "model.embed_tokens.weight": ["dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]],
                "model.norm.weight": ["dtype": "F16", "shape": [2], "data_offsets": [8, 12]],
            ]
            payload = Data(repeating: 0x2a, count: 12)
        }
        try writeSafetensor(
            tensors: tensors,
            payload: payload,
            headerAlignment: executorReady ? 4_096 : 8,
            to: root.appendingPathComponent(shardName))
        try writeJSON([
            "weight_map": [
                "model.embed_tokens.weight": shardName,
                "model.norm.weight": shardName,
            ]
        ], to: root.appendingPathComponent("model.safetensors.index.json"))
        return root
    }

    private func writeSafetensor(
        tensors: [String: Any], payload: Data, headerAlignment: Int, to url: URL
    ) throws {
        var header = try JSONSerialization.data(withJSONObject: tensors, options: [.sortedKeys])
        while !(8 + header.count).isMultiple(of: headerAlignment) { header.append(0x20) }
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        data.append(payload)
        try data.write(to: url)
    }

    private func readSafetensorHeader(
        from url: URL
    ) throws -> (payloadStart: Int, object: [String: Any]) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try XCTUnwrap(try handle.read(upToCount: 8))
        let size = prefix.enumerated().reduce(UInt64(0)) { result, item in
            result | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        let data = try XCTUnwrap(try handle.read(upToCount: Int(size)))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        return (8 + Int(size), object)
    }

    private func readRealTensorPayloads(from url: URL) throws -> [String: Data] {
        let header = try readSafetensorHeader(from: url)
        let file = try Data(contentsOf: url)
        var result: [String: Data] = [:]
        for (name, value) in header.object
        where name != "__metadata__" && !name.hasPrefix("__afm_padding_") {
            let metadata = try XCTUnwrap(value as? [String: Any])
            let offsets = try XCTUnwrap(metadata["data_offsets"] as? [Int])
            let start = header.payloadStart + offsets[0]
            let end = header.payloadStart + offsets[1]
            result[name] = file[start..<end]
        }
        return result
    }

    private func writeJSON(_ object: Any, to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        try data.write(to: url)
    }

    private struct GGUFLayout {
        let tensorDataOffset: UInt64
        let relativeOffsetPositions: [Int]
    }

    @discardableResult
    private func writeGGUF(to url: URL, normElements: UInt64 = 2) throws -> GGUFLayout {
        var data = Data("GGUF".utf8)
        data.appendLittleEndian(UInt32(3))
        data.appendLittleEndian(UInt64(2))
        data.appendLittleEndian(UInt64(1))
        data.appendGGUFString("general.alignment")
        data.appendLittleEndian(UInt32(4))
        data.appendLittleEndian(UInt32(32))

        var positions: [Int] = []
        data.appendGGUFString("token_embd.weight")
        data.appendLittleEndian(UInt32(2))
        data.appendLittleEndian(UInt64(2))
        data.appendLittleEndian(UInt64(2))
        data.appendLittleEndian(UInt32(1))
        positions.append(data.count)
        data.appendLittleEndian(UInt64(0))

        data.appendGGUFString("output_norm.weight")
        data.appendLittleEndian(UInt32(1))
        data.appendLittleEndian(normElements)
        data.appendLittleEndian(UInt32(1))
        positions.append(data.count)
        data.appendLittleEndian(UInt64(8))

        while !data.count.isMultiple(of: 32) { data.append(0) }
        let tensorDataOffset = UInt64(data.count)
        data.append(Data((0..<8).map(UInt8.init)))
        data.append(Data(repeating: 0x7f, count: Int(normElements * 2)))
        try data.write(to: url)
        return GGUFLayout(
            tensorDataOffset: tensorDataOffset,
            relativeOffsetPositions: positions)
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }

    mutating func appendGGUFString(_ value: String) {
        appendLittleEndian(UInt64(value.utf8.count))
        append(contentsOf: value.utf8)
    }

    func readUInt64(at offset: Int) -> UInt64 {
        self[offset..<(offset + 8)].withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
    }
}
