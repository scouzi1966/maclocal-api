import Foundation
import XCTest
@testable import AFMKitMLX

final class AFMMLXModelStoreTests: XCTestCase {
    func testLocalDescriptorsValidateDeduplicateAndPreserveOrder() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try makeModel(at: root.appendingPathComponent("models/org/first"))
        try makeModel(at: root.appendingPathComponent("models/org/second"))

        let store = AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: root))
        let descriptors = store.localDescriptors(
            for: ["org/second", "missing/model", "org/first", "org/second"]
        )

        XCTAssertEqual(descriptors.map(\.modelID.rawValue), ["org/second", "org/first"])
        XCTAssertTrue(store.isAvailableLocally("org/first"))
        XCTAssertFalse(store.isAvailableLocally("missing/model"))
    }

    func testAbsoluteModelDirectoryUsesSharedValidation() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try makeModel(at: directory)

        let store = AFMMLXModelStore()

        XCTAssertEqual(store.localDirectory(for: directory.path)?.path, directory.path)
        XCTAssertEqual(store.descriptor(for: directory.path).requiresNetwork, false)
    }

    func testDiscoveryReturnsTypedFlatAndHuggingFaceModels() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let flat = root.appendingPathComponent("flat", isDirectory: true)
        let hub = root.appendingPathComponent("hub", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try makeModel(
            at: flat.appendingPathComponent("org/flat-model"),
            contextWindow: 16_384
        )
        try makeModel(
            at: hub.appendingPathComponent(
                "models--org--hub-model/snapshots/revision"
            ),
            contextWindow: 32_768
        )
        try FileManager.default.createDirectory(
            at: flat.appendingPathComponent("org/incomplete"),
            withIntermediateDirectories: true
        )

        let store = AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: flat))
        let models = store.discoverLocalModels(
            in: [
                .init(
                    directory: flat,
                    layout: .flat,
                    origin: .configuredCache
                ),
                .init(
                    directory: hub,
                    layout: .huggingFaceHub,
                    origin: .huggingFace
                )
            ]
        )

        XCTAssertEqual(
            models.map(\.id.rawValue),
            ["org/flat-model", "org/hub-model"]
        )
        XCTAssertEqual(models[0].loadIdentifier, "org/flat-model")
        XCTAssertEqual(models[0].descriptor.contextWindow, 16_384)
        XCTAssertEqual(models[0].origin, .configuredCache)
        XCTAssertEqual(
            models[1].loadIdentifier,
            models[1].localDirectory.path
        )
        XCTAssertEqual(models[1].descriptor.contextWindow, 32_768)
        XCTAssertEqual(models[1].origin, .huggingFace)
    }

    func testDiscoveryDeduplicatesCanonicalIDByLocationPrecedence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let first = root.appendingPathComponent("first", isDirectory: true)
        let second = root.appendingPathComponent("second", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try makeModel(at: first.appendingPathComponent("org/model"))
        try makeModel(at: second.appendingPathComponent("org/model"))

        let models = AFMMLXModelStore().discoverLocalModels(
            in: [
                .init(
                    directory: first,
                    layout: .flat,
                    origin: .swiftHub
                ),
                .init(
                    directory: second,
                    layout: .flat,
                    origin: .lmStudio
                )
            ]
        )

        XCTAssertEqual(models.count, 1)
        XCTAssertEqual(models.first?.origin, .swiftHub)
        XCTAssertEqual(
            models.first?.localDirectory.path,
            first.appendingPathComponent("org/model").path
        )
    }

    private func makeModel(
        at directory: URL,
        contextWindow: Int? = nil
    ) throws {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let config: [String: Any] = contextWindow.map {
            ["max_position_embeddings": $0]
        } ?? [:]
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json")
        )
        try Data("weights".utf8).write(
            to: directory.appendingPathComponent("weights.safetensors")
        )
    }
}
