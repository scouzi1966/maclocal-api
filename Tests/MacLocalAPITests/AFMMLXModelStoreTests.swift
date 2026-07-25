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

    private func makeModel(at directory: URL) throws {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try Data("{}".utf8).write(
            to: directory.appendingPathComponent("config.json")
        )
        try Data("weights".utf8).write(
            to: directory.appendingPathComponent("weights.safetensors")
        )
    }
}
