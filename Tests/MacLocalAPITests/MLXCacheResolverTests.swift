import Foundation
@testable import AFMKit
@testable import AFMKitMLX
import XCTest

final class MLXCacheResolverTests: XCTestCase {
    func testShardedSnapshotIsCompleteOnlyWhenEveryIndexedShardExists() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("afmkit-sharded-model-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
        let index: [String: Any] = [
            "weight_map": [
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors"
            ]
        ]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))

        let resolver = MLXCacheResolver()
        XCTAssertFalse(resolver.hasRequiredFiles(root))

        try Data().write(to: root.appendingPathComponent("model-00001-of-00002.safetensors"))
        XCTAssertFalse(resolver.hasRequiredFiles(root))

        try Data().write(to: root.appendingPathComponent("model-00002-of-00002.safetensors"))
        XCTAssertTrue(resolver.hasRequiredFiles(root))
    }
}
