import XCTest

@testable import MacLocalAPI

final class EmbeddingModelRegistryTests: XCTestCase {
    func testAppleEntriesResolve() throws {
        let registry = EmbeddingModelRegistry()

        let english = try registry.resolve(modelID: EmbeddingModelRegistry.defaultModelID)
        let multilingual = try registry.resolve(modelID: EmbeddingModelRegistry.multilingualModelID)

        XCTAssertEqual(english?.id, EmbeddingModelRegistry.defaultModelID)
        XCTAssertEqual(english?.backend, .nlContextual)
        XCTAssertEqual(multilingual?.id, EmbeddingModelRegistry.multilingualModelID)
        XCTAssertEqual(multilingual?.backend, .nlContextual)
    }

    func testUnknownModelReturnsNilWithoutMLXOverride() throws {
        let registry = EmbeddingModelRegistry()

        let entry = try registry.resolve(modelID: "unknown-model")

        XCTAssertNil(entry)
    }

    func testResolveMLXEntryFromMockedLocalDirectory() throws {
        let modelDirectory = try makeMockMLXModelDirectory()
        let registry = EmbeddingModelRegistry()

        let entry = try registry.resolve(modelID: modelDirectory.path, backendOverride: .mlx)

        XCTAssertEqual(entry?.id, modelDirectory.path)
        XCTAssertEqual(entry?.backend, .mlx)
        XCTAssertEqual(entry?.nativeDimension, 384)
        XCTAssertEqual(entry?.pooling, .mean)
        XCTAssertEqual(entry?.maxInputTokens, 512)
    }

    private func makeMockMLXModelDirectory() throws -> URL {
        let modelDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("embedding-model-registry-tests")
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)

        let configURL = modelDirectory.appendingPathComponent("config.json")
        let config = """
        {
          "model_type": "bert",
          "hidden_size": 384,
          "max_position_embeddings": 512
        }
        """
        try Data(config.utf8).write(to: configURL)

        let weightsURL = modelDirectory.appendingPathComponent("model.safetensors")
        try Data().write(to: weightsURL)

        let poolingDirectory = modelDirectory.appendingPathComponent("1_Pooling")
        try FileManager.default.createDirectory(at: poolingDirectory, withIntermediateDirectories: true)

        let poolingURL = poolingDirectory.appendingPathComponent("config.json")
        let poolingConfig = """
        {
          "word_embedding_dimension": 384,
          "pooling_mode_cls_token": false,
          "pooling_mode_mean_tokens": true,
          "pooling_mode_max_tokens": false,
          "pooling_mode_lasttoken": false
        }
        """
        try Data(poolingConfig.utf8).write(to: poolingURL)

        addTeardownBlock {
            try? FileManager.default.removeItem(at: modelDirectory)
        }

        return modelDirectory
    }
}
