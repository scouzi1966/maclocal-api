import XCTest

@testable import MacLocalAPI

final class EmbeddingModelRegistryTests: XCTestCase {
    func testAppleEntriesResolve() {
        let registry = EmbeddingModelRegistry()

        let english = registry.resolve(modelID: EmbeddingModelRegistry.defaultModelID)
        let multilingual = registry.resolve(modelID: EmbeddingModelRegistry.multilingualModelID)

        XCTAssertEqual(english?.id, EmbeddingModelRegistry.defaultModelID)
        XCTAssertEqual(english?.backend, .nlContextual)
        XCTAssertEqual(multilingual?.id, EmbeddingModelRegistry.multilingualModelID)
        XCTAssertEqual(multilingual?.backend, .nlContextual)
    }

    func testUnknownModelReturnsNil() {
        let registry = EmbeddingModelRegistry()

        let entry = registry.resolve(modelID: "unknown-model")

        XCTAssertNil(entry)
    }

    func testWhitespaceModelIDReturnsNil() {
        let registry = EmbeddingModelRegistry()

        XCTAssertNil(registry.resolve(modelID: ""))
        XCTAssertNil(registry.resolve(modelID: "   "))
    }
}
