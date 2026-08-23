import XCTest

@testable import AFMKit
import AFMKitServices
@testable import AFMServer

final class EmbeddingModelRegistryTests: XCTestCase {
    func testAppleEntriesResolve() {
        let registry = EmbeddingModelRegistry()

        let english = registry.resolve(modelID: EmbeddingModelRegistry.defaultModelID)
        let multilingual = registry.resolve(modelID: "apple-nl-contextual-multi")

        XCTAssertEqual(english?.id, EmbeddingModelRegistry.defaultModelID)
        XCTAssertEqual(english?.backend, .nlContextual)
        XCTAssertEqual(multilingual?.id, "apple-nl-contextual-multi")
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
