import XCTest
import AFMKitServicesCompatibility

final class AFMKitServicesCompatibilityTests: XCTestCase {
    func testLegacyProductMakesUpstreamServiceModuleImportable() {
        let registry = EmbeddingModelRegistry()
        XCTAssertEqual(
            registry.resolve(modelID: EmbeddingModelRegistry.defaultModelID)?.id,
            EmbeddingModelRegistry.defaultModelID
        )
    }
}
