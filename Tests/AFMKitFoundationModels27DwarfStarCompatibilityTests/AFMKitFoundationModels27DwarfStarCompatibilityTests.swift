#if compiler(>=6.4) && canImport(FoundationModels)
import XCTest
import AFMKitFoundationModels27DwarfStar

@available(macOS 27.0, *)
final class AFMKitFoundationModels27DwarfStarCompatibilityTests: XCTestCase {
    func testLegacyModuleReexportsDwarfStarModel() {
        let model = DwarfStarLanguageModel(modelPath: "/tmp/model.gguf")

        XCTAssertEqual(model.executorConfiguration.modelPath, "/tmp/model.gguf")
        XCTAssertEqual(model.executorConfiguration.contextWindow, 32_768)
    }
}
#endif
