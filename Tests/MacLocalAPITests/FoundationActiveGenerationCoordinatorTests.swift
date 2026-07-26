#if canImport(FoundationModels)
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
@MainActor
final class FoundationActiveGenerationCoordinatorTests: XCTestCase {
    func testNewGenerationReportsReplacedProvider() {
        let coordinator = AFMFoundationActiveGenerationCoordinator<String>()
        let first = coordinator.begin(provider: "apple")
        let second = coordinator.begin(provider: "pcc")

        XCTAssertNil(first.replacedProvider)
        XCTAssertEqual(second.replacedProvider, "apple")
        XCTAssertEqual(coordinator.activeProvider, "pcc")
        XCTAssertFalse(coordinator.isActive(first.generation))
        XCTAssertTrue(coordinator.isActive(second.generation))
    }

    func testStaleGenerationCannotFinishCurrentGeneration() {
        let coordinator = AFMFoundationActiveGenerationCoordinator<String>()
        let first = coordinator.begin(provider: "apple").generation
        let second = coordinator.begin(provider: "claude").generation

        XCTAssertFalse(coordinator.finish(first))
        XCTAssertEqual(coordinator.activeProvider, "claude")
        XCTAssertTrue(coordinator.finish(second))
        XCTAssertNil(coordinator.activeProvider)
    }

    func testCancelReturnsProviderAndClearsGeneration() {
        let coordinator = AFMFoundationActiveGenerationCoordinator<String>()
        _ = coordinator.begin(provider: "gemini")

        XCTAssertEqual(coordinator.cancelActiveGeneration(), "gemini")
        XCTAssertNil(coordinator.activeProvider)
        XCTAssertNil(coordinator.cancelActiveGeneration())
    }

    func testResetClearsActiveGeneration() {
        let coordinator = AFMFoundationActiveGenerationCoordinator<String>()
        _ = coordinator.begin(provider: "mlx")

        coordinator.reset()

        XCTAssertNil(coordinator.activeGeneration)
        XCTAssertNil(coordinator.activeProvider)
    }
}
#endif
