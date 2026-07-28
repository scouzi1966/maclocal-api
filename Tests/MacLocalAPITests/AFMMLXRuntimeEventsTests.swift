@testable import AFMKitMLX
import XCTest

final class AFMMLXRuntimeEventsTests: XCTestCase {
    func testRuntimeInfoCarriesGenerationMetrics() {
        let info = AFMMLXRuntimeInfo(
            promptTime: 1.25,
            tokensPerSecond: 42.5
        )

        XCTAssertEqual(info.promptTime, 1.25)
        XCTAssertEqual(info.tokensPerSecond, 42.5)
    }

    func testRuntimeEventsAreEquatableForContractTests() {
        XCTAssertEqual(
            AFMMLXRuntimeEvent.chunk("hello"),
            AFMMLXRuntimeEvent.chunk("hello")
        )
        XCTAssertNotEqual(
            AFMMLXRuntimeEvent.chunk("hello"),
            AFMMLXRuntimeEvent.chunk("bye")
        )
        XCTAssertEqual(
            AFMMLXRuntimeEvent.info(
                AFMMLXRuntimeInfo(promptTime: 0.4, tokensPerSecond: 12.0)
            ),
            AFMMLXRuntimeEvent.info(
                AFMMLXRuntimeInfo(promptTime: 0.4, tokensPerSecond: 12.0)
            )
        )
    }

    func testDefaultImageProcessingSizeMatchesLegacyVLMInputSize() {
        XCTAssertEqual(AFMMLXRuntimePolicy.defaultImageProcessingSize, 1024)
    }
}
