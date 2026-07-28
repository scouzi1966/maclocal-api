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

    func testRuntimeAdapterUsesSharedImageProcessingSize() {
        XCTAssertEqual(
            AFMMLXRuntimeAdapter.imageProcessingSize,
            1024
        )
    }

    func testRuntimeAdapterDetectsLikelyVisionModelPaths() {
        XCTAssertTrue(AFMMLXRuntimeAdapter.pathSuggestsVisionModel("/models/Qwen3-VL-4bit"))
        XCTAssertTrue(AFMMLXRuntimeAdapter.pathSuggestsVisionModel("/models/qwen3-vl_4bit"))
        XCTAssertTrue(AFMMLXRuntimeAdapter.pathSuggestsVisionModel("/models/local-vision-model"))
        XCTAssertFalse(AFMMLXRuntimeAdapter.pathSuggestsVisionModel("/models/Qwen3-4bit"))
    }
}
