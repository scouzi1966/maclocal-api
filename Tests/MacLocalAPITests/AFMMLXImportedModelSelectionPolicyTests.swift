import XCTest
import AFMKitMLX

final class AFMMLXImportedModelSelectionPolicyTests: XCTestCase {
    func testUsesMTPCompatibilityForTextOnlyDefault() {
        XCTAssertEqual(
            AFMMLXImportedModelSelectionPolicy.make(
                name: "Local Qwen",
                path: "/Volumes/models/local-qwen",
                isVision: true,
                mtpCompatible: true
            ),
            AFMMLXImportedModelSelectionPlan(
                name: "Local Qwen",
                path: "/Volumes/models/local-qwen",
                isVision: true,
                textOnlyMode: true
            )
        )
    }

    func testDefaultsTextOnlyFromVisionFlag() {
        XCTAssertEqual(
            AFMMLXImportedModelSelectionPolicy.make(
                name: "Local Text",
                path: "/Volumes/models/local-text",
                isVision: false,
                mtpCompatible: false
            ),
            AFMMLXImportedModelSelectionPlan(
                name: "Local Text",
                path: "/Volumes/models/local-text",
                isVision: false,
                textOnlyMode: true
            )
        )

        XCTAssertEqual(
            AFMMLXImportedModelSelectionPolicy.make(
                name: "Local Vision",
                path: "/Volumes/models/local-vision",
                isVision: true,
                mtpCompatible: false
            ),
            AFMMLXImportedModelSelectionPlan(
                name: "Local Vision",
                path: "/Volumes/models/local-vision",
                isVision: true,
                textOnlyMode: false
            )
        )
    }
}
