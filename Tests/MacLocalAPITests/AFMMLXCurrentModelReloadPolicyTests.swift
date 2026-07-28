import XCTest
import AFMKitMLX

final class AFMMLXCurrentModelReloadPolicyTests: XCTestCase {
    func testUnavailableWithoutLoadedModel() {
        XCTAssertEqual(
            AFMMLXCurrentModelReloadPolicy.make(
                loadedModelRepoID: nil,
                targetVLM: true
            ),
            .unavailable
        )

        XCTAssertEqual(
            AFMMLXCurrentModelReloadPolicy.make(
                loadedModelRepoID: "   ",
                targetVLM: false
            ),
            .unavailable
        )
    }

    func testImportedPathReloadUsesLastPathComponentAndTargetMode() {
        let plan = AFMMLXCurrentModelReloadPolicy.make(
            loadedModelRepoID: "/Volumes/edata/models/Qwen3.5",
            targetVLM: true
        )

        XCTAssertEqual(
            plan,
            .imported(
                name: "Qwen3.5",
                path: "/Volumes/edata/models/Qwen3.5",
                isVision: true
            )
        )
    }

    func testRepositoryReloadForVisionDoesNotForceLLMOnly() {
        let plan = AFMMLXCurrentModelReloadPolicy.make(
            loadedModelRepoID: "mlx-community/Qwen3.5-VL",
            targetVLM: true
        )

        XCTAssertEqual(
            plan,
            .repository(
                repoID: "mlx-community/Qwen3.5-VL",
                isVision: true,
                forceLLMOnly: false
            )
        )
    }

    func testRepositoryReloadForTextOnlyForcesLLMOnly() {
        let plan = AFMMLXCurrentModelReloadPolicy.make(
            loadedModelRepoID: "mlx-community/Qwen3.5-VL",
            targetVLM: false
        )

        XCTAssertEqual(
            plan,
            .repository(
                repoID: "mlx-community/Qwen3.5-VL",
                isVision: false,
                forceLLMOnly: true
            )
        )
    }
}
