import XCTest
@testable import AFMKitMLX

final class AFMMLXModelPresentationPolicyTests: XCTestCase {
    struct TestParameters: Equatable {
        let temperature: Double
        let maxTokens: Int
    }

    func testDisplayNamePrefersLoadedModelThenCuratedDisplayName() {
        let candidates = [
            AFMMLXDisplayModelCandidate(name: "Qwen3-4B-4bit", displayName: "Qwen3 4B"),
            AFMMLXDisplayModelCandidate(name: "Gemma-4B-4bit", displayName: "Gemma 4B")
        ]

        XCTAssertEqual(
            AFMMLXModelPresentationPolicy.displayName(
                forSelection: "Qwen3-4B-4bit",
                loadedModelName: " Loaded Model ",
                curatedCandidates: candidates
            ),
            "Loaded Model"
        )

        XCTAssertEqual(
            AFMMLXModelPresentationPolicy.displayName(
                forSelection: " Qwen3-4B-4bit ",
                loadedModelName: nil,
                curatedCandidates: candidates
            ),
            "Qwen3 4B"
        )
    }

    func testDisplayNameFallsBackToSelectionForUserDownloadedModel() {
        XCTAssertEqual(
            AFMMLXModelPresentationPolicy.displayName(
                forSelection: "example-org/Downloaded-Model-4bit",
                loadedModelName: " ",
                curatedCandidates: [
                    AFMMLXDisplayModelCandidate(name: "Curated-Model", displayName: "Curated Model")
                ]
            ),
            "example-org/Downloaded-Model-4bit"
        )
    }

    func testParameterPresetResolvesOnlyCuratedSelection() {
        let preset = TestParameters(temperature: 0.2, maxTokens: 2048)

        XCTAssertEqual(
            AFMMLXModelPresentationPolicy.parameterPreset(
                forSelection: " Curated-Model ",
                curatedCandidates: [
                    AFMMLXParameterPresetCandidate(name: "Curated-Model", parameters: preset),
                    AFMMLXParameterPresetCandidate<TestParameters>(name: "Other-Model", parameters: nil)
                ]
            ),
            preset
        )

        XCTAssertNil(
            AFMMLXModelPresentationPolicy.parameterPreset(
                forSelection: "example-org/Downloaded-Model-4bit",
                curatedCandidates: [
                    AFMMLXParameterPresetCandidate(name: "Curated-Model", parameters: preset)
                ]
            )
        )
    }

    func testModelBitDepthParsesTrailingBitSuffix() {
        XCTAssertEqual(
            AFMMLXModelPresentationPolicy.modelBitDepth(for: "Qwen3-VL-4B-Instruct-5bit"),
            5
        )
        XCTAssertNil(
            AFMMLXModelPresentationPolicy.modelBitDepth(for: "Qwen3-VL-4B-Instruct-bf16")
        )
    }

    func testCuratedModelNamesFiltersFamilyAndSortsQuantizationBeforeBF16() {
        XCTAssertEqual(
            AFMMLXModelPresentationPolicy.curatedModelNames(
                in: "Qwen3-VL-4B",
                candidates: [
                    AFMMLXCuratedModelCandidate(name: "Qwen3-VL-8B-Instruct-4bit"),
                    AFMMLXCuratedModelCandidate(name: "Qwen3-VL-4B-Instruct-bf16"),
                    AFMMLXCuratedModelCandidate(name: "Qwen3-VL-4B-Instruct-8bit"),
                    AFMMLXCuratedModelCandidate(name: "Qwen3-VL-4B-Instruct-4bit"),
                    AFMMLXCuratedModelCandidate(name: "Qwen3-VL-4B-Instruct-5bit")
                ]
            ),
            [
                "Qwen3-VL-4B-Instruct-4bit",
                "Qwen3-VL-4B-Instruct-5bit",
                "Qwen3-VL-4B-Instruct-8bit",
                "Qwen3-VL-4B-Instruct-bf16"
            ]
        )
    }

    func testAutoLoadModelNamePrefersSelectedThenDefaultAvailableModel() {
        XCTAssertEqual(
            AFMMLXModelPresentationPolicy.autoLoadModelName(
                selectedModelName: "Selected",
                defaultModelName: "Default",
                candidates: [
                    AFMMLXAutoLoadCandidate(name: "Default", isAvailable: true),
                    AFMMLXAutoLoadCandidate(name: "Selected", isAvailable: true)
                ]
            ),
            "Selected"
        )

        XCTAssertEqual(
            AFMMLXModelPresentationPolicy.autoLoadModelName(
                selectedModelName: "Selected",
                defaultModelName: "Default",
                candidates: [
                    AFMMLXAutoLoadCandidate(name: "Default", isAvailable: true),
                    AFMMLXAutoLoadCandidate(name: "Selected", isAvailable: false)
                ]
            ),
            "Default"
        )
    }

    func testAutoLoadModelNameReturnsNilWithoutAvailableCandidate() {
        XCTAssertNil(
            AFMMLXModelPresentationPolicy.autoLoadModelName(
                selectedModelName: "Selected",
                defaultModelName: "Default",
                candidates: [
                    AFMMLXAutoLoadCandidate(name: "Default", isAvailable: false),
                    AFMMLXAutoLoadCandidate(name: "Selected", isAvailable: false)
                ]
            )
        )
    }

    func testBenchmarkModelSelectionIDsKeepsAvailableSortedUniqueIDs() {
        XCTAssertEqual(
            AFMMLXModelPresentationPolicy.benchmarkModelSelectionIDs(
                curatedCandidates: [
                    AFMMLXBenchmarkModelCandidate(id: "Zeta", isAvailable: true),
                    AFMMLXBenchmarkModelCandidate(id: "Alpha", isAvailable: true),
                    AFMMLXBenchmarkModelCandidate(id: "Missing", isAvailable: false)
                ],
                downloadedCandidates: [
                    AFMMLXBenchmarkModelCandidate(id: "custom/Beta", isAvailable: true),
                    AFMMLXBenchmarkModelCandidate(id: "Alpha", isAvailable: true),
                    AFMMLXBenchmarkModelCandidate(id: "custom/Gamma", isAvailable: false)
                ]
            ),
            [
                "Alpha",
                "Zeta",
                "custom/Beta"
            ]
        )
    }
}
