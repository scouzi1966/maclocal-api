import XCTest
import AFMKitMLX

final class AFMMLXLoadSelectionPolicyTests: XCTestCase {
    func testQuickLoadPlanResolvesImportedSelection() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.quickLoadPlan(
                for: " imported:/Volumes/models/local-model ",
                curatedCandidates: [],
                downloadedIDs: [],
                isDualMode: false,
                loadAsVLM: false
            ),
            .imported(rawPath: "/Volumes/models/local-model")
        )
    }

    func testQuickLoadPlanResolvesCuratedStandardAndDualMode() {
        let curated = [
            AFMMLXQuickCuratedLoadCandidate(
                id: "mlx-community/curated-model",
                name: "curated-model",
                repoID: "mlx-community/curated-model"
            )
        ]

        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.quickLoadPlan(
                for: "mlx-community/curated-model",
                curatedCandidates: curated,
                downloadedIDs: [],
                isDualMode: false,
                loadAsVLM: false
            ),
            .curatedStandard(selectionID: "mlx-community/curated-model")
        )
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.quickLoadPlan(
                for: "mlx-community/curated-model",
                curatedCandidates: curated,
                downloadedIDs: [],
                isDualMode: true,
                loadAsVLM: false
            ),
            .curatedDualMode(
                repoID: "mlx-community/curated-model",
                isVision: false,
                forceLLMOnly: true
            )
        )
    }

    func testQuickLoadPlanResolvesUserDownloadedOverrideOnlyForDualMode() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.quickLoadPlan(
                for: "custom/downloaded-model",
                curatedCandidates: [],
                downloadedIDs: ["custom/downloaded-model"],
                isDualMode: false,
                loadAsVLM: true
            ),
            .userDownloaded(repoID: "custom/downloaded-model", isVisionOverride: nil)
        )
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.quickLoadPlan(
                for: "custom/downloaded-model",
                curatedCandidates: [],
                downloadedIDs: ["custom/downloaded-model"],
                isDualMode: true,
                loadAsVLM: true
            ),
            .userDownloaded(repoID: "custom/downloaded-model", isVisionOverride: true)
        )
    }

    func testQuickLoadPlanFallsBackToSelectionPathAndDisplayName() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.quickLoadPlan(
                for: "unknown-org/unknown-model",
                curatedCandidates: [],
                downloadedIDs: [],
                isDualMode: true,
                loadAsVLM: true
            ),
            .fallback(path: "unknown-org/unknown-model", name: "unknown-model")
        )
    }

    func testSelectedLoadPlanResolvesImportedAbsolutePath() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.selectedLoadPlan(
                modelName: "Local Model",
                customModelPath: " /Volumes/models/local-model ",
                downloadedCandidates: [],
                isDualMode: false,
                textOnlyMode: false
            ),
            .imported(rawPath: "/Volumes/models/local-model")
        )
    }

    func testSelectedLoadPlanResolvesUserDownloadedOverrideOnlyForDualMode() {
        let downloaded = [
            AFMMLXSelectedLoadDownloadedCandidate(
                id: "custom/downloaded-model",
                name: "downloaded-model"
            )
        ]

        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.selectedLoadPlan(
                modelName: "downloaded-model",
                customModelPath: "custom/downloaded-model",
                downloadedCandidates: downloaded,
                isDualMode: false,
                textOnlyMode: false
            ),
            .userDownloaded(repoID: "custom/downloaded-model", isVisionOverride: nil)
        )
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.selectedLoadPlan(
                modelName: "downloaded-model",
                customModelPath: "custom/downloaded-model",
                downloadedCandidates: downloaded,
                isDualMode: true,
                textOnlyMode: true
            ),
            .userDownloaded(repoID: "custom/downloaded-model", isVisionOverride: false)
        )
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.selectedLoadPlan(
                modelName: "downloaded-model",
                customModelPath: "custom/downloaded-model",
                downloadedCandidates: downloaded,
                isDualMode: true,
                textOnlyMode: false
            ),
            .userDownloaded(repoID: "custom/downloaded-model", isVisionOverride: true)
        )
    }

    func testSelectedLoadPlanFallsBackToCuratedModel() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.selectedLoadPlan(
                modelName: "Qwen3-VL-4B-Instruct-5bit",
                customModelPath: "mlx-community/Qwen3-VL-4B-Instruct-5bit",
                downloadedCandidates: [],
                isDualMode: true,
                textOnlyMode: false
            ),
            .curated
        )
    }

    func testNamedLoadPlanPrefersUserDownloadedName() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.namedLoadPlan(
                modelName: "SharedName",
                downloadedCandidates: [
                    AFMMLXSelectedLoadDownloadedCandidate(
                        id: "custom/shared-name",
                        name: "SharedName"
                    )
                ],
                curatedModelNames: ["SharedName"]
            ),
            .userDownloaded(repoID: "custom/shared-name")
        )
    }

    func testNamedLoadPlanFallsBackToCuratedModelName() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.namedLoadPlan(
                modelName: " CuratedModel ",
                downloadedCandidates: [
                    AFMMLXSelectedLoadDownloadedCandidate(
                        id: "custom/other",
                        name: "Other"
                    )
                ],
                curatedModelNames: ["CuratedModel"]
            ),
            .curated(modelName: "CuratedModel")
        )
    }

    func testNamedLoadPlanReturnsUnavailableForBlankOrUnknownSelection() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.namedLoadPlan(
                modelName: "   ",
                downloadedCandidates: [
                    AFMMLXSelectedLoadDownloadedCandidate(
                        id: "custom/downloaded",
                        name: "Downloaded"
                    )
                ],
                curatedModelNames: ["Curated"]
            ),
            .unavailable
        )

        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.namedLoadPlan(
                modelName: "Missing",
                downloadedCandidates: [],
                curatedModelNames: ["Curated"]
            ),
            .unavailable
        )
    }

    func testCuratedSelectionPrefersCustomModelPathAsAFM27ID() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.curatedSelection(
                modelName: " CuratedModel ",
                customModelPath: " mlx-community/CuratedModel "
            ),
            AFMMLXCuratedSelection(
                modelName: "CuratedModel",
                afm27ModelID: "mlx-community/CuratedModel"
            )
        )
    }

    func testCuratedSelectionFallsBackToCommunityRepoID() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.curatedSelection(
                modelName: " CuratedModel ",
                customModelPath: nil
            ),
            AFMMLXCuratedSelection(
                modelName: "CuratedModel",
                afm27ModelID: "mlx-community/CuratedModel"
            )
        )
    }

    func testSelectedNameChangePlanUnloadsLoadedModelOnFamilyChangeAfterAppear() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.selectedNameChangePlan(
                oldModelName: "Qwen3-VL-4B-Instruct-5bit",
                newModelName: "Qwen3-VL-8B-Instruct-5bit",
                loadedModelName: "Qwen3-VL-4B-Instruct-5bit",
                hasAppearedOnce: true,
                isModelLoaded: true,
                selectedModelCustomPath: nil,
                importedModelNames: [],
                curatedModelNames: ["Qwen3-VL-8B-Instruct-5bit"]
            ),
            AFMMLXSelectedNameChangePlan(
                shouldUnloadLoadedModel: true,
                curatedModelName: "Qwen3-VL-8B-Instruct-5bit"
            )
        )
    }

    func testSelectedNameChangePlanDoesNotUnloadWhenSyncingToLoadedModel() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.selectedNameChangePlan(
                oldModelName: "Qwen3-VL-4B-Instruct-5bit",
                newModelName: "Qwen3-VL-8B-Instruct-5bit",
                loadedModelName: " Qwen3-VL-8B-Instruct-5bit ",
                hasAppearedOnce: true,
                isModelLoaded: true,
                selectedModelCustomPath: nil,
                importedModelNames: [],
                curatedModelNames: ["Qwen3-VL-8B-Instruct-5bit"]
            ),
            AFMMLXSelectedNameChangePlan(
                shouldUnloadLoadedModel: false,
                curatedModelName: "Qwen3-VL-8B-Instruct-5bit"
            )
        )
    }

    func testSelectedNameChangePlanSkipsCuratedUpdateForImportedSelection() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.selectedNameChangePlan(
                oldModelName: "Qwen3-VL-4B-Instruct-5bit",
                newModelName: "ImportedModel",
                loadedModelName: nil,
                hasAppearedOnce: true,
                isModelLoaded: false,
                selectedModelCustomPath: nil,
                importedModelNames: ["ImportedModel"],
                curatedModelNames: ["ImportedModel"]
            ),
            AFMMLXSelectedNameChangePlan(
                shouldUnloadLoadedModel: false,
                curatedModelName: nil
            )
        )
    }

    func testSelectedNameChangePlanSkipsCuratedUpdateForAbsoluteCustomPath() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.selectedNameChangePlan(
                oldModelName: "Qwen3-VL-4B-Instruct-5bit",
                newModelName: "Qwen3-VL-8B-Instruct-5bit",
                loadedModelName: nil,
                hasAppearedOnce: true,
                isModelLoaded: false,
                selectedModelCustomPath: " /Volumes/models/local-model ",
                importedModelNames: [],
                curatedModelNames: ["Qwen3-VL-8B-Instruct-5bit"]
            ),
            AFMMLXSelectedNameChangePlan(
                shouldUnloadLoadedModel: false,
                curatedModelName: nil
            )
        )
    }

    func testFallbackDisplayNameUsesLastPathComponent() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.fallbackDisplayName(for: " org/model "),
            "model"
        )
    }

    func testQuickSelectionIsLoadedMatchesImportedPath() {
        XCTAssertTrue(
            AFMMLXLoadSelectionPolicy.quickSelectionIsLoaded(
                selectionID: "imported:/Volumes/models/local-model",
                loadedModelID: "/Volumes/models/local-model",
                loadedModelName: "local-model"
            )
        )
        XCTAssertFalse(
            AFMMLXLoadSelectionPolicy.quickSelectionIsLoaded(
                selectionID: "imported:/Volumes/models/other-model",
                loadedModelID: "/Volumes/models/local-model",
                loadedModelName: "local-model"
            )
        )
    }

    func testQuickSelectionIsLoadedMatchesRepositoryIDOrDisplayName() {
        XCTAssertTrue(
            AFMMLXLoadSelectionPolicy.quickSelectionIsLoaded(
                selectionID: "mlx-community/Qwen3-4B",
                loadedModelID: "mlx-community/Qwen3-4B",
                loadedModelName: nil
            )
        )
        XCTAssertTrue(
            AFMMLXLoadSelectionPolicy.quickSelectionIsLoaded(
                selectionID: "mlx-community/Qwen3-4B",
                loadedModelID: nil,
                loadedModelName: "Qwen3-4B"
            )
        )
        XCTAssertFalse(
            AFMMLXLoadSelectionPolicy.quickSelectionIsLoaded(
                selectionID: "mlx-community/Qwen3-4B",
                loadedModelID: "mlx-community/Other",
                loadedModelName: "Other"
            )
        )
    }

    func testQuickSelectionIsLoadedReturnsFalseForEmptySelection() {
        XCTAssertFalse(
            AFMMLXLoadSelectionPolicy.quickSelectionIsLoaded(
                selectionID: " ",
                loadedModelID: "mlx-community/Qwen3-4B",
                loadedModelName: "Qwen3-4B"
            )
        )
    }

    func testInitialQuickSelectionPrefersLoadedModel() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.initialQuickSelection(
                loadedModelID: " mlx-community/loaded-model ",
                loadedModelIsVLM: true,
                curatedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "mlx-community/curated-model", isVision: false)
                ],
                downloadedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "custom/downloaded-model", isVision: false)
                ],
                importedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "imported:/Volumes/models/imported", isVision: true)
                ]
            ),
            AFMMLXQuickSelection(id: "mlx-community/loaded-model", loadAsVLM: true)
        )
    }

    func testInitialQuickSelectionFallsThroughCategoryOrder() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.initialQuickSelection(
                loadedModelID: nil,
                loadedModelIsVLM: false,
                curatedCandidates: [],
                downloadedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "custom/downloaded-model", isVision: true)
                ],
                importedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "imported:/Volumes/models/imported", isVision: false)
                ]
            ),
            AFMMLXQuickSelection(id: "custom/downloaded-model", loadAsVLM: true)
        )
    }

    func testInitialQuickSelectionReturnsNilWhenNoCandidates() {
        XCTAssertNil(
            AFMMLXLoadSelectionPolicy.initialQuickSelection(
                loadedModelID: " ",
                loadedModelIsVLM: false,
                curatedCandidates: [],
                downloadedCandidates: [],
                importedCandidates: []
            )
        )
    }

    func testQuickSelectionLoadAsVLMPrefersImportedSelection() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.quickSelectionLoadAsVLM(
                for: " imported:/Volumes/models/local-vlm ",
                curatedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "imported:/Volumes/models/local-vlm", isVision: false)
                ],
                downloadedCandidates: [],
                importedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "imported:/Volumes/models/local-vlm", isVision: true)
                ]
            ),
            true
        )
    }

    func testQuickSelectionLoadAsVLMFallsBackToCuratedThenDownloaded() {
        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.quickSelectionLoadAsVLM(
                for: "mlx-community/curated-model",
                curatedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "mlx-community/curated-model", isVision: true)
                ],
                downloadedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "mlx-community/curated-model", isVision: false),
                    AFMMLXQuickSelectionCandidate(id: "custom/downloaded-model", isVision: false)
                ],
                importedCandidates: []
            ),
            true
        )

        XCTAssertEqual(
            AFMMLXLoadSelectionPolicy.quickSelectionLoadAsVLM(
                for: "custom/downloaded-model",
                curatedCandidates: [],
                downloadedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "custom/downloaded-model", isVision: false)
                ],
                importedCandidates: []
            ),
            false
        )
    }

    func testQuickSelectionLoadAsVLMReturnsNilForUnknownSelection() {
        XCTAssertNil(
            AFMMLXLoadSelectionPolicy.quickSelectionLoadAsVLM(
                for: "unknown/model",
                curatedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "mlx-community/curated-model", isVision: true)
                ],
                downloadedCandidates: [
                    AFMMLXQuickSelectionCandidate(id: "custom/downloaded-model", isVision: false)
                ],
                importedCandidates: []
            )
        )
    }
}
