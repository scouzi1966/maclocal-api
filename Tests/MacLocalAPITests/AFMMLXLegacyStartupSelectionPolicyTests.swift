import XCTest
import AFMKitMLX

final class AFMMLXLegacyStartupSelectionPolicyTests: XCTestCase {
    func testPrefersLoadedAvailableModel() {
        XCTAssertEqual(
            AFMMLXLegacyStartupSelectionPolicy.select(
                loadedModelName: "Loaded",
                loadedModelRepoID: "custom/loaded-repo",
                selectedModelName: "Selected",
                candidates: [
                    AFMMLXLegacyStartupModelCandidate(name: "Selected", id: "custom/selected", isAvailable: true),
                    AFMMLXLegacyStartupModelCandidate(name: "Loaded", id: "custom/loaded", isAvailable: true)
                ]
            ),
            AFMMLXLegacyStartupSelection(
                modelName: "Loaded",
                afm27ModelID: "custom/loaded-repo"
            )
        )
    }

    func testFallsBackToLoadedCandidateIDWhenLoadedRepoIDIsBlank() {
        XCTAssertEqual(
            AFMMLXLegacyStartupSelectionPolicy.select(
                loadedModelName: "Loaded",
                loadedModelRepoID: " ",
                selectedModelName: "Selected",
                candidates: [
                    AFMMLXLegacyStartupModelCandidate(name: "Selected", id: "custom/selected", isAvailable: true),
                    AFMMLXLegacyStartupModelCandidate(name: "Loaded", id: "custom/loaded", isAvailable: true)
                ]
            ),
            AFMMLXLegacyStartupSelection(
                modelName: "Loaded",
                afm27ModelID: "custom/loaded"
            )
        )
    }

    func testPrefersPersistedSelectionAfterLoadedMiss() {
        XCTAssertEqual(
            AFMMLXLegacyStartupSelectionPolicy.select(
                loadedModelName: "MissingLoaded",
                loadedModelRepoID: "custom/missing",
                selectedModelName: "Selected",
                candidates: [
                    AFMMLXLegacyStartupModelCandidate(name: "Selected", id: "custom/selected", isAvailable: true),
                    AFMMLXLegacyStartupModelCandidate(name: "Model-5bit", id: "custom/default", isAvailable: true)
                ]
            ),
            AFMMLXLegacyStartupSelection(
                modelName: "Selected",
                afm27ModelID: "custom/selected"
            )
        )
    }

    func testFallsBackToFiveBitThenFirstAvailable() {
        XCTAssertEqual(
            AFMMLXLegacyStartupSelectionPolicy.select(
                loadedModelName: nil,
                loadedModelRepoID: nil,
                selectedModelName: "Missing",
                candidates: [
                    AFMMLXLegacyStartupModelCandidate(name: "Alpha", id: "custom/alpha", isAvailable: true),
                    AFMMLXLegacyStartupModelCandidate(name: "Model-5bit", id: "custom/default", isAvailable: true)
                ]
            ),
            AFMMLXLegacyStartupSelection(
                modelName: "Model-5bit",
                afm27ModelID: "custom/default"
            )
        )

        XCTAssertEqual(
            AFMMLXLegacyStartupSelectionPolicy.select(
                loadedModelName: nil,
                loadedModelRepoID: nil,
                selectedModelName: "Missing",
                candidates: [
                    AFMMLXLegacyStartupModelCandidate(name: "Alpha", id: "custom/alpha", isAvailable: true),
                    AFMMLXLegacyStartupModelCandidate(name: "Beta", id: "custom/beta", isAvailable: true)
                ]
            ),
            AFMMLXLegacyStartupSelection(
                modelName: "Alpha",
                afm27ModelID: "custom/alpha"
            )
        )
    }

    func testReturnsNilWithoutAvailableModels() {
        XCTAssertNil(
            AFMMLXLegacyStartupSelectionPolicy.select(
                loadedModelName: "Loaded",
                loadedModelRepoID: "custom/loaded",
                selectedModelName: "Selected",
                candidates: [
                    AFMMLXLegacyStartupModelCandidate(name: "Selected", id: "custom/selected", isAvailable: false),
                    AFMMLXLegacyStartupModelCandidate(name: "Loaded", id: "custom/loaded", isAvailable: false)
                ]
            )
        )
    }
}
