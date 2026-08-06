import XCTest
import AFMKitCore
import AFMKitMLX

final class AFMMLXStartupModelSelectionPolicyTests: XCTestCase {
    func testLoadedOptionWinsOverPersistedSelection() {
        let selection = AFMMLXStartupModelSelectionPolicy.select(
            options: [
                option(id: "org/persisted", displayName: "Persisted", sourceTag: "curated"),
                option(id: "org/loaded", displayName: "Loaded", sourceTag: "loaded")
            ],
            selectedDisplayName: "Persisted"
        )

        XCTAssertEqual(selection?.id, "org/loaded")
    }

    func testPersistedSelectionWinsWhenNoLoadedOptionExists() {
        let selection = AFMMLXStartupModelSelectionPolicy.select(
            options: [
                option(id: "org/alpha", displayName: "Alpha", sourceTag: "curated"),
                option(id: "org/beta", displayName: "Beta", sourceTag: "curated")
            ],
            selectedDisplayName: "Beta"
        )

        XCTAssertEqual(selection?.id, "org/beta")
    }

    func testPersistedSelectionCanMatchRepositoryDisplayName() {
        let selection = AFMMLXStartupModelSelectionPolicy.select(
            options: [
                option(id: "mlx-community/Qwen3.5-VL", displayName: "Different Label", sourceTag: "curated")
            ],
            selectedDisplayName: "Qwen3.5-VL"
        )

        XCTAssertEqual(selection?.id, "mlx-community/Qwen3.5-VL")
    }

    func testDefaultOptionWinsBeforeFirstAvailableFallback() {
        let selection = AFMMLXStartupModelSelectionPolicy.select(
            options: [
                option(id: "org/alpha", displayName: "Alpha", sourceTag: "curated"),
                option(id: "org/default", displayName: "Default", sourceTag: "curated")
            ],
            selectedDisplayName: "",
            defaultOptionIDs: ["org/default"]
        )

        XCTAssertEqual(selection?.id, "org/default")
    }

    func testUnavailableOptionsDoNotAutoSelect() {
        let selection = AFMMLXStartupModelSelectionPolicy.select(
            options: [
                option(id: "org/unavailable", displayName: "Unavailable", sourceTag: "curated", available: false)
            ],
            selectedDisplayName: "Unavailable",
            defaultOptionIDs: ["org/unavailable"]
        )

        XCTAssertNil(selection)
    }

    func testFirstAvailableOptionIsFallback() {
        let selection = AFMMLXStartupModelSelectionPolicy.select(
            options: [
                option(id: "org/alpha", displayName: "Alpha", sourceTag: "curated"),
                option(id: "org/beta", displayName: "Beta", sourceTag: "curated")
            ],
            selectedDisplayName: ""
        )

        XCTAssertEqual(selection?.id, "org/alpha")
    }

    private func option(
        id: String,
        displayName: String,
        sourceTag: String,
        available: Bool = true
    ) -> AFMMLXModelSelectionOption {
        AFMMLXModelSelectionOption(
            id: id,
            displayName: displayName,
            sourceTag: sourceTag,
            loadReference: available ? loadReference(id: id, displayName: displayName) : nil
        )
    }

    private func loadReference(id: String, displayName: String) -> AFMMLXModelLoadReference {
        AFMMLXModelLoadReference(
            requestedID: id,
            loadIdentifier: id,
            localDirectory: URL(fileURLWithPath: "/models/\(displayName)"),
            descriptor: AFMModelDescriptor(
                providerID: "mlx",
                modelID: AFMModelID(rawValue: id),
                displayName: displayName,
                capabilities: [.text],
                contextWindow: 4096,
                privacyBoundary: .device,
                requiresNetwork: false
            )
        )
    }
}
