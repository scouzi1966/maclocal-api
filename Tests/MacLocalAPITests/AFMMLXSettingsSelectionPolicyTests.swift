import XCTest
import AFMKitMLX

final class AFMMLXSettingsSelectionPolicyTests: XCTestCase {
    func testSyncUsesLegacySelectedDisplayNameWhenCurrentSelectionIsEmpty() {
        let synced = AFMMLXSettingsSelectionPolicy.syncedModelID(
            currentModelID: " ",
            selectedLegacyModelID: "Downloaded model",
            options: Self.options
        )

        XCTAssertEqual(synced, "mlx-community/Downloaded-model")
    }

    func testSyncMatchesLegacySelectedRepoLeaf() {
        let synced = AFMMLXSettingsSelectionPolicy.syncedModelID(
            currentModelID: "",
            selectedLegacyModelID: "Downloaded-model",
            options: Self.options
        )

        XCTAssertEqual(synced, "mlx-community/Downloaded-model")
    }

    func testSyncDoesNotOverrideExistingSelectionUnlessForced() {
        let synced = AFMMLXSettingsSelectionPolicy.syncedModelID(
            currentModelID: "mlx-community/Current",
            selectedLegacyModelID: "Downloaded model",
            options: Self.options
        )

        XCTAssertNil(synced)

        let forced = AFMMLXSettingsSelectionPolicy.syncedModelID(
            currentModelID: "mlx-community/Current",
            selectedLegacyModelID: "Downloaded model",
            options: Self.options,
            force: true
        )

        XCTAssertEqual(forced, "mlx-community/Downloaded-model")
    }

    func testLegacySelectionDisplayNameIgnoresCustomOptionsAndUnchangedNames() {
        XCTAssertNil(
            AFMMLXSettingsSelectionPolicy.legacySelectionDisplayName(
                for: "/custom/path",
                options: Self.options,
                currentLegacyDisplayName: "Other"
            )
        )
        XCTAssertNil(
            AFMMLXSettingsSelectionPolicy.legacySelectionDisplayName(
                for: "mlx-community/Downloaded-model",
                options: Self.options,
                currentLegacyDisplayName: "Downloaded model"
            )
        )
    }

    func testLegacySelectionDisplayNameReturnsManagedDisplayName() {
        let legacyName = AFMMLXSettingsSelectionPolicy.legacySelectionDisplayName(
            for: "mlx-community/Downloaded-model",
            options: Self.options,
            currentLegacyDisplayName: "Other"
        )

        XCTAssertEqual(legacyName, "Downloaded model")
    }

    private static let options: [AFMMLXModelSelectionOption] = [
        AFMMLXModelSelectionOption(
            id: "mlx-community/Downloaded-model",
            displayName: "Downloaded model",
            sourceTag: "downloaded",
            loadReference: nil
        ),
        AFMMLXModelSelectionOption(
            id: "/custom/path",
            displayName: "path",
            sourceTag: "custom",
            loadReference: nil
        )
    ]
}
