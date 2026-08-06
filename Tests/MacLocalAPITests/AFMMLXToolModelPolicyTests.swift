import XCTest
import AFMKitMLX

final class AFMMLXToolModelPolicyTests: XCTestCase {
    func testModelListEntriesIncludeDownloadedAndImportedModels() {
        let entries = AFMMLXToolModelPolicy.modelListEntries(
            downloadedModels: [
                AFMMLXToolDownloadedModel(
                    id: "mlx-community/Qwen3-0.6B-4bit",
                    name: "Qwen3-0.6B-4bit",
                    isVision: false
                ),
                AFMMLXToolDownloadedModel(
                    id: "example-org/Vision-Model",
                    name: "Vision-Model",
                    isVision: true
                )
            ],
            importedModels: [
                AFMMLXToolImportedModel(
                    id: "/Volumes/models/imported",
                    name: "Imported",
                    path: "/Volumes/models/imported"
                )
            ]
        )

        XCTAssertEqual(entries.map(\.id), [
            "mlx-community/Qwen3-0.6B-4bit",
            "example-org/Vision-Model",
            "/Volumes/models/imported"
        ])
        XCTAssertEqual(entries[0].source, "curated")
        XCTAssertEqual(entries[1].source, "downloaded")
        XCTAssertEqual(entries[1].isVision, true)
        XCTAssertEqual(entries[2].source, "imported")
        XCTAssertNil(entries[2].url)
    }

    func testResolvePrefersDownloadedModelID() {
        let resolution = AFMMLXToolModelPolicy.resolve(
            modelID: "example-org/Text-Model",
            downloadedModels: Self.downloadedModels,
            importedModels: [],
            isModelRepoOnDisk: { _ in false },
            detectIsVisionFromDisk: { _ in true }
        )

        XCTAssertEqual(
            resolution,
            .downloaded(id: "example-org/Text-Model", displayName: "Text-Model")
        )
    }

    func testResolveRegistersRepoAlreadyOnDiskBeforeNameFallback() {
        let resolution = AFMMLXToolModelPolicy.resolve(
            modelID: "example-org/Local-Vision",
            downloadedModels: Self.downloadedModels,
            importedModels: [],
            isModelRepoOnDisk: { $0 == "example-org/Local-Vision" },
            detectIsVisionFromDisk: { _ in true }
        )

        XCTAssertEqual(
            resolution,
            .repositoryOnDisk(
                id: "example-org/Local-Vision",
                displayName: "Local-Vision",
                isVision: true
            )
        )
    }

    func testResolveFallsBackToDownloadedDisplayName() {
        let resolution = AFMMLXToolModelPolicy.resolve(
            modelID: "Text-Model",
            downloadedModels: Self.downloadedModels,
            importedModels: [],
            isModelRepoOnDisk: { _ in false },
            detectIsVisionFromDisk: { _ in false }
        )

        XCTAssertEqual(
            resolution,
            .downloaded(id: "example-org/Text-Model", displayName: "Text-Model")
        )
    }

    func testResolveImportedByNameLeafOrPath() {
        let imported = [
            AFMMLXToolImportedModel(
                id: "/Volumes/models/imported",
                name: "Imported",
                path: "/Volumes/models/imported"
            )
        ]

        for id in ["Imported", "imported", "/Volumes/models/imported"] {
            XCTAssertEqual(
                AFMMLXToolModelPolicy.resolve(
                    modelID: id,
                    downloadedModels: [],
                    importedModels: imported,
                    isModelRepoOnDisk: { _ in false },
                    detectIsVisionFromDisk: { _ in false }
                ),
                .imported(
                    id: "/Volumes/models/imported",
                    name: "Imported",
                    path: "/Volumes/models/imported"
                )
            )
        }
    }

    func testResolveMissingModel() {
        XCTAssertEqual(
            AFMMLXToolModelPolicy.resolve(
                modelID: "missing/model",
                downloadedModels: [],
                importedModels: [],
                isModelRepoOnDisk: { _ in false },
                detectIsVisionFromDisk: { _ in false }
            ),
            .missing(modelID: "missing/model")
        )
    }

    private static let downloadedModels = [
        AFMMLXToolDownloadedModel(
            id: "example-org/Text-Model",
            name: "Text-Model",
            isVision: false
        )
    ]
}
