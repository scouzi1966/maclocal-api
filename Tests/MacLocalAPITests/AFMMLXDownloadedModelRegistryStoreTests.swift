import XCTest
@testable import AFMKitMLX

final class AFMMLXDownloadedModelRegistryStoreTests: XCTestCase {
    private let suiteName = "AFMMLXDownloadedModelRegistryStoreTests"
    private let modelsKey = "models"

    override func tearDownWithError() throws {
        UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName)
        try super.tearDownWithError()
    }

    func testModelsRoundTripThroughUserDefaults() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        let store = AFMMLXDownloadedModelRegistryStore(defaults: defaults)
        let models = [
            DownloadRecord(repoID: "example-org/Text-Model-4bit", displayName: "Text-Model-4bit", isVision: false),
            DownloadRecord(repoID: "example-org/Vision-Model-4bit", displayName: "Vision-Model-4bit", isVision: true)
        ]

        try store.save(models, forKey: modelsKey)
        let loaded = try store.load(DownloadRecord.self, forKey: modelsKey)

        XCTAssertEqual(loaded, models)
    }

    func testMissingKeyLoadsEmptyArray() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        let store = AFMMLXDownloadedModelRegistryStore(defaults: defaults)

        let loaded = try store.load(DownloadRecord.self, forKey: modelsKey)

        XCTAssertEqual(loaded, [])
    }

    func testLoadCleanedPreservesOriginalAndRemovedCounts() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        let store = AFMMLXDownloadedModelRegistryStore(defaults: defaults)
        let models = [
            DownloadRecord(repoID: "custom/Alpha-4bit", displayName: "Alpha-4bit", isVision: false),
            DownloadRecord(repoID: "other/Alpha-4bit", displayName: "Alpha-4bit", isVision: true),
            DownloadRecord(repoID: "curated/Gemma-4bit", displayName: "Gemma-4bit", isVision: false)
        ]

        try store.save(models, forKey: modelsKey)
        let loaded = try store.loadCleaned(
            DownloadRecord.self,
            forKey: modelsKey,
            id: \.repoID,
            displayName: \.displayName,
            isCurated: { $0 == "curated/Gemma-4bit" }
        )

        XCTAssertEqual(loaded.originalCount, 3)
        XCTAssertEqual(loaded.removedCount, 2)
        XCTAssertEqual(loaded.models.map(\.repoID), ["custom/Alpha-4bit"])
    }
}

private struct DownloadRecord: Codable, Equatable {
    let repoID: String
    let displayName: String
    let isVision: Bool
}
