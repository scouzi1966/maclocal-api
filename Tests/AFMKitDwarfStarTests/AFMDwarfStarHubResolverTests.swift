import XCTest
@testable import AFMKitDwarfStar

final class AFMDwarfStarHubResolverTests: XCTestCase {
    func testSelectorExcludesDSparkSupportArtifact() throws {
        let selected = try AFMDwarfStarHubSelector.selectModel(
            from: [
                .init(path: "DeepSeek-DSpark.gguf", size: 6_000),
                .init(path: "DeepSeek-main.gguf", size: 150_000)
            ],
            repositoryID: "owner/model",
            physicalMemory: 1_000_000)

        XCTAssertEqual(selected.path, "DeepSeek-main.gguf")
    }

    func testSelectorChoosesLargestModelThatFitsMemoryBudget() throws {
        let selected = try AFMDwarfStarHubSelector.selectModel(
            from: [
                .init(path: "q2.gguf", size: 20),
                .init(path: "q4.gguf", size: 70),
                .init(path: "q8.gguf", size: 95)
            ],
            repositoryID: "owner/model",
            physicalMemory: 100,
            memoryFraction: 0.8)

        XCTAssertEqual(selected.path, "q4.gguf")
    }

    func testExplicitFileOverridesMemorySelection() throws {
        let selected = try AFMDwarfStarHubSelector.selectModel(
            from: [
                .init(path: "small.gguf", size: 20),
                .init(path: "large.gguf", size: 95)
            ],
            repositoryID: "owner/model",
            requestedPath: "large.gguf",
            physicalMemory: 100)

        XCTAssertEqual(selected.path, "large.gguf")
    }

    func testSelectorRejectsRepositoryWithOnlySupportGGUF() {
        XCTAssertThrowsError(try AFMDwarfStarHubSelector.selectModel(
            from: [.init(path: "dspark-support.gguf", size: 10)],
            repositoryID: "owner/model")) { error in
                XCTAssertEqual(
                    error as? AFMDwarfStarHubSelectionError,
                    .noModelGGUF("owner/model"))
            }
    }
}
