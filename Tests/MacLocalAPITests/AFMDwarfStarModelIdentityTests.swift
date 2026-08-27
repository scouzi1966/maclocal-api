@testable import AFMServer
import XCTest

final class AFMDwarfStarModelIdentityTests: XCTestCase {
    func testRepositoryIDRemainsStableAtAPIBoundary() {
        XCTAssertEqual(
            AFMDwarfStarModelIdentity.advertisedModelID(
                requestedModel: "owner/repository",
                checkpointPath: "/cache/model-00001.gguf",
                requestedPathExists: { _ in false }
            ),
            "owner/repository"
        )
    }

    func testExistingTwoComponentRelativePathUsesCheckpointBasename() {
        XCTAssertEqual(
            AFMDwarfStarModelIdentity.advertisedModelID(
                requestedModel: "models/checkpoint.gguf",
                checkpointPath: "/workspace/models/checkpoint.gguf",
                requestedPathExists: { $0 == "models/checkpoint.gguf" }
            ),
            "checkpoint.gguf"
        )
    }

    func testExplicitLocalPathUsesCheckpointBasenameEvenBeforeExistenceCheck() {
        XCTAssertEqual(
            AFMDwarfStarModelIdentity.advertisedModelID(
                requestedModel: "./models/checkpoint.gguf",
                checkpointPath: "/workspace/models/checkpoint.gguf",
                requestedPathExists: { _ in false }
            ),
            "checkpoint.gguf"
        )
    }
}
