import XCTest
@testable import AFMKitMLX

final class AFMMLXLegacyLoadPolicyTests: XCTestCase {
    func testPlansImportedDirectoryWithoutDownloadPhase() {
        let directory = URL(fileURLWithPath: "/Volumes/models/Imported")
        let model = model(name: "Imported", customModelPath: directory.path, kind: .llm)

        let plan = AFMMLXLegacyLoadPolicy.make(
            model: model,
            localDirectoryForRepo: { _ in nil },
            factoryDecision: { directory in
                AFMMLXRuntimeFactoryDecision(
                    isVisionModelDirectory: directory?.path.contains("Imported") == true,
                    requiresVisionFactory: false
                )
            }
        )

        XCTAssertEqual(plan.repoID, directory.path)
        XCTAssertEqual(plan.source, .importedDirectory(directory))
        XCTAssertEqual(plan.localDirectory, directory)
        XCTAssertFalse(plan.isDownloadingPhase)
        XCTAssertFalse(plan.forceVisionFactory)
        XCTAssertTrue(plan.configIsVision)
    }

    func testPlansCachedRepositoryBeforeRemoteDownload() {
        let cachedDirectory = URL(fileURLWithPath: "/Users/test/Library/Caches/models/mlx-community/Qwen")
        let model = model(name: "Qwen", customModelPath: "mlx-community/Qwen", kind: .llm)

        let plan = AFMMLXLegacyLoadPolicy.make(
            model: model,
            localDirectoryForRepo: { repoID in
                repoID == "mlx-community/Qwen" ? cachedDirectory : nil
            },
            factoryDecision: { _ in
                AFMMLXRuntimeFactoryDecision(isVisionModelDirectory: false, requiresVisionFactory: false)
            }
        )

        XCTAssertEqual(plan.repoID, "mlx-community/Qwen")
        XCTAssertEqual(plan.source, .cachedDirectory(cachedDirectory))
        XCTAssertEqual(plan.localDirectory, cachedDirectory)
        XCTAssertFalse(plan.isDownloadingPhase)
    }

    func testPlansRemoteRepositoryWhenNotCached() {
        let model = model(name: "Remote", customModelPath: "mlx-community/Remote", kind: .llm)

        let plan = AFMMLXLegacyLoadPolicy.make(
            model: model,
            localDirectoryForRepo: { _ in nil },
            factoryDecision: { directory in
                XCTAssertNil(directory)
                return AFMMLXRuntimeFactoryDecision(isVisionModelDirectory: false, requiresVisionFactory: false)
            }
        )

        XCTAssertEqual(plan.repoID, "mlx-community/Remote")
        XCTAssertEqual(plan.source, .remoteRepository("mlx-community/Remote"))
        XCTAssertNil(plan.localDirectory)
        XCTAssertTrue(plan.isDownloadingPhase)
    }

    func testPlansDefaultConfigurationForCuratedModelWithoutCustomPath() {
        let model = model(name: "Curated", customModelPath: nil, kind: .vlm)

        let plan = AFMMLXLegacyLoadPolicy.make(
            model: model,
            localDirectoryForRepo: { _ in nil },
            factoryDecision: { directory in
                XCTAssertNil(directory)
                return AFMMLXRuntimeFactoryDecision(isVisionModelDirectory: false, requiresVisionFactory: false)
            }
        )

        XCTAssertEqual(plan.repoID, "mlx-community/Curated")
        XCTAssertEqual(plan.source, .defaultConfiguration)
        XCTAssertNil(plan.localDirectory)
        XCTAssertTrue(plan.isDownloadingPhase)
        XCTAssertFalse(plan.configIsVision)
    }

    func testResolvesLocalModelToVisionFactoryWhenRequiredByDescriptor() {
        let directory = URL(fileURLWithPath: "/Volumes/models/Sparse-VLM")
        let model = model(name: "Sparse-VLM", customModelPath: directory.path, kind: .llm)

        let resolution = AFMMLXLegacyLoadPolicy.resolveModelForLoading(
            model: model,
            localDirectoryForRepo: { _ in nil },
            factoryDecision: { receivedDirectory in
                XCTAssertEqual(receivedDirectory, directory)
                return AFMMLXRuntimeFactoryDecision(
                    isVisionModelDirectory: true,
                    requiresVisionFactory: true
                )
            }
        )

        XCTAssertTrue(resolution.wasCorrected)
        XCTAssertEqual(resolution.correctedFromKind, .llm)
        XCTAssertEqual(resolution.resolvedKind, .vlm)
        XCTAssertEqual(resolution.localDirectory, directory)
    }

    func testKeepsLocalModelKindWhenDescriptorMatchesRequest() {
        let directory = URL(fileURLWithPath: "/Volumes/models/Text-Model")
        let model = model(name: "Text-Model", customModelPath: directory.path, kind: .llm)

        let resolution = AFMMLXLegacyLoadPolicy.resolveModelForLoading(
            model: model,
            localDirectoryForRepo: { _ in nil },
            factoryDecision: { _ in
                AFMMLXRuntimeFactoryDecision(
                    isVisionModelDirectory: false,
                    requiresVisionFactory: false
                )
            }
        )

        XCTAssertFalse(resolution.wasCorrected)
        XCTAssertNil(resolution.correctedFromKind)
        XCTAssertEqual(resolution.resolvedKind, .llm)
        XCTAssertEqual(resolution.localDirectory, directory)
    }

    func testLeavesRemoteModelUnresolvedUntilDownloadProvidesLocalDescriptor() {
        let model = model(name: "Remote", customModelPath: "mlx-community/Remote", kind: .vlm)

        let resolution = AFMMLXLegacyLoadPolicy.resolveModelForLoading(
            model: model,
            localDirectoryForRepo: { _ in nil },
            factoryDecision: { _ in
                XCTFail("Remote uncached models do not have a local descriptor yet")
                return AFMMLXRuntimeFactoryDecision(isVisionModelDirectory: false, requiresVisionFactory: false)
            }
        )

        XCTAssertFalse(resolution.wasCorrected)
        XCTAssertNil(resolution.localDirectory)
        XCTAssertEqual(resolution.resolvedKind, .vlm)
    }

    private func model(
        name: String,
        customModelPath: String?,
        kind: AFMMLXLegacyLoadModelKind
    ) -> AFMMLXLegacyLoadModelDescriptor {
        AFMMLXLegacyLoadModelDescriptor(
            name: name,
            customModelPath: customModelPath,
            configuration: AFMMLXModelCatalog.genericModelConfiguration(isVision: kind == .vlm),
            kind: kind
        )
    }
}
