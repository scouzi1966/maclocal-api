import Foundation
import XCTest
@testable import AFMKitMLX

final class AFMMLXModelStoreTests: XCTestCase {
    func testCompleteModelDirectoryRequiresConfigAndWeights() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        XCTAssertFalse(AFMMLXModelStore.isCompleteModelDirectory(root))

        try JSONSerialization.data(withJSONObject: [:]).write(
            to: root.appendingPathComponent("config.json")
        )
        XCTAssertFalse(AFMMLXModelStore.isCompleteModelDirectory(root))

        try Data("weights".utf8).write(
            to: root.appendingPathComponent("weights.safetensors")
        )
        XCTAssertTrue(AFMMLXModelStore.isCompleteModelDirectory(root))
    }

    func testCompleteModelDirectoryRequiresEveryIndexedShard() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        try JSONSerialization.data(withJSONObject: [:]).write(
            to: root.appendingPathComponent("config.json")
        )
        let index: [String: Any] = [
            "weight_map": [
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors"
            ]
        ]
        try JSONSerialization.data(withJSONObject: index).write(
            to: root.appendingPathComponent("model.safetensors.index.json")
        )

        XCTAssertFalse(AFMMLXModelStore.isCompleteModelDirectory(root))

        try Data("weights".utf8).write(
            to: root.appendingPathComponent("model-00001-of-00002.safetensors")
        )
        XCTAssertFalse(AFMMLXModelStore.isCompleteModelDirectory(root))

        try Data("weights".utf8).write(
            to: root.appendingPathComponent("model-00002-of-00002.safetensors")
        )
        XCTAssertTrue(AFMMLXModelStore.isCompleteModelDirectory(root))
    }

    func testIdentifierCandidatesPreferCuratedRepositoryID() {
        XCTAssertEqual(
            AFMMLXModelStore.identifierCandidates(forModelName: "Qwen3-VL-4B-Instruct-5bit").first,
            "mlx-community/Qwen3-VL-4B-Instruct-5bit"
        )
    }

    func testIdentifierCandidatesIncludeCommunityFallbacksForPlainNames() {
        XCTAssertEqual(
            AFMMLXModelStore.identifierCandidates(forModelName: "Custom-Model-4bit"),
            [
                "mlx-community/Custom-Model-4bit",
                "lmstudio-community/Custom-Model-4bit",
            ]
        )
    }

    func testIdentifierCandidatesKeepExplicitRepositoryIDs() {
        XCTAssertEqual(
            AFMMLXModelStore.identifierCandidates(forModelName: "example-org/custom-model"),
            ["example-org/custom-model"]
        )
    }

    func testIdentifierCandidatesDropDuplicatesAndBlankInput() {
        let curated = [
            AFMMLXCuratedModel(
                displayName: "Local",
                repoID: "mlx-community/Local",
                capabilities: [.text],
                generationPreset: AFMMLXGenerationPreset()
            )
        ]

        XCTAssertEqual(
            AFMMLXModelStore.identifierCandidates(
                forModelName: "Local",
                curatedModels: curated,
                defaultOrganizations: ["mlx-community", "custom"]
            ),
            ["mlx-community/Local", "custom/Local"]
        )
        XCTAssertEqual(AFMMLXModelStore.identifierCandidates(forModelName: "  "), [])
    }

    func testLikelyRepositoryIdentifierRejectsPersistedFilesystemPaths() {
        XCTAssertTrue(AFMMLXModelStore.isLikelyRepositoryIdentifier("mlx-community/Qwen3"))
        XCTAssertTrue(AFMMLXModelStore.isLikelyRepositoryIdentifier("Qwen3"))

        XCTAssertFalse(AFMMLXModelStore.isLikelyRepositoryIdentifier(""))
        XCTAssertFalse(AFMMLXModelStore.isLikelyRepositoryIdentifier("   "))
        XCTAssertFalse(AFMMLXModelStore.isLikelyRepositoryIdentifier("/Volumes/models/Qwen3"))
        XCTAssertFalse(AFMMLXModelStore.isLikelyRepositoryIdentifier("Volumes/models/Qwen3"))
        XCTAssertFalse(AFMMLXModelStore.isLikelyRepositoryIdentifier(" users/example "))
    }

    func testProjectedRepositoryIDKeepsExplicitRepositoryID() {
        XCTAssertEqual(
            AFMMLXModelStore.projectedRepositoryID(from: " mlx-community/Qwen3 "),
            "mlx-community/Qwen3"
        )
    }

    func testProjectedRepositoryIDDerivesImportedPathSuffix() {
        XCTAssertEqual(
            AFMMLXModelStore.projectedRepositoryID(
                from: "imported:/Volumes/models/mlx-community/Qwen3-4B"
            ),
            "mlx-community/Qwen3-4B"
        )
    }

    func testProjectedRepositoryIDDerivesPlainPathSuffix() {
        XCTAssertEqual(
            AFMMLXModelStore.projectedRepositoryID(
                from: "/Volumes/models/lmstudio-community/Gemma-4bit"
            ),
            "lmstudio-community/Gemma-4bit"
        )
    }

    func testProjectedRepositoryIDRejectsInvalidSystemPathAuthors() {
        XCTAssertNil(AFMMLXModelStore.projectedRepositoryID(from: "/Volumes/models/local-model"))
        XCTAssertNil(AFMMLXModelStore.projectedRepositoryID(from: "Volumes/models/local-model"))
        XCTAssertNil(AFMMLXModelStore.projectedRepositoryID(from: " "))
    }

    func testProjectedRepositoryIDDoesNotProjectPlainDisplayNames() {
        XCTAssertNil(AFMMLXModelStore.projectedRepositoryID(from: "Qwen3-VL-4B-Instruct-5bit"))
    }

    func testSpecialtyModelIdentifierDetectsSpeechAndAudioModels() {
        XCTAssertTrue(AFMMLXModelStore.isSpecialtyModelIdentifier("prince-canuma/Kokoro-82M"))
        XCTAssertTrue(AFMMLXModelStore.isSpecialtyModelIdentifier("mlx-community/Orpheus-3B-0.1-ft-4bit"))
        XCTAssertTrue(AFMMLXModelStore.isSpecialtyModelIdentifier("example-org/fast-whisper-large"))
        XCTAssertTrue(AFMMLXModelStore.isSpecialtyModelIdentifier("marvis-ai/anything"))

        XCTAssertFalse(AFMMLXModelStore.isSpecialtyModelIdentifier("mlx-community/Qwen3-4B-4bit"))
        XCTAssertFalse(AFMMLXModelStore.isSpecialtyModelIdentifier("lmstudio-community/gemma-3-4b-it"))
    }

    func testCleanedPersistedModelRecordsPrefersCanonicalIDsAndFiltersInvalidEntries() {
        struct Record: Equatable {
            var id: String
            var name: String
        }

        let records = [
            Record(id: "Qwen3", name: "Qwen3"),
            Record(id: "Volumes/Crucial4TB/Qwen3", name: "Qwen3"),
            Record(id: "mlx-community/Qwen3", name: "Qwen3"),
            Record(id: "mlx-community/Curated", name: "Curated"),
            Record(id: "CustomBare", name: "CustomBare"),
            Record(id: "example-org/Other", name: "Other"),
            Record(id: "lmstudio-community/Other", name: "Other"),
        ]

        let cleaned = AFMMLXModelStore.cleanedPersistedModelRecords(
            records,
            id: \.id,
            displayName: \.name,
            isCurated: { $0 == "mlx-community/Curated" }
        )

        XCTAssertEqual(
            cleaned,
            [
                Record(id: "mlx-community/Qwen3", name: "Qwen3"),
                Record(id: "example-org/Other", name: "Other"),
                Record(id: "CustomBare", name: "CustomBare"),
            ]
        )
    }

    func testCompleteSnapshotDirectoryUsesExplicitRevision() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let snapshots = root.appendingPathComponent("snapshots", isDirectory: true)
        let resolved = snapshots.appendingPathComponent("aaa-resolved", isDirectory: true)
        let other = snapshots.appendingPathComponent("zzz-other", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try makeModel(at: resolved)
        try makeModel(at: other)

        XCTAssertEqual(
            AFMMLXModelStore.completeSnapshotDirectory(
                in: root,
                revision: "aaa-resolved"
            )?.path,
            resolved.path
        )
    }

    func testNewestCompleteSnapshotDirectoryUsesNewestCompleteSnapshot() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let snapshots = root.appendingPathComponent("snapshots", isDirectory: true)
        let incomplete = snapshots.appendingPathComponent("000-incomplete", isDirectory: true)
        let olderLexicographicallyLarger = snapshots.appendingPathComponent("zzz-older", isDirectory: true)
        let newerLexicographicallySmaller = snapshots.appendingPathComponent("aaa-newer", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try FileManager.default.createDirectory(at: incomplete, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: [:]).write(
            to: incomplete.appendingPathComponent("config.json")
        )
        try makeModel(at: olderLexicographicallyLarger)
        try makeModel(at: newerLexicographicallySmaller)

        let oldDate = Date(timeIntervalSince1970: 1_700_000_000)
        let newDate = Date(timeIntervalSince1970: 1_800_000_000)
        try FileManager.default.setAttributes(
            [.modificationDate: oldDate],
            ofItemAtPath: olderLexicographicallyLarger.path
        )
        try FileManager.default.setAttributes(
            [.modificationDate: newDate],
            ofItemAtPath: newerLexicographicallySmaller.path
        )

        XCTAssertEqual(
            AFMMLXModelStore.newestCompleteSnapshotDirectory(in: root)?.path,
            newerLexicographicallySmaller.path
        )
    }

    func testLocalDescriptorsValidateDeduplicateAndPreserveOrder() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try makeModel(at: root.appendingPathComponent("models/org/first"))
        try makeModel(at: root.appendingPathComponent("models/org/second"))

        let store = AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: root))
        let descriptors = store.localDescriptors(
            for: ["org/second", "missing/model", "org/first", "org/second"]
        )

        XCTAssertEqual(descriptors.map(\.modelID.rawValue), ["org/second", "org/first"])
        XCTAssertTrue(store.isAvailableLocally("org/first"))
        XCTAssertFalse(store.isAvailableLocally("missing/model"))
    }

    func testModelSelectionOptionsDeduplicateAndPreserveLoadedSource() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try makeModel(at: root.appendingPathComponent("models/org/loaded"))
        try makeModel(at: root.appendingPathComponent("models/org/other"))

        let store = AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: root))
        let options = store.modelSelectionOptions(
            selectedModelID: "org/missing",
            loadedModelID: "org/loaded",
            loadedDisplayName: "Loaded display",
            candidates: [
                AFMMLXModelCandidate(id: "org/loaded", displayName: "Downloaded display", sourceTag: "downloaded"),
                AFMMLXModelCandidate(id: "org/other", displayName: "Other display", sourceTag: "curated"),
                AFMMLXModelCandidate(id: "org/missing", displayName: "Missing display", sourceTag: "curated")
            ]
        )

        XCTAssertEqual(options.map(\.id), ["org/loaded", "org/other", "org/missing"])
        XCTAssertEqual(options[0].displayName, "Loaded display")
        XCTAssertEqual(options[0].sourceTag, "loaded")
        XCTAssertTrue(options[0].isAvailableLocally)
        XCTAssertEqual(options[1].displayName, "Other display")
        XCTAssertEqual(options[1].sourceTag, "curated")
        XCTAssertTrue(options[1].isAvailableLocally)
        XCTAssertEqual(options[2].sourceTag, "custom")
        XCTAssertFalse(options[2].isAvailableLocally)
    }

    func testIsVisionModelUsesLocalDescriptorCapabilities() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try makeModel(
            at: root.appendingPathComponent("models/org/vision-model"),
            config: [
                "architectures": ["QwenVLForConditionalGeneration"],
                "model_type": "qwen2"
            ]
        )
        try makeModel(
            at: root.appendingPathComponent("models/org/text-model"),
            config: ["model_type": "qwen3"]
        )

        let store = AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: root))

        XCTAssertTrue(store.isVisionModel("org/vision-model"))
        XCTAssertFalse(store.isVisionModel("org/text-model"))
        XCTAssertFalse(store.isVisionModel("org/missing"))
    }

    func testAbsoluteModelDirectoryUsesSharedValidation() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try makeModel(at: directory)

        let store = AFMMLXModelStore()

        XCTAssertEqual(store.localDirectory(for: directory.path)?.path, directory.path)
        XCTAssertEqual(store.descriptor(for: directory.path).requiresNetwork, false)
    }

    func testLoadReferenceKeepsRepoIdentifierForConfiguredCacheModel() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try makeModel(at: root.appendingPathComponent("models/org/repo-model"))

        let reference = try XCTUnwrap(
            AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: root))
                .loadReference(for: "org/repo-model")
        )

        XCTAssertEqual(reference.requestedID, "org/repo-model")
        XCTAssertEqual(reference.loadIdentifier, "org/repo-model")
        XCTAssertEqual(
            reference.localDirectory.path,
            root.appendingPathComponent("models/org/repo-model").path
        )
        XCTAssertEqual(reference.descriptor.requiresNetwork, false)
    }

    func testModelPresentationOverridesConfiguredIDDisplayNameAndCapabilityLabels() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("models/org/vision-model")
        try makeModel(
            at: directory,
            config: [
                "architectures": ["QwenVLForConditionalGeneration"],
                "model_type": "qwen2"
            ],
            tokenizer: ["chat_template": "<think>{{ tools }}<tool_call>"]
        )
        let reference = try XCTUnwrap(
            AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: root))
                .loadReference(for: "org/vision-model")
        )

        let repoPresentation = AFMMLXModelPresentation.make(
            configuredID: "custom/RuntimeID",
            loadReference: reference
        )
        let pathPresentation = AFMMLXModelPresentation.make(
            configuredID: "/custom/path/RuntimeID",
            loadReference: reference
        )

        XCTAssertEqual(repoPresentation.configuredID, "custom/RuntimeID")
        XCTAssertEqual(repoPresentation.localDirectory.path, directory.path)
        XCTAssertEqual(repoPresentation.descriptor.modelID.rawValue, "custom/RuntimeID")
        XCTAssertEqual(repoPresentation.descriptor.displayName, "RuntimeID")
        XCTAssertEqual(repoPresentation.descriptor.requiresNetwork, false)
        XCTAssertEqual(pathPresentation.descriptor.displayName, directory.lastPathComponent)
        XCTAssertTrue(repoPresentation.capabilityLabels.contains("vision"))
        XCTAssertTrue(repoPresentation.capabilityLabels.contains("tools"))
        XCTAssertTrue(repoPresentation.capabilityLabels.contains("structured"))
    }

    func testModelPresentationResolvesAlternateLoadIdentifier() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("models/org/imported-model")
        try makeModel(
            at: directory,
            config: [
                "model_type": "qwen3_vl",
                "max_position_embeddings": 65_536,
                "vision_config": ["hidden_size": 128]
            ],
            tokenizer: ["chat_template": "<think>{{ tools }}<tool_call>"]
        )

        let store = AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: root))
        let presentation = try XCTUnwrap(
            store.modelPresentation(
                configuredID: "/Volumes/External/ImportedModel",
                resolvedID: "org/imported-model"
            )
        )

        XCTAssertEqual(presentation.configuredID, "/Volumes/External/ImportedModel")
        XCTAssertEqual(presentation.localDirectory.path, directory.path)
        XCTAssertEqual(presentation.descriptor.modelID.rawValue, "/Volumes/External/ImportedModel")
        XCTAssertEqual(presentation.descriptor.displayName, directory.lastPathComponent)
        XCTAssertEqual(presentation.descriptor.contextWindow, 65_536)
        XCTAssertEqual(presentation.descriptor.requiresNetwork, false)
        XCTAssertTrue(presentation.capabilityLabels.contains("vision"))
        XCTAssertTrue(presentation.capabilityLabels.contains("tools"))
    }

    func testModelPresentationReturnsNilForBlankOrMissingModel() {
        let store = AFMMLXModelStore()

        XCTAssertNil(store.modelPresentation(configuredID: "  "))
        XCTAssertNil(store.modelPresentation(
            configuredID: "configured/model",
            resolvedID: "/missing/model-\(UUID().uuidString)"
        ))
    }

    func testLoadReferenceResolvesSwiftHubFlatModelByRepoID() throws {
        let documents = try XCTUnwrap(
            FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        )
        let org = "afmkit-test-\(UUID().uuidString)"
        let directory = documents
            .appendingPathComponent("huggingface/models")
            .appendingPathComponent(org)
            .appendingPathComponent("doc-model")
        defer {
            try? FileManager.default.removeItem(
                at: documents
                    .appendingPathComponent("huggingface/models")
                    .appendingPathComponent(org)
            )
        }
        try makeModel(at: directory)

        let reference = try XCTUnwrap(
            AFMMLXModelStore().loadReference(for: "\(org)/doc-model")
        )

        XCTAssertEqual(reference.loadIdentifier, "\(org)/doc-model")
        XCTAssertEqual(reference.localDirectory.path, directory.path)
    }

    func testLoadReferenceResolvesLibraryFlatModelByRepoID() throws {
        let library = try XCTUnwrap(
            FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first
        )
        let org = "afmkit-test-\(UUID().uuidString)"
        let directory = library
            .appendingPathComponent("Caches/models")
            .appendingPathComponent(org)
            .appendingPathComponent("cache-model")
        defer {
            try? FileManager.default.removeItem(
                at: library
                    .appendingPathComponent("Caches/models")
                    .appendingPathComponent(org)
            )
        }
        try makeModel(at: directory)

        let reference = try XCTUnwrap(
            AFMMLXModelStore().loadReference(for: "\(org)/cache-model")
        )

        XCTAssertEqual(reference.loadIdentifier, "\(org)/cache-model")
        XCTAssertEqual(reference.localDirectory.path, directory.path)
    }

    func testLoadReferenceResolvesAbsoluteSnapshotPath() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let snapshot = root
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent("revision", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try makeModel(at: snapshot)

        let reference = try XCTUnwrap(
            AFMMLXModelStore().loadReference(for: root.path)
        )

        XCTAssertEqual(reference.requestedID, root.path)
        XCTAssertEqual(reference.loadIdentifier, snapshot.path)
        XCTAssertEqual(reference.localDirectory.path, snapshot.path)
        XCTAssertEqual(reference.descriptor.requiresNetwork, false)
    }

    func testLoadReferenceUsesNewestCompleteSnapshotPath() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let snapshots = root.appendingPathComponent("snapshots", isDirectory: true)
        let incomplete = snapshots.appendingPathComponent("000-incomplete", isDirectory: true)
        let olderLexicographicallyLarger = snapshots.appendingPathComponent("zzz-older", isDirectory: true)
        let newerLexicographicallySmaller = snapshots.appendingPathComponent("aaa-newer", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try FileManager.default.createDirectory(at: incomplete, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: [:]).write(
            to: incomplete.appendingPathComponent("config.json")
        )
        try makeModel(at: olderLexicographicallyLarger)
        try makeModel(at: newerLexicographicallySmaller)

        let oldDate = Date(timeIntervalSince1970: 1_700_000_000)
        let newDate = Date(timeIntervalSince1970: 1_800_000_000)
        try FileManager.default.setAttributes(
            [.modificationDate: oldDate],
            ofItemAtPath: olderLexicographicallyLarger.path
        )
        try FileManager.default.setAttributes(
            [.modificationDate: newDate],
            ofItemAtPath: newerLexicographicallySmaller.path
        )

        let reference = try XCTUnwrap(
            AFMMLXModelStore().loadReference(for: root.path)
        )

        XCTAssertEqual(reference.localDirectory.path, newerLexicographicallySmaller.path)
        XCTAssertEqual(reference.loadIdentifier, newerLexicographicallySmaller.path)
    }

    func testRemovablePackageDirectoryUsesFlatModelDirectory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let directory = root.appendingPathComponent("models/org/flat-model")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeModel(at: directory)

        let store = AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: root))

        XCTAssertEqual(
            store.removablePackageDirectory(for: "org/flat-model")?.path,
            directory.path
        )
    }

    func testRemovablePackageDirectoryUsesHuggingFacePackageRoot() throws {
        let package = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let snapshot = package
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent("revision", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: package) }
        try makeModel(at: snapshot)

        XCTAssertEqual(
            AFMMLXModelStore().removablePackageDirectory(for: package.path)?.path,
            package.path
        )
    }

    func testDeleteLocalModelPackageRemovesFlatPackage() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let directory = root.appendingPathComponent("models/org/delete-me")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeModel(at: directory)

        let store = AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: root))
        let result = try store.deleteLocalModelPackage(for: "org/delete-me")

        XCTAssertEqual(result.requestedID, "org/delete-me")
        XCTAssertEqual(result.removedDirectory.path, directory.path)
        XCTAssertTrue(result.deleted)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

    func testDeleteLocalModelPackageRemovesHuggingFacePackageRoot() throws {
        let package = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let snapshot = package
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent("revision", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: package) }
        try makeModel(at: snapshot)

        let result = try AFMMLXModelStore().deleteLocalModelPackage(for: package.path)

        XCTAssertEqual(result.removedDirectory.path, package.path)
        XCTAssertTrue(result.deleted)
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.path))
    }

    func testDownloadModelPackageSkipsDownloaderWhenAlreadyCached() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let directory = root.appendingPathComponent("models/org/cached")
        defer { try? FileManager.default.removeItem(at: root) }
        try makeModel(at: directory)

        final class DownloadProbe: @unchecked Sendable {
            var called = false
        }
        let probe = DownloadProbe()
        let store = AFMMLXModelStore(
            resolver: MLXCacheResolver(cacheRoot: root),
            downloadSnapshot: { _, _, _ in
                probe.called = true
                return directory
            }
        )

        let result = try await store.downloadModelPackage(for: "org/cached")

        XCTAssertFalse(probe.called)
        XCTAssertEqual(result.loadReference.loadIdentifier, "org/cached")
        XCTAssertEqual(result.loadReference.localDirectory.path, directory.path)
    }

    func testDownloadModelPackageInvokesDownloaderAndReturnsLoadReference() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let directory = root.appendingPathComponent("models/org/downloaded")
        defer { try? FileManager.default.removeItem(at: root) }

        final class DownloadProbe: @unchecked Sendable {
            var progressSeen = false
        }
        let probe = DownloadProbe()
        let store = AFMMLXModelStore(
            resolver: MLXCacheResolver(cacheRoot: root),
            downloadSnapshot: { modelID, matching, progress in
                XCTAssertEqual(modelID, "org/downloaded")
                XCTAssertTrue(matching.contains("*.safetensors"))
                progress?(Progress(totalUnitCount: 1))
                probe.progressSeen = true
                try FileManager.default.createDirectory(
                    at: directory,
                    withIntermediateDirectories: true
                )
                try JSONSerialization.data(withJSONObject: [:]).write(
                    to: directory.appendingPathComponent("config.json")
                )
                try Data("weights".utf8).write(
                    to: directory.appendingPathComponent("weights.safetensors")
                )
                return directory
            }
        )

        let result = try await store.downloadModelPackage(for: "org/downloaded")

        XCTAssertTrue(probe.progressSeen)
        XCTAssertEqual(result.requestedID, "org/downloaded")
        XCTAssertEqual(result.downloadedDirectory.path, directory.path)
        XCTAssertEqual(result.loadReference.loadIdentifier, "org/downloaded")
        XCTAssertEqual(result.loadReference.localDirectory.path, directory.path)
    }

    func testDownloadTTSModelPackageUsesSpecialtyPatterns() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let directory = root.appendingPathComponent("models/org/tts-model")
        defer { try? FileManager.default.removeItem(at: root) }

        final class DownloadProbe: @unchecked Sendable {
            var matching: [String] = []
        }
        let probe = DownloadProbe()
        let store = AFMMLXModelStore(
            resolver: MLXCacheResolver(cacheRoot: root),
            downloadSnapshot: { _, matching, _ in
                probe.matching = matching
                try FileManager.default.createDirectory(
                    at: directory,
                    withIntermediateDirectories: true
                )
                try JSONSerialization.data(withJSONObject: [:]).write(
                    to: directory.appendingPathComponent("config.json")
                )
                try Data("weights".utf8).write(
                    to: directory.appendingPathComponent("weights.safetensors")
                )
                return directory
            }
        )

        let result = try await store.downloadTTSModelPackage(for: "org/tts-model")

        XCTAssertEqual(probe.matching, AFMMLXModelStore.ttsDownloadPatterns)
        XCTAssertEqual(result.requestedID, "org/tts-model")
        XCTAssertEqual(result.loadReference.localDirectory.path, directory.path)
    }

    func testDiscoveryReturnsTypedFlatAndHuggingFaceModels() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let flat = root.appendingPathComponent("flat", isDirectory: true)
        let hub = root.appendingPathComponent("hub", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try makeModel(
            at: flat.appendingPathComponent("org/flat-model"),
            contextWindow: 16_384
        )
        try makeModel(
            at: hub.appendingPathComponent(
                "models--org--hub-model/snapshots/revision"
            ),
            contextWindow: 32_768
        )
        try FileManager.default.createDirectory(
            at: flat.appendingPathComponent("org/incomplete"),
            withIntermediateDirectories: true
        )

        let store = AFMMLXModelStore(resolver: MLXCacheResolver(cacheRoot: flat))
        let models = store.discoverLocalModels(
            in: [
                .init(
                    directory: flat,
                    layout: .flat,
                    origin: .configuredCache
                ),
                .init(
                    directory: hub,
                    layout: .huggingFaceHub,
                    origin: .huggingFace
                )
            ]
        )

        XCTAssertEqual(
            models.map(\.id.rawValue),
            ["org/flat-model", "org/hub-model"]
        )
        XCTAssertEqual(models[0].loadIdentifier, "org/flat-model")
        XCTAssertEqual(models[0].descriptor.contextWindow, 16_384)
        XCTAssertEqual(models[0].origin, .configuredCache)
        XCTAssertEqual(
            models[0].packageDirectory.path,
            flat.appendingPathComponent("org/flat-model").path
        )
        XCTAssertGreaterThan(models[0].sizeBytes, 0)
        XCTAssertEqual(
            models[1].loadIdentifier,
            models[1].localDirectory.path
        )
        XCTAssertEqual(models[1].descriptor.contextWindow, 32_768)
        XCTAssertEqual(models[1].origin, .huggingFace)
        XCTAssertEqual(
            models[1].packageDirectory.path,
            hub.appendingPathComponent("models--org--hub-model").path
        )
        XCTAssertGreaterThan(models[1].sizeBytes, 0)
    }

    func testDiscoveryDeduplicatesCanonicalIDByLocationPrecedence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let first = root.appendingPathComponent("first", isDirectory: true)
        let second = root.appendingPathComponent("second", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try makeModel(at: first.appendingPathComponent("org/model"))
        try makeModel(at: second.appendingPathComponent("org/model"))

        let models = AFMMLXModelStore().discoverLocalModels(
            in: [
                .init(
                    directory: first,
                    layout: .flat,
                    origin: .swiftHub
                ),
                .init(
                    directory: second,
                    layout: .flat,
                    origin: .lmStudio
                )
            ]
        )

        XCTAssertEqual(models.count, 1)
        XCTAssertEqual(models.first?.origin, .swiftHub)
        XCTAssertEqual(
            models.first?.localDirectory.path,
            first.appendingPathComponent("org/model").path
        )
    }

    private func makeModel(
        at directory: URL,
        contextWindow: Int? = nil,
        config explicitConfig: [String: Any]? = nil,
        tokenizer: [String: Any]? = nil
    ) throws {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        var config = explicitConfig ?? [:]
        if let contextWindow {
            config["max_position_embeddings"] = contextWindow
        }
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json")
        )
        if let tokenizer {
            try JSONSerialization.data(withJSONObject: tokenizer).write(
                to: directory.appendingPathComponent("tokenizer_config.json")
            )
        }
        try Data("weights".utf8).write(
            to: directory.appendingPathComponent("weights.safetensors")
        )
    }
}
