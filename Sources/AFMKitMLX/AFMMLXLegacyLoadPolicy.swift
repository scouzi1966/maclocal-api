import Foundation
@preconcurrency import MLXLMCommon

public struct AFMMLXRuntimeFactoryDecision: Equatable, Sendable {
    public let isVisionModelDirectory: Bool
    public let requiresVisionFactory: Bool

    public init(isVisionModelDirectory: Bool, requiresVisionFactory: Bool) {
        self.isVisionModelDirectory = isVisionModelDirectory
        self.requiresVisionFactory = requiresVisionFactory
    }
}

public enum AFMMLXLegacyLoadSource: Equatable, Sendable {
    case importedDirectory(URL)
    case cachedDirectory(URL)
    case remoteRepository(String)
    case defaultConfiguration
}

public enum AFMMLXLegacyLoadModelKind: Equatable, Sendable {
    case llm
    case vlm
}

public struct AFMMLXLegacyLoadModelDescriptor {
    public let name: String
    public let customModelPath: String?
    public let configuration: ModelConfiguration
    public let kind: AFMMLXLegacyLoadModelKind

    public init(
        name: String,
        customModelPath: String?,
        configuration: ModelConfiguration,
        kind: AFMMLXLegacyLoadModelKind
    ) {
        self.name = name
        self.customModelPath = customModelPath
        self.configuration = configuration
        self.kind = kind
    }
}

public struct AFMMLXLegacyLoadPlan {
    public let repoID: String
    public let source: AFMMLXLegacyLoadSource
    public let configuration: ModelConfiguration
    public let localDirectory: URL?
    public let isDownloadingPhase: Bool
    public let factoryDecision: AFMMLXRuntimeFactoryDecision

    public var forceVisionFactory: Bool {
        factoryDecision.requiresVisionFactory
    }

    public var configIsVision: Bool {
        localDirectory.map { _ in factoryDecision.isVisionModelDirectory } ?? false
    }

    public init(
        repoID: String,
        source: AFMMLXLegacyLoadSource,
        configuration: ModelConfiguration,
        localDirectory: URL?,
        isDownloadingPhase: Bool,
        factoryDecision: AFMMLXRuntimeFactoryDecision
    ) {
        self.repoID = repoID
        self.source = source
        self.configuration = configuration
        self.localDirectory = localDirectory
        self.isDownloadingPhase = isDownloadingPhase
        self.factoryDecision = factoryDecision
    }
}

public struct AFMMLXLegacyModelLoadResolutionPlan: Equatable, Sendable {
    public let localDirectory: URL?
    public let factoryDecision: AFMMLXRuntimeFactoryDecision?
    public let resolvedKind: AFMMLXLegacyLoadModelKind
    public let correctedFromKind: AFMMLXLegacyLoadModelKind?

    public var wasCorrected: Bool {
        correctedFromKind != nil
    }

    public init(
        localDirectory: URL?,
        factoryDecision: AFMMLXRuntimeFactoryDecision?,
        resolvedKind: AFMMLXLegacyLoadModelKind,
        correctedFromKind: AFMMLXLegacyLoadModelKind?
    ) {
        self.localDirectory = localDirectory
        self.factoryDecision = factoryDecision
        self.resolvedKind = resolvedKind
        self.correctedFromKind = correctedFromKind
    }
}

public enum AFMMLXLegacyLoadPolicy {
    public static func resolveModelForLoading(
        model: AFMMLXLegacyLoadModelDescriptor,
        localDirectoryForRepo: (String) -> URL?,
        factoryDecision: (URL) -> AFMMLXRuntimeFactoryDecision
    ) -> AFMMLXLegacyModelLoadResolutionPlan {
        guard let localDirectory = localDirectory(for: model, localDirectoryForRepo: localDirectoryForRepo) else {
            return AFMMLXLegacyModelLoadResolutionPlan(
                localDirectory: nil,
                factoryDecision: nil,
                resolvedKind: model.kind,
                correctedFromKind: nil
            )
        }

        let decision = factoryDecision(localDirectory)
        let resolvedKind: AFMMLXLegacyLoadModelKind = decision.requiresVisionFactory ? .vlm : .llm
        let correctedFromKind = resolvedKind == model.kind ? nil : model.kind

        return AFMMLXLegacyModelLoadResolutionPlan(
            localDirectory: localDirectory,
            factoryDecision: decision,
            resolvedKind: resolvedKind,
            correctedFromKind: correctedFromKind
        )
    }

    public static func make(
        model: AFMMLXLegacyLoadModelDescriptor,
        localDirectoryForRepo: (String) -> URL?,
        factoryDecision: (URL?) -> AFMMLXRuntimeFactoryDecision
    ) -> AFMMLXLegacyLoadPlan {
        let repoID = model.customModelPath ?? "mlx-community/\(model.name)"

        if let customPath = model.customModelPath, customPath.hasPrefix("/") {
            let localDirectory = URL(fileURLWithPath: customPath)
            var configuration = model.configuration
            configuration.id = .directory(localDirectory)
            return AFMMLXLegacyLoadPlan(
                repoID: repoID,
                source: .importedDirectory(localDirectory),
                configuration: configuration,
                localDirectory: localDirectory,
                isDownloadingPhase: false,
                factoryDecision: factoryDecision(localDirectory)
            )
        }

        if let localDirectory = localDirectoryForRepo(repoID) {
            var configuration = model.configuration
            configuration.id = .directory(localDirectory)
            return AFMMLXLegacyLoadPlan(
                repoID: repoID,
                source: .cachedDirectory(localDirectory),
                configuration: configuration,
                localDirectory: localDirectory,
                isDownloadingPhase: false,
                factoryDecision: factoryDecision(localDirectory)
            )
        }

        if let customPath = model.customModelPath {
            var configuration = model.configuration
            configuration.id = .id(customPath, revision: "main")
            return AFMMLXLegacyLoadPlan(
                repoID: repoID,
                source: .remoteRepository(customPath),
                configuration: configuration,
                localDirectory: nil,
                isDownloadingPhase: true,
                factoryDecision: factoryDecision(nil)
            )
        }

        return AFMMLXLegacyLoadPlan(
            repoID: repoID,
            source: .defaultConfiguration,
            configuration: model.configuration,
            localDirectory: nil,
            isDownloadingPhase: true,
            factoryDecision: factoryDecision(nil)
        )
    }

    private static func localDirectory(
        for model: AFMMLXLegacyLoadModelDescriptor,
        localDirectoryForRepo: (String) -> URL?
    ) -> URL? {
        if let customPath = model.customModelPath, customPath.hasPrefix("/") {
            return URL(fileURLWithPath: customPath)
        }
        let repoID = model.customModelPath ?? "mlx-community/\(model.name)"
        return localDirectoryForRepo(repoID)
    }
}
