import Foundation

public enum AFMMLXVisionAssetIssue: String, CaseIterable, Hashable, Sendable {
    case conditionalGenerationArchitecture
    case visionConfiguration
    case imageTokenIdentifiers
    case processorConfiguration
    case visionWeights
}

public struct AFMMLXVisionAssetQualification: Hashable, Sendable {
    public let snapshotIdentity: String
    public let modelType: String
    public let canonicalModelType: String
    public let isConditionalGeneration: Bool
    public let declaresVision: Bool
    public let processorClass: String?
    public let visionTensorCount: Int
    public let missingAssets: Set<AFMMLXVisionAssetIssue>

    public init(
        snapshotIdentity: String,
        modelType: String,
        canonicalModelType: String,
        isConditionalGeneration: Bool,
        declaresVision: Bool,
        processorClass: String?,
        visionTensorCount: Int,
        missingAssets: Set<AFMMLXVisionAssetIssue>
    ) {
        self.snapshotIdentity = snapshotIdentity
        self.modelType = modelType
        self.canonicalModelType = canonicalModelType
        self.isConditionalGeneration = isConditionalGeneration
        self.declaresVision = declaresVision
        self.processorClass = processorClass
        self.visionTensorCount = visionTensorCount
        self.missingAssets = missingAssets
    }

    public var isAssetUsable: Bool {
        declaresVision && missingAssets.isEmpty
    }

    public var isUsableQwenConditionalGeneration: Bool {
        Self.qwenConditionalModelTypes.contains(canonicalModelType)
            && isConditionalGeneration
            && isAssetUsable
    }

    public var missingAssetNames: [String] {
        missingAssets.map(\.rawValue).sorted()
    }

    private static let qwenConditionalModelTypes: Set<String> = [
        "qwen3_5",
        "qwen3_5_moe",
    ]
}
