import Foundation

public enum AFMMLXLoadedModeSwitchPlan: Equatable, Sendable {
    case imported(rawPath: String, targetVLM: Bool)
    case currentLoadedModel(targetVLM: Bool)

    public var targetVLM: Bool {
        switch self {
        case .imported(_, let targetVLM), .currentLoadedModel(let targetVLM):
            targetVLM
        }
    }
}

public enum AFMMLXLoadedModeSwitchPolicy {
    public static func make(
        loadedModelRepoID: String?,
        loadedModelType: String?,
        isLoadedModelVLM: Bool,
        loadedModelDirectoryIsVision: Bool
    ) -> AFMMLXLoadedModeSwitchPlan? {
        let trimmedModelType = loadedModelType?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmedModelType.isEmpty,
              AFMMLXModelArchitecture.isDualModeModelType(trimmedModelType),
              loadedModelDirectoryIsVision else {
            return nil
        }

        let targetVLM = !isLoadedModelVLM
        let trimmedRepoID = loadedModelRepoID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if let importedPath = AFMMLXQuickReloadPolicy.importedPath(from: trimmedRepoID) {
            return .imported(rawPath: importedPath, targetVLM: targetVLM)
        }
        return .currentLoadedModel(targetVLM: targetVLM)
    }
}

public enum AFMMLXModelFactoryKind: Equatable, Sendable {
    case llm
    case vlm
}

public enum AFMMLXModelFactoryPolicy {
    public static func initialFactory(
        forceVLM: Bool,
        architecture: AFMMLXModelArchitecturePreflight,
        visionQualification: AFMMLXVisionAssetQualification? = nil
    ) -> AFMMLXModelFactoryKind {
        if forceVLM || architecture.requiresVisionModelFactory {
            return .vlm
        }
        if visionQualification?.isUsableQwenConditionalGeneration == true {
            return .vlm
        }
        return .llm
    }
}

public enum AFMMLXRequestMediaKind: Hashable, Sendable {
    case image
    case video
    case audio

    public var label: String {
        switch self {
        case .image: "image"
        case .video: "video"
        case .audio: "audio"
        }
    }
}

public enum AFMMLXRequestMediaPolicy {
    private static let videoModelTypes: Set<String> = [
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_6",
        "qwen3_6_moe",
        "smolvlm",
    ]

    public static func kind(contentPartType: String, mediaURL: String? = nil) -> AFMMLXRequestMediaKind? {
        switch contentPartType {
        case "input_audio":
            return .audio
        case "image_url":
            guard let mediaURL else { return .image }
            if mediaURL.lowercased().hasPrefix("data:video/") {
                return .video
            }
            if let url = URL(string: mediaURL), videoExtensions.contains(url.pathExtension.lowercased()) {
                return .video
            }
            return .image
        default:
            return nil
        }
    }

    public static func supports(
        _ kind: AFMMLXRequestMediaKind,
        architecture: AFMMLXModelArchitecturePreflight
    ) -> Bool {
        switch kind {
        case .audio:
            return false
        case .image:
            return architecture.isVisionConfiguration
        case .video:
            return architecture.isVisionConfiguration
                && videoModelTypes.contains(architecture.canonicalModelType)
        }
    }

    private static let videoExtensions: Set<String> = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]
}
