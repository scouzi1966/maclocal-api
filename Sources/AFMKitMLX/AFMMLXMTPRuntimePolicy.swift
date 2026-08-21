import Foundation

enum AFMMLXMTPRuntimeModelKind: Equatable, Sendable {
    case qwenText
    case qwenVision
}

/// Pure lifecycle decisions shared by model loading and request dispatch.
/// Keeping these rules independent of MLX objects makes failure/retry and
/// model-switch behavior deterministic and directly testable.
enum AFMMLXMTPRuntimePolicy {
    static func compatibleModelKind(
        mtpEnabled: Bool,
        factory: AFMMLXModelFactoryKind,
        canonicalModelType: String
    ) -> AFMMLXMTPRuntimeModelKind? {
        guard mtpEnabled,
              canonicalModelType == "qwen3_5" || canonicalModelType == "qwen3_5_moe"
        else { return nil }
        return factory == .vlm ? .qwenVision : .qwenText
    }

    static func canReuseLoadedModel(
        loadedModelID: String?,
        requestedModelID: String,
        mtpEnabled: Bool,
        bindingModelID: String?
    ) -> Bool {
        guard loadedModelID == requestedModelID else { return false }
        return !mtpEnabled || bindingModelID == requestedModelID
    }

    static func bindingIsUsable(
        for requestedModelID: String,
        mtpEnabled: Bool,
        bindingModelID: String?
    ) -> Bool {
        mtpEnabled && bindingModelID == requestedModelID
    }

    static func allowSynchronousSidecarDownload(mtpEnabled: Bool) -> Bool {
        mtpEnabled
    }

    static func shouldPrefetchInBackground(
        mtpEnabled: Bool,
        resolvedSidecar: String?,
        automaticRepositoryID: String?
    ) -> Bool {
        !mtpEnabled && resolvedSidecar == nil && automaticRepositoryID != nil
    }

    static func directSidecarHasRequiredMetadata(_ sidecarURL: URL) -> Bool {
        AFMMLXSpeculativeRuntimeResourceResolver.mtpQuantization(
            resourceDirectory: sidecarURL.deletingLastPathComponent()
        ) != nil
    }
}
