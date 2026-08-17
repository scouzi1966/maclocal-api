import AFMKitCore

public enum AFMMLXRuntimeMediaAdmission: Equatable, Sendable {
    case allowed
    case unsupported
    case visionAssetsUnavailable(missing: [String])
}

public enum AFMMLXRuntimeVisionPolicy {
    public static func supportsVision(
        architecture: AFMMLXModelArchitecturePreflight,
        qualification: AFMMLXVisionAssetQualification,
        factory: AFMMLXModelFactoryKind
    ) -> Bool {
        guard factory == .vlm, architecture.isVisionConfiguration else {
            return false
        }
        if qualification.isQwenConditionalGeneration {
            return qualification.isAssetUsable
        }
        return true
    }

    public static func admission(
        for kind: AFMMLXRequestMediaKind,
        architecture: AFMMLXModelArchitecturePreflight,
        qualification: AFMMLXVisionAssetQualification,
        factory: AFMMLXModelFactoryKind
    ) -> AFMMLXRuntimeMediaAdmission {
        guard AFMMLXRequestMediaPolicy.supports(kind, architecture: architecture) else {
            return .unsupported
        }
        if qualification.isQwenConditionalGeneration && !qualification.isAssetUsable {
            return .visionAssetsUnavailable(missing: qualification.missingAssetNames)
        }
        guard factory == .vlm else {
            return .unsupported
        }
        return .allowed
    }

    public static func runtimeDescriptor(
        declared descriptor: AFMModelDescriptor,
        architecture: AFMMLXModelArchitecturePreflight,
        qualification: AFMMLXVisionAssetQualification,
        factory: AFMMLXModelFactoryKind
    ) -> AFMModelDescriptor {
        var descriptor = descriptor
        if supportsVision(
            architecture: architecture,
            qualification: qualification,
            factory: factory
        ) {
            descriptor.capabilities.insert(.vision)
        } else {
            descriptor.capabilities.remove(.vision)
        }
        return descriptor
    }
}
