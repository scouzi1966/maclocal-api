public struct AFMMLXImportedModelSelectionPlan: Equatable, Sendable {
    public let name: String
    public let path: String
    public let isVision: Bool
    public let textOnlyMode: Bool

    public init(
        name: String,
        path: String,
        isVision: Bool,
        textOnlyMode: Bool
    ) {
        self.name = name
        self.path = path
        self.isVision = isVision
        self.textOnlyMode = textOnlyMode
    }
}

public enum AFMMLXImportedModelSelectionPolicy {
    public static func make(
        name: String,
        path: String,
        isVision: Bool,
        mtpCompatible: Bool
    ) -> AFMMLXImportedModelSelectionPlan {
        AFMMLXImportedModelSelectionPlan(
            name: name,
            path: path,
            isVision: isVision,
            textOnlyMode: mtpCompatible || !isVision
        )
    }
}
