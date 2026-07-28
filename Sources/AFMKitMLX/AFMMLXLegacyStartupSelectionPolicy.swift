import Foundation

public struct AFMMLXLegacyStartupModelCandidate: Equatable, Sendable {
    public let name: String
    public let id: String
    public let isAvailable: Bool

    public init(name: String, id: String, isAvailable: Bool) {
        self.name = name
        self.id = id
        self.isAvailable = isAvailable
    }
}

public struct AFMMLXLegacyStartupSelection: Equatable, Sendable {
    public let modelName: String
    public let afm27ModelID: String

    public init(modelName: String, afm27ModelID: String) {
        self.modelName = modelName
        self.afm27ModelID = afm27ModelID
    }
}

public enum AFMMLXLegacyStartupSelectionPolicy {
    public static func select(
        loadedModelName: String?,
        loadedModelRepoID: String?,
        selectedModelName: String,
        candidates: [AFMMLXLegacyStartupModelCandidate]
    ) -> AFMMLXLegacyStartupSelection? {
        let available = candidates.filter(\.isAvailable)
        guard !available.isEmpty else { return nil }

        let trimmedLoadedName = loadedModelName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmedLoadedName.isEmpty,
           let loaded = available.first(where: { $0.name == trimmedLoadedName }) {
            let loadedID = loadedModelRepoID?.trimmingCharacters(in: .whitespacesAndNewlines)
            let afm27ModelID: String
            if let loadedID, !loadedID.isEmpty {
                afm27ModelID = loadedID
            } else {
                afm27ModelID = loaded.id
            }
            return AFMMLXLegacyStartupSelection(
                modelName: loaded.name,
                afm27ModelID: afm27ModelID
            )
        }

        let selected = selectedModelName.trimmingCharacters(in: .whitespacesAndNewlines)
        if !selected.isEmpty,
           let persisted = available.first(where: { $0.name == selected }) {
            return AFMMLXLegacyStartupSelection(
                modelName: persisted.name,
                afm27ModelID: persisted.id
            )
        }

        if let defaultFiveBit = available.first(where: { $0.name.contains("5bit") }) {
            return AFMMLXLegacyStartupSelection(
                modelName: defaultFiveBit.name,
                afm27ModelID: defaultFiveBit.id
            )
        }

        let first = available[0]
        return AFMMLXLegacyStartupSelection(
            modelName: first.name,
            afm27ModelID: first.id
        )
    }
}
