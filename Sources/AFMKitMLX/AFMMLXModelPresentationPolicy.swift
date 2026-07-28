import Foundation

public struct AFMMLXAutoLoadCandidate: Equatable, Sendable {
    public let name: String
    public let isAvailable: Bool

    public init(name: String, isAvailable: Bool) {
        self.name = name
        self.isAvailable = isAvailable
    }
}

public struct AFMMLXBenchmarkModelCandidate: Equatable, Sendable {
    public let id: String
    public let isAvailable: Bool

    public init(id: String, isAvailable: Bool) {
        self.id = id
        self.isAvailable = isAvailable
    }
}

public struct AFMMLXDisplayModelCandidate: Equatable, Sendable {
    public let name: String
    public let displayName: String

    public init(name: String, displayName: String) {
        self.name = name
        self.displayName = displayName
    }
}

public struct AFMMLXParameterPresetCandidate<Parameters> {
    public let name: String
    public let parameters: Parameters?

    public init(name: String, parameters: Parameters?) {
        self.name = name
        self.parameters = parameters
    }
}

public struct AFMMLXCuratedModelCandidate: Equatable, Sendable {
    public let name: String

    public init(name: String) {
        self.name = name
    }
}

public enum AFMMLXModelPresentationPolicy {
    public static func displayName(
        forSelection selection: String,
        loadedModelName: String?,
        curatedCandidates: [AFMMLXDisplayModelCandidate]
    ) -> String {
        let trimmedLoadedName = loadedModelName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmedLoadedName.isEmpty {
            return trimmedLoadedName
        }

        let trimmedSelection = selection.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedSelection.isEmpty else { return trimmedSelection }
        return curatedCandidates.first(where: { $0.name == trimmedSelection })?.displayName
            ?? trimmedSelection
    }

    public static func parameterPreset<Parameters>(
        forSelection selection: String,
        curatedCandidates: [AFMMLXParameterPresetCandidate<Parameters>]
    ) -> Parameters? {
        let trimmedSelection = selection.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedSelection.isEmpty else { return nil }
        return curatedCandidates.first(where: { $0.name == trimmedSelection })?.parameters
    }

    public static func modelBitDepth(for modelName: String) -> Int? {
        guard let last = modelName.split(separator: "-").last else {
            return nil
        }
        let lastString = String(last)
        guard lastString.hasSuffix("bit"), lastString.count > 3 else {
            return nil
        }
        return Int(lastString.dropLast(3))
    }

    public static func curatedModelNames(
        in family: String,
        candidates: [AFMMLXCuratedModelCandidate]
    ) -> [String] {
        candidates
            .map(\.name)
            .filter { AFMMLXLoadSelectionPolicy.modelFamily(for: $0) == family }
            .sorted { lhs, rhs in
                if lhs.contains("bf16") && !rhs.contains("bf16") {
                    return false
                }
                if !lhs.contains("bf16") && rhs.contains("bf16") {
                    return true
                }
                return (modelBitDepth(for: lhs) ?? 0) < (modelBitDepth(for: rhs) ?? 0)
            }
    }

    public static func autoLoadModelName(
        selectedModelName: String,
        defaultModelName: String,
        candidates: [AFMMLXAutoLoadCandidate]
    ) -> String? {
        let selected = selectedModelName.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallback = defaultModelName.trimmingCharacters(in: .whitespacesAndNewlines)

        if !selected.isEmpty,
           candidates.contains(where: { $0.name == selected && $0.isAvailable }) {
            return selected
        }

        if !fallback.isEmpty,
           candidates.contains(where: { $0.name == fallback && $0.isAvailable }) {
            return fallback
        }

        return nil
    }

    public static func benchmarkModelSelectionIDs(
        curatedCandidates: [AFMMLXBenchmarkModelCandidate],
        downloadedCandidates: [AFMMLXBenchmarkModelCandidate]
    ) -> [String] {
        var seen = Set<String>()
        return (curatedCandidates + downloadedCandidates)
            .filter(\.isAvailable)
            .map(\.id)
            .filter { !$0.isEmpty }
            .filter { seen.insert($0).inserted }
            .sorted()
    }
}
