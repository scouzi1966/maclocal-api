import Foundation

public enum AFMMLXStartupModelSelectionPolicy {
    public static func select(
        options: [AFMMLXModelSelectionOption],
        selectedDisplayName: String,
        defaultOptionIDs: Set<String> = [],
        defaultDisplayNames: Set<String> = [],
        loadedSourceTag: String = "loaded"
    ) -> AFMMLXModelSelectionOption? {
        let availableOptions = options.filter(\.isAvailableLocally)
        guard !availableOptions.isEmpty else { return nil }

        let selectedName = selectedDisplayName.trimmingCharacters(in: .whitespacesAndNewlines)
        let selectedOption = selectedName.isEmpty
            ? nil
            : availableOptions.first { option in
                option.displayName == selectedName || displayName(for: option.id) == selectedName
            }

        let defaultOption = availableOptions.first { option in
            defaultOptionIDs.contains(option.id) || defaultDisplayNames.contains(option.displayName)
        }

        return availableOptions.first { $0.sourceTag == loadedSourceTag }
            ?? selectedOption
            ?? defaultOption
            ?? availableOptions.first
    }

    private static func displayName(for modelID: String) -> String {
        if modelID.hasPrefix("/") {
            return URL(fileURLWithPath: modelID).lastPathComponent
        }
        return modelID.split(separator: "/").last.map(String.init) ?? modelID
    }
}
