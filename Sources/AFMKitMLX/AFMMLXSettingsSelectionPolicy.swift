public enum AFMMLXSettingsSelectionPolicy {
    public static func syncedModelID(
        currentModelID: String,
        selectedLegacyModelID: String,
        options: [AFMMLXModelSelectionOption],
        force: Bool = false
    ) -> String? {
        guard let option = option(matchingLegacySelection: selectedLegacyModelID, options: options) else {
            return nil
        }

        if force || currentModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return option.id
        }
        return nil
    }

    public static func legacySelectionDisplayName(
        for modelID: String,
        options: [AFMMLXModelSelectionOption],
        currentLegacyDisplayName: String,
        customSourceTag: String = "custom"
    ) -> String? {
        guard let option = options.first(where: { $0.id == modelID && $0.sourceTag != customSourceTag }),
              currentLegacyDisplayName != option.displayName else {
            return nil
        }
        return option.displayName
    }

    private static func option(
        matchingLegacySelection selectedLegacyModelID: String,
        options: [AFMMLXModelSelectionOption]
    ) -> AFMMLXModelSelectionOption? {
        options.first {
            $0.displayName == selectedLegacyModelID
                || $0.id == selectedLegacyModelID
                || $0.id.split(separator: "/").last.map(String.init) == selectedLegacyModelID
        }
    }
}
