#if canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, *)
public enum AFMFoundationGeneratedContentReader {
    public static func property(_ name: String, in content: GeneratedContent) -> GeneratedContent? {
        guard case .structure(let properties, _) = content.kind else { return nil }
        return properties[name]
    }

    public static func string(_ name: String, in content: GeneratedContent) -> String? {
        guard let value = property(name, in: content), case .string(let string) = value.kind else {
            return nil
        }
        return string
    }

    public static func number(_ name: String, in content: GeneratedContent) -> Double? {
        guard let value = property(name, in: content), case .number(let number) = value.kind else {
            return nil
        }
        return number
    }

    public static func strings(_ name: String, in content: GeneratedContent) -> [String] {
        guard let value = property(name, in: content), case .array(let elements) = value.kind else {
            return []
        }
        return elements.compactMap { element in
            guard case .string(let string) = element.kind else { return nil }
            return string
        }
    }

    public static func joinedSections(
        _ sections: [String],
        fallback content: GeneratedContent,
        separator: String = "\n\n"
    ) -> String {
        sections.isEmpty ? content.jsonString : sections.joined(separator: separator)
    }
}
#endif
