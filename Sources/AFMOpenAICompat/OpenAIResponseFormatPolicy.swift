import Foundation

/// Shared policy for OpenAI-compatible `response_format` handling.
///
/// Servers may expose defaults such as `--guided-json`; callers may override
/// them per request. The policy lives with the DTOs so AFMKit providers, CLIs,
/// and HTTP layers apply the same precedence and structured-output cleanup.
public enum OpenAIResponseFormatPolicy {
    private static let fencedStructuredOutputRegex = try! NSRegularExpression(
        pattern: #"^\s*```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)\s*```\s*$"#,
        options: []
    )

    public static func effectiveResponseFormat(
        requestFormat: ResponseFormat?,
        serverDefault: ResponseFormat?
    ) -> ResponseFormat? {
        requestFormat ?? serverDefault
    }

    public static func effectiveStrictJsonSchema(
        requestFormat: ResponseFormat?,
        serverDefault: ResponseFormat?
    ) -> ResponseJsonSchema? {
        guard let format = effectiveResponseFormat(
            requestFormat: requestFormat,
            serverDefault: serverDefault
        ),
              format.type == "json_schema",
              let jsonSchema = format.jsonSchema,
              jsonSchema.strict == true else {
            return nil
        }
        return jsonSchema
    }

    public static func requiresStructuredOutputSanitization(_ responseFormat: ResponseFormat?) -> Bool {
        guard let type = responseFormat?.type else { return false }
        return type == "json_schema" || type == "json_object"
    }

    public static func sanitizeStructuredOutput(
        _ text: String,
        responseFormat: ResponseFormat?
    ) -> String {
        guard requiresStructuredOutputSanitization(responseFormat) else {
            return text
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let match = fencedStructuredOutputRegex.firstMatch(
            in: trimmed,
            range: NSRange(trimmed.startIndex..., in: trimmed)
        ),
              let contentRange = Range(match.range(at: 1), in: trimmed) else {
            return trimmed
        }

        return String(trimmed[contentRange]).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
