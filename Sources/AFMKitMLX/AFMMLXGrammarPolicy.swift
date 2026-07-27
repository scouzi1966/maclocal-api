import AFMOpenAICompat

public enum AFMMLXGrammarPolicy {
    /// Check whether any tool in the request has `strict: true`.
    public static func hasStrictTools(_ tools: [RequestTool]?) -> Bool {
        tools?.contains { $0.function.strict == true } ?? false
    }

    /// Check whether a response_format has json_schema with strict: true.
    public static func hasStrictSchema(_ responseFormat: ResponseFormat?) -> Bool {
        responseFormat?.type == "json_schema" && responseFormat?.jsonSchema?.strict == true
    }

    public static func shouldDowngradeGrammarConstraints(
        responseFormat: ResponseFormat?,
        tools: [RequestTool]?,
        supportsStrictToolGrammar: Bool,
        enableGrammarConstraints: Bool
    ) -> Bool {
        let strictSchema = hasStrictSchema(responseFormat)
        let strictTools = hasStrictTools(tools) && supportsStrictToolGrammar
        return (strictSchema || strictTools) && !enableGrammarConstraints
    }
}
