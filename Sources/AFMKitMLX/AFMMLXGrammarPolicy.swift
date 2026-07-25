import AFMOpenAICompat

public enum AFMMLXGrammarPolicy {
    public static func shouldDowngradeGrammarConstraints(
        responseFormat: ResponseFormat?,
        tools: [RequestTool]?,
        supportsStrictToolGrammar: Bool,
        enableGrammarConstraints: Bool
    ) -> Bool {
        MLXModelService.shouldDowngradeGrammarConstraints(
            responseFormat: responseFormat,
            tools: tools,
            supportsStrictToolGrammar: supportsStrictToolGrammar,
            enableGrammarConstraints: enableGrammarConstraints
        )
    }
}
