import AFMOpenAICompat
import MLXLMCommon

public enum AFMMLXToolCallPolicy {
    public static func isToolCallParserDisabled(_ parser: String?) -> Bool {
        MLXModelService.isToolCallParserDisabled(parser)
    }

    public static func normalizeToolCalls(
        _ toolCalls: [ToolCall],
        startIndex: Int = 0,
        paramNameMapping: [String: String] = [:],
        tools: [RequestTool]? = nil,
        fixToolArgs: Bool = false
    ) -> [ResponseToolCall] {
        MLXModelService.normalizeToolCalls(
            toolCalls,
            startIndex: startIndex,
            paramNameMapping: paramNameMapping,
            tools: tools,
            fixToolArgs: fixToolArgs
        )
    }

    public static func coerceArgumentTypes(
        _ toolCall: ResponseToolCall,
        tools: [RequestTool]?
    ) -> ResponseToolCall {
        MLXModelService.coerceArgumentTypes(toolCall, tools: tools)
    }

    public static func remapArgumentKeys(
        _ arguments: [String: any Sendable],
        toolName: String,
        tools: [RequestTool]?
    ) -> [String: any Sendable] {
        guard let tools, !tools.isEmpty else { return arguments }
        return MLXModelService.remapArgumentKeys(arguments, toolName: toolName, tools: tools)
    }

    public static func remapResponseToolCallArguments(
        _ toolCall: ResponseToolCall,
        tools: [RequestTool]?
    ) -> ResponseToolCall {
        MLXModelService.remapResponseToolCallArguments(toolCall, tools: tools)
    }
}
