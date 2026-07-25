import Foundation
import AFMOpenAICompat
import MLXLMCommon

public struct AFMMLXServingConfiguration: Sendable, Equatable {
    public var toolCallParser: String?
    public var supportsStrictToolGrammar: Bool
    public var thinkStartTag: String?
    public var thinkEndTag: String?
    public var harmonyChannels: Bool
    public var structuralStripTags: [String]
    public var fixToolArguments: Bool
    public var grammarConstraintsEnabled: Bool

    public init(
        toolCallParser: String? = nil,
        supportsStrictToolGrammar: Bool = false,
        thinkStartTag: String? = nil,
        thinkEndTag: String? = nil,
        harmonyChannels: Bool = false,
        structuralStripTags: [String] = [],
        fixToolArguments: Bool = false,
        grammarConstraintsEnabled: Bool = false
    ) {
        self.toolCallParser = toolCallParser
        self.supportsStrictToolGrammar = supportsStrictToolGrammar
        self.thinkStartTag = thinkStartTag
        self.thinkEndTag = thinkEndTag
        self.harmonyChannels = harmonyChannels
        self.structuralStripTags = structuralStripTags
        self.fixToolArguments = fixToolArguments
        self.grammarConstraintsEnabled = grammarConstraintsEnabled
    }
}

public protocol AFMMLXServingConfigurationProviding: Sendable {
    var servingConfiguration: AFMMLXServingConfiguration { get }

    func normalizeModel(_ raw: String) -> String
    func resolvedToolCallParser(logBypass: Bool) -> String?
}

public extension AFMMLXServingConfigurationProviding {
    var toolCallParser: String? { servingConfiguration.toolCallParser }
    var supportsStrictToolGrammar: Bool { servingConfiguration.supportsStrictToolGrammar }
    var thinkStartTag: String? { servingConfiguration.thinkStartTag }
    var thinkEndTag: String? { servingConfiguration.thinkEndTag }
    var harmonyChannels: Bool { servingConfiguration.harmonyChannels }
    var structuralStripTags: [String] { servingConfiguration.structuralStripTags }
    var fixToolArgs: Bool { servingConfiguration.fixToolArguments }
    var enableGrammarConstraints: Bool { servingConfiguration.grammarConstraintsEnabled }

    func shouldDowngradeGrammarConstraints(
        responseFormat: ResponseFormat?,
        tools: [RequestTool]?
    ) -> Bool {
        AFMMLXGrammarPolicy.shouldDowngradeGrammarConstraints(
            responseFormat: responseFormat,
            tools: tools,
            supportsStrictToolGrammar: supportsStrictToolGrammar,
            enableGrammarConstraints: enableGrammarConstraints
        )
    }

    func isToolCallParserDisabled(_ parser: String?) -> Bool {
        AFMMLXToolCallPolicy.isToolCallParserDisabled(parser)
    }

    func normalizeToolCalls(
        _ toolCalls: [ToolCall],
        startIndex: Int = 0,
        paramNameMapping: [String: String] = [:],
        tools: [RequestTool]? = nil
    ) -> [ResponseToolCall] {
        AFMMLXToolCallPolicy.normalizeToolCalls(
            toolCalls,
            startIndex: startIndex,
            paramNameMapping: paramNameMapping,
            tools: tools,
            fixToolArgs: fixToolArgs
        )
    }

    func coerceToolCallArguments(
        _ toolCall: ResponseToolCall,
        tools: [RequestTool]?
    ) -> ResponseToolCall {
        AFMMLXToolCallPolicy.coerceArgumentTypes(toolCall, tools: tools)
    }

    func remapArgumentKeys(
        _ arguments: [String: any Sendable],
        toolName: String,
        tools: [RequestTool]?
    ) -> [String: any Sendable] {
        guard fixToolArgs else { return arguments }
        return AFMMLXToolCallPolicy.remapArgumentKeys(arguments, toolName: toolName, tools: tools)
    }

    func remapToolCallArguments(
        _ toolCall: ResponseToolCall,
        tools: [RequestTool]?
    ) -> ResponseToolCall {
        guard fixToolArgs else { return toolCall }
        return AFMMLXToolCallPolicy.remapResponseToolCallArguments(toolCall, tools: tools)
    }
}
