import Foundation

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

