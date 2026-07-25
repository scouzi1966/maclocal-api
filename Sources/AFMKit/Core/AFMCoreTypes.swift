import Foundation

public struct AFMProviderID: RawRepresentable, Hashable, Codable, Sendable,
    ExpressibleByStringLiteral, CustomStringConvertible
{
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(stringLiteral value: String) {
        self.init(rawValue: value)
    }

    public var description: String { rawValue }
}

public struct AFMModelID: RawRepresentable, Hashable, Codable, Sendable,
    ExpressibleByStringLiteral, CustomStringConvertible
{
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(stringLiteral value: String) {
        self.init(rawValue: value)
    }

    public var description: String { rawValue }
}

public struct AFMModelCapabilities: OptionSet, Hashable, Codable, Sendable {
    public let rawValue: UInt64

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    public static let text = Self(rawValue: 1 << 0)
    public static let vision = Self(rawValue: 1 << 1)
    public static let audioInput = Self(rawValue: 1 << 2)
    public static let audioOutput = Self(rawValue: 1 << 3)
    public static let reasoning = Self(rawValue: 1 << 4)
    public static let toolCalling = Self(rawValue: 1 << 5)
    public static let structuredOutput = Self(rawValue: 1 << 6)
    public static let streaming = Self(rawValue: 1 << 7)
    public static let embeddings = Self(rawValue: 1 << 8)
    public static let speculativeDecoding = Self(rawValue: 1 << 9)
    public static let prefixCaching = Self(rawValue: 1 << 10)
}

public enum AFMPrivacyBoundary: String, Codable, Hashable, Sendable {
    case device
    case privateCloud
    case providerCloud
    case configurable
    case unknown
}

public enum AFMModelAvailability: Hashable, Sendable {
    case available
    case loading(progress: Double?)
    case requiresConfiguration(keys: [String])
    case unavailable(reason: String)

    public var isAvailable: Bool {
        if case .available = self { return true }
        return false
    }
}

public struct AFMModelDescriptor: Hashable, Sendable {
    public var providerID: AFMProviderID
    public var modelID: AFMModelID
    public var displayName: String
    public var capabilities: AFMModelCapabilities
    public var contextWindow: Int?
    public var privacyBoundary: AFMPrivacyBoundary
    public var requiresNetwork: Bool?
    public var metadata: [String: AFMJSONValue]

    public init(
        providerID: AFMProviderID,
        modelID: AFMModelID,
        displayName: String,
        capabilities: AFMModelCapabilities,
        contextWindow: Int? = nil,
        privacyBoundary: AFMPrivacyBoundary = .unknown,
        requiresNetwork: Bool? = nil,
        metadata: [String: AFMJSONValue] = [:]
    ) {
        self.providerID = providerID
        self.modelID = modelID
        self.displayName = displayName
        self.capabilities = capabilities
        self.contextWindow = contextWindow
        self.privacyBoundary = privacyBoundary
        self.requiresNetwork = requiresNetwork
        self.metadata = metadata
    }
}

public struct AFMProviderDescriptor: Hashable, Sendable {
    public var id: AFMProviderID
    public var displayName: String
    public var privacyBoundary: AFMPrivacyBoundary
    public var configurationKeys: [String]
    public var metadata: [String: AFMJSONValue]

    public init(
        id: AFMProviderID,
        displayName: String,
        privacyBoundary: AFMPrivacyBoundary = .unknown,
        configurationKeys: [String] = [],
        metadata: [String: AFMJSONValue] = [:]
    ) {
        self.id = id
        self.displayName = displayName
        self.privacyBoundary = privacyBoundary
        self.configurationKeys = configurationKeys
        self.metadata = metadata
    }
}

public indirect enum AFMJSONValue: Hashable, Sendable {
    case null
    case bool(Bool)
    case integer(Int)
    case number(Double)
    case string(String)
    case array([AFMJSONValue])
    case object([String: AFMJSONValue])
}

public enum AFMMessageRole: String, Codable, Hashable, Sendable {
    case system
    case user
    case assistant
    case tool
}

public enum AFMContentPart: Hashable, Sendable {
    case text(String)
    case data(mimeType: String, value: Data)
    case reference(URL)
    case custom(type: String, payload: Data)
}

public struct AFMMessage: Hashable, Sendable {
    public var role: AFMMessageRole
    public var content: [AFMContentPart]
    public var name: String?
    public var toolCallID: String?
    public var toolCalls: [AFMToolCall]
    public var metadata: [String: AFMJSONValue]

    public init(
        role: AFMMessageRole,
        content: [AFMContentPart],
        name: String? = nil,
        toolCallID: String? = nil,
        toolCalls: [AFMToolCall] = [],
        metadata: [String: AFMJSONValue] = [:]
    ) {
        self.role = role
        self.content = content
        self.name = name
        self.toolCallID = toolCallID
        self.toolCalls = toolCalls
        self.metadata = metadata
    }

    public init(role: AFMMessageRole, text: String) {
        self.init(role: role, content: [.text(text)])
    }
}

public struct AFMToolDefinition: Hashable, Sendable {
    public var name: String
    public var description: String?
    public var inputSchema: AFMJSONValue

    public init(name: String, description: String? = nil, inputSchema: AFMJSONValue) {
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
    }
}

public enum AFMResponseConstraint: Hashable, Sendable {
    case jsonObject
    case jsonSchema(name: String?, schema: AFMJSONValue, strict: Bool)
    case grammar(String)
}

public struct AFMGenerationOptions: Hashable, Sendable {
    public var temperature: Double?
    public var maximumResponseTokens: Int?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var presencePenalty: Double?
    public var seed: Int?
    public var stopSequences: [String]
    public var responseConstraint: AFMResponseConstraint?

    public init(
        temperature: Double? = nil,
        maximumResponseTokens: Int? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        stopSequences: [String] = [],
        responseConstraint: AFMResponseConstraint? = nil
    ) {
        self.temperature = temperature
        self.maximumResponseTokens = maximumResponseTokens
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.stopSequences = stopSequences
        self.responseConstraint = responseConstraint
    }
}

public struct AFMRequest: Hashable, Sendable {
    public var messages: [AFMMessage]
    public var tools: [AFMToolDefinition]
    public var options: AFMGenerationOptions
    public var metadata: [String: AFMJSONValue]

    public init(
        messages: [AFMMessage],
        tools: [AFMToolDefinition] = [],
        options: AFMGenerationOptions = .init(),
        metadata: [String: AFMJSONValue] = [:]
    ) {
        self.messages = messages
        self.tools = tools
        self.options = options
        self.metadata = metadata
    }
}

public struct AFMToolCall: Hashable, Sendable {
    public var id: String
    public var name: String
    public var arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public struct AFMUsage: Hashable, Sendable {
    public var inputTokens: Int
    public var cachedInputTokens: Int
    public var outputTokens: Int
    public var reasoningTokens: Int

    public init(
        inputTokens: Int = 0,
        cachedInputTokens: Int = 0,
        outputTokens: Int = 0,
        reasoningTokens: Int = 0
    ) {
        self.inputTokens = inputTokens
        self.cachedInputTokens = cachedInputTokens
        self.outputTokens = outputTokens
        self.reasoningTokens = reasoningTokens
    }
}

public enum AFMFinishReason: String, Codable, Hashable, Sendable {
    case stop
    case length
    case toolCalls
    case cancelled
    case contentFilter
    case error
    case unknown
}

public struct AFMModelResponse: Hashable, Sendable {
    public var text: String
    public var reasoning: String?
    public var toolCalls: [AFMToolCall]
    public var usage: AFMUsage
    public var finishReason: AFMFinishReason
    public var metadata: [String: AFMJSONValue]

    public init(
        text: String = "",
        reasoning: String? = nil,
        toolCalls: [AFMToolCall] = [],
        usage: AFMUsage = .init(),
        finishReason: AFMFinishReason = .stop,
        metadata: [String: AFMJSONValue] = [:]
    ) {
        self.text = text
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.usage = usage
        self.finishReason = finishReason
        self.metadata = metadata
    }
}

public enum AFMTextUpdateAction: String, Codable, Hashable, Sendable {
    case append
    case replace
}

public enum AFMToolCallStage: Hashable, Sendable {
    case started
    case argumentsDelta(String)
    case completed
    case retracted
}

public enum AFMGenerationEvent: Hashable, Sendable {
    case responseText(action: AFMTextUpdateAction, text: String, tokenCount: Int)
    case reasoningText(action: AFMTextUpdateAction, text: String, tokenCount: Int)
    case toolCall(call: AFMToolCall, stage: AFMToolCallStage)
    case usage(AFMUsage)
    case metadata([String: AFMJSONValue])
    case custom(type: String, payload: Data)
    case completed(AFMFinishReason)
}

public enum AFMError: Error, Hashable, Sendable, LocalizedError {
    case providerNotRegistered(AFMProviderID)
    case providerAlreadyRegistered(AFMProviderID)
    case modelNotFound(provider: AFMProviderID, model: AFMModelID)
    case unavailable(String)
    case unsupportedCapability(String)
    case invalidRequest(String)
    case loadingFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .providerNotRegistered(let id):
            return "AFM provider '\(id)' is not registered."
        case .providerAlreadyRegistered(let id):
            return "AFM provider '\(id)' is already registered."
        case .modelNotFound(let provider, let model):
            return "AFM model '\(model)' was not found for provider '\(provider)'."
        case .unavailable(let reason):
            return "The AFM model is unavailable: \(reason)"
        case .unsupportedCapability(let capability):
            return "The AFM model does not support \(capability)."
        case .invalidRequest(let reason):
            return "The AFM request is invalid: \(reason)"
        case .loadingFailed(let reason):
            return "The AFM model failed to load: \(reason)"
        case .generationFailed(let reason):
            return "AFM generation failed: \(reason)"
        }
    }
}
