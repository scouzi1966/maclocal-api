import Foundation
import AFMKitCore
import AFMOpenAICompat

public struct AFMMLXRuntimeConfiguration: Sendable {
    public var kvBits: Int?
    public var enablePrefixCaching: Bool
    public var mtpEnabled: Bool
    public var mtpDepth: Int
    public var eagle3DrafterPath: String?
    public var maxConcurrent: Int
    public var toolCallParser: String?
    public var enableGrammarConstraints: Bool
    public var prefillStepSize: Int?
    public var kvEvictionPolicy: String
    public var fixToolArguments: Bool
    public var forceVLM: Bool
    public var cacheProfilePath: String?
    public var trace: Bool
    public var gpuCapturePath: String?
    public var gpuTraceDuration: Int?
    public var gpuProfile: Bool
    public var gpuProfileBandwidth: Bool
    public var defaultChatTemplateKwargs: [String: AFMJSONValue]?
    public var defaultGuidedJsonSchema: ResponseFormat?

    public init(
        kvBits: Int? = nil,
        enablePrefixCaching: Bool = true,
        mtpEnabled: Bool = false,
        mtpDepth: Int = 3,
        eagle3DrafterPath: String? = nil,
        maxConcurrent: Int = 0,
        toolCallParser: String? = nil,
        enableGrammarConstraints: Bool = false,
        prefillStepSize: Int? = nil,
        kvEvictionPolicy: String = "none",
        fixToolArguments: Bool = false,
        forceVLM: Bool = false,
        cacheProfilePath: String? = nil,
        trace: Bool = false,
        gpuCapturePath: String? = nil,
        gpuTraceDuration: Int? = nil,
        gpuProfile: Bool = false,
        gpuProfileBandwidth: Bool = false,
        defaultChatTemplateKwargs: [String: AFMJSONValue]? = nil,
        defaultGuidedJsonSchema: ResponseFormat? = nil
    ) {
        self.kvBits = kvBits
        self.enablePrefixCaching = enablePrefixCaching
        self.mtpEnabled = mtpEnabled
        self.mtpDepth = mtpDepth
        self.eagle3DrafterPath = eagle3DrafterPath
        self.maxConcurrent = max(0, maxConcurrent)
        self.toolCallParser = toolCallParser
        self.enableGrammarConstraints = enableGrammarConstraints
        self.prefillStepSize = prefillStepSize
        self.kvEvictionPolicy = kvEvictionPolicy
        self.fixToolArguments = fixToolArguments
        self.forceVLM = forceVLM
        self.cacheProfilePath = cacheProfilePath
        self.trace = trace
        self.gpuCapturePath = gpuCapturePath
        self.gpuTraceDuration = gpuTraceDuration
        self.gpuProfile = gpuProfile
        self.gpuProfileBandwidth = gpuProfileBandwidth
        self.defaultChatTemplateKwargs = defaultChatTemplateKwargs
        self.defaultGuidedJsonSchema = defaultGuidedJsonSchema
    }

    public init(providerConfiguration configuration: AFMProviderConfiguration) {
        self.init()
        apply(configuration)
    }

    public mutating func apply(_ configuration: AFMProviderConfiguration) {
        if let value = configuration.integer("kvBits") {
            kvBits = value
        }
        if let value = configuration.bool("enablePrefixCaching") {
            enablePrefixCaching = value
        }
        if let value = configuration.bool("mtpEnabled") {
            mtpEnabled = value
        }
        if let value = configuration.integer("mtpDepth") {
            mtpDepth = value
        }
        if let value = configuration.string("eagle3DrafterPath") {
            eagle3DrafterPath = value
        }
        if let value = configuration.integer("maxConcurrent") {
            maxConcurrent = max(0, value)
        }
        if let value = configuration.string("toolCallParser") {
            toolCallParser = value
        }
        if let value = configuration.bool("enableGrammarConstraints") {
            enableGrammarConstraints = value
        }
        if let value = configuration.integer("prefillStepSize") {
            prefillStepSize = value
        }
        if let value = configuration.string("kvEvictionPolicy") {
            kvEvictionPolicy = value
        }
        if let value = configuration.bool("fixToolArguments") {
            fixToolArguments = value
        }
        if let value = configuration.bool("forceVLM") {
            forceVLM = value
        }
        if let value = configuration.string("cacheProfilePath") {
            cacheProfilePath = value
        }
        if let value = configuration.bool("trace") {
            trace = value
        }
        if let value = configuration.string("gpuCapturePath") {
            gpuCapturePath = value
        }
        if let value = configuration.integer("gpuTraceDuration") {
            gpuTraceDuration = value
        }
        if let value = configuration.bool("gpuProfile") {
            gpuProfile = value
        }
        if let value = configuration.bool("gpuProfileBandwidth") {
            gpuProfileBandwidth = value
        }
    }

    public func apply(to service: MLXModelService) {
        service.kvBits = kvBits
        service.enablePrefixCaching = enablePrefixCaching
        service.mtpEnabled = mtpEnabled
        service.mtpDepth = mtpDepth
        service.eagle3DrafterPath = eagle3DrafterPath
        service.maxConcurrent = maxConcurrent >= 2 ? maxConcurrent : 0
        service.toolCallParser = toolCallParser
        service.enableGrammarConstraints = enableGrammarConstraints
        service.prefillStepSize = prefillStepSize ?? service.prefillStepSize
        service.kvEvictionPolicy = kvEvictionPolicy
        service.fixToolArgs = fixToolArguments
        service.forceVLM = forceVLM
        service.cacheProfilePath = cacheProfilePath
        service.trace = trace
        service.gpuCapturePath = gpuCapturePath
        service.gpuTraceDuration = gpuTraceDuration
        service.gpuProfile = gpuProfile
        service.gpuProfileBandwidth = gpuProfileBandwidth
        service.defaultChatTemplateKwargs =
            defaultChatTemplateKwargs?.mapValues(Self.anyValue)
        service.defaultGuidedJsonSchema = defaultGuidedJsonSchema
    }

    private static func anyValue(_ value: AFMJSONValue) -> Any {
        switch value {
        case .null:
            return NSNull()
        case .bool(let value):
            return value
        case .integer(let value):
            return value
        case .number(let value):
            return value
        case .string(let value):
            return value
        case .array(let values):
            return values.map(anyValue)
        case .object(let values):
            return values.mapValues(anyValue)
        }
    }
}

public final class AFMMLXRuntime: @unchecked Sendable {
    public let modelID: String
    public let descriptor: AFMModelDescriptor
    public let service: MLXModelService

    private let configuration: AFMMLXRuntimeConfiguration
    let initializesSchedulerOnLoad: Bool

    public init(
        modelID: String,
        configuration: AFMMLXRuntimeConfiguration = .init(),
        resolver: MLXCacheResolver = .init(),
        service providedService: MLXModelService? = nil
    ) {
        let service = providedService ?? MLXModelService(resolver: resolver)
        configuration.apply(to: service)

        self.configuration = configuration
        self.initializesSchedulerOnLoad = true
        self.service = service
        self.modelID = service.normalizeModel(modelID)
        self.descriptor = AFMMLXModelDescriptor.describe(
            modelID: self.modelID,
            resolver: resolver
        )
    }

    /// Attach AFMKit lifecycle and model metadata to a service that the host has
    /// already configured. Request adapters must not reapply provider defaults.
    public init(
        modelID: String,
        attaching service: MLXModelService,
        resolver: MLXCacheResolver = .init()
    ) {
        self.configuration = AFMMLXRuntimeConfiguration(
            enablePrefixCaching: service.enablePrefixCaching,
            maxConcurrent: service.maxConcurrent,
            enableGrammarConstraints: service.enableGrammarConstraints
        )
        self.initializesSchedulerOnLoad = false
        self.service = service
        self.modelID = service.normalizeModel(modelID)
        self.descriptor = AFMMLXModelDescriptor.describe(
            modelID: self.modelID,
            resolver: resolver
        )
    }

    public convenience init(
        modelID: String,
        providerConfiguration: AFMProviderConfiguration,
        resolver: MLXCacheResolver = .init(),
        service providedService: MLXModelService? = nil
    ) {
        self.init(
            modelID: modelID,
            configuration: AFMMLXRuntimeConfiguration(
                providerConfiguration: providerConfiguration
            ),
            resolver: resolver,
            service: providedService
        )
    }

    public func load(
        progress: (@Sendable (Progress) -> Void)? = nil,
        stage: (@Sendable (MLXLoadStage) -> Void)? = nil
    ) async throws -> AFMModelDescriptor {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = try await service.ensureLoaded(
            model: modelID,
            progress: progress,
            stage: stage
        )
        if initializesSchedulerOnLoad && configuration.maxConcurrent >= 2 {
            try await service.initScheduler()
        }
        return descriptor
    }

    public func prewarm(
        messages: [Message] = [Message(role: "user", content: "warmup")],
        maxTokens: Int = 4
    ) async throws {
        _ = try await service.generate(
            model: modelID,
            messages: messages,
            temperature: 0,
            maxTokens: maxTokens,
            topP: nil,
            repetitionPenalty: nil
        )
    }

    public func unload(
        verbose: Bool = false,
        timeoutSeconds: TimeInterval = 30
    ) async {
        await service.shutdownAndReleaseResources(
            verbose: verbose,
            timeoutSeconds: timeoutSeconds
        )
    }
}

extension AFMProviderConfiguration {
    func bool(_ key: String) -> Bool? {
        guard case .bool(let value) = values[key] else { return nil }
        return value
    }

    func integer(_ key: String) -> Int? {
        guard case .integer(let value) = values[key] else { return nil }
        return value
    }

    func string(_ key: String) -> String? {
        guard case .string(let value) = values[key] else { return nil }
        return value
    }
}
