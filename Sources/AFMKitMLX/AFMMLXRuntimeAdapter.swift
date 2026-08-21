import Foundation
import AFMKitCore
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXVLM
import Tokenizers

public final class AFMMLXDFlash2Runtime: @unchecked Sendable {
    public let draft: DFlash2DraftModel
    public let preflight: AFMMLXDFlash2Preflight

    init(draft: DFlash2DraftModel, preflight: AFMMLXDFlash2Preflight) {
        self.draft = draft
        self.preflight = preflight
    }
}

public enum AFMMLXSpeculativeRuntime {
    case none
    case mtpLLM(Qwen3_5MoEMTPGenerator)
    case mtpVLM(MTPGenerator)
    case eagle3(Gemma4Eagle3Drafter)
    case dflash2(AFMMLXDFlash2Runtime)

    public var kind: AFMMLXSpeculativeRuntimeKind {
        switch self {
        case .none: return .none
        case .mtpLLM, .mtpVLM: return .mtp
        case .eagle3: return .eagle3
        case .dflash2: return .dflash2
        }
    }
}

extension AFMMLXSpeculativeRuntime: @unchecked Sendable {}

public struct AFMMLXRuntimeAdapter: Sendable {
    public typealias RuntimeEvent = AFMMLXRuntimeEvent
    public nonisolated static let imageProcessingSize = AFMMLXRuntimePolicy.defaultImageProcessingSize

    public struct LoadedContainer {
        public let container: ModelContainer
        public let isVision: Bool
        public let loadTime: Double

        public init(
            container: ModelContainer,
            isVision: Bool,
            loadTime: Double
        ) {
            self.container = container
            self.isVision = isVision
            self.loadTime = loadTime
        }
    }

    public init() {}

    @MainActor public func makeUserInput(
        chat: [Chat.Message],
        additionalContext: [String: any Sendable]?,
        modelType: String? = nil
    ) throws -> UserInput {
        if modelType == "deepseek_v4" {
            let prompt = try DeepseekV4ChatEncoder.renderOpenAIChat(
                messages: Self.openAIMessages(from: chat),
                tools: nil,
                additionalContext: additionalContext,
                addGenerationPrompt: true
            )
            return UserInput(
                prompt: .text(prompt),
                processing: .init(
                    resize: .init(
                        width: Self.imageProcessingSize,
                        height: Self.imageProcessingSize
                    )
                ),
                additionalContext: additionalContext
            )
        }

        return UserInput(
            chat: chat,
            processing: .init(
                resize: .init(
                    width: Self.imageProcessingSize,
                    height: Self.imageProcessingSize
                )
            ),
            additionalContext: additionalContext
        )
    }

    private static func openAIMessages(
        from chat: [Chat.Message]
    ) -> [[String: any Sendable]] {
        chat.map { message in
            var raw: [String: any Sendable] = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if let name = message.name {
                raw["name"] = name
            }
            if let toolCalls = message.toolCalls {
                raw["tool_calls"] = toolCalls.map { Self.sendableJSONDictionary($0) }
            }
            if let toolResponses = message.toolResponses {
                raw["tool_responses"] = toolResponses.map { Self.sendableJSONDictionary($0) }
            }
            return raw
        }
    }

    private static func sendableJSONDictionary(
        _ dictionary: [String: Any]
    ) -> [String: any Sendable] {
        var result: [String: any Sendable] = [:]
        for (key, value) in dictionary {
            if let sendable = sendableJSONValue(value) {
                result[key] = sendable
            }
        }
        return result
    }

    private static func sendableJSONValue(_ value: Any) -> (any Sendable)? {
        switch value {
        case let value as String:
            return value
        case let value as Bool:
            return value
        case let value as Int:
            return value
        case let value as Double:
            return value
        case let value as Float:
            return Double(value)
        case let value as [Any]:
            return value.compactMap(sendableJSONValue)
        case let value as [String: Any]:
            return sendableJSONDictionary(value)
        default:
            return nil
        }
    }

    @MainActor public func runGeneration(
        container: ModelContainer,
        chat: [Chat.Message],
        additionalContext: [String: any Sendable]?,
        modelType: String? = nil,
        parameters parameterRequest: AFMMLXGenerationParameterRequest,
        onEvent: (RuntimeEvent) async throws -> Bool
    ) async throws {
        let userInput = try makeUserInput(
            chat: chat,
            additionalContext: additionalContext,
            modelType: modelType
        )
        let input = try await container.prepare(input: userInput)
        let parameters = AFMMLXGenerationParameterFactory.make(parameterRequest)
        let stream = try await container.generate(input: input, parameters: parameters)

        for await item in stream {
            let shouldContinue: Bool
            switch item {
            case .chunk(let string):
                shouldContinue = try await onEvent(.chunk(string))
            case .info(let info):
                shouldContinue = try await onEvent(
                    .info(
                        AFMMLXRuntimeInfo(
                            promptTime: info.promptTime,
                            tokensPerSecond: info.tokensPerSecond
                        )
                    )
                )
            case .toolCall:
                shouldContinue = try await onEvent(.toolCall)
            case .tokenLogprobs:
                shouldContinue = try await onEvent(.tokenLogprobs)
            }

            if !shouldContinue {
                break
            }
        }
    }

    public func runReducedGeneration(
        container: ModelContainer,
        userInput: consuming sending UserInput,
        parameters parameterRequest: AFMMLXGenerationParameterRequest,
        hasReasoningOutput: Bool,
        updateInterval: TimeInterval = AFMGenerationLoopPolicy.defaultDirectUIUpdateInterval,
        stopRequested: @escaping @Sendable () -> Bool,
        onProgress: @escaping @MainActor @Sendable (AFMMLXReducedGenerationProgress) -> Void
    ) async throws -> AFMMLXReducedGenerationResult {
        let (progressStream, progressContinuation) = AsyncStream.makeStream(
            of: AFMMLXReducedGenerationProgress.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        let progressConsumer = Task { @MainActor in
            for await progress in progressStream {
                onProgress(progress)
            }
        }

        let input = try await container.prepare(input: userInput)
        let parameters = AFMMLXGenerationParameterFactory.make(parameterRequest)
        let stream = try await container.generate(input: input, parameters: parameters)

        var reducer = AFMReasoningStreamReducer(hasReasoningOutput: hasReasoningOutput)
        var stopPollState = AFMStopPollState()
        var runtimeInfo: AFMMLXRuntimeInfo?
        var lastUpdateTime = CFAbsoluteTimeGetCurrent()
        var pendingReasoningContent: String?
        var accumulatedReasoningContent: String?
        var pendingReasoningState = false
        var pendingReasoningDuration: TimeInterval?

        func mergeReasoningUpdate(_ update: AFMReasoningPendingUpdate?) {
            guard let update else { return }
            if let content = update.content {
                pendingReasoningContent = (pendingReasoningContent ?? "") + content
                accumulatedReasoningContent = (accumulatedReasoningContent ?? "") + content
            }
            pendingReasoningState = update.isReasoning
            pendingReasoningDuration = update.duration
        }

        func drainMergedReasoningUpdate() -> AFMReasoningPendingUpdate? {
            guard pendingReasoningContent != nil || pendingReasoningDuration != nil else {
                return nil
            }
            let update = AFMReasoningPendingUpdate(
                content: accumulatedReasoningContent,
                isReasoning: pendingReasoningState,
                duration: pendingReasoningDuration
            )
            pendingReasoningContent = nil
            pendingReasoningDuration = nil
            return update
        }

        for await item in stream {
            switch item {
            case .chunk(let string):
                if stopPollState.observeToken(stopRequested: stopRequested()) {
                    break
                }

                let chunkUpdate = reducer.processChunk(string)
                mergeReasoningUpdate(chunkUpdate.reasoningUpdate)

                let now = CFAbsoluteTimeGetCurrent()
                if now - lastUpdateTime >= updateInterval {
                    mergeReasoningUpdate(reducer.drainReasoningUpdate())
                    let progress = AFMMLXReducedGenerationProgress(
                        accumulatedText: reducer.accumulatedText,
                        reasoningUpdate: drainMergedReasoningUpdate()
                    )
                    progressContinuation.yield(progress)
                    lastUpdateTime = now
                }

            case .info(let info):
                runtimeInfo = AFMMLXRuntimeInfo(
                    promptTime: info.promptTime,
                    tokensPerSecond: info.tokensPerSecond
                )

            case .toolCall, .tokenLogprobs:
                break
            }

            if stopPollState.shouldStop {
                break
            }
        }

        mergeReasoningUpdate(reducer.drainReasoningUpdate())
        let finalReasoningUpdate = drainMergedReasoningUpdate()
        progressContinuation.yield(
            AFMMLXReducedGenerationProgress(
                accumulatedText: reducer.accumulatedText,
                reasoningUpdate: finalReasoningUpdate
            )
        )
        progressContinuation.finish()
        await progressConsumer.value

        return AFMMLXReducedGenerationResult(
            finalState: reducer.finalState(),
            finalReasoningUpdate: finalReasoningUpdate,
            tokenCount: reducer.tokenCount,
            stopped: stopPollState.shouldStop,
            runtimeInfo: runtimeInfo
        )
    }

    @MainActor public func loadBenchmarkContainer(
        modelPath: String,
        logger: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> LoadedContainer {
        let modelURL = URL(fileURLWithPath: modelPath)
        let configuration = ModelConfiguration(directory: modelURL)
        let isVision = Self.pathSuggestsVisionModel(modelPath)
        logger("Model type: \(isVision ? "VLM" : "LLM")")
        logger("Loading model...")

        let loadStart = CFAbsoluteTimeGetCurrent()
        let container: ModelContainer
        if isVision {
            container = try await VLMModelFactory.shared.loadContainer(configuration: configuration)
        } else {
            container = try await LLMModelFactory.shared.loadContainer(configuration: configuration)
        }
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        logger("Model loaded in \(String(format: "%.2f", loadTime))s")
        return LoadedContainer(container: container, isVision: isVision, loadTime: loadTime)
    }

    @MainActor public func loadAppContainer(
        configuration: ModelConfiguration,
        forceVisionFactory: Bool,
        configIsVision: Bool,
        cachedContainer: (Bool) -> ModelContainer?,
        cacheContainer: (ModelContainer, Bool) -> Void,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws -> LoadedContainer {
        func load(with factory: ModelFactory, isVision: Bool) async throws -> LoadedContainer {
            let loadStart = CFAbsoluteTimeGetCurrent()
            let container = try await factory.loadContainer(configuration: configuration) { progress in
                onProgress(progress.fractionCompleted)
            }
            cacheContainer(container, isVision)
            return LoadedContainer(
                container: container,
                isVision: isVision,
                loadTime: CFAbsoluteTimeGetCurrent() - loadStart
            )
        }

        if forceVisionFactory, let cached = cachedContainer(true) {
            return LoadedContainer(container: cached, isVision: true, loadTime: 0)
        }

        if !forceVisionFactory, let cached = cachedContainer(false) {
            return LoadedContainer(container: cached, isVision: false, loadTime: 0)
        }

        if forceVisionFactory {
            return try await load(with: VLMModelFactory.shared, isVision: true)
        }

        do {
            return try await load(with: LLMModelFactory.shared, isVision: false)
        } catch {
            if configIsVision {
                if let cached = cachedContainer(true) {
                    return LoadedContainer(container: cached, isVision: true, loadTime: 0)
                }
                return try await load(with: VLMModelFactory.shared, isVision: true)
            }
            throw error
        }
    }

    @MainActor public func runPromptStreamBenchmark(
        container: ModelContainer,
        prompt: String,
        parameters parameterRequest: AFMMLXGenerationParameterRequest,
        logger: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> Double {
        try await container.perform { context in
            let userInput = UserInput(prompt: prompt)
            let input = try await context.processor.prepare(input: userInput)
            logger("Prompt tokens: \(input.text.tokens.size)")

            let parameters = AFMMLXGenerationParameterFactory.make(parameterRequest)
            var completionInfo: GenerateCompletionInfo?
            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: context
            )

            for await generation in stream {
                switch generation {
                case .chunk:
                    break
                case .info(let info):
                    completionInfo = info
                default:
                    break
                }
            }

            if let info = completionInfo {
                logger("Generation: \(info.generationTokenCount) tokens, \(String(format: "%.3f", info.tokensPerSecond)) tok/s")
                return info.tokensPerSecond
            }
            return 0
        }
    }

    @MainActor public func runPromptTokenIteratorBenchmark(
        container: ModelContainer,
        prompt: String,
        parameters parameterRequest: AFMMLXGenerationParameterRequest,
        logger: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> Double {
        try await container.perform { context in
            let userInput = UserInput(prompt: prompt)
            let input = try await context.processor.prepare(input: userInput)
            logger("Prompt tokens: \(input.text.tokens.size)")

            let parameters = AFMMLXGenerationParameterFactory.make(parameterRequest)
            let iterator = try TokenIterator(
                input: input,
                model: context.model,
                parameters: parameters
            )

            var eosTokenIds = context.configuration.eosTokenIds
            if let tokenizerEos = context.tokenizer.eosTokenId {
                eosTokenIds.insert(tokenizerEos)
            }

            var tokenCount = 0
            let startTime = CFAbsoluteTimeGetCurrent()
            for token in iterator {
                if token == context.tokenizer.unknownTokenId || eosTokenIds.contains(token) {
                    break
                }
                tokenCount += 1
            }

            let generateTime = CFAbsoluteTimeGetCurrent() - startTime
            let tokensPerSecond = Double(tokenCount) / generateTime
            MLX.Stream().synchronize()
            logger("Generation: \(tokenCount) tokens, \(String(format: "%.3f", tokensPerSecond)) tok/s")
            return tokensPerSecond
        }
    }

    @MainActor public func isDenseGemma4Verifier(container: ModelContainer) async -> Bool {
        await container.perform { context in
            context.model is Gemma4Model
        }
    }

    @MainActor public func makeMTPRuntime(
        container: ModelContainer,
        sidecarPath: String
    ) async throws -> AFMMLXSpeculativeRuntime? {
        try await container.perform { context -> AFMMLXSpeculativeRuntime? in
            if let qwen = context.model as? Qwen3_5MoEModel {
                let head = try qwen.loadMTPHead(sidecarPath: sidecarPath)
                return .mtpLLM(Qwen3_5MoEMTPGenerator(model: qwen, head: head, depth: 1))
            }
            if let qwen = context.model as? Qwen3_5MoEVL {
                let head = try qwen.loadMTPHead(sidecarPath: sidecarPath)
                return .mtpVLM(MTPGenerator(model: qwen, head: head, depth: 3))
            }
            return nil
        }
    }

    public nonisolated func makeEagle3Runtime(drafterDirectory: URL) throws -> AFMMLXSpeculativeRuntime {
        let drafter = try Gemma4Eagle3Drafter.load(directory: drafterDirectory.path)
        return .eagle3(drafter)
    }

    public func makeDFlash2Runtime(
        container: ModelContainer,
        targetDirectory: URL,
        drafterDirectory: URL,
        blockSize: Int
    ) async throws -> AFMMLXSpeculativeRuntime {
        let preflight = try AFMMLXDFlash2PreflightValidator.validate(
            targetDirectory: targetDirectory,
            drafterDirectory: drafterDirectory,
            requestedBlockSize: blockSize)
        let draft = try DFlash2DraftModel.load(directory: drafterDirectory.path)
        let targetCompatible = try await container.perform { context in
            let loadedTargetData = try Data(contentsOf: context.configuration
                .modelDirectory()
                .appendingPathComponent("config.json"))
            guard loadedTargetData == preflight.targetConfigurationData else {
                throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                    "validated target metadata does not match the loaded model container")
            }
            guard let target = context.model as? any DFlash2Target else { return false }
            _ = try DFlash2Generator(
                target: target,
                draft: draft,
                blockSize: preflight.blockSize)
            return true
        }
        guard targetCompatible else {
            throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                "loaded target runtime does not expose DFlash2 hidden-state and rollback hooks")
        }
        return .dflash2(AFMMLXDFlash2Runtime(draft: draft, preflight: preflight))
    }

    @MainActor public func runSpeculativeGeneration(
        container: ModelContainer,
        userInput: UserInput,
        runtime: AFMMLXSpeculativeRuntime,
        maxTokens: Int,
        shouldStop: @escaping @Sendable () -> Bool,
        onChunk: @escaping @Sendable (String) -> Void,
        onTelemetry: @escaping @Sendable (AFMMLXSpeculativeTelemetry) -> Void = { _ in }
    ) async throws -> Int {
        try await container.perform { context -> Int in
            let input = try await context.processor.prepare(input: userInput)
            guard !Self.isMultimodalInput(input) else { return 0 }
            let promptIds = Self.extractTokenArray(input)
            guard !promptIds.isEmpty else { return 0 }

            let eos = Self.completeEOSTokenIDs(
                configuration: context.configuration,
                tokenizer: context.tokenizer)
            var allTokens: [Int] = []
            var previousText = ""

            let emit: (Int) -> Bool = { token in
                if Task.isCancelled || shouldStop() {
                    return false
                }
                if eos.contains(token) {
                    return false
                }

                allTokens.append(token)
                let fullText = context.tokenizer.decode(tokens: allTokens)
                if fullText.count > previousText.count {
                    onChunk(String(fullText.dropFirst(previousText.count)))
                    previousText = fullText
                }
                return true
            }

            switch runtime {
            case .mtpLLM(let generator):
                _ = generator.generate(promptIds: promptIds, maxTokens: maxTokens, eosIds: eos, onToken: emit)
            case .mtpVLM(let generator):
                _ = generator.generate(promptIds: promptIds, maxTokens: maxTokens, eosIds: eos, onToken: emit)
            case .eagle3(let drafter):
                guard let model = context.model as? Gemma4Model else { return 0 }
                let generator = Gemma4Eagle3Generator(drafter: drafter)
                _ = generator.generateSpeculative(
                    model: model,
                    promptIds: promptIds,
                    maxTokens: maxTokens,
                    eosIds: eos,
                    blockSize: 2,
                    onToken: emit
                )
            case .dflash2(let runtime):
                guard let target = context.model as? any DFlash2Target else { return 0 }
                let generator = try DFlash2Generator(
                    target: target,
                    draft: runtime.draft,
                    blockSize: runtime.preflight.blockSize)
                let result = try generator.generate(
                    promptIDs: promptIds,
                    maxTokens: maxTokens,
                    stopTokenIDs: eos,
                    shouldStop: { Task.isCancelled || shouldStop() },
                    onToken: emit)
                let stats = result.statistics
                onTelemetry(AFMMLXSpeculativeTelemetry(
                    strategy: "dflash2",
                    draftedTokens: stats.draftedTokens,
                    acceptedDraftTokens: stats.acceptedDraftTokens,
                    emittedTokens: stats.emittedTokens,
                    verificationCycles: stats.verificationCycles,
                    draftTime: stats.draftSeconds,
                    verificationTime: stats.verificationSeconds,
                    rollbackTime: stats.rollbackSeconds))
            case .none:
                return 0
            }

            try Task.checkCancellation()
            if shouldStop() {
                throw CancellationError()
            }
            return allTokens.count
        }
    }

    public nonisolated static func pathSuggestsVisionModel(_ modelPath: String) -> Bool {
        let lowercased = modelPath.lowercased()
        return lowercased.contains("-vl-")
            || lowercased.contains("-vl_")
            || lowercased.contains("vision")
    }

    private nonisolated static func extractTokenArray(_ input: LMInput) -> [Int] {
        input.text.tokens.reshaped(-1).asArray(Int.self)
    }

    private nonisolated static func isMultimodalInput(_ input: LMInput) -> Bool {
        input.image != nil || input.video != nil
    }

    private nonisolated static func completeEOSTokenIDs(
        configuration: ModelConfiguration,
        tokenizer: any Tokenizer
    ) -> Set<Int> {
        var result = configuration.eosTokenIds
        if let eosTokenID = tokenizer.eosTokenId {
            result.insert(eosTokenID)
        }
        for token in configuration.extraEOSTokens {
            if let tokenID = tokenizer.convertTokenToId(token) {
                result.insert(tokenID)
            }
        }
        return result
    }
}
