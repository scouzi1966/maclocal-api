import Foundation
import MLX
import Cmlx
import MLXLLM
import MLXVLM
import MLXLMCommon
import Tokenizers
import Hub

/// Resolved log probability entry with token strings (ready for API response).
struct ResolvedLogprob: Sendable {
    let token: String
    let tokenId: Int
    let logprob: Float
    let topTokens: [(token: String, tokenId: Int, logprob: Float)]
}

/// A chunk of streaming output, optionally carrying per-token log probabilities or tool calls.
struct StreamChunk: Sendable {
    let text: String
    let logprobs: [ResolvedLogprob]?
    let toolCalls: [ResponseToolCall]?
    let promptTokens: Int?
    let completionTokens: Int?

    init(text: String, logprobs: [ResolvedLogprob]? = nil, toolCalls: [ResponseToolCall]? = nil, promptTokens: Int? = nil, completionTokens: Int? = nil) {
        self.text = text
        self.logprobs = logprobs
        self.toolCalls = toolCalls
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

enum MLXLoadStage: String {
    case checkingCache = "checking cache"
    case downloading = "downloading"
    case loadingModel = "loading model"
    case ready = "ready"
}

enum MLXServiceError: Error, LocalizedError {
    case invalidModel(String)
    case modelNotFoundInCache(String)
    case downloadFailed(String)
    case loadFailed(String)
    case noModelLoaded
    case serviceShuttingDown

    var errorDescription: String? {
        switch self {
        case .invalidModel(let value):
            return "Invalid model identifier: \(value)"
        case .modelNotFoundInCache(let value):
            return "Model not found in cache: \(value)"
        case .downloadFailed(let value):
            return "Failed to download model: \(value)"
        case .loadFailed(let value):
            return "Failed to load model: \(value)"
        case .noModelLoaded:
            return "No MLX model loaded"
        case .serviceShuttingDown:
            return "MLX service is shutting down"
        }
    }
}

/// Thread-safe container for KV cache state across requests.
/// Marked @unchecked Sendable because access is serialized through ModelContainer.perform.
private final class PromptCacheBox: @unchecked Sendable {
    var promptTokens: [Int] = []
    var cache: [KVCache] = []
    var modelID: String = ""
    var isValid: Bool = false

    func invalidate() {
        promptTokens = []
        cache = []
        modelID = ""
        isValid = false
    }
}

private let debugLogging = ProcessInfo.processInfo.environment["AFM_DEBUG"].map { $0 == "1" } ?? false

final class MLXModelService: @unchecked Sendable {
    private let resolver: MLXCacheResolver
    private let registry = MLXModelRegistry()
    private let stateLock = NSLock()
    private var currentModelID: String?
    private var currentContainer: ModelContainer?
    private var activeOperations: Int = 0
    private var isShuttingDown = false
    private var gpuInitialized = false
    private let promptCache = PromptCacheBox()
    private var currentToolCallFormat: ToolCallFormat?
    var enablePrefixCaching: Bool = false
    var prefillStepSize: Int = 2048
    var toolCallParser: String?
    var fixToolArgs: Bool = false
    init(resolver: MLXCacheResolver) {
        self.resolver = resolver
        self.resolver.applyEnvironment()
    }

    /// Configure MLX GPU settings once, before first model load.
    /// Must be called after Metal is available (not during early init).
    private func ensureGPUConfigured() {
        guard !gpuInitialized else { return }
        gpuInitialized = true

        let totalMemoryGB = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        let cacheMB: Int
        switch totalMemoryGB {
        case 0..<12:  cacheMB = 128
        case 12..<24: cacheMB = 256
        case 24..<48: cacheMB = 512
        default:      cacheMB = 1024
        }
        Memory.cacheLimit = cacheMB * 1024 * 1024

        let maxWorkingSet = GPU.deviceInfo().maxRecommendedWorkingSetSize
        let wiredLimitBytes = Int(Double(maxWorkingSet) * 0.9)
        var previousWired: size_t = 0
        mlx_set_wired_limit(&previousWired, size_t(wiredLimitBytes))

        print("MLX GPU: cache=\(cacheMB)MB wired=\(wiredLimitBytes / (1024*1024))MB (system \(totalMemoryGB)GB)")
    }

    func normalizeModel(_ raw: String) -> String {
        resolver.normalizedModelID(raw)
    }

    func revalidateRegistry() throws -> [String] {
        try registry.revalidate(using: resolver)
    }

    func ensureLoaded(
        model rawModel: String,
        progress: (@Sendable (Progress) -> Void)? = nil,
        stage: (@Sendable (MLXLoadStage) -> Void)? = nil,
        countOperation: Bool = true
    ) async throws -> String {
        var didBeginOperation = false
        if countOperation {
            try beginOperation()
            didBeginOperation = true
        }
        defer {
            if didBeginOperation {
                endOperation()
            }
        }

        let modelID = normalizeModel(rawModel)
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MLXServiceError.invalidModel(rawModel)
        }
        stage?(.checkingCache)

        if let cached = withStateLock({ () -> (String, ModelContainer)? in
            guard currentModelID == modelID, let container = currentContainer else { return nil }
            return (modelID, container)
        }) {
            stage?(.ready)
            return cached.0
        }

        ensureGPUConfigured()

        if resolver.localModelDirectory(repoId: modelID) == nil {
            stage?(.downloading)
            try await downloadModel(modelID: modelID, progress: progress)
        }

        guard let directory = resolver.localModelDirectory(repoId: modelID) else {
            throw MLXServiceError.modelNotFoundInCache(modelID)
        }

        var config = ModelConfiguration(directory: directory)
        let isVLM = try isVisionModel(directory: directory)

        // Auto-detect tool call format from model type (vendor LLMModelFactory lost this code)
        var detectedFormat = inferToolCallFormat(directory: directory)
        // --tool-call-parser override: force format for the specified parser
        if let parser = toolCallParser {
            switch parser {
            case "qwen3_xml":
                detectedFormat = .xmlFunction
            case "hermes", "llama3_json", "mistral":
                detectedFormat = .json
            case "gemma":
                detectedFormat = .gemma
            default:
                break
            }
            if debugLogging {
                print("[ToolCallParser] Forcing \(String(describing: detectedFormat)) format for \(parser) parser")
            }
        }
        config.toolCallFormat = detectedFormat
        stage?(.loadingModel)
        do {
            let loaded: ModelContainer
            if isVLM {
                loaded = try await VLMModelFactory.shared.loadContainer(configuration: config)
            } else {
                loaded = try await LLMModelFactory.shared.loadContainer(configuration: config)
            }
            withStateLock {
                currentContainer = loaded
                currentModelID = modelID
                currentToolCallFormat = detectedFormat
            }
            promptCache.invalidate()
            try registry.registerModel(modelID)
            stage?(.ready)
            return modelID
        } catch {
            throw MLXServiceError.loadFailed("\(modelID): \(error.localizedDescription)")
        }
    }

    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        tools: [RequestTool]? = nil,
        stop: [String]? = nil,
        responseFormat: ResponseFormat? = nil
    ) async throws -> (modelID: String, content: String, promptTokens: Int, completionTokens: Int, tokenLogprobs: [ResolvedLogprob]?, toolCalls: [ResponseToolCall]?) {
        try beginOperation()
        defer { endOperation() }

        let modelID = try await ensureLoaded(model: model, countOperation: false)
        guard let container = withStateLock({ currentContainer }) else { throw MLXServiceError.noModelLoaded }

        let promptText = buildPrompt(from: messages)
        let toolSpecs = convertToToolSpecs(tools)
        let userInput = try buildUserInput(from: messages, tools: toolSpecs, responseFormat: responseFormat)
        let wantLogprobs = logprobs == true
        let params = GenerateParameters(
            maxTokens: maxTokens ?? 2000,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: normalizedTemperature(temperature),
            topP: normalizedTopP(topP),
            repetitionPenalty: normalizedRepetitionPenalty(repetitionPenalty),
            repetitionContextSize: 64,
            topK: normalizedTopK(topK),
            minP: normalizedMinP(minP),
            presencePenalty: normalizedPresencePenalty(presencePenalty),
            seed: normalizedSeed(seed),
            computeLogprobs: wantLogprobs,
            topLogprobsCount: wantLogprobs ? min(max(topLogprobs ?? 0, 0), 20) : 0,
            prefillStepSize: self.prefillStepSize
        )

        var collectedLogprobs = [TokenLogprobData]()
        var resolvedLogprobs: [ResolvedLogprob]? = nil
        var collectedToolCalls = [ToolCall]()
        var completionInfo: GenerateCompletionInfo? = nil
        let promptCache = self.promptCache
        let generated: String = try await container.perform { context in
            let input = try await context.processor.prepare(input: userInput)

            // DEBUG: decode and print the full prompt to see what the template produced
            if debugLogging {
                let allTokens = input.text.tokens.reshaped(-1).asArray(Int.self)
                let decoded = context.tokenizer.decode(tokens: allTokens)
                print("[DEBUG] Full tokenized prompt (\(allTokens.count) tokens):\n\(decoded)\n[/DEBUG]")
            }

            // If the chat template appended <think>, prepend it so extractors can detect it
            let tokens = input.text.tokens
            let ndim = tokens.ndim
            let seqLen = tokens.dim(ndim - 1)
            var out = ""
            if seqLen >= 2 {
                let flat = tokens.reshaped(-1)
                let lastTwo = flat[seqLen - 2 ..< seqLen].asArray(Int.self)
                let decoded = context.tokenizer.decode(tokens: lastTwo)
                if decoded.contains("<think>") {
                    out = "<think>"
                }
            }

            // Prompt caching: determine cache hit/miss
            let useCache = self.enablePrefixCaching && !self.isMultimodalInput(input)
            let inputTokens = useCache ? self.extractTokenArray(input) : []
            var generationCache: [KVCache]
            var generateInput: LMInput

            if useCache {
                var prefixLen = self.findPrefixLength(incoming: inputTokens, currentModelID: modelID)
                if prefixLen > 0 {
                    // Near/exact match: ensure we re-feed at least minSuffix tokens
                    // to give the model enough context for stable generation.
                    // 1 token is fragile (can cause immediate EOS); 16 is a safe margin.
                    let minSuffix = 16
                    let maxPrefix = inputTokens.count - minSuffix
                    if prefixLen > maxPrefix {
                        prefixLen = max(0, maxPrefix)
                    }
                    if prefixLen > 0 {
                        // Reuse cached KV cache, trimmed to prefix length
                        generationCache = promptCache.cache
                        self.trimCacheToLength(generationCache, keepTokens: prefixLen)
                        // Build suffix-only input
                        let suffixTokens = Array(inputTokens[prefixLen...])
                        generateInput = LMInput(text: .init(tokens: MLXArray(suffixTokens)))
                        if debugLogging {
                            print("[KVCache] Prefix match: \(prefixLen)/\(inputTokens.count) tokens cached, processing \(suffixTokens.count) suffix tokens")
                        }
                    } else {
                        // Prefix too short after minSuffix adjustment, do full prefill
                        generationCache = context.model.newCache(parameters: params)
                        generateInput = input
                        if debugLogging {
                            print("[KVCache] Cache miss (prefix \(self.findPrefixLength(incoming: inputTokens, currentModelID: modelID)) < minSuffix), full prefill for \(inputTokens.count) tokens")
                        }
                    }
                } else {
                    // Cache miss: fresh cache
                    generationCache = context.model.newCache(parameters: params)
                    generateInput = input
                    if debugLogging {
                        print("[KVCache] Cache miss, full prefill for \(inputTokens.count) tokens")
                    }
                }
            } else {
                // Multimodal input: no caching
                generationCache = context.model.newCache(parameters: params)
                generateInput = input
                if debugLogging {
                    print("[KVCache] Multimodal input, skipping cache")
                }
            }

            let activeStops = stop?.filter { !$0.isEmpty } ?? []
            let genStart = Date()
            var firstTokenTime: Date?
            for await piece in try MLXLMCommon.generate(input: generateInput, cache: generationCache, parameters: params, context: context) {
                if debugLogging {
                    print("[DEBUG] Generation piece: \(piece)")
                }
                if case .chunk(let text) = piece {
                    if firstTokenTime == nil { firstTokenTime = Date() }
                    out += text
                    if !activeStops.isEmpty, let match = activeStops.first(where: { out.contains($0) }) {
                        if let range = out.range(of: match) {
                            out = String(out[..<range.lowerBound])
                        }
                        break
                    }
                } else if case .tokenLogprobs(let lps) = piece {
                    collectedLogprobs.append(contentsOf: lps)
                } else if case .toolCall(let tc) = piece {
                    if debugLogging {
                        print("[DEBUG] Tool call detected: \(tc.function.name)(\(tc.function.arguments))")
                    }
                    collectedToolCalls.append(tc)
                } else if case .info(let info) = piece {
                    completionInfo = info
                }
            }

            Stream.gpu.synchronize()
            if debugLogging {
                let ttft = firstTokenTime.map { $0.timeIntervalSince(genStart) } ?? 0
                let total = Date().timeIntervalSince(genStart)
                let promptTok = completionInfo?.promptTokenCount ?? 0
                let genTok = completionInfo?.generationTokenCount ?? 0
                print("[KVCache] Timing: TTFT=\(String(format: "%.3f", ttft))s total=\(String(format: "%.3f", total))s prompt_tokens=\(promptTok) gen_tokens=\(genTok)")
            }

            // Save prompt cache state (trim generation tokens, keep prompt-only)
            if useCache && !inputTokens.isEmpty {
                self.savePromptCacheState(cache: generationCache, promptTokens: inputTokens, modelID: modelID)
            }

            if wantLogprobs && !collectedLogprobs.isEmpty {
                resolvedLogprobs = self.resolveLogprobs(collectedLogprobs, tokenizer: context.tokenizer)
            }

            return out
        }

        // If the vendor ToolCallProcessor didn't detect tool calls, try fallback parsing.
        // Qwen3-Coder outputs <tool_call><function=name>...</function></tool_call> which
        // the vendor's XMLFunctionParser misses (regex doesn't match multiline content).
        var finalToolCalls = collectedToolCalls
        var finalContent = generated
        if finalToolCalls.isEmpty && tools != nil {
            let (parsed, remaining) = Self.extractToolCallsFallback(from: generated)
            if !parsed.isEmpty {
                finalToolCalls = parsed
                finalContent = remaining
            }
        }

        let responseToolCalls: [ResponseToolCall]? = finalToolCalls.isEmpty ? nil : finalToolCalls.enumerated().map { (i, tc) in
            var converted = Self.convertToolCall(tc, index: i)
            if self.fixToolArgs, let requestTools = tools {
                // Re-parse arguments JSON, remap keys, re-serialize
                if let data = converted.function.arguments.data(using: .utf8),
                   let argsDict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    var sendableArgs = [String: any Sendable]()
                    for (k, v) in argsDict { sendableArgs[k] = v }
                    let remapped = Self.remapArgumentKeys(sendableArgs, toolName: converted.function.name, tools: requestTools)
                    let remappedAny = remapped.mapValues { $0 as Any }
                    if let newData = try? JSONSerialization.data(withJSONObject: remappedAny, options: [.sortedKeys]),
                       let newStr = String(data: newData, encoding: .utf8) {
                        converted = ResponseToolCall(
                            id: converted.id,
                            type: converted.type,
                            function: ResponseToolCallFunction(name: converted.function.name, arguments: newStr)
                        )
                    }
                }
            }
            return converted
        }

        let promptTokens = completionInfo?.promptTokenCount ?? estimateTokens(promptText)
        let completionTokens = completionInfo?.generationTokenCount ?? estimateTokens(generated)
        return (modelID, finalContent, promptTokens, completionTokens, resolvedLogprobs, responseToolCalls)
    }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        tools: [RequestTool]? = nil,
        stop: [String]? = nil,
        responseFormat: ResponseFormat? = nil
    ) async throws -> (modelID: String, stream: AsyncThrowingStream<StreamChunk, Error>, promptTokens: Int, toolCallStartTag: String?, toolCallEndTag: String?) {
        try beginOperation()
        defer { endOperation() }

        let modelID = try await ensureLoaded(model: model, countOperation: false)
        guard let container = withStateLock({ currentContainer }) else { throw MLXServiceError.noModelLoaded }

        let promptText = buildPrompt(from: messages)
        let toolSpecs = convertToToolSpecs(tools)
        let userInput = try buildUserInput(from: messages, tools: toolSpecs, responseFormat: responseFormat)
        let promptTokens = estimateTokens(promptText)
        let wantLogprobs = logprobs == true
        let params = GenerateParameters(
            maxTokens: maxTokens ?? 2000,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: normalizedTemperature(temperature),
            topP: normalizedTopP(topP),
            repetitionPenalty: normalizedRepetitionPenalty(repetitionPenalty),
            repetitionContextSize: 64,
            topK: normalizedTopK(topK),
            minP: normalizedMinP(minP),
            presencePenalty: normalizedPresencePenalty(presencePenalty),
            seed: normalizedSeed(seed),
            computeLogprobs: wantLogprobs,
            topLogprobsCount: wantLogprobs ? min(max(topLogprobs ?? 0, 0), 20) : 0,
            prefillStepSize: self.prefillStepSize
        )

        let promptCache = self.promptCache
        let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
            let task = Task {
                do {
                    try await container.perform { context in
                        let input = try await context.processor.prepare(input: userInput)

                        // If the chat template appended <think> to the prompt, inject it
                        // into the stream so the reasoning extractor can detect it.
                        let tokens = input.text.tokens
                        let ndim = tokens.ndim
                        let seqLen = tokens.dim(ndim - 1)
                        if seqLen >= 2 {
                            let flat = tokens.reshaped(-1)
                            let lastTwo = flat[seqLen - 2 ..< seqLen].asArray(Int.self)
                            let decoded = context.tokenizer.decode(tokens: lastTwo)
                            if decoded.contains("<think>") {
                                continuation.yield(StreamChunk(text: "<think>"))
                            }
                        }

                        // Prompt caching: determine cache hit/miss
                        let useCache = self.enablePrefixCaching && !self.isMultimodalInput(input)
                        let inputTokens = useCache ? self.extractTokenArray(input) : []
                        var generationCache: [KVCache]
                        var generateInput: LMInput

                        if useCache {
                            var prefixLen = self.findPrefixLength(incoming: inputTokens, currentModelID: modelID)
                            if prefixLen > 0 {
                                // Near/exact match: ensure we re-feed at least minSuffix tokens
                                let minSuffix = 16
                                let maxPrefix = inputTokens.count - minSuffix
                                if prefixLen > maxPrefix {
                                    prefixLen = max(0, maxPrefix)
                                }
                                if prefixLen > 0 {
                                    generationCache = promptCache.cache
                                    self.trimCacheToLength(generationCache, keepTokens: prefixLen)
                                    let suffixTokens = Array(inputTokens[prefixLen...])
                                    generateInput = LMInput(text: .init(tokens: MLXArray(suffixTokens)))
                                    if debugLogging {
                                        print("[KVCache] Prefix match: \(prefixLen)/\(inputTokens.count) tokens cached, processing \(suffixTokens.count) suffix tokens")
                                    }
                                } else {
                                    generationCache = context.model.newCache(parameters: params)
                                    generateInput = input
                                    if debugLogging {
                                        print("[KVCache] Cache miss (prefix < minSuffix), full prefill for \(inputTokens.count) tokens")
                                    }
                                }
                            } else {
                                generationCache = context.model.newCache(parameters: params)
                                generateInput = input
                                if debugLogging {
                                    print("[KVCache] Cache miss, full prefill for \(inputTokens.count) tokens")
                                }
                            }
                        } else {
                            generationCache = context.model.newCache(parameters: params)
                            generateInput = input
                            if debugLogging {
                                print("[KVCache] Multimodal input, skipping cache")
                            }
                        }

                        let activeStops = stop?.filter { !$0.isEmpty } ?? []
                        // Buffer to handle stop strings that span chunk boundaries
                        let maxStopLen = activeStops.map(\.count).max() ?? 0
                        var stopBuffer = ""
                        let genStart = Date()
                        var firstTokenTime: Date?

                        var pendingLogprobs: [TokenLogprobData]? = nil
                        for await piece in try MLXLMCommon.generate(input: generateInput, cache: generationCache, parameters: params, context: context) {
                            if Task.isCancelled {
                                print("[MLX] Generation cancelled by client")
                                break
                            }
                            if case .tokenLogprobs(let lps) = piece {
                                pendingLogprobs = lps
                            } else if case .chunk(let text) = piece {
                                if firstTokenTime == nil { firstTokenTime = Date() }
                                let resolved: [ResolvedLogprob]?
                                if let lps = pendingLogprobs {
                                    resolved = self.resolveLogprobs(lps, tokenizer: context.tokenizer)
                                } else {
                                    resolved = nil
                                }

                                if !activeStops.isEmpty {
                                    stopBuffer += text
                                    // Check for a complete stop string match
                                    if let match = activeStops.first(where: { stopBuffer.contains($0) }) {
                                        // Emit text up to the stop string
                                        if let range = stopBuffer.range(of: match) {
                                            let before = String(stopBuffer[..<range.lowerBound])
                                            if !before.isEmpty {
                                                continuation.yield(StreamChunk(text: before, logprobs: resolved))
                                            }
                                        }
                                        break
                                    }
                                    // Flush safe portion of the buffer (keep tail that could be partial stop match)
                                    if stopBuffer.count > maxStopLen {
                                        let flushEnd = stopBuffer.index(stopBuffer.endIndex, offsetBy: -maxStopLen)
                                        let flushText = String(stopBuffer[..<flushEnd])
                                        stopBuffer = String(stopBuffer[flushEnd...])
                                        continuation.yield(StreamChunk(text: flushText, logprobs: resolved))
                                    }
                                } else {
                                    continuation.yield(StreamChunk(text: text, logprobs: resolved))
                                }
                                pendingLogprobs = nil
                            } else if case .toolCall(let tc) = piece {
                                // Emit tool call as a stream chunk with empty text
                                let responseTC = Self.convertToolCall(tc, index: 0)
                                continuation.yield(StreamChunk(text: "", toolCalls: [responseTC]))
                            } else if case .info(let info) = piece {
                                // Emit real token counts as a final info chunk
                                continuation.yield(StreamChunk(text: "", promptTokens: info.promptTokenCount, completionTokens: info.generationTokenCount))
                            }
                        }
                        // Flush any remaining buffered text (no stop match found)
                        if !activeStops.isEmpty && !stopBuffer.isEmpty {
                            continuation.yield(StreamChunk(text: stopBuffer))
                        }
                        // Synchronize GPU after generation completes (or breaks early).
                        Stream.gpu.synchronize()
                        if debugLogging {
                            let ttft = firstTokenTime.map { $0.timeIntervalSince(genStart) } ?? 0
                            let total = Date().timeIntervalSince(genStart)
                            print("[KVCache] Timing: TTFT=\(String(format: "%.3f", ttft))s total=\(String(format: "%.3f", total))s (streaming)")
                        }

                        // Save prompt cache state (trim generation tokens, keep prompt-only)
                        if useCache && !inputTokens.isEmpty && !Task.isCancelled {
                            self.savePromptCacheState(cache: generationCache, promptTokens: inputTokens, modelID: modelID)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }

        // Derive tool call start/end tags for streaming detection
        let toolTags: (start: String, end: String)?
        if let tools, !tools.isEmpty {
            let format = withStateLock({ currentToolCallFormat })
            if let format {
                switch format {
                case .xmlFunction:
                    // XMLFunctionParser has nil tags; chat template wraps in <tool_call>
                    toolTags = ("<tool_call>", "</tool_call>")
                default:
                    let parser = format.createParser()
                    toolTags = (parser.startTag ?? "<tool_call>", parser.endTag ?? "</tool_call>")
                }
            } else {
                toolTags = ("<tool_call>", "</tool_call>")
            }
        } else {
            toolTags = nil
        }

        return (modelID, stream, promptTokens, toolTags?.start, toolTags?.end)
    }

    func shutdownAndReleaseResources(verbose: Bool = false, timeoutSeconds: TimeInterval = 30) async {
        let start = Date()
        withStateLock { isShuttingDown = true }

        while Date().timeIntervalSince(start) < timeoutSeconds {
            if withStateLock({ activeOperations == 0 }) {
                break
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }

        promptCache.invalidate()
        autoreleasepool {
            withStateLock {
                currentContainer = nil
                currentModelID = nil
            }
        }

        // Ensure queued GPU work is complete before clearing recycled buffers.
        Stream.gpu.synchronize()
        Stream.cpu.synchronize()
        Memory.clearCache()
        Stream.gpu.synchronize()
        Memory.clearCache()

        if verbose {
            let snapshot = Memory.snapshot()
            print("MLX memory after shutdown - active: \(formatBytes(snapshot.activeMemory)), cache: \(formatBytes(snapshot.cacheMemory)), peak: \(formatBytes(snapshot.peakMemory))")
        }
    }

    // MARK: - Tool conversion helpers

    /// Convert OpenAI-format RequestTool array to vendor ToolSpec array.
    private func convertToToolSpecs(_ tools: [RequestTool]?) -> [ToolSpec]? {
        guard let tools, !tools.isEmpty else { return nil }
        return tools.map { tool -> ToolSpec in
            var funcDict: [String: any Sendable] = [
                "name": tool.function.name
            ]
            if let desc = tool.function.description {
                funcDict["description"] = desc
            }
            if let params = tool.function.parameters {
                funcDict["parameters"] = params.toSendable()
            }
            return [
                "type": tool.type,
                "function": funcDict
            ]
        }
    }

    /// Convert a vendor ToolCall to an OpenAI-compatible ResponseToolCall.
    static func convertToolCall(_ tc: ToolCall, index: Int, paramNameMapping: [String: String] = [:]) -> ResponseToolCall {
        // Apply parameter name mapping (e.g. snake_case → camelCase) if provided.
        // Qwen3-Coder converts camelCase param names to snake_case in XML output.
        let argsDict: [String: Any]
        if paramNameMapping.isEmpty {
            argsDict = tc.function.arguments.mapValues { $0.anyValue }
        } else {
            var mapped = [String: Any]()
            for (key, value) in tc.function.arguments {
                let mappedKey = paramNameMapping[key] ?? key
                mapped[mappedKey] = value.anyValue
            }
            argsDict = mapped
        }
        let argsJSON: String
        if let data = try? JSONSerialization.data(withJSONObject: argsDict, options: [.sortedKeys]),
           let str = String(data: data, encoding: .utf8) {
            argsJSON = str
        } else {
            argsJSON = "{}"
        }
        return ResponseToolCall(
            id: "call_\(generateCallID())",
            type: "function",
            function: ResponseToolCallFunction(
                name: tc.function.name,
                arguments: argsJSON
            )
        )
    }

    /// Remap tool call argument keys to match the original tool schema.
    /// Heuristics (in priority order):
    /// 1. Exact match — key exists in schema → keep as-is
    /// 2. Case-insensitive match — e.g. "filepath" → "filePath"
    /// 3. Snake↔Camel match — e.g. "file_path" → "filePath" or "filePath" → "file_path"
    /// 4. Suffix match — e.g. "path" matches "filePath" (only if exactly one candidate)
    static func remapArgumentKeys(_ arguments: [String: any Sendable], toolName: String, tools: [RequestTool]) -> [String: any Sendable] {
        // Find the matching tool schema
        guard let tool = tools.first(where: { $0.function.name == toolName }),
              let paramsAny = tool.function.parameters?.toSendable() as? [String: Any],
              let props = paramsAny["properties"] as? [String: Any] else {
            return arguments
        }
        let schemaKeys = Array(props.keys)
        let schemaKeysLower = schemaKeys.map { $0.lowercased() }

        var remapped = [String: any Sendable]()
        for (key, value) in arguments {
            // 1. Exact match
            if props[key] != nil {
                remapped[key] = value
                continue
            }

            // 2. Case-insensitive match
            let keyLower = key.lowercased()
            if let idx = schemaKeysLower.firstIndex(of: keyLower) {
                let mapped = schemaKeys[idx]
                if debugLogging { print("[ToolCallRemap] \(key) → \(mapped) (case-insensitive)") }
                remapped[mapped] = value
                continue
            }

            // 3. Snake↔Camel conversion
            // Try converting key from snake_case to camelCase
            let camelized = snakeToCamel(key)
            if camelized != key, props[camelized] != nil {
                if debugLogging { print("[ToolCallRemap] \(key) → \(camelized) (snake→camel)") }
                remapped[camelized] = value
                continue
            }
            // Try converting key from camelCase to snake_case
            let snaked = camelToSnake(key)
            if snaked != key, props[snaked] != nil {
                if debugLogging { print("[ToolCallRemap] \(key) → \(snaked) (camel→snake)") }
                remapped[snaked] = value
                continue
            }

            // 4. Suffix match — model's key is a suffix of exactly one schema key
            let suffixCandidates = schemaKeys.filter {
                $0.lowercased().hasSuffix(keyLower) && $0.count > key.count
            }
            if suffixCandidates.count == 1 {
                let mapped = suffixCandidates[0]
                if debugLogging { print("[ToolCallRemap] \(key) → \(mapped) (suffix)") }
                remapped[mapped] = value
                continue
            }

            // No match — keep original key
            remapped[key] = value
        }
        return remapped
    }

    /// Convert snake_case to camelCase: "file_path" → "filePath"
    private static func snakeToCamel(_ s: String) -> String {
        let parts = s.split(separator: "_", omittingEmptySubsequences: false)
        guard parts.count > 1 else { return s }
        return String(parts[0]) + parts.dropFirst().map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined()
    }

    /// Convert camelCase to snake_case: "filePath" → "file_path"
    private static func camelToSnake(_ s: String) -> String {
        var result = ""
        for (i, char) in s.enumerated() {
            if char.isUppercase {
                if i > 0 { result += "_" }
                result += char.lowercased()
            } else {
                result += String(char)
            }
        }
        return result
    }

    /// Generate a random alphanumeric ID for tool call IDs.
    private static func generateCallID() -> String {
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return String((0..<24).map { _ in chars.randomElement()! })
    }

    /// Fallback tool call extraction for formats the vendor parser misses.
    /// Handles <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    /// and <tool_call>{"name":"func","arguments":{...}}</tool_call> patterns.
    /// Returns extracted ToolCalls and remaining non-tool-call content.
    static func extractToolCallsFallback(from text: String) -> ([ToolCall], String) {
        var toolCalls = [ToolCall]()
        var remaining = text

        // Match <tool_call>...</tool_call> blocks (dotMatchesLineSeparators for multiline)
        let toolCallRegex = try! NSRegularExpression(
            pattern: #"<tool_call>\s*(.*?)\s*</tool_call>"#,
            options: [.dotMatchesLineSeparators]
        )
        let matches = toolCallRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            guard let innerRange = Range(match.range(at: 1), in: text) else { continue }
            let inner = String(text[innerRange])

            // Try XML function format: <function=name><parameter=key>value</parameter></function>
            if let tc = parseXMLFunction(inner) {
                toolCalls.insert(tc, at: 0)
                if let fullRange = Range(match.range, in: remaining) {
                    remaining.removeSubrange(fullRange)
                }
                continue
            }

            // Try JSON format: {"name":"func","arguments":{...}}
            if let tc = parseJSONToolCall(inner) {
                toolCalls.insert(tc, at: 0)
                if let fullRange = Range(match.range, in: remaining) {
                    remaining.removeSubrange(fullRange)
                }
            }
        }

        // Fallback: Mistral models may emit [TOOL_CALLS] in various formats
        if toolCalls.isEmpty {
            let trimmed = remaining.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("[TOOL_CALLS]") {
                let afterPrefix = String(trimmed.dropFirst("[TOOL_CALLS]".count)).trimmingCharacters(in: .whitespacesAndNewlines)

                // Format 1: [TOOL_CALLS][{"name":"func","arguments":{...}}]
                if let data = afterPrefix.data(using: .utf8),
                   let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    for item in arr {
                        if let tc = parseJSONToolCall(String(data: (try? JSONSerialization.data(withJSONObject: item)) ?? Data(), encoding: .utf8) ?? "{}") {
                            toolCalls.append(tc)
                        }
                    }
                    if !toolCalls.isEmpty {
                        remaining = ""
                    }
                }

                // Format 2: [TOOL_CALLS]func_name[ARGS]{"key":"value"}
                if toolCalls.isEmpty {
                    let argsPattern = try! NSRegularExpression(
                        pattern: #"([a-zA-Z_][a-zA-Z0-9_]*)\[ARGS\](\{[\s\S]*?\})(?:\s|$)"#,
                        options: [])
                    let matches = argsPattern.matches(in: afterPrefix, range: NSRange(afterPrefix.startIndex..., in: afterPrefix))
                    for match in matches {
                        if let nameRange = Range(match.range(at: 1), in: afterPrefix),
                           let argsRange = Range(match.range(at: 2), in: afterPrefix) {
                            let name = String(afterPrefix[nameRange])
                            let argsStr = String(afterPrefix[argsRange])
                            if let argsData = argsStr.data(using: .utf8),
                               let argsDict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any] {
                                var args: [String: String] = [:]
                                for (k, v) in argsDict {
                                    args[k] = "\(v)"
                                }
                                toolCalls.append(ToolCall(function: .init(name: name, arguments: args)))
                            }
                        }
                    }
                    if !toolCalls.isEmpty {
                        remaining = ""
                    }
                }
            }
        }

        // Fallback: bare JSON tool call (no wrapper tags)
        // e.g. {"name":"get_weather","arguments":{"city":"Tokyo"}} or with "parameters"
        if toolCalls.isEmpty {
            let trimmed = remaining.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("{") && trimmed.hasSuffix("}") {
                if let tc = parseJSONToolCall(trimmed) {
                    toolCalls.append(tc)
                    remaining = ""
                }
            }
        }

        // Trim leftover whitespace/think tags from remaining
        remaining = remaining
            .replacingOccurrences(of: #"<think>\s*</think>"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return (toolCalls, remaining)
    }

    /// Parse <function=name><parameter=key>value</parameter></function>
    private static func parseXMLFunction(_ content: String) -> ToolCall? {
        let funcRegex = try! NSRegularExpression(
            pattern: #"<function=([^>]+)>(.*?)</function>"#,
            options: [.dotMatchesLineSeparators]
        )
        guard let funcMatch = funcRegex.firstMatch(in: content, range: NSRange(content.startIndex..., in: content)),
              let nameRange = Range(funcMatch.range(at: 1), in: content),
              let bodyRange = Range(funcMatch.range(at: 2), in: content) else {
            return nil
        }

        let funcName = String(content[nameRange])
        let body = String(content[bodyRange])

        var arguments: [String: any Sendable] = [:]
        let paramRegex = try! NSRegularExpression(
            pattern: #"<parameter=([^>]+)>(.*?)</parameter>"#,
            options: [.dotMatchesLineSeparators]
        )
        let paramMatches = paramRegex.matches(in: body, range: NSRange(body.startIndex..., in: body))
        for pm in paramMatches {
            guard let keyRange = Range(pm.range(at: 1), in: body),
                  let valRange = Range(pm.range(at: 2), in: body) else { continue }
            let key = String(body[keyRange])
            var val = String(body[valRange])
            if val.hasPrefix("\n") { val = String(val.dropFirst()) }
            if val.hasSuffix("\n") { val = String(val.dropLast()) }
            // Keep first non-empty value — Qwen3-Coder-Next sometimes emits
            // duplicate parameters where the second is malformed/empty.
            // See: https://github.com/anomalyco/opencode/issues/6918
            if !val.isEmpty, arguments[key] == nil {
                arguments[key] = val
            }
        }

        // Salvage unclosed parameters (e.g. model hit max_tokens mid-content).
        // Look for <parameter=KEY>VALUE... without a closing </parameter>.
        let unclosedRegex = try! NSRegularExpression(
            pattern: #"<parameter=([^>]+)>([\s\S]+)$"#,
            options: []
        )
        if let unclosedMatch = unclosedRegex.firstMatch(in: body, range: NSRange(body.startIndex..., in: body)),
           let keyRange = Range(unclosedMatch.range(at: 1), in: body),
           let valRange = Range(unclosedMatch.range(at: 2), in: body) {
            let key = String(body[keyRange])
            if arguments[key] == nil {
                var val = String(body[valRange])
                if val.hasPrefix("\n") { val = String(val.dropFirst()) }
                if val.hasSuffix("\n") { val = String(val.dropLast()) }
                // Strip any trailing </function> tag that may be part of the parent
                if let funcEnd = val.range(of: "</function>") {
                    val = String(val[..<funcEnd.lowerBound])
                    if val.hasSuffix("\n") { val = String(val.dropLast()) }
                }
                if !val.isEmpty {
                    arguments[key] = val
                    if debugLogging {
                        print("[ToolCallParser] Salvaged unclosed parameter '\(key)' (\(val.count) chars)")
                    }
                }
            }
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Parse {"name":"func","arguments":{...}} JSON tool call
    private static func parseJSONToolCall(_ content: String) -> ToolCall? {
        guard let data = content.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = json["name"] as? String else {
            return nil
        }
        var arguments: [String: any Sendable] = [:]
        if let args = (json["arguments"] as? [String: Any]) ?? (json["parameters"] as? [String: Any]) {
            for (k, v) in args {
                arguments[k] = v
            }
        }
        return ToolCall(function: .init(name: name, arguments: arguments))
    }

    // MARK: - Private helpers

    private func beginOperation() throws {
        try withStateLock {
            if isShuttingDown {
                throw MLXServiceError.serviceShuttingDown
            }
            activeOperations += 1
        }
    }

    private func endOperation() {
        withStateLock {
            activeOperations = max(0, activeOperations - 1)
        }
    }

    private func withStateLock<T>(_ body: () throws -> T) rethrows -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return try body()
    }

    private func formatBytes(_ bytes: Int) -> String {
        let gb = Double(bytes) / 1_073_741_824.0
        return String(format: "%.2f GB", gb)
    }

    private func downloadModel(modelID: String, progress: (@Sendable (Progress) -> Void)?) async throws {
        do {
            _ = try await Hub.snapshot(
                from: modelID,
                matching: ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "tokenizer*", "*.bpe", "*.bin"],
                progressHandler: { p in progress?(p) }
            )
        } catch {
            throw MLXServiceError.downloadFailed("\(modelID): \(error.localizedDescription)")
        }
    }

    private func inferToolCallFormat(directory: URL) -> ToolCallFormat? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = json["model_type"] as? String else {
            return nil
        }
        return ToolCallFormat.infer(from: modelType)
    }

    private func isVisionModel(directory: URL) throws -> Bool {
        let config = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: config),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        let modelType = (json["model_type"] as? String ?? "").lowercased()
        if modelType.contains("vl") || modelType.contains("vision") {
            return true
        }
        // Multimodal models (e.g. gemma3) have both text_config and vision_config
        if json["text_config"] != nil && json["vision_config"] != nil {
            return true
        }
        // Some VLMs (e.g. Qwen3.5-MoE) have vision token IDs without a vision_config block
        if json["image_token_id"] != nil || json["vision_start_token_id"] != nil {
            return true
        }
        return false
    }

    private func buildPrompt(from messages: [Message]) -> String {
        messages.map { "\($0.role): \($0.textContent)" }.joined(separator: "\n")
    }

    private func buildUserInput(from messages: [Message], tools: [ToolSpec]? = nil, responseFormat: ResponseFormat? = nil) throws -> UserInput {
        var chatMessages: [Chat.Message] = []
        var hasSystemMessage = false
        for m in messages {
            let text = m.textContent
            let images = try extractImages(from: m)
            switch m.role {
            case "system", "developer":
                hasSystemMessage = true
                chatMessages.append(.system(text))
            case "assistant":
                if let toolCalls = m.toolCalls, !toolCalls.isEmpty {
                    // Reconstruct assistant tool-call message as text for the chat template.
                    // Models expect tool calls in a specific format that the template handles.
                    var parts: [String] = []
                    if !text.isEmpty {
                        parts.append(text)
                    }
                    for tc in toolCalls {
                        parts.append("<tool_call>\n{\"name\": \"\(tc.function.name)\", \"arguments\": \(tc.function.arguments)}\n</tool_call>")
                    }
                    chatMessages.append(.assistant(parts.joined(separator: "\n")))
                } else {
                    chatMessages.append(.assistant(text))
                }
            case "tool":
                // Tool result messages — use the vendor's .tool() factory
                let toolContent: String
                if let name = m.name {
                    toolContent = "<tool_response>\n{\"name\": \"\(name)\", \"content\": \(text)}\n</tool_response>"
                } else {
                    toolContent = text
                }
                chatMessages.append(.tool(toolContent))
            default:
                chatMessages.append(.user(text, images: images))
            }
        }

        // Align with Vesta behavior: always include a base system instruction
        // when callers don't explicitly provide one.
        if !hasSystemMessage {
            chatMessages.insert(.system("You are a helpful assistant!"), at: 0)
        }

        // Inject JSON format instructions when response_format is requested
        if let format = responseFormat {
            let jsonInstruction: String?
            switch format.type {
            case "json_object":
                jsonInstruction = "Respond with valid JSON only. Do not include any text outside the JSON object."
            case "json_schema":
                if let schema = format.jsonSchema {
                    var parts = ["Respond with valid JSON only. Do not include any text outside the JSON object."]
                    if let schemaValue = schema.schema {
                        let encoder = JSONEncoder()
                        encoder.outputFormatting = [.sortedKeys]
                        if let data = try? encoder.encode(schemaValue),
                           let schemaStr = String(data: data, encoding: .utf8) {
                            parts.append("Your response must conform to this JSON schema: \(schemaStr)")
                        }
                    }
                    if let name = schema.name {
                        parts.append("The response object is: \(name)")
                    }
                    jsonInstruction = parts.joined(separator: "\n")
                } else {
                    jsonInstruction = "Respond with valid JSON only. Do not include any text outside the JSON object."
                }
            default:
                jsonInstruction = nil
            }
            if let instruction = jsonInstruction {
                chatMessages.append(.system(instruction))
            }
        }

        if chatMessages.isEmpty {
            return UserInput(prompt: "")
        }

        var input = UserInput(chat: chatMessages, processing: .init(resize: .init(width: 1024, height: 1024)), tools: tools)

        // When --tool-call-parser is set and tools are present, override the chat template
        if let parser = toolCallParser, tools != nil, !tools!.isEmpty {
            let templateOverride: String?
            switch parser {
            case "qwen3_xml":
                templateOverride = Self.qwen3XMLTemplate
            case "hermes":
                templateOverride = Self.hermesTemplate
            case "llama3_json":
                templateOverride = Self.llama3JSONTemplate
            case "mistral":
                templateOverride = Self.mistralTemplate
            case "gemma":
                // Gemma uses the model's built-in template; no override needed
                templateOverride = nil
            default:
                print("Warning: unknown tool-call-parser '\(parser)', using default chat template")
                templateOverride = nil
            }
            if let tpl = templateOverride {
                input.additionalContext = (input.additionalContext ?? [:])
                input.additionalContext?["chatTemplateOverride"] = tpl
            }
            if debugLogging {
                print("[ToolCallParser] Using \(parser) chat template override")
            }
        }

        return input
    }

    private func extractImages(from message: Message) throws -> [UserInput.Image] {
        guard let content = message.content, case .parts(let parts) = content else { return [] }
        var images: [UserInput.Image] = []
        for part in parts where part.type == "image_url" {
            guard let raw = part.image_url?.url, let url = URL(string: raw) else { continue }
            if let scheme = url.scheme, scheme == "http" || scheme == "https" {
                let (data, _) = try awaitURL(url: url)
                let temp = FileManager.default.temporaryDirectory
                    .appendingPathComponent("afm_mlx_image_\(UUID().uuidString).\(url.pathExtension.isEmpty ? "jpg" : url.pathExtension)")
                try data.write(to: temp)
                images.append(.url(temp))
            } else {
                images.append(.url(url))
            }
        }
        return images
    }

    private func awaitURL(url: URL) throws -> (Data, URLResponse) {
        let sem = DispatchSemaphore(value: 0)
        var result: Result<(Data, URLResponse), Error>?
        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let error {
                result = .failure(error)
            } else if let data, let response {
                result = .success((data, response))
            } else {
                result = .failure(MLXServiceError.downloadFailed("image download failed"))
            }
            sem.signal()
        }
        task.resume()
        sem.wait()
        switch result {
        case .success(let pair):
            return pair
        case .failure(let error):
            throw error
        case .none:
            throw MLXServiceError.downloadFailed("image download failed")
        }
    }

    private func estimateTokens(_ text: String) -> Int {
        let words = text.split(whereSeparator: \.isWhitespace).count
        let charBased = Double(text.count) / 4.0
        let wordBased = Double(words) / 0.75
        return Int(max(charBased, wordBased))
    }

    private func resolveLogprobs(_ data: [TokenLogprobData], tokenizer: any Tokenizer) -> [ResolvedLogprob] {
        data.map { entry in
            let token = tokenizer.decode(tokens: [entry.tokenId])
            let topTokens = zip(entry.topTokenIds, entry.topLogprobs).map { (id, lp) in
                (token: tokenizer.decode(tokens: [id]), tokenId: id, logprob: lp)
            }
            return ResolvedLogprob(
                token: token,
                tokenId: entry.tokenId,
                logprob: entry.logprob,
                topTokens: topTokens
            )
        }
    }

    // MARK: - Prompt cache helpers

    /// Extract a flat array of token IDs from prepared LMInput.
    private func extractTokenArray(_ input: LMInput) -> [Int] {
        input.text.tokens.reshaped(-1).asArray(Int.self)
    }

    /// Find the length of the common token prefix between incoming tokens and the cached state.
    /// Returns 0 on cache miss (different model, no cache, or no common prefix).
    private func findPrefixLength(incoming: [Int], currentModelID: String) -> Int {
        guard promptCache.isValid,
              promptCache.modelID == currentModelID,
              !promptCache.cache.isEmpty,
              !promptCache.promptTokens.isEmpty else {
            return 0
        }
        let cached = promptCache.promptTokens
        let limit = min(incoming.count, cached.count)
        var prefixLen = 0
        while prefixLen < limit && incoming[prefixLen] == cached[prefixLen] {
            prefixLen += 1
        }
        return prefixLen
    }

    /// Trim KV cache to keep only the first `keepTokens` tokens across all layers.
    private func trimCacheToLength(_ cache: [KVCache], keepTokens: Int) {
        for layer in cache {
            let excess = layer.offset - keepTokens
            if excess > 0 {
                layer.trim(excess)
            }
        }
    }

    /// Save prompt-only cache state after generation completes.
    /// Trims generation tokens from the cache so only prompt tokens remain.
    private func savePromptCacheState(cache: [KVCache], promptTokens: [Int], modelID: String) {
        let promptLen = promptTokens.count
        trimCacheToLength(cache, keepTokens: promptLen)
        promptCache.cache = cache
        promptCache.promptTokens = promptTokens
        promptCache.modelID = modelID
        promptCache.isValid = true
        if debugLogging {
            print("[KVCache] Saved prompt cache: \(promptLen) tokens for model \(modelID)")
        }
    }

    /// Check if the LMInput contains multimodal content (images/video) which we don't cache.
    private func isMultimodalInput(_ input: LMInput) -> Bool {
        input.image != nil || input.video != nil
    }

    private func normalizedRepetitionPenalty(_ value: Double?) -> Float? {
        guard let value else { return nil }
        if abs(value - 1.0) < 0.000_001 {
            return nil
        }
        return Float(value)
    }

    private func normalizedTopP(_ value: Double?) -> Float {
        guard let value else { return 1.0 }  // MLX library default
        return Float(min(max(value, 0.0), 1.0))
    }

    private func normalizedTemperature(_ value: Double?) -> Float {
        guard let value else { return 0.6 }  // MLX library default
        return Float(min(max(value, 0.0), 1.0))
    }

    private func normalizedTopK(_ value: Int?) -> Int {
        guard let value else { return 0 }  // 0 = disabled
        return max(0, value)
    }

    private func normalizedMinP(_ value: Double?) -> Float {
        guard let value else { return 0.0 }  // 0.0 = disabled
        return Float(min(max(value, 0.0), 1.0))
    }

    private func normalizedPresencePenalty(_ value: Double?) -> Float {
        guard let value else { return 0.0 }  // 0.0 = disabled
        return Float(value)
    }

    private func normalizedSeed(_ value: Int?) -> UInt64? {
        guard let value else { return nil }
        return UInt64(max(0, value))
    }

    // MARK: - Tool call parser templates

    /// vLLM's tool_chat_template_qwen3coder.jinja — teaches the model the exact XML tool call format.
    /// Source: https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_qwen3coder.jinja
    static let qwen3XMLTemplate = """
    {% macro render_extra_keys(json_dict, handled_keys) %}
        {%- if json_dict is mapping %}
            {%- for json_key in json_dict if json_key not in handled_keys %}
                {%- if json_dict[json_key] is mapping or (json_dict[json_key] is sequence and json_dict[json_key] is not string) %}
                    {{- '\\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | tojson | safe) ~ '</' ~ json_key ~ '>' }}
                {%- else %}
                    {{-'\\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | string) ~ '</' ~ json_key ~ '>' }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
    {% endmacro %}

    {%- if messages[0]["role"] == "system" %}
        {%- set system_message = messages[0]["content"] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set loop_messages = messages %}
    {%- endif %}

    {%- if not tools is defined %}
        {%- set tools = [] %}
    {%- endif %}

    {%- if system_message is defined %}
        {{- "<|im_start|>system\\n" + system_message }}
    {%- else %}
        {%- if tools is iterable and tools | length > 0 %}
            {{- "<|im_start|>system\\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks." }}
        {%- endif %}
    {%- endif %}
    {%- if tools is iterable and tools | length > 0 %}
        {{- "\\n\\n# Tools\\n\\nYou have access to the following functions:\\n\\n" }}
        {{- "<tools>" }}
        {%- for tool in tools %}
            {%- if tool.function is defined %}
                {%- set tool = tool.function %}
            {%- endif %}
            {{- "\\n<function>\\n<name>" ~ tool.name ~ "</name>" }}
            {%- if tool.description is defined %}
                {{- '\\n<description>' ~ (tool.description | trim) ~ '</description>' }}
            {%- endif %}
            {{- '\\n<parameters>' }}
            {%- if tool.parameters is defined and tool.parameters is mapping and tool.parameters.properties is defined and tool.parameters.properties is mapping %}
                {%- for param_name, param_fields in tool.parameters.properties|items %}
                    {{- '\\n<parameter>' }}
                    {{- '\\n<name>' ~ param_name ~ '</name>' }}
                    {%- if param_fields.type is defined %}
                        {{- '\\n<type>' ~ (param_fields.type | string) ~ '</type>' }}
                    {%- endif %}
                    {%- if param_fields.description is defined %}
                        {{- '\\n<description>' ~ (param_fields.description | trim) ~ '</description>' }}
                    {%- endif %}
                    {%- set handled_keys = ['name', 'type', 'description'] %}
                    {{- render_extra_keys(param_fields, handled_keys) }}
                    {{- '\\n</parameter>' }}
                {%- endfor %}
            {%- endif %}
            {% set handled_keys = ['type', 'properties'] %}
            {{- render_extra_keys(tool.parameters, handled_keys) }}
            {{- '\\n</parameters>' }}
            {%- set handled_keys = ['type', 'name', 'description', 'parameters'] %}
            {{- render_extra_keys(tool, handled_keys) }}
            {{- '\\n</function>' }}
        {%- endfor %}
        {{- "\\n</tools>" }}
        {{- '\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}
    {%- endif %}
    {%- if system_message is defined %}
        {{- '<|im_end|>\\n' }}
    {%- else %}
        {%- if tools is iterable and tools | length > 0 %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
    {%- for message in loop_messages %}
        {%- if message.role == "assistant" and message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 %}
            {{- '<|im_start|>' + message.role }}
            {%- if message.content is defined and message.content is string and message.content | trim | length > 0 %}
                {{- '\\n' + message.content | trim + '\\n' }}
            {%- endif %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\\n' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '\\n</parameter>\\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\\n</tool_call>' }}
            {%- endfor %}
            {{- '<|im_end|>\\n' }}
        {%- elif message.role == "user" or message.role == "system" or message.role == "assistant" %}
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
        {%- elif message.role == "tool" %}
            {%- if loop.previtem and loop.previtem.role != "tool" %}
                {{- '<|im_start|>user\\n' }}
            {%- endif %}
            {{- '<tool_response>\\n' }}
            {{- message.content }}
            {{- '\\n</tool_response>\\n' }}
            {%- if not loop.last and loop.nextitem.role != "tool" %}
                {{- '<|im_end|>\\n' }}
            {%- elif loop.last %}
                {{- '<|im_end|>\\n' }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n' }}
    {%- endif %}
    """

    /// vLLM's tool_chat_template_hermes.jinja — ChatML format with <tool_call> JSON wrapping.
    /// Source: https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_hermes.jinja
    static let hermesTemplate = """
    {%- macro json_to_python_type(json_spec) %}
        {%- set basic_type_map = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool"
    } %}
        {%- if basic_type_map[json_spec.type] is defined %}
            {{- basic_type_map[json_spec.type] }}
        {%- elif json_spec.type == "array" %}
            {{- "list[" +  json_to_python_type(json_spec|items) + "]" }}
        {%- elif json_spec.type == "object" %}
            {%- if json_spec.additionalProperties is defined %}
                {{- "dict[str, " + json_to_python_type(json_spec.additionalProperties) + ']' }}
            {%- else %}
                {{- "dict" }}
            {%- endif %}
        {%- elif json_spec.type is iterable %}
            {{- "Union[" }}
            {%- for t in json_spec.type %}
                {{- json_to_python_type({"type": t}) }}
                {%- if not loop.last %}
                    {{- "," }}
                {%- endif %}
            {%- endfor %}
            {{- "]" }}
        {%- else %}
            {{- "Any" }}
        {%- endif %}
    {%- endmacro %}

    {{- bos_token }}
    {{- "<|im_start|>system\\nYou are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> " }}
    {%- if tools is iterable and tools | length > 0 %}
        {%- for tool in tools %}
            {%- if tool.function is defined %}
                {%- set tool = tool.function %}
            {%- endif %}
            {{- '{"type": "function", "function": ' }}
            {{- '{"name": "' + tool.name + '", ' }}
            {{- '"description": "' + tool.name + '(' }}
            {%- for param_name, param_fields in tool.parameters.properties|items %}
                {{- param_name + ": " + json_to_python_type(param_fields) }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- ")" }}
            {%- if tool.return is defined %}
                {{- " -> " + json_to_python_type(tool.return) }}
            {%- endif %}
            {{- " - " + tool.description + "\\n\\n" }}
            {%- for param_name, param_fields in tool.parameters.properties|items %}
                {%- if loop.first %}
                    {{- "    Args:\\n" }}
                {%- endif %}
                {{- "        " + param_name + "(" + json_to_python_type(param_fields) + "): " + param_fields.description|trim }}
            {%- endfor %}
            {%- if tool.return is defined and tool.return.description is defined %}
                {{- "\\n    Returns:\\n        " + tool.return.description }}
            {%- endif %}
            {{- '"' }}
            {{- ', "parameters": ' }}
            {%- if tool.parameters.properties | length == 0 %}
                {{- "{}" }}
            {%- else %}
                {{- tool.parameters|tojson }}
            {%- endif %}
            {{- "}" }}
            {%- if not loop.last %}
                {{- "\\n" }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- " </tools>" }}
    {{- 'Use the following pydantic model json schema for each tool call you will make: {"properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"title": "Arguments", "type": "object"}}, "required": ["name", "arguments"], "title": "FunctionCall", "type": "object"}\\n' }}
    {{- "For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\\n" }}
    {{- "<tool_call>\\n" }}
    {{- '{"name": <function-name>, "arguments": <args-dict>}\\n' }}
    {{- '</tool_call><|im_end|>' }}
    {%- for message in messages %}
        {%- if message.role == "user" or message.role == "system" or (message.role == "assistant" and message.tool_calls is not defined) %}
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
        {%- elif message.role == "assistant" and message.tool_calls is defined %}
            {{- '<|im_start|>' + message.role }}
            {%- for tool_call in message.tool_calls %}
                {{- '\\n<tool_call>\\n' }}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '{' }}
                {{- '"name": "' }}
                {{- tool_call.name }}
                {{- '"' }}
                {%- if tool_call.arguments is defined %}
                    {{- ', ' }}
                    {{- '"arguments": ' }}
                    {{- tool_call.arguments|tojson }}
                {%- endif %}
                {{- '}' }}
                {{- '\\n</tool_call>' }}
            {%- endfor %}
            {{- '<|im_end|>\\n' }}
        {%- elif message.role == "tool" %}
            {%- if loop.previtem and loop.previtem.role != "tool" %}
                {{- '<|im_start|>tool\\n' }}
            {%- endif %}
            {{- '<tool_response>\\n' }}
            {{- message.content }}
            {%- if not loop.last %}
                {{- '\\n</tool_response>\\n' }}
            {%- else %}
                {{- '\\n</tool_response>' }}
            {%- endif %}
            {%- if not loop.last and loop.nextitem.role != "tool" %}
                {{- '<|im_end|>' }}
            {%- elif loop.last %}
                {{- '<|im_end|>' }}
            {%- endif %}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\\n' }}
    {%- endif %}
    """

    /// Adapted from vLLM's tool_chat_template_llama3.1_json.jinja — Llama 3.1/3.3 format.
    /// Modifications: wraps tool calls in <tool_call>/<\/tool_call> for streaming detection,
    /// uses "arguments" key, removes raise_exception, hardcodes date fallback.
    /// Source: https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.1_json.jinja
    static let llama3JSONTemplate = """
    {{- bos_token }}
    {%- if custom_tools is defined %}
        {%- set tools = custom_tools %}
    {%- endif %}
    {%- if not tools_in_user_message is defined %}
        {%- set tools_in_user_message = true %}
    {%- endif %}
    {%- if not date_string is defined %}
        {%- set date_string = "23 Feb 2026" %}
    {%- endif %}
    {%- if not tools is defined %}
        {%- set tools = none %}
    {%- endif %}

    {%- if messages[0]['role'] == 'system' %}
        {%- if messages[0]['content'] is string %}
            {%- set system_message = messages[0]['content']|trim %}
        {%- else %}
            {%- set system_message = messages[0]['content'][0]['text']|trim %}
        {%- endif %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {%- if tools is not none %}
            {%- set system_message = "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question." %}
        {%- else %}
            {%- set system_message = "" %}
        {%- endif %}
    {%- endif %}

    {{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}
    {%- if tools is not none %}
        {{- "Environment: ipython\\n" }}
    {%- endif %}
    {{- "Cutting Knowledge Date: December 2023\\n" }}
    {{- "Today Date: " + date_string + "\\n\\n" }}
    {%- if tools is not none and not tools_in_user_message %}
        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call " }}
        {{- 'wrapped in <tool_call></tool_call> tags with the keys "name" and "arguments".\\n\\n' }}
        {%- for t in tools %}
            {{- t | tojson(indent=4) }}
            {{- "\\n\\n" }}
        {%- endfor %}
    {%- endif %}
    {{- system_message }}
    {{- "<|eot_id|>" }}

    {%- if tools_in_user_message and not tools is none %}
        {%- if messages | length != 0 %}
            {%- if messages[0]['content'] is string %}
                {%- set first_user_message = messages[0]['content']|trim %}
            {%- else %}
                {%- set first_user_message = messages[0]['content'] | selectattr('type', 'equalto', 'text') | map(attribute='text') | map('trim') | join('\\n') %}
            {%- endif %}
            {%- set messages = messages[1:] %}
        {%- else %}
            {%- set first_user_message = "" %}
        {%- endif %}
        {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}
        {{- "Given the following functions, please respond with a JSON for a function call " }}
        {{- 'wrapped in <tool_call></tool_call> tags with the keys "name" and "arguments".\\n\\n' }}
        {%- for t in tools %}
            {{- t | tojson(indent=4) }}
            {{- "\\n\\n" }}
        {%- endfor %}
        {{- first_user_message + "<|eot_id|>"}}
    {%- endif %}

    {%- for message in messages %}
        {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}
            {%- if message['content'] is string %}
                {{- message['content'] | trim}}
            {%- else %}
                {%- for content in message['content'] %}
                    {%- if content['type'] == 'text' %}
                        {{- content['text'] | trim }}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {{- '<|eot_id|>' }}
        {%- elif 'tool_calls' in message %}
            {%- set tool_call = message.tool_calls[0].function %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
            {{- '<tool_call>\\n' }}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n' }}
            {{- '</tool_call>' }}
            {{- "<|eot_id|>" }}
        {%- elif message.role == "tool" or message.role == "ipython" %}
            {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}
            {%- if message.content is string %}
                {{- message.content }}
            {%- else %}
                {%- for content in message['content'] %}
                    {%- if content['type'] == 'text' %}
                        {{- content['text'] }}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
    {%- endif %}
    """

    /// Adapted from vLLM's tool_chat_template_mistral.jinja — Mistral v7 format.
    /// Modifications: wraps tool calls in <tool_call>/<\/tool_call> instead of [TOOL_CALLS],
    /// removes raise_exception calls, simplified tool_call_id handling.
    /// Source: https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_mistral.jinja
    static let mistralTemplate = """
    {%- if messages[0]["role"] == "system" %}
        {%- set system_message = messages[0]["content"] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set loop_messages = messages %}
    {%- endif %}
    {%- if not tools is defined %}
        {%- set tools = none %}
    {%- endif %}
    {%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}

    {{- bos_token }}
    {%- for message in loop_messages %}
        {%- if message["role"] == "user" %}
            {%- if tools is not none and (message == user_messages[-1]) %}
                {{- "[AVAILABLE_TOOLS] [" }}
                {%- for tool in tools %}
                    {%- set tool = tool.function %}
                    {{- '{"type": "function", "function": {' }}
                    {%- for key, val in tool.items() if key != "return" %}
                        {%- if val is string %}
                            {{- '"' + key + '": "' + val + '"' }}
                        {%- else %}
                            {{- '"' + key + '": ' + val|tojson }}
                        {%- endif %}
                        {%- if not loop.last %}
                            {{- ", " }}
                        {%- endif %}
                    {%- endfor %}
                    {{- "}}" }}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- else %}
                        {{- "]" }}
                    {%- endif %}
                {%- endfor %}
                {{- "[/AVAILABLE_TOOLS]" }}
            {%- endif %}
            {%- if loop.last and system_message is defined %}
                {{- "[INST] " + system_message + "\\n\\n" + message["content"] + "[/INST]" }}
            {%- else %}
                {{- "[INST] " + message["content"] + "[/INST]" }}
            {%- endif %}
        {%- elif message["role"] == "tool_calls" or message.tool_calls is defined %}
            {%- if message.tool_calls is defined %}
                {%- set tool_calls = message.tool_calls %}
            {%- else %}
                {%- set tool_calls = message.content %}
            {%- endif %}
            {%- for tool_call in tool_calls %}
                {{- "\\n<tool_call>\\n" }}
                {%- if tool_call.function is defined %}
                    {{- tool_call.function|tojson }}
                {%- else %}
                    {{- tool_call|tojson }}
                {%- endif %}
                {{- "\\n</tool_call>" }}
            {%- endfor %}
            {{- eos_token }}
        {%- elif message["role"] == "assistant" %}
            {{- " " + message["content"] + eos_token }}
        {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
            {%- if message.content is defined and message.content.content is defined %}
                {%- set content = message.content.content %}
            {%- else %}
                {%- set content = message.content %}
            {%- endif %}
            {{- '[TOOL_RESULTS] {"content": ' + content|string + '}[/TOOL_RESULTS]' }}
        {%- endif %}
    {%- endfor %}
    """

}
