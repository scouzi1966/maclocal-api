import Vapor
import Foundation
import MLX
import MLXLMCommon

struct FinalizedAssistantTurn {
    let finishReason: String
    let content: String?
    let reasoningContent: String?
    let toolCalls: [ResponseToolCall]?
}

struct MLXChatCompletionsController: RouteCollection {
    private let streamingEnabled: Bool
    private let modelID: String
    private let service: any MLXChatServing
    private let temperature: Double?
    private let topP: Double?
    private let maxTokens: Int?
    private let repetitionPenalty: Double?
    private let topK: Int?
    private let minP: Double?
    private let presencePenalty: Double?
    private let seed: Int?
    private let maxLogprobs: Int
    private let veryVerbose: Bool
    private let trace: Bool
    private let rawOutput: Bool
    private let stop: String?

    init(
        streamingEnabled: Bool = true,
        modelID: String,
        service: any MLXChatServing,
        temperature: Double?,
        topP: Double? = nil,
        maxTokens: Int? = nil,
        repetitionPenalty: Double?,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        maxLogprobs: Int = 20,
        veryVerbose: Bool = false,
        trace: Bool = false,
        rawOutput: Bool = false,
        stop: String? = nil
    ) {
        self.streamingEnabled = streamingEnabled
        self.modelID = modelID
        self.service = service
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.topK = topK
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.maxLogprobs = maxLogprobs
        self.veryVerbose = veryVerbose
        self.trace = trace
        self.rawOutput = rawOutput
        self.stop = stop
    }

    /// Merge CLI --stop sequences with API-level stop sequences, deduplicating.
    private func mergeStopSequences(cliStop: String?, apiStop: [String]?) -> [String]? {
        var merged: [String] = []
        if let cliStopString = cliStop {
            let cliArray = cliStopString.split(separator: ",").map { String($0.trimmingCharacters(in: .whitespaces)) }
            merged.append(contentsOf: cliArray)
        }
        if let apiArray = apiStop { merged.append(contentsOf: apiArray) }
        guard !merged.isEmpty else { return nil }
        var seen = Set<String>()
        return merged.filter { seen.insert($0).inserted }
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "chat", "completions", body: .collect(maxSize: "100mb"), use: chatCompletions)
        v1.on(.OPTIONS, "chat", "completions", use: handleOptions)
    }

    func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization, X-AFM-Profile")
        return response
    }

    func chatCompletions(req: Request) async throws -> Response {
        do {
            let chatRequest = try req.content.decode(ChatCompletionRequest.self)
            if veryVerbose {
                print("\(Self.pink)[\(Self.timestamp())] RECV MLX full request:\n\(encodeJSON(chatRequest))\(Self.reset)"); fflush(stdout)
                if let lastUser = chatRequest.messages.last(where: { $0.role == "user" }) {
                    let prompt = lastUser.textContent
                    let truncated = prompt.count > 500 ? String(prompt.prefix(500)) + "..." : prompt
                    print("\(Self.red)[\(Self.timestamp())] RECV MLX user prompt:\n  \(truncated)\(Self.reset)"); fflush(stdout)
                }
            }
            // Detect strict-mode downgrade: user requested grammar enforcement but admin didn't enable the engine
            let strictRequested = MLXModelService.hasStrictSchema(chatRequest.responseFormat)
                || MLXModelService.hasStrictTools(chatRequest.tools)
            let grammarDowngraded = strictRequested && !service.enableGrammarConstraints

            guard !chatRequest.messages.isEmpty else {
                return try await createErrorResponse(req: req, error: OpenAIError(message: "At least one message is required"), status: .badRequest)
            }

            // Validate top_logprobs against server max (vLLM-compatible)
            if let requestedTopLogprobs = chatRequest.topLogprobs, requestedTopLogprobs > maxLogprobs {
                return try await createErrorResponse(
                    req: req,
                    error: OpenAIError(
                        message: "top_logprobs must be <= \(maxLogprobs). Received \(requestedTopLogprobs). Use --max-logprobs to increase the server limit.",
                        type: "invalid_request_error"
                    ),
                    status: .badRequest
                )
            }

            if let requestedModelRaw = chatRequest.model?.trimmingCharacters(in: .whitespacesAndNewlines),
               !requestedModelRaw.isEmpty,
               service.normalizeModel(requestedModelRaw) != modelID {
                // WebUI may send transformed model identifiers; afm mlx always serves the active model.
                print("[\(Self.timestamp())] MLX request model '\(requestedModelRaw)' does not match active model '\(modelID)'; serving active model"); fflush(stdout)
            }

            let effectiveTools = try Self.resolveEffectiveTools(
                chatRequest.tools,
                toolChoice: chatRequest.toolChoice
            )

            let hasTools = effectiveTools != nil && !(effectiveTools?.isEmpty ?? true)
            if hasTools && veryVerbose {
                let toolNames = chatRequest.tools!.map { $0.function.name }.joined(separator: ", ")
                print("\(Self.gold)[\(Self.timestamp())] RECV tools: [\(toolNames)]\(Self.reset)")
            }
            // -VV: Full tool schemas as received from client
            if hasTools && trace {
                for tool in effectiveTools! {
                    let schemaJSON: String
                    if let params = tool.function.parameters,
                       let data = try? JSONSerialization.data(withJSONObject: params.toJinjaCompatible(), options: [.prettyPrinted, .sortedKeys]),
                       let str = String(data: data, encoding: .utf8) {
                        schemaJSON = str
                    } else {
                        schemaJSON = "(no parameters)"
                    }
                    print("\(Self.cyan)[\(Self.timestamp())] [VV] RECV tool schema: \(tool.function.name)\n\(schemaJSON)\(Self.reset)")
                }
                fflush(stdout)
            }

            // Atomically reserve a concurrent slot (check + increment in one lock).
            // Serial mode always returns true. Returns 503 if at capacity.
            guard service.tryReserveSlot() else {
                let peer = req.peerAddress?.description ?? "unknown"
                let ua = req.headers.first(name: .userAgent) ?? "unknown"
                req.logger.warning("Connection refused: at capacity (\(service.maxConcurrent)/\(service.maxConcurrent)) — client=\(peer) ua=\(ua)")
                let response = Response(status: .serviceUnavailable)
                response.headers.add(name: .contentType, value: "application/json")
                response.headers.add(name: .accessControlAllowOrigin, value: "*")
                response.headers.add(name: "Retry-After", value: "2")
                try response.content.encode(OpenAIError(
                    message: "Server at capacity (\(service.maxConcurrent) concurrent requests). Please retry shortly.",
                    type: "server_busy"
                ))
                return response
            }

            // Reset peak memory before each request so usage.peak_memory_gib
            // reflects this request only (matches mlx_lm's mx.reset_peak_memory())
            GPU.resetPeakMemory()

            let isWebUI = req.headers.first(name: .origin) != nil
            let extractThinking = !rawOutput || isWebUI

            if chatRequest.stream == true && streamingEnabled {
                return try await createStreamingResponse(req: req, chatRequest: chatRequest, extractThinking: extractThinking, grammarDowngraded: grammarDowngraded)
            }

            // In concurrent mode, non-streaming requests currently bypass the
            // BatchScheduler decode loop, so the controller must release the
            // reservation itself. Streaming requests are released by the
            // scheduler when the stream finishes.
            defer { service.releaseSlot() }

            // AFM Profile: start GPU monitoring if client requests it
            let profileHeader = req.headers.first(name: "X-AFM-Profile")?.lowercased()
            let wantProfile = profileHeader == "true" || profileHeader == "extended"
            let wantExtended = profileHeader == "extended"
            if wantProfile { service.startAPIProfile() }

            let effectiveTemp = chatRequest.temperature ?? temperature
            let effectiveTopP = chatRequest.topP ?? topP
            let effectiveMaxTokens = normalizedMaxTokens(chatRequest.effectiveMaxTokens)
            let effectiveRepetitionPenalty = chatRequest.effectiveRepetitionPenalty ?? repetitionPenalty
            let effectiveTopK = chatRequest.topK ?? topK
            let effectiveMinP = chatRequest.minP ?? minP
            let effectivePresencePenalty = chatRequest.presencePenalty ?? presencePenalty
            let effectiveSeed = chatRequest.seed ?? seed
            let effectiveStop = mergeStopSequences(cliStop: stop, apiStop: chatRequest.stop)
            let started = Date()
            if veryVerbose {
                let promptChars = chatRequest.messages.map { $0.textContent.count }.reduce(0, +)
                let stopDesc = effectiveStop.map { $0.map { $0.debugDescription }.joined(separator: ", ") }
                print(
                    "\(Self.orange)[\(Self.timestamp())] MLX start: stream=false\n  prompt_chars=\(promptChars) max_tokens=\(effectiveMaxTokens)\n  temperature=\(effectiveTemp?.description ?? "default") top_p=\(effectiveTopP?.description ?? "default") rep_penalty=\(effectiveRepetitionPenalty?.description ?? "none")\n  top_k=\(effectiveTopK?.description ?? "none") min_p=\(effectiveMinP?.description ?? "none") presence_penalty=\(effectivePresencePenalty?.description ?? "none")\n  seed=\(effectiveSeed?.description ?? "none") stop=\(stopDesc ?? "none")\(Self.reset)"
                ); fflush(stdout)
            }
            let result: ChatGenerationResult
            if service.maxConcurrent >= 2 {
                // Batch mode: route through scheduler for batched decode
                let streamResult: ChatStreamingResult = try await service.generateStreaming(
                    model: modelID,
                    messages: chatRequest.messages,
                    temperature: effectiveTemp,
                    maxTokens: effectiveMaxTokens,
                    topP: effectiveTopP,
                    repetitionPenalty: effectiveRepetitionPenalty,
                    topK: effectiveTopK,
                    minP: effectiveMinP,
                    presencePenalty: effectivePresencePenalty,
                    seed: effectiveSeed,
                    logprobs: chatRequest.logprobs,
                    topLogprobs: chatRequest.topLogprobs,
                    tools: effectiveTools,
                    stop: effectiveStop,
                    responseFormat: chatRequest.responseFormat,
                    chatTemplateKwargs: chatRequest.chatTemplateKwargs
                )

                // Collect stream into complete response
                var fullText = ""
                var allLogprobs: [ResolvedLogprob] = []
                var finalToolCalls: [ResponseToolCall]? = nil
                var promptTokens = streamResult.promptTokens
                var completionTokens = 0
                var cachedTokens = 0
                var promptTime: Double = 0
                var generateTime: Double = 0
                var stoppedBySequence = false

                for try await chunk in streamResult.stream {
                    fullText += chunk.text
                    if let lp = chunk.logprobs { allLogprobs.append(contentsOf: lp) }
                    if let tc = chunk.toolCalls { finalToolCalls = tc }
                    if let pt = chunk.promptTokens { promptTokens = pt }
                    if let ct = chunk.completionTokens { completionTokens = ct }
                    if let cached = chunk.cachedTokens { cachedTokens = cached }
                    if let pt = chunk.promptTime { promptTime = pt }
                    if let gt = chunk.generateTime { generateTime = gt }
                    if let sbs = chunk.stoppedBySequence { stoppedBySequence = sbs }
                }

                // Fallback tool call parsing: if no tool calls were detected by
                // streaming runtime but content contains tool call patterns, try
                // full-text parsing (handles missing <tool_call> wrapper, etc.)
                if finalToolCalls == nil && chatRequest.tools != nil && !fullText.isEmpty {
                    let trimmed = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
                    let parserName = self.service.toolCallParser ?? "auto"
                    let looksLikeToolCall =
                        fullText.contains("<function=") ||
                        fullText.contains("<tool_call>") ||
                        fullText.contains("[TOOL_CALLS]") ||
                        fullText.contains("[ARGS]") ||
                        (trimmed.hasPrefix("{") && trimmed.contains("\"name\""))
                    if looksLikeToolCall {
                        print("\(Self.gold)[\(Self.timestamp())] [ToolCallParser] Post-generation parse (parser=\(parserName))\(Self.reset)")
                        fflush(stdout)
                        let (parsed, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
                            from: fullText,
                            toolCallParser: self.service.toolCallParser,
                            tools: chatRequest.tools
                        )
                        if !parsed.isEmpty {
                            let names = parsed.map { $0.function.name }.joined(separator: ", ")
                            print("\(Self.gold)[\(Self.timestamp())] [ToolCallParser] Parsed \(parsed.count) call(s): \(names) (parser=\(parserName))\(Self.reset)")
                            fflush(stdout)
                            finalToolCalls = MLXModelService.normalizeToolCalls(
                                parsed,
                                tools: chatRequest.tools,
                                fixToolArgs: service.fixToolArgs
                            )
                            fullText = remaining
                        }
                    }
                }

                result = (
                    modelID: streamResult.modelID,
                    content: fullText,
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    tokenLogprobs: allLogprobs.isEmpty ? nil : allLogprobs,
                    toolCalls: finalToolCalls,
                    cachedTokens: cachedTokens,
                    promptTime: promptTime,
                    generateTime: generateTime,
                    stoppedBySequence: stoppedBySequence
                )
            } else {
                // Serial mode: use existing generate() path
                result = try await service.generate(
                    model: modelID,
                    messages: chatRequest.messages,
                    temperature: effectiveTemp,
                    maxTokens: effectiveMaxTokens,
                    topP: effectiveTopP,
                    repetitionPenalty: effectiveRepetitionPenalty,
                    topK: effectiveTopK,
                    minP: effectiveMinP,
                    presencePenalty: effectivePresencePenalty,
                    seed: effectiveSeed,
                    logprobs: chatRequest.logprobs,
                    topLogprobs: chatRequest.topLogprobs,
                    tools: effectiveTools,
                    stop: effectiveStop,
                    responseFormat: chatRequest.responseFormat,
                    chatTemplateKwargs: chatRequest.chatTemplateKwargs
                )
            }
            let completionTok = result.completionTokens
            let promptTime = result.promptTime
            let generateTime = result.generateTime
            let tokPerSec = generateTime > 0 ? Double(completionTok) / generateTime : 0
            let promptTokPerSec = promptTime > 0 ? Double(result.promptTokens) / promptTime : 0
            let finalizedTurn = Self.finalizeAssistantTurn(
                content: result.content,
                toolCalls: result.toolCalls,
                toolChoice: chatRequest.toolChoice,
                extractThinking: extractThinking,
                thinkStartTag: service.thinkStartTag ?? "<think>",
                thinkEndTag: service.thinkEndTag ?? "</think>",
                stoppedBySequence: result.stoppedBySequence,
                completionTokens: completionTok,
                maxTokens: effectiveMaxTokens,
                sanitizeContent: sanitizeDegenerateTail
            )

            // If we got tool calls, return a tool_calls response
            if let toolCalls = finalizedTurn.toolCalls, !toolCalls.isEmpty {
                if veryVerbose {
                    let toolNames = toolCalls.map { $0.function.name }.joined(separator: ", ")
                    print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=false\n  prompt_tokens=\(result.promptTokens) completion_tokens=\(completionTok)\n  prompt=\(String(format: "%.2f", promptTime))s gen=\(String(format: "%.2f", generateTime))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=tool_calls\(Self.reset)"); fflush(stdout)
                    for tc in toolCalls {
                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call: \(tc.function.name)\n  id=\(tc.id)\n  args=\(tc.function.arguments)\(Self.reset)")
                    }
                    fflush(stdout)
                }

                let choiceLogprobs = Self.buildChoiceLogprobs(result.tokenLogprobs)
                let timings = StreamTimings(prompt_n: result.promptTokens, prompt_ms: promptTime * 1000, predicted_n: completionTok, predicted_ms: generateTime * 1000)
                let extended = wantExtended ? service.stopAPIProfileExtended(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil
                let profile = wantExtended ? nil : (wantProfile ? service.stopAPIProfile(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil)
                let response = ChatCompletionResponse(
                    model: result.modelID,
                    toolCalls: toolCalls,
                    logprobs: choiceLogprobs,
                    promptTokens: result.promptTokens,
                    completionTokens: completionTok,
                    cachedTokens: result.cachedTokens,
                    completionTime: generateTime,
                    promptTime: promptTime,
                    timings: timings,
                    afmProfile: profile,
                    afmProfileExtended: extended
                )
                let cacheInfo1 = Self.cacheStatsSummary(
                    cachedTokens: result.cachedTokens,
                    totalPromptTokens: result.promptTokens
                )
                print("\(Self.orange)[\(Self.timestamp())] [STATS] pp: \(result.promptTokens) tok, \(String(format: "%.2f", promptTime))s (\(String(format: "%.1f", promptTokPerSec)) tok/s) | tg: \(completionTok) tok, \(String(format: "%.2f", generateTime))s (\(String(format: "%.1f", tokPerSec)) tok/s)\(cacheInfo1) stream=false\(Self.reset)")
                let tcSummary = toolCalls.map { "\($0.function.name)(\(Self.argKeysPreview($0.function.arguments)))" }.joined(separator: ", ")
                print("\(Self.gold)[\(Self.timestamp())] [TOOL_CALLS] \(toolCalls.count) call(s): \(tcSummary)\(Self.reset)")
                fflush(stdout)
                if veryVerbose {
                    print("\(Self.teal)[\(Self.timestamp())] SEND full response:\n\(encodeJSON(response))\(Self.reset)"); fflush(stdout)
                }
                // -VV: Non-streaming tool call details
                if trace {
                    for tc in toolCalls {
                        print("\(Self.cyan)[\(Self.timestamp())] [VV] SEND→CLIENT (non-stream) tool_call \(tc.function.name):\n  \(tc.function.arguments)\(Self.reset)")
                    }
                    fflush(stdout)
                }
                return try await createSuccessResponse(req: req, response: response, grammarDowngraded: grammarDowngraded)
            }

            let stopReason = finalizedTurn.finishReason
            if veryVerbose {
                print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=false\n  prompt_tokens=\(result.promptTokens) completion_tokens=\(completionTok)\n  prompt=\(String(format: "%.2f", promptTime))s gen=\(String(format: "%.2f", generateTime))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=\(stopReason)\(Self.reset)"); fflush(stdout)
            }

            let choiceLogprobs = Self.buildChoiceLogprobs(result.tokenLogprobs)
            let timings = StreamTimings(prompt_n: result.promptTokens, prompt_ms: promptTime * 1000, predicted_n: completionTok, predicted_ms: generateTime * 1000)
            let extended = wantExtended ? service.stopAPIProfileExtended(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil
            let profile = wantExtended ? nil : (wantProfile ? service.stopAPIProfile(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil)
            let response = ChatCompletionResponse(
                model: result.modelID,
                content: finalizedTurn.content ?? "",
                reasoningContent: finalizedTurn.reasoningContent,
                logprobs: choiceLogprobs,
                finishReason: stopReason,
                promptTokens: result.promptTokens,
                completionTokens: completionTok,
                cachedTokens: result.cachedTokens,
                completionTime: generateTime,
                promptTime: promptTime,
                timings: timings,
                afmProfile: profile,
                afmProfileExtended: extended
            )
            let cacheInfo2 = Self.cacheStatsSummary(
                cachedTokens: result.cachedTokens,
                totalPromptTokens: result.promptTokens
            )
            print("\(Self.orange)[\(Self.timestamp())] [STATS] pp: \(result.promptTokens) tok, \(String(format: "%.2f", promptTime))s (\(String(format: "%.1f", promptTokPerSec)) tok/s) | tg: \(completionTok) tok, \(String(format: "%.2f", generateTime))s (\(String(format: "%.1f", tokPerSec)) tok/s)\(cacheInfo2) stream=false\(Self.reset)")
            fflush(stdout)
            if veryVerbose {
                print("\(Self.teal)[\(Self.timestamp())] SEND full response:\n\(encodeJSON(response))\(Self.reset)"); fflush(stdout)
            }
            return try await createSuccessResponse(req: req, response: response, grammarDowngraded: grammarDowngraded)
        } catch let abort as Abort {
            req.logger.error("[\(Self.timestamp())] MLX completions error: \(abort)")
            return try await createErrorResponse(
                req: req,
                error: OpenAIError(
                    message: abort.reason,
                    type: abort.status == .badRequest ? "invalid_request_error" : "mlx_error"
                ),
                status: abort.status
            )
        } catch {
            req.logger.error("[\(Self.timestamp())] MLX completions error: \(error)")
            return try await createErrorResponse(req: req, error: OpenAIError(message: error.localizedDescription, type: "mlx_error"), status: .badRequest)
        }
    }

    private func createStreamingResponse(req: Request, chatRequest: ChatCompletionRequest, extractThinking: Bool, grammarDowngraded: Bool = false) async throws -> Response {
        let httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "text/event-stream")
        httpResponse.headers.add(name: .cacheControl, value: "no-cache")
        httpResponse.headers.add(name: .connection, value: "keep-alive")
        httpResponse.headers.add(name: "Access-Control-Allow-Origin", value: "*")
        httpResponse.headers.add(name: "Access-Control-Allow-Headers", value: "Content-Type, X-AFM-Profile")
        httpResponse.headers.add(name: "X-Accel-Buffering", value: "no")
        if grammarDowngraded {
            httpResponse.headers.add(name: "X-Grammar-Constraints", value: "downgraded")
        }

        let streamId = UUID().uuidString

        // AFM Profile: start GPU monitoring if client requests it
        let streamProfileHeader = req.headers.first(name: "X-AFM-Profile")?.lowercased()
        let wantStreamProfile = streamProfileHeader == "true" || streamProfileHeader == "extended"
        let wantStreamExtended = streamProfileHeader == "extended"
        if wantStreamProfile { service.startAPIProfile() }

        httpResponse.body = .init(asyncStream: { writer in
            let encoder = JSONEncoder()
            var fullContent = ""
            let started = Date()
            let effectiveTemp = chatRequest.temperature ?? self.temperature
            let effectiveTopP = chatRequest.topP ?? self.topP
            let effectiveMaxTokens = self.normalizedMaxTokens(chatRequest.effectiveMaxTokens)
            let effectiveRepetitionPenalty = chatRequest.effectiveRepetitionPenalty ?? self.repetitionPenalty
            let effectiveTopK = chatRequest.topK ?? self.topK
            let effectiveMinP = chatRequest.minP ?? self.minP
            let effectivePresencePenalty = chatRequest.presencePenalty ?? self.presencePenalty
            let effectiveSeed = chatRequest.seed ?? self.seed
            let effectiveStop = self.mergeStopSequences(cliStop: self.stop, apiStop: chatRequest.stop)

            do {
                let effectiveTools = try Self.resolveEffectiveTools(
                    chatRequest.tools,
                    toolChoice: chatRequest.toolChoice
                )
                if self.veryVerbose {
                    let promptChars = chatRequest.messages.map { $0.textContent.count }.reduce(0, +)
                    let stopDesc = effectiveStop.map { $0.map { $0.debugDescription }.joined(separator: ", ") }
                    print(
                        "\(Self.orange)[\(Self.timestamp())] MLX start: stream=true\n  prompt_chars=\(promptChars) max_tokens=\(effectiveMaxTokens)\n  temperature=\(effectiveTemp?.description ?? "default") top_p=\(effectiveTopP?.description ?? "default") rep_penalty=\(effectiveRepetitionPenalty?.description ?? "none")\n  top_k=\(effectiveTopK?.description ?? "none") min_p=\(effectiveMinP?.description ?? "none") presence_penalty=\(effectivePresencePenalty?.description ?? "none")\n  seed=\(effectiveSeed?.description ?? "none") stop=\(stopDesc ?? "none")\(Self.reset)"
                    ); fflush(stdout)
                }
                let res = try await service.generateStreaming(
                    model: modelID,
                    messages: chatRequest.messages,
                    temperature: effectiveTemp,
                    maxTokens: effectiveMaxTokens,
                    topP: effectiveTopP,
                    repetitionPenalty: effectiveRepetitionPenalty,
                    topK: effectiveTopK,
                    minP: effectiveMinP,
                    presencePenalty: effectivePresencePenalty,
                    seed: effectiveSeed,
                    logprobs: chatRequest.logprobs,
                    topLogprobs: chatRequest.topLogprobs,
                    tools: effectiveTools,
                    stop: effectiveStop,
                    responseFormat: chatRequest.responseFormat,
                    chatTemplateKwargs: chatRequest.chatTemplateKwargs
                )
                // Emit an initial assistant delta so clients always open a response container.
                let initialChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: res.modelID,
                    content: "",
                    isFirst: true
                )
                let initialData = try encoder.encode(initialChunk)
                if let jsonString = String(data: initialData, encoding: .utf8) {
                    try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                }

                // State for <think> tag extraction (Qwen, DeepSeek R1, etc.)
                var insideThinkBlock = false
                var thinkBuffer = ""
                var verboseReasoningBuf = ""
                var verboseContentBuf = ""
                var logprobBuffer = [ResolvedLogprob]()
                var collectedToolCalls = [ResponseToolCall]()
                var hasToolCalls = false
                let allowedToolName: String? = {
                    if case .function(let functionChoice) = chatRequest.toolChoice {
                        return functionChoice.function.name
                    }
                    return nil
                }()
                var permittedToolIndices = Set<Int>()
                var stoppedBySequence = false
                var realPromptTokens: Int? = nil
                var realCompletionTokens: Int? = nil
                var realCachedTokens: Int? = nil
                var realPromptTime: Double? = nil
                var realGenerateTime: Double? = nil

                // Token-level tool call detection (mlx-lm style).
                // Instead of buffering ALL content when tools are present, detect
                // tool call start/end tags per-token. Content outside tool calls
                // streams normally; only the tool call body is buffered and parsed.
                let toolCallStartTag = res.toolCallStartTag
                let toolCallEndTag = res.toolCallEndTag
                let thinkStartTag = res.thinkStartTag
                let thinkEndTag = res.thinkEndTag
                let toolRuntime = (toolCallStartTag != nil && toolCallEndTag != nil) ? ToolCallStreamingRuntime(
                    toolCallStartTag: toolCallStartTag!,
                    toolCallEndTag: toolCallEndTag!,
                    toolCallParser: self.service.toolCallParser,
                    tools: effectiveTools,
                    applyFixToolArgs: { rtc in
                        return self.applyFixToolArgs(rtc, tools: effectiveTools)
                    },
                    remapSingleKey: { key, toolName in
                        return self.remapSingleKey(key, toolName: toolName, tools: effectiveTools)
                    }
                ) : nil
                let fallbackParamNameMapping = toolRuntime?.paramNameMapping ?? [:]

                var pendingRawTag: String? = nil
                for try await streamChunk in res.stream {
                    let piece = streamChunk.text

                    // Capture real token counts and timing from the info chunk
                    if let pt = streamChunk.promptTokens { realPromptTokens = pt }
                    if let ct = streamChunk.completionTokens { realCompletionTokens = ct }
                    if let cached = streamChunk.cachedTokens { realCachedTokens = cached }
                    if streamChunk.stoppedBySequence == true { stoppedBySequence = true }
                    if let pt = streamChunk.promptTime { realPromptTime = pt }
                    if let gt = streamChunk.generateTime { realGenerateTime = gt }

                    if let deltas = streamChunk.toolCallDeltas, !deltas.isEmpty {
                        let filtered = deltas.filter {
                            Self.isToolDeltaAllowed(
                                $0,
                                toolChoice: chatRequest.toolChoice,
                                allowedFunctionName: allowedToolName,
                                permittedToolIndices: &permittedToolIndices
                            )
                        }
                        guard !filtered.isEmpty else { continue }
                        hasToolCalls = true
                        let tcChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            toolCalls: filtered
                        )
                        let tcData = try encoder.encode(tcChunk)
                        if let jsonString = String(data: tcData, encoding: .utf8) {
                            try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                        continue
                    }

                    // Handle tool call chunks from the vendor parser
                    if let tcs = streamChunk.toolCalls, !tcs.isEmpty {
                        for tc in tcs {
                            guard Self.isToolCallAllowed(
                                tc,
                                toolChoice: chatRequest.toolChoice,
                                allowedFunctionName: allowedToolName,
                                permittedToolIndices: &permittedToolIndices
                            ) else { continue }
                            hasToolCalls = true
                            collectedToolCalls.append(tc)
                            if self.veryVerbose {
                                print("\(Self.gold)[\(Self.timestamp())] SEND tool_call (vendor): \(tc.function.name)\n  id=\(tc.id)\n  args=\(tc.function.arguments)\(Self.reset)")
                                fflush(stdout)
                            }
                            let delta = StreamDeltaToolCall(
                                index: collectedToolCalls.count - 1,
                                id: tc.id,
                                type: tc.type,
                                function: StreamDeltaFunction(
                                    name: tc.function.name,
                                    arguments: tc.function.arguments
                                )
                            )
                            let tcChunk = ChatCompletionStreamResponse(
                                id: streamId,
                                model: res.modelID,
                                toolCalls: [delta]
                            )
                            let tcData = try encoder.encode(tcChunk)
                            if let jsonString = String(data: tcData, encoding: .utf8) {
                                try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                            }
                        }
                        continue
                    }

                    if let toolRuntime {
                        fullContent += piece
                        let runtimeOutput = toolRuntime.process(piece: piece)
                        if runtimeOutput.handled {
                            if runtimeOutput.events.contains(where: {
                                if case .started = $0 { return true }
                                return false
                            }), extractThinking && !thinkBuffer.isEmpty {
                                let flushed: String?
                                let flushedReasoning: String?
                                if insideThinkBlock {
                                    flushedReasoning = thinkBuffer
                                    flushed = nil
                                } else {
                                    flushed = thinkBuffer
                                    flushedReasoning = nil
                                }
                                thinkBuffer = ""
                                if flushed != nil || flushedReasoning != nil {
                                    let flushChunk = ChatCompletionStreamResponse(
                                        id: streamId,
                                        model: res.modelID,
                                        content: flushed ?? "",
                                        reasoningContent: flushedReasoning,
                                        isFirst: false
                                    )
                                    if let flushData = try? encoder.encode(flushChunk),
                                       let jsonString = String(data: flushData, encoding: .utf8) {
                                        try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                                    }
                                }
                            }

                            for event in runtimeOutput.events {
                                switch event {
                                case .started:
                                    break
                                case .appendCollected(let toolCall):
                                    hasToolCalls = true
                                    collectedToolCalls.append(toolCall)
                                case .replaceCollected(let index, let toolCall):
                                    hasToolCalls = true
                                    if index < collectedToolCalls.count {
                                        collectedToolCalls[index] = toolCall
                                    } else {
                                        collectedToolCalls.append(toolCall)
                                    }
                                case .delta(let delta):
                                    let tcChunk = ChatCompletionStreamResponse(
                                        id: streamId,
                                        model: res.modelID,
                                        toolCalls: [delta]
                                    )
                                    let tcData = try encoder.encode(tcChunk)
                                    if let jsonString = String(data: tcData, encoding: .utf8) {
                                        try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                                    }
                                }
                            }
                            hasToolCalls = hasToolCalls || toolRuntime.hasToolCalls
                            continue
                        }
                    }

                    if let lps = streamChunk.logprobs {
                        logprobBuffer.append(contentsOf: lps)
                    }
                    fullContent += piece

                    // Detect RAW think tags but defer logging until after extraction flush
                    let tst = thinkStartTag ?? "<think>"
                    let tet = thinkEndTag ?? "</think>"
                    if self.veryVerbose && (piece.contains(tst) || piece.contains(tet)) {
                        pendingRawTag = piece.debugDescription
                    }

                    if extractThinking {
                        // If the piece is exactly the think start tag (template-injected or
                        // model-generated), just flip the state without adding the literal
                        // tag to the buffer. Prevents double-tag leaks when the template
                        // injects a think tag and the model also generates one.
                        let trimmedPiece = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                        if let tst = thinkStartTag, trimmedPiece == tst && !insideThinkBlock {
                            insideThinkBlock = true
                        } else {
                            thinkBuffer += piece
                        }

                        let extracted = Self.extractThinkTags(
                            buffer: &thinkBuffer,
                            insideThinkBlock: &insideThinkBlock,
                            startTag: thinkStartTag ?? "<think>",
                            endTag: thinkEndTag ?? "</think>"
                        )

                        let emitContent = extracted.content
                        let emitReasoning = extracted.reasoning

                        let flushLogprobs = logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer)

                        // Emit a chunk whenever we have visible content, extracted reasoning,
                        // or buffered logprobs. Without this, per-token logprobs can be lost
                        // when detokenized content is still buffered for think-tag extraction.
                        let hasReasoning = emitReasoning != nil
                        let hasContent = emitContent != nil
                        if hasReasoning || hasContent || flushLogprobs != nil {
                            if self.veryVerbose {
                                if let r = emitReasoning { verboseReasoningBuf += r }
                                if let c = emitContent { verboseContentBuf += c }
                                // Flush verbose log on newlines or when buffer gets large
                                if verboseReasoningBuf.hasSuffix("\n") || verboseReasoningBuf.count > 200 {
                                    print("\(Self.purple)[\(Self.timestamp())] SEND reasoning:\n  \(verboseReasoningBuf)\(Self.reset)"); fflush(stdout)
                                    verboseReasoningBuf = ""
                                }
                                if verboseContentBuf.hasSuffix("\n") || verboseContentBuf.count > 200 {
                                    print("\(Self.teal)[\(Self.timestamp())] SEND content (chunk):\n  \(verboseContentBuf)\(Self.reset)"); fflush(stdout)
                                    verboseContentBuf = ""
                                }
                            }
                            logprobBuffer = []
                            let contentChunk = ChatCompletionStreamResponse(
                                id: streamId,
                                model: res.modelID,
                                content: emitContent ?? "",
                                reasoningContent: emitReasoning,
                                logprobs: flushLogprobs,
                                isFirst: false
                            )
                            let chunkData = try encoder.encode(contentChunk)
                            if let jsonString = String(data: chunkData, encoding: .utf8) {
                                try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                            }
                        }
                        // Log RAW think tag AFTER extracted reasoning/content is flushed
                        if let tag = pendingRawTag {
                            print("\(Self.purple)[\(Self.timestamp())] MLX RAW token: \(tag)\(Self.reset)")
                            fflush(stdout)
                            pendingRawTag = nil
                        }
                    } else {
                        let flushLogprobs = logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer)
                        logprobBuffer = []
                        let contentChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            content: piece,
                            logprobs: flushLogprobs,
                            isFirst: false
                        )
                        let chunkData = try encoder.encode(contentChunk)
                        if let jsonString = String(data: chunkData, encoding: .utf8) {
                            try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                    }
                }

                // Handle incomplete tool call (model hit max tokens mid-tool-call)
                if let toolRuntime {
                    let trailingEvents = toolRuntime.finishIncompleteToolCall()
                    for event in trailingEvents {
                        switch event {
                        case .started:
                            break
                        case .appendCollected(let toolCall):
                            guard Self.isToolCallAllowed(
                                toolCall,
                                toolChoice: chatRequest.toolChoice,
                                allowedFunctionName: allowedToolName,
                                permittedToolIndices: &permittedToolIndices
                            ) else { continue }
                            hasToolCalls = true
                            collectedToolCalls.append(toolCall)
                        case .replaceCollected(let index, let toolCall):
                            guard Self.isToolCallAllowed(
                                toolCall,
                                toolChoice: chatRequest.toolChoice,
                                allowedFunctionName: allowedToolName,
                                permittedToolIndices: &permittedToolIndices
                            ) else { continue }
                            hasToolCalls = true
                            if index < collectedToolCalls.count {
                                collectedToolCalls[index] = toolCall
                            } else {
                                collectedToolCalls.append(toolCall)
                            }
                        case .delta(let delta):
                            guard Self.isToolDeltaAllowed(
                                delta,
                                toolChoice: chatRequest.toolChoice,
                                allowedFunctionName: allowedToolName,
                                permittedToolIndices: &permittedToolIndices
                            ) else { continue }
                            let tcChunk = ChatCompletionStreamResponse(
                                id: streamId,
                                model: res.modelID,
                                toolCalls: [delta]
                            )
                            let tcData = try encoder.encode(tcChunk)
                            if let jsonString = String(data: tcData, encoding: .utf8) {
                                try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                            }
                        }
                    }
                    hasToolCalls = hasToolCalls || toolRuntime.hasToolCalls
                }
                if toolRuntime?.madeToolCall == true { hasToolCalls = true }

                // Post-loop fallback: if tools were present but no tool calls detected
                // by token-level matching, try full-content regex parsing.
                // This handles edge cases where tags aren't single tokens.
                let trimmedFull = fullContent.trimmingCharacters(in: .whitespacesAndNewlines)
                let looksLikeBareJsonToolCall = trimmedFull.hasPrefix("{") && trimmedFull.contains("\"name\"")
                let parserName = self.service.toolCallParser ?? "auto"
                if !hasToolCalls && (
                    (toolCallStartTag != nil && fullContent.contains(toolCallStartTag!)) ||
                    fullContent.contains("[TOOL_CALLS]") ||
                    fullContent.contains("[ARGS]") ||
                    (chatRequest.tools != nil && looksLikeBareJsonToolCall) ||
                    (chatRequest.tools != nil && fullContent.contains("<function="))
                ) {
                    print("\(Self.gold)[\(Self.timestamp())] [ToolCallParser] Post-generation parse (parser=\(parserName))\(Self.reset)")
                    fflush(stdout)
                    let (parsed, _) = ToolCallStreamingRuntime.parseCompletedToolCalls(
                        from: fullContent,
                        toolCallParser: self.service.toolCallParser,
                        tools: chatRequest.tools
                    )
                    if !parsed.isEmpty {
                        let fallbackNames = parsed.map { $0.function.name }.joined(separator: ", ")
                        print("\(Self.gold)[\(Self.timestamp())] [ToolCallParser] Parsed \(parsed.count) call(s): \(fallbackNames) (parser=\(parserName))\(Self.reset)")
                        fflush(stdout)
                        for rtc in MLXModelService.normalizeToolCalls(
                            parsed,
                            startIndex: collectedToolCalls.count,
                            paramNameMapping: fallbackParamNameMapping,
                            tools: chatRequest.tools,
                            fixToolArgs: service.fixToolArgs
                        ) {
                            guard Self.isToolCallAllowed(
                                rtc,
                                toolChoice: chatRequest.toolChoice,
                                allowedFunctionName: allowedToolName,
                                permittedToolIndices: &permittedToolIndices
                            ) else { continue }
                            hasToolCalls = true
                            collectedToolCalls.append(rtc)
                            let delta = StreamDeltaToolCall(
                                index: collectedToolCalls.count - 1,
                                id: rtc.id,
                                type: rtc.type,
                                function: StreamDeltaFunction(
                                    name: rtc.function.name,
                                    arguments: rtc.function.arguments
                                )
                            )
                            let tcChunk = ChatCompletionStreamResponse(
                                id: streamId,
                                model: res.modelID,
                                toolCalls: [delta]
                            )
                            let tcData = try encoder.encode(tcChunk)
                            if let jsonString = String(data: tcData, encoding: .utf8) {
                                try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                            }
                        }
                    }
                }

                // === LOG IMMEDIATELY after generation, BEFORE any writer.write() calls ===
                if self.veryVerbose {
                    if !verboseReasoningBuf.isEmpty { print("\(Self.purple)[\(Self.timestamp())] SEND reasoning:\n  \(verboseReasoningBuf)\(Self.reset)") }
                    if !verboseContentBuf.isEmpty { print("\(Self.teal)[\(Self.timestamp())] SEND content (chunk):\n  \(verboseContentBuf)\(Self.reset)") }
                }
                let promptTokens = realPromptTokens ?? res.promptTokens
                let completionTokens = realCompletionTokens ?? self.estimateTokens(fullContent)
                let generationDuration = max(Date().timeIntervalSince(started), 0.001)
                let tokPerSec = generationDuration > 0 ? Double(completionTokens) / generationDuration : 0
                let finalizedTurn = Self.finalizeAssistantTurn(
                    content: fullContent,
                    toolCalls: hasToolCalls ? collectedToolCalls : nil,
                    toolChoice: chatRequest.toolChoice,
                    extractThinking: extractThinking,
                    thinkStartTag: thinkStartTag ?? "<think>",
                    thinkEndTag: thinkEndTag ?? "</think>",
                    stoppedBySequence: stoppedBySequence,
                    completionTokens: completionTokens,
                    maxTokens: effectiveMaxTokens,
                    sanitizeContent: self.sanitizeDegenerateTail
                )
                let finishReason = finalizedTurn.finishReason
                if self.veryVerbose {
                    if finishReason == "tool_calls" {
                        print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true\n  prompt_tokens=\(promptTokens) completion_tokens=\(completionTokens)\n  elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=tool_calls\(Self.reset)")
                    } else {
                        let trimmedAnswer = (finalizedTurn.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmedAnswer.isEmpty {
                            print("\(Self.teal)[\(Self.timestamp())] MLX full answer:\n  \(trimmedAnswer)\(Self.reset)")
                        }
                        print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true\n  prompt_tokens=\(promptTokens) completion_tokens=\(completionTokens)\n  elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=\(finishReason)\(Self.reset)")
                    }
                }
                let sPromptTime = realPromptTime ?? 0
                let sPromptTokPerSec = sPromptTime > 0 ? Double(promptTokens) / sPromptTime : 0
                let sGenTime = realGenerateTime ?? generationDuration
                let sGenTokPerSec = sGenTime > 0 ? Double(completionTokens) / sGenTime : 0
                let sCached = realCachedTokens ?? 0
                let sCacheInfo = Self.cacheStatsSummary(
                    cachedTokens: sCached,
                    totalPromptTokens: promptTokens
                )
                print("\(Self.orange)[\(Self.timestamp())] [STATS] pp: \(promptTokens) tok, \(String(format: "%.2f", sPromptTime))s (\(String(format: "%.1f", sPromptTokPerSec)) tok/s) | tg: \(completionTokens) tok, \(String(format: "%.2f", sGenTime))s (\(String(format: "%.1f", sGenTokPerSec)) tok/s)\(sCacheInfo) stream=true\(Self.reset)")
                if let finalToolCalls = finalizedTurn.toolCalls, !finalToolCalls.isEmpty {
                    let tcSummary = finalToolCalls.map { "\($0.function.name)(\(Self.argKeysPreview($0.function.arguments)))" }.joined(separator: ", ")
                    print("\(Self.gold)[\(Self.timestamp())] [TOOL_CALLS] \(finalToolCalls.count) call(s): \(tcSummary)\(Self.reset)")
                }
                if self.veryVerbose {
                    let usageLog = StreamUsage(promptTokens: promptTokens, completionTokens: completionTokens, completionTime: generationDuration, promptTime: 0)
                    print("\(Self.teal)[\(Self.timestamp())] SEND usage:\n  \(self.encodeJSON(usageLog))\(Self.reset)")
                }
                fflush(stdout)

                // === Now flush remaining buffer to client (writer calls may hang/throw) ===
                // Flush remaining thinkBuffer content (tool call tags are handled
                // above and never enter the thinkBuffer, so this is safe).
                if extractThinking && !thinkBuffer.isEmpty {
                    let remaining: String?
                    let remainingReasoning: String?
                    if insideThinkBlock {
                        remainingReasoning = thinkBuffer
                        remaining = nil
                    } else {
                        remaining = thinkBuffer
                        remainingReasoning = nil
                    }
                    if remaining != nil || remainingReasoning != nil {
                        let flushLogprobs = logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer)
                        logprobBuffer = []
                        let flushChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            content: remaining ?? "",
                            reasoningContent: remainingReasoning,
                            logprobs: flushLogprobs,
                            isFirst: false
                        )
                        if let flushData = try? encoder.encode(flushChunk),
                           let jsonString = String(data: flushData, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                    } else if !logprobBuffer.isEmpty {
                        let flushChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            content: "",
                            logprobs: Self.buildChoiceLogprobs(logprobBuffer),
                            isFirst: false
                        )
                        logprobBuffer = []
                        if let flushData = try? encoder.encode(flushChunk),
                           let jsonString = String(data: flushData, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                    }
                }

                let promptTime = realPromptTime ?? 0
                let generateTime = realGenerateTime ?? generationDuration
                let usage = StreamUsage(
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    completionTime: generateTime,
                    promptTime: promptTime,
                    cachedTokens: realCachedTokens
                )
                let finalChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: res.modelID,
                    content: "",
                    isFinished: true,
                    finishReason: finishReason
                )
                let finalData = try encoder.encode(finalChunk)
                if let jsonString = String(data: finalData, encoding: .utf8) {
                    try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                }
                let usageChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: res.modelID,
                    usage: usage,
                    timings: StreamTimings(prompt_n: promptTokens, prompt_ms: promptTime * 1000, predicted_n: completionTokens, predicted_ms: generateTime * 1000)
                )
                let usageData = try encoder.encode(usageChunk)
                if let jsonString = String(data: usageData, encoding: .utf8) {
                    try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                }
                // AFM Profile: send as a final SSE event before [DONE]
                if wantStreamProfile {
                    if wantStreamExtended {
                        let extended = self.service.stopAPIProfileExtended(
                            promptTokens: promptTokens,
                            completionTokens: completionTokens,
                            promptTime: promptTime,
                            generateTime: generateTime
                        )
                        if let data = try? JSONEncoder().encode(["afm_profile_extended": extended]),
                           let json = String(data: data, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
                        }
                    } else {
                        let profile = self.service.stopAPIProfile(
                            promptTokens: promptTokens,
                            completionTokens: completionTokens,
                            promptTime: promptTime,
                            generateTime: generateTime
                        )
                        if let data = try? JSONEncoder().encode(["afm_profile": profile]),
                           let json = String(data: data, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
                        }
                    }
                }
                try? await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                try? await writer.write(.end)
            } catch {
                // Cleanup profile timer on error to prevent leak
                if wantStreamProfile || wantStreamExtended {
                    _ = self.service.stopAPIProfile(promptTokens: 0, completionTokens: 0, promptTime: 0, generateTime: 0)
                }
                // Log summary even on error/cancellation
                let completionTokens = self.estimateTokens(fullContent)
                let generationDuration = max(Date().timeIntervalSince(started), 0.001)
                let tokPerSec = generationDuration > 0 ? Double(completionTokens) / generationDuration : 0
                if self.veryVerbose {
                    let (finalAnswer, _) = Self.extractThinkContent(from: fullContent, startTag: self.service.thinkStartTag ?? "<think>", endTag: self.service.thinkEndTag ?? "</think>")
                    let trimmedAnswer = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmedAnswer.isEmpty {
                        print("\(Self.teal)[\(Self.timestamp())] MLX full answer (before error):\n  \(trimmedAnswer)\(Self.reset)")
                    }
                    print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true\n  completion_tokens=\(completionTokens)\n  elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec))\n  error=\(error.localizedDescription)\(Self.reset)")
                    fflush(stdout)
                }
                req.logger.error("[\(Self.timestamp())] MLX stream error: \(error)")
                let errorChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: self.modelID,
                    content: "⚠️ **Error**\n\n\(error.localizedDescription)",
                    isFirst: true
                )
                if let data = try? encoder.encode(errorChunk), let json = String(data: data, encoding: .utf8) {
                    try? await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
                }
                try? await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                try? await writer.write(.end)
            }
        })

        return httpResponse
    }

    /// Remap a single argument key using the full heuristic chain when --fix-tool-args is enabled.
    /// Returns the original key if no match or fixToolArgs is off.
    private func remapSingleKey(_ key: String, toolName: String, tools: [RequestTool]?) -> String {
        guard service.fixToolArgs, let tools, !tools.isEmpty else { return key }
        // Build a single-entry dict, remap, return the (possibly changed) key
        let dummy: [String: any Sendable] = [key: "" as String]
        let remapped = MLXModelService.remapArgumentKeys(dummy, toolName: toolName, tools: tools)
        return remapped.keys.first ?? key
    }

    /// Apply heuristic argument key remapping to a ResponseToolCall when --fix-tool-args is enabled.
    private func applyFixToolArgs(_ rtc: ResponseToolCall, tools: [RequestTool]?) -> ResponseToolCall {
        guard service.fixToolArgs else { return rtc }
        return MLXModelService.remapResponseToolCallArguments(rtc, tools: tools)
    }

    private func normalizedMaxTokens(_ requested: Int?) -> Int {
        if let requested, requested > 0 {
            return requested
        }
        if let maxTokens, maxTokens > 0 {
            return maxTokens
        }
        return 4096
    }

    private func sanitizeDegenerateTail(_ text: String) -> String {
        var cleaned = text

        if let badChar = cleaned.lastIndex(of: "�"), cleaned.distance(from: badChar, to: cleaned.endIndex) < 512 {
            cleaned = String(cleaned[..<badChar])
        }

        let pattern = "([!?.:,;`~_\\-*=|])\\1{79,}$"
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return cleaned
        }
        let nsrange = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
        guard let match = regex.firstMatch(in: cleaned, range: nsrange),
              let range = Range(match.range, in: cleaned) else {
            return cleaned
        }

        return String(cleaned[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func createSuccessResponse(req: Request, response: ChatCompletionResponse, grammarDowngraded: Bool = false) async throws -> Response {
        let httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
        if grammarDowngraded {
            httpResponse.headers.add(name: "X-Grammar-Constraints", value: "downgraded")
        }
        try httpResponse.content.encode(response)
        return httpResponse
    }

    private func createErrorResponse(req: Request, error: OpenAIError, status: HTTPStatus) async throws -> Response {
        let httpResponse = Response(status: status)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
        try httpResponse.content.encode(error)
        return httpResponse
    }

    private func estimateTokens(_ text: String) -> Int {
        let words = text.split(whereSeparator: \.isWhitespace).count
        return Int(max(Double(text.count) / 4.0, Double(words) / 0.75))
    }

    private static let isoFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    private static func timestamp() -> String {
        isoFormatter.string(from: Date())
    }

    /// Fuzzy-match a hallucinated tool name against valid candidates.
    /// Returns the best match if edit distance ≤ 3, otherwise nil.
    private static func fuzzyMatchToolName(_ name: String, candidates: [String]) -> String? {
        var bestMatch: String?
        var bestDist = Int.max
        for candidate in candidates {
            let d = editDistance(name.lowercased(), candidate.lowercased())
            if d < bestDist {
                bestDist = d
                bestMatch = candidate
            }
        }
        return bestDist <= 3 ? bestMatch : nil
    }

    /// Levenshtein edit distance between two strings.
    private static func editDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        let m = a.count, n = b.count
        if m == 0 { return n }
        if n == 0 { return m }
        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)
        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                curr[j] = a[i-1] == b[j-1] ? prev[j-1] : 1 + Swift.min(prev[j], curr[j-1], prev[j-1])
            }
            prev = curr
        }
        return prev[n]
    }

    /// Extract argument key names from a JSON arguments string for log preview.
    private static func argKeysPreview(_ json: String) -> String {
        guard let data = json.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return json.count > 40 ? "\(json.prefix(40))..." : json
        }
        return dict.keys.sorted().joined(separator: ", ")
    }

    private static func cacheStatsSummary(cachedTokens: Int, totalPromptTokens: Int) -> String {
        let total = max(totalPromptTokens, cachedTokens)
        let suffix = max(0, total - cachedTokens)
        guard cachedTokens > 0 else { return " | cache: MISS suffix=\(suffix)" }
        let ratio = total > 0 ? Int(Double(cachedTokens) / Double(total) * 100) : 0
        return " | cache: HIT \(cachedTokens)/\(total) (\(ratio)%) suffix=\(suffix)"
    }

    /// ANSI color codes
    private static let orange = "\u{1B}[38;5;208m"
    private static let pink = "\u{1B}[38;5;213m"
    private static let red = "\u{1B}[38;5;196m"
    private static let teal = "\u{1B}[38;5;43m"
    private static let purple = "\u{1B}[38;5;135m"
    private static let gold = "\u{1B}[38;5;178m"
    private static let cyan = "\u{1B}[38;5;87m"   // -VV trace logging
    private static let reset = "\u{1B}[0m"

    /// Extract `<think>...</think>` content from a streaming buffer.
    /// Returns any reasoning and regular content that can be flushed.
    /// The buffer retains incomplete tag fragments for the next call.
    static func extractThinkTags(
        buffer: inout String,
        insideThinkBlock: inout Bool,
        startTag: String = "<think>",
        endTag: String = "</think>"
    ) -> (reasoning: String?, content: String?) {
        var reasoning = ""
        var content = ""
        let startTagLen = startTag.count
        let endTagLen = endTag.count

        while !buffer.isEmpty {
            if insideThinkBlock {
                if let endRange = buffer.range(of: endTag) {
                    reasoning += String(buffer[buffer.startIndex..<endRange.lowerBound])
                    buffer = String(buffer[endRange.upperBound...])
                    insideThinkBlock = false
                } else if buffer.count > endTagLen {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -endTagLen)
                    reasoning += String(buffer[buffer.startIndex..<safeEnd])
                    buffer = String(buffer[safeEnd...])
                    break
                } else {
                    break
                }
            } else {
                if let startRange = buffer.range(of: startTag) {
                    let before = String(buffer[buffer.startIndex..<startRange.lowerBound])
                    content += before
                    buffer = String(buffer[startRange.upperBound...])
                    insideThinkBlock = true
                } else if buffer.count > startTagLen {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -startTagLen)
                    content += String(buffer[buffer.startIndex..<safeEnd])
                    buffer = String(buffer[safeEnd...])
                    break
                } else {
                    break
                }
            }
        }

        let r: String? = reasoning.isEmpty ? nil : reasoning
        let c: String? = content.isEmpty ? nil : content

        if r == nil && c == nil {
            return (reasoning: nil, content: nil)
        }

        return (reasoning: r, content: c)
    }

    /// Extract think tags from a complete (non-streaming) response.
    static func extractThinkContent(from text: String, startTag: String = "<think>", endTag: String = "</think>") -> (content: String, reasoning: String?) {
        guard text.contains(startTag) else { return (text, nil) }
        var buffer = text
        var inside = false
        var allReasoning = ""
        var allContent = ""

        while !buffer.isEmpty {
            let extracted = extractThinkTags(buffer: &buffer, insideThinkBlock: &inside, startTag: startTag, endTag: endTag)
            if let r = extracted.reasoning { allReasoning += r }
            if let c = extracted.content { allContent += c }
            if extracted.reasoning == nil && extracted.content == nil { break }
        }
        // Flush remaining buffer
        if !buffer.isEmpty {
            if inside {
                allReasoning += buffer
            } else {
                allContent += buffer
            }
        }

        let reasoning: String? = allReasoning.isEmpty ? nil : allReasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        let content = allContent.trimmingCharacters(in: .whitespacesAndNewlines)
        return (content, reasoning)
    }

    static func finalizeAssistantTurn(
        content: String,
        toolCalls: [ResponseToolCall]?,
        toolChoice: ToolChoice?,
        extractThinking: Bool,
        thinkStartTag: String,
        thinkEndTag: String,
        stoppedBySequence: Bool,
        completionTokens: Int,
        maxTokens: Int,
        sanitizeContent: (String) -> String
    ) -> FinalizedAssistantTurn {
        let effectiveToolCalls = applyToolChoice(toolCalls, toolChoice: toolChoice)
        if let effectiveToolCalls, !effectiveToolCalls.isEmpty {
            return FinalizedAssistantTurn(
                finishReason: "tool_calls",
                content: nil,
                reasoningContent: nil,
                toolCalls: effectiveToolCalls
            )
        }

        let cleanedContent = sanitizeContent(content)
        let finalContent: String
        let reasoningContent: String?
        if extractThinking {
            (finalContent, reasoningContent) = extractThinkContent(
                from: cleanedContent,
                startTag: thinkStartTag,
                endTag: thinkEndTag
            )
        } else {
            finalContent = cleanedContent
            reasoningContent = nil
        }

        let finishReason = stoppedBySequence ? "stop" : (completionTokens >= maxTokens ? "length" : "stop")
        return FinalizedAssistantTurn(
            finishReason: finishReason,
            content: finalContent,
            reasoningContent: reasoningContent,
            toolCalls: nil
        )
    }

    static func applyToolChoice(_ toolCalls: [ResponseToolCall]?, toolChoice: ToolChoice?) -> [ResponseToolCall]? {
        guard let toolCalls, !toolCalls.isEmpty else { return nil }
        guard let toolChoice else { return toolCalls }

        switch toolChoice {
        case .mode(let mode):
            return mode == "none" ? nil : toolCalls
        case .function(let functionChoice):
            let name = functionChoice.function.name
            let filtered = toolCalls.filter { $0.function.name == name }
            return filtered.isEmpty ? nil : filtered
        }
    }

    static func resolveEffectiveTools(_ tools: [RequestTool]?, toolChoice: ToolChoice?) throws -> [RequestTool]? {
        guard let tools, !tools.isEmpty else { return nil }
        guard let toolChoice else { return tools }

        switch toolChoice {
        case .mode(let mode):
            return mode == "none" ? nil : tools
        case .function(let functionChoice):
            let requestedName = functionChoice.function.name
            let filtered = tools.filter { $0.function.name == requestedName }
            guard !filtered.isEmpty else {
                throw Abort(
                    .badRequest,
                    reason: "tool_choice specifies function '\(requestedName)', but that tool was not provided"
                )
            }
            return filtered
        }
    }

    static func isToolCallAllowed(
        _ toolCall: ResponseToolCall,
        toolChoice: ToolChoice?,
        allowedFunctionName: String?,
        permittedToolIndices: inout Set<Int>
    ) -> Bool {
        switch toolChoice {
        case .mode(let mode) where mode == "none":
            return false
        case .function:
            guard let allowedFunctionName else { return false }
            let isAllowed = toolCall.function.name == allowedFunctionName
            if isAllowed {
                permittedToolIndices.insert(toolCall.index ?? permittedToolIndices.count)
            }
            return isAllowed
        default:
            if let index = toolCall.index {
                permittedToolIndices.insert(index)
            }
            return true
        }
    }

    static func isToolDeltaAllowed(
        _ delta: StreamDeltaToolCall,
        toolChoice: ToolChoice?,
        allowedFunctionName: String?,
        permittedToolIndices: inout Set<Int>
    ) -> Bool {
        switch toolChoice {
        case .mode(let mode) where mode == "none":
            return false
        case .function:
            if let name = delta.function?.name {
                let isAllowed = name == allowedFunctionName
                if isAllowed {
                    permittedToolIndices.insert(delta.index)
                }
                return isAllowed
            }
            return permittedToolIndices.contains(delta.index)
        default:
            permittedToolIndices.insert(delta.index)
            return true
        }
    }

    static func buildChoiceLogprobs(_ resolved: [ResolvedLogprob]?) -> ChoiceLogprobs? {
        guard let resolved, !resolved.isEmpty else { return nil }
        let content = resolved.map { entry in
            let topEntries = entry.topTokens.map { top in
                TopLogprobEntry(
                    token: top.token,
                    logprob: Double(top.logprob),
                    bytes: Array(top.token.utf8).map { Int($0) }
                )
            }
            return TokenLogprobContent(
                token: entry.token,
                logprob: Double(entry.logprob),
                bytes: Array(entry.token.utf8).map { Int($0) },
                topLogprobs: topEntries
            )
        }
        return ChoiceLogprobs(content: content)
    }

    /// Look up the schema type for a parameter in a tool's function schema.
    static func schemaTypeForParam(_ paramName: String, toolName: String, tools: [RequestTool]?) -> String? {
        guard let tools else { return nil }
        guard let tool = tools.first(where: { $0.function.name == toolName }),
              let paramsAny = tool.function.parameters?.toSendable() as? [String: Any],
              let props = paramsAny["properties"] as? [String: Any],
              let propSchema = props[paramName] as? [String: Any],
              let schemaType = propSchema["type"] as? String else { return nil }
        return schemaType
    }

    /// JSON-encode a parameter value: if it parses as a JSON array or object,
    /// return it as-is (structured); otherwise encode as a JSON string.
    static func jsonEncodeValue(_ s: String) -> String {
        if let data = s.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data),
           (parsed is [Any] || parsed is [String: Any]),
           let reencoded = try? JSONSerialization.data(withJSONObject: parsed),
           let result = String(data: reencoded, encoding: .utf8) {
            return result
        }
        return jsonEncodeString(s)
    }

    /// JSON-encode a string value with proper escaping, including surrounding quotes.
    static func jsonEncodeString(_ s: String) -> String {
        // Wrap in array so JSONSerialization accepts it, then strip the brackets.
        if let data = try? JSONSerialization.data(withJSONObject: [s]),
           let str = String(data: data, encoding: .utf8),
           str.hasPrefix("["), str.hasSuffix("]") {
            // str is e.g. ["hello \"world\""] — strip [ and ]
            let inner = str.dropFirst().dropLast()
            return String(inner)
        }
        // Fallback: manual escaping
        let escaped = s.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
        return "\"\(escaped)\""
    }

    /// Escape a JSON object key (minimal: just backslash and quote).
    static func jsonEscapeKey(_ s: String) -> String {
        s.replacingOccurrences(of: "\\", with: "\\\\")
         .replacingOccurrences(of: "\"", with: "\\\"")
    }

    /// Convert camelCase to snake_case (e.g. "filePath" → "file_path").
    /// Used to build reverse mapping for Qwen3-Coder's parameter name conversion.
    static func toSnakeCase(_ s: String) -> String {
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

    private func encodeJSON<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(value),
              let text = String(data: data, encoding: .utf8) else {
            return "<json-encode-failed>"
        }
        return text
    }
}
