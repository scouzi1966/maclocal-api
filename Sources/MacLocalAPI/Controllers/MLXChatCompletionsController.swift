import Vapor
import Foundation
import MLXLMCommon

struct MLXChatCompletionsController: RouteCollection {
    private let streamingEnabled: Bool
    private let modelID: String
    private let service: MLXModelService
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
        service: MLXModelService,
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
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization")
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

            // Suppress tools when tool_choice=none
            let toolChoiceNone: Bool
            if case .mode(let m) = chatRequest.toolChoice, m == "none" {
                toolChoiceNone = true
            } else {
                toolChoiceNone = false
            }
            let effectiveTools: [RequestTool]? = toolChoiceNone ? nil : chatRequest.tools

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

            let isWebUI = req.headers.first(name: .origin) != nil
            let extractThinking = !rawOutput || isWebUI

            if chatRequest.stream == true && streamingEnabled {
                return try await createStreamingResponse(req: req, chatRequest: chatRequest, extractThinking: extractThinking)
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
            let result = try await service.generate(
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
            let cleanedContent = sanitizeDegenerateTail(result.content)
            let completionTok = result.completionTokens
            let promptTime = result.promptTime
            let generateTime = result.generateTime
            let tokPerSec = generateTime > 0 ? Double(completionTok) / generateTime : 0
            let promptTokPerSec = promptTime > 0 ? Double(result.promptTokens) / promptTime : 0

            // If we got tool calls, return a tool_calls response
            if let toolCalls = result.toolCalls, !toolCalls.isEmpty {
                if veryVerbose {
                    let toolNames = toolCalls.map { $0.function.name }.joined(separator: ", ")
                    print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=false\n  prompt_tokens=\(result.promptTokens) completion_tokens=\(completionTok)\n  prompt=\(String(format: "%.2f", promptTime))s gen=\(String(format: "%.2f", generateTime))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=tool_calls\(Self.reset)"); fflush(stdout)
                    for tc in toolCalls {
                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call: \(tc.function.name)\n  id=\(tc.id)\n  args=\(tc.function.arguments)\(Self.reset)")
                    }
                    fflush(stdout)
                }

                let choiceLogprobs = buildChoiceLogprobs(result.tokenLogprobs)
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
                return try await createSuccessResponse(req: req, response: response)
            }

            let stopReason = result.stoppedBySequence ? "stop" : (completionTok >= effectiveMaxTokens ? "length" : "stop")
            if veryVerbose {
                print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=false\n  prompt_tokens=\(result.promptTokens) completion_tokens=\(completionTok)\n  prompt=\(String(format: "%.2f", promptTime))s gen=\(String(format: "%.2f", generateTime))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=\(stopReason)\(Self.reset)"); fflush(stdout)
            }

            // Extract <think>...</think> tags into reasoning_content
            let finalContent: String
            let reasoningContent: String?
            if extractThinking {
                (finalContent, reasoningContent) = Self.extractThinkContent(
                    from: cleanedContent,
                    startTag: service.thinkStartTag ?? "<think>",
                    endTag: service.thinkEndTag ?? "</think>"
                )
            } else {
                finalContent = cleanedContent
                reasoningContent = nil
            }

            let choiceLogprobs = buildChoiceLogprobs(result.tokenLogprobs)
            let timings = StreamTimings(prompt_n: result.promptTokens, prompt_ms: promptTime * 1000, predicted_n: completionTok, predicted_ms: generateTime * 1000)
            let extended = wantExtended ? service.stopAPIProfileExtended(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil
            let profile = wantExtended ? nil : (wantProfile ? service.stopAPIProfile(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil)
            let response = ChatCompletionResponse(
                model: result.modelID,
                content: finalContent,
                reasoningContent: reasoningContent,
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
            return try await createSuccessResponse(req: req, response: response)
        } catch {
            req.logger.error("[\(Self.timestamp())] MLX completions error: \(error)")
            return try await createErrorResponse(req: req, error: OpenAIError(message: error.localizedDescription, type: "mlx_error"), status: .badRequest)
        }
    }

    private func createStreamingResponse(req: Request, chatRequest: ChatCompletionRequest, extractThinking: Bool) async throws -> Response {
        let httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "text/event-stream")
        httpResponse.headers.add(name: .cacheControl, value: "no-cache")
        httpResponse.headers.add(name: .connection, value: "keep-alive")
        httpResponse.headers.add(name: "Access-Control-Allow-Origin", value: "*")
        httpResponse.headers.add(name: "Access-Control-Allow-Headers", value: "Content-Type")
        httpResponse.headers.add(name: "X-Accel-Buffering", value: "no")

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
            let effectiveTools: [RequestTool]? = {
                if case .mode(let m) = chatRequest.toolChoice, m == "none" { return nil }
                return chatRequest.tools
            }()

            do {
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
                var inToolCall = false
                var madeToolCall = false
                var currentToolText = ""

                // Incremental tool call streaming state (OpenAI-compatible).
                // Emits function name and argument fragments as XML tags complete,
                // rather than waiting for the entire tool call body.
                var incrementalEmittedFirst = false
                var incrementalCallId = ""
                var incrementalToolIndex = 0
                var incrementalParamCount = 0
                var incrementalEmittedKeys = Set<String>()

                // Build parameter name mapping from tool schemas.
                // When --fix-tool-args is enabled, we build a comprehensive mapping
                // that handles case-insensitive, snake↔camel, and suffix matches.
                // Otherwise, just the basic snake_case → camelCase mapping for Qwen3-Coder.
                var paramNameMapping = [String: String]()  // model-emitted-key → original
                if let tools = chatRequest.tools {
                    for tool in tools {
                        if let paramsAny = tool.function.parameters?.toSendable() as? [String: Any],
                           let props = paramsAny["properties"] as? [String: Any] {
                            for key in props.keys {
                                let snaked = Self.toSnakeCase(key)
                                if snaked != key {
                                    paramNameMapping[snaked] = key
                                }
                            }
                        }
                    }
                }

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

                    // Handle tool call chunks from the vendor parser
                    if let tcs = streamChunk.toolCalls, !tcs.isEmpty {
                        hasToolCalls = true
                        madeToolCall = true
                        for tc in tcs {
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

                    // Token-level tool call tag detection (before think extraction).
                    // Uses contains() for robustness: detokenizer may merge tokens
                    // or add whitespace around special tokens.
                    if let startTag = toolCallStartTag, !inToolCall, piece.contains(startTag) {
                        inToolCall = true
                        madeToolCall = true
                        fullContent += piece
                        // Flush any remaining thinkBuffer content BEFORE tool call deltas,
                        // so clients receive text content before tool_calls (correct ordering).
                        if extractThinking && !thinkBuffer.isEmpty {
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
                        // Any text after the start tag begins the tool call body
                        if let range = piece.range(of: startTag) {
                            let after = String(piece[range.upperBound...])
                            if !after.isEmpty { currentToolText += after }
                        }
                        if self.veryVerbose {
                            print("\(Self.gold)[\(Self.timestamp())] RECV <tool_call> start tag\(Self.reset)")
                            fflush(stdout)
                        }
                        continue
                    }
                    if inToolCall {
                        fullContent += piece
                        if let endTag = toolCallEndTag, piece.contains(endTag) {
                            // Text before end tag is part of tool call body
                            if let range = piece.range(of: endTag) {
                                let before = String(piece[..<range.lowerBound])
                                if !before.isEmpty { currentToolText += before }
                            }
                            if self.veryVerbose {
                                let bodyPreviewForLog = currentToolText.count > 500
                                    ? "\(currentToolText.prefix(250))...\(currentToolText.suffix(250))"
                                    : currentToolText
                                print("\(Self.gold)[\(Self.timestamp())] RECV </tool_call> end tag\n  body=\(currentToolText.count) chars\n  raw=\(bodyPreviewForLog)\(Self.reset)")
                                fflush(stdout)
                            }
                            // -VV: Full verbatim body from model (no truncation)
                            if self.trace {
                                print("\(Self.cyan)[\(Self.timestamp())] [VV] RECV←MODEL tool_call body verbatim (\(currentToolText.count) chars):\n\(currentToolText)\(Self.reset)")
                                fflush(stdout)
                            }

                            if incrementalEmittedFirst {
                                // Run fallback parser on complete tool body — this does
                                // cross-param dedup, key remapping, and type coercion.
                                // Emit the CLEAN args via SSE (deferred from midstream).
                                let wrapped = "\(toolCallStartTag!)\(currentToolText)\(toolCallEndTag!)"
                                let (parsed, _) = MLXModelService.extractToolCallsFallback(from: wrapped)
                                for tc in parsed {
                                    hasToolCalls = true
                                    let rtc = MLXModelService.coerceArgumentTypes(self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: incrementalToolIndex, paramNameMapping: paramNameMapping), tools: chatRequest.tools), tools: chatRequest.tools)
                                    // Replace the placeholder we added earlier
                                    if incrementalToolIndex < collectedToolCalls.count {
                                        collectedToolCalls[incrementalToolIndex] = rtc
                                    } else {
                                        collectedToolCalls.append(rtc)
                                    }

                                    // Emit clean args as single SSE chunk
                                    let argsJson = rtc.function.arguments
                                    let argsDelta = StreamDeltaToolCall(
                                        index: incrementalToolIndex,
                                        id: nil, type: nil,
                                        function: StreamDeltaFunction(name: nil, arguments: argsJson)
                                    )
                                    let argsChunk = ChatCompletionStreamResponse(
                                        id: streamId, model: res.modelID, toolCalls: [argsDelta]
                                    )
                                    let argsData = try encoder.encode(argsChunk)
                                    if let jsonString = String(data: argsData, encoding: .utf8) {
                                        if self.trace {
                                            print("\(Self.cyan)[\(Self.timestamp())] [SSE] endtag deferred args: \(argsJson.prefix(200))\(Self.reset)")
                                            fflush(stdout)
                                        }
                                        try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                                    }

                                    if self.veryVerbose {
                                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call (incremental): \(rtc.function.name)\n  id=\(rtc.id)\n  args=\(rtc.function.arguments)\(Self.reset)")
                                        fflush(stdout)
                                    }
                                }
                            } else {
                                // Incremental parsing never kicked in (non-XML format).
                                // Fall back to single-chunk emission.
                                var parsed: [ToolCall] = []

                                // afm_adaptive_xml: try direct JSON parse first (handles model format-switching)
                                if self.service.toolCallParser == "afm_adaptive_xml" {
                                    let trimmedBody = currentToolText.trimmingCharacters(in: .whitespacesAndNewlines)
                                    if trimmedBody.hasPrefix("{"),
                                       let data = trimmedBody.data(using: .utf8),
                                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                                       let name = json["name"] as? String {
                                        var arguments: [String: any Sendable] = [:]
                                        if let args = (json["arguments"] as? [String: Any]) ?? (json["parameters"] as? [String: Any]) {
                                            for (k, v) in args { arguments[k] = v as Sendable }
                                        }
                                        // Validate tool name against request schema; fuzzy-match if hallucinated
                                        let validNames = effectiveTools?.map { $0.function.name } ?? []
                                        let resolvedName: String
                                        if validNames.contains(name) {
                                            resolvedName = name
                                        } else if let match = Self.fuzzyMatchToolName(name, candidates: validNames) {
                                            resolvedName = match
                                            if self.veryVerbose {
                                                print("\(Self.gold)[\(Self.timestamp())] afm_adaptive_xml: corrected hallucinated tool '\(name)' → '\(match)'\(Self.reset)")
                                                fflush(stdout)
                                            }
                                        } else {
                                            resolvedName = name  // pass through, let client report the error
                                        }
                                        parsed = [ToolCall(function: .init(name: resolvedName, arguments: arguments))]
                                        if self.veryVerbose {
                                            print("\(Self.gold)[\(Self.timestamp())] afm_adaptive_xml: JSON-in-XML fallback parsed '\(resolvedName)' with \(arguments.count) args\(Self.reset)")
                                            fflush(stdout)
                                        }
                                    }
                                }

                                // Fall back to regex extraction (existing behavior)
                                if parsed.isEmpty {
                                    let wrapped = "\(toolCallStartTag!)\(currentToolText)\(toolCallEndTag!)"
                                    let (regexParsed, _) = MLXModelService.extractToolCallsFallback(from: wrapped)
                                    parsed = regexParsed
                                }

                                // Validate/correct tool names against request schema
                                if self.service.toolCallParser == "afm_adaptive_xml" {
                                    let validNames = effectiveTools?.map { $0.function.name } ?? []
                                    if !validNames.isEmpty {
                                        parsed = parsed.map { tc in
                                            if validNames.contains(tc.function.name) { return tc }
                                            if let match = Self.fuzzyMatchToolName(tc.function.name, candidates: validNames) {
                                                if self.veryVerbose {
                                                    print("\(Self.gold)[\(Self.timestamp())] afm_adaptive_xml: corrected hallucinated tool '\(tc.function.name)' → '\(match)'\(Self.reset)")
                                                    fflush(stdout)
                                                }
                                                return ToolCall(function: .init(name: match, arguments: tc.function.arguments))
                                            }
                                            return tc
                                        }
                                    }
                                }

                                if parsed.isEmpty {
                                    if self.veryVerbose {
                                        let bodyPreview = currentToolText.count > 200 ? "\(currentToolText.prefix(100))...\(currentToolText.suffix(100))" : currentToolText
                                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call fallback: found 0 tool calls\n  body=\(bodyPreview)\(Self.reset)")
                                        fflush(stdout)
                                    }
                                }
                                for tc in parsed {
                                    hasToolCalls = true
                                    let rtc = MLXModelService.coerceArgumentTypes(self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: collectedToolCalls.count, paramNameMapping: paramNameMapping), tools: chatRequest.tools), tools: chatRequest.tools)
                                    collectedToolCalls.append(rtc)
                                    if self.veryVerbose {
                                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call (fallback): \(rtc.function.name)\n  id=\(rtc.id)\n  args=\(rtc.function.arguments)\(Self.reset)")
                                        fflush(stdout)
                                    }
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
                                        // -VV: Log SSE chunk sent to client
                                        if self.trace {
                                            print("\(Self.cyan)[\(Self.timestamp())] [VV] SEND→CLIENT SSE tool_call:\n  data: \(jsonString)\(Self.reset)")
                                            fflush(stdout)
                                        }
                                        try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                                    }
                                }
                            }
                            // Reset state for next tool call
                            currentToolText = ""
                            inToolCall = false
                            incrementalEmittedFirst = false
                            incrementalCallId = ""
                            incrementalParamCount = 0
                            incrementalEmittedKeys = Set<String>()
                        } else {
                            currentToolText += piece

                            // --- Incremental XML tag scanning ---
                            // Try to detect <function=NAME> and complete <parameter=KEY>...</parameter> pairs
                            // in the accumulated currentToolText and emit JSON argument fragments.

                            // 1. Detect <function=NAME> — emit first delta with name
                            if !incrementalEmittedFirst,
                               let funcRange = currentToolText.range(of: #"<function=([^>]+)>"#, options: .regularExpression) {
                                let matchStr = String(currentToolText[funcRange])
                                // Extract function name from <function=NAME>
                                if let nameRange = matchStr.range(of: "="),
                                   let closeRange = matchStr.range(of: ">", options: .backwards) {
                                    let funcName = String(matchStr[nameRange.upperBound..<closeRange.lowerBound])
                                    // If funcName looks like JSON (contains " or {), the model emitted
                                    // JSON inside XML tags. Skip incremental emission — the end-tag
                                    // handler will try JSON fallback parsing instead.
                                    if funcName.contains("\"") || funcName.contains("{") {
                                        if self.veryVerbose {
                                            print("\(Self.gold)[\(Self.timestamp())] Skipping incremental emit: funcName looks like JSON: \(funcName.prefix(60))\(Self.reset)")
                                            fflush(stdout)
                                        }
                                    } else {
                                    incrementalCallId = "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
                                    incrementalToolIndex = collectedToolCalls.count
                                    // Add a placeholder to collectedToolCalls
                                    let placeholder = ResponseToolCall(
                                        index: incrementalToolIndex,
                                        id: incrementalCallId,
                                        type: "function",
                                        function: ResponseToolCallFunction(name: funcName, arguments: "")
                                    )
                                    collectedToolCalls.append(placeholder)
                                    hasToolCalls = true
                                    incrementalEmittedFirst = true

                                    // Emit first delta: id, type, name, empty arguments
                                    let firstDelta = StreamDeltaToolCall(
                                        index: incrementalToolIndex,
                                        id: incrementalCallId,
                                        type: "function",
                                        function: StreamDeltaFunction(name: funcName, arguments: "")
                                    )
                                    let firstChunk = ChatCompletionStreamResponse(
                                        id: streamId,
                                        model: res.modelID,
                                        toolCalls: [firstDelta]
                                    )
                                    let firstData = try encoder.encode(firstChunk)
                                    if let jsonString = String(data: firstData, encoding: .utf8) {
                                        try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                                    }
                                    if self.veryVerbose {
                                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call name: \(funcName)\n  id=\(incrementalCallId)\(Self.reset)")
                                        fflush(stdout)
                                    }
                                    } // else (funcName is not JSON)
                                }
                            }

                            // 2. Detect complete <parameter=KEY>VALUE</parameter> pairs — track but DON'T emit yet.
                            // SSE emission is deferred until </tool_call> so cross-param dedup can clean
                            // leaked JSON fragments before anything goes over the wire.
                            if incrementalEmittedFirst {
                                let paramPattern = #"<parameter=([^>]+)>([\s\S]*?)</parameter>"#
                                if let paramRegex = try? NSRegularExpression(pattern: paramPattern, options: [.dotMatchesLineSeparators]) {
                                    let nsText = currentToolText as NSString
                                    let matches = paramRegex.matches(in: currentToolText, range: NSRange(location: 0, length: nsText.length))
                                    for match in matches {
                                        guard match.numberOfRanges >= 3,
                                              let keyRange = Range(match.range(at: 1), in: currentToolText),
                                              let _ = Range(match.range(at: 2), in: currentToolText) else { continue }
                                        let rawKey = String(currentToolText[keyRange])
                                        incrementalEmittedKeys.insert(rawKey)
                                    }
                                }
                            }
                        }
                        continue
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

                        let flushLogprobs = logprobBuffer.isEmpty ? nil : self.buildChoiceLogprobs(logprobBuffer)

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
                        let flushLogprobs = logprobBuffer.isEmpty ? nil : self.buildChoiceLogprobs(logprobBuffer)
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
                if inToolCall && !currentToolText.isEmpty {
                    if incrementalEmittedFirst {
                        // Salvage unclosed parameter (e.g. model hit max_tokens mid-content).
                        // Look for <parameter=KEY>VALUE... without a closing </parameter>.
                        let unclosedPattern = #"<parameter=([^>]+)>([\s\S]+)$"#
                        if let unclosedRegex = try? NSRegularExpression(pattern: unclosedPattern, options: []),
                           let unclosedMatch = unclosedRegex.firstMatch(in: currentToolText, range: NSRange(currentToolText.startIndex..., in: currentToolText)),
                           let keyRange = Range(unclosedMatch.range(at: 1), in: currentToolText),
                           let valRange = Range(unclosedMatch.range(at: 2), in: currentToolText) {
                            let rawKey = String(currentToolText[keyRange])
                            if !incrementalEmittedKeys.contains(rawKey) {
                                var value = String(currentToolText[valRange])
                                if value.hasPrefix("\n") { value = String(value.dropFirst()) }
                                if value.hasSuffix("\n") { value = String(value.dropLast()) }
                                value = MLXModelService.decodeJSONEscapes(MLXModelService.decodeXMLEntities(value))
                                if !value.isEmpty {
                                    incrementalEmittedKeys.insert(rawKey)
                                    var emitKey = paramNameMapping[rawKey] ?? rawKey
                                    if emitKey == rawKey {
                                        let funcName = incrementalToolIndex < collectedToolCalls.count ? collectedToolCalls[incrementalToolIndex].function.name : ""
                                        emitKey = self.remapSingleKey(rawKey, toolName: funcName, tools: chatRequest.tools)
                                    }
                                    let jsonValue = Self.jsonEncodeString(value)
                                    let fragment: String
                                    if incrementalParamCount == 0 {
                                        fragment = "{\"\(Self.jsonEscapeKey(emitKey))\":\(jsonValue)"
                                    } else {
                                        fragment = ",\"\(Self.jsonEscapeKey(emitKey))\":\(jsonValue)"
                                    }
                                    incrementalParamCount += 1
                                    let paramDelta = StreamDeltaToolCall(
                                        index: incrementalToolIndex,
                                        id: nil, type: nil,
                                        function: StreamDeltaFunction(name: nil, arguments: fragment)
                                    )
                                    let paramChunk = ChatCompletionStreamResponse(
                                        id: streamId, model: res.modelID, toolCalls: [paramDelta]
                                    )
                                    let paramData = try encoder.encode(paramChunk)
                                    if let jsonString = String(data: paramData, encoding: .utf8) {
                                        if self.trace {
                                            print("\(Self.cyan)[\(Self.timestamp())] [SSE] salvage arg fragment: \(fragment)\(Self.reset)")
                                            fflush(stdout)
                                        }
                                        try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                                    }
                                    if self.veryVerbose {
                                        print("\(Self.gold)[\(Self.timestamp())] SEND salvaged unclosed param '\(rawKey)' (\(value.count) chars)\(Self.reset)")
                                        fflush(stdout)
                                    }
                                }
                            }
                        }

                        // Close the JSON arguments object we started incrementally.
                        // If no parameters were emitted, send "{}" as a complete object.
                        let incCloseArgs = incrementalParamCount == 0 ? "{}" : "}"
                        let closeDelta = StreamDeltaToolCall(
                            index: incrementalToolIndex,
                            id: nil,
                            type: nil,
                            function: StreamDeltaFunction(name: nil, arguments: incCloseArgs)
                        )
                        let closeChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            toolCalls: [closeDelta]
                        )
                        let closeData = try encoder.encode(closeChunk)
                        if let jsonString = String(data: closeData, encoding: .utf8) {
                            if self.trace {
                                print("\(Self.cyan)[\(Self.timestamp())] [SSE] salvage arg close: \(incCloseArgs)\(Self.reset)")
                                fflush(stdout)
                            }
                            try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                        // Update the placeholder with final parsed data
                        let wrapped = "\(toolCallStartTag!)\(currentToolText)\(toolCallEndTag!)"
                        let (parsed, _) = MLXModelService.extractToolCallsFallback(from: wrapped)
                        for tc in parsed {
                            hasToolCalls = true
                            let rtc = MLXModelService.coerceArgumentTypes(self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: incrementalToolIndex, paramNameMapping: paramNameMapping), tools: chatRequest.tools), tools: chatRequest.tools)
                            if incrementalToolIndex < collectedToolCalls.count {
                                collectedToolCalls[incrementalToolIndex] = rtc
                            } else {
                                collectedToolCalls.append(rtc)
                            }
                        }
                    } else {
                        let wrapped = "\(toolCallStartTag!)\(currentToolText)\(toolCallEndTag!)"
                        let (parsed, _) = MLXModelService.extractToolCallsFallback(from: wrapped)
                        for tc in parsed {
                            hasToolCalls = true
                            let rtc = MLXModelService.coerceArgumentTypes(self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: collectedToolCalls.count, paramNameMapping: paramNameMapping), tools: chatRequest.tools), tools: chatRequest.tools)
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
                if madeToolCall { hasToolCalls = true }

                // Post-loop fallback: if tools were present but no tool calls detected
                // by token-level matching, try full-content regex parsing.
                // This handles edge cases where tags aren't single tokens.
                let trimmedFull = fullContent.trimmingCharacters(in: .whitespacesAndNewlines)
                let looksLikeBareJsonToolCall = trimmedFull.hasPrefix("{") && trimmedFull.contains("\"name\"")
                if !hasToolCalls && (
                    (toolCallStartTag != nil && fullContent.contains(toolCallStartTag!)) ||
                    fullContent.contains("[TOOL_CALLS]") ||
                    (chatRequest.tools != nil && looksLikeBareJsonToolCall)
                ) {
                    if self.veryVerbose {
                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call: token-level missed, trying fallback parser\(Self.reset)")
                        fflush(stdout)
                    }
                    let (parsed, _) = MLXModelService.extractToolCallsFallback(from: fullContent)
                    if !parsed.isEmpty {
                        hasToolCalls = true
                        for tc in parsed {
                            let rtc = MLXModelService.coerceArgumentTypes(self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: collectedToolCalls.count, paramNameMapping: paramNameMapping), tools: chatRequest.tools), tools: chatRequest.tools)
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
                let finishReason: String
                if hasToolCalls {
                    finishReason = "tool_calls"
                    if self.veryVerbose {
                        print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true\n  prompt_tokens=\(promptTokens) completion_tokens=\(completionTokens)\n  elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=tool_calls\(Self.reset)")
                    }
                } else {
                    finishReason = stoppedBySequence ? "stop" : (completionTokens >= effectiveMaxTokens ? "length" : "stop")
                    if self.veryVerbose {
                        let (finalAnswer, _) = Self.extractThinkContent(from: fullContent, startTag: thinkStartTag ?? "<think>", endTag: thinkEndTag ?? "</think>")
                        let trimmedAnswer = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
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
                if hasToolCalls && !collectedToolCalls.isEmpty {
                    let tcSummary = collectedToolCalls.map { "\($0.function.name)(\(Self.argKeysPreview($0.function.arguments)))" }.joined(separator: ", ")
                    print("\(Self.gold)[\(Self.timestamp())] [TOOL_CALLS] \(collectedToolCalls.count) call(s): \(tcSummary)\(Self.reset)")
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
                        let flushLogprobs = logprobBuffer.isEmpty ? nil : self.buildChoiceLogprobs(logprobBuffer)
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
                            logprobs: self.buildChoiceLogprobs(logprobBuffer),
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
        guard service.fixToolArgs, let tools, !tools.isEmpty else { return rtc }
        guard let data = rtc.function.arguments.data(using: .utf8),
              let argsDict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return rtc }
        var sendableArgs = [String: any Sendable]()
        for (k, v) in argsDict { sendableArgs[k] = v }
        let remapped = MLXModelService.remapArgumentKeys(sendableArgs, toolName: rtc.function.name, tools: tools)
        let remappedAny = remapped.mapValues { $0 as Any }
        guard let newData = try? JSONSerialization.data(withJSONObject: remappedAny, options: [.sortedKeys]),
              let newStr = String(data: newData, encoding: .utf8) else { return rtc }
        return ResponseToolCall(
            index: rtc.index,
            id: rtc.id,
            type: rtc.type,
            function: ResponseToolCallFunction(name: rtc.function.name, arguments: newStr)
        )
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

    private func createSuccessResponse(req: Request, response: ChatCompletionResponse) async throws -> Response {
        let httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
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
    private static func extractThinkTags(
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
    private static func extractThinkContent(from text: String, startTag: String = "<think>", endTag: String = "</think>") -> (content: String, reasoning: String?) {
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

    private func buildChoiceLogprobs(_ resolved: [ResolvedLogprob]?) -> ChoiceLogprobs? {
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
