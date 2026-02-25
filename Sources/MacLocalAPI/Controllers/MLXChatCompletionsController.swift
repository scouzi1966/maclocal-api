import Vapor
import Foundation

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
    private let rawOutput: Bool

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
        rawOutput: Bool = false
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
        self.rawOutput = rawOutput
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
                req.logger.info("\(Self.pink)[\(Self.timestamp())] RECV MLX full request:\n\(encodeJSON(chatRequest))\(Self.reset)")
                if let lastUser = chatRequest.messages.last(where: { $0.role == "user" }) {
                    let prompt = lastUser.textContent
                    let truncated = prompt.count > 500 ? String(prompt.prefix(500)) + "..." : prompt
                    req.logger.info("\(Self.red)[\(Self.timestamp())] RECV MLX user prompt:\n  \(truncated)\(Self.reset)")
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
               requestedModelRaw != modelID {
                // WebUI may send transformed model identifiers; afm mlx always serves the active model.
                req.logger.info("[\(Self.timestamp())] MLX request model '\(requestedModelRaw)' does not match active model '\(modelID)'; serving active model")
            }

            let hasTools = chatRequest.tools != nil && !(chatRequest.tools?.isEmpty ?? true)
            if hasTools && veryVerbose {
                let toolNames = chatRequest.tools!.map { $0.function.name }.joined(separator: ", ")
                req.logger.info("\(Self.gold)[\(Self.timestamp())] RECV tools: [\(toolNames)]\(Self.reset)")
            }

            let isWebUI = req.headers.first(name: .origin) != nil
            let extractThinking = !rawOutput || isWebUI

            if chatRequest.stream == true && streamingEnabled {
                return try await createStreamingResponse(req: req, chatRequest: chatRequest, extractThinking: extractThinking)
            }

            let effectiveTemp = chatRequest.temperature ?? temperature
            let effectiveTopP = chatRequest.topP ?? topP
            let effectiveMaxTokens = normalizedMaxTokens(chatRequest.effectiveMaxTokens)
            let effectiveRepetitionPenalty = chatRequest.effectiveRepetitionPenalty ?? repetitionPenalty
            let effectiveTopK = chatRequest.topK ?? topK
            let effectiveMinP = chatRequest.minP ?? minP
            let effectivePresencePenalty = chatRequest.presencePenalty ?? presencePenalty
            let effectiveSeed = chatRequest.seed ?? seed
            let started = Date()
            if veryVerbose {
                let promptChars = chatRequest.messages.map { $0.textContent.count }.reduce(0, +)
                let stopDesc = chatRequest.stop.map { $0.map { $0.debugDescription }.joined(separator: ", ") }
                req.logger.info(
                    "\(Self.orange)[\(Self.timestamp())] MLX start: stream=false\n  prompt_chars=\(promptChars) max_tokens=\(effectiveMaxTokens)\n  temperature=\(effectiveTemp?.description ?? "default") top_p=\(effectiveTopP?.description ?? "default") rep_penalty=\(effectiveRepetitionPenalty?.description ?? "none")\n  top_k=\(effectiveTopK?.description ?? "none") min_p=\(effectiveMinP?.description ?? "none") presence_penalty=\(effectivePresencePenalty?.description ?? "none")\n  seed=\(effectiveSeed?.description ?? "none") stop=\(stopDesc ?? "none")\(Self.reset)"
                )
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
                tools: chatRequest.tools,
                stop: chatRequest.stop,
                responseFormat: chatRequest.responseFormat
            )
            let cleanedContent = sanitizeDegenerateTail(result.content)
            let elapsed = Date().timeIntervalSince(started)
            let completionTok = result.completionTokens
            let tokPerSec = elapsed > 0 ? Double(completionTok) / elapsed : 0

            // If we got tool calls, return a tool_calls response
            if let toolCalls = result.toolCalls, !toolCalls.isEmpty {
                if veryVerbose {
                    let toolNames = toolCalls.map { $0.function.name }.joined(separator: ", ")
                    req.logger.info("\(Self.orange)[\(Self.timestamp())] MLX done: stream=false\n  prompt_tokens=\(result.promptTokens) completion_tokens=\(completionTok)\n  elapsed=\(String(format: "%.2f", elapsed))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=tool_calls\(Self.reset)")
                    for tc in toolCalls {
                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call: \(tc.function.name)\n  id=\(tc.id)\n  args=\(tc.function.arguments)\(Self.reset)")
                    }
                    fflush(stdout)
                }

                let choiceLogprobs = buildChoiceLogprobs(result.tokenLogprobs)
                let response = ChatCompletionResponse(
                    model: result.modelID,
                    toolCalls: toolCalls,
                    logprobs: choiceLogprobs,
                    promptTokens: result.promptTokens,
                    completionTokens: completionTok
                )
                if veryVerbose {
                    req.logger.info("\(Self.teal)[\(Self.timestamp())] SEND full response:\n\(encodeJSON(response))\(Self.reset)")
                }
                return try await createSuccessResponse(req: req, response: response)
            }

            let stopReason = completionTok >= effectiveMaxTokens ? "length" : "stop"
            if veryVerbose {
                req.logger.info("\(Self.orange)[\(Self.timestamp())] MLX done: stream=false\n  prompt_tokens=\(result.promptTokens) completion_tokens=\(completionTok)\n  elapsed=\(String(format: "%.2f", elapsed))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=\(stopReason)\(Self.reset)")
            }

            // Extract <think>...</think> tags into reasoning_content
            let finalContent: String
            let reasoningContent: String?
            if extractThinking {
                (finalContent, reasoningContent) = Self.extractThinkContent(from: cleanedContent)
            } else {
                finalContent = cleanedContent
                reasoningContent = nil
            }

            let choiceLogprobs = buildChoiceLogprobs(result.tokenLogprobs)
            let response = ChatCompletionResponse(
                model: result.modelID,
                content: finalContent,
                reasoningContent: reasoningContent,
                logprobs: choiceLogprobs,
                promptTokens: result.promptTokens,
                completionTokens: completionTok
            )
            if veryVerbose {
                req.logger.info("\(Self.teal)[\(Self.timestamp())] SEND full response:\n\(encodeJSON(response))\(Self.reset)")
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

            do {
                if self.veryVerbose {
                    let promptChars = chatRequest.messages.map { $0.textContent.count }.reduce(0, +)
                    let stopDesc = chatRequest.stop.map { $0.map { $0.debugDescription }.joined(separator: ", ") }
                    req.logger.info(
                        "\(Self.orange)[\(Self.timestamp())] MLX start: stream=true\n  prompt_chars=\(promptChars) max_tokens=\(effectiveMaxTokens)\n  temperature=\(effectiveTemp?.description ?? "default") top_p=\(effectiveTopP?.description ?? "default") rep_penalty=\(effectiveRepetitionPenalty?.description ?? "none")\n  top_k=\(effectiveTopK?.description ?? "none") min_p=\(effectiveMinP?.description ?? "none") presence_penalty=\(effectivePresencePenalty?.description ?? "none")\n  seed=\(effectiveSeed?.description ?? "none") stop=\(stopDesc ?? "none")\(Self.reset)"
                    )
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
                    tools: chatRequest.tools,
                    stop: chatRequest.stop,
                    responseFormat: chatRequest.responseFormat
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
                var realPromptTokens: Int? = nil
                var realCompletionTokens: Int? = nil

                // Token-level tool call detection (mlx-lm style).
                // Instead of buffering ALL content when tools are present, detect
                // tool call start/end tags per-token. Content outside tool calls
                // streams normally; only the tool call body is buffered and parsed.
                let toolCallStartTag = res.toolCallStartTag
                let toolCallEndTag = res.toolCallEndTag
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

                    // Capture real token counts from the info chunk
                    if let pt = streamChunk.promptTokens { realPromptTokens = pt }
                    if let ct = streamChunk.completionTokens { realCompletionTokens = ct }

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
                                print("\(Self.gold)[\(Self.timestamp())] RECV </tool_call> end tag\n  body=\(currentToolText.count) chars\(Self.reset)")
                                fflush(stdout)
                            }

                            if incrementalEmittedFirst {
                                // Run one final parameter scan on the complete tool body
                                // to catch parameters that completed in the same token as </tool_call>.
                                let paramPattern = #"<parameter=([^>]+)>([\s\S]*?)</parameter>"#
                                if let paramRegex = try? NSRegularExpression(pattern: paramPattern, options: [.dotMatchesLineSeparators]) {
                                    let nsText = currentToolText as NSString
                                    let matches = paramRegex.matches(in: currentToolText, range: NSRange(location: 0, length: nsText.length))
                                    for match in matches {
                                        guard match.numberOfRanges >= 3,
                                              let keyRange = Range(match.range(at: 1), in: currentToolText),
                                              let valRange = Range(match.range(at: 2), in: currentToolText) else { continue }
                                        let rawKey = String(currentToolText[keyRange])
                                        if incrementalEmittedKeys.contains(rawKey) { continue }
                                        var value = String(currentToolText[valRange])
                                        if value.hasPrefix("\n") { value = String(value.dropFirst()) }
                                        if value.hasSuffix("\n") { value = String(value.dropLast()) }
                                        if value.isEmpty { continue }
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
                                            try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                                        }
                                    }
                                }

                                // Emit closing for the arguments JSON object.
                                // If no parameters were emitted, send "{}" as a complete object;
                                // otherwise just close with "}".
                                let closeArgs = incrementalParamCount == 0 ? "{}" : "}"
                                let closeDelta = StreamDeltaToolCall(
                                    index: incrementalToolIndex,
                                    id: nil,
                                    type: nil,
                                    function: StreamDeltaFunction(name: nil, arguments: closeArgs)
                                )
                                let closeChunk = ChatCompletionStreamResponse(
                                    id: streamId,
                                    model: res.modelID,
                                    toolCalls: [closeDelta]
                                )
                                let closeData = try encoder.encode(closeChunk)
                                if let jsonString = String(data: closeData, encoding: .utf8) {
                                    try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                                }
                                // Build the full tool call for collectedToolCalls record
                                let wrapped = "\(toolCallStartTag!)\(currentToolText)\(toolCallEndTag!)"
                                let (parsed, _) = MLXModelService.extractToolCallsFallback(from: wrapped)
                                for tc in parsed {
                                    hasToolCalls = true
                                    let rtc = self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: incrementalToolIndex, paramNameMapping: paramNameMapping), tools: chatRequest.tools)
                                    // Replace the placeholder we added earlier
                                    if incrementalToolIndex < collectedToolCalls.count {
                                        collectedToolCalls[incrementalToolIndex] = rtc
                                    } else {
                                        collectedToolCalls.append(rtc)
                                    }
                                    if self.veryVerbose {
                                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call (incremental): \(rtc.function.name)\n  id=\(rtc.id)\n  args=\(rtc.function.arguments)\(Self.reset)")
                                        fflush(stdout)
                                    }
                                }
                            } else {
                                // Incremental parsing never kicked in (non-XML format).
                                // Fall back to single-chunk emission.
                                let wrapped = "\(toolCallStartTag!)\(currentToolText)\(toolCallEndTag!)"
                                let (parsed, _) = MLXModelService.extractToolCallsFallback(from: wrapped)
                                if parsed.isEmpty {
                                    if self.veryVerbose {
                                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call fallback: found 0 tool calls\n  body=\(currentToolText)\(Self.reset)")
                                        fflush(stdout)
                                    }
                                }
                                for tc in parsed {
                                    hasToolCalls = true
                                    let rtc = self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: collectedToolCalls.count, paramNameMapping: paramNameMapping), tools: chatRequest.tools)
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
                                    incrementalCallId = "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
                                    incrementalToolIndex = collectedToolCalls.count
                                    // Add a placeholder to collectedToolCalls
                                    let placeholder = ResponseToolCall(
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
                                }
                            }

                            // 2. Detect complete <parameter=KEY>VALUE</parameter> pairs — emit argument fragments
                            if incrementalEmittedFirst {
                                // Scan for all complete parameter tags not yet emitted
                                let paramPattern = #"<parameter=([^>]+)>([\s\S]*?)</parameter>"#
                                if let paramRegex = try? NSRegularExpression(pattern: paramPattern, options: [.dotMatchesLineSeparators]) {
                                    let nsText = currentToolText as NSString
                                    let matches = paramRegex.matches(in: currentToolText, range: NSRange(location: 0, length: nsText.length))
                                    for match in matches {
                                        guard match.numberOfRanges >= 3,
                                              let keyRange = Range(match.range(at: 1), in: currentToolText),
                                              let valRange = Range(match.range(at: 2), in: currentToolText) else { continue }
                                        let rawKey = String(currentToolText[keyRange])
                                        // Skip duplicate parameters (dedup)
                                        if incrementalEmittedKeys.contains(rawKey) { continue }
                                        var value = String(currentToolText[valRange])
                                        // Strip leading/trailing newlines to match parseXMLFunction behavior
                                        if value.hasPrefix("\n") { value = String(value.dropFirst()) }
                                        if value.hasSuffix("\n") { value = String(value.dropLast()) }
                                        // Skip empty values — Qwen3-Coder sometimes emits
                                        // an empty duplicate first, then the real value.
                                        // Match extractToolCallsFallback's "first non-empty" logic.
                                        if value.isEmpty { continue }
                                        incrementalEmittedKeys.insert(rawKey)

                                        // Map model-emitted key back to original schema name.
                                        // Basic: snake_case→camelCase (Qwen3-Coder converts filePath → file_path).
                                        // With --fix-tool-args: also case-insensitive, camel→snake, and suffix match.
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
                                            id: nil,
                                            type: nil,
                                            function: StreamDeltaFunction(name: nil, arguments: fragment)
                                        )
                                        let paramChunk = ChatCompletionStreamResponse(
                                            id: streamId,
                                            model: res.modelID,
                                            toolCalls: [paramDelta]
                                        )
                                        let paramData = try encoder.encode(paramChunk)
                                        if let jsonString = String(data: paramData, encoding: .utf8) {
                                            try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                                        }
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
                    if self.veryVerbose && (piece.contains("<think>") || piece.contains("</think>")) {
                        pendingRawTag = piece.debugDescription
                    }

                    if extractThinking {
                        thinkBuffer += piece

                        let extracted = Self.extractThinkTags(
                            buffer: &thinkBuffer,
                            insideThinkBlock: &insideThinkBlock
                        )

                        let emitContent = extracted.content
                        let emitReasoning = extracted.reasoning

                        // Only emit a chunk if we have something to send
                        let hasReasoning = emitReasoning != nil
                        let hasContent = emitContent != nil
                        if hasReasoning || hasContent {
                            if self.veryVerbose {
                                if let r = emitReasoning { verboseReasoningBuf += r }
                                if let c = emitContent { verboseContentBuf += c }
                                // Flush verbose log on newlines or when buffer gets large
                                if verboseReasoningBuf.hasSuffix("\n") || verboseReasoningBuf.count > 200 {
                                    req.logger.info("\(Self.purple)[\(Self.timestamp())] SEND reasoning:\n  \(verboseReasoningBuf)\(Self.reset)")
                                    verboseReasoningBuf = ""
                                }
                                if verboseContentBuf.hasSuffix("\n") || verboseContentBuf.count > 200 {
                                    req.logger.info("\(Self.teal)[\(Self.timestamp())] SEND content (chunk):\n  \(verboseContentBuf)\(Self.reset)")
                                    verboseContentBuf = ""
                                }
                            }
                            // Flush accumulated logprobs with this chunk
                            let flushLogprobs = logprobBuffer.isEmpty ? nil : self.buildChoiceLogprobs(logprobBuffer)
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
                            try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                        // Update the placeholder with final parsed data
                        let wrapped = "\(toolCallStartTag!)\(currentToolText)\(toolCallEndTag!)"
                        let (parsed, _) = MLXModelService.extractToolCallsFallback(from: wrapped)
                        for tc in parsed {
                            hasToolCalls = true
                            let rtc = self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: incrementalToolIndex, paramNameMapping: paramNameMapping), tools: chatRequest.tools)
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
                            let rtc = self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: collectedToolCalls.count, paramNameMapping: paramNameMapping), tools: chatRequest.tools)
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
                            let rtc = self.applyFixToolArgs(MLXModelService.convertToolCall(tc, index: collectedToolCalls.count, paramNameMapping: paramNameMapping), tools: chatRequest.tools)
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
                    finishReason = completionTokens >= effectiveMaxTokens ? "length" : "stop"
                    if self.veryVerbose {
                        let (finalAnswer, _) = Self.extractThinkContent(from: fullContent)
                        let trimmedAnswer = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmedAnswer.isEmpty {
                            print("\(Self.teal)[\(Self.timestamp())] MLX full answer:\n  \(trimmedAnswer)\(Self.reset)")
                        }
                        print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true\n  prompt_tokens=\(promptTokens) completion_tokens=\(completionTokens)\n  elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=\(finishReason)\(Self.reset)")
                    }
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
                        let flushChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            content: remaining ?? "",
                            reasoningContent: remainingReasoning,
                            isFirst: false
                        )
                        if let flushData = try? encoder.encode(flushChunk),
                           let jsonString = String(data: flushData, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                    }
                }

                let usage = StreamUsage(
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    completionTime: generationDuration,
                    promptTime: 0
                )
                let finalChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: res.modelID,
                    content: "",
                    isFinished: true,
                    finishReason: finishReason,
                    usage: usage,
                    timings: StreamTimings(prompt_n: promptTokens, prompt_ms: 0, predicted_n: completionTokens, predicted_ms: generationDuration * 1000)
                )
                let finalData = try encoder.encode(finalChunk)
                if let jsonString = String(data: finalData, encoding: .utf8) {
                    try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                }
                try? await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                try? await writer.write(.end)
            } catch {
                // Log summary even on error/cancellation
                let completionTokens = self.estimateTokens(fullContent)
                let generationDuration = max(Date().timeIntervalSince(started), 0.001)
                let tokPerSec = generationDuration > 0 ? Double(completionTokens) / generationDuration : 0
                if self.veryVerbose {
                    let (finalAnswer, _) = Self.extractThinkContent(from: fullContent)
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

    /// ANSI color codes
    private static let orange = "\u{1B}[38;5;208m"
    private static let pink = "\u{1B}[38;5;213m"
    private static let red = "\u{1B}[38;5;196m"
    private static let teal = "\u{1B}[38;5;43m"
    private static let purple = "\u{1B}[38;5;135m"
    private static let gold = "\u{1B}[38;5;178m"
    private static let reset = "\u{1B}[0m"

    /// Extract `<think>...</think>` content from a streaming buffer.
    /// Returns any reasoning and regular content that can be flushed.
    /// The buffer retains incomplete tag fragments for the next call.
    private static func extractThinkTags(
        buffer: inout String,
        insideThinkBlock: inout Bool
    ) -> (reasoning: String?, content: String?) {
        var reasoning = ""
        var content = ""

        while !buffer.isEmpty {
            if insideThinkBlock {
                if let endRange = buffer.range(of: "</think>") {
                    reasoning += String(buffer[buffer.startIndex..<endRange.lowerBound])
                    buffer = String(buffer[endRange.upperBound...])
                    insideThinkBlock = false
                } else if buffer.count > 8 {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -8)
                    reasoning += String(buffer[buffer.startIndex..<safeEnd])
                    buffer = String(buffer[safeEnd...])
                    break
                } else {
                    break
                }
            } else {
                if let startRange = buffer.range(of: "<think>") {
                    let before = String(buffer[buffer.startIndex..<startRange.lowerBound])
                    content += before
                    buffer = String(buffer[startRange.upperBound...])
                    insideThinkBlock = true
                } else if buffer.count > 7 {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -7)
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

    /// Extract `<think>...</think>` from a complete (non-streaming) response.
    private static func extractThinkContent(from text: String) -> (content: String, reasoning: String?) {
        guard text.contains("<think>") else { return (text, nil) }
        var buffer = text
        var inside = false
        var allReasoning = ""
        var allContent = ""

        while !buffer.isEmpty {
            let extracted = extractThinkTags(buffer: &buffer, insideThinkBlock: &inside)
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
