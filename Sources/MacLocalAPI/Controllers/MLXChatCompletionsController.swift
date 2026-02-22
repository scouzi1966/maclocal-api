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
                req.logger.info("\(Self.pink)[\(Self.timestamp())] MLX full request: \(encodeJSON(chatRequest))\(Self.reset)")
                // Log the user's prompt in red for quick scanning
                if let lastUser = chatRequest.messages.last(where: { $0.role == "user" }) {
                    let prompt = lastUser.textContent
                    let truncated = prompt.count > 500 ? String(prompt.prefix(500)) + "..." : prompt
                    req.logger.info("\(Self.red)[\(Self.timestamp())] MLX user prompt: \(truncated)\(Self.reset)")
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
            let promptChars = chatRequest.messages.map { $0.textContent.count }.reduce(0, +)
            req.logger.info(
                "\(Self.orange)[\(Self.timestamp())] MLX start: stream=false prompt_chars=\(promptChars) max_tokens=\(effectiveMaxTokens) temperature=\(effectiveTemp?.description ?? "default") top_p=\(effectiveTopP?.description ?? "default") rep_penalty=\(effectiveRepetitionPenalty?.description ?? "none") top_k=\(effectiveTopK?.description ?? "none") min_p=\(effectiveMinP?.description ?? "none") presence_penalty=\(effectivePresencePenalty?.description ?? "none") seed=\(effectiveSeed?.description ?? "none")\(Self.reset)"
            )
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
                topLogprobs: chatRequest.topLogprobs
            )
            let cleanedContent = sanitizeDegenerateTail(result.content)
            let elapsed = Date().timeIntervalSince(started)
            let completionTok = result.completionTokens
            let tokPerSec = elapsed > 0 ? Double(completionTok) / elapsed : 0
            let stopReason = completionTok >= effectiveMaxTokens ? "length" : "stop"
            req.logger.info("\(Self.orange)[\(Self.timestamp())] MLX done: stream=false prompt_tokens=\(result.promptTokens) completion_tokens=\(completionTok) elapsed=\(String(format: "%.2f", elapsed))s tok/s=\(String(format: "%.1f", tokPerSec)) finish_reason=\(stopReason)\(Self.reset)")

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
                completionTokens: estimateTokens(cleanedContent)
            )
            if veryVerbose {
                req.logger.info("\(Self.teal)[\(Self.timestamp())] MLX full response: \(encodeJSON(response))\(Self.reset)")
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
                let promptChars = chatRequest.messages.map { $0.textContent.count }.reduce(0, +)
                req.logger.info(
                    "\(Self.orange)[\(Self.timestamp())] MLX start: stream=true prompt_chars=\(promptChars) max_tokens=\(effectiveMaxTokens) temperature=\(effectiveTemp?.description ?? "default") top_p=\(effectiveTopP?.description ?? "default") rep_penalty=\(effectiveRepetitionPenalty?.description ?? "none") top_k=\(effectiveTopK?.description ?? "none") min_p=\(effectiveMinP?.description ?? "none") presence_penalty=\(effectivePresencePenalty?.description ?? "none") seed=\(effectiveSeed?.description ?? "none")\(Self.reset)"
                )
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
                    topLogprobs: chatRequest.topLogprobs
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

                // True token streaming: forward every chunk as it is generated.
                var pendingRawTag: String? = nil
                for try await streamChunk in res.stream {
                    let piece = streamChunk.text
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
                        if emitContent != nil || emitReasoning != nil {
                            if self.veryVerbose {
                                if let r = emitReasoning { verboseReasoningBuf += r }
                                if let c = emitContent { verboseContentBuf += c }
                                // Flush verbose log on newlines or when buffer gets large
                                if verboseReasoningBuf.hasSuffix("\n") || verboseReasoningBuf.count > 200 {
                                    req.logger.info("\(Self.purple)[\(Self.timestamp())] MLX reasoning (extracted): \(verboseReasoningBuf)\(Self.reset)")
                                    verboseReasoningBuf = ""
                                }
                                if verboseContentBuf.hasSuffix("\n") || verboseContentBuf.count > 200 {
                                    req.logger.info("\(Self.teal)[\(Self.timestamp())] MLX content (extracted): \(verboseContentBuf)\(Self.reset)")
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

                // === LOG IMMEDIATELY after generation, BEFORE any writer.write() calls ===
                if self.veryVerbose {
                    if !verboseReasoningBuf.isEmpty { print("\(Self.purple)[\(Self.timestamp())] MLX reasoning (extracted): \(verboseReasoningBuf)\(Self.reset)") }
                    if !verboseContentBuf.isEmpty { print("\(Self.teal)[\(Self.timestamp())] MLX content (extracted): \(verboseContentBuf)\(Self.reset)") }
                }
                let completionTokens = self.estimateTokens(fullContent)
                let generationDuration = max(Date().timeIntervalSince(started), 0.001)
                let tokPerSec = generationDuration > 0 ? Double(completionTokens) / generationDuration : 0
                let stopReason = completionTokens >= effectiveMaxTokens ? "length" : "stop"
                let (finalAnswer, _) = Self.extractThinkContent(from: fullContent)
                let trimmedAnswer = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedAnswer.isEmpty {
                    print("\(Self.teal)[\(Self.timestamp())] MLX answer: \(trimmedAnswer)\(Self.reset)")
                }
                print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true prompt_tokens=\(res.promptTokens) completion_tokens=\(completionTokens) elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec)) finish_reason=\(stopReason)\(Self.reset)")
                if self.veryVerbose {
                    let usageLog = StreamUsage(promptTokens: res.promptTokens, completionTokens: completionTokens, completionTime: generationDuration, promptTime: 0)
                    print("\(Self.teal)[\(Self.timestamp())] MLX stream final usage: \(self.encodeJSON(usageLog))\(Self.reset)")
                }
                fflush(stdout)

                // === Now flush remaining buffer to client (writer calls may hang/throw) ===
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
                    promptTokens: res.promptTokens,
                    completionTokens: completionTokens,
                    completionTime: generationDuration,
                    promptTime: 0
                )
                let finalChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: res.modelID,
                    content: "",
                    isFinished: true,
                    usage: usage,
                    timings: StreamTimings(prompt_n: res.promptTokens, prompt_ms: 0, predicted_n: completionTokens, predicted_ms: generationDuration * 1000)
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
                let (finalAnswer, _) = Self.extractThinkContent(from: fullContent)
                let trimmedAnswer = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedAnswer.isEmpty {
                    print("\(Self.teal)[\(Self.timestamp())] MLX answer: \(trimmedAnswer)\(Self.reset)")
                }
                print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true completion_tokens=\(completionTokens) elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec)) error=\(error.localizedDescription)\(Self.reset)")
                fflush(stdout)
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
