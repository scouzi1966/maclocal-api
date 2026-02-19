import Vapor
import Foundation

struct MLXChatCompletionsController: RouteCollection {
    private let streamingEnabled: Bool
    private let modelID: String
    private let service: MLXModelService
    private let temperature: Double?
    private let repetitionPenalty: Double?
    private let veryVerbose: Bool

    init(
        streamingEnabled: Bool = true,
        modelID: String,
        service: MLXModelService,
        temperature: Double?,
        repetitionPenalty: Double?,
        veryVerbose: Bool = false
    ) {
        self.streamingEnabled = streamingEnabled
        self.modelID = modelID
        self.service = service
        self.temperature = temperature
        self.repetitionPenalty = repetitionPenalty
        self.veryVerbose = veryVerbose
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
                req.logger.info("MLX full request: \(encodeJSON(chatRequest))")
            }
            guard !chatRequest.messages.isEmpty else {
                return try await createErrorResponse(req: req, error: OpenAIError(message: "At least one message is required"), status: .badRequest)
            }

            if let requestedModelRaw = chatRequest.model?.trimmingCharacters(in: .whitespacesAndNewlines),
               !requestedModelRaw.isEmpty,
               requestedModelRaw != modelID {
                // WebUI may send transformed model identifiers; afm mlx always serves the active model.
                req.logger.info("MLX request model '\(requestedModelRaw)' does not match active model '\(modelID)'; serving active model")
            }

            if chatRequest.stream == true && streamingEnabled {
                return try await createStreamingResponse(req: req, chatRequest: chatRequest)
            }

            let effectiveTemp = chatRequest.temperature ?? temperature
            let effectiveMaxTokens = normalizedMaxTokens(chatRequest.maxTokens)
            let effectiveRepetitionPenalty = chatRequest.effectiveRepetitionPenalty ?? repetitionPenalty
            let started = Date()
            req.logger.info(
                "MLX generation start: model=\(modelID) stream=false max_tokens=\(effectiveMaxTokens) temperature=\(effectiveTemp?.description ?? "nil") top_p=\(chatRequest.topP?.description ?? "nil") repetition_penalty=\(effectiveRepetitionPenalty?.description ?? "nil")"
            )
            let result = try await service.generate(
                model: modelID,
                messages: chatRequest.messages,
                temperature: effectiveTemp,
                maxTokens: effectiveMaxTokens,
                topP: chatRequest.topP,
                repetitionPenalty: effectiveRepetitionPenalty
            )
            let cleanedContent = sanitizeDegenerateTail(result.content)
            req.logger.info("MLX generation done: model=\(modelID) stream=false completion_tokens=\(result.completionTokens) elapsed=\(String(format: "%.2f", Date().timeIntervalSince(started)))s")
            let response = ChatCompletionResponse(
                model: result.modelID,
                content: cleanedContent,
                promptTokens: result.promptTokens,
                completionTokens: estimateTokens(cleanedContent)
            )
            if veryVerbose {
                req.logger.info("MLX full response: \(encodeJSON(response))")
            }
            return try await createSuccessResponse(req: req, response: response)
        } catch {
            req.logger.error("MLX completions error: \(error)")
            return try await createErrorResponse(req: req, error: OpenAIError(message: error.localizedDescription, type: "mlx_error"), status: .badRequest)
        }
    }

    private func createStreamingResponse(req: Request, chatRequest: ChatCompletionRequest) async throws -> Response {
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
            do {
                let effectiveTemp = chatRequest.temperature ?? self.temperature
                let effectiveMaxTokens = self.normalizedMaxTokens(chatRequest.maxTokens)
                let effectiveRepetitionPenalty = chatRequest.effectiveRepetitionPenalty ?? self.repetitionPenalty
                let started = Date()
                req.logger.info(
                    "MLX generation start: model=\(self.modelID) stream=true max_tokens=\(effectiveMaxTokens) temperature=\(effectiveTemp?.description ?? "nil") top_p=\(chatRequest.topP?.description ?? "nil") repetition_penalty=\(effectiveRepetitionPenalty?.description ?? "nil")"
                )
                let res = try await service.generateStreaming(
                    model: modelID,
                    messages: chatRequest.messages,
                    temperature: effectiveTemp,
                    maxTokens: effectiveMaxTokens,
                    topP: chatRequest.topP,
                    repetitionPenalty: effectiveRepetitionPenalty
                )
                var fullContent = ""

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

                // True token streaming: forward every chunk as it is generated.
                for try await piece in res.stream {
                    fullContent += piece
                    let contentChunk = ChatCompletionStreamResponse(
                        id: streamId,
                        model: res.modelID,
                        content: piece,
                        isFirst: false
                    )
                    let chunkData = try encoder.encode(contentChunk)
                    if let jsonString = String(data: chunkData, encoding: .utf8) {
                        try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                    }
                }

                let completionTokens = self.estimateTokens(fullContent)
                let generationDuration = max(Date().timeIntervalSince(started), 0.001)
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
                    try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                }
                req.logger.info("MLX generation done: model=\(self.modelID) stream=true completion_tokens=\(completionTokens) elapsed=\(String(format: "%.2f", generationDuration))s")
                if self.veryVerbose {
                    req.logger.info("MLX full streamed response content: \(fullContent)")
                    req.logger.info("MLX stream final usage: \(encodeJSON(usage))")
                    req.logger.info("MLX stream final chunk: \(encodeJSON(finalChunk))")
                }
                try await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                try await writer.write(.end)
            } catch {
                req.logger.error("MLX stream error: \(error)")
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
        guard let requested, requested > 0 else {
            return 2000
        }
        return requested
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
