import Vapor
import Foundation

struct MLXChatCompletionsController: RouteCollection {
    private let streamingEnabled: Bool
    private let modelID: String
    private let service: MLXModelService
    private let temperature: Double?

    init(streamingEnabled: Bool = true, modelID: String, service: MLXModelService, temperature: Double?) {
        self.streamingEnabled = streamingEnabled
        self.modelID = modelID
        self.service = service
        self.temperature = temperature
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
            let started = Date()
            req.logger.info("MLX generation start: model=\(modelID) stream=false max_tokens=\(effectiveMaxTokens)")
            let result = try await service.generate(
                model: modelID,
                messages: chatRequest.messages,
                temperature: effectiveTemp,
                maxTokens: effectiveMaxTokens
            )
            req.logger.info("MLX generation done: model=\(modelID) stream=false completion_tokens=\(result.completionTokens) elapsed=\(String(format: "%.2f", Date().timeIntervalSince(started)))s")
            let response = ChatCompletionResponse(
                model: result.modelID,
                content: result.content,
                promptTokens: result.promptTokens,
                completionTokens: result.completionTokens
            )
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
                let started = Date()
                req.logger.info("MLX generation start: model=\(self.modelID) stream=true max_tokens=\(effectiveMaxTokens)")
                let res = try await service.generate(
                    model: modelID,
                    messages: chatRequest.messages,
                    temperature: effectiveTemp,
                    maxTokens: effectiveMaxTokens
                )
                req.logger.info("MLX generation done: model=\(self.modelID) stream=true completion_tokens=\(res.completionTokens) elapsed=\(String(format: "%.2f", Date().timeIntervalSince(started)))s")

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

                let start = Date()
                let contentChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: res.modelID,
                    content: res.content,
                    isFirst: false
                )
                let chunkData = try encoder.encode(contentChunk)
                if let jsonString = String(data: chunkData, encoding: .utf8) {
                    try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                }

                let duration = Date().timeIntervalSince(start)
                let usage = StreamUsage(promptTokens: res.promptTokens, completionTokens: res.completionTokens, completionTime: duration, promptTime: 0)
                let finalChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: res.modelID,
                    content: "",
                    isFinished: true,
                    usage: usage,
                    timings: StreamTimings(prompt_n: res.promptTokens, prompt_ms: 0, predicted_n: res.completionTokens, predicted_ms: duration * 1000)
                )
                let finalData = try encoder.encode(finalChunk)
                if let jsonString = String(data: finalData, encoding: .utf8) {
                    try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
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
            return 256
        }
        return min(requested, 2048)
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
}
