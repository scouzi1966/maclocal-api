import Vapor
import Foundation

struct ChatCompletionsController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.post("chat", "completions", use: chatCompletions)
        v1.on(.OPTIONS, "chat", "completions", use: handleOptions)
    }
    
    func handleOptions(req: Request) async throws -> Response {
        var response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization")
        return response
    }
    
    func chatCompletions(req: Request) async throws -> Response {
        do {
            let chatRequest = try req.content.decode(ChatCompletionRequest.self)
            
            guard !chatRequest.messages.isEmpty else {
                let error = OpenAIError(message: "At least one message is required")
                return try await createErrorResponse(req: req, error: error, status: .badRequest)
            }
            
            let foundationService: FoundationModelService
            if #available(macOS 26.0, *) {
                foundationService = try await FoundationModelService()
            } else {
                throw FoundationModelError.notAvailable
            }
            
            // Check if streaming is requested
            if chatRequest.stream == true {
                return try await createStreamingResponse(req: req, chatRequest: chatRequest, foundationService: foundationService)
            }
            
            let content = try await foundationService.generateResponse(for: chatRequest.messages)
            
            let promptTokens = estimateTokens(for: chatRequest.messages)
            let completionTokens = estimateTokens(for: content)
            
            let response = ChatCompletionResponse(
                model: chatRequest.model,
                content: content,
                promptTokens: promptTokens,
                completionTokens: completionTokens
            )
            
            return try await createSuccessResponse(req: req, response: response)
            
        } catch let foundationError as FoundationModelError {
            let error = OpenAIError(
                message: foundationError.localizedDescription,
                type: "foundation_model_error"
            )
            return try await createErrorResponse(req: req, error: error, status: .serviceUnavailable)
            
        } catch {
            req.logger.error("Unexpected error: \(error)")
            let error = OpenAIError(
                message: "Internal server error occurred",
                type: "internal_error"
            )
            return try await createErrorResponse(req: req, error: error, status: .internalServerError)
        }
    }
    
    private func createSuccessResponse(req: Request, response: ChatCompletionResponse) async throws -> Response {
        var httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
        try httpResponse.content.encode(response)
        return httpResponse
    }
    
    private func createErrorResponse(req: Request, error: OpenAIError, status: HTTPStatus) async throws -> Response {
        var httpResponse = Response(status: status)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
        try httpResponse.content.encode(error)
        return httpResponse
    }
    
    private func estimateTokens(for messages: [Message]) -> Int {
        let totalText = messages.map { $0.content }.joined(separator: " ")
        return estimateTokens(for: totalText)
    }
    
    private func estimateTokens(for text: String) -> Int {
        let words = text.split(separator: " ")
        return Int(Double(words.count) * 1.3)
    }
    
    private func createStreamingResponse(req: Request, chatRequest: ChatCompletionRequest, foundationService: FoundationModelService) async throws -> Response {
        var httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "text/event-stream")
        httpResponse.headers.add(name: .cacheControl, value: "no-cache")
        httpResponse.headers.add(name: .connection, value: "keep-alive")
        httpResponse.headers.add(name: "Access-Control-Allow-Origin", value: "*")
        httpResponse.headers.add(name: "Access-Control-Allow-Headers", value: "Content-Type")
        httpResponse.headers.add(name: "X-Accel-Buffering", value: "no")
        
        let streamId = UUID().uuidString
        
        httpResponse.body = .init(asyncStream: { writer in
            do {
                let encoder = JSONEncoder()
                let stream = try await foundationService.generateStreamingResponse(for: chatRequest.messages)
                
                var isFirst = true
                
                // Stream content chunks
                for try await chunk in stream {
                    let streamChunk = ChatCompletionStreamResponse(
                        id: streamId,
                        model: chatRequest.model,
                        content: chunk,
                        isFirst: isFirst
                    )
                    isFirst = false
                    
                    let chunkData = try encoder.encode(streamChunk)
                    if let jsonString = String(data: chunkData, encoding: .utf8) {
                        try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                    }
                }
                
                // Send final chunk
                let finalChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: chatRequest.model,
                    content: "",
                    isFinished: true
                )
                let finalData = try encoder.encode(finalChunk)
                if let jsonString = String(data: finalData, encoding: .utf8) {
                    try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                }
                
                // Send done marker
                try await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                
                try await writer.write(.end)
                
            } catch {
                req.logger.error("Streaming error: \(error)")
                // Send a proper error response in OpenAI format
                let errorResponse = """
                data: {"error": {"message": "\(error.localizedDescription)", "type": "server_error"}}

                """
                try? await writer.write(.buffer(.init(string: errorResponse)))
                try? await writer.write(.end)
            }
        })
        
        return httpResponse
    }
}