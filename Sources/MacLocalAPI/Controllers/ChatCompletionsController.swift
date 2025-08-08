import Vapor
import Foundation

struct ChatCompletionsController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.post("chat", "completions", use: chatCompletions)
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
        try httpResponse.content.encode(response)
        return httpResponse
    }
    
    private func createErrorResponse(req: Request, error: OpenAIError, status: HTTPStatus) async throws -> Response {
        var httpResponse = Response(status: status)
        httpResponse.headers.add(name: .contentType, value: "application/json")
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
}