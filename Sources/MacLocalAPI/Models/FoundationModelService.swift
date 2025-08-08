import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

enum FoundationModelError: Error, LocalizedError {
    case notAvailable
    case sessionCreationFailed
    case responseGenerationFailed(String)
    case invalidInput
    
    var errorDescription: String? {
        switch self {
        case .notAvailable:
            return "Foundation Models framework is not available. Requires macOS 26+ with Apple Intelligence enabled."
        case .sessionCreationFailed:
            return "Failed to create Foundation Models session. Ensure Apple Intelligence is enabled in System Settings."
        case .responseGenerationFailed(let message):
            return "Failed to generate response: \(message)"
        case .invalidInput:
            return "Invalid input provided to Foundation Models"
        }
    }
}

@available(macOS 26.0, *)
class FoundationModelService {
    
    #if canImport(FoundationModels)
    private var session: LanguageModelSession?
    #endif
    
    init() async throws {
        #if canImport(FoundationModels)
        self.session = LanguageModelSession()
        #else
        throw FoundationModelError.notAvailable
        #endif
    }
    
    func generateResponse(for messages: [Message]) async throws -> String {
        #if canImport(FoundationModels)
        guard let session = session else {
            throw FoundationModelError.sessionCreationFailed
        }
        
        let prompt = formatMessagesAsPrompt(messages)
        
        do {
            let response = try await session.respond(to: prompt)
            return response.content
        } catch {
            throw FoundationModelError.responseGenerationFailed(error.localizedDescription)
        }
        #else
        throw FoundationModelError.notAvailable
        #endif
    }
    
    private func formatMessagesAsPrompt(_ messages: [Message]) -> String {
        var prompt = ""
        
        for message in messages {
            switch message.role {
            case "system":
                prompt += "System: \(message.content)\n\n"
            case "user":
                prompt += "User: \(message.content)\n\n"
            case "assistant":
                prompt += "Assistant: \(message.content)\n\n"
            default:
                prompt += "\(message.content)\n\n"
            }
        }
        
        prompt += "Assistant: "
        return prompt
    }
    
    static func isAvailable() -> Bool {
        #if canImport(FoundationModels)
        return true
        #else
        return false
        #endif
    }
}

