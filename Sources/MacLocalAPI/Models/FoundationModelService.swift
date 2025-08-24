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
    
    // Shared singleton instance
    static var shared: FoundationModelService?
    
    // Shared adapter for reuse across instances
    static var sharedAdapter: SystemLanguageModel.Adapter?
    static var sharedAdapterPath: String?
    
    init(instructions: String = "You are a helpful assistant", adapter: String? = nil) async throws {
        #if canImport(FoundationModels)
        // Check if adapter path is provided
        if let adapterPath = adapter {
            do {
                // Expand tilde and resolve relative paths
                let expandedPath = NSString(string: adapterPath).expandingTildeInPath
                let adapterURL = URL(fileURLWithPath: expandedPath)
                
                // Validate adapter file exists and has correct extension
                guard FileManager.default.fileExists(atPath: adapterURL.path) else {
                    print("Warning: Adapter file not found at '\(adapterPath)', falling back to default model")
                    self.session = LanguageModelSession {
                        instructions
                    }
                    return
                }
                
                guard adapterURL.pathExtension.lowercased() == "fmadapter" else {
                    print("Warning: Adapter file must have .fmadapter extension, falling back to default model")
                    self.session = LanguageModelSession {
                        instructions
                    }
                    return
                }
                
                // Try to load the adapter
                let adapter = try SystemLanguageModel.Adapter(fileURL: adapterURL)
                let customModel = SystemLanguageModel(adapter: adapter)
                
                // Store adapter for reuse if this is the first time loading
                if Self.sharedAdapter == nil {
                    Self.sharedAdapter = adapter
                    Self.sharedAdapterPath = adapterPath
                }
                
                self.session = LanguageModelSession(model: customModel) {
                    instructions
                }
                
                print("✅ Successfully loaded LoRA adapter: \(adapterURL.lastPathComponent)")
                
            } catch {
                print("Warning: Failed to load adapter '\(adapterPath)': \(error.localizedDescription)")
                print("Falling back to default model")
                
                // Fallback to default model
                self.session = LanguageModelSession {
                    instructions
                }
            }
        } else {
            // No adapter specified, use default model
            self.session = LanguageModelSession {
                instructions
            }
        }
        #else
        throw FoundationModelError.notAvailable
        #endif
    }
    
    // Private initializer for creating instances with shared adapter
    private init(instructions: String, useSharedAdapter: Bool) async throws {
        #if canImport(FoundationModels)
        if useSharedAdapter, let sharedAdapter = Self.sharedAdapter {
            // Use the shared adapter
            let customModel = SystemLanguageModel(adapter: sharedAdapter)
            self.session = LanguageModelSession(model: customModel) {
                instructions
            }
        } else {
            // No shared adapter available, use default model
            self.session = LanguageModelSession {
                instructions
            }
        }
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
    
    func generateStreamingResponseWithTiming(for messages: [Message]) async throws -> (content: String, promptTime: Double) {
        #if canImport(FoundationModels)
        guard let session = session else {
            throw FoundationModelError.sessionCreationFailed
        }
        
        let prompt = formatMessagesAsPrompt(messages)
        
        // Measure actual Foundation Model processing time
        let promptStartTime = Date()
        let response = try await session.respond(to: prompt)
        let promptTime = Date().timeIntervalSince(promptStartTime)
        
        let content = response.content
        
        // Handle empty or nil content
        guard !content.isEmpty else {
            return (content: "I'm unable to generate a response at the moment.", promptTime: promptTime)
        }
        
        return (content: content, promptTime: promptTime)
        #else
        throw FoundationModelError.notAvailable
        #endif
    }
    
    func generateStreamingResponse(for messages: [Message]) async throws -> AsyncThrowingStream<String, Error> {
        #if canImport(FoundationModels)
        guard let session = session else {
            throw FoundationModelError.sessionCreationFailed
        }
        
        let prompt = formatMessagesAsPrompt(messages)
        
        return AsyncThrowingStream<String, Error> { continuation in
            Task {
                do {
                    // Since FoundationModels may not have streaming support yet,
                    // we'll simulate streaming by chunking the complete response
                    let response = try await session.respond(to: prompt)
                    let content = response.content
                    
                    // Handle empty or nil content
                    guard !content.isEmpty else {
                        continuation.yield("I'm unable to generate a response at the moment.")
                        continuation.finish()
                        return
                    }
                    
                    // ChatGPT-style smooth streaming with natural delays
                    await streamContentSmoothly(content: content, continuation: continuation)
                    
                    continuation.finish()
                } catch {
                    // Log the error and provide a fallback response
                    print("FoundationModel error: \(error)")
                    continuation.finish(throwing: FoundationModelError.responseGenerationFailed(error.localizedDescription))
                }
            }
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
    
    private func streamContentSmoothly(content: String, continuation: AsyncThrowingStream<String, Error>.Continuation) async {
        // Handle code blocks specially to preserve formatting
        let codeBlockRanges = findCodeBlockRanges(in: content)
        var currentIndex = content.startIndex
        
        while currentIndex < content.endIndex {
            // Check if we're at the start of a code block
            if let codeBlockRange = codeBlockRanges.first(where: { $0.lowerBound == currentIndex }) {
                // Stream entire code block at once to preserve formatting
                let codeBlockContent = String(content[codeBlockRange])
                continuation.yield(codeBlockContent)
                currentIndex = codeBlockRange.upperBound
                
                // Brief pause after code blocks
                do {
                    try await Task.sleep(nanoseconds: 100_000_000) // 100ms
                } catch {
                    // Continue if sleep is interrupted
                }
            } else {
                // Stream character by character or small tokens for smooth flow
                let remainingContent = String(content[currentIndex...])
                let nextChunk = getNextStreamingChunk(from: remainingContent, codeBlockRanges: codeBlockRanges, currentIndex: currentIndex, fullContent: content)
                
                if !nextChunk.isEmpty {
                    continuation.yield(nextChunk)
                    
                    // Natural streaming delay - varies based on content type
                    let delay = getStreamingDelay(for: nextChunk)
                    if delay > 0 {
                        do {
                            try await Task.sleep(nanoseconds: delay)
                        } catch {
                            // Continue if sleep is interrupted
                        }
                    }
                    
                    currentIndex = content.index(currentIndex, offsetBy: nextChunk.count)
                } else {
                    currentIndex = content.index(after: currentIndex)
                }
            }
        }
    }
    
    private func findCodeBlockRanges(in content: String) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []
        var searchIndex = content.startIndex
        
        while searchIndex < content.endIndex {
            // Look for code block start
            if let startRange = content.range(of: "```", options: [], range: searchIndex..<content.endIndex) {
                // Find the end of this code block
                let afterStart = startRange.upperBound
                if let endRange = content.range(of: "```", options: [], range: afterStart..<content.endIndex) {
                    let fullRange = startRange.lowerBound..<endRange.upperBound
                    ranges.append(fullRange)
                    searchIndex = endRange.upperBound
                } else {
                    // No closing ```, treat rest as code block
                    ranges.append(startRange.lowerBound..<content.endIndex)
                    break
                }
            } else {
                break
            }
        }
        
        return ranges
    }
    
    private func getNextStreamingChunk(from content: String, codeBlockRanges: [Range<String.Index>], currentIndex: String.Index, fullContent: String) -> String {
        // Don't break if we're inside a code block
        if codeBlockRanges.contains(where: { $0.contains(currentIndex) }) {
            return ""
        }
        
        // For smooth streaming, send 1-3 characters at a time, preferring word boundaries
        let maxChunkSize = 3
        let chunkSize = min(maxChunkSize, content.count)
        
        if chunkSize > 1 {
            // Try to break at word boundaries when possible
            let endIndex = content.index(content.startIndex, offsetBy: chunkSize)
            if endIndex < content.endIndex {
                let nextChar = content[endIndex]
                if nextChar.isWhitespace || nextChar.isPunctuation {
                    // Good place to break
                    return String(content.prefix(chunkSize))
                }
                // Look backwards for a space within the chunk
                for i in (1..<chunkSize).reversed() {
                    let testIndex = content.index(content.startIndex, offsetBy: i)
                    if content[testIndex].isWhitespace {
                        return String(content.prefix(i + 1))
                    }
                }
            }
        }
        
        // Fallback to single characters for very smooth streaming
        return chunkSize > 0 ? String(content.prefix(1)) : ""
    }
    
    private func getStreamingDelay(for chunk: String) -> UInt64 {
        // Variable delay based on content type for natural feel
        if chunk.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return 10_000_000 // 10ms for whitespace
        }
        
        if chunk.contains("\n") {
            return 50_000_000 // 50ms for line breaks
        }
        
        if chunk.hasSuffix(".") || chunk.hasSuffix("!") || chunk.hasSuffix("?") {
            return 80_000_000 // 80ms for sentence endings
        }
        
        if chunk.hasSuffix(",") || chunk.hasSuffix(";") || chunk.hasSuffix(":") {
            return 40_000_000 // 40ms for punctuation
        }
        
        // Base delay for regular characters - very fast for smooth flow
        return 15_000_000 // 15ms
    }
    
    private func smartChunkContent(_ content: String) -> [String] {
        var chunks: [String] = []
        var currentChunk = ""
        var inCodeBlock = false
        // Note: inInlineCode and codeBlockMarker reserved for future inline code detection
        
        let lines = content.components(separatedBy: .newlines)
        
        for line in lines {
            let trimmedLine = line.trimmingCharacters(in: .whitespaces)
            
            // Detect code block start/end
            if trimmedLine.hasPrefix("```") {
                if !inCodeBlock {
                    // Starting a code block
                    inCodeBlock = true
                    // codeBlockMarker = trimmedLine // Reserved for future use
                    if !currentChunk.isEmpty {
                        chunks.append(currentChunk)
                        currentChunk = ""
                    }
                    currentChunk = line + "\n"
                } else if trimmedLine == "```" || trimmedLine.hasPrefix("```") {
                    // Ending a code block
                    currentChunk += line + "\n"
                    chunks.append(currentChunk)
                    currentChunk = ""
                    inCodeBlock = false
                    // codeBlockMarker = "" // Reserved for future use
                } else {
                    // Inside code block
                    currentChunk += line + "\n"
                }
                continue
            }
            
            // If we're in a code block, just accumulate
            if inCodeBlock {
                currentChunk += line + "\n"
                continue
            }
            
            // For non-code content, check if we should chunk
            currentChunk += line
            if !line.isEmpty {
                currentChunk += "\n"
            }
            
            // Chunk on natural breaks (paragraphs, sentences) but preserve formatting
            if line.isEmpty || 
               line.hasSuffix(".") || 
               line.hasSuffix("!") || 
               line.hasSuffix("?") ||
               line.hasSuffix(":") ||
               currentChunk.count > 200 { // Fallback size limit
                
                if !currentChunk.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    chunks.append(currentChunk)
                    currentChunk = ""
                }
            }
        }
        
        // Add any remaining content
        if !currentChunk.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            chunks.append(currentChunk)
        }
        
        // If no good chunks were made, fall back to word-based chunking
        if chunks.isEmpty && !content.isEmpty {
            let words = content.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
            let chunkSize = max(5, words.count / 8)
            
            for i in stride(from: 0, to: words.count, by: chunkSize) {
                let endIndex = min(i + chunkSize, words.count)
                let chunk = words[i..<endIndex].joined(separator: " ") + " "
                chunks.append(chunk)
            }
        }
        
        return chunks.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }
    
    static func isAvailable() -> Bool {
        #if canImport(FoundationModels)
        return true
        #else
        return false
        #endif
    }
    
    // Initialize the shared instance once at server startup
    static func initialize(instructions: String = "You are a helpful assistant", adapter: String? = nil) async throws {
        shared = try await FoundationModelService(instructions: instructions, adapter: adapter)
    }
    
    // Get the shared instance
    static func getShared() throws -> FoundationModelService {
        guard let shared = shared else {
            throw FoundationModelError.sessionCreationFailed
        }
        return shared
    }
    
    // Create a new instance that reuses the shared adapter (for per-request use)
    static func createWithSharedAdapter(instructions: String = "You are a helpful assistant") async throws -> FoundationModelService {
        return try await FoundationModelService(instructions: instructions, useSharedAdapter: true)
    }
}

