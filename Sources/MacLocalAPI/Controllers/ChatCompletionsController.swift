import Vapor
import Foundation

struct ChatCompletionsController: RouteCollection {
    private let streamingEnabled: Bool
    private let instructions: String
    private let adapter: String?
    private let temperature: Double?
    private let randomness: String?
    private let permissiveGuardrails: Bool
    
    init(streamingEnabled: Bool = true, instructions: String = "You are a helpful assistant", adapter: String? = nil, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool) {
        self.streamingEnabled = streamingEnabled
        self.instructions = instructions
        self.adapter = adapter
        self.temperature = temperature
        self.randomness = randomness
        self.permissiveGuardrails = permissiveGuardrails
    }
    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.post("chat", "completions", use: chatCompletions)
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
                let error = OpenAIError(message: "At least one message is required")
                return try await createErrorResponse(req: req, error: error, status: .badRequest)
            }
            
            let foundationService: FoundationModelService
            if #available(macOS 26.0, *) {
                foundationService = try await FoundationModelService.createWithSharedAdapter(instructions: instructions, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails)
            } else {
                throw FoundationModelError.notAvailable
            }
            
            // Check if streaming is requested and enabled
            if chatRequest.stream == true && streamingEnabled {
                return try await createStreamingResponse(req: req, chatRequest: chatRequest, foundationService: foundationService)
            }
            
            // Use temperature from API request if provided, otherwise use CLI parameter
            let effectiveTemperature = chatRequest.temperature ?? temperature
            let effectiveRandomness = randomness

            let content = try await foundationService.generateResponse(for: chatRequest.messages, temperature: effectiveTemperature, randomness: effectiveRandomness)
            
            let promptTokens = estimateTokens(for: chatRequest.messages)
            let completionTokens = estimateTokens(for: content)
            
            let response = ChatCompletionResponse(
                model: chatRequest.model ?? "foundation",
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
    
    private func estimateTokens(for messages: [Message]) -> Int {
        let totalText = messages.map { $0.content }.joined(separator: " ")
        return estimateTokens(for: totalText)
    }
    
    private func estimateTokens(for text: String) -> Int {
        // GPT-Style estimation based on OpenAI's rough estimates:
        // - 1 token ≈ 4 characters of English text
        // - 1 token ≈ ¾ words
        // - 100 tokens ≈ 75 words
        
        let wordCount = text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }.count
        
        // Use the more conservative estimate
        let charBasedTokens = Double(text.count) / 4.0
        let wordBasedTokens = Double(wordCount) / 0.75
        
        return Int(max(charBasedTokens, wordBasedTokens))
    }
    
    private func createStreamingResponse(req: Request, chatRequest: ChatCompletionRequest, foundationService: FoundationModelService) async throws -> Response {
        let httpResponse = Response(status: .ok)
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
                
                // Use temperature from API request if provided, otherwise use CLI parameter
                let effectiveTemperature = chatRequest.temperature ?? self.temperature
                let effectiveRandomness = self.randomness

                // Get response with proper timing measurement
                let (content, promptTime) = try await foundationService.generateStreamingResponseWithTiming(for: chatRequest.messages, temperature: effectiveTemperature, randomness: effectiveRandomness)
                
                // Start streaming timing
                let completionStartTime = Date()
                var isFirst = true
                var completionTokens = 0
                
                // ChatGPT-style smooth streaming: token-by-token with natural timing
                try await streamContentSmoothly(
                    content: content,
                    streamId: streamId,
                    model: chatRequest.model ?? "foundation",
                    encoder: encoder,
                    writer: writer,
                    isFirst: &isFirst,
                    completionTokens: &completionTokens
                )
                
                // Calculate timing metrics
                let completionTime = Date().timeIntervalSince(completionStartTime)
                let promptTokens = estimateTokens(for: chatRequest.messages)
                
                let usage = StreamUsage(
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    completionTime: completionTime,
                    promptTime: promptTime
                )
                
                // Send final chunk with metrics
                let finalChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: chatRequest.model ?? "foundation",
                    content: "",
                    isFinished: true,
                    usage: usage
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
    
    private func streamContentSmoothly(
        content: String,
        streamId: String,
        model: String,
        encoder: JSONEncoder,
        writer: any AsyncBodyStreamWriter,
        isFirst: inout Bool,
        completionTokens: inout Int
    ) async throws {
        // Handle code blocks specially to preserve formatting
        let codeBlockRanges = findCodeBlockRanges(in: content)
        var currentIndex = content.startIndex
        
        while currentIndex < content.endIndex {
            // Check if we're at the start of a code block
            if let codeBlockRange = codeBlockRanges.first(where: { $0.lowerBound == currentIndex }) {
                // Stream entire code block at once to preserve formatting
                let codeBlockContent = String(content[codeBlockRange])
                try await sendStreamChunk(
                    content: codeBlockContent,
                    streamId: streamId,
                    model: model,
                    encoder: encoder,
                    writer: writer,
                    isFirst: &isFirst,
                    completionTokens: &completionTokens
                )
                currentIndex = codeBlockRange.upperBound
                
                // Brief pause after code blocks
                try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            } else {
                // Stream character by character or small tokens for smooth flow
                let remainingContent = String(content[currentIndex...])
                let nextChunk = getNextStreamingChunk(from: remainingContent, codeBlockRanges: codeBlockRanges, currentIndex: currentIndex, fullContent: content)
                
                if !nextChunk.isEmpty {
                    try await sendStreamChunk(
                        content: nextChunk,
                        streamId: streamId,
                        model: model,
                        encoder: encoder,
                        writer: writer,
                        isFirst: &isFirst,
                        completionTokens: &completionTokens
                    )
                    
                    // Natural streaming delay - varies based on content type
                    let delay = getStreamingDelay(for: nextChunk)
                    if delay > 0 {
                        try await Task.sleep(nanoseconds: delay)
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
    
    private func sendStreamChunk(
        content: String,
        streamId: String,
        model: String,
        encoder: JSONEncoder,
        writer: any AsyncBodyStreamWriter,
        isFirst: inout Bool,
        completionTokens: inout Int
    ) async throws {
        let streamChunk = ChatCompletionStreamResponse(
            id: streamId,
            model: model,
            content: content,
            isFirst: isFirst
        )
        isFirst = false
        completionTokens += estimateTokens(for: content)
        
        let chunkData = try encoder.encode(streamChunk)
        if let jsonString = String(data: chunkData, encoding: .utf8) {
            try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
        }
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
}
