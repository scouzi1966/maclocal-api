import Foundation
import AFMOpenAICompat

#if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
import FoundationModels
#endif

// Parsed randomness parameter structure
//
// This structure represents the randomness configuration for Apple Foundation Models generation.
// It supports the sampling modes available in Apple's GenerationOptions API:
//
// Design Constraints (per Apple Foundation Models API):
// - Only ONE sampling method can be active at a time (greedy, random, top-p, OR top-k)
// - top-p and top-k cannot be combined in a single request
// - Seeds are optional and can be combined with any sampling method for reproducibility
//
// Supported Formats:
// - "greedy" - Deterministic sampling (always selects most likely token)
// - "random" - Apple's default random sampling
// - "random:top-p=<0.0-1.0>" - Nucleus sampling with probability threshold
// - "random:top-k=<int>" - Top-k sampling limiting to K most likely tokens
// - "random:seed=<int>" - Random sampling with specific seed
// - "random:top-p=0.9:seed=42" - Nucleus sampling with seed (combining is allowed)
// - "random:top-k=50:seed=42" - Top-k sampling with seed (combining is allowed)
//
// Invalid Combinations:
// - "random:top-p=0.9:top-k=50" - REJECTED: Cannot mix sampling methods
public struct RandomnessConfig {
    enum SamplingMode {
        case greedy
        case random
        case topP(Double)  // Nucleus sampling: 0.0-1.0 probability threshold
        case topK(Int)     // Top-k sampling: positive integer for k value
    }

    let mode: SamplingMode
    let seed: UInt64?

    public static func parse(_ randomnessString: String) throws -> RandomnessConfig {
        let trimmed = randomnessString.trimmingCharacters(in: .whitespacesAndNewlines)

        // Handle simple cases (backward compatibility)
        if trimmed == "greedy" {
            return RandomnessConfig(mode: .greedy, seed: nil)
        }
        if trimmed == "random" {
            return RandomnessConfig(mode: .random, seed: nil)
        }

        // Parse structured format: "random:top-p=0.9:seed=42"
        let components = trimmed.components(separatedBy: ":")
        guard components.count >= 1, components[0] == "random" else {
            throw FoundationModelError.invalidRandomnessParameter("Randomness must start with 'greedy', 'random', or 'random:...'")
        }

        var mode: SamplingMode = .random
        var seed: UInt64? = nil
        var hasSamplingParameter = false

        // Parse additional parameters
        // NOTE: Apple Foundation Models API does not support combining top-p and top-k simultaneously
        for i in 1..<components.count {
            let param = components[i]
            if param.hasPrefix("top-p=") {
                // Check for conflicting sampling parameters
                if hasSamplingParameter {
                    throw FoundationModelError.invalidRandomnessParameter("Cannot combine top-p and top-k sampling parameters. Apple Foundation Models API supports only one sampling method at a time.")
                }

                let valueStr = String(param.dropFirst(6))
                guard let value = Double(valueStr), value >= 0.0, value <= 1.0 else {
                    throw FoundationModelError.invalidRandomnessParameter("top-p must be between 0.0 and 1.0")
                }
                mode = .topP(value)
                hasSamplingParameter = true
            } else if param.hasPrefix("top-k=") {
                // Check for conflicting sampling parameters
                if hasSamplingParameter {
                    throw FoundationModelError.invalidRandomnessParameter("Cannot combine top-p and top-k sampling parameters. Apple Foundation Models API supports only one sampling method at a time.")
                }

                let valueStr = String(param.dropFirst(6))
                guard let value = Int(valueStr), value > 0 else {
                    throw FoundationModelError.invalidRandomnessParameter("top-k must be a positive integer")
                }
                mode = .topK(value)
                hasSamplingParameter = true
            } else if param.hasPrefix("seed=") {
                let valueStr = String(param.dropFirst(5))
                guard let value = UInt64(valueStr) else {
                    throw FoundationModelError.invalidRandomnessParameter("seed must be a non-negative integer")
                }
                seed = value
            } else {
                throw FoundationModelError.invalidRandomnessParameter("Unknown parameter: \(param)")
            }
        }

        return RandomnessConfig(mode: mode, seed: seed)
    }
}

public enum FoundationModelError: Error, LocalizedError {
    private static let structuredTruncationMarker = "Failed to deserialize a Generable type from model output"

    case notAvailable
    case sessionCreationFailed
    case responseGenerationFailed(String)
    case responseTruncated(maxTokens: Int)
    case invalidInput
    case invalidRandomnessParameter(String)
    case contextWindowExceeded(provided: Int, maximum: Int)
    case guardrailViolation(String)
    case schemaConversionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notAvailable:
            return "Foundation Models framework is not available. Requires macOS 26+ with Apple Intelligence enabled."
        case .sessionCreationFailed:
            return "Failed to create Foundation Models session. Ensure Apple Intelligence is enabled in System Settings."
        case .responseGenerationFailed(let message):
            return "Failed to generate response: \(message)"
        case .responseTruncated:
            return "Response generation reached the maximum token budget before a complete structured value was produced."
        case .invalidInput:
            return "Invalid input provided to Foundation Models"
        case .invalidRandomnessParameter(let message):
            return "Invalid randomness parameter: \(message)"
        case .contextWindowExceeded(let provided, let maximum):
            return "Context window exceeded: Your conversation has \(provided) tokens but the maximum is \(maximum). Please start a new conversation or reduce the message length."
        case .guardrailViolation(let message):
            return "Content policy violation: \(message)"
        case .schemaConversionFailed(let message):
            return "Schema conversion failed: \(message)"
        }
    }

    /// Check if an error is a guardrail violation and extract the message
    static func parseGuardrailError(_ error: Error) -> FoundationModelError? {
        let errorString = String(describing: error)
        if errorString.contains("guardrailViolation") || errorString.contains("unsafe content") {
            // Extract the debug description if available
            if let range = errorString.range(of: "debugDescription: \"") {
                let start = range.upperBound
                if let endRange = errorString[start...].range(of: "\"") {
                    let message = String(errorString[start..<endRange.lowerBound])
                    return .guardrailViolation(message)
                }
            }
            return .guardrailViolation("The request was blocked due to content policy restrictions.")
        }
        return nil
    }

    /// Check if an error is a context window exceeded error and extract token counts
    static func parseContextWindowError(_ error: Error) -> FoundationModelError? {
        let errorString = String(describing: error)
        guard errorString.contains("exceededContextWindowSize") || (errorString.contains("context") && errorString.contains("exceeds")) else {
            return nil
        }

        // Try to extract token counts from the error message
        // Pattern: "Provided 4,089 tokens, but the maximum allowed is 4,096"
        let providedPattern = try? NSRegularExpression(pattern: "Provided ([0-9,]+) tokens")
        let maxPattern = try? NSRegularExpression(pattern: "maximum allowed is ([0-9,]+)")

        var provided = 0
        var maximum = 4096 // Default

        if let match = providedPattern?.firstMatch(in: errorString, range: NSRange(errorString.startIndex..., in: errorString)),
           let range = Range(match.range(at: 1), in: errorString) {
            let numStr = String(errorString[range]).replacingOccurrences(of: ",", with: "")
            provided = Int(numStr) ?? 0
        }

        if let match = maxPattern?.firstMatch(in: errorString, range: NSRange(errorString.startIndex..., in: errorString)),
           let range = Range(match.range(at: 1), in: errorString) {
            let numStr = String(errorString[range]).replacingOccurrences(of: ",", with: "")
            maximum = Int(numStr) ?? 4096
        }

        return .contextWindowExceeded(provided: provided, maximum: maximum)
    }

    static func parseStructuredTruncationError(_ error: Error, maxTokens: Int?) -> FoundationModelError? {
        guard let maxTokens, maxTokens > 0 else { return nil }
        let errorString = String(describing: error)
        if errorString.contains(Self.structuredTruncationMarker) {
            return .responseTruncated(maxTokens: maxTokens)
        }
        return nil
    }
}

struct FoundationStopSequenceFilter {
    private let stopSequences: [String]
    private var pending = ""
    private(set) var stopped = false

    init(stopSequences: [String]?) {
        self.stopSequences = stopSequences?.filter { !$0.isEmpty } ?? []
    }

    mutating func consume(_ delta: String) -> String {
        guard !stopped else { return "" }
        guard !stopSequences.isEmpty else { return delta }
        pending += delta

        let earliest = stopSequences.compactMap { pending.range(of: $0) }
            .min { $0.lowerBound < $1.lowerBound }
        if let earliest {
            let output = String(pending[..<earliest.lowerBound])
            pending = ""
            stopped = true
            return output
        }

        let heldCount = stopSequences.reduce(0) { longest, sequence in
            let maximum = min(pending.count, max(0, sequence.count - 1))
            guard maximum > longest else { return longest }
            for count in stride(from: maximum, through: longest + 1, by: -1) {
                if pending.suffix(count) == sequence.prefix(count) { return count }
            }
            return longest
        }
        let emittedCount = pending.count - heldCount
        let output = String(pending.prefix(emittedCount))
        pending = String(pending.suffix(heldCount))
        return output
    }

    mutating func finish() -> String {
        guard !stopped else { return "" }
        let output = pending
        pending = ""
        return output
    }
}

@available(macOS 26.0, *)
public class FoundationModelService: @unchecked Sendable {
    
    #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
    private var session: LanguageModelSession?
    private let model: SystemLanguageModel
    private let instructions: String
    #endif
    
    // Shared singleton instance
    nonisolated(unsafe) static var shared: FoundationModelService?

    // Shared adapter for reuse across instances
    #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS && !DISABLE_FOUNDATION_MODELS
    nonisolated(unsafe) static var sharedAdapter: SystemLanguageModel.Adapter?
    #else
    nonisolated(unsafe) static var sharedAdapter: Any?
    #endif
    nonisolated(unsafe) static var sharedAdapterPath: String?
    
    public init(instructions: String = "You are a helpful assistant", adapter: String? = nil, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool) async throws {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS && !DISABLE_FOUNDATION_MODELS
        let fallbackModel = SystemLanguageModel(guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
        let selectedModel: SystemLanguageModel
        // Check if adapter path is provided
        if let adapterPath = adapter {
            do {
                // Expand tilde and resolve relative paths
                let expandedPath = NSString(string: adapterPath).expandingTildeInPath
                let adapterURL = URL(fileURLWithPath: expandedPath)
                
                // Validate adapter file exists and has correct extension
                guard FileManager.default.fileExists(atPath: adapterURL.path) else {
                    print("Warning: Adapter file not found at '\(adapterPath)', falling back to default model")
                    selectedModel = fallbackModel
                    self.model = selectedModel
                    self.instructions = instructions
                    self.session = LanguageModelSession(model: selectedModel) { instructions }
                    return
                }
                
                guard adapterURL.pathExtension.lowercased() == "fmadapter" else {
                    print("Warning: Adapter file must have .fmadapter extension, falling back to default model")
                    selectedModel = fallbackModel
                    self.model = selectedModel
                    self.instructions = instructions
                    self.session = LanguageModelSession(model: selectedModel) { instructions }
                    return
                }
                
                // Try to load the adapter
                let adapter = try SystemLanguageModel.Adapter(fileURL: adapterURL)
                selectedModel = SystemLanguageModel(adapter: adapter, guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
                
                // Store adapter for reuse if this is the first time loading
                if Self.sharedAdapter == nil {
                    Self.sharedAdapter = adapter
                    Self.sharedAdapterPath = adapterPath
                }
                
                print("✅ Successfully loaded LoRA adapter: \(adapterURL.lastPathComponent)")
                
            } catch {
                print("Warning: Failed to load adapter '\(adapterPath)': \(error.localizedDescription)")
                print("Falling back to default model")
                
                // Fallback to default model
                selectedModel = fallbackModel
            }
        } else {
            // No adapter specified, use default model
            selectedModel = fallbackModel
        }
        self.model = selectedModel
        self.instructions = instructions
        self.session = LanguageModelSession(model: selectedModel) { instructions }
        #else
        throw FoundationModelError.notAvailable
        #endif
    }
    
    // Private initializer for creating instances with shared adapter
    private init(instructions: String, useSharedAdapter: Bool, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool) async throws {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        let selectedModel: SystemLanguageModel
        if useSharedAdapter, let sharedAdapter = Self.sharedAdapter {
            // Use the shared adapter
            selectedModel = SystemLanguageModel(adapter: sharedAdapter, guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
        } else {
            // No shared adapter available, use default model
            selectedModel = SystemLanguageModel(guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
        }
        self.model = selectedModel
        self.instructions = instructions
        self.session = LanguageModelSession(model: selectedModel) { instructions }
        #else
        throw FoundationModelError.notAvailable
        #endif
    }

    /// Replaces the stateful native session and optionally restores a saved conversation.
    /// Callers must subsequently submit only messages that are not already in `history`.
    public func resetConversation(with history: [Message] = []) {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        guard !history.isEmpty else {
            session = LanguageModelSession(model: model) { instructions }
            return
        }
        session = LanguageModelSession(model: model, transcript: transcript(from: history))
        #endif
    }

    #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
    private func transcript(from messages: [Message]) -> Transcript {
        let restoredInstructions = messages
            .filter { $0.role == "system" || $0.role == "developer" }
            .map(\.textContent)
            .filter { !$0.isEmpty }
        let instructionText = ([instructions] + restoredInstructions).joined(separator: "\n\n")
        var entries: [Transcript.Entry] = [
            .instructions(.init(
                segments: [.text(.init(content: instructionText))],
                toolDefinitions: []
            ))
        ]
        for message in messages where message.role != "system" && message.role != "developer" {
            if message.role == "assistant" {
                let segment = Transcript.Segment.text(.init(content: message.textContent))
                entries.append(.response(.init(assetIDs: [], segments: [segment])))
            } else {
                let segment = Transcript.Segment.text(.init(content: formatMessagesAsPrompt([message])))
                entries.append(.prompt(.init(segments: [segment])))
            }
        }
        return Transcript(entries: entries)
    }

    static func acceptedTranscript(
        from previousTranscript: Transcript,
        prompt: String,
        response: String,
        options: GenerationOptions
    ) -> Transcript {
        var entries = Array(previousTranscript)
        entries.append(.prompt(.init(
            segments: [.text(.init(content: prompt))],
            options: options
        )))
        entries.append(.response(.init(
            assetIDs: [],
            segments: [.text(.init(content: response))]
        )))
        return Transcript(entries: entries)
    }

    private func restoreAcceptedResponse(
        previousTranscript: Transcript,
        prompt: String,
        response: String,
        options: GenerationOptions
    ) {
        session = LanguageModelSession(
            model: model,
            transcript: Self.acceptedTranscript(
                from: previousTranscript,
                prompt: prompt,
                response: response,
                options: options
            )
        )
    }
    #endif
    
    public func generateResponse(for messages: [Message], temperature: Double? = nil, randomness: String? = nil, maxTokens: Int? = nil, stop: [String]? = nil) async throws -> String {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        guard let session = session else {
            throw FoundationModelError.sessionCreationFailed
        }

        let prompt = formatMessagesAsPrompt(messages)

        do {
            let options = try createGenerationOptions(temperature: temperature, randomness: randomness, maxTokens: maxTokens)
            let previousTranscript = session.transcript
            let response = try await session.respond(to: prompt, options: options)
            let acceptedResponse = applyStopSequences(to: response.content, stopSequences: stop)
            if acceptedResponse != response.content {
                restoreAcceptedResponse(
                    previousTranscript: previousTranscript,
                    prompt: prompt,
                    response: acceptedResponse,
                    options: options
                )
            }
            return acceptedResponse
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // Check for context window exceeded error and wrap it
            if let contextError = FoundationModelError.parseContextWindowError(error) {
                throw contextError
            }
            // Check for guardrail violation
            if let guardrailError = FoundationModelError.parseGuardrailError(error) {
                throw guardrailError
            }
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
            case "system", "developer":
                prompt += "System: \(message.textContent)\n\n"
            case "user":
                prompt += "User: \(message.textContent)\n\n"
            case "assistant":
                prompt += "Assistant: \(message.textContent)\n\n"
            default:
                prompt += "\(message.textContent)\n\n"
            }
        }

        prompt += "Assistant: "
        return prompt
    }

    /// Pre-warm the session for faster first response
    public func prewarm() async throws {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        guard let session = session else {
            throw FoundationModelError.sessionCreationFailed
        }
        try await session.prewarm()
        #endif
    }

    /// Generate response with native streaming (real token-by-token output)
    public func generateNativeStreamingResponse(
        for messages: [Message],
        temperature: Double? = nil,
        randomness: String? = nil,
        maxTokens: Int? = nil,
        stop: [String]? = nil
    ) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task {
                #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
                guard let session = self.session else {
                    continuation.finish(throwing: FoundationModelError.sessionCreationFailed)
                    return
                }

                let prompt = self.formatMessagesAsPrompt(messages)

                do {
                    let options = try self.createGenerationOptions(temperature: temperature, randomness: randomness, maxTokens: maxTokens)
                    let previousTranscript = session.transcript
                    // Use native streaming API — partialResponse.content is cumulative,
                    // so we must extract only the new delta each iteration.
                    let stream = session.streamResponse(to: prompt, options: options)
                    var previousContent = ""
                    var acceptedResponse = ""
                    var stopFilter = FoundationStopSequenceFilter(stopSequences: stop)
                    for try await partialResponse in stream {
                        try Task.checkCancellation()
                        if stopFilter.stopped { break }
                        let full = partialResponse.content
                        if full.count > previousContent.count {
                            let delta = String(full.dropFirst(previousContent.count))
                            let output = stopFilter.consume(delta)
                            if !output.isEmpty {
                                acceptedResponse += output
                                continuation.yield(output)
                            }
                        }
                        previousContent = full
                    }
                    let finalOutput = stopFilter.finish()
                    if !finalOutput.isEmpty {
                        acceptedResponse += finalOutput
                        continuation.yield(finalOutput)
                    }
                    if stopFilter.stopped {
                        self.restoreAcceptedResponse(
                            previousTranscript: previousTranscript,
                            prompt: prompt,
                            response: acceptedResponse,
                            options: options
                        )
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    if let contextError = FoundationModelError.parseContextWindowError(error) {
                        continuation.finish(throwing: contextError)
                    } else if let guardrailError = FoundationModelError.parseGuardrailError(error) {
                        continuation.finish(throwing: guardrailError)
                    } else {
                        continuation.finish(throwing: FoundationModelError.responseGenerationFailed(error.localizedDescription))
                    }
                }
                #else
                continuation.finish(throwing: FoundationModelError.notAvailable)
                #endif
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
    
    /// Generate a guided (structured) response using constrained decoding.
    /// Converts the OpenAI JSON Schema to Apple's GenerationSchema internally.
    public func generateGuidedResponse(
        for messages: [Message],
        jsonSchema: ResponseJsonSchema,
        temperature: Double? = nil,
        randomness: String? = nil,
        maxTokens: Int? = nil,
        stop: [String]? = nil
    ) async throws -> String {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        guard let session = session else {
            throw FoundationModelError.sessionCreationFailed
        }

        let schema: GenerationSchema
        do {
            schema = try JSONSchemaConverter.convert(jsonSchema)
        } catch {
            throw FoundationModelError.schemaConversionFailed(error.localizedDescription)
        }

        let prompt = formatMessagesAsPrompt(messages)

        do {
            let options = try createGenerationOptions(temperature: temperature, randomness: randomness, maxTokens: maxTokens)
            let previousTranscript = session.transcript
            let response = try await session.respond(to: prompt, schema: schema, options: options)
            let rawResponse = response.content.jsonString
            let acceptedResponse = applyStopSequences(to: rawResponse, stopSequences: stop)
            if acceptedResponse != rawResponse {
                restoreAcceptedResponse(
                    previousTranscript: previousTranscript,
                    prompt: prompt,
                    response: acceptedResponse,
                    options: options
                )
            }
            return acceptedResponse
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            if let contextError = FoundationModelError.parseContextWindowError(error) {
                throw contextError
            }
            if let guardrailError = FoundationModelError.parseGuardrailError(error) {
                throw guardrailError
            }
            if let truncationError = FoundationModelError.parseStructuredTruncationError(error, maxTokens: maxTokens) {
                throw truncationError
            }
            throw FoundationModelError.responseGenerationFailed(error.localizedDescription)
        }
        #else
        throw FoundationModelError.notAvailable
        #endif
    }

    /// Generate a guided (structured) streaming response using constrained decoding.
    /// Converts the OpenAI JSON Schema to Apple's GenerationSchema internally.
    ///
    /// Note: Apple's guided generation streams partial JSON *snapshots* (the entire
    /// structure mutates, not just appends). We diff successive snapshots to emit
    /// append-only deltas that are compatible with SSE streaming consumers.
    public func generateGuidedStreamingResponse(
        for messages: [Message],
        jsonSchema: ResponseJsonSchema,
        temperature: Double? = nil,
        randomness: String? = nil,
        maxTokens: Int? = nil,
        stop: [String]? = nil
    ) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task {
                #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
                guard let session = self.session else {
                    continuation.finish(throwing: FoundationModelError.sessionCreationFailed)
                    return
                }

                let schema: GenerationSchema
                do {
                    schema = try JSONSchemaConverter.convert(jsonSchema)
                } catch {
                    continuation.finish(throwing: FoundationModelError.schemaConversionFailed(error.localizedDescription))
                    return
                }

                let prompt = self.formatMessagesAsPrompt(messages)

                do {
                    let options = try self.createGenerationOptions(temperature: temperature, randomness: randomness, maxTokens: maxTokens)
                    let previousTranscript = session.transcript
                    let stream = session.streamResponse(to: prompt, schema: schema, options: options)
                    var previousJson = ""
                    var processedPrefixCount = 0
                    var acceptedResponse = ""
                    var stopFilter = FoundationStopSequenceFilter(stopSequences: stop)
                    for try await partialResponse in stream {
                        try Task.checkCancellation()
                        if stopFilter.stopped { break }
                        let currentJson = partialResponse.content.jsonString
                        let stablePrefixCount = Self.commonPrefixLength(previousJson, currentJson)
                        if stablePrefixCount > processedPrefixCount {
                            let stablePrefix = String(currentJson.prefix(stablePrefixCount))
                            let delta = String(stablePrefix.dropFirst(processedPrefixCount))
                            let output = stopFilter.consume(delta)
                            if !output.isEmpty {
                                acceptedResponse += output
                                continuation.yield(output)
                            }
                            processedPrefixCount = stablePrefixCount
                        }
                        previousJson = currentJson
                    }
                    if !stopFilter.stopped, previousJson.count > processedPrefixCount {
                        let output = stopFilter.consume(String(previousJson.dropFirst(processedPrefixCount)))
                        if !output.isEmpty {
                            acceptedResponse += output
                            continuation.yield(output)
                        }
                    }
                    let finalOutput = stopFilter.finish()
                    if !finalOutput.isEmpty {
                        acceptedResponse += finalOutput
                        continuation.yield(finalOutput)
                    }
                    if stopFilter.stopped {
                        self.restoreAcceptedResponse(
                            previousTranscript: previousTranscript,
                            prompt: prompt,
                            response: acceptedResponse,
                            options: options
                        )
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    if let contextError = FoundationModelError.parseContextWindowError(error) {
                        continuation.finish(throwing: contextError)
                    } else if let guardrailError = FoundationModelError.parseGuardrailError(error) {
                        continuation.finish(throwing: guardrailError)
                    } else if let truncationError = FoundationModelError.parseStructuredTruncationError(error, maxTokens: maxTokens) {
                        continuation.finish(throwing: truncationError)
                    } else {
                        continuation.finish(throwing: FoundationModelError.responseGenerationFailed(error.localizedDescription))
                    }
                }
                #else
                continuation.finish(throwing: FoundationModelError.notAvailable)
                #endif
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
    private func createGenerationOptions(temperature: Double?, randomness: String?, maxTokens: Int? = nil) throws -> GenerationOptions {
        // Default to 2000 tokens when max_tokens is absent or non-positive.
        let effectiveMaxTokens: Int = if let mt = maxTokens, mt > 0 { mt } else { 2000 }
        guard let randomnessString = randomness else {
            // Default behavior when randomness is not specified
            return GenerationOptions(temperature: temperature, maximumResponseTokens: effectiveMaxTokens)
        }

        let config = try RandomnessConfig.parse(randomnessString)
        switch config.mode {
        case .greedy:
            return GenerationOptions(
                sampling: .greedy,
                temperature: temperature,
                maximumResponseTokens: effectiveMaxTokens
            )
        case .random:
            if let seed = config.seed {
                return GenerationOptions(
                    sampling: .random(probabilityThreshold: 1.0, seed: seed),
                    temperature: temperature,
                    maximumResponseTokens: effectiveMaxTokens
                )
            } else {
                return GenerationOptions(
                    temperature: temperature,
                    maximumResponseTokens: effectiveMaxTokens
                )
            }
        case .topP(let threshold):
            if let seed = config.seed {
                return GenerationOptions(
                    sampling: .random(probabilityThreshold: threshold, seed: seed),
                    temperature: temperature,
                    maximumResponseTokens: effectiveMaxTokens
                )
            } else {
                return GenerationOptions(
                    sampling: .random(probabilityThreshold: threshold),
                    temperature: temperature,
                    maximumResponseTokens: effectiveMaxTokens
                )
            }
        case .topK(let k):
            if let seed = config.seed {
                return GenerationOptions(
                    sampling: .random(top: k, seed: seed),
                    temperature: temperature,
                    maximumResponseTokens: effectiveMaxTokens
                )
            } else {
                return GenerationOptions(
                    sampling: .random(top: k),
                    temperature: temperature,
                    maximumResponseTokens: effectiveMaxTokens
                )
            }
        }
    }
    #endif
    
    /// Truncate content at the earliest occurrence of any stop sequence.
    /// The stop sequence itself is excluded from the output.
    private func applyStopSequences(to content: String, stopSequences: [String]?) -> String {
        guard let stopSequences = stopSequences, !stopSequences.isEmpty else {
            return content
        }
        var earliestIndex: String.Index? = nil
        for seq in stopSequences {
            if let range = content.range(of: seq) {
                if earliestIndex == nil || range.lowerBound < earliestIndex! {
                    earliestIndex = range.lowerBound
                }
            }
        }
        if let idx = earliestIndex {
            return String(content[..<idx])
        }
        return content
    }

    static func commonPrefixLength(_ lhs: String, _ rhs: String) -> Int {
        var count = 0
        for (left, right) in zip(lhs, rhs) {
            guard left == right else { break }
            count += 1
        }
        return count
    }

    static func isAvailable() -> Bool {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        return true
        #else
        return false
        #endif
    }
    
    // Initialize the shared instance once at server startup
    public static func initialize(instructions: String = "You are a helpful assistant", adapter: String? = nil, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool, prewarm: Bool = false) async throws {
        shared = try await FoundationModelService(instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails)
        if prewarm {
            print("🔥 Pre-warming model...")
            try await shared?.prewarm()
            print("✅ Model pre-warmed and ready")
        }
    }
    
    // Get the shared instance
    static func getShared() throws -> FoundationModelService {
        guard let shared = shared else {
            throw FoundationModelError.sessionCreationFailed
        }
        return shared
    }
    
    // Create a new instance that reuses the shared adapter (for per-request use)
    public static func createWithSharedAdapter(instructions: String = "You are a helpful assistant", temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool) async throws -> FoundationModelService {
        return try await FoundationModelService(instructions: instructions, useSharedAdapter: true, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails)
    }
}
