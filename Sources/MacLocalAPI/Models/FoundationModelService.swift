import Foundation

#if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS && !DISABLE_FOUNDATION_MODELS
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
struct RandomnessConfig {
    enum SamplingMode {
        case greedy
        case random
        case topP(Double)  // Nucleus sampling: 0.0-1.0 probability threshold
        case topK(Int)     // Top-k sampling: positive integer for k value
    }

    let mode: SamplingMode
    let seed: UInt64?

    static func parse(_ randomnessString: String) throws -> RandomnessConfig {
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

enum FoundationModelError: Error, LocalizedError {
    case notAvailable
    case sessionCreationFailed
    case responseGenerationFailed(String)
    case invalidInput
    case invalidRandomnessParameter(String)
    case contextWindowExceeded(provided: Int, maximum: Int)
    case guardrailViolation(String)

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
        case .invalidRandomnessParameter(let message):
            return "Invalid randomness parameter: \(message)"
        case .contextWindowExceeded(let provided, let maximum):
            return "Context window exceeded: Your conversation has \(provided) tokens but the maximum is \(maximum). Please start a new conversation or reduce the message length."
        case .guardrailViolation(let message):
            return "Content policy violation: \(message)"
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
}

@available(macOS 26.0, *)
class FoundationModelService {
    
    #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
    private var session: LanguageModelSession?
    #endif
    
    // Shared singleton instance
    static var shared: FoundationModelService?
    
    // Shared adapter for reuse across instances
    #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS && !DISABLE_FOUNDATION_MODELS
    static var sharedAdapter: SystemLanguageModel.Adapter?
    #else
    static var sharedAdapter: Any?
    #endif
    static var sharedAdapterPath: String?
    
    init(instructions: String = "You are a helpful assistant", adapter: String? = nil, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool) async throws {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS && !DISABLE_FOUNDATION_MODELS
        // Check if adapter path is provided
        if let adapterPath = adapter {
            do {
                // Expand tilde and resolve relative paths
                let expandedPath = NSString(string: adapterPath).expandingTildeInPath
                let adapterURL = URL(fileURLWithPath: expandedPath)
                
                // Validate adapter file exists and has correct extension
                guard FileManager.default.fileExists(atPath: adapterURL.path) else {
                    print("Warning: Adapter file not found at '\(adapterPath)', falling back to default model")
                    let model = SystemLanguageModel(guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
                    self.session = LanguageModelSession(model: model) {
                        instructions
                    }
                    return
                }
                
                guard adapterURL.pathExtension.lowercased() == "fmadapter" else {
                    print("Warning: Adapter file must have .fmadapter extension, falling back to default model")
                    let model = SystemLanguageModel(guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
                    self.session = LanguageModelSession(model: model) {
                        instructions
                    }
                    return
                }
                
                // Try to load the adapter
                let adapter = try SystemLanguageModel.Adapter(fileURL: adapterURL)
                let customModel = SystemLanguageModel(adapter: adapter, guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
                
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
                let model = SystemLanguageModel(guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
                self.session = LanguageModelSession(model: model) {
                    instructions
                }
            }
        } else {
            // No adapter specified, use default model
            let model = SystemLanguageModel(guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
            self.session = LanguageModelSession(model: model) {
                instructions
            }
        }
        #else
        throw FoundationModelError.notAvailable
        #endif
    }
    
    // Private initializer for creating instances with shared adapter
    private init(instructions: String, useSharedAdapter: Bool, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool) async throws {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        if useSharedAdapter, let sharedAdapter = Self.sharedAdapter {
            // Use the shared adapter
            let customModel = SystemLanguageModel(adapter: sharedAdapter, guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
            self.session = LanguageModelSession(model: customModel) {
                instructions
            }
        } else {
            // No shared adapter available, use default model
            let model = SystemLanguageModel(guardrails: permissiveGuardrails ? .permissiveContentTransformations : .default)
            self.session = LanguageModelSession(model: model) {
                instructions
            }
        }
        #else
        throw FoundationModelError.notAvailable
        #endif
    }
    
    func generateResponse(for messages: [Message], temperature: Double? = nil, randomness: String? = nil, maxTokens: Int? = nil) async throws -> String {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        guard let session = session else {
            throw FoundationModelError.sessionCreationFailed
        }

        let prompt = formatMessagesAsPrompt(messages)

        do {
            let options = try createGenerationOptions(temperature: temperature, randomness: randomness, maxTokens: maxTokens)
            let response = try await session.respond(to: prompt, options: options)
            return response.content
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
    func prewarm() async throws {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        guard let session = session else {
            throw FoundationModelError.sessionCreationFailed
        }
        try await session.prewarm()
        #endif
    }

    /// Generate response with native streaming (real token-by-token output)
    func generateNativeStreamingResponse(
        for messages: [Message],
        temperature: Double? = nil,
        randomness: String? = nil,
        maxTokens: Int? = nil
    ) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
                guard let session = self.session else {
                    continuation.finish(throwing: FoundationModelError.sessionCreationFailed)
                    return
                }

                let prompt = self.formatMessagesAsPrompt(messages)

                do {
                    let options = try self.createGenerationOptions(temperature: temperature, randomness: randomness, maxTokens: maxTokens)
                    // Use native streaming API — partialResponse.content is cumulative,
                    // so we must extract only the new delta each iteration.
                    let stream = session.streamResponse(to: prompt, options: options)
                    var previousContent = ""
                    for try await partialResponse in stream {
                        let full = partialResponse.content
                        if full.count > previousContent.count {
                            let delta = String(full.dropFirst(previousContent.count))
                            continuation.yield(delta)
                        }
                        previousContent = full
                    }
                    continuation.finish()
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
        }
    }
    
    #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
    private func createGenerationOptions(temperature: Double?, randomness: String?, maxTokens: Int? = nil) throws -> GenerationOptions {
        // Default to 2000 tokens when max_tokens is absent or non-positive.
        let effectiveMaxTokens: Int = if let mt = maxTokens, mt > 0 { mt } else { 2000 }
        DebugLogger.log("createGenerationOptions called with temperature: \(temperature?.description ?? "nil"), randomness: \(randomness ?? "nil"), maxTokens: \(effectiveMaxTokens)")

        guard let randomnessString = randomness else {
            // Default behavior when randomness is not specified
            return GenerationOptions(temperature: temperature, maximumResponseTokens: effectiveMaxTokens)
        }

        let config = try RandomnessConfig.parse(randomnessString)
        DebugLogger.log("Parsed randomness config: mode=\(config.mode), seed=\(config.seed?.description ?? "nil")")

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
    
    static func isAvailable() -> Bool {
        #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
        return true
        #else
        return false
        #endif
    }
    
    // Initialize the shared instance once at server startup
    static func initialize(instructions: String = "You are a helpful assistant", adapter: String? = nil, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool, prewarm: Bool = false) async throws {
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
    static func createWithSharedAdapter(instructions: String = "You are a helpful assistant", temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool) async throws -> FoundationModelService {
        return try await FoundationModelService(instructions: instructions, useSharedAdapter: true, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails)
    }
}
