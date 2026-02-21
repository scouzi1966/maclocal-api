import Foundation
import MLX
import Cmlx
import MLXLLM
import MLXVLM
import MLXLMCommon
import Hub

enum MLXLoadStage: String {
    case checkingCache = "checking cache"
    case downloading = "downloading"
    case loadingModel = "loading model"
    case ready = "ready"
}

enum MLXServiceError: Error, LocalizedError {
    case invalidModel(String)
    case modelNotFoundInCache(String)
    case downloadFailed(String)
    case loadFailed(String)
    case noModelLoaded
    case serviceShuttingDown

    var errorDescription: String? {
        switch self {
        case .invalidModel(let value):
            return "Invalid model identifier: \(value)"
        case .modelNotFoundInCache(let value):
            return "Model not found in cache: \(value)"
        case .downloadFailed(let value):
            return "Failed to download model: \(value)"
        case .loadFailed(let value):
            return "Failed to load model: \(value)"
        case .noModelLoaded:
            return "No MLX model loaded"
        case .serviceShuttingDown:
            return "MLX service is shutting down"
        }
    }
}

final class MLXModelService: @unchecked Sendable {
    private let resolver: MLXCacheResolver
    private let registry = MLXModelRegistry()
    private let stateLock = NSLock()
    private var currentModelID: String?
    private var currentContainer: ModelContainer?
    private var activeOperations: Int = 0
    private var isShuttingDown = false
    private var gpuInitialized = false
    init(resolver: MLXCacheResolver) {
        self.resolver = resolver
        self.resolver.applyEnvironment()
    }

    /// Configure MLX GPU settings once, before first model load.
    /// Must be called after Metal is available (not during early init).
    private func ensureGPUConfigured() {
        guard !gpuInitialized else { return }
        gpuInitialized = true

        let totalMemoryGB = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        let cacheMB: Int
        switch totalMemoryGB {
        case 0..<12:  cacheMB = 128
        case 12..<24: cacheMB = 256
        case 24..<48: cacheMB = 512
        default:      cacheMB = 1024
        }
        Memory.cacheLimit = cacheMB * 1024 * 1024

        let maxWorkingSet = GPU.deviceInfo().maxRecommendedWorkingSetSize
        let wiredLimitBytes = Int(Double(maxWorkingSet) * 0.9)
        var previousWired: size_t = 0
        mlx_set_wired_limit(&previousWired, size_t(wiredLimitBytes))

        print("MLX GPU: cache=\(cacheMB)MB wired=\(wiredLimitBytes / (1024*1024))MB (system \(totalMemoryGB)GB)")
    }

    func normalizeModel(_ raw: String) -> String {
        resolver.normalizedModelID(raw)
    }

    func revalidateRegistry() throws -> [String] {
        try registry.revalidate(using: resolver)
    }

    func ensureLoaded(
        model rawModel: String,
        progress: (@Sendable (Progress) -> Void)? = nil,
        stage: (@Sendable (MLXLoadStage) -> Void)? = nil,
        countOperation: Bool = true
    ) async throws -> String {
        var didBeginOperation = false
        if countOperation {
            try beginOperation()
            didBeginOperation = true
        }
        defer {
            if didBeginOperation {
                endOperation()
            }
        }

        let modelID = normalizeModel(rawModel)
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MLXServiceError.invalidModel(rawModel)
        }
        stage?(.checkingCache)

        if let cached = withStateLock({ () -> (String, ModelContainer)? in
            guard currentModelID == modelID, let container = currentContainer else { return nil }
            return (modelID, container)
        }) {
            stage?(.ready)
            return cached.0
        }

        ensureGPUConfigured()

        if resolver.localModelDirectory(repoId: modelID) == nil {
            stage?(.downloading)
            try await downloadModel(modelID: modelID, progress: progress)
        }

        guard let directory = resolver.localModelDirectory(repoId: modelID) else {
            throw MLXServiceError.modelNotFoundInCache(modelID)
        }

        let config = ModelConfiguration(directory: directory)
        let isVLM = try isVisionModel(directory: directory)
        stage?(.loadingModel)
        do {
            let loaded: ModelContainer
            if isVLM {
                loaded = try await VLMModelFactory.shared.loadContainer(configuration: config)
            } else {
                loaded = try await LLMModelFactory.shared.loadContainer(configuration: config)
            }
            withStateLock {
                currentContainer = loaded
                currentModelID = modelID
            }
            try registry.registerModel(modelID)
            stage?(.ready)
            return modelID
        } catch {
            throw MLXServiceError.loadFailed("\(modelID): \(error.localizedDescription)")
        }
    }

    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?
    ) async throws -> (modelID: String, content: String, promptTokens: Int, completionTokens: Int) {
        try beginOperation()
        defer { endOperation() }

        let modelID = try await ensureLoaded(model: model, countOperation: false)
        guard let container = withStateLock({ currentContainer }) else { throw MLXServiceError.noModelLoaded }

        let promptText = buildPrompt(from: messages)
        let userInput = try buildUserInput(from: messages)
        let params = GenerateParameters(
            maxTokens: maxTokens ?? 2000,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: normalizedTemperature(temperature),
            topP: normalizedTopP(topP),
            repetitionPenalty: normalizedRepetitionPenalty(repetitionPenalty),
            repetitionContextSize: 64,
            prefillStepSize: 2048
        )

        let generated: String = try await container.perform { context in
            let input = try await context.processor.prepare(input: userInput)

            // If the chat template appended <think>, prepend it so extractors can detect it
            let tokens = input.text.tokens
            let ndim = tokens.ndim
            let seqLen = tokens.dim(ndim - 1)
            var out = ""
            if seqLen >= 2 {
                let flat = tokens.reshaped(-1)
                let lastTwo = flat[seqLen - 2 ..< seqLen].asArray(Int.self)
                let decoded = context.tokenizer.decode(tokens: lastTwo)
                if decoded.contains("<think>") {
                    out = "<think>"
                }
            }

            for await piece in try MLXLMCommon.generate(input: input, parameters: params, context: context) {
                if case .chunk(let text) = piece {
                    out += text
                }
            }

            Stream.gpu.synchronize()
            return out
        }

        return (modelID, generated, estimateTokens(promptText), estimateTokens(generated))
    }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?
    ) async throws -> (modelID: String, stream: AsyncThrowingStream<String, Error>, promptTokens: Int) {
        try beginOperation()
        defer { endOperation() }

        let modelID = try await ensureLoaded(model: model, countOperation: false)
        guard let container = withStateLock({ currentContainer }) else { throw MLXServiceError.noModelLoaded }

        let promptText = buildPrompt(from: messages)
        let userInput = try buildUserInput(from: messages)
        let promptTokens = estimateTokens(promptText)
        let params = GenerateParameters(
            maxTokens: maxTokens ?? 2000,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: normalizedTemperature(temperature),
            topP: normalizedTopP(topP),
            repetitionPenalty: normalizedRepetitionPenalty(repetitionPenalty),
            repetitionContextSize: 64,
            prefillStepSize: 2048
        )

        let stream = AsyncThrowingStream<String, Error> { continuation in
            let task = Task {
                do {
                    try await container.perform { context in
                        let input = try await context.processor.prepare(input: userInput)

                        // If the chat template appended <think> to the prompt, inject it
                        // into the stream so the reasoning extractor can detect it.
                        let tokens = input.text.tokens
                        let ndim = tokens.ndim
                        let seqLen = tokens.dim(ndim - 1)
                        if seqLen >= 2 {
                            let flat = tokens.reshaped(-1)
                            let lastTwo = flat[seqLen - 2 ..< seqLen].asArray(Int.self)
                            let decoded = context.tokenizer.decode(tokens: lastTwo)
                            if decoded.contains("<think>") {
                                continuation.yield("<think>")
                            }
                        }

                        for await piece in try MLXLMCommon.generate(input: input, parameters: params, context: context) {
                            if Task.isCancelled {
                                print("[MLX] Generation cancelled by client")
                                break
                            }
                            if case .chunk(let text) = piece {
                                continuation.yield(text)
                            }
                        }
                        // Synchronize GPU after generation completes (or breaks early).
                        Stream.gpu.synchronize()
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }

        return (modelID, stream, promptTokens)
    }

    func shutdownAndReleaseResources(verbose: Bool = false, timeoutSeconds: TimeInterval = 30) async {
        let start = Date()
        withStateLock { isShuttingDown = true }

        while Date().timeIntervalSince(start) < timeoutSeconds {
            if withStateLock({ activeOperations == 0 }) {
                break
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }

        autoreleasepool {
            withStateLock {
                currentContainer = nil
                currentModelID = nil
            }
        }

        // Ensure queued GPU work is complete before clearing recycled buffers.
        Stream.gpu.synchronize()
        Stream.cpu.synchronize()
        Memory.clearCache()
        Stream.gpu.synchronize()
        Memory.clearCache()

        if verbose {
            let snapshot = Memory.snapshot()
            print("MLX memory after shutdown - active: \(formatBytes(snapshot.activeMemory)), cache: \(formatBytes(snapshot.cacheMemory)), peak: \(formatBytes(snapshot.peakMemory))")
        }
    }

    private func beginOperation() throws {
        try withStateLock {
            if isShuttingDown {
                throw MLXServiceError.serviceShuttingDown
            }
            activeOperations += 1
        }
    }

    private func endOperation() {
        withStateLock {
            activeOperations = max(0, activeOperations - 1)
        }
    }

    private func withStateLock<T>(_ body: () throws -> T) rethrows -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return try body()
    }

    private func formatBytes(_ bytes: Int) -> String {
        let gb = Double(bytes) / 1_073_741_824.0
        return String(format: "%.2f GB", gb)
    }

    private func downloadModel(modelID: String, progress: (@Sendable (Progress) -> Void)?) async throws {
        do {
            _ = try await Hub.snapshot(
                from: modelID,
                matching: ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "tokenizer*", "*.bpe", "*.bin"],
                progressHandler: { p in progress?(p) }
            )
        } catch {
            throw MLXServiceError.downloadFailed("\(modelID): \(error.localizedDescription)")
        }
    }

    private func isVisionModel(directory: URL) throws -> Bool {
        let config = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: config),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        let modelType = (json["model_type"] as? String ?? "").lowercased()
        if modelType.contains("vl") || modelType.contains("vision") {
            return true
        }
        // Multimodal models (e.g. gemma3) have both text_config and vision_config
        if json["text_config"] != nil && json["vision_config"] != nil {
            return true
        }
        // Some VLMs (e.g. Qwen3.5-MoE) have vision token IDs without a vision_config block
        if json["image_token_id"] != nil || json["vision_start_token_id"] != nil {
            return true
        }
        return false
    }

    private func buildPrompt(from messages: [Message]) -> String {
        messages.map { "\($0.role): \($0.textContent)" }.joined(separator: "\n")
    }

    private func buildUserInput(from messages: [Message]) throws -> UserInput {
        var chatMessages: [Chat.Message] = []
        var hasSystemMessage = false
        for m in messages {
            let text = m.textContent
            let images = try extractImages(from: m)
            switch m.role {
            case "system", "developer":
                hasSystemMessage = true
                chatMessages.append(.system(text))
            case "assistant":
                chatMessages.append(.assistant(text))
            default:
                chatMessages.append(.user(text, images: images))
            }
        }

        // Align with Vesta behavior: always include a base system instruction
        // when callers don't explicitly provide one.
        if !hasSystemMessage {
            chatMessages.insert(.system("You are a helpful assistant!"), at: 0)
        }

        if chatMessages.isEmpty {
            return UserInput(prompt: "")
        }

        return UserInput(chat: chatMessages, processing: .init(resize: .init(width: 1024, height: 1024)))
    }

    private func extractImages(from message: Message) throws -> [UserInput.Image] {
        guard case .parts(let parts) = message.content else { return [] }
        var images: [UserInput.Image] = []
        for part in parts where part.type == "image_url" {
            guard let raw = part.image_url?.url, let url = URL(string: raw) else { continue }
            if let scheme = url.scheme, scheme == "http" || scheme == "https" {
                let (data, _) = try awaitURL(url: url)
                let temp = FileManager.default.temporaryDirectory
                    .appendingPathComponent("afm_mlx_image_\(UUID().uuidString).\(url.pathExtension.isEmpty ? "jpg" : url.pathExtension)")
                try data.write(to: temp)
                images.append(.url(temp))
            } else {
                images.append(.url(url))
            }
        }
        return images
    }

    private func awaitURL(url: URL) throws -> (Data, URLResponse) {
        let sem = DispatchSemaphore(value: 0)
        var result: Result<(Data, URLResponse), Error>?
        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let error {
                result = .failure(error)
            } else if let data, let response {
                result = .success((data, response))
            } else {
                result = .failure(MLXServiceError.downloadFailed("image download failed"))
            }
            sem.signal()
        }
        task.resume()
        sem.wait()
        switch result {
        case .success(let pair):
            return pair
        case .failure(let error):
            throw error
        case .none:
            throw MLXServiceError.downloadFailed("image download failed")
        }
    }

    private func estimateTokens(_ text: String) -> Int {
        let words = text.split(whereSeparator: \.isWhitespace).count
        let charBased = Double(text.count) / 4.0
        let wordBased = Double(words) / 0.75
        return Int(max(charBased, wordBased))
    }

    private func normalizedRepetitionPenalty(_ value: Double?) -> Float? {
        guard let value else { return nil }
        if abs(value - 1.0) < 0.000_001 {
            return nil
        }
        return Float(value)
    }

    private func normalizedTopP(_ value: Double?) -> Float {
        guard let value else { return 1.0 }  // MLX library default
        return Float(min(max(value, 0.0), 1.0))
    }

    private func normalizedTemperature(_ value: Double?) -> Float {
        guard let value else { return 0.6 }  // MLX library default
        return Float(min(max(value, 0.0), 1.0))
    }

}
