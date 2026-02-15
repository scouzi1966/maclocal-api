import Foundation
import MLXLLM
import MLXVLM
import MLXLMCommon
import Hub

enum MLXServiceError: Error, LocalizedError {
    case invalidModel(String)
    case modelNotFoundInCache(String)
    case downloadFailed(String)
    case loadFailed(String)
    case noModelLoaded

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
        }
    }
}

final class MLXModelService: @unchecked Sendable {
    private let resolver: MLXCacheResolver
    private let registry = MLXModelRegistry()
    private var currentModelID: String?
    private var currentContainer: ModelContainer?

    init(resolver: MLXCacheResolver) {
        self.resolver = resolver
        self.resolver.applyEnvironment()
    }

    func normalizeModel(_ raw: String) -> String {
        resolver.normalizedModelID(raw)
    }

    func revalidateRegistry() throws -> [String] {
        try registry.revalidate(using: resolver)
    }

    func ensureLoaded(model rawModel: String, progress: (@Sendable (Progress) -> Void)? = nil) async throws -> String {
        let modelID = normalizeModel(rawModel)
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MLXServiceError.invalidModel(rawModel)
        }

        if currentModelID == modelID, currentContainer != nil {
            return modelID
        }

        if resolver.localModelDirectory(repoId: modelID) == nil {
            try await downloadModel(modelID: modelID, progress: progress)
        }

        guard let directory = resolver.localModelDirectory(repoId: modelID) else {
            throw MLXServiceError.modelNotFoundInCache(modelID)
        }

        let config = ModelConfiguration(directory: directory)
        let isVLM = try isVisionModel(directory: directory)
        do {
            if isVLM {
                currentContainer = try await VLMModelFactory.shared.loadContainer(configuration: config)
            } else {
                currentContainer = try await LLMModelFactory.shared.loadContainer(configuration: config)
            }
            currentModelID = modelID
            try registry.registerModel(modelID)
            return modelID
        } catch {
            throw MLXServiceError.loadFailed("\(modelID): \(error.localizedDescription)")
        }
    }

    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?
    ) async throws -> (modelID: String, content: String, promptTokens: Int, completionTokens: Int) {
        let modelID = try await ensureLoaded(model: model)
        guard let container = currentContainer else { throw MLXServiceError.noModelLoaded }

        let promptText = buildPrompt(from: messages)
        let userInput = try buildUserInput(from: messages)
        let params = GenerateParameters(
            maxTokens: maxTokens ?? 2048,
            temperature: Float(temperature ?? 0.7),
            topP: 0.95
        )

        let generated: String = try await container.perform { context in
            let input = try await context.processor.prepare(input: userInput)
            var out = ""
            for await piece in try MLXLMCommon.generate(input: input, parameters: params, context: context) {
                if case .chunk(let text) = piece {
                    out += text
                }
            }
            return out
        }

        return (modelID, generated, estimateTokens(promptText), estimateTokens(generated))
    }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?
    ) async throws -> (modelID: String, stream: AsyncThrowingStream<String, Error>, promptTokens: Int) {
        let modelID = try await ensureLoaded(model: model)
        guard let container = currentContainer else { throw MLXServiceError.noModelLoaded }

        let promptText = buildPrompt(from: messages)
        let userInput = try buildUserInput(from: messages)
        let promptTokens = estimateTokens(promptText)
        let params = GenerateParameters(
            maxTokens: maxTokens ?? 2048,
            temperature: Float(temperature ?? 0.7),
            topP: 0.95
        )

        let stream = AsyncThrowingStream<String, Error> { continuation in
            Task {
                do {
                    try await container.perform { context in
                        let input = try await context.processor.prepare(input: userInput)
                        for await piece in try MLXLMCommon.generate(input: input, parameters: params, context: context) {
                            if case .chunk(let text) = piece {
                                continuation.yield(text)
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        return (modelID, stream, promptTokens)
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
        return modelType.contains("vl") || modelType.contains("vision")
    }

    private func buildPrompt(from messages: [Message]) -> String {
        messages.map { "\($0.role): \($0.textContent)" }.joined(separator: "\n")
    }

    private func buildUserInput(from messages: [Message]) throws -> UserInput {
        var chatMessages: [Chat.Message] = []
        for m in messages {
            let text = m.textContent
            let images = try extractImages(from: m)
            switch m.role {
            case "system":
                chatMessages.append(.system(text))
            case "assistant":
                chatMessages.append(.assistant(text))
            default:
                chatMessages.append(.user(text, images: images))
            }
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
}
