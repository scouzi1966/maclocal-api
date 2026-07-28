import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXVLM
import Tokenizers

public struct AFMMLXRuntimeAdapter {
    public typealias RuntimeEvent = AFMMLXRuntimeEvent
    public nonisolated static let imageProcessingSize = AFMMLXRuntimePolicy.defaultImageProcessingSize

    public struct LoadedContainer {
        public let container: ModelContainer
        public let isVision: Bool
        public let loadTime: Double

        public init(
            container: ModelContainer,
            isVision: Bool,
            loadTime: Double
        ) {
            self.container = container
            self.isVision = isVision
            self.loadTime = loadTime
        }
    }

    public init() {}

    @MainActor public func makeUserInput(
        chat: [Chat.Message],
        additionalContext: [String: any Sendable]?
    ) -> UserInput {
        UserInput(
            chat: chat,
            processing: .init(
                resize: .init(
                    width: Self.imageProcessingSize,
                    height: Self.imageProcessingSize
                )
            ),
            additionalContext: additionalContext
        )
    }

    @MainActor public func runGeneration(
        container: ModelContainer,
        chat: [Chat.Message],
        additionalContext: [String: any Sendable]?,
        parameters parameterRequest: AFMMLXGenerationParameterRequest,
        onEvent: (RuntimeEvent) async throws -> Bool
    ) async throws {
        let userInput = makeUserInput(
            chat: chat,
            additionalContext: additionalContext
        )
        let input = try await container.prepare(input: userInput)
        let parameters = AFMMLXGenerationParameterFactory.make(parameterRequest)
        let stream = try await container.generate(input: input, parameters: parameters)

        for await item in stream {
            let shouldContinue: Bool
            switch item {
            case .chunk(let string):
                shouldContinue = try await onEvent(.chunk(string))
            case .info(let info):
                shouldContinue = try await onEvent(
                    .info(
                        AFMMLXRuntimeInfo(
                            promptTime: info.promptTime,
                            tokensPerSecond: info.tokensPerSecond
                        )
                    )
                )
            case .toolCall:
                shouldContinue = try await onEvent(.toolCall)
            case .tokenLogprobs:
                shouldContinue = try await onEvent(.tokenLogprobs)
            }

            if !shouldContinue {
                break
            }
        }
    }

    @MainActor public func loadBenchmarkContainer(
        modelPath: String,
        logger: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> LoadedContainer {
        let modelURL = URL(fileURLWithPath: modelPath)
        let configuration = ModelConfiguration(directory: modelURL)
        let isVision = Self.pathSuggestsVisionModel(modelPath)
        logger("Model type: \(isVision ? "VLM" : "LLM")")
        logger("Loading model...")

        let loadStart = CFAbsoluteTimeGetCurrent()
        let container: ModelContainer
        if isVision {
            container = try await VLMModelFactory.shared.loadContainer(configuration: configuration)
        } else {
            container = try await LLMModelFactory.shared.loadContainer(configuration: configuration)
        }
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        logger("Model loaded in \(String(format: "%.2f", loadTime))s")
        return LoadedContainer(container: container, isVision: isVision, loadTime: loadTime)
    }

    @MainActor public func runPromptStreamBenchmark(
        container: ModelContainer,
        prompt: String,
        parameters parameterRequest: AFMMLXGenerationParameterRequest,
        logger: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> Double {
        try await container.perform { context in
            let userInput = UserInput(prompt: prompt)
            let input = try await context.processor.prepare(input: userInput)
            logger("Prompt tokens: \(input.text.tokens.size)")

            let parameters = AFMMLXGenerationParameterFactory.make(parameterRequest)
            var completionInfo: GenerateCompletionInfo?
            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: context
            )

            for await generation in stream {
                switch generation {
                case .chunk:
                    break
                case .info(let info):
                    completionInfo = info
                default:
                    break
                }
            }

            if let info = completionInfo {
                logger("Generation: \(info.generationTokenCount) tokens, \(String(format: "%.3f", info.tokensPerSecond)) tok/s")
                return info.tokensPerSecond
            }
            return 0
        }
    }

    @MainActor public func runPromptTokenIteratorBenchmark(
        container: ModelContainer,
        prompt: String,
        parameters parameterRequest: AFMMLXGenerationParameterRequest,
        logger: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> Double {
        try await container.perform { context in
            let userInput = UserInput(prompt: prompt)
            let input = try await context.processor.prepare(input: userInput)
            logger("Prompt tokens: \(input.text.tokens.size)")

            let parameters = AFMMLXGenerationParameterFactory.make(parameterRequest)
            let iterator = try TokenIterator(
                input: input,
                model: context.model,
                parameters: parameters
            )

            var eosTokenIds = context.configuration.eosTokenIds
            if let tokenizerEos = context.tokenizer.eosTokenId {
                eosTokenIds.insert(tokenizerEos)
            }

            var tokenCount = 0
            let startTime = CFAbsoluteTimeGetCurrent()
            for token in iterator {
                if token == context.tokenizer.unknownTokenId || eosTokenIds.contains(token) {
                    break
                }
                tokenCount += 1
            }

            let generateTime = CFAbsoluteTimeGetCurrent() - startTime
            let tokensPerSecond = Double(tokenCount) / generateTime
            MLX.Stream().synchronize()
            logger("Generation: \(tokenCount) tokens, \(String(format: "%.3f", tokensPerSecond)) tok/s")
            return tokensPerSecond
        }
    }

    public nonisolated static func pathSuggestsVisionModel(_ modelPath: String) -> Bool {
        let lowercased = modelPath.lowercased()
        return lowercased.contains("-vl-")
            || lowercased.contains("-vl_")
            || lowercased.contains("vision")
    }
}
