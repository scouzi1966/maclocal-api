import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

@testable import AFMKitMLX

final class MLXSerialIgnoreEOSTests: XCTestCase {
    func testSerialStreamingStopsAtEOSByDefaultAndHonorsIgnoreEOS() async throws {
        let defaultTokenCount = try await streamingGenerationTokenCount(ignoreEOS: false)
        let ignoredEOSTokenCount = try await streamingGenerationTokenCount(ignoreEOS: true)

        XCTAssertEqual(defaultTokenCount, 0)
        XCTAssertEqual(ignoredEOSTokenCount, 3)
    }

    func testSerialNonStreamingStopsAtEOSByDefaultAndHonorsIgnoreEOS() async throws {
        let defaultTokenCount = try await nonStreamingGenerationTokenCount(ignoreEOS: false)
        let ignoredEOSTokenCount = try await nonStreamingGenerationTokenCount(ignoreEOS: true)

        XCTAssertEqual(defaultTokenCount, 0)
        XCTAssertEqual(ignoredEOSTokenCount, 3)
    }

    private func streamingGenerationTokenCount(ignoreEOS: Bool) async throws -> Int {
        let tokenizer = FixedEOSTokenizer()
        let model = FixedEOSTokenModel()
        var configuration = ModelConfiguration(id: "serial-eos-test")
        configuration.eosTokenIds = [FixedEOSTokenizer.eosTokenID]
        let container = ModelContainer(context: ModelContext(
            configuration: configuration,
            model: model,
            processor: StandInUserInputProcessor(),
            tokenizer: tokenizer
        ))
        let parameters = GenerateParameters(
            maxTokens: 3,
            temperature: 0,
            ignoreEndOfSequence: ignoreEOS
        )
        let input = LMInput(tokens: MLXArray([1]))
        let (stream, task) = try await container.generateTask(
            input: input,
            parameters: parameters
        )

        let tokenCount = await completionTokenCount(from: stream)
        await task.value
        return tokenCount
    }

    private func nonStreamingGenerationTokenCount(ignoreEOS: Bool) async throws -> Int {
        let tokenizer = FixedEOSTokenizer()
        let model = FixedEOSTokenModel()
        var configuration = ModelConfiguration(id: "serial-eos-test")
        configuration.eosTokenIds = [FixedEOSTokenizer.eosTokenID]
        let context = ModelContext(
            configuration: configuration,
            model: model,
            processor: StandInUserInputProcessor(),
            tokenizer: tokenizer
        )
        let parameters = GenerateParameters(
            maxTokens: 3,
            temperature: 0,
            ignoreEndOfSequence: ignoreEOS
        )
        let input = LMInput(tokens: MLXArray([1]))
        let stream = try MLXLMCommon.generate(
            input: input,
            cache: model.newCache(parameters: parameters),
            parameters: parameters,
            context: context
        )
        return await completionTokenCount(from: stream)
    }

    private func completionTokenCount(from stream: AsyncStream<Generation>) async -> Int {
        var tokenCount = -1
        for await generation in stream {
            if case .info(let info) = generation {
                tokenCount = info.generationTokenCount
            }
        }
        return tokenCount
    }
}

private final class FixedEOSTokenModel: Module, LanguageModel {
    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray([Float(0), 0, 10, 0]).reshaped(1, 1, 4)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

private struct FixedEOSTokenizer: Tokenizer {
    static let eosTokenID = 2

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { "<eos>" }
    var eosTokenId: Int? { Self.eosTokenID }
    var unknownToken: String? { "<unknown>" }
    var unknownTokenId: Int? { 3 }
    var hasChatTemplate: Bool { false }

    func tokenize(text: String) -> [String] { [text] }
    func encode(text: String) -> [Int] { [1] }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.map { $0 == Self.eosTokenID ? "eos" : "token\($0)" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? {
        token == "<eos>" ? Self.eosTokenID : nil
    }
    func convertIdToToken(_ id: Int) -> String? {
        id == Self.eosTokenID ? "<eos>" : "token\(id)"
    }

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] { [1] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] { [1] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] { [1] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: String
    ) throws -> [Int] { [1] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] { [1] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1] }
}
