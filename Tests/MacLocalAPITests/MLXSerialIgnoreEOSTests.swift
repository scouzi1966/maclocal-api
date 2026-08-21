import AFMKitCore
import AFMKitServices
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

@testable import AFMKitMLX

final class MLXSerialIgnoreEOSTests: XCTestCase {
    func testDSparkNonStreamingIgnoreEOSPreservesVisibleBudgetAndUsage() {
        assertSpeculativeIgnoreEOS(engine: .dspark, mode: .nonStreaming)
    }

    func testDSparkStreamingIgnoreEOSPreservesVisibleBudgetAndTelemetry() {
        assertSpeculativeIgnoreEOS(engine: .dspark, mode: .streaming)
    }

    func testMTPNonStreamingIgnoreEOSPreservesVisibleBudgetAndUsage() {
        assertSpeculativeIgnoreEOS(engine: .mtp, mode: .nonStreaming)
    }

    func testMTPStreamingIgnoreEOSPreservesVisibleBudgetAndTelemetry() {
        assertSpeculativeIgnoreEOS(engine: .mtp, mode: .streaming)
    }

    func testEagle3NonStreamingIgnoreEOSPreservesVisibleBudgetAndUsage() {
        assertSpeculativeIgnoreEOS(engine: .eagle3, mode: .nonStreaming)
    }

    func testEagle3StreamingIgnoreEOSPreservesVisibleBudgetAndTelemetry() {
        assertSpeculativeIgnoreEOS(engine: .eagle3, mode: .streaming)
    }

    func testSerialStreamingStopsAtEOSByDefaultAndHonorsIgnoreEOS() async throws {
        let defaultObservation = try await streamingGenerationObservation(ignoreEOS: false)
        let ignoredEOSObservation = try await streamingGenerationObservation(ignoreEOS: true)

        XCTAssertEqual(defaultObservation.tokenCount, 0)
        XCTAssertEqual(defaultObservation.text, "")
        XCTAssertEqual(ignoredEOSObservation.tokenCount, 0)
        XCTAssertEqual(ignoredEOSObservation.text, "")
    }

    func testSerialNonStreamingStopsAtEOSByDefaultAndHonorsIgnoreEOS() async throws {
        let defaultObservation = try await nonStreamingGenerationObservation(ignoreEOS: false)
        let ignoredEOSObservation = try await nonStreamingGenerationObservation(ignoreEOS: true)

        XCTAssertEqual(defaultObservation.tokenCount, 0)
        XCTAssertEqual(defaultObservation.text, "")
        XCTAssertEqual(ignoredEOSObservation.tokenCount, 0)
        XCTAssertEqual(ignoredEOSObservation.text, "")
    }

    func testNativeSynchronousPathsExcludeSuppressedEOSFromUsageAndCallbacks() throws {
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
        let input = LMInput(tokens: MLXArray([1]))
        let parameters = GenerateParameters(
            maxTokens: 3,
            temperature: 0,
            ignoreEndOfSequence: true
        )
        var callbackTokens = [Int]()
        let callbackResult = MLXLMCommon.generate(
            input: input,
            context: context,
            iterator: try TokenIterator(
                input: input,
                model: model,
                parameters: parameters
            ),
            ignoreEndOfSequence: true
        ) { (token: Int) in
            callbackTokens.append(token)
            return .more
        }
        let arrayResult = MLXLMCommon.generate(
            input: input,
            context: context,
            iterator: try TokenIterator(
                input: input,
                model: model,
                parameters: parameters
            ),
            ignoreEndOfSequence: true
        ) { (_: [Int]) in .more }

        XCTAssertEqual(callbackTokens, [])
        XCTAssertEqual(callbackResult.generationTokenCount, 0)
        XCTAssertEqual(arrayResult.tokens, [])
        XCTAssertEqual(arrayResult.generationTokenCount, 0)
    }

    func testSingleSchedulerPrefillAndDecodeExcludeSuppressedEOS() async throws {
        let collector = InferenceTelemetryCollector()
        let scheduler = makeScheduler(
            collector: collector,
            admissionWindowNanoseconds: 0
        )
        let stream = AFMGenerationContext.$ignoreEndOfSequence.withValue(true) {
            scheduler.submit(
                input: LMInput(tokens: MLXArray([1])),
                parameters: GenerateParameters(maxTokens: 3, temperature: 0),
                promptTokens: 1
            )
        }

        let observation = try await Self.schedulerObservation(from: stream)
        await scheduler.shutdown()

        XCTAssertEqual(observation.text, "")
        XCTAssertEqual(observation.tokenCount, 0)
        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.generatedTokensTotal, 0)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "stop" }?.count, 1)
    }

    func testConcurrentSchedulerBatchPrefillAndDecodeExcludeSuppressedEOS() async throws {
        let collector = InferenceTelemetryCollector()
        let scheduler = makeScheduler(
            collector: collector,
            admissionWindowNanoseconds: 10_000_000
        )
        let input = LMInput(tokens: MLXArray([1]))
        let parameters = GenerateParameters(maxTokens: 3, temperature: 0)
        let streams = AFMGenerationContext.$ignoreEndOfSequence.withValue(true) {
            (
                scheduler.submit(input: input, parameters: parameters, promptTokens: 1),
                scheduler.submit(input: input, parameters: parameters, promptTokens: 1)
            )
        }

        async let first = Self.schedulerObservation(from: streams.0)
        async let second = Self.schedulerObservation(from: streams.1)
        let (firstObservation, secondObservation) = try await (first, second)
        let observations = [firstObservation, secondObservation]
        await scheduler.shutdown()

        XCTAssertEqual(observations.map(\.text), ["", ""])
        XCTAssertEqual(observations.map(\.tokenCount), [0, 0])
        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.generatedTokensTotal, 0)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 2)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "stop" }?.count, 2)
    }

    func testSchedulerCancellationImmediatelyAfterSubmissionCountsSingleAbort() async throws {
        let collector = InferenceTelemetryCollector()
        let scheduler = makeScheduler(
            collector: collector,
            admissionWindowNanoseconds: 20_000_000
        )
        let stream = scheduler.submit(
            input: LMInput(tokens: MLXArray([1])),
            parameters: GenerateParameters(maxTokens: 3, temperature: 0),
            promptTokens: 1
        )
        let consumer = Task {
            for try await _ in stream {}
        }

        consumer.cancel()
        _ = try? await consumer.value
        try await waitForTerminalRequest(collector)
        await scheduler.shutdown()

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.generatedTokensTotal, 0)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "abort" }?.count, 1)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "cancelled" }?.count, 1)
    }

    private func streamingGenerationObservation(ignoreEOS: Bool) async throws -> Observation {
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

        let observation = await generationObservation(from: stream)
        await task.value
        return observation
    }

    private func nonStreamingGenerationObservation(ignoreEOS: Bool) async throws -> Observation {
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
        return await generationObservation(from: stream)
    }

    private func generationObservation(from stream: AsyncStream<Generation>) async -> Observation {
        var text = ""
        var tokenCount = -1
        for await generation in stream {
            if case .chunk(let chunk) = generation {
                text += chunk
            } else if case .info(let info) = generation {
                tokenCount = info.generationTokenCount
            }
        }
        return Observation(text: text, tokenCount: tokenCount)
    }

    private static func schedulerObservation(
        from stream: AsyncThrowingStream<StreamChunk, Error>
    ) async throws -> Observation {
        var text = ""
        var tokenCount = -1
        for try await chunk in stream {
            text += chunk.text
            if let completionTokens = chunk.completionTokens {
                tokenCount = completionTokens
            }
        }
        return Observation(text: text, tokenCount: tokenCount)
    }

    private func makeScheduler(
        collector: InferenceTelemetryCollector,
        admissionWindowNanoseconds: UInt64
    ) -> BatchScheduler {
        var configuration = ModelConfiguration(id: "scheduler-eos-test")
        configuration.eosTokenIds = [FixedEOSTokenizer.eosTokenID]
        return BatchScheduler(
            model: FixedEOSTokenModel(),
            tokenizer: FixedEOSTokenizer(),
            processor: StandInUserInputProcessor(),
            configuration: configuration,
            telemetryObserver: collector,
            maxConcurrent: 2,
            admissionWindowNanoseconds: admissionWindowNanoseconds
        )
    }

    private func waitForTerminalRequest(
        _ collector: InferenceTelemetryCollector
    ) async throws {
        for _ in 0..<100 {
            if collector.metricsSnapshot().terminalRequestsTotal == 1 { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("scheduler cancellation did not reach telemetry")
    }

    private func assertSpeculativeIgnoreEOS(
        engine: MLXSpeculativeEngine,
        mode: MLXSpeculativeGenerationMode,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let eos = FixedEOSTokenizer.eosTokenID
        let accounting = MLXSpeculativeOutputAccounting(
            engine: engine,
            mode: mode,
            maximumVisibleTokens: 2,
            endOfSequenceTokenIDs: [eos],
            ignoreEndOfSequence: true
        )
        var telemetryTokens = 0

        for token in [eos, 10, eos, 11] {
            let disposition = accounting.consume(token)
            if case .emit = disposition { telemetryTokens += 1 }
            if !disposition.shouldContinue { break }
        }

        XCTAssertEqual(accounting.engine, engine, file: file, line: line)
        XCTAssertEqual(accounting.mode, mode, file: file, line: line)
        XCTAssertEqual(accounting.generatorMaximumTokens, Int.max, file: file, line: line)
        XCTAssertEqual(accounting.visibleTokenIDs, [10, 11], file: file, line: line)
        XCTAssertEqual(accounting.visibleTokenCount, 2, file: file, line: line)
        XCTAssertEqual(telemetryTokens, 2, file: file, line: line)
    }
}

private struct Observation {
    let text: String
    let tokenCount: Int
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
        let batchSize = inputs.ndim > 1 ? inputs.dim(0) : 1
        let logits: [Float] = Array(
            repeating: [Float(0), 0, 10, 0],
            count: batchSize
        ).flatMap { $0 }
        return MLXArray(logits).reshaped(batchSize, 1, 4)
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
