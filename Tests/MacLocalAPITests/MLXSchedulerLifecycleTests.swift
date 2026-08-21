import AFMKit
import AFMOpenAICompat
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers
@testable import AFMKitMLX
@testable import AFMServer
import XCTest

final class MLXSchedulerLifecycleTests: XCTestCase {
    func testAttachedAdapterTransfersControllerReservationWithoutDuplicatingIt() async throws {
        let service = try await makeScheduledService(maxConcurrent: 2)
        let adapter = AFMKitMLXChatServingAdapter(service: service)

        let reserved = await adapter.waitForSlot(timeout: 0)
        XCTAssertTrue(reserved)
        XCTAssertEqual(
            service.schedulerAdmissionSnapshot,
            .init(inFlightCount: 1, reservedCount: 1))

        let result = try await adapter.generateStreaming(
            model: Self.modelID,
            messages: [.init(role: "user", content: "hello")],
            temperature: 0,
            maxTokens: 1,
            topP: nil,
            repetitionPenalty: nil,
            topK: nil,
            minP: nil,
            presencePenalty: nil,
            seed: 7,
            logprobs: nil,
            topLogprobs: nil,
            tools: nil,
            toolChoice: nil,
            parallelToolCalls: nil,
            stop: nil,
            responseFormat: nil,
            chatTemplateKwargs: nil,
            speculativeDecoding: nil,
            preserveStructuralTags: false,
            requestId: "attached-controller")

        try await waitUntil {
            service.schedulerAdmissionSnapshot
                == .init(inFlightCount: 1, reservedCount: 0)
        }
        for try await _ in result.stream {}
        XCTAssertEqual(
            service.schedulerAdmissionSnapshot,
            .init(inFlightCount: 0, reservedCount: 0))

        await service.shutdownAndReleaseResources(timeoutSeconds: 1)
    }

    func testDirectGenerateCollectsSchedulerStreamWithoutDoubleAdmission() async throws {
        let service = try await makeScheduledService(maxConcurrent: 2)

        let result = try await service.generate(
            model: Self.modelID,
            messages: [.init(role: "user", content: "hello")],
            temperature: 0,
            maxTokens: 1,
            topP: nil,
            repetitionPenalty: nil,
            seed: 11)

        XCTAssertEqual(result.modelID, Self.modelID)
        XCTAssertLessThanOrEqual(result.completionTokens, 1)
        XCTAssertEqual(
            service.schedulerAdmissionSnapshot,
            .init(inFlightCount: 0, reservedCount: 0))

        await service.shutdownAndReleaseResources(timeoutSeconds: 1)
    }

    func testShutdownClosesOperationAdmissionBeforeSchedulerRemoval() async throws {
        let service = try await makeScheduledService(maxConcurrent: 2)

        XCTAssertNotNil(service.beginShutdown())
        XCTAssertTrue(service.shuttingDown)
        XCTAssertThrowsError(try service.beginOperation()) { error in
            guard case MLXServiceError.serviceShuttingDown = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }

        await service.shutdownAndReleaseResources(timeoutSeconds: 1)
    }

    func testGPUCaptureRejectsSchedulerInstallationWithoutConsumingCapture() async throws {
        let service = makeLoadedService()
        service.maxConcurrent = 2
        service.gpuCapturePath = "/tmp/should-not-capture.gputrace"

        do {
            try await service.initScheduler()
            XCTFail("concurrent GPU capture should be rejected")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains(
                AFMMLXGPUCapturePolicy.concurrentIncompatibility))
        }

        XCTAssertNil(service.installedScheduler)
        XCTAssertEqual(
            service.gpuCapturePath,
            "/tmp/should-not-capture.gputrace")
    }

    func testSchedulerShutdownBalancesPendingAndActiveRequestAccounting() async throws {
        StatsAggregator.shared.reset()
        defer { StatsAggregator.shared.reset() }

        let service = try await makeScheduledService(
            maxConcurrent: 2,
            requiresFixedDecodeCohorts: true)
        let scheduler = try XCTUnwrap(service.installedScheduler)
        let parameters = GenerateParameters(maxTokens: 100_000, temperature: 0)
        let input = LMInput(tokens: MLXArray([1, 2, 3]))
        let activeStream = scheduler.submit(
            input: input,
            parameters: parameters,
            promptTokens: 3,
            requestId: "active")

        try await waitUntil { scheduler.pendingRequestCount == 0 }
        let pendingStream = scheduler.submit(
            input: input,
            parameters: parameters,
            promptTokens: 3,
            requestId: "pending")
        try await waitUntil { scheduler.pendingRequestCount == 1 }

        let before = StatsAggregator.shared.snapshot()
        XCTAssertEqual(before.requestsStartedTotal, 2)
        XCTAssertEqual(before.numRunning, 1)
        XCTAssertEqual(before.numWaiting, 1)

        await service.shutdownAndReleaseResources(timeoutSeconds: 1)

        await assertShutdown(stream: activeStream)
        await assertShutdown(stream: pendingStream)
        let after = StatsAggregator.shared.snapshot()
        XCTAssertEqual(after.requestsStartedTotal, 2)
        XCTAssertEqual(after.requestsCompletedTotal, 2)
        XCTAssertEqual(after.requestSuccessByReason["abort"], 2)
        XCTAssertEqual(
            scheduler.admissionSnapshot,
            .init(inFlightCount: 0, reservedCount: 0))
        XCTAssertEqual(scheduler.pendingRequestCount, 0)
    }

    private static let modelID = "tests/tiny-scheduler"

    private func makeScheduledService(
        maxConcurrent: Int,
        requiresFixedDecodeCohorts: Bool? = nil
    ) async throws -> MLXModelService {
        let service = makeLoadedService()
        service.maxConcurrent = maxConcurrent
        try await service.initScheduler(
            requiresFixedDecodeCohorts: requiresFixedDecodeCohorts)
        return service
    }

    private func makeLoadedService() -> MLXModelService {
        let fixture = makeModelFixture()
        let container = ModelContainer(context: .init(
            configuration: fixture.configuration,
            model: fixture.model,
            processor: fixture.processor,
            tokenizer: fixture.tokenizer))
        let architecture = AFMMLXModelArchitecturePreflight(
            modelID: Self.modelID,
            modelType: "llama",
            canonicalModelType: "llama",
            isVisionConfiguration: false,
            requiresVisionModelFactory: false)
        return MLXModelService(
            resolver: MLXCacheResolver(),
            testingModelID: Self.modelID,
            container: container,
            architecture: architecture)
    }

    private func makeModelFixture() -> (
        model: LlamaModel,
        tokenizer: SchedulerTestTokenizer,
        processor: SchedulerTestInputProcessor,
        configuration: ModelConfiguration
    ) {
        let tokenizer = SchedulerTestTokenizer()
        let processor = SchedulerTestInputProcessor()
        let model = LlamaModel(LlamaConfiguration(
            hiddenSize: 16,
            hiddenLayers: 1,
            intermediateSize: 32,
            attentionHeads: 2,
            rmsNormEps: 0.00001,
            vocabularySize: 32,
            kvHeads: 1))
        eval(model)
        let configuration = ModelConfiguration(id: Self.modelID)
        return (model, tokenizer, processor, configuration)
    }

    private func waitUntil(
        timeout: TimeInterval = 1,
        condition: @escaping @Sendable () -> Bool
    ) async throws {
        let deadline = ContinuousClock.now + .seconds(timeout)
        while !condition() {
            guard ContinuousClock.now < deadline else {
                XCTFail("scheduler state did not settle before timeout")
                throw NSError(
                    domain: "MLXSchedulerLifecycleTests",
                    code: 1)
            }
            try await Task.sleep(for: .milliseconds(1))
        }
    }

    private func assertShutdown(
        stream: AsyncThrowingStream<StreamChunk, Error>,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            for try await _ in stream {}
            XCTFail("expected shutdown error", file: file, line: line)
        } catch {
            guard case MLXServiceError.serviceShuttingDown = error else {
                return XCTFail("unexpected error: \(error)", file: file, line: line)
            }
        }
    }
}

private struct SchedulerTestInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray([1, 2, 3]))
    }
}

private struct SchedulerTestTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }
    var hasChatTemplate: Bool { true }

    func tokenize(text: String) -> [String] { [text] }
    func encode(text: String) -> [Int] { [1, 2, 3] }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { encode(text: text) }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.filter { !skipSpecialTokens || $0 != eosTokenId }.map(String.init).joined(separator: " ")
    }
    func convertTokenToId(_ token: String) -> Int? { token == eosToken ? eosTokenId : nil }
    func convertIdToToken(_ id: Int) -> String? { id == eosTokenId ? eosToken : String(id) }

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] { [1, 2, 3] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] { [1, 2, 3] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2, 3] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] { [1, 2, 3] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: String
    ) throws -> [Int] { [1, 2, 3] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] { [1, 2, 3] }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2, 3] }
}
