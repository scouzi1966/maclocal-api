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
    private struct ExpectedLoadFailure: Error {}

    func testCallerReservedLoadFailureReleasesReservation() async throws {
        let service = try await makeScheduledService(maxConcurrent: 2)
        let reservation = try reserved(service.tryReserveSlot())
        let model = AFMMLXModel(
            modelID: AFMModelID(rawValue: Self.modelID),
            attachedService: service,
            schedulerAdmissionOwnership: .caller(.reserved(reservation)),
            testingStreamingLoad: { throw ExpectedLoadFailure() })

        do {
            for try await _ in model.streamResponse(to: AFMRequest(messages: [])) {}
            XCTFail("expected load failure")
        } catch is ExpectedLoadFailure {
            // Expected.
        }

        XCTAssertEqual(
            service.schedulerAdmissionSnapshot,
            .init(inFlightCount: 0, reservedCount: 0))
        await service.shutdownAndReleaseResources(timeoutSeconds: 1)
    }

    func testCallerReservedRespondLoadFailureReleasesReservation() async throws {
        let service = try await makeScheduledService(maxConcurrent: 2)
        let reservation = try reserved(service.tryReserveSlot())
        let model = AFMMLXModel(
            modelID: AFMModelID(rawValue: Self.modelID),
            attachedService: service,
            schedulerAdmissionOwnership: .caller(.reserved(reservation)))
        _ = service.beginShutdown()

        do {
            _ = try await model.respond(to: AFMRequest(messages: []))
            XCTFail("expected shutdown load failure")
        } catch {
            guard case .loadingFailed = error as? AFMError else {
                return XCTFail("unexpected error: \(error)")
            }
        }

        XCTAssertEqual(
            service.schedulerAdmissionSnapshot,
            .init(inFlightCount: 0, reservedCount: 0))
        await service.shutdownAndReleaseResources(timeoutSeconds: 1)
    }

    func testCallerReservedLoadCancellationReleasesReservation() async throws {
        let service = try await makeScheduledService(maxConcurrent: 2)
        let reservation = try reserved(service.tryReserveSlot())
        let probe = SchedulerCancellationProbe()
        let model = AFMMLXModel(
            modelID: AFMModelID(rawValue: Self.modelID),
            attachedService: service,
            schedulerAdmissionOwnership: .caller(.reserved(reservation)),
            testingStreamingLoad: { try await probe.run() })
        let stream = model.streamResponse(to: AFMRequest(messages: []))
        let consumer = Task {
            do {
                for try await _ in stream {}
            } catch {
                // Cancellation is the expected terminal state.
            }
        }

        await probe.waitUntilEntered()
        consumer.cancel()
        await consumer.value
        try await waitUntil {
            service.schedulerAdmissionSnapshot
                == .init(inFlightCount: 0, reservedCount: 0)
        }

        await service.shutdownAndReleaseResources(timeoutSeconds: 1)
    }

    func testUnreservedSubmissionCannotConsumeAnotherRequestsReservation() {
        let admission = BatchSchedulerAdmissionState(maxConcurrent: 1)

        let reservation = admission.tryReserve()
        XCTAssertNotNil(reservation)
        XCTAssertFalse(admission.reserveForUnreservedSubmission())
        XCTAssertEqual(
            admission.snapshot,
            .init(inFlightCount: 1, reservedCount: 1))
        XCTAssertTrue(admission.consumeReservationForSubmission(reservation!))
        XCTAssertEqual(
            admission.snapshot,
            .init(inFlightCount: 1, reservedCount: 0))
    }

    func testReservedAndUnreservedSubmissionsRemainIndependentWhenInterleaved() {
        let admission = BatchSchedulerAdmissionState(maxConcurrent: 2)

        let reservation = admission.tryReserve()
        XCTAssertNotNil(reservation)
        XCTAssertTrue(admission.reserveForUnreservedSubmission())
        XCTAssertEqual(
            admission.snapshot,
            .init(inFlightCount: 2, reservedCount: 1))
        XCTAssertTrue(admission.consumeReservationForSubmission(reservation!))
        XCTAssertEqual(
            admission.snapshot,
            .init(inFlightCount: 2, reservedCount: 0))
        admission.finish(count: 2)
        XCTAssertEqual(
            admission.snapshot,
            .init(inFlightCount: 0, reservedCount: 0))
    }

    func testAttachedAdapterTransfersControllerReservationWithoutDuplicatingIt() async throws {
        let service = try await makeScheduledService(maxConcurrent: 2)
        let adapter = AFMKitMLXChatServingAdapter(service: service)

        let schedulerAdmission = await adapter.waitForSlot(timeout: 0)
        XCTAssertTrue(schedulerAdmission.isAdmitted)
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
            requestId: "attached-controller",
            schedulerAdmission: schedulerAdmission)

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

    func testDirectGenerationCannotStealReservedAdapterCapacity() async throws {
        let service = try await makeScheduledService(maxConcurrent: 2)
        let firstAdmission = service.tryReserveSlot()
        _ = try reserved(firstAdmission)
        let secondReservation = try reserved(service.tryReserveSlot())

        do {
            _ = try await service.generate(
                model: Self.modelID,
                messages: [.init(role: "user", content: "direct")],
                temperature: 0,
                maxTokens: 1,
                topP: nil,
                repetitionPenalty: nil,
                seed: 11)
            XCTFail("unreserved generation should not consume reserved capacity")
        } catch {
            guard case .serverBusy = error as? MLXServiceError else {
                return XCTFail("unexpected error: \(error)")
            }
        }
        XCTAssertEqual(
            service.schedulerAdmissionSnapshot,
            .init(inFlightCount: 2, reservedCount: 2))

        let adapter = AFMKitMLXChatServingAdapter(service: service)
        let result = try await adapter.generateStreaming(
            model: Self.modelID,
            messages: [.init(role: "user", content: "reserved")],
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
            requestId: "reserved-adapter",
            schedulerAdmission: firstAdmission)
        for try await _ in result.stream {}
        XCTAssertEqual(
            service.schedulerAdmissionSnapshot,
            .init(inFlightCount: 1, reservedCount: 1))

        XCTAssertTrue(service.releaseSlot(secondReservation))
        XCTAssertEqual(
            service.schedulerAdmissionSnapshot,
            .init(inFlightCount: 0, reservedCount: 0))
        await service.shutdownAndReleaseResources(timeoutSeconds: 1)
    }

    func testSerialAdmissionDoesNotCreateSchedulerReservation() {
        let service = makeLoadedService()

        XCTAssertEqual(service.tryReserveSlot(), .serial)
        XCTAssertNil(service.schedulerAdmissionSnapshot)
    }

    func testReplacementAndForeignSchedulerCannotReleaseReservation() async throws {
        let firstService = try await makeScheduledService(maxConcurrent: 2)
        let secondService = try await makeScheduledService(maxConcurrent: 2)
        let firstReservation = try reserved(firstService.tryReserveSlot())
        let secondReservation = try reserved(secondService.tryReserveSlot())

        XCTAssertFalse(secondService.releaseSlot(firstReservation))
        XCTAssertFalse(firstService.releaseSlot(secondReservation))
        let firstScheduler = try XCTUnwrap(firstService.installedScheduler)
        let secondScheduler = try XCTUnwrap(secondService.installedScheduler)
        XCTAssertTrue(
            firstService.replaceInstalledSchedulerForTesting(with: secondScheduler)
                === firstScheduler)
        defer {
            _ = firstService.replaceInstalledSchedulerForTesting(with: firstScheduler)
        }
        XCTAssertFalse(firstService.releaseSlot(firstReservation))
        XCTAssertEqual(
            firstService.schedulerAdmissionSnapshot,
            .init(inFlightCount: 1, reservedCount: 1))
        XCTAssertEqual(
            secondService.schedulerAdmissionSnapshot,
            .init(inFlightCount: 1, reservedCount: 1))

        _ = firstService.replaceInstalledSchedulerForTesting(with: firstScheduler)
        XCTAssertTrue(firstService.releaseSlot(firstReservation))
        XCTAssertTrue(secondService.releaseSlot(secondReservation))
        await firstService.shutdownAndReleaseResources(timeoutSeconds: 1)
        await secondService.shutdownAndReleaseResources(timeoutSeconds: 1)
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

    func testShutdownInvalidatesPromotionBeforeSchedulerPublication() async throws {
        let service = makeLoadedService()
        service.maxConcurrent = 2
        let barrier = SchedulerPromotionBarrier()
        service.schedulerPromotionBarrier = { await barrier.suspend() }
        let promotion = Task {
            try await service.initScheduler()
        }

        await barrier.waitUntilEntered()
        let shutdown = Task {
            await service.shutdownAndReleaseResources(timeoutSeconds: 1)
        }
        try await waitUntil { service.shuttingDown }
        XCTAssertNil(service.installedScheduler)

        await barrier.release()
        do {
            try await promotion.value
            XCTFail("promotion should be invalidated by shutdown")
        } catch {
            guard case MLXServiceError.serviceShuttingDown = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
        await shutdown.value

        XCTAssertNil(service.installedScheduler)
    }

    func testShutdownClosesAdmissionBeforeSetupBlockedByActiveDecode() async throws {
        let decodeBarrier = SchedulerDecodeBarrier()
        let setupBarrier = SchedulerPromotionBarrier()
        let service = try await makeScheduledService(
            maxConcurrent: 2,
            decodeBarrier: { decodeBarrier.block() })
        addTeardownBlock {
            decodeBarrier.release()
            await setupBarrier.release()
        }
        let scheduler = try XCTUnwrap(service.installedScheduler)
        let activeStream = scheduler.submit(
            input: LMInput(tokens: MLXArray([1, 2, 3])),
            parameters: GenerateParameters(maxTokens: 100_000, temperature: 0),
            promptTokens: 3,
            requestId: "active-decode")
        await decodeBarrier.waitUntilEntered()

        let reservation = try reserved(service.tryReserveSlot())
        service.schedulerSetupBarrier = { await setupBarrier.suspend() }
        let setup = Task<Void, Error> { @Sendable [service, reservation] in
            defer { service.releaseSlot(reservation) }
            _ = try await service.generateStreamingWithSchedulerAdmission(
                model: Self.modelID,
                messages: [.init(role: "user", content: "blocked setup")],
                temperature: 0,
                maxTokens: 1,
                topP: nil,
                repetitionPenalty: nil,
                admission: .reserved(reservation))
        }
        await setupBarrier.waitUntilEntered()
        await setupBarrier.release()

        let shutdown = Task {
            await service.shutdownAndReleaseResources(timeoutSeconds: 1)
        }
        try await waitUntil { scheduler.isAdmissionClosed }
        decodeBarrier.release()

        do {
            try await setup.value
            XCTFail("setup queued behind decode should observe shutdown")
        } catch {
            guard case MLXServiceError.serviceShuttingDown = error else {
                return XCTFail("unexpected setup error: \(error)")
            }
        }
        await shutdown.value
        await assertShutdown(stream: activeStream)
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

    func testSchedulerShutdownDrainsPendingAndActiveRequests() async throws {
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

        await service.shutdownAndReleaseResources(timeoutSeconds: 1)

        await assertShutdown(stream: activeStream)
        await assertShutdown(stream: pendingStream)
        XCTAssertEqual(
            scheduler.admissionSnapshot,
            .init(inFlightCount: 0, reservedCount: 0))
        XCTAssertEqual(scheduler.pendingRequestCount, 0)
    }

    private static let modelID = "tests/tiny-scheduler"

    private func makeScheduledService(
        maxConcurrent: Int,
        requiresFixedDecodeCohorts: Bool? = nil,
        decodeBarrier: (@Sendable () -> Void)? = nil
    ) async throws -> MLXModelService {
        let service = makeLoadedService()
        service.maxConcurrent = maxConcurrent
        service.schedulerDecodeBarrier = decodeBarrier
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
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            testingModelID: Self.modelID,
            container: container,
            architecture: architecture)
        addTeardownBlock {
            await service.shutdownAndReleaseResources(timeoutSeconds: 1)
        }
        return service
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

    private func reserved(
        _ admission: AFMMLXSchedulerAdmission,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> AFMMLXSchedulerReservation {
        guard case .reserved(let reservation) = admission else {
            XCTFail("expected scheduler reservation, got \(admission)", file: file, line: line)
            throw NSError(domain: "MLXSchedulerLifecycleTests", code: 2)
        }
        return reservation
    }
}

private actor SchedulerCancellationProbe {
    private var entered = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []

    func run() async throws {
        entered = true
        let waiters = enteredWaiters
        enteredWaiters.removeAll()
        for waiter in waiters {
            waiter.resume()
        }
        try await Task.sleep(for: .seconds(5))
    }

    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { continuation in
            enteredWaiters.append(continuation)
        }
    }
}

private actor SchedulerPromotionBarrier {
    private var entered = false
    private var released = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func suspend() async {
        entered = true
        let waiters = enteredWaiters
        enteredWaiters.removeAll()
        for waiter in waiters {
            waiter.resume()
        }
        if released { return }
        await withCheckedContinuation { continuation in
            releaseWaiters.append(continuation)
        }
    }

    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { continuation in
            enteredWaiters.append(continuation)
        }
    }

    func release() {
        released = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waiters {
            waiter.resume()
        }
    }
}

private final class SchedulerDecodeBarrier: @unchecked Sendable {
    private let lock = NSLock()
    private let releaseSemaphore = DispatchSemaphore(value: 0)
    private var entered = false
    private var released = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []

    func block() {
        let state = lock.withLock { () -> (Bool, [CheckedContinuation<Void, Never>]) in
            guard !entered else { return (false, []) }
            entered = true
            let waiters = enteredWaiters
            enteredWaiters.removeAll()
            return (!released, waiters)
        }
        for waiter in state.1 {
            waiter.resume()
        }
        if state.0 {
            releaseSemaphore.wait()
        }
    }

    func waitUntilEntered() async {
        if lock.withLock({ entered }) { return }
        await withCheckedContinuation { continuation in
            let resumeImmediately = lock.withLock { () -> Bool in
                if entered { return true }
                enteredWaiters.append(continuation)
                return false
            }
            if resumeImmediately {
                continuation.resume()
            }
        }
    }

    func release() {
        let shouldSignal = lock.withLock { () -> Bool in
            guard !released else { return false }
            released = true
            return entered
        }
        if shouldSignal {
            releaseSemaphore.signal()
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
