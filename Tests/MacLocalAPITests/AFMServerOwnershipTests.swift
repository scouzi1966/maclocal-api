@testable import AFMKitMLX
@testable import AFMServer
import AFMKit
import os
import XCTest

final class AFMServerOwnershipTests: XCTestCase {
    func testActiveConnectionTrackerMaintainsCurrentAndPeakCounts() {
        let tracker = ActiveConnectionTracker()

        DispatchQueue.concurrentPerform(iterations: 32) { _ in
            tracker.connectionStarted()
        }

        var snapshot = tracker.snapshot()
        XCTAssertEqual(snapshot.activeConnections, 32)
        XCTAssertEqual(snapshot.activeConnectionsPeak, 32)

        DispatchQueue.concurrentPerform(iterations: 32) { _ in
            tracker.connectionEnded()
        }
        tracker.connectionEnded()

        snapshot = tracker.snapshot()
        XCTAssertEqual(snapshot.activeConnections, 0)
        XCTAssertEqual(snapshot.activeConnectionsPeak, 32)
    }

    func testPrometheusRenderingUsesServerOwnedConnectionSnapshot() {
        let tracker = ActiveConnectionTracker()
        tracker.connectionStarted()
        tracker.connectionStarted()

        let body = MetricsController.renderPrometheus(
            StatsAggregator.shared.snapshot(),
            connections: tracker.snapshot()
        )
        let lines = body.split(separator: "\n").map(String.init)

        XCTAssertTrue(lines.contains {
            $0.hasPrefix("afm:num_active_connections{") && $0.hasSuffix("} 2")
        })
        XCTAssertTrue(lines.contains {
            $0.hasPrefix("afm:active_connections_peak{") && $0.hasSuffix("} 2")
        })
    }

    func testConcreteMLXAdapterPreservesProviderServingConfiguration() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.maxConcurrent = 8
        service.toolCallParser = "afm_adaptive_xml"
        service.enableGrammarConstraints = true
        service.fixToolArgs = true
        let model = AFMMLXModel(
            modelID: "test/concrete-adapter",
            attachedService: service
        )

        let adapter = AFMKitMLXChatServingAdapter(model: model)

        XCTAssertEqual(adapter.maxConcurrent, 8)
        XCTAssertEqual(adapter.servingConfiguration.toolCallParser, "afm_adaptive_xml")
        XCTAssertTrue(adapter.servingConfiguration.grammarConstraintsEnabled)
        XCTAssertTrue(adapter.servingConfiguration.fixToolArguments)
        XCTAssertTrue(adapter.tryReserveSlot())
        adapter.releaseSlot()
    }

    func testGenericAdapterEnforcesConfiguredAdmissionLimitConcurrently() {
        let adapter = makeGenericAdapter(maxConcurrent: 4)
        let accepted = OSAllocatedUnfairLock(initialState: 0)

        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            if adapter.tryReserveSlot() {
                accepted.withLock { $0 += 1 }
            }
        }

        XCTAssertEqual(accepted.withLock { $0 }, 4)
        XCTAssertFalse(adapter.tryReserveSlot())

        for _ in 0..<4 {
            adapter.releaseSlot()
        }
        XCTAssertEqual((0..<4).filter { _ in adapter.tryReserveSlot() }.count, 4)
        XCTAssertFalse(adapter.tryReserveSlot())
        for _ in 0..<4 {
            adapter.releaseSlot()
        }
    }

    func testGenericAdapterWaitsForCapacityAndTimesOutWhenFull() async {
        let adapter = makeGenericAdapter(maxConcurrent: 1)
        XCTAssertTrue(adapter.tryReserveSlot())

        let waiter = Task { await adapter.waitForSlot(timeout: 1) }
        try? await Task.sleep(nanoseconds: 30_000_000)
        adapter.releaseSlot()
        let admittedAfterRelease = await waiter.value
        XCTAssertTrue(admittedAfterRelease)

        let admittedWhileFull = await adapter.waitForSlot(timeout: 0.03)
        XCTAssertFalse(admittedWhileFull)
        adapter.releaseSlot()
    }

    func testGenericAdapterReleasesTransferredStreamingReservation() async throws {
        let adapter = makeGenericAdapter(maxConcurrent: 1)
        XCTAssertTrue(adapter.tryReserveSlot())

        let result = try await genericStream(
            adapter: adapter,
            messages: [Message(role: "user", content: "hello")]
        )
        for try await _ in result.stream {}

        let admittedAfterStream = await adapter.waitForSlot(timeout: 0.5)
        XCTAssertTrue(admittedAfterStream)
        adapter.releaseSlot()
    }

    func testGenericAdapterReleasesExactlyOneReservationWhenStreamingRequestIsInvalid() async throws {
        let adapter = makeGenericAdapter(maxConcurrent: 2)
        XCTAssertTrue(adapter.tryReserveSlot())
        XCTAssertTrue(adapter.tryReserveSlot())

        let result = try await genericStream(
            adapter: adapter,
            messages: [Message(role: "unsupported", content: "hello")]
        )
        do {
            for try await _ in result.stream {}
            XCTFail("invalid OpenAI message role unexpectedly completed its stream")
        } catch {}

        XCTAssertTrue(adapter.tryReserveSlot())
        XCTAssertFalse(adapter.tryReserveSlot())
        adapter.releaseSlot()
        adapter.releaseSlot()
    }

    private func genericStream(
        adapter: AFMKitMLXChatServingAdapter,
        messages: [Message]
    ) async throws -> AFMChatStreamingResult {
        try await adapter.generateStreaming(
            model: "test/generic-admission",
            messages: messages,
            temperature: nil,
            maxTokens: nil,
            topP: nil,
            repetitionPenalty: nil,
            topK: nil,
            minP: nil,
            presencePenalty: nil,
            seed: nil,
            logprobs: nil,
            topLogprobs: nil,
            tools: nil,
            parallelToolCalls: nil,
            stop: nil,
            responseFormat: nil,
            chatTemplateKwargs: nil,
            preserveStructuralTags: false,
            requestId: nil
        )
    }

    private func makeGenericAdapter(maxConcurrent: Int) -> AFMKitMLXChatServingAdapter {
        let model = GenericAdmissionTestModel(maxConcurrent: maxConcurrent)
        return AFMKitMLXChatServingAdapter(
            model: AnyAFMModel(model),
            modelID: model.descriptor.modelID.rawValue
        )
    }
}

private struct GenericAdmissionTestModel: AFMModel {
    let descriptor: AFMModelDescriptor

    init(maxConcurrent: Int) {
        descriptor = AFMModelDescriptor(
            providerID: "test",
            modelID: "test/generic-admission",
            displayName: "Generic admission test",
            capabilities: [.text, .streaming],
            metadata: ["maxConcurrent": .integer(maxConcurrent)]
        )
    }

    func availability() async -> AFMModelAvailability { .available }

    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
        descriptor
    }

    func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        AFMModelResponse(text: "ok")
    }

    func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.responseText(action: .append, text: "ok", tokenCount: 1))
            continuation.yield(.completed(.stop))
            continuation.finish()
        }
    }
}
