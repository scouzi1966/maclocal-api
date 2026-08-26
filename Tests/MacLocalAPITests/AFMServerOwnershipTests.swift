@testable import AFMServer
import AFMKit
import Vapor
import XCTVapor
import os
import XCTest

final class AFMServerOwnershipTests: XCTestCase {
    func testPrometheusRenderingUsesServerOwnedConnectionSnapshot() {
        let telemetry = AFMServerTelemetryAdapter.standalone()
        telemetry.configure(
            modelName: "test-model",
            maximumConcurrentRequests: 2
        )
        let first = telemetry.connectionOpened()
        let second = telemetry.connectionOpened()

        let body = MetricsController.renderPrometheus(telemetry.metricsSnapshot())
        let lines = body.split(separator: "\n").map(String.init)

        XCTAssertTrue(lines.contains {
            $0.hasPrefix("afm:num_active_connections{") && $0.hasSuffix("} 2")
        })
        XCTAssertTrue(lines.contains {
            $0.hasPrefix("afm:active_connections_peak{") && $0.hasSuffix("} 2")
        })

        telemetry.connectionClosed(first)
        telemetry.connectionClosed(second)
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

    func testNonStreamingControllerDoesNotReleaseQueuedThirdRequestReservation() async throws {
        let state = ControlledGenericAdmissionState()
        let adapter = makeGenericAdapter(maxConcurrent: 2, state: state)
        let app = try await Application.make(.testing)
        try MLXChatCompletionsController(
            modelID: "test/generic-admission",
            service: adapter,
            temperature: nil,
            repetitionPenalty: nil
        ).boot(routes: app)
        let testable = try app.testable()
        let sender = ConcurrentRequestSender(tester: testable)

        let first = Task { try await sender.send("first") }
        let second = Task { try await sender.send("second") }
        await state.waitUntilStarted(count: 2)

        let third = Task { try await sender.send("third") }
        try await Task.sleep(nanoseconds: 150_000_000)
        let startedWhileFull = await state.startedCount()
        XCTAssertEqual(startedWhileFull, 2, "third request bypassed admission queue")

        await state.release("first")
        await state.waitUntilStarted(count: 3)
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertFalse(
            adapter.tryReserveSlot(),
            "first request cleanup consumed the queued third request's reservation"
        )

        await state.releaseAll()
        try await first.value
        try await second.value
        try await third.value
        try await app.asyncShutdown()
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

    private func makeGenericAdapter(
        maxConcurrent: Int,
        state: ControlledGenericAdmissionState? = nil
    ) -> AFMKitMLXChatServingAdapter {
        let model = GenericAdmissionTestModel(maxConcurrent: maxConcurrent, state: state)
        return AFMKitMLXChatServingAdapter(
            model: AnyAFMModel(model),
            modelID: model.descriptor.modelID.rawValue
        )
    }

}

private struct ConcurrentRequestSender: Sendable {
    let tester: any XCTApplicationTester

    func send(_ prompt: String) async throws {
        let body = ByteBuffer(string: """
        {"model":"test/generic-admission","stream":false,"messages":[{"role":"user","content":"\(prompt)"}]}
        """)
        var headers = HTTPHeaders()
        headers.contentType = .json
        try await tester.test(
            .POST,
            "/v1/chat/completions",
            headers: headers,
            body: body
        ) { response async in
            XCTAssertEqual(response.status, HTTPStatus.ok)
        }
    }
}

private struct GenericAdmissionTestModel: AFMModel {
    let descriptor: AFMModelDescriptor
    let state: ControlledGenericAdmissionState?

    init(maxConcurrent: Int, state: ControlledGenericAdmissionState? = nil) {
        descriptor = AFMModelDescriptor(
            providerID: "test",
            modelID: "test/generic-admission",
            displayName: "Generic admission test",
            capabilities: [.text, .streaming],
            metadata: ["maxConcurrent": .integer(maxConcurrent)]
        )
        self.state = state
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
            let task = Task {
                if let state {
                    let prompt = request.messages
                        .flatMap(\.content)
                        .compactMap { content -> String? in
                            guard case .text(let text) = content else { return nil }
                            return text
                        }
                        .joined()
                    await state.start(prompt)
                    await state.waitUntilReleased(prompt)
                }
                continuation.yield(.responseText(action: .append, text: "ok", tokenCount: 1))
                continuation.yield(.completed(.stop))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

private actor ControlledGenericAdmissionState {
    private var started = Set<String>()
    private var released = Set<String>()

    func start(_ prompt: String) {
        started.insert(prompt)
    }

    func release(_ prompt: String) {
        released.insert(prompt)
    }

    func releaseAll() {
        released.formUnion(started)
    }

    func startedCount() -> Int {
        started.count
    }

    func waitUntilStarted(count: Int) async {
        while started.count < count {
            try? await Task.sleep(nanoseconds: 1_000_000)
        }
    }

    func waitUntilReleased(_ prompt: String) async {
        while !released.contains(prompt), !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 1_000_000)
        }
    }
}
