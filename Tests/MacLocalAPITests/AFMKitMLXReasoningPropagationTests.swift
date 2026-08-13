@testable import AFMServer
import AFMKit
import XCTest

final class AFMKitMLXReasoningPropagationTests: XCTestCase {
    private actor RequestCapture {
        private var requests: [AFMRequest] = []

        func append(_ request: AFMRequest) {
            requests.append(request)
        }

        func request(at index: Int) -> AFMRequest {
            requests[index]
        }
    }

    private struct CapturingModel: AFMModel {
        let descriptor = AFMModelDescriptor(
            providerID: "test.capture",
            modelID: "capture",
            displayName: "Capture",
            capabilities: [.text, .reasoning],
            metadata: ["maxConcurrent": .integer(1)]
        )
        let capture: RequestCapture

        func availability() async -> AFMModelAvailability { .available }

        func load(
            progress: (@Sendable (Double) -> Void)?
        ) async throws -> AFMModelDescriptor {
            descriptor
        }

        func respond(to request: AFMRequest) async throws -> AFMModelResponse {
            await capture.append(request)
            return AFMModelResponse(text: "captured", reasoning: "thought")
        }

        func streamResponse(
            to request: AFMRequest
        ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
            AsyncThrowingStream { continuation in
                continuation.finish()
            }
        }
    }

    func testStartupReasoningDefaultsReachAFMRequestMetadata() async throws {
        let capture = RequestCapture()
        let adapter = makeAdapter(
            capture: capture,
            defaults: [
                "enable_thinking": AnyCodable(true),
                "reasoning_effort": AnyCodable("high"),
            ]
        )

        _ = try await generate(adapter: adapter, kwargs: nil)

        let request = await capture.request(at: 0)
        XCTAssertEqual(
            request.metadata["chatTemplateKwargs"],
            .object([
                "enable_thinking": .bool(true),
                "reasoning_effort": .string("high"),
            ])
        )
    }

    func testReasoningCapableFixedModelPublishesStructuralTags() async throws {
        let adapter = makeAdapter(capture: RequestCapture(), defaults: [:])

        let result = try await generate(adapter: adapter, kwargs: nil)

        XCTAssertEqual(adapter.thinkStartTag, "<think>")
        XCTAssertEqual(adapter.thinkEndTag, "</think>")
        XCTAssertEqual(result.content, "<think>thought</think>captured")
    }

    func testRequestReasoningEffortOverridesStartupDefault() async throws {
        let capture = RequestCapture()
        let adapter = makeAdapter(
            capture: capture,
            defaults: [
                "enable_thinking": AnyCodable(true),
                "reasoning_effort": AnyCodable("low"),
            ]
        )

        _ = try await generate(
            adapter: adapter,
            kwargs: ["reasoning_effort": AnyCodable("max")]
        )

        let request = await capture.request(at: 0)
        XCTAssertEqual(
            request.metadata["chatTemplateKwargs"],
            .object([
                "enable_thinking": .bool(true),
                "reasoning_effort": .string("max"),
            ])
        )
    }

    private func makeAdapter(
        capture: RequestCapture,
        defaults: [String: AnyCodable]
    ) -> AFMKitMLXChatServingAdapter {
        AFMKitMLXChatServingAdapter(
            model: AnyAFMModel(CapturingModel(capture: capture)),
            modelID: "capture",
            defaultChatTemplateKwargs: defaults
        )
    }

    private func generate(
        adapter: AFMKitMLXChatServingAdapter,
        kwargs: [String: AnyCodable]?
    ) async throws -> AFMMLXChatGenerationResult {
        try await adapter.generate(
            model: "capture",
            messages: [Message(role: "user", content: "hello")],
            temperature: nil,
            maxTokens: 8,
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
            chatTemplateKwargs: kwargs
        )
    }
}
