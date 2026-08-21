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
        let capture: RequestCapture
        let maxConcurrent: Int

        var descriptor: AFMModelDescriptor {
            AFMModelDescriptor(
                providerID: "test.capture",
                modelID: "capture",
                displayName: "Capture",
                capabilities: [.text, .reasoning],
                metadata: ["maxConcurrent": .integer(maxConcurrent)]
            )
        }

        func availability() async -> AFMModelAvailability { .available }

        func load(
            progress: (@Sendable (Double) -> Void)?
        ) async throws -> AFMModelDescriptor {
            descriptor
        }

        func respond(to request: AFMRequest) async throws -> AFMModelResponse {
            await capture.append(request)
            return AFMModelResponse(
                text: "captured",
                reasoning: "thought",
                metadata: [
                    AFMMLXSpeculativeTelemetry.metadataKey:
                        Self.telemetry.metadataValue,
                ]
            )
        }

        func streamResponse(
            to request: AFMRequest
        ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
            AsyncThrowingStream { continuation in
                Task {
                    await capture.append(request)
                    continuation.yield(.metadata([
                        AFMMLXSpeculativeTelemetry.metadataKey:
                            Self.telemetry.metadataValue,
                    ]))
                    continuation.yield(.completed(.stop))
                    continuation.finish()
                }
            }
        }

        static let telemetry = AFMMLXSpeculativeTelemetry.fallback(
            strategy: "dflash2", reason: "disabled")
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
        XCTAssertEqual(result.speculativeTelemetry, CapturingModel.telemetry)
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

    func testSpeculativeControlsReachAFMRequestMetadata() async throws {
        let capture = RequestCapture()
        let adapter = makeAdapter(capture: capture, defaults: [:])
        let speculative = SpeculativeDecodingOptions(
            mode: "dflash2",
            requirement: "required",
            drafter: "incoai/example",
            maxDraftTokens: 4
        )

        _ = try await generate(
            adapter: adapter,
            kwargs: nil,
            speculativeDecoding: speculative
        )

        let request = await capture.request(at: 0)
        XCTAssertEqual(
            request.metadata["speculativeDecoding"],
            .object([
                "mode": .string("dflash2"),
                "requirement": .string("required"),
                "drafter": .string("incoai/example"),
                "maxDraftTokens": .integer(4),
            ])
        )
    }

    func testStreamingSpeculativeControlsReachAFMRequestMetadata() async throws {
        let capture = RequestCapture()
        let adapter = makeAdapter(capture: capture, defaults: [:])
        let result = try await adapter.generateStreaming(
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
            toolChoice: nil,
            parallelToolCalls: nil,
            stop: nil,
            responseFormat: nil,
            chatTemplateKwargs: nil,
            speculativeDecoding: SpeculativeDecodingOptions(mode: "off"),
            preserveStructuralTags: false,
            requestId: "spec-test"
        )
        var streamedTelemetry: AFMMLXSpeculativeTelemetry?
        for try await chunk in result.stream {
            streamedTelemetry = chunk.speculativeTelemetry ?? streamedTelemetry
        }

        let request = await capture.request(at: 0)
        XCTAssertEqual(
            request.metadata["speculativeDecoding"],
            .object(["mode": .string("off")])
        )
        XCTAssertEqual(
            request.metadata[AFMMLXRequestMetadata.preserveStructuralTags],
            .bool(false))
        XCTAssertEqual(streamedTelemetry, CapturingModel.telemetry)
    }

    func testStreamingStructuralTagsHaveSerialConcurrentParity() async throws {
        let serialCapture = RequestCapture()
        let concurrentCapture = RequestCapture()
        let serial = makeAdapter(
            capture: serialCapture,
            defaults: [:],
            maxConcurrent: 1)
        let concurrent = makeAdapter(
            capture: concurrentCapture,
            defaults: [:],
            maxConcurrent: 4)

        for adapter in [serial, concurrent] {
            let result = try await adapter.generateStreaming(
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
                toolChoice: nil,
                parallelToolCalls: nil,
                stop: nil,
                responseFormat: nil,
                chatTemplateKwargs: nil,
                speculativeDecoding: nil,
                preserveStructuralTags: true,
                requestId: nil)
            for try await _ in result.stream {}
        }

        let serialRequest = await serialCapture.request(at: 0)
        let concurrentRequest = await concurrentCapture.request(at: 0)
        XCTAssertEqual(
            serialRequest.metadata[AFMMLXRequestMetadata.preserveStructuralTags],
            .bool(true))
        XCTAssertEqual(
            concurrentRequest.metadata[AFMMLXRequestMetadata.preserveStructuralTags],
            .bool(true))
    }

    private func makeAdapter(
        capture: RequestCapture,
        defaults: [String: AnyCodable],
        maxConcurrent: Int = 1
    ) -> AFMKitMLXChatServingAdapter {
        AFMKitMLXChatServingAdapter(
            model: AnyAFMModel(CapturingModel(
                capture: capture,
                maxConcurrent: maxConcurrent)),
            modelID: "capture",
            defaultChatTemplateKwargs: defaults
        )
    }

    private func generate(
        adapter: AFMKitMLXChatServingAdapter,
        kwargs: [String: AnyCodable]?,
        speculativeDecoding: SpeculativeDecodingOptions? = nil
    ) async throws -> AFMMLXChatGenerationResultWithTelemetry {
        try await adapter.generateWithTelemetry(
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
            toolChoice: nil,
            parallelToolCalls: nil,
            stop: nil,
            responseFormat: nil,
            chatTemplateKwargs: kwargs,
            speculativeDecoding: speculativeDecoding
        )
    }
}
