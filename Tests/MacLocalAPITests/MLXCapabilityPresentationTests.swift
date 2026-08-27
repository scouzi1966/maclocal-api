import AFMKitCore
import AFMOpenAICompat
@testable import AFMServer
import XCTest

final class MLXCapabilityPresentationTests: XCTestCase {
    func testLoadedDescriptorPublishesOnlyProvenCapabilities() {
        let descriptor = AFMModelDescriptor(
            providerID: "mlx",
            modelID: "qualified-model",
            displayName: "qualified-model",
            capabilities: [
                .text,
                .vision,
                .reasoning,
                .toolCalling,
                .structuredOutput,
                .streaming,
                .prefixCaching,
                .speculativeDecoding,
            ],
            privacyBoundary: .device
        )

        XCTAssertTrue(AFMMLXCapabilityPresentation.supportsVision(descriptor: descriptor))
        XCTAssertEqual(
            Set(AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: descriptor)),
            [
                "chat",
                "completion",
                "vision",
                "reasoning",
                "tools",
                "structured",
                "streaming",
                "prefix_cache",
                "speculative_decoding",
                "mlx_runtime",
                "batch",
                "context_window_override",
                "kv_quantization",
                "logprobs",
                "penalties",
                "prefill_tuning",
            ]
        )
    }

    func testMissingDescriptorFailsClosedForOptionalCapabilities() {
        XCTAssertFalse(AFMMLXCapabilityPresentation.supportsVision(descriptor: nil))
        XCTAssertEqual(
            AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: nil),
            ["chat", "completion"]
        )
    }

    func testTextDescriptorDoesNotInventVisionOrSpeculation() {
        let descriptor = AFMModelDescriptor(
            providerID: "mlx",
            modelID: "text-model",
            displayName: "text-model",
            capabilities: [.text, .streaming],
            privacyBoundary: .device
        )

        XCTAssertFalse(AFMMLXCapabilityPresentation.supportsVision(descriptor: descriptor))
        XCTAssertEqual(
            Set(AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: descriptor)),
            [
                "chat", "completion", "streaming", "mlx_runtime", "batch", "context_window_override", "kv_quantization",
                "logprobs", "penalties", "prefill_tuning",
            ]
        )
    }

    func testDwarfStarDescriptorDoesNotAdvertiseMLXOnlyEngineFeatures() {
        let descriptor = AFMModelDescriptor(
            providerID: "dwarfstar",
            modelID: "deepseek",
            displayName: "deepseek",
            capabilities: [.text, .streaming, .reasoning, .toolCalling, .prefixCaching],
            privacyBoundary: .device
        )

        let labels = Set(
            AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: descriptor)
        )
        XCTAssertTrue(labels.contains("tools"))
        XCTAssertTrue(labels.contains("prefix_cache"))
        XCTAssertTrue(labels.contains("dwarfstar_runtime"))
        XCTAssertFalse(labels.contains("batch"))
        XCTAssertFalse(labels.contains("logprobs"))
        XCTAssertFalse(labels.contains("structured"))
        XCTAssertFalse(labels.contains("penalties"))
    }

    func testDeclaredMediaDetectionDoesNotRequirePayloadField() {
        let messages = [
            Message(
                role: "user",
                content: .parts([ContentPart(type: "image_url")])
            )
        ]

        XCTAssertTrue(MLXChatCompletionsController.containsDeclaredMedia(messages))
    }

    func testVerboseRequestRenderingRedactsMediaSecrets() throws {
        let json = #"""
        {
          "model": "test-model",
          "messages": [{
            "role": "user",
            "content": [
              {"type":"image_url","image_url":{"url":"https://example.com/a.png?token=signed-secret"}},
              {"type":"input_audio","input_audio":{"data":"private-base64-audio","format":"wav"}}
            ]
          }]
        }
        """#
        let request = try JSONDecoder().decode(
            ChatCompletionRequest.self,
            from: Data(json.utf8)
        )

        let rendered = MLXChatCompletionsController.redactedRequestJSON(request)

        XCTAssertFalse(rendered.contains("signed-secret"))
        XCTAssertFalse(rendered.contains("private-base64-audio"))
        XCTAssertTrue(rendered.contains("<redacted-media-reference>"))
        XCTAssertTrue(rendered.contains("<redacted-audio-data>"))
    }
}
