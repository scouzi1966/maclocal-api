#if canImport(FoundationModels)
import CoreImage
import FoundationModels
@testable import AFMKit
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class MLXFoundationLanguageModelTests: XCTestCase {
    private struct TestCustomSegment: Transcript.CustomSegment {
        struct Content: Codable, Equatable, Sendable {
            let value: String
        }

        let id: String
        let content: Content
    }

    func testExecutorConfigurationIncludesModelAndRuntimeIdentity() {
        let first = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsVision: true,
            supportsReasoning: true,
            supportsToolCalling: true,
            supportsGuidedGeneration: true
        )
        let same = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsVision: true,
            supportsReasoning: true,
            supportsToolCalling: true,
            supportsGuidedGeneration: true
        )
        let otherModel = MLXLanguageModel(
            modelID: "mlx-community/model-b",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsVision: true,
            supportsReasoning: true,
            supportsToolCalling: true,
            supportsGuidedGeneration: true
        )

        XCTAssertEqual(first.executorConfiguration, same.executorConfiguration)
        XCTAssertNotEqual(first.executorConfiguration, otherModel.executorConfiguration)
        XCTAssertTrue(first.capabilities.contains(.vision))
        XCTAssertTrue(first.capabilities.contains(.reasoning))
        XCTAssertTrue(first.capabilities.contains(.toolCalling))
        XCTAssertTrue(first.capabilities.contains(.guidedGeneration))
    }

    func testTranscriptTranslationPreservesMultiTurnRoles() throws {
        let transcript = Transcript(entries: [
            .instructions(
                .init(
                    segments: [.text(.init(content: "Be concise."))],
                    toolDefinitions: []
                )
            ),
            .prompt(
                .init(segments: [.text(.init(content: "First question"))])
            ),
            .response(
                .init(
                    metadata: [:],
                    segments: [.text(.init(content: "First answer"))]
                )
            ),
            .prompt(
                .init(segments: [.text(.init(content: "Follow-up question"))])
            )
        ])

        let messages = try MLXFoundationRequestAdapter.messages(from: transcript)

        XCTAssertEqual(messages.map(\.role), ["system", "user", "assistant", "user"])
        XCTAssertEqual(
            messages.map(\.textContent),
            ["Be concise.", "First question", "First answer", "Follow-up question"]
        )
    }

    func testTranscriptTranslationPreservesToolCallsAndOutputs() throws {
        let call = Transcript.ToolCall(
            id: "call_1",
            toolName: "weather",
            arguments: try GeneratedContent(json: #"{"city":"Toronto"}"#)
        )
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Weather?"))])),
            .toolCalls(.init([call])),
            .toolOutput(
                .init(
                    id: "call_1",
                    toolName: "weather",
                    segments: [.text(.init(content: "Sunny"))]
                )
            )
        ])

        let messages = try MLXFoundationRequestAdapter.messages(from: transcript)

        XCTAssertEqual(messages.map(\.role), ["user", "assistant", "tool"])
        XCTAssertEqual(messages[1].toolCalls?.first?.id, "call_1")
        XCTAssertEqual(messages[1].toolCalls?.first?.function.name, "weather")
        let arguments = try XCTUnwrap(
            messages[1].toolCalls?.first?.function.arguments.data(using: .utf8)
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: arguments) as? [String: String]
        )
        XCTAssertEqual(object, ["city": "Toronto"])
        XCTAssertEqual(messages[2].toolCallId, "call_1")
        XCTAssertEqual(messages[2].name, "weather")
        XCTAssertEqual(messages[2].textContent, "Sunny")
    }

    func testTranscriptTranslationPreservesImageAttachments() throws {
        let image = CIImage(
            color: CIColor(red: 1, green: 0, blue: 0, alpha: 1)
        ).cropped(to: CGRect(x: 0, y: 0, width: 1, height: 1))
        let transcript = Transcript(entries: [
            .prompt(
                .init(
                    segments: [
                        .text(.init(content: "Describe this image.")),
                        .attachment(
                            .init(
                                content: .image(.init(image)),
                                label: "Reference image"
                            )
                        )
                    ]
                )
            )
        ])

        let messages = try MLXFoundationRequestAdapter.messages(from: transcript)
        guard case .parts(let parts)? = messages.first?.content else {
            return XCTFail("Expected multimodal message parts.")
        }
        XCTAssertEqual(parts.compactMap(\.text), [
            "Describe this image.",
            "Reference image"
        ])
        XCTAssertTrue(
            parts.compactMap(\.image_url?.url)
                .contains(where: { $0.hasPrefix("data:image/png;base64,") })
        )
    }

    func testTranscriptTranslationPreservesCustomSegments() throws {
        let transcript = Transcript(entries: [
            .prompt(
                .init(
                    segments: [
                        .custom(
                            TestCustomSegment(
                                id: "custom_1",
                                content: .init(value: "project-state")
                            )
                        )
                    ]
                )
            )
        ])

        let messages = try MLXFoundationRequestAdapter.messages(from: transcript)

        XCTAssertEqual(messages.count, 1)
        XCTAssertTrue(messages[0].textContent.contains("project-state"))
    }

    func testGenerationConfigMapsSamplingContextAndToolMode() throws {
        let model = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            defaultMaximumResponseTokens: 2_048,
            supportsToolCalling: true
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(),
            enabledTools: [],
            generationOptions: GenerationOptions(
                samplingMode: .random(top: 17, seed: 42),
                temperature: 0.7,
                maximumResponseTokens: 321,
                toolCallingMode: .disallowed
            ),
            contextOptions: ContextOptions(
                includeSchemaInPrompt: false,
                reasoningLevel: .deep
            ),
            metadata: ["requestID": "request-1"]
        )

        let config = try MLXFoundationRequestAdapter.generationConfig(
            from: request,
            model: model
        )

        XCTAssertEqual(config.temperature, 0.7)
        XCTAssertEqual(config.topK, 17)
        XCTAssertEqual(config.seed, 42)
        XCTAssertEqual(config.maxTokens, 321)
        XCTAssertNil(config.tools)
        XCTAssertEqual(config.metadata["includeSchemaInPrompt"], .bool(false))
        XCTAssertEqual(config.metadata["toolCallingMode"], .string("disallowed"))
        XCTAssertEqual(config.metadata["reasoningLevel"], .string("deep"))
        XCTAssertEqual(config.metadata["requestID"], .string("request-1"))
    }
}
#endif
