#if canImport(FoundationModels)
import CoreImage
import FoundationModels
@testable import AFMKit
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class MLXFoundationLanguageModelTests: XCTestCase {
    @Generable
    struct TestStructuredAnswer {
        @Guide(description: "Short answer text")
        let answer: String
    }

    @Generable
    struct TestWeatherToolArguments {
        @Guide(description: "City name")
        let city: String
    }

    private struct TestCustomSegment: Transcript.CustomSegment {
        struct Content: Codable, Equatable, Sendable {
            let value: String
        }

        let id: String
        let content: Content
    }

    func testLanguageModelPlanProjectsAFMKitDescriptorCapabilities() {
        let descriptor = AFMModelDescriptor(
            providerID: "afmkit.mlx",
            modelID: "mlx-community/model-a",
            displayName: "Model A",
            capabilities: [.text, .vision, .reasoning, .toolCalling, .structuredOutput],
            contextWindow: 16_384,
            privacyBoundary: .device,
            requiresNetwork: false
        )

        let plan = AFMMLXFoundationLanguageModelPlan.make(
            modelID: "/cache/model-a",
            descriptor: descriptor,
            defaultMaximumResponseTokens: 768
        )

        XCTAssertEqual(plan.modelID, "/cache/model-a")
        XCTAssertEqual(plan.defaultMaximumResponseTokens, 768)
        XCTAssertTrue(plan.enablePrefixCaching)
        XCTAssertTrue(plan.supportsVision)
        XCTAssertTrue(plan.supportsReasoning)
        XCTAssertTrue(plan.supportsToolCalling)
        XCTAssertTrue(plan.supportsGuidedGeneration)
        XCTAssertTrue(plan.acceptsImageInput(true))
        XCTAssertFalse(plan.acceptsImageInput(false))
    }

    func testLanguageModelPlanBuildsMLXLanguageModelConfiguration() {
        let plan = AFMMLXFoundationLanguageModelPlan(
            modelID: "/cache/model-a",
            defaultMaximumResponseTokens: 384,
            enablePrefixCaching: false,
            supportsVision: true,
            supportsReasoning: true,
            supportsToolCalling: false,
            supportsGuidedGeneration: true
        )

        let model = plan.languageModel()

        XCTAssertEqual(model.modelID, "/cache/model-a")
        XCTAssertEqual(model.executorConfiguration.defaultMaximumResponseTokens, 384)
        XCTAssertFalse(model.executorConfiguration.enablePrefixCaching)
        XCTAssertTrue(model.executorConfiguration.supportsVision)
        XCTAssertTrue(model.executorConfiguration.supportsReasoning)
        XCTAssertFalse(model.executorConfiguration.supportsToolCalling)
        XCTAssertTrue(model.executorConfiguration.supportsGuidedGeneration)
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
        XCTAssertNil(config.metadata["chatTemplateKwargs"])
        XCTAssertEqual(config.metadata["requestID"], .string("request-1"))
    }

    func testGenerationConfigDisablesThinkingByDefaultForReasoningModels() throws {
        let model = MLXLanguageModel(
            modelID: "mlx-community/qwen-reasoning",
            defaultMaximumResponseTokens: 2_048,
            supportsReasoning: true
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(),
            enabledTools: [],
            generationOptions: GenerationOptions(maximumResponseTokens: 128),
            contextOptions: ContextOptions(),
            metadata: [:]
        )

        let config = try MLXFoundationRequestAdapter.generationConfig(
            from: request,
            model: model
        )

        guard case .object(let kwargs)? = config.metadata["chatTemplateKwargs"] else {
            return XCTFail("Expected chatTemplateKwargs metadata.")
        }
        XCTAssertEqual(kwargs["enable_thinking"], .bool(false))
    }

    func testGenerationConfigEnablesThinkingForExplicitReasoningLevel() throws {
        let model = MLXLanguageModel(
            modelID: "mlx-community/qwen-reasoning",
            defaultMaximumResponseTokens: 2_048,
            supportsReasoning: true
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(),
            enabledTools: [],
            generationOptions: GenerationOptions(maximumResponseTokens: 128),
            contextOptions: ContextOptions(reasoningLevel: .moderate),
            metadata: [:]
        )

        let config = try MLXFoundationRequestAdapter.generationConfig(
            from: request,
            model: model
        )

        XCTAssertEqual(config.metadata["reasoningLevel"], .string("moderate"))
        guard case .object(let kwargs)? = config.metadata["chatTemplateKwargs"] else {
            return XCTFail("Expected chatTemplateKwargs metadata.")
        }
        XCTAssertEqual(kwargs["enable_thinking"], .bool(true))
    }

    func testGenerationConfigMapsGenerationSchemaToStrictJSONResponseFormat() throws {
        let model = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            defaultMaximumResponseTokens: 2_048,
            supportsGuidedGeneration: true
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(),
            enabledTools: [],
            schema: GenerationSchema(
                type: TestStructuredAnswer.self,
                properties: [
                    GenerationSchema.Property(
                        name: "answer",
                        description: "Short answer text",
                        type: String.self
                    )
                ]
            ),
            generationOptions: GenerationOptions(maximumResponseTokens: 128),
            contextOptions: ContextOptions(),
            metadata: [:]
        )

        let config = try MLXFoundationRequestAdapter.generationConfig(
            from: request,
            model: model
        )

        XCTAssertEqual(config.responseFormat?.type, "json_schema")
        XCTAssertEqual(config.responseFormat?.jsonSchema?.name, "TestStructuredAnswer")
        XCTAssertEqual(config.responseFormat?.jsonSchema?.strict, true)
        let schemaPayload = try XCTUnwrap(config.responseFormat?.jsonSchema?.schema)
        guard case .object(let schema) = schemaPayload.value else {
            XCTFail("Expected object schema, got \(schemaPayload.value)")
            return
        }
        guard case .string("object")? = schema["type"] else {
            XCTFail("Expected object schema type, got \(String(describing: schema["type"]))")
            return
        }
        guard case .object(let properties)? = schema["properties"] else {
            XCTFail("Expected object properties, got \(String(describing: schema["properties"]))")
            return
        }
        XCTAssertNotNil(properties["answer"])
    }

    func testGenerationConfigForwardsEnabledToolDefinitions() throws {
        let model = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            defaultMaximumResponseTokens: 2_048,
            supportsToolCalling: true
        )
        let weatherTool = Transcript.ToolDefinition(
            name: "weather",
            description: "Look up weather by city.",
            parameters: GenerationSchema(
                type: TestWeatherToolArguments.self,
                properties: [
                    GenerationSchema.Property(
                        name: "city",
                        description: "City name",
                        type: String.self
                    )
                ]
            )
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [
                .prompt(.init(segments: [.text(.init(content: "Weather in Toronto?"))]))
            ]),
            enabledTools: [weatherTool],
            generationOptions: GenerationOptions(toolCallingMode: .required),
            contextOptions: ContextOptions(),
            metadata: [:]
        )

        let config = try MLXFoundationRequestAdapter.generationConfig(
            from: request,
            model: model
        )

        let tool = try XCTUnwrap(config.tools?.first)
        XCTAssertEqual(tool.type, "function")
        XCTAssertEqual(tool.function.name, "weather")
        XCTAssertEqual(tool.function.description, "Look up weather by city.")
        XCTAssertEqual(tool.function.strict, true)
        guard case .object(let parameters) = try XCTUnwrap(tool.function.parameters).value else {
            XCTFail("Expected object parameters, got \(String(describing: tool.function.parameters?.value))")
            return
        }
        guard case .object(let properties)? = parameters["properties"] else {
            XCTFail("Expected tool parameter properties, got \(String(describing: parameters["properties"]))")
            return
        }
        XCTAssertNotNil(properties["city"])
        XCTAssertEqual(config.metadata["toolCallingMode"], .string("required"))
    }

    func testEventChannelAdapterTracksFallbackUsageFromTextTokens() {
        var adapter = MLXFoundationEventChannelAdapter()

        XCTAssertEqual(
            adapter.consume(.text("Hello", tokenCount: 2)),
            .responseText("Hello", tokenCount: 2)
        )
        XCTAssertEqual(
            adapter.consume(.reasoning("Think", tokenCount: 3)),
            .reasoningText("Think", tokenCount: 3)
        )

        XCTAssertEqual(adapter.finishPlan(), .usage(AFMUsage(outputTokens: 2)))
    }

    func testEventChannelAdapterSuppressesFallbackAfterUsageEvent() {
        var adapter = MLXFoundationEventChannelAdapter()

        XCTAssertEqual(
            adapter.consume(.text("Hello", tokenCount: 2)),
            .responseText("Hello", tokenCount: 2)
        )
        XCTAssertEqual(
            adapter.consume(.usage(promptTokens: 7, completionTokens: 11, cachedTokens: 5)),
            .usage(AFMUsage(inputTokens: 7, cachedInputTokens: 5, outputTokens: 11))
        )

        XCTAssertNil(adapter.finishPlan())
    }

    func testEventChannelAdapterMapsToolMetadataAndFinishEvents() {
        var adapter = MLXFoundationEventChannelAdapter()
        let call = AFMToolCall(id: "call_1", name: "weather", arguments: "")

        XCTAssertEqual(
            adapter.consume(.toolCall(call, stage: .started)),
            .toolArguments(id: "call_1", name: "weather", arguments: "")
        )
        XCTAssertEqual(
            adapter.consume(.toolCall(call, stage: .argumentsDelta("{\"city\":\"Paris\"}"))),
            .toolArguments(id: "call_1", name: "weather", arguments: "{\"city\":\"Paris\"}")
        )
        XCTAssertNil(adapter.consume(.toolCall(call, stage: .completed)))
        XCTAssertEqual(
            adapter.consume(.metadata(["provider": .string("mlx")])),
            .metadata(["provider": .string("mlx")])
        )
        XCTAssertEqual(
            adapter.consume(.custom(type: "blob", payload: Data([0x01, 0x02]))),
            .customMetadata(key: "afm.custom.blob", value: "AQI=")
        )
        XCTAssertEqual(
            adapter.consume(.completed(.length)),
            .finishReason("length")
        )
        XCTAssertNil(adapter.consume(.tokenLogprobs([])))
    }
}
#endif
