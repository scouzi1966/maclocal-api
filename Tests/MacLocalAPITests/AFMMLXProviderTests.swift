import AFMKitCore
import AFMOpenAICompat
@testable import AFMKitMLX
import MLXLMCommon
import XCTest

final class AFMMLXProviderTests: XCTestCase {
    func testFactoryExposesStableProviderIdentityAndConfiguration() {
        let descriptor = AFMMLXProviderFactory().descriptor

        XCTAssertEqual(descriptor.id, "mlx")
        XCTAssertEqual(descriptor.privacyBoundary, .device)
        XCTAssertTrue(descriptor.configurationKeys.contains("enablePrefixCaching"))
        XCTAssertTrue(descriptor.configurationKeys.contains("maxConcurrent"))
    }

    func testMLXModelExposesPortableTokenizationCapability() {
        let model = AFMMLXModel(modelID: "test/model")

        requirePortableTokenizer(model)
    }

    func testMLXModelServiceExposesAFMKitProfilingContract() {
        requireProfilingContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testMLXModelServiceExposesAFMKitRequestSchedulingContract() {
        requireRequestSchedulingContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testMLXModelServiceExposesAFMKitBatchControlContract() {
        requireBatchControlContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testMLXModelServiceExposesAFMKitServingConfigurationContract() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.toolCallParser = "qwen3_xml"
        service.enableGrammarConstraints = true
        service.fixToolArgs = true

        requireServingConfigurationContract(service)

        let configuration = service.servingConfiguration
        XCTAssertEqual(configuration.toolCallParser, "qwen3_xml")
        XCTAssertTrue(configuration.supportsStrictToolGrammar)
        XCTAssertTrue(configuration.fixToolArguments)
        XCTAssertTrue(configuration.grammarConstraintsEnabled)
    }

    func testMLXModelServiceExposesAFMKitOpenAIChatGenerationContract() {
        requireOpenAIChatGenerationContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testMLXModelServiceExposesAFMKitOpenAIChatServingContract() {
        requireOpenAIChatServingContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testServingConfigurationProviderOwnsPolicyConveniences() throws {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.toolCallParser = "qwen3_xml"
        service.fixToolArgs = true

        let strictSchema = ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: "answer",
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: true
            )
        )

        XCTAssertEqual(service.toolCallParser, "qwen3_xml")
        XCTAssertTrue(service.supportsStrictToolGrammar)
        XCTAssertTrue(service.isToolCallParserDisabled(" none "))
        XCTAssertTrue(
            service.shouldDowngradeGrammarConstraints(
                responseFormat: strictSchema,
                tools: nil
            )
        )

        let coerced = service.coerceToolCallArguments(
            ResponseToolCall(
                index: 0,
                id: "call_test",
                type: "function",
                function: ResponseToolCallFunction(
                    name: "get_weather",
                    arguments: #"{"days":"5"}"#
                )
            ),
            tools: [
                RequestTool(
                    type: "function",
                    function: RequestToolFunction(
                        name: "get_weather",
                        description: nil,
                        parameters: AnyCodable([
                            "type": "object",
                            "properties": ["days": ["type": "integer"]],
                            "required": ["days"]
                        ]),
                        strict: nil
                    )
                )
            ]
        )
        let data = try XCTUnwrap(coerced.function.arguments.data(using: .utf8))
        let arguments = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(arguments["days"] as? Int, 5)
    }

    func testAFMKitOwnsChatGenerationResultShape() {
        let result: AFMMLXChatGenerationResult = (
            modelID: "test/model",
            content: "ok",
            promptTokens: 1,
            completionTokens: 1,
            tokenLogprobs: nil,
            toolCalls: nil,
            cachedTokens: 0,
            promptTime: 0,
            generateTime: 0,
            stoppedBySequence: false
        )

        XCTAssertEqual(result.modelID, "test/model")
        XCTAssertEqual(result.content, "ok")
    }

    func testDescriptorInfersCapabilitiesFromModelAssets() throws {
        let root = try makeModelCache(
            config: [
                "max_position_embeddings": 65_536,
                "vision_config": ["model_type": "vision"]
            ],
            tokenizer: [
                "chat_template": "{% if tools %}<tool_call>{% endif %}<think>"
            ],
            generation: ["enable_thinking": true],
            includeMTP: true
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let descriptor = AFMMLXModelDescriptor.describe(
            modelID: "test/model",
            resolver: MLXCacheResolver(cacheRoot: root)
        )

        XCTAssertEqual(descriptor.contextWindow, 65_536)
        XCTAssertEqual(descriptor.requiresNetwork, false)
        XCTAssertTrue(descriptor.capabilities.contains(.text))
        XCTAssertTrue(descriptor.capabilities.contains(.vision))
        XCTAssertTrue(descriptor.capabilities.contains(.reasoning))
        XCTAssertTrue(descriptor.capabilities.contains(.toolCalling))
        XCTAssertTrue(descriptor.capabilities.contains(.structuredOutput))
        XCTAssertTrue(descriptor.capabilities.contains(.speculativeDecoding))
    }

    func testDescriptorInfersVisionFromArchitectureMetadata() throws {
        let root = try makeModelCache(
            config: [
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "model_type": "qwen2"
            ]
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let descriptor = AFMMLXModelDescriptor.describe(
            modelID: "test/model",
            resolver: MLXCacheResolver(cacheRoot: root)
        )

        XCTAssertTrue(descriptor.capabilities.contains(.vision))
    }

    func testDescriptorReadsVisionConfigurationFromModelDirectory() throws {
        let root = try makeModelCache(
            config: [
                "model_type": "gemma3",
                "text_config": ["model_type": "gemma"],
                "vision_config": ["model_type": "siglip"],
            ]
        )
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertTrue(
            AFMMLXModelDescriptor.isVisionModelConfiguration(
                in: root.appendingPathComponent("test/model")
            )
        )
    }

    func testDescriptorRequiresVisionFactoryForSparseVLMTextConfig() {
        XCTAssertTrue(
            AFMMLXModelDescriptor.requiresVisionModelFactory([
                "model_type": "gemma3",
                "text_config": ["model_type": "gemma"],
                "vision_config": ["model_type": "siglip"],
            ])
        )
    }

    func testDescriptorDoesNotRequireVisionFactoryWhenTextConfigHasArchitectureFields() {
        XCTAssertFalse(
            AFMMLXModelDescriptor.requiresVisionModelFactory([
                "model_type": "qwen3_5_moe",
                "text_config": [
                    "model_type": "qwen3_5_moe",
                    "num_attention_heads": 16,
                ],
                "vision_config": ["model_type": "qwen3_vl"],
            ])
        )
    }

    func testDescriptorReadsVisionFactoryRequirementFromModelDirectory() throws {
        let root = try makeModelCache(
            config: [
                "model_type": "gemma3",
                "text_config": ["model_type": "gemma"],
                "vision_config": ["model_type": "siglip"],
            ]
        )
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertTrue(
            AFMMLXModelDescriptor.requiresVisionModelFactory(
                in: root.appendingPathComponent("test/model")
            )
        )
    }

    func testUncachedDescriptorReportsNetworkRequirement() {
        let descriptor = AFMMLXModelDescriptor.describe(
            modelID: "missing/model",
            resolver: MLXCacheResolver()
        )

        XCTAssertEqual(descriptor.providerID, "mlx")
        XCTAssertEqual(descriptor.requiresNetwork, true)
        XCTAssertEqual(descriptor.privacyBoundary, .device)
    }

    func testGrammarPolicyDowngradesStrictSchemaWithoutAdminOptIn() {
        let strictSchema = ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: "answer",
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: true
            )
        )

        XCTAssertTrue(
            AFMMLXGrammarPolicy.shouldDowngradeGrammarConstraints(
                responseFormat: strictSchema,
                tools: nil,
                supportsStrictToolGrammar: true,
                enableGrammarConstraints: false
            )
        )
        XCTAssertFalse(
            AFMMLXGrammarPolicy.shouldDowngradeGrammarConstraints(
                responseFormat: strictSchema,
                tools: nil,
                supportsStrictToolGrammar: true,
                enableGrammarConstraints: true
            )
        )
    }

    func testToolPolicyDisablesExplicitNoneParser() {
        XCTAssertTrue(AFMMLXToolCallPolicy.isToolCallParserDisabled(" none "))
        XCTAssertFalse(AFMMLXToolCallPolicy.isToolCallParserDisabled("afm_adaptive_xml"))
    }

    func testToolPolicyNormalizesAndCoercesArguments() throws {
        let tool = RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: "get_weather",
                description: nil,
                parameters: AnyCodable([
                    "type": "object",
                    "properties": [
                        "days": ["type": "integer"],
                        "includeWind": ["type": "boolean"]
                    ]
                ]),
                strict: nil
            )
        )
        let rawCall = ToolCall(function: .init(
            name: "get_weather",
            arguments: [
                "days": "5",
                "includeWind": "true"
            ]
        ))

        let normalized = AFMMLXToolCallPolicy.normalizeToolCalls([rawCall], tools: [tool])

        XCTAssertEqual(normalized.count, 1)
        let data = try XCTUnwrap(normalized.first?.function.arguments.data(using: .utf8))
        let arguments = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(arguments["days"] as? Int, 5)
        XCTAssertEqual(arguments["includeWind"] as? Bool, true)
    }

    func testRequiredToolPolicyRejectsRequestWithoutEnabledTools() {
        let request = AFMRequest(
            messages: [],
            metadata: ["toolCallingMode": .string("required")]
        )

        XCTAssertThrowsError(
            try AFMMLXToolPolicy.validateCompletedToolCalls([], for: request)
        ) { error in
            XCTAssertEqual(
                error as? AFMError,
                .invalidRequest("Tool calling is required, but no tools are enabled.")
            )
        }
    }

    func testRequiredToolPolicyRejectsTextOnlyCompletion() {
        let request = requiredToolRequest()

        XCTAssertThrowsError(
            try AFMMLXToolPolicy.validateCompletedToolCalls([], for: request)
        ) { error in
            XCTAssertEqual(
                error as? AFMError,
                .generationFailed(
                    "The model returned no tool call while tool calling was required."
                )
            )
        }
    }

    func testRequiredToolPolicyAcceptsCompletedToolCall() throws {
        try AFMMLXToolPolicy.validateCompletedToolCalls(
            [
                AFMToolCall(
                    id: "call_1",
                    name: "weather",
                    arguments: #"{"city":"Toronto"}"#
                )
            ],
            for: requiredToolRequest()
        )
    }

    func testAllowedToolPolicyAcceptsTextOnlyCompletion() throws {
        let request = AFMRequest(
            messages: [],
            tools: requiredToolRequest().tools,
            metadata: ["toolCallingMode": .string("allowed")]
        )

        try AFMMLXToolPolicy.validateCompletedToolCalls([], for: request)
    }

    private func makeModelCache(
        config: [String: Any],
        tokenizer: [String: Any] = [:],
        generation: [String: Any] = [:],
        includeMTP: Bool = false
    ) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("afmkit-provider-\(UUID().uuidString)")
        let model = root.appendingPathComponent("test/model")
        try FileManager.default.createDirectory(
            at: model,
            withIntermediateDirectories: true
        )
        try writeJSON(config, to: model.appendingPathComponent("config.json"))
        try writeJSON(
            tokenizer,
            to: model.appendingPathComponent("tokenizer_config.json")
        )
        try writeJSON(
            generation,
            to: model.appendingPathComponent("generation_config.json")
        )
        try Data().write(to: model.appendingPathComponent("model.safetensors"))
        if includeMTP {
            try Data().write(to: model.appendingPathComponent("mtp.safetensors"))
        }
        return root
    }

    private func requirePortableTokenizer<Tokenizer: AFMTextTokenizing>(
        _ tokenizer: Tokenizer
    ) {}

    private func requireProfilingContract<Profiler: AFMMLXAPIProfiling>(
        _ profiler: Profiler
    ) {}

    private func requireRequestSchedulingContract<Scheduler: AFMMLXRequestScheduling>(
        _ scheduler: Scheduler
    ) {}

    private func requireBatchControlContract<Controller: AFMMLXBatchControlling>(
        _ controller: Controller
    ) {}

    private func requireServingConfigurationContract<Provider: AFMMLXServingConfigurationProviding>(
        _ provider: Provider
    ) {}

    private func requireOpenAIChatGenerationContract<Generator: AFMMLXOpenAIChatGenerating>(
        _ generator: Generator
    ) {}

    private func requireOpenAIChatServingContract<Serving: AFMMLXOpenAIChatServing>(
        _ serving: Serving
    ) {}

    private func requiredToolRequest() -> AFMRequest {
        AFMRequest(
            messages: [],
            tools: [
                AFMToolDefinition(
                    name: "weather",
                    description: "Get weather.",
                    inputSchema: .object([
                        "type": .string("object"),
                        "properties": .object([:])
                    ])
                )
            ],
            metadata: ["toolCallingMode": .string("required")]
        )
    }

    private func writeJSON(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value).write(to: url)
    }
}
