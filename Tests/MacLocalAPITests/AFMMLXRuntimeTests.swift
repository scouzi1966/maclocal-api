import AFMKitCore
import AFMOpenAICompat
@testable import AFMKitMLX
import XCTest

final class AFMMLXRuntimeTests: XCTestCase {
    func testRuntimeConfigurationAppliesTypedSettingsToService() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        let guidedSchema = ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: "answer",
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: true
            )
        )

        AFMMLXRuntimeConfiguration(
            kvBits: 4,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 5,
            eagle3DrafterPath: "/tmp/eagle",
            maxConcurrent: 4,
            toolCallParser: "qwen3_xml",
            enableGrammarConstraints: true,
            prefillStepSize: 256,
            kvEvictionPolicy: "streaming",
            fixToolArguments: true,
            forceVLM: true,
            cacheProfilePath: "/tmp/cache.json",
            trace: true,
            gpuCapturePath: "/tmp/capture.gputrace",
            gpuTraceDuration: 3,
            gpuProfile: true,
            gpuProfileBandwidth: true,
            defaultChatTemplateKwargs: [
                "enable_thinking": .bool(false),
                "top_k": .integer(20)
            ],
            defaultGuidedJsonSchema: guidedSchema
        ).apply(to: service)

        XCTAssertEqual(service.kvBits, 4)
        XCTAssertTrue(service.enablePrefixCaching)
        XCTAssertTrue(service.mtpEnabled)
        XCTAssertEqual(service.mtpDepth, 5)
        XCTAssertEqual(service.eagle3DrafterPath, "/tmp/eagle")
        XCTAssertEqual(service.maxConcurrent, 4)
        XCTAssertEqual(service.toolCallParser, "qwen3_xml")
        XCTAssertTrue(service.enableGrammarConstraints)
        XCTAssertEqual(service.prefillStepSize, 256)
        XCTAssertEqual(service.kvEvictionPolicy, "streaming")
        XCTAssertTrue(service.fixToolArgs)
        XCTAssertTrue(service.forceVLM)
        XCTAssertEqual(service.cacheProfilePath, "/tmp/cache.json")
        XCTAssertTrue(service.trace)
        XCTAssertEqual(service.gpuCapturePath, "/tmp/capture.gputrace")
        XCTAssertEqual(service.gpuTraceDuration, 3)
        XCTAssertTrue(service.gpuProfile)
        XCTAssertTrue(service.gpuProfileBandwidth)
        let templateKwargs = service.defaultChatTemplateKwargs
        let enableThinking = templateKwargs?["enable_thinking"] as? Bool
        let topK = templateKwargs?["top_k"] as? Int
        XCTAssertEqual(enableThinking, false)
        XCTAssertEqual(topK, 20)
        XCTAssertEqual(service.defaultGuidedJsonSchema?.type, "json_schema")
    }

    func testRuntimeConfigurationDisablesBatchModeForSingleConcurrency() {
        let service = MLXModelService(resolver: MLXCacheResolver())

        AFMMLXRuntimeConfiguration(maxConcurrent: 1).apply(to: service)

        XCTAssertEqual(service.maxConcurrent, 0)
    }

    func testRuntimeNormalizesModelIDAndAppliesProviderConfiguration() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        let runtime = AFMMLXRuntime(
            modelID: "Qwen3.6-35B-A3B-4bit",
            providerConfiguration: AFMProviderConfiguration(values: [
                "enablePrefixCaching": .bool(false),
                "maxConcurrent": .integer(8),
                "toolCallParser": .string("none")
            ]),
            service: service
        )

        XCTAssertEqual(runtime.modelID, "mlx-community/Qwen3.6-35B-A3B-4bit")
        XCTAssertFalse(service.enablePrefixCaching)
        XCTAssertEqual(service.maxConcurrent, 8)
        XCTAssertEqual(service.toolCallParser, "none")
    }
}
