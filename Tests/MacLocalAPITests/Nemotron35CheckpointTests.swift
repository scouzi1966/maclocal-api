import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

@testable import AFMKitMLX

final class Nemotron35CheckpointTests: XCTestCase {
    private let modelID = "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-mxfp4"

    func testCheckpointArchitectureUsesNemotronHLanguageModelFactory() throws {
        let preflight = try AFMMLXModelArchitecture.preflightConfiguration(
            [
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "layers_block_type": ["mamba", "moe", "attention"],
                "quantization": ["mode": "mxfp4", "bits": 4, "group_size": 32],
            ],
            modelID: modelID
        )

        XCTAssertEqual(preflight.canonicalModelType, "nemotron_h")
        XCTAssertFalse(preflight.isVisionConfiguration)
        XCTAssertFalse(preflight.requiresVisionModelFactory)
    }

    func testCheckpointConfigurationNormalizesNamedBlockLayout() throws {
        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self,
            from: Data(checkpointConfigurationJSON.utf8)
        )

        XCTAssertEqual(config.modelType, "nemotron_h")
        XCTAssertEqual(config.hybridOverridePattern, "MEM*E")
        XCTAssertEqual(config.numHiddenLayers, 5)
        XCTAssertEqual(config.timeStepLimitMin, 0.001)
        XCTAssertEqual(config.timeStepLimitMax, .infinity)
        XCTAssertEqual(config.nSharedExperts, 1)
        XCTAssertEqual(config.routedScalingFactor, 2.5)
    }

    func testProductionCheckpointNormalizesAll52Layers() throws {
        let productionLayout = [
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
            "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba",
            "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
            "moe", "mamba", "moe",
        ]
        var productionConfiguration = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(checkpointConfigurationJSON.utf8))
                as? [String: Any]
        )
        productionConfiguration["layers_block_type"] = productionLayout
        let data = try JSONSerialization.data(withJSONObject: productionConfiguration)

        let config = try JSONDecoder().decode(NemotronHConfiguration.self, from: data)

        XCTAssertEqual(config.numHiddenLayers, 52)
        XCTAssertEqual(config.hybridOverridePattern.count, 52)
        XCTAssertEqual(config.hybridOverridePattern, productionLayout.map {
            switch $0 {
            case "mamba": return "M"
            case "attention": return "*"
            case "moe": return "E"
            default: XCTFail("Unexpected production block type: \($0)"); return "?"
            }
        }.joined())
    }

    func testCheckpointRejectsUnknownNamedBlockType() {
        let invalid = checkpointConfigurationJSON.replacingOccurrences(
            of: "\"attention\"",
            with: "\"unsupported_attention\""
        )

        XCTAssertThrowsError(try JSONDecoder().decode(
            NemotronHConfiguration.self,
            from: Data(invalid.utf8)
        ))
    }

    func testCheckpointToolFormatUsesQwen3CoderXML() {
        XCTAssertEqual(ToolCallFormat.infer(from: "nemotron_h"), .xmlFunction)
    }

    func testDisabledThinkingPromptDoesNotReopenClosedBlock() {
        XCTAssertFalse(MLXModelService.promptSuffixOpensThink(
            "assistant\n<think>\n\n</think>\n\n",
            startTag: "<think>",
            endTag: "</think>"
        ))
        XCTAssertTrue(MLXModelService.promptSuffixOpensThink(
            "assistant\n<think>\n",
            startTag: "<think>",
            endTag: "</think>"
        ))
    }

    func testCheckpointJinjaAdvertisesReasoningToggle() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Nemotron35Checkpoint-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try Data(#"{"model_type":"nemotron_h","max_position_embeddings":262144}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        try Data(#"{"temperature":1.0,"top_p":0.95}"#.utf8)
            .write(to: directory.appendingPathComponent("generation_config.json"))
        try Data("""
        {%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}
        {%- if add_generation_prompt %}
            {%- if enable_thinking %}<|im_start|>assistant\n<think>\n
            {%- else %}<|im_start|>assistant\n<think></think>
            {%- endif %}
        {%- endif %}
        """.utf8).write(to: directory.appendingPathComponent("chat_template.jinja"))

        let metadata = AFMMLXLocalModelMetadata.inspect(
            modelDirectory: directory,
            modelName: "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-mxfp4"
        )

        XCTAssertEqual(metadata.modelType, "nemotron_h")
        XCTAssertEqual(metadata.contextWindow, 262_144)
        XCTAssertEqual(metadata.generationPreset?.temperature, 1.0)
        XCTAssertEqual(metadata.generationPreset?.topP, 0.95)
        XCTAssertTrue(metadata.hasImplicitReasoning)
        XCTAssertTrue(metadata.supportsThinkingToggle)
    }

    private var checkpointConfigurationJSON: String {
        """
        {
          "architectures": ["NemotronHForCausalLM"],
          "model_type": "nemotron_h",
          "vocab_size": 131072,
          "hidden_size": 64,
          "num_hidden_layers": 52,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "mamba_num_heads": 4,
          "mamba_head_dim": 16,
          "mamba_proj_bias": false,
          "ssm_state_size": 16,
          "conv_kernel": 4,
          "n_groups": 2,
          "intermediate_size": 128,
          "moe_intermediate_size": 64,
          "moe_shared_expert_intermediate_size": 128,
          "moe_latent_size": null,
          "n_routed_experts": 8,
          "n_shared_experts": 1,
          "num_experts_per_tok": 2,
          "layers_block_type": ["mamba", "moe", "mamba", "attention", "moe"],
          "layer_norm_epsilon": 1e-5,
          "mlp_bias": false,
          "use_bias": false,
          "use_conv_bias": true,
          "tie_word_embeddings": false,
          "head_dim": 16,
          "n_group": 1,
          "topk_group": 1,
          "norm_topk_prob": true,
          "routed_scaling_factor": 2.5,
          "time_step_min": 0.001,
          "time_step_max": 0.1
        }
        """
    }
}
