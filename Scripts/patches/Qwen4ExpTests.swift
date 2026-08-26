import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class Qwen4ExpTests: XCTestCase {
    private let minimalConfiguration = """
        {
          "model_type": "qwen4_exp",
          "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts_per_tok": 1,
            "num_experts": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "hc_count": 4,
            "hc_lowrank": 16,
            "ple_layer_ids": [],
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 64,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid",
            "eos_token_id": 31,
            "rope_parameters": {
              "partial_rotary_factor": 0.25,
              "rope_theta": 10000000
            }
          }
        }
        """

    func testToolCallFormatUsesQwenXMLProtocol() {
        XCTAssertEqual(ToolCallFormat.infer(from: "qwen4_exp"), .xmlFunction)
    }

    func testRegistryCreatesQwen4ExpModelFromNestedConfig() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )

        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        XCTAssertEqual(qwen.vocabularySize, 32)
        XCTAssertEqual(qwen.kvHeads, [0, 1])
        XCTAssertEqual(qwen.newCache(parameters: nil).count, 2)
    }

    func testSanitizeKeepsTextWeightsAndRenamesPLEShards() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([1]),
            "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight":
                MLXArray([Float(1.25)]),
            "language_model.model.layers.0.linear_attn.norm.weight": MLXArray([Float(1.25)]),
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight":
                MLXArray.zeros([1]),
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets":
                MLXArray.zeros([1]),
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes":
                MLXArray.zeros([1]),
            "language_model.mtp.proj.weight": MLXArray.zeros([1]),
            "visual.patch_embed.weight": MLXArray.zeros([1]),
        ]

        let sanitized = qwen.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertEqual(
            try XCTUnwrap(sanitized[
                "model.layers.0.attn_hyper_connection.hc_norm.weight"
            ]).item(Float.self),
            0.25, accuracy: 0.0001)
        XCTAssertEqual(
            try XCTUnwrap(sanitized["model.layers.0.linear_attn.norm.weight"]).item(Float.self),
            1.25, accuracy: 0.0001)
        XCTAssertNotNil(sanitized[
            "model.layers.1.ple.ple_embedding.ngram_embedding.shards.7.weight"
        ])
        XCTAssertNil(sanitized["mtp.proj.weight"])
        XCTAssertNil(sanitized["visual.patch_embed.weight"])
        XCTAssertNil(sanitized["model.layers.1.ple.ple_embedding.ngram_heads_offsets"])
        XCTAssertNil(sanitized["model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"])
    }

    func testTinyModelRunsPromptAndCachedToken() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        let promptLogits = qwen(MLXArray([1, 2, 3]).reshaped(1, 3), cache: cache)
        eval(promptLogits)
        XCTAssertEqual(promptLogits.shape, [1, 3, 32])

        let tokenLogits = qwen(MLXArray([4]).reshaped(1, 1), cache: cache)
        eval(tokenLogits)
        XCTAssertEqual(tokenLogits.shape, [1, 1, 32])
    }

    func testTinyPLEPathRunsPromptAndCachedToken() async throws {
        let pleConfiguration = minimalConfiguration.replacingOccurrences(
            of: "\"ple_layer_ids\": [],",
            with: """
                "ple_layer_ids": [1],
                "ple_embed_dim": 128,
                "ple_conv_kernel_size": 2,
                "ngram_size": 3,
                "heads_per_ngram": 2,
                "ngram_vocab_size_base": 101,
                "make_ngram_vocab_size_divisible_by": 4,
                "split_ngram_parts": 4,
                """
        )
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        let promptLogits = qwen(MLXArray([1, 2, 31, 3]).reshaped(1, 4), cache: cache)
        eval(promptLogits)
        XCTAssertEqual(promptLogits.shape, [1, 4, 32])

        let tokenLogits = qwen(MLXArray([4]).reshaped(1, 1), cache: cache)
        eval(tokenLogits)
        XCTAssertEqual(tokenLogits.shape, [1, 1, 32])
    }

    func testQSAPathRunsBeyondTokenBudget() async throws {
        let qsaConfiguration = minimalConfiguration
            .replacingOccurrences(of: "\"indexer_budget\": 2048", with: "\"indexer_budget\": 8")
            .replacingOccurrences(of: "\"indexer_compress_ratio\": 4", with: "\"indexer_compress_ratio\": 2")
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(qsaConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        let prompt = MLXArray(Array(1 ... 12).map { $0 % 31 }).reshaped(1, 12)
        let promptLogits = qwen(prompt, cache: cache)
        eval(promptLogits)
        XCTAssertEqual(promptLogits.shape, [1, 12, 32])

        let tokenLogits = qwen(MLXArray([13]).reshaped(1, 1), cache: cache)
        eval(tokenLogits)
        XCTAssertEqual(tokenLogits.shape, [1, 1, 32])
    }
}
