// Tests for the GLM-5.2 DSA fixes ported from mlx-lm PR #1454:
//  1. Indexer head reduction via elementwise adds (workaround for ml-explore/mlx#3784)
//     must equal the fused sum(axis: 1).
//  2. sparseGatherAttention (long-context prefill) must produce the same output as the
//     dense top-k-masked prefill path on identical inputs — it attends over the exact
//     same selected key set.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

public class GLM5DsaTests: XCTestCase {

    /// Tolerance for float32 path equivalence. Upstream measured ~3e-7 for the
    /// dense-vs-sparse parity; 1e-4 leaves headroom for op-ordering differences.
    private static let parityTolerance: Float = 1e-4

    private func tinyConfig() throws -> GLM5MoeDsaConfiguration {
        let json = """
            {
              "vocab_size": 128,
              "hidden_size": 64,
              "intermediate_size": 128,
              "moe_intermediate_size": 64,
              "num_hidden_layers": 1,
              "num_attention_heads": 4,
              "num_key_value_heads": 4,
              "kv_lora_rank": 32,
              "q_lora_rank": 32,
              "qk_rope_head_dim": 16,
              "qk_nope_head_dim": 32,
              "v_head_dim": 32,
              "num_experts_per_tok": 2,
              "n_routed_experts": 4,
              "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
              "indexer_types": ["full", "shared"],
              "index_head_dim": 32,
              "index_n_heads": 4,
              "index_topk": 32
            }
            """
        return try JSONDecoder().decode(
            GLM5MoeDsaConfiguration.self, from: json.data(using: .utf8)!)
    }

    /// Boolean causal mask matching GLM5MoeDsaModelInner.createBoolMask.
    private func boolCausalMask(length: Int, offset: Int) -> MLXArray {
        let rowIdx = MLXArray(0 ..< length).reshaped(length, 1) + offset
        let colIdx = MLXArray(0 ..< (length + offset)).reshaped(1, length + offset)
        return rowIdx .>= colIdx
    }

    /// GLM-5.2 ships "rope_parameters": {"rope_theta": 8000000} — a whole JSON number,
    /// which StringOrNumber decodes as .int. A .float-only match silently fell back to
    /// the 1M default and scrambled RoPE at depth. indexer_types gates per-layer
    /// indexer creation ("shared" layers have no indexer weights in the checkpoint).
    func testConfigParsesRopeParametersAndIndexerTypes() throws {
        let config = try tinyConfig()
        XCTAssertEqual(config.ropeTheta, 8_000_000)
        XCTAssertEqual(config.indexerTypes, ["full", "shared"])
    }

    func testIndexerHeadReductionEquivalence() {
        MLXRandom.seed(3)
        let nHeads = 8
        let scores = MLXRandom.uniform(0.0 ..< 1.0, [1, nHeads, 16, 256])

        let fused = scores.sum(axis: 1, keepDims: true)
        var summed = scores[0..., 0 ..< 1]
        for h in 1 ..< nHeads {
            summed = summed + scores[0..., h ..< (h + 1)]
        }

        let maxDiff = abs(fused - summed).max().item(Float.self)
        XCTAssertLessThanOrEqual(
            maxDiff, Self.parityTolerance,
            "elementwise-add head reduction diverges from fused sum")
    }

    func testSparseGatherMatchesDenseMaskedPrefill() throws {
        let config = try tinyConfig()
        MLXRandom.seed(42)
        let attention = GLM5MoeDsaAttention(config, layerIdx: 0)

        // embedQ/unembedOut (GLM5MoeDsaMultiLinear) init with SCALAR placeholders —
        // real tensors normally arrive from the checkpoint. Inject random weights so
        // the bare module is runnable. Logical shape is [numHeads, outputDims, inputDims].
        let h = config.numAttentionHeads
        let scale: Float = 0.05
        try attention.update(
            parameters: ModuleParameters.unflattened([
                "embed_q.weight": MLXRandom.normal([h, config.kvLoraRank, config.qkNopeHeadDim]) * scale,
                "unembed_out.weight": MLXRandom.normal([h, config.vHeadDim, config.kvLoraRank]) * scale,
            ]), verify: [])

        let priorLength = 64  // cached context before the chunk under test
        let chunkLength = 96  // totalSeq = 160 > indexTopk (32) => sparse selection active

        MLXRandom.seed(7)
        let priorX = MLXRandom.normal([1, priorLength, config.hiddenSize])
        let chunkX = MLXRandom.normal([1, chunkLength, config.hiddenSize])

        /// Run prior + chunk through the attention layer with fresh caches, with the
        /// sparse-prefill gate set so the chunk takes either the dense-masked path
        /// (gate above totalSeq) or the sparse-gather path (gate below totalSeq).
        func run(gate: Int) -> MLXArray {
            let savedGate = GLM5MoeDsaAttention.sparsePrefillMinContext
            GLM5MoeDsaAttention.sparsePrefillMinContext = gate
            defer { GLM5MoeDsaAttention.sparsePrefillMinContext = savedGate }

            let cache = CacheList(KVCacheSimple(), KVCacheSimple())
            _ = attention(
                priorX, mask: boolCausalMask(length: priorLength, offset: 0), cache: cache,
                prevTopkIndices: nil)
            let (out, _) = attention(
                chunkX, mask: boolCausalMask(length: chunkLength, offset: priorLength),
                cache: cache, prevTopkIndices: nil)
            eval(out)
            return out
        }

        let dense = run(gate: Int.max)
        let sparse = run(gate: 1)

        XCTAssertEqual(dense.shape, sparse.shape)
        let maxDiff = abs(dense - sparse).max().item(Float.self)
        XCTAssertLessThanOrEqual(
            maxDiff, Self.parityTolerance,
            "sparseGatherAttention diverges from dense top-k-masked prefill")
    }

    /// Model-level end-to-end parity: full GLM5MoeDsaModel (embeddings → decoder
    /// layers incl. MoE → logits) over multi-chunk prefill, dense vs sparse. Covers
    /// the internal mask creation (createBoolMask), cache offsets across chunks, and
    /// the layer stack — everything the server-side path exercises below the
    /// generation loop.
    func testModelLevelDenseSparseLogitParity() throws {
        var config = try tinyConfig()
        config.numHiddenLayers = 2

        MLXRandom.seed(99)
        let model = GLM5MoeDsaModel(config)

        // Inject weights for the scalar-placeholder MultiLinear params (see above).
        let h = config.numAttentionHeads
        let scale: Float = 0.05
        var inject = [String: MLXArray]()
        for l in 0 ..< config.numHiddenLayers {
            inject["model.layers.\(l).self_attn.embed_q.weight"] =
                MLXRandom.normal([h, config.kvLoraRank, config.qkNopeHeadDim]) * scale
            inject["model.layers.\(l).self_attn.unembed_out.weight"] =
                MLXRandom.normal([h, config.vHeadDim, config.kvLoraRank]) * scale
        }
        try model.update(parameters: ModuleParameters.unflattened(inject), verify: [])

        let chunkLength = 200
        let numChunks = 3
        MLXRandom.seed(5)
        let tokens = MLXRandom.randInt(0 ..< config.vocabSize, [1, chunkLength * numChunks])

        func run(gate: Int) -> MLXArray {
            let savedGate = GLM5MoeDsaAttention.sparsePrefillMinContext
            GLM5MoeDsaAttention.sparsePrefillMinContext = gate
            defer { GLM5MoeDsaAttention.sparsePrefillMinContext = savedGate }

            let cache = model.newCache(parameters: nil)
            var logits: MLXArray!
            for c in 0 ..< numChunks {
                let chunk = tokens[0..., (c * chunkLength) ..< ((c + 1) * chunkLength)]
                logits = model(chunk, cache: cache)
            }
            eval(logits!)
            return logits
        }

        let dense = run(gate: Int.max)
        let sparse = run(gate: 64)  // chunks 2 and 3 (totalSeq 400, 600) go sparse

        XCTAssertEqual(dense.shape, sparse.shape)
        let maxDiff = abs(dense - sparse).max().item(Float.self)
        XCTAssertLessThanOrEqual(
            maxDiff, Self.parityTolerance,
            "model-level logits diverge between dense and sparse prefill")
    }
}
