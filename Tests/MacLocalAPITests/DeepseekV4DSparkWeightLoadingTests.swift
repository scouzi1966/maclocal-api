import MLX
import MLXLMCommon
import XCTest
@testable import MLXLLM

final class DeepseekV4DSparkWeightLoadingTests: XCTestCase {
    func testOfficialPackedWeightsNormalizeWithoutRequantization() {
        let model = DeepseekV4Model(makeConfig())
        let officialMXFP8 = MLXArray(
            [UInt8](repeating: 0x11, count: 128 * 128)
        ).reshaped([128, 128])
        let officialMXFP4 = MLXArray(
            [Int8](repeating: 0x22, count: 64)
        ).reshaped([4, 16])
        let expertScales = MLXArray(
            [UInt8](repeating: 0x7f, count: 4)
        ).reshaped([4, 1])
        let attentionScales = MLXArray([UInt8(0x7f)]).reshaped([1, 1])

        let sanitized = model.sanitize(weights: [
            "layers.0.attn.wq_a.weight": officialMXFP8,
            "layers.0.attn.wq_a.scale": attentionScales,
            "layers.0.ffn.experts.0.w1.weight": officialMXFP4,
            "layers.0.ffn.experts.0.w1.scale": expertScales,
            "layers.0.ffn.experts.1.w1.weight": officialMXFP4,
            "layers.0.ffn.experts.1.w1.scale": expertScales,
        ])

        let attentionWeight = sanitized["model.layers.0.self_attn.wq_a.weight"]
        XCTAssertEqual(attentionWeight?.dtype, .uint32)
        XCTAssertEqual(attentionWeight?.shape, [128, 32])
        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.wq_a.scales"]?.dtype,
            .uint8)
        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.wq_a.scales"]?.shape,
            [128, 4])

        let expertWeight = sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
        XCTAssertEqual(expertWeight?.dtype, .uint32)
        XCTAssertEqual(expertWeight?.shape, [2, 4, 4])
        XCTAssertEqual(
            sanitized["model.layers.0.mlp.switch_mlp.gate_proj.scales"]?.shape,
            [2, 4, 1])
    }

    func testOfficialPackedQuantizationModeUsesShapeRatio() {
        let mxfp4 = inferOfficialBlockQuantization(
            weightShape: [256, 4], scaleShape: [256, 1])
        XCTAssertEqual(mxfp4?.groupSize, 32)
        XCTAssertEqual(mxfp4?.bits, 4)
        XCTAssertEqual(mxfp4?.mode, .mxfp4)

        let mxfp8 = inferOfficialBlockQuantization(
            weightShape: [256, 8], scaleShape: [256, 1])
        XCTAssertEqual(mxfp8?.groupSize, 32)
        XCTAssertEqual(mxfp8?.bits, 8)
        XCTAssertEqual(mxfp8?.mode, .mxfp8)

        XCTAssertNil(inferOfficialBlockQuantization(
            weightShape: [256, 3], scaleShape: [256, 1]))
    }

    func testEmbeddedDrafterTopologyMatchesSanitizedCheckpointPaths() {
        let config = makeConfig()
        let model = DeepseekV4Model(config)
        let parameterKeys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(parameterKeys.contains("mtp.0.main_proj.weight"))
        XCTAssertTrue(parameterKeys.contains("mtp.0.attn.wq_a.weight"))
        XCTAssertTrue(parameterKeys.contains("mtp.1.markov_head.markov_w1.weight"))
        XCTAssertTrue(parameterKeys.contains("mtp.1.confidence_head.proj.weight"))
        XCTAssertTrue(parameterKeys.contains("mtp.1.hc_head_fn"))
    }

    func testSanitizeRetainsDrafterAndStacksItsExperts() {
        let config = makeConfig()
        let model = DeepseekV4Model(config)
        let weights: [String: MLXArray] = [
            "mtp.0.main_proj.weight": zeros([8, 8]),
            "mtp.0.hc_attn_fn": zeros([8, 16]),
            "mtp.0.attn.wq_a.weight": zeros([4, 8]),
            "mtp.0.ffn.shared_experts.w1.weight": zeros([4, 8]),
            "mtp.0.ffn.experts.0.w1.weight": zeros([4, 8]),
            "mtp.0.ffn.experts.1.w1.weight": ones([4, 8]),
            "mtp.1.markov_head.markov_w1.weight": zeros([32, 4]),
            "mtp.1.confidence_head.proj.weight": zeros([1, 12]),
            "mtp.1.hc_head_fn": zeros([2, 16]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["mtp.0.main_proj.weight"])
        XCTAssertNotNil(sanitized["mtp.0.attn_hc.fn"])
        XCTAssertNotNil(sanitized["mtp.0.attn.wq_a.weight"])
        XCTAssertNotNil(sanitized["mtp.0.ffn.shared_experts.gate_proj.weight"])
        XCTAssertEqual(
            sanitized["mtp.0.ffn.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8])
        XCTAssertNil(sanitized["mtp.0.ffn.experts.0.gate_proj.weight"])
        XCTAssertNotNil(sanitized["mtp.1.markov_head.markov_w1.weight"])
        XCTAssertNotNil(sanitized["mtp.1.confidence_head.proj.weight"])
        XCTAssertNotNil(sanitized["mtp.1.hc_head_fn"])
    }

    func testDrafterPrefillAndProposalAdvanceOnlyTargetCacheRows() throws {
        let config = makeConfig()
        let model = DeepseekV4Model(config)
        let cache = model.newDSparkCache()
        let anchor = MLXArray([Int32(3)])

        XCTAssertTrue(model.prefillDSpark(
            anchorTokenIds: anchor,
            capturedHidden: zeros([1, 2, 8]),
            cache: cache))
        XCTAssertEqual(cache.map(\.offset), [2, 2])

        let proposal = try XCTUnwrap(model.proposeDSpark(
            anchorTokenIds: anchor,
            capturedHidden: zeros([1, 1, 8]),
            cache: cache))
        MLX.eval(proposal.tokenIds, proposal.logits, proposal.confidence)

        XCTAssertEqual(proposal.tokenIds.shape, [1, 4])
        XCTAssertEqual(proposal.logits.shape, [1, 3, 32])
        XCTAssertEqual(proposal.confidence.shape, [1, 3])
        XCTAssertEqual(cache.map(\.offset), [3, 3])
    }

    func testSpeculativeGeneratorMatchesGreedyAutoregressiveTokens() {
        let model = DeepseekV4Model(makeConfig())
        let prompt = [1, 7, 4, 9]
        let maxTokens = 12

        let expected = greedyTokens(
            model: model, prompt: prompt, maxTokens: maxTokens)
        let actual = DeepseekV4DSparkGenerator(
            model: model, draftLimit: 2
        ).generate(promptIds: prompt, maxTokens: maxTokens)

        XCTAssertEqual(actual, expected)
    }

    func testSpeculativeGeneratorMatchesGreedyWithCompressedAttention() {
        let config = makeThreeLayerConfig(compressRatios: [0, 2, 0])
        let model = DeepseekV4Model(config)
        let prompt = [1, 7, 4, 9, 2]
        let maxTokens = 16

        let expected = greedyTokens(
            model: model, prompt: prompt, maxTokens: maxTokens)
        let actual = DeepseekV4DSparkGenerator(
            model: model, draftLimit: 2
        ).generate(promptIds: prompt, maxTokens: maxTokens)

        XCTAssertEqual(actual, expected)
    }

    func testSpeculativeGeneratorMatchesGreedyAcrossSlidingLayers() {
        let config = makeThreeLayerConfig(compressRatios: [0, 0, 0])
        let model = DeepseekV4Model(config)
        let prompt = [1, 7, 4, 9, 2]
        let maxTokens = 16

        let expected = greedyTokens(
            model: model, prompt: prompt, maxTokens: maxTokens)
        let actual = DeepseekV4DSparkGenerator(
            model: model, draftLimit: 2
        ).generate(promptIds: prompt, maxTokens: maxTokens)

        XCTAssertEqual(actual, expected)
    }

    func testCompressedVerifierBatchMatchesSequentialRows() throws {
        let model = DeepseekV4Model(
            makeThreeLayerConfig(compressRatios: [0, 2, 0]))
        let prompt = [1, 7, 4, 9, 2]
        let promptArray = MLXArray(prompt.map(Int32.init)).reshaped([1, prompt.count])
        func primedCache() throws -> [KVCache] {
            let cache = model.newCache(parameters: nil)
            let prefill = try XCTUnwrap(model.forwardDSparkVerifier(
                promptArray, cache: cache))
            let first = argMax(
                prefill.logits[0, -1, 0...], axis: -1).item(Int.self)
            _ = try XCTUnwrap(model.forwardDSparkVerifier(
                MLXArray([Int32(first)]).reshaped([1, 1]), cache: cache))
            return cache
        }

        let verifierIds = [13, 9, 19]
        let batchedCache = try primedCache()
        let sequentialCache = try primedCache()
        let batchedInput = MLXArray(verifierIds.map(Int32.init)).reshaped([1, verifierIds.count])
        let batched = try XCTUnwrap(model.forwardDSparkVerifier(
            batchedInput, cache: batchedCache))
        let batchedTargets = argMax(
            batched.logits[0, 0..., 0...], axis: -1).asArray(Int32.self)

        var sequentialTargets: [Int32] = []
        for id in verifierIds {
            let row = try XCTUnwrap(model.forwardDSparkVerifier(
                MLXArray([Int32(id)]).reshaped([1, 1]), cache: sequentialCache))
            sequentialTargets.append(Int32(
                argMax(row.logits[0, -1, 0...], axis: -1).item(Int.self)))
        }

        XCTAssertEqual(batchedTargets, sequentialTargets)
    }

    func testSpeculativeRollbackRestoresCompressedPoolAndBuffers() {
        let cache = DeepseekV4Cache(
            slidingWindow: 8,
            compressRatio: 2,
            poolQuantizationEnabled: false)
        let initialKeys = ones([1, 1, 3, 4])
        _ = cache.update(keys: initialKeys, values: initialKeys)
        cache.setHybridPool(
            branch: .compressor, value: ones([1, 2, 4]))
        cache.setHybridBuffers(
            branch: .compressor,
            kv: ones([1, 1, 4]),
            gate: ones([1, 1, 1]))
        let snapshot = cache.captureSpeculativeSnapshot()

        let verifierKeys = ones([1, 1, 3, 4]) * 2
        _ = cache.update(keys: verifierKeys, values: verifierKeys)
        _ = cache.appendPooled(.compressor, value: ones([1, 1, 4]) * 3)
        cache.setHybridBuffers(
            branch: .compressor,
            kv: ones([1, 2, 4]) * 4,
            gate: ones([1, 2, 1]) * 4)
        cache.rollbackSpeculative(rejected: 2, to: snapshot)

        XCTAssertEqual(cache.offset, 4)
        XCTAssertEqual(cache.hybridPool(branch: .compressor)?.shape, [1, 2, 4])
        XCTAssertEqual(
            cache.hybridBuffers(branch: .compressor).kv?.shape,
            [1, 1, 4])
    }

    private func greedyTokens(
        model: DeepseekV4Model,
        prompt: [Int],
        maxTokens: Int
    ) -> [Int] {
        let cache = model.newCache(parameters: nil)
        func array(_ ids: [Int]) -> MLXArray {
            MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
        }
        func next(_ logits: MLXArray) -> Int {
            argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
        }

        var pending = next(model(array(prompt), cache: cache))
        var output = [pending]
        while output.count < maxTokens {
            pending = next(model(array([pending]), cache: cache))
            output.append(pending)
        }
        return output
    }

    private func makeConfig() -> DeepseekV4Configuration {
        var config = DeepseekV4Configuration()
        config.vocabSize = 32
        config.hiddenSize = 8
        config.numHiddenLayers = 1
        config.numAttentionHeads = 2
        config.numKeyValueHeads = 1
        config.headDim = 4
        config.qkRopeHeadDim = 2
        config.qLoraRank = 4
        config.oGroups = 2
        config.oLoraRank = 4
        config.nRoutedExperts = 2
        config.nSharedExperts = 1
        config.numExpertsPerTok = 1
        config.moeIntermediateSize = 4
        config.numHashLayers = 0
        config.hcMult = 2
        config.compressRatios = [0, 0, 0]
        config.dsparkBlockSize = 3
        config.dsparkNoiseTokenId = 31
        config.dsparkTargetLayerIds = [0]
        config.dsparkMarkovRank = 4
        config.activationQATEnabled = false
        return config
    }

    private func makeThreeLayerConfig(
        compressRatios: [Int]
    ) -> DeepseekV4Configuration {
        precondition(compressRatios.count == 3)
        var config = makeConfig()
        config.numHiddenLayers = 3
        config.compressRatios = compressRatios + [0, 0, 0]
        config.dsparkTargetLayerIds = [0, 1, 2]
        return config
    }
}
