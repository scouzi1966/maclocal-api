import MLX
import MLXLLM
@testable import AFMKitMLX
import XCTest

final class DeepseekV4NativeCheckpointTests: XCTestCase {
    func testNativeCheckpointMarkerDecodes() throws {
        let data = Data(#"{"model_type":"deepseek_v4","afm_native_checkpoint":true}"#.utf8)
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        XCTAssertTrue(config.afmNativeCheckpoint)
    }

    func testNativeCheckpointMarkerDefaultsToFalse() throws {
        let data = Data(#"{"model_type":"deepseek_v4"}"#.utf8)
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        XCTAssertFalse(config.afmNativeCheckpoint)
    }

    func testNativeCheckpointBypassesSanitizer() {
        var config = DeepseekV4Configuration()
        config.afmNativeCheckpoint = true
        config.vocabSize = 16
        config.hiddenSize = 8
        config.numHiddenLayers = 0
        config.numAttentionHeads = 1
        config.numKeyValueHeads = 1
        config.headDim = 8
        config.qkRopeHeadDim = 2
        config.qLoraRank = 4
        config.oGroups = 1
        config.oLoraRank = 4
        config.nRoutedExperts = 2
        config.numExpertsPerTok = 1
        config.moeIntermediateSize = 4

        let model = DeepseekV4Model(config)
        let input = ["model.already_normalized": MLXArray([Float(1)])]
        let output = model.sanitize(weights: input)

        XCTAssertEqual(Set(output.keys), Set(input.keys))
        XCTAssertEqual(output["model.already_normalized"]?.shape, [1])
    }

    func testDwarfstarQ8ProfileSelectsOnlyAdvertisedRoles() {
        XCTAssertTrue(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control("lm_head"))
        XCTAssertTrue(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.self_attn.wq_a"))
        XCTAssertTrue(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.mlp.shared_experts.gate_proj"))

        XCTAssertFalse(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.mlp.switch_mlp.gate_proj"))
        XCTAssertFalse(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.self_attn.compressor.kv_a_proj"))
        XCTAssertFalse(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.layers.4.self_attn.indexer.weight"))
        XCTAssertFalse(DeepseekV4CheckpointConverter.usesDwarfstarQ8Control(
            "model.embed_tokens"))
    }

    func testAlignedMXFP4SuperblocksPrefixScalesAndPreserveWords() throws {
        let words = MLXArray((0..<64).map(UInt32.init)).reshaped([1, 64])
        let scales = MLXArray((0..<16).map(UInt8.init)).reshaped([1, 16])

        let aligned = try DeepseekV4CheckpointConverter.alignedMXFP4Superblocks(
            weight: words, scales: scales)
        MLX.eval(aligned)

        XCTAssertEqual(aligned.shape, [1, 68])
        let values = aligned.asArray(UInt32.self)
        XCTAssertEqual(values[0...3], [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c])
        XCTAssertEqual(Array(values[4...]), (0..<64).map(UInt32.init))
    }
}
