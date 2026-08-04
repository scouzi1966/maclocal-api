import MLX
import XCTest
@testable import MLXLLM

final class DeepseekV4DSparkCaptureTests: XCTestCase {
    func testCaptureForwardPreservesVerifierOutputAndLayerOrder() {
        var config = DeepseekV4Configuration()
        config.vocabSize = 32
        config.hiddenSize = 8
        config.numHiddenLayers = 2
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
        config.hcSinkhornIters = 2
        config.compressRatios = [0, 0]
        config.activationQATEnabled = false

        let model = DeepseekV4ModelInner(config: config)
        let tokens = MLXArray([1, 2]).reshaped(1, 2)
        let ordinary = model(tokens, cache: nil)
        let captured = model.forwardCapturingHiddenStates(
            tokens,
            cache: nil,
            layerIds: [1, 0])

        MLX.eval(ordinary, captured.hidden, captured.captured)
        XCTAssertEqual(captured.hidden.shape, [1, 2, 8])
        XCTAssertEqual(captured.captured.shape, [1, 2, 16])
        XCTAssertTrue(MLX.allClose(ordinary, captured.hidden).item(Bool.self))
    }
}
