import XCTest
@testable import MLXLLM
@testable import AFMKitMLX

final class DeepseekV4DSparkCapabilityTests: XCTestCase {
    func testRuntimeSwitchDefaultsOffAndRequiresExplicitOnValue() {
        XCTAssertFalse(afmDSparkEnabled(environment: [:]))
        XCTAssertTrue(afmDSparkEnabled(environment: ["AFM_DSPARK": "1"]))
        XCTAssertTrue(afmDSparkEnabled(environment: ["AFM_DSPARK": "true"]))
        XCTAssertTrue(afmDSparkEnabled(environment: ["AFM_DSPARK": "on"]))
        XCTAssertTrue(afmDSparkEnabled(environment: ["AFM_DSPARK": " yes "]))

        for value in ["0", "false", "FALSE", "off", " no ", "maybe", ""] {
            XCTAssertFalse(afmDSparkEnabled(environment: ["AFM_DSPARK": value]))
        }
    }

    func testDetectsEmbeddedDSparkFromCheckpointMetadata() throws {
        let json = """
        {
          "vocab_size": 129280,
          "num_hidden_layers": 43,
          "compress_ratios": [0, 4, 128, 0, 4, 128, 0, 4, 128, 0,
                              4, 128, 0, 4, 128, 0, 4, 128, 0, 4,
                              128, 0, 4, 128, 0, 4, 128, 0, 4, 128,
                              0, 4, 128, 0, 4, 128, 0, 4, 128, 0,
                              4, 128, 0, 0, 0, 0],
          "num_nextn_predict_layers": 1,
          "dspark_block_size": 5,
          "dspark_noise_token_id": 128799,
          "dspark_target_layer_ids": [40, 41, 42],
          "dspark_markov_rank": 256
        }
        """

        let config = try JSONDecoder().decode(
            DeepseekV4Configuration.self,
            from: Data(json.utf8)
        )

        XCTAssertEqual(config.dsparkStageCount, 3)
        XCTAssertEqual(config.dsparkBlockSize, 5)
        XCTAssertEqual(config.dsparkTargetLayerIds, [40, 41, 42])
        XCTAssertTrue(config.hasEmbeddedDSpark)
    }

    func testRejectsIncompleteOrInvalidDSparkMetadata() throws {
        var config = DeepseekV4Configuration()
        config.compressRatios = Array(repeating: 0, count: 46)
        config.dsparkBlockSize = 5
        config.dsparkNoiseTokenId = 128_799
        config.dsparkTargetLayerIds = [40, 41, 42]
        config.dsparkMarkovRank = 256
        XCTAssertTrue(config.hasEmbeddedDSpark)

        config.dsparkNoiseTokenId = config.vocabSize
        XCTAssertFalse(config.hasEmbeddedDSpark)

        config.dsparkNoiseTokenId = 128_799
        config.compressRatios = Array(repeating: 0, count: config.numHiddenLayers)
        XCTAssertFalse(config.hasEmbeddedDSpark)
    }
}
