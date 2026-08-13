import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class MuseGlimmerModelTests: XCTestCase {
    private func configurationJSON() throws -> Data {
        let object: [String: Any] = [
            "model_type": "muse_glimmer",
            "text_config": [
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 16,
                "layer_types": ["sliding_attention", "full_attention"],
                "layer_rope_theta": [500000.0, 0.0],
                "sliding_window": 4
            ],
            "vision_config": [
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "patch_size": 2,
                "patch_temporal": 1,
                "merge_size": 1,
                "pos_emb_height": 2,
                "pos_emb_width": 2,
                "layer_types": ["window_attention", "full_attention"]
            ],
            "image_token_id": 14,
            "video_token_id": 15,
            "out_hidden_size": 8,
            "projector_hidden_size": 8
        ]
        return try JSONSerialization.data(withJSONObject: object)
    }

    private func makeModel() throws -> MuseGlimmer {
        let config = try JSONDecoder().decode(
            MuseGlimmerConfiguration.self, from: configurationJSON())
        return MuseGlimmer(config)
    }

    func testPublishedConfigurationDecodesNestedTextAndVisionFields() throws {
        let config = try JSONDecoder().decode(
            MuseGlimmerConfiguration.self, from: configurationJSON())

        XCTAssertEqual(config.modelType, "muse_glimmer")
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 2)
        XCTAssertEqual(config.textConfiguration.layerRopeTheta, [500000, 0])
        XCTAssertEqual(config.visionConfiguration.layerTypes, ["window_attention", "full_attention"])
        XCTAssertEqual(config.imageTokenId, 14)
    }

    func testPublishedProcessorConfigurationDecodesNestedImageSettings() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "processor_class": "MuseGlimmerProcessor",
            "image_processor": [
                "patch_size": 14,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "max_image_tokens": 4096,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            ],
        ])
        let config = try JSONDecoder().decode(
            MuseGlimmerProcessorConfiguration.self, from: data)

        XCTAssertEqual(config.imageProcessor.patchSize, 14)
        XCTAssertEqual(config.imageProcessor.temporalPatchSize, 2)
        XCTAssertEqual(config.imageProcessor.mergeSize, 2)
        XCTAssertEqual(config.imageProcessor.maxImageTokens, 4096)
    }

    func testPatchifyUsesRasterTokenAndTemporalChannelPatchLayout() {
        let pixels = MLXArray([Float(0), 1, 2, 3]).reshaped(1, 1, 2, 2)
        let patches = museGlimmerPatchify(
            pixels, gridH: 2, gridW: 2, patchSize: 1, temporalPatchSize: 2)
        MLX.eval(patches)

        XCTAssertEqual(patches.shape, [4, 2])
        XCTAssertEqual(
            patches.asArray(Float.self),
            [0, 0, 1, 1, 2, 2, 3, 3])
    }

    func testPixelShuffleUsesChannelMajorMergeLayout() {
        let hidden = MLXArray((0 ..< 8).map(Float.init)).reshaped(4, 2)
        let shuffled = musePixelShuffle(
            hidden, grid: [THW(1, 2, 2)], mergeSize: 2)
        MLX.eval(shuffled)

        XCTAssertEqual(shuffled.shape, [1, 8])
        XCTAssertEqual(
            shuffled.asArray(Float.self),
            [0, 2, 4, 6, 1, 3, 5, 7])
    }

    func testSanitizeRewritesCheckpointPrefixesAndDropsRotaryArtifacts() throws {
        let model = try makeModel()
        let value = MLXArray(1.0)
        let sanitized = model.sanitize(weights: [
            "model.language_model.layers.0.weight": value,
            "model.vision_tower.patch_embedder.weight": value,
            "lm_head.weight": value,
            "model.language_model.layers.0.rotary_emb.inv_freq": value
        ])

        XCTAssertNotNil(sanitized["language_model.model.layers.0.weight"])
        XCTAssertNotNil(sanitized["vision_tower.patch_embedder.weight"])
        XCTAssertNotNil(sanitized["language_model.lm_head.weight"])
        XCTAssertNil(sanitized["language_model.model.layers.0.rotary_emb.inv_freq"])
    }

    func testVLMRegistrySelectsMuseImplementationFromArchitecture() async throws {
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: configurationJSON(),
            modelType: "muse_glimmer"
        )

        XCTAssertTrue(model is MuseGlimmer)
    }

    func testCacheConstructionMatchesPerLayerAttentionKind() throws {
        let model = try makeModel()
        let cache = model.newCache(parameters: nil)

        XCTAssertEqual(cache.count, 2)
        XCTAssertTrue(cache[0] is RotatingKVCache)
        XCTAssertTrue(cache[1] is StandardKVCache)
    }

    func testTextForwardSupportsIncrementalCacheReuse() throws {
        let model = try makeModel()
        let cache = model.newCache(parameters: nil)

        let prompt = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let promptLogits = model.callAsFunction(prompt, cache: cache)
        MLX.eval(promptLogits)
        XCTAssertEqual(promptLogits.shape, [1, 3, 16])

        let nextToken = MLXArray([4])[.newAxis, .ellipsis]
        let nextLogits = model.callAsFunction(nextToken, cache: cache)
        MLX.eval(nextLogits)
        XCTAssertEqual(nextLogits.shape, [1, 1, 16])
    }

    func testVisionPromptRunsThroughProjectionAndTextPrefill() throws {
        let model = try makeModel()
        let cache = model.newCache(parameters: nil)
        let imageToken = MLXArray([14])[.newAxis, .ellipsis]
        let flattenedPatch = MLXArray(
            Array(repeating: Float(0.25), count: 3 * 2 * 2)
        ).reshaped(1, 3 * 2 * 2)
        let input = LMInput(
            text: .init(tokens: imageToken),
            image: .init(pixels: flattenedPatch, frames: [THW(1, 1, 1)])
        )

        let result = try model.prepare(input, cache: cache, windowSize: nil)
        guard case .logits(let output) = result else {
            return XCTFail("Muse vision preparation should consume the full prompt")
        }
        MLX.eval(output.logits)
        XCTAssertEqual(output.logits.shape, [1, 1, 16])
    }
}
