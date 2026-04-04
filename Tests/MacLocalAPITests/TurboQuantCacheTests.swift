import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MacLocalAPI

struct TurboQuantCacheTests {
    init() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    @Test("GenerateParameters exposes TurboQuant configuration")
    func generateParametersExposeTurboQuantConfiguration() {
        let parameters = GenerateParameters(
            maxTokens: 16,
            kvCacheFormat: .turboQuant,
            turboQuantVariant: .turboQuant35,
            turboQuantMetadataPath: "/tmp/turboquant.json"
        )

        #expect(parameters.usesTurboQuantKVCache)
        #expect(!parameters.usesDynamicKVQuantization)
        #expect(parameters.turboQuantConfiguration?.variant == .turboQuant35)
        #expect(parameters.turboQuantConfiguration?.metadataPath == "/tmp/turboquant.json")
    }

    @Test("Attention cache factory keeps sliding-window layers on rotating cache")
    func attentionCacheFactoryRespectsSlidingWindow() {
        let turboParameters = GenerateParameters(
            kvCacheFormat: .turboQuant,
            turboQuantVariant: .turboQuant25
        )

        let fullAttention = makeAttentionKVCache(parameters: turboParameters)
        #expect(fullAttention is TurboQuantKVCache)

        let slidingWindow = makeAttentionKVCache(parameters: turboParameters, maxSize: 4096, keep: 0)
        #expect(slidingWindow is RotatingKVCache)
    }

    @Test("TurboQuant metadata JSON round-trips")
    func turboQuantMetadataRoundTrip() throws {
        let metadata = TurboQuantMetadataArtifact(
            recipe: TurboQuantVariant.turboQuant25.rawValue,
            headSize: 128,
            modelName: "test-model",
            layers: [
                "layers.0.self_attn": TurboQuantLayerMetadata(
                    key: TurboQuantTensorMetadata(highPrecisionIndices: [[0, 1, 2, 3]]),
                    value: TurboQuantTensorMetadata(highPrecisionIndices: [[4, 5, 6, 7]])
                )
            ]
        )

        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("json")
        defer { try? FileManager.default.removeItem(at: url) }

        try saveTurboQuantMetadata(metadata, url: url)
        let decoded = try loadTurboQuantMetadata(url: url)

        #expect(decoded == metadata)
    }

    @Test("Prompt cache serialization preserves TurboQuant cache identity")
    func turboQuantPromptCacheRoundTrip() throws {
        let cache = TurboQuantKVCache(
            configuration: TurboQuantConfiguration(
                variant: .turboQuant35,
                metadataPath: "/tmp/turboquant.json"
            )
        )

        let _ = cache.update(
            keys: MLXArray.ones([1, 2, 6, 8]),
            values: MLXArray.ones([1, 2, 6, 8]) * 2
        )

        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache], metadata: ["source": "turboquant-test"])
        let (loadedCaches, userMetadata) = try loadPromptCache(url: url)

        #expect(userMetadata?["source"] == "turboquant-test")
        #expect(loadedCaches.count == 1)
        #expect(loadedCaches[0] is TurboQuantKVCache)

        let loaded = loadedCaches[0] as! TurboQuantKVCache
        #expect(loaded.configuration.variant == .turboQuant35)
        #expect(loaded.configuration.metadataPath == "/tmp/turboquant.json")
        #expect(loaded.offset == 6)

        let state = loaded.state
        #expect(state.count == 2)
        #expect(state[0].shape == [1, 2, 6, 8])
        #expect(state[1].shape == [1, 2, 6, 8])
    }
}
