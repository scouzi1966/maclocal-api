import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MacLocalAPI

struct TurboQuantCacheTests {
    init() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    private func makeFilledSimpleCache(tokens: Int = 6) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let _ = cache.update(
            keys: MLXArray.ones([1, 2, tokens, 64]),
            values: MLXArray.ones([1, 2, tokens, 64]) * 2
        )
        return cache
    }

    @Test("GenerateParameters auto-enable TurboQuant for fractional bit-widths")
    func generateParametersAutoEnableTurboQuant() {
        let parameters = GenerateParameters(
            maxTokens: 16,
            kvBits: 3.5,
            kvQuantScheme: .uniform,
            turboQuantVariant: .turboQuant35,
            turboQuantMetadataPath: "/tmp/turboquant.json"
        )

        #expect(parameters.usesTurboQuantKVCache)
        #expect(!parameters.usesDynamicKVQuantization)
        #expect(parameters.turboQuantConfiguration?.bits == 3.5)
        #expect(parameters.turboQuantConfiguration?.variant == .turboQuant35)
        #expect(parameters.turboQuantConfiguration?.metadataPath == "/tmp/turboquant.json")
    }

    @Test("GenerateParameters keep uniform integer KV quantization on legacy path")
    func generateParametersKeepUniformQuantization() {
        let parameters = GenerateParameters(
            kvBits: 4.0,
            kvQuantScheme: .uniform
        )

        #expect(!parameters.usesTurboQuantKVCache)
        #expect(parameters.usesDynamicKVQuantization)
        #expect(parameters.uniformKVBits == 4)
    }

    @Test("Attention cache factory keeps sliding-window layers on rotating cache")
    func attentionCacheFactoryRespectsSlidingWindow() {
        let turboParameters = GenerateParameters(
            kvBits: 4.0,
            kvQuantScheme: .turboQuant,
            turboQuantVariant: .turboQuant25
        )

        let fullAttention = makeAttentionKVCache(parameters: turboParameters)
        #expect(fullAttention is TurboQuantKVCache)

        let slidingWindow = makeAttentionKVCache(parameters: turboParameters, maxSize: 4096, keep: 0)
        #expect(slidingWindow is RotatingKVCache)
    }

    @Test("KV cache replacement follows TurboQuant runtime rules")
    func turboQuantReplacementRules() {
        let nested = CacheList(caches: [makeFilledSimpleCache(), MambaCache()])
        var caches: [KVCache] = [
            makeFilledSimpleCache(),
            RotatingKVCache(maxSize: 128, keep: 4),
            nested,
            makeFilledSimpleCache(),
        ]

        maybeQuantizeKVCache(
            cache: &caches,
            kvBits: 3.5,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            kvQuantScheme: .uniform
        )

        #expect(caches[0] is TurboQuantKVCache)
        #expect(caches[1] is RotatingKVCache)
        let nestedCache = caches[2] as! CacheList
        #expect(nestedCache[0] is TurboQuantKVCache)
        #expect(nestedCache[1] is MambaCache)
        #expect(caches[3] is KVCacheSimple)
    }

    @Test("KV cache replacement keeps uniform quantization on integer path")
    func uniformReplacementRules() {
        var caches: [KVCache] = [makeFilledSimpleCache()]

        maybeQuantizeKVCache(
            cache: &caches,
            kvBits: 4.0,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            kvQuantScheme: .uniform
        )

        #expect(caches[0] is QuantizedKVCache)
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
                bits: 3.5,
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
        #expect(loaded.configuration.bits == 3.5)
        #expect(loaded.configuration.variant == .turboQuant35)
        #expect(loaded.configuration.metadataPath == "/tmp/turboquant.json")
        #expect(loaded.offset == 6)

        let state = loaded.state
        #expect(state.count == 2)
        #expect(state[0].shape == [1, 2, 6, 8])
        #expect(state[1].shape == [1, 2, 6, 8])
    }

    @Test("MlxCommand parses fractional KV bits and quant scheme")
    func mlxCommandParsesTurboQuantFlags() throws {
        let command = try MlxCommand.parse([
            "--kv-bits", "3.5",
            "--kv-quant-scheme", "turboquant",
        ])

        #expect(command.kvBits == 3.5)
        #expect(command.kvQuantScheme == "turboquant")
    }
}
