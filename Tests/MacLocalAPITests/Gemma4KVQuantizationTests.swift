import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXLLM
@testable import MLXVLM

@testable import AFMKitMLX

@Suite(.serialized)
struct Gemma4KVQuantizationTests {
    static let modelDirectory: URL = {
        let root = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]
            ?? "/Volumes/edata/models/vesta-test-cache"
        return URL(fileURLWithPath: root)
            .appendingPathComponent("mlx-community/gemma-4-e4b-it-4bit")
    }()

    static let modelAvailable = FileManager.default.fileExists(
        atPath: modelDirectory.appendingPathComponent("config.json").path)

    init() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    @Test(
        "Gemma 4 mixed cache supports 4-bit dynamic KV quantization",
        .enabled(if: Self.modelAvailable, "Gemma 4 E4B fixture is not installed"))
    func mixedCacheDecodeAfterQuantization() async throws {
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(directory: Self.modelDirectory))

        await container.perform { context in
            guard let model = context.model as? Gemma4Model else {
                Issue.record("Expected Gemma4Model, got \(type(of: context.model))")
                return
            }

            let config = model.configuration.textConfig
            let expectedRotatingCount = config.layerTypes.filter {
                $0 == "sliding_attention"
            }.count
            let sharedBoundary = min(config.firstKvSharedLayerIdx, config.numHiddenLayers)
            let prefixLayerTypes = Array(config.layerTypes[..<sharedBoundary])
            let expectedSharedCount = prefixLayerTypes.lastIndex(of: "full_attention") == nil
                ? 0 : 1

            let baselineCache = model.newCache(parameters: nil)
            var quantizedCache = model.newCache(parameters: nil)
            #expect(quantizedCache.filter { $0 is RotatingKVCache }.count == expectedRotatingCount)
            #expect(quantizedCache.filter { $0 is SharedKVCache }.count == expectedSharedCount)
            #expect(expectedSharedCount > 0)

            let prompt = MLXArray([1, 2, 3]).reshaped([1, 3])
            var baselineLogits = model(prompt, cache: baselineCache)
            let quantizedPrefill = model(prompt, cache: quantizedCache)
            MLX.eval(baselineLogits, quantizedPrefill)
            baselineCache.forEach { MLX.eval($0.innerState()) }
            quantizedCache.forEach { MLX.eval($0.innerState()) }

            maybeQuantizeKVCache(cache: &quantizedCache, kvBits: 4)

            #expect(quantizedCache.contains { $0 is QuantizedKVCache })
            #expect(quantizedCache.filter { $0 is RotatingKVCache }.count == expectedRotatingCount)
            #expect(quantizedCache.filter { $0 is SharedKVCache }.count == expectedSharedCount)

            for _ in 0..<3 {
                let token = MLX.argMax(baselineLogits[0, -1, 0...], axis: -1)
                    .item(Int.self)
                let input = MLXArray([token]).reshaped([1, 1])
                baselineLogits = model(input, cache: baselineCache)
                let quantizedLogits = model(input, cache: quantizedCache)
                MLX.eval(baselineLogits, quantizedLogits)

                let baselineToken = MLX.argMax(
                    baselineLogits[0, -1, 0...], axis: -1).item(Int.self)
                let quantizedToken = MLX.argMax(
                    quantizedLogits[0, -1, 0...], axis: -1).item(Int.self)
                #expect(quantizedToken == baselineToken)
            }

            let expectedOffset = 6
            #expect(baselineCache.filter { $0 is SharedKVCache }.allSatisfy {
                $0.offset == expectedOffset
            })
            #expect(quantizedCache.filter { $0 is SharedKVCache }.allSatisfy {
                $0.offset == expectedOffset
            })
        }
    }

    @Test("Gemma shared KV caches preserve history and remain materialized")
    func quantizationExemptionIsHonored() throws {
        let shared = SharedKVCache()
        let firstKeys = MLXArray(Array(0..<4)).asType(.float32).reshaped([1, 1, 2, 2])
        let firstValues = firstKeys + 10
        _ = shared.update(keys: firstKeys, values: firstValues)
        let nextKeys = MLXArray([4, 5]).asType(.float32).reshaped([1, 1, 1, 2])
        let nextValues = nextKeys + 10
        let (allKeys, allValues) = shared.update(keys: nextKeys, values: nextValues)
        MLX.eval(allKeys, allValues)

        #expect(allKeys.shape == [1, 1, 3, 2])
        #expect(allValues.shape == [1, 1, 3, 2])
        #expect(allKeys.asArray(Float.self) == [0, 1, 2, 3, 4, 5])
        #expect(allValues.asArray(Float.self) == [10, 11, 12, 13, 14, 15])

        var caches: [KVCache] = [shared]
        maybeQuantizeKVCache(cache: &caches, kvBits: 4)

        #expect(caches[0] is SharedKVCache)
        #expect(!(caches[0] is QuantizedKVCache))

        let cacheURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma-shared-kv-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: cacheURL) }

        try savePromptCache(url: cacheURL, cache: caches)
        var (restored, _) = try loadPromptCache(url: cacheURL)

        #expect(restored.count == 1)
        #expect(restored[0] is SharedKVCache)
        #expect(restored[0].offset == 3)

        maybeQuantizeKVCache(cache: &restored, kvBits: 4)
        #expect(restored[0] is SharedKVCache)
        #expect(!(restored[0] is QuantizedKVCache))

        let finalKeys = MLXArray([6, 7]).asType(.float32).reshaped([1, 1, 1, 2])
        let finalValues = finalKeys + 10
        let (roundTrippedKeys, roundTrippedValues) = restored[0].update(
            keys: finalKeys, values: finalValues)
        MLX.eval(roundTrippedKeys, roundTrippedValues)
        #expect(roundTrippedKeys.asArray(Float.self) == [0, 1, 2, 3, 4, 5, 6, 7])
        #expect(roundTrippedValues.asArray(Float.self) == [10, 11, 12, 13, 14, 15, 16, 17])
    }

    @Test("Gemma VLM preserves unequal per-sequence RoPE offsets")
    func vlmUsesPerSequenceOffsets() {
        let cache = BatchKVCacheSimple(batchSize: 2, leftPadding: [0, 2])
        let keys = MLXArray.ones([2, 1, 3, 2])
        _ = cache.update(keys: keys, values: keys)

        let offsets = gemma4VLRopeOffsets(cache: cache, batchSize: 2)
        #expect(offsets.scalar == 3)
        #expect(offsets.batched != nil)
        if let batched = offsets.batched {
            MLX.eval(batched)
            #expect(batched.asArray(Int32.self) == [3, 1])
        }

        let uniform = BatchKVCacheSimple(batchSize: 2, leftPadding: [0, 0])
        _ = uniform.update(keys: keys, values: keys)
        let uniformOffsets = gemma4VLRopeOffsets(cache: uniform, batchSize: 2)
        #expect(uniformOffsets.scalar == 3)
        #expect(uniformOffsets.batched == nil)
    }

    @Test("Gemma VLM shared layers inherit unequal reference offsets")
    func vlmSharedCacheInheritsReferenceOffsets() throws {
        let reference = BatchKVCacheSimple(batchSize: 2, leftPadding: [0, 2])
        let keys = MLXArray.ones([2, 1, 3, 2])
        _ = reference.update(keys: keys, values: keys)

        let sharedLayer = BatchKVCacheSimple(batchSize: 2, leftPadding: [0, 0])
        gemma4VLSyncSharedCache(
            sharedLayer,
            scalarOffset: reference.offset,
            offsets: reference.offsetArray,
            allOffsetsEqual: reference.allOffsetsEqual)

        #expect(sharedLayer.offset == reference.offset)
        #expect(!sharedLayer.allOffsetsEqual)
        let synchronized = try #require(sharedLayer.offsetArray)
        MLX.eval(synchronized)
        #expect(synchronized.asArray(Int32.self) == [3, 1])

        let ropeOffsets = gemma4VLRopeOffsets(cache: sharedLayer, batchSize: 2)
        #expect(ropeOffsets.scalar == 3)
        let batched = try #require(ropeOffsets.batched)
        MLX.eval(batched)
        #expect(batched.asArray(Int32.self) == [3, 1])
    }
}
