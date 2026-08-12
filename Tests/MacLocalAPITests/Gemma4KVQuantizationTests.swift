import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXLLM

@testable import AFMKitMLX

@Suite(.serialized)
struct Gemma4KVQuantizationTests {
    static let modelDirectory: URL = {
        let root = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]
            ?? "/Volumes/edata/models/vesta-test-cache"
        return URL(fileURLWithPath: root)
            .appendingPathComponent("mlx-community/gemma-4-31b-it-8bit")
    }()

    init() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    @Test("Gemma 4 mixed cache supports 4-bit dynamic KV quantization")
    func mixedCacheDecodeAfterQuantization() async throws {
        guard FileManager.default.fileExists(
            atPath: Self.modelDirectory.appendingPathComponent("config.json").path)
        else {
            return
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(directory: Self.modelDirectory))

        await container.perform { context in
            guard let model = context.model as? Gemma4Model else {
                Issue.record("Expected Gemma4Model, got \(type(of: context.model))")
                return
            }

            var cache = model.newCache(parameters: nil)
            let sharedCount = cache.filter { $0 is SharedKVCache }.count
            let prefill = model(MLXArray([1, 2]).reshaped([1, 2]), cache: cache)
            MLX.eval(prefill)
            cache.forEach { MLX.eval($0.innerState()) }

            maybeQuantizeKVCache(cache: &cache, kvBits: 4)

            #expect(cache.contains { $0 is QuantizedKVCache })
            #expect(cache.filter { $0 is RotatingKVCache }.count == 50)
            #expect(cache.filter { $0 is SharedKVCache }.count == sharedCount)

            let decode = model(MLXArray([3]).reshaped([1, 1]), cache: cache)
            MLX.eval(decode)
            #expect(decode.dim(1) == 1)
        }
    }

    @Test("Gemma shared KV caches remain materialized")
    func quantizationExemptionIsHonored() {
        let shared = SharedKVCache()
        _ = shared.update(
            keys: MLXArray.zeros([1, 1, 1, 64]),
            values: MLXArray.zeros([1, 1, 1, 64]))
        var caches: [KVCache] = [shared]

        maybeQuantizeKVCache(cache: &caches, kvBits: 4)

        #expect(caches[0] is SharedKVCache)
        #expect(!(caches[0] is QuantizedKVCache))
    }
}
