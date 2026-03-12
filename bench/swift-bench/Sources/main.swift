import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon

// Minimal benchmark: load model, generate 512 tokens, report speed.
// This uses vanilla mlx-swift-lm with NO afm code in the path.

@main
struct Bench {
    static func main() async throws {
        let modelId = CommandLine.arguments.count > 1
            ? CommandLine.arguments[1]
            : "mlx-community/Qwen3.5-35B-A3B-4bit"
        let maxTokens = 512
        let cacheDir = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]
            ?? "/Volumes/edata/models/vesta-test-cache"

        print("Model:      \(modelId)")
        print("Max tokens: \(maxTokens)")
        print("Cache:      \(cacheDir)")
        print("")

        // Configure HF hub to use our cache
        let hub = HubApi(downloadBase: URL(fileURLWithPath: cacheDir))

        // Load model
        print("Loading model...")
        let loadStart = Date()
        let config = ModelConfiguration(id: modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            hub: hub, configuration: config
        ) { progress in
            // silent
        }
        let loadTime = Date().timeIntervalSince(loadStart)
        print("  Loaded in \(String(format: "%.1f", loadTime))s")

        // Build prompt
        let prompt = "Explain calculus concepts from limits through multivariable calculus with rigorous mathematical notation."
        nonisolated(unsafe) let userInput = UserInput(prompt: .chat([.user(prompt)]))

        // Warmup
        print("Warmup...")
        let _ = try await container.perform { (context: ModelContext) in
            let input = try await context.processor.prepare(input: userInput)
            var count = 0
            let params = GenerateParameters(maxTokens: 10, temperature: 0.0)
            for try await _ in try MLXLMCommon.generate(
                input: input, parameters: params, context: context
            ) {
                count += 1
                if count >= 10 { break }
            }
            return count
        }

        // Timed run 1: no explicit cache (default path)
        print("Generating \(maxTokens) tokens (no explicit cache)...")
        let genStart1 = Date()
        let tokenCount1 = try await container.perform { (context: ModelContext) in
            let input = try await context.processor.prepare(input: userInput)
            var count = 0
            let params = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)
            for try await piece in try MLXLMCommon.generate(
                input: input, parameters: params, context: context
            ) {
                if case .chunk(let text) = piece {
                    count += 1
                }
            }
            return count
        }
        let genTime1 = Date().timeIntervalSince(genStart1)
        let tps1 = Double(maxTokens) / genTime1
        print("  tok/s: \(String(format: "%.1f", tps1)) (no cache)")

        // Timed run 2: with explicit cache (afm-style path)
        print("Generating \(maxTokens) tokens (explicit cache, afm-style params)...")
        let genStart2 = Date()
        let tokenCount2 = try await container.perform { (context: ModelContext) in
            let input = try await context.processor.prepare(input: userInput)
            var count = 0
            let params = GenerateParameters(
                maxTokens: maxTokens,
                kvGroupSize: 64,
                quantizedKVStart: 0,
                temperature: 0.0,
                topP: 1.0,
                repetitionContextSize: 64
            )
            let cache = context.model.newCache(parameters: params)
            for try await piece in try MLXLMCommon.generate(
                input: input, cache: cache, parameters: params, context: context
            ) {
                if case .chunk(let text) = piece {
                    count += 1
                }
            }
            return count
        }
        let genTime2 = Date().timeIntervalSince(genStart2)
        let tps2 = Double(maxTokens) / genTime2
        print("  tok/s: \(String(format: "%.1f", tps2)) (explicit cache)")

        print("")
        print("Results:")
        print("  Run 1 (no cache): \(String(format: "%.1f", tps1)) tok/s (\(String(format: "%.2f", genTime1))s)")
        print("  Run 2 (cache):    \(String(format: "%.1f", tps2)) tok/s (\(String(format: "%.2f", genTime2))s)")
    }
}
