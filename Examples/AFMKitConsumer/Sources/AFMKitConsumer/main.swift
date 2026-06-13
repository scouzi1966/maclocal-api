import AFMKit
import Foundation

// Demonstrates embedding afm as a headless Swift library via AFMKit.
//
// Run (set the model cache to avoid re-downloading):
//   MACAFM_MLX_MODEL_CACHE=/path/to/cache swift run AFMKitConsumer
//
// This file's purpose is to prove the AFMKit public API is usable from an
// external SPM package: model selection, engine config, loading, and both
// blocking and streaming generation.

@main
struct AFMKitConsumer {
    static func main() async {
        do {
            // 1. Create an engine over the MLX backend with engine-level config.
            let engine = AFMEngine(
                backend: .mlx(modelID: "mlx-community/Qwen3-4B-MLX-4bit"),
                config: EngineConfig(enablePrefixCaching: true)
            )

            // 2. Load the model (downloads on first run).
            let modelID = try await engine.load { fraction in
                if fraction > 0 { print("loading… \(Int(fraction * 100))%") }
            }
            print("loaded: \(modelID)")

            let messages = [
                Message(role: "system", content: "You are concise."),
                Message(role: "user", content: "Name three primary colors."),
            ]
            let config = GenerationConfig(temperature: 0, maxTokens: 64)

            // 3a. Blocking generation.
            let result = try await engine.respond(to: messages, config)
            print("--- response ---")
            print(result.content)
            print("(prompt: \(result.promptTokens) tok, completion: \(result.completionTokens) tok)")

            // 3b. Streaming generation.
            print("--- streamed ---")
            for try await delta in engine.streamRespond(to: messages, config) {
                print(delta, terminator: "")
            }
            print()

            // 4. AFMLanguageModel protocol — the WWDC26-shaped abstraction.
            let model: any AFMLanguageModel = engine
            print("available: \(model.isAvailable)")
        } catch {
            FileHandle.standardError.write(Data("AFMKitConsumer error: \(error)\n".utf8))
        }
    }
}
