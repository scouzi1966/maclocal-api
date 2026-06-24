import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXVLM

@testable import AFMKit
@testable import AFMServer

/// P2 gate: the MTP self-speculative generator must produce output IDENTICAL to greedy
/// autoregressive decoding (every emitted token is one the trunk would have produced), while
/// using fewer trunk forwards (the speedup). This proves correctness; the speed is validated
/// separately end-to-end. temperature=0 (greedy) only here; temp>0 exact sampling is P3.
struct MTPGeneratorP2Tests {
    static let modelDir: String = {
        let root = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]
            ?? "/Volumes/Crucial4TB/models/vesta-test-cache"
        return root + "/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    }()
    init() throws { try MLXMetalLibrary.ensureAvailable(verbose: false) }

    private func tok2D(_ ids: [Int]) -> MLXArray {
        MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
    }

    /// Plain greedy AR reference using the same trunk hooks (no MTP head).
    private func greedyAR(_ model: Qwen3_5MoEVL, prompt: [Int], maxTokens: Int) -> [Int] {
        let cache = model.newCache(parameters: nil)
        let (_, l0) = model.forwardHidden(tok2D(prompt), cache: cache)
        var logits = l0[0..., (l0.dim(1) - 1)..., 0...]
        var out: [Int] = []
        while out.count < maxTokens {
            let t = MLX.argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
            out.append(t)
            let (_, nl) = model.forwardHidden(tok2D([t]), cache: cache)
            logits = nl
        }
        return out
    }

    @Test("MTP greedy output is identical to greedy AR")
    func mtpMatchesGreedyAR() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelDir + "/config.json"),
              FileManager.default.fileExists(atPath: Self.modelDir + "/mtp.safetensors") else {
            print("[P2] model/sidecar absent — skipping")
            return
        }
        let ctx = try await loadModel(directory: URL(fileURLWithPath: Self.modelDir))
        guard let model = ctx.model as? Qwen3_5MoEVL else {
            Issue.record("not Qwen3_5MoEVL"); return
        }
        let head = try model.loadMTPHead(sidecarPath: Self.modelDir + "/mtp.safetensors")

        let prompt = [760, 6511, 314, 9338, 369]   // "The capital of France is"
        let N = 32

        let ref = greedyAR(model, prompt: prompt, maxTokens: N)

        var cyclesUsed = 0
        let gen = MTPGenerator(model: model, head: head, depth: 3)
        // Count trunk cycles by tapping the generator's loop via token count vs cycles isn't
        // exposed; we instead just compare tokens and infer acceptance from the printed log.
        let mtp = gen.generate(promptIds: prompt, maxTokens: N)
        cyclesUsed = mtp.count  // placeholder; real cycle count printed inside if instrumented

        let n = Swift.min(ref.count, mtp.count)
        var firstMismatch = -1
        for i in 0..<n where ref[i] != mtp[i] { firstMismatch = i; break }
        print("[P2] AR=\(ref.prefix(12))... MTP=\(mtp.prefix(12))...  len ref=\(ref.count) mtp=\(mtp.count) firstMismatch=\(firstMismatch)")
        _ = cyclesUsed

        #expect(mtp.count == ref.count, "length mismatch MTP=\(mtp.count) AR=\(ref.count)")
        #expect(firstMismatch == -1, "MTP diverged from greedy AR at index \(firstMismatch)")
    }
}
