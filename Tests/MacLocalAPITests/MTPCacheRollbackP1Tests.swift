import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXVLM

@testable import AFMKit
@testable import AFMServer

/// P1 gate for the MTP Swift port — the make-or-break test.
///
/// Self-speculative decoding advances the trunk cache over K speculated tokens during verify,
/// then must roll back to the M<=K accepted tokens. This test proves the snapshot/restore +
/// re-forward mechanism (MTPCacheSnapshot) reconstructs the EXACT per-layer state — including
/// the 48 GatedDeltaNet recurrent states (conv + delta-rule SSM) and the 16 full-attention KV
/// caches — that a direct M-token forward would have produced.
///
/// Method:
///   reference: cache_A := prefill(prompt); forward(extra[0..<M])         // ground truth
///   rollback:  cache_B := prefill(prompt); snap=capture(cache_B);
///              forward(extra[0..<K]);  restore(snap); forward(extra[0..<M])
///   assert: cache_A and cache_B hold bit/near-identical state for every layer,
///           AND the next-token logits from each match.
///
/// Requires the model on disk; skips cleanly if absent.
struct MTPCacheRollbackP1Tests {
    static let modelDir: String = {
        let root = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]
            ?? "/Volumes/Crucial4TB/models/vesta-test-cache"
        return root + "/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    }()

    init() throws { try MLXMetalLibrary.ensureAvailable(verbose: false) }

    @Test("MTP cache snapshot/restore reconstructs exact trunk state after rollback")
    func rollbackReconstructsState() async throws {
        guard FileManager.default.fileExists(atPath: Self.modelDir + "/config.json") else {
            print("[P1] model absent at \(Self.modelDir) — skipping")
            return
        }

        // Load the model via the standard factory (handles config + quantization).
        let ctx = try await loadModel(directory: URL(fileURLWithPath: Self.modelDir))
        guard let model = ctx.model as? Qwen3_5MoEVL else {
            Issue.record("loaded model is not Qwen3_5MoEVL (got \(type(of: ctx.model)))")
            return
        }

        // Build a [1, n] int32 token array (MLXArray has no nested-array literal init).
        func tok2D(_ ids: [Int]) -> MLXArray {
            MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
        }

        // A fixed prompt + a fixed continuation we'll "speculate".
        let prompt = tok2D([760, 6511, 314, 9338, 369])               // "The capital of France is"
        let extra = [11751, 13, 374, 264, 3283]                        // " Paris", ".", " is", " a", " city"
        let K = 4, M = 2

        func freshCacheAfter(_ trailing: Int) -> [any KVCache] {
            let cache = model.newCache(parameters: nil)
            _ = model.forwardHidden(prompt, cache: cache)              // prefill
            if trailing > 0 {
                _ = model.forwardHidden(tok2D(Array(extra[0..<trailing])), cache: cache)
            }
            return cache
        }

        // Reference: prefill + M tokens, directly.
        let refCache = freshCacheAfter(M)

        // Rollback path: prefill, snapshot, advance K, restore, advance M.
        let rbCache = model.newCache(parameters: nil)
        _ = model.forwardHidden(prompt, cache: rbCache)
        let snap = MTPCacheSnapshot.capture(rbCache)
        _ = model.forwardHidden(tok2D(Array(extra[0..<K])), cache: rbCache)   // verify K
        MTPCacheSnapshot.restore(snap, into: rbCache)                         // roll back
        _ = model.forwardHidden(tok2D(Array(extra[0..<M])), cache: rbCache)   // accept M

        // Compare per-layer state arrays.
        var maxDiff: Float = 0
        var worstLayer = -1
        for (i, (a, b)) in zip(refCache, rbCache).enumerated() {
            let sa = a.state, sb = b.state
            #expect(sa.count == sb.count, "layer \(i) state count \(sa.count) vs \(sb.count)")
            #expect(a.offset == b.offset, "layer \(i) offset \(a.offset) vs \(b.offset)")
            for (x, y) in zip(sa, sb) {
                let d = MLX.abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self)
                if d > maxDiff { maxDiff = d; worstLayer = i }
            }
        }
        print(String(format: "[P1] per-layer state max|Δ|=%.6f (worst layer %d of %d)",
                     maxDiff, worstLayer, refCache.count))

        // Next-token logits must agree too (the thing that actually matters downstream).
        let (_, refLogits) = model.forwardHidden(tok2D([extra[M]]), cache: refCache)
        let (_, rbLogits) = model.forwardHidden(tok2D([extra[M]]), cache: rbCache)
        let refArg = MLX.argMax(refLogits[0, -1, 0...]).item(Int.self)
        let rbArg = MLX.argMax(rbLogits[0, -1, 0...]).item(Int.self)
        let logitDiff = MLX.abs(refLogits.asType(.float32) - rbLogits.asType(.float32))
            .max().item(Float.self)
        print("[P1] next-token argmax ref=\(refArg) rb=\(rbArg)  logit max|Δ|=\(logitDiff)")

        // GDN recurrent state is fp32 and the re-forward is deterministic, so this should be
        // exact or within tiny fp noise. Gate generously but meaningfully.
        #expect(maxDiff < 1e-2, "rolled-back state diverges (max|Δ|=\(maxDiff)) — GDN rollback broken")
        #expect(refArg == rbArg, "next-token argmax differs after rollback (ref \(refArg) vs rb \(rbArg))")
    }
}
