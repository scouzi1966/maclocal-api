import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXLLM

@testable import AFMKit

/// P1 gate for the EAGLE3 port: the greedy draft -> verify -> accept -> KV-trim loop must produce
/// EXACTLY the same token sequence as plain greedy autoregressive decoding from the same verifier.
/// Speculative decoding is lossless under greedy sampling — every emitted token is either a draft
/// that equals the verifier's argmax or the verifier's own argmax bonus — so equality is the
/// correctness gate (acceptance rate is reported, not gated).
///
/// Single sequence, all-`KVCacheSimple` verifier cache (generation stays well under the sliding
/// window, so simple == rotating numerically and the rollback is a uniform trim). Requires the
/// dense 31B verifier + EAGLE3 drafter on disk; skips cleanly if absent.
struct Eagle3SpecLoopP1Tests {
    static let modelDir: String = {
        let root = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]
            ?? "/Volumes/Crucial4TB/models/vesta-test-cache"
        return root + "/mlx-community/gemma-4-31b-it-4bit"
    }()
    static let fixtureDir: String = {
        let here = URL(fileURLWithPath: #filePath)
        return here.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Scripts/eagle3-port/fixtures").path
    }()
    static func drafterDir() -> String? {
        let hub = (ProcessInfo.processInfo.environment["HF_HOME"]
            ?? "/Volumes/Crucial4TB/models/huggingface") + "/hub"
        let base = hub + "/models--RedHatAI--gemma-4-31B-it-speculator.eagle3/snapshots"
        guard let snaps = try? FileManager.default.contentsOfDirectory(atPath: base) else { return nil }
        for s in snaps where FileManager.default.fileExists(atPath: base + "/" + s + "/config.json") {
            return base + "/" + s
        }
        return nil
    }

    init() throws { try MLXMetalLibrary.ensureAvailable(verbose: false) }

    /// Carries the non-Sendable drafter/generator into the @Sendable model-lock closure. Safe: the
    /// model lock serializes all access and the test is single-threaded.
    final class GenBox: @unchecked Sendable {
        let gen: Gemma4Eagle3Generator
        init(_ d: Gemma4Eagle3Drafter) { gen = Gemma4Eagle3Generator(drafter: d) }
    }

    struct LoopOut: @unchecked Sendable {
        let ar: [Int]; let spec: [Int]
        let rounds: Int; let acceptedTotal: Int; let draftedTotal: Int
    }

    @Test("EAGLE3 greedy speculative loop output == greedy AR (Gemma4-31B)")
    func specEqualsGreedyAR() async throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: Self.modelDir + "/config.json"),
              let drafterDir = Self.drafterDir(),
              fm.fileExists(atPath: Self.fixtureDir + "/eagle3_ref.json") else {
            print("[E3-P1] model/drafter/fixture absent — skipping"); return
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(directory: URL(fileURLWithPath: Self.modelDir)))
        let drafter = try Gemma4Eagle3Drafter.load(directory: drafterDir)
        let box = GenBox(drafter)
        let capIds = drafter.config.captureLayerIds

        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: Self.fixtureDir + "/eagle3_ref.json"))) as! [String: Any]
        let promptIds = (meta["prompt_ids"] as! [Any]).map { Int($0 as! Int) }

        let nGen = 48
        let blockSize = 4   // bs-1 = 3 draft tokens (matches the speculator's design)

        let result: LoopOut? = try await container.perform { ctx -> LoopOut? in
            guard let model = ctx.model as? Gemma4Model else {
                Issue.record("loaded model is not Gemma4Model (\(type(of: ctx.model)))"); return nil
            }
            let nLayers = model.newCache(parameters: nil).count
            func simpleCaches() -> [KVCache] { (0 ..< nLayers).map { _ in KVCacheSimple() } }
            func argmaxLast(_ logits: MLXArray) -> Int {
                MLX.argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
            }
            let promptArr = MLXArray(promptIds.map { Int32($0) }).reshaped([1, promptIds.count])

            // ---- greedy AR reference (verifier only) ----
            let arCache = simpleCaches()
            var bAR = argmaxLast(model.callAsFunction(promptArr, cache: arCache))
            var ar = [bAR]
            while ar.count < nGen {
                let lg = model.callAsFunction(MLXArray([Int32(bAR)]).reshaped([1, 1]), cache: arCache)
                bAR = argmaxLast(lg)
                ar.append(bAR)
            }

            // ---- speculative loop ----
            box.gen.reset()
            let vCache = simpleCaches()
            let (pLogits, pCaps) = model.forwardCapture(promptArr, cache: vCache, captureLayerIds: capIds)
            var b = argmaxLast(pLogits)
            box.gen.prefill(promptTokens: promptIds,
                            verifierHidden3x: concatenated(pCaps, axis: -1), bonus: b)
            var spec = [b]
            var rounds = 0, acceptedTotal = 0, draftedTotal = 0

            while spec.count < nGen {
                let draftTokens = box.gen.draftBlock(blockSize: blockSize)
                if draftTokens.isEmpty { break }

                var verifyIds = [b]; verifyIds.append(contentsOf: draftTokens)
                let vArr = MLXArray(verifyIds.map { Int32($0) }).reshaped([1, verifyIds.count])
                let (vLogits, vCaps) = model.forwardCapture(vArr, cache: vCache, captureLayerIds: capIds)
                let target = MLX.argMax(vLogits[0, 0..., 0...], axis: -1).asArray(Int32.self).map { Int($0) }

                // acceptance walk: longest matching prefix, then the verifier's bonus token
                let nd = draftTokens.count
                var accepted = nd
                for i in 0 ..< nd where draftTokens[i] != target[i] { accepted = i; break }
                let bonus = target[accepted]
                var newTokens = Array(draftTokens[0 ..< accepted]); newTokens.append(bonus)

                let budget = nGen - spec.count
                spec.append(contentsOf: newTokens.prefix(budget))
                rounds += 1; acceptedTotal += accepted; draftedTotal += nd

                // verifier KV rollback: keep b + accepted drafts (= accepted+1 positions)
                let trim = verifyIds.count - (accepted + 1)
                if trim > 0 { for c in vCache { _ = c.trim(trim) } }

                if spec.count >= nGen { break }
                box.gen.accept(verifyHidden3x: concatenated(vCaps, axis: -1),
                               draftTokens: draftTokens, accepted: accepted, newLastToken: bonus)
                b = bonus
            }
            return LoopOut(ar: ar, spec: Array(spec.prefix(nGen)),
                           rounds: rounds, acceptedTotal: acceptedTotal, draftedTotal: draftedTotal)
        }
        guard let r = result else { return }

        let firstDiff = Array(zip(r.ar, r.spec)).firstIndex(where: { $0 != $1 }) ?? -1
        let acceptRate = r.draftedTotal > 0 ? Float(r.acceptedTotal) / Float(r.draftedTotal) : 0
        let tokPerRound = r.rounds > 0 ? Float(r.spec.count) / Float(r.rounds) : 0
        print(String(format: "[E3-P1] gen=%d rounds=%d accept=%d/%d (%.1f%%) tok/round=%.2f firstDiff=%d",
                     r.spec.count, r.rounds, r.acceptedTotal, r.draftedTotal,
                     acceptRate * 100, tokPerRound, firstDiff))
        print("[E3-P1] AR  : \(r.ar.prefix(16))")
        print("[E3-P1] SPEC: \(r.spec.prefix(16))")

        #expect(r.spec.count == r.ar.count, "length mismatch spec \(r.spec.count) vs ar \(r.ar.count)")
        #expect(r.spec == r.ar, "speculative output diverges from greedy AR at index \(firstDiff)")
    }
}
