import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXLLM

@testable import AFMKit

/// P0 gate for the EAGLE3 Swift port: `Gemma4Eagle3Drafter` must reproduce the mlx-vlm reference
/// for one draft step. Fixture (Scripts/eagle3-port/fixtures/, from capture_eagle3_reference.py):
///   inputs   — fused_hidden (1,1,5376) [= fc over captured hidden@2,30,57], primary token (json)
///   expected — draft_hidden (1,1,5376), draft_logits_hot (1,32000), draft_full token (json)
/// We feed primary + fused_hidden through forwardTokens -> logits -> argmax -> d2t, and check the
/// post-layer hidden and the drafted full-vocab token match.
///
/// Requires the EAGLE3 drafter on disk (HF cache); skips cleanly if absent.
struct Eagle3DrafterP0Tests {
    static let fixtureDir: String = {
        let here = URL(fileURLWithPath: #filePath)
        return here.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Scripts/eagle3-port/fixtures").path
    }()

    /// Resolve the drafter snapshot dir under the HF cache (hash subdir).
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

    @Test("EAGLE3 drafter reproduces the Python reference draft step")
    func drafterMatchesReference() throws {
        let fm = FileManager.default
        guard let dir = Self.drafterDir() else {
            print("[E3-P0] drafter absent — skipping"); return
        }
        let fixST = Self.fixtureDir + "/eagle3_ref.safetensors"
        let fixJSON = Self.fixtureDir + "/eagle3_ref.json"
        guard fm.fileExists(atPath: fixST), fm.fileExists(atPath: fixJSON) else {
            print("[E3-P0] fixture absent — run capture_eagle3_reference.py"); return
        }

        let drafter = try Gemma4Eagle3Drafter.load(directory: dir)

        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: fixJSON))) as! [String: Any]
        let primary = meta["primary"] as! Int
        let expectedFull = meta["draft_full"] as! Int

        let f = try MLX.loadArrays(url: URL(fileURLWithPath: fixST))
        let fused = f["fused_hidden"]!.asType(.bfloat16)      // (1,1,5376) already fc-fused
        let expectedHidden = f["draft_hidden"]!.asType(.float32)

        // Drafter forward: one token (primary) with the fused hidden. positionOffset=1 (mirrors
        // the reference's _next_position after prefill).
        let cache = drafter.newCache()
        let tok = MLXArray([Int32(primary)]).reshaped([1, 1])
        let h = drafter.forwardTokens(tok, hidden: fused, cache: cache, positionOffset: 1).asType(.float32)
        eval(h)

        let cos = (MLX.sum(h * expectedHidden)
            / (MLX.sqrt(MLX.sum(h * h)) * MLX.sqrt(MLX.sum(expectedHidden * expectedHidden)))).item(Float.self)
        let meanAbs = MLX.abs(h - expectedHidden).mean().item(Float.self)
        let refScale = MLX.abs(expectedHidden).mean().item(Float.self)

        let hotLogits = drafter.logits(h)
        let hot = MLX.argMax(hotLogits[0, -1, 0...], axis: -1)
        let full = drafter.draftToTarget(hot).item(Int.self)

        print(String(format: "[E3-P0] draft_hidden cos=%.6f mean|Δ|=%.4f refScale=%.4f | drafted full=%d expected=%d",
                     cos, meanAbs, refScale, full, expectedFull))

        #expect(h.shape == expectedHidden.shape)
        #expect(cos > 0.999, "draft hidden diverges (cos \(cos))")
        #expect(full == expectedFull, "drafted token \(full) != reference \(expectedFull)")
    }
}
