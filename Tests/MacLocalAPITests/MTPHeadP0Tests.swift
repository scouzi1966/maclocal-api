import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXVLM

@testable import MacLocalAPI

/// P0 gate for the MTP Swift port: the Swift `Qwen3_5MTPHead` must reproduce the Python
/// (mtplx) reference for one draft step.
///
/// Fixture (Scripts/mtp-port/fixtures/, captured by capture_mtp_reference.py):
///   inputs   — last_hidden (1,1,5120), primary_embed (1,1,5120)
///   expected — draft_hidden (1,1,5120) = head's post_norm output
/// The head's final projection reuses the trunk lm_head, so validating the post_norm hidden
/// fully exercises the head (fc fusion + the attention/MLP block + norms). Logit/argmax parity
/// is checked at P2 once the trunk lm_head is wired in.
///
/// Requires the MTP sidecar on disk; skips cleanly if absent so CI without the model passes.
struct MTPHeadP0Tests {
    static let sidecar =
        "/Volumes/Crucial4TB/models/mtplx/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed/mtp.safetensors"
    static let fixtureDir: String = {
        // Resolve from THIS source file's location (CWD is unreliable: MLXMetalLibrary
        // chdir's to the metallib dir during setup). #filePath =
        // <repo>/Tests/MacLocalAPITests/MTPHeadP0Tests.swift → up 3 → <repo>.
        let here = URL(fileURLWithPath: #filePath)
        let repo = here.deletingLastPathComponent()  // MacLocalAPITests
            .deletingLastPathComponent()              // Tests
            .deletingLastPathComponent()              // repo root
        return repo.appendingPathComponent("Scripts/mtp-port/fixtures").path
    }()

    init() throws { try MLXMetalLibrary.ensureAvailable(verbose: false) }

    /// Build the Qwen3.6-27B text config matching the model (only the fields the head needs).
    private func headConfig() -> Qwen3_5MoEVLTextConfiguration? {
        // Decode from the model's config.json text_config so we don't hand-maintain constants.
        let cfgPath =
            "/Volumes/Crucial4TB/models/mtplx/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed/config.json"
        guard let data = FileManager.default.contents(atPath: cfgPath) else { return nil }
        let top = try? JSONDecoder().decode(Qwen3_5MoEVLConfiguration.self, from: data)
        return top?.textConfig
    }

    @Test("MTP head reproduces the Python reference draft_hidden")
    func headMatchesReference() throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: Self.sidecar) else {
            print("[P0] sidecar absent — skipping (download Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed)")
            return
        }
        let fixture = Self.fixtureDir + "/mtp_ref_step.safetensors"
        guard fm.fileExists(atPath: fixture) else {
            print("[P0] fixture absent — run Scripts/mtp-port/capture_mtp_reference.py first")
            return
        }
        guard let cfg = headConfig() else {
            Issue.record("could not load text config")
            return
        }

        // Load + quantize the head from the sidecar.
        let head = try Qwen3_5MTPHead.load(sidecarPath: Self.sidecar, config: cfg)

        // Load fixture inputs/expected (float32).
        let f = try MLX.loadArrays(url: URL(fileURLWithPath: fixture))
        let lastHidden = f["last_hidden"]!.asType(.bfloat16)      // run in bf16 like the model
        let primaryEmbed = f["primary_embed"]!.asType(.bfloat16)
        let expected = f["draft_hidden"]!.asType(.float32)

        // Single-token step: a fresh single-layer KV cache, causal mask is nil for L=1.
        let cache = KVCacheSimple()
        let out = head(hiddenStates: lastHidden, tokenEmbeds: primaryEmbed, mask: nil, cache: cache)
            .asType(.float32)
        eval(out)

        #expect(out.shape == expected.shape, "shape \(out.shape) vs \(expected.shape)")

        // Numerical closeness: bf16 compute vs the reference's bf16 path. Use a relative metric
        // robust to the ~1e-2 scale of bf16 rounding across a 64-layer-equivalent block.
        let diff = MLX.abs(out - expected)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)
        let refScale = MLX.abs(expected).mean().item(Float.self)
        let cos = (MLX.sum(out * expected)
            / (MLX.sqrt(MLX.sum(out * out)) * MLX.sqrt(MLX.sum(expected * expected)))).item(Float.self)
        print(String(format: "[P0] draft_hidden: max|Δ|=%.4f mean|Δ|=%.4f refScale=%.4f cos=%.6f",
                     maxAbs, meanAbs, refScale, cos))

        // Cosine similarity is the primary gate (direction must match almost exactly);
        // mean abs error should be small relative to the activation scale.
        #expect(cos > 0.999, "cosine \(cos) — head output diverges from reference")
        #expect(meanAbs < 0.1 * refScale + 0.05, "mean|Δ| \(meanAbs) too large vs scale \(refScale)")
    }
}
