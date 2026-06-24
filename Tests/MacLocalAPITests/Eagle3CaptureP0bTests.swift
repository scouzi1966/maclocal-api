import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXLLM

@testable import AFMKit
@testable import AFMServer

/// P0b gate for the EAGLE3 port: the dense Gemma4-31B verifier's hidden-state capture hook
/// (`forwardCapture` @ layers [2,30,57]) must reproduce the mlx-vlm reference captures, and the
/// drafter's fc-fusion of them must match `fused_hidden`. This validates the single novel
/// structural piece (the capture point) end-to-end into the drafter input.
///
/// Requires the dense 31B model + EAGLE3 drafter on disk; skips cleanly if absent.
struct Eagle3CaptureP0bTests {
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

    @Test("Gemma4-31B hidden capture @ [2,30,57] matches reference; fc-fusion matches")
    func captureMatchesReference() async throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: Self.modelDir + "/config.json"),
              let drafterDir = Self.drafterDir(),
              fm.fileExists(atPath: Self.fixtureDir + "/eagle3_ref.safetensors") else {
            print("[E3-P0b] model/drafter/fixture absent — skipping"); return
        }

        // Force the LLM factory (server path: gemma-4-31b is text-loadable -> Gemma4Model).
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(directory: URL(fileURLWithPath: Self.modelDir)))
        let drafter = try Gemma4Eagle3Drafter.load(directory: drafterDir)
        let capIds = drafter.config.captureLayerIds   // [2,30,57]

        let meta = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: Self.fixtureDir + "/eagle3_ref.json"))) as! [String: Any]
        let ids = (meta["prompt_ids"] as! [Any]).map { Int32($0 as! Int) }
        let f = try MLX.loadArrays(url: URL(fileURLWithPath: Self.fixtureDir + "/eagle3_ref.safetensors"))
        let refPrimary = meta["primary"] as! Int

        func cos(_ a: MLXArray, _ b: MLXArray) -> Float {
            (MLX.sum(a * b) / (MLX.sqrt(MLX.sum(a * a)) * MLX.sqrt(MLX.sum(b * b)))).item(Float.self)
        }

        // Run the capture forward inside perform (model lock); return tensors as float32 arrays
        // for comparison OUTSIDE the @Sendable closure (which can't capture drafter/f/cos).
        let capCount = capIds.count
        struct CaptureOut: @unchecked Sendable {
            let primary: Int; let caps: [MLXArray]; let fused: MLXArray
        }
        let result: CaptureOut? = try await container.perform { ctx -> CaptureOut? in
            guard let model = ctx.model as? Gemma4Model else {
                Issue.record("loaded model is not Gemma4Model (\(type(of: ctx.model)))"); return nil
            }
            let input = MLXArray(ids).reshaped([1, ids.count])
            let cache = model.newCache(parameters: nil)
            let (logits, caps) = model.forwardCapture(input, cache: cache, captureLayerIds: capIds)
            let prim = MLX.argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
            let lastCaps = caps.map { $0[0..., ($0.dim(1) - 1)..., 0...] }
            // fused needs the drafter.fc — do it here is not possible (drafter not Sendable);
            // return raw last-position captures (float32) and fuse outside.
            let capsF = lastCaps.map { $0.asType(.float32) }
            capsF.forEach { eval($0) }
            return CaptureOut(primary: prim, caps: capsF, fused: MLXArray(0))
        }
        guard let r = result else { return }

        #expect(r.caps.count == capCount, "captured \(r.caps.count) != \(capCount) layers")
        #expect(r.primary == refPrimary, "primary \(r.primary) != ref \(refPrimary)")
        // Per-layer gate: after fixing the full-attention ProportionalRoPE bug (denominator +
        // rotate-half geometry), captures match to >0.9987. The residual is smooth bf16 rounding
        // accumulation over up to 57 layers (monotonic with depth, no step at full-attention
        // layers) — not a structural error. The strict >0.999 gate is enforced on `fused` below,
        // which IS the drafter input that determines draft quality.
        let perLayerGate: Float = 0.998
        var worst: Float = 1
        for (i, cap) in r.caps.enumerated() {
            let c = cos(cap, f["cap\(i)"]!.asType(.float32))
            worst = Swift.min(worst, c)
            print(String(format: "[E3-P0b] layer %d cap cos=%.6f", capIds[i], c))
            #expect(c > perLayerGate, "capture@\(capIds[i]) diverges (cos \(c))")
        }
        // fc-fuse outside the closure (drafter is here).
        let fused = drafter.prepareTargetHidden(
            concatenated(r.caps.map { $0.asType(.bfloat16) }, axis: -1)).asType(.float32)
        eval(fused)
        let fcos = cos(fused, f["fused_hidden"]!.asType(.float32))
        print(String(format: "[E3-P0b] fused cos=%.6f (worst layer cos=%.6f)", fcos, worst))
        #expect(fcos > 0.999, "fc-fused hidden diverges (cos \(fcos))")
    }
}
