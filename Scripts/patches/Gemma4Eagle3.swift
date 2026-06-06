//
//  Gemma4Eagle3.swift
//
//  EAGLE3 speculative-decoding drafter for the dense Gemma4-31B verifier.
//  Port of mlx-vlm v0.6.0 mlx_vlm/speculative/drafters/eagle3/eagle3.py (Eagle3DraftModel).
//
//  The drafter is a single Llama-style transformer layer whose first layer ("Eagle3FirstLayer")
//  attends over concat(token_embedding, fused_target_hidden). The fused hidden = fc over the
//  concatenation of the verifier's residual-stream hidden states captured at layers [2,30,57].
//  It projects to a reduced 32000-token "hot" vocab via its own lm_head, mapped back to the full
//  262144 vocab via a d2t (draft->target) index table. Weights ship as bf16 PyTorch safetensors
//  (RedHatAI/gemma-4-31B-it-speculator.eagle3); loaded here without quantization.
//
//  Validated target: dense gemma-4-31b-it-4bit, +25% decode vs AR (see EAGLE3-PORT-PLAN.md).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Drafter config (subset of the speculator config.json we need)

public struct Gemma4Eagle3Config: Codable, Sendable {
    public var hiddenSize: Int          // transformer_layer_config.hidden_size (5376)
    public var intermediateSize: Int    // 21504
    public var numHiddenLayers: Int     // 1
    public var numAttentionHeads: Int   // 32
    public var numKeyValueHeads: Int    // 16
    public var headDim: Int             // 256
    public var rmsNormEps: Float        // 1e-6
    public var ropeTheta: Float         // 10000.0
    public var ropeTraditional: Bool    // false
    public var attentionBias: Bool      // false
    public var draftVocabSize: Int      // 32000
    public var targetVocabSize: Int     // 262144
    public var targetHiddenSize: Int    // 5376 (== hiddenSize here)
    public var normBeforeResidual: Bool // true
    public var normBeforeFc: Bool       // false
    public var captureLayerIds: [Int]   // [2,30,57]

    /// Decode from the speculator's config.json (nested transformer_layer_config + top-level eagle keys).
    public static func from(configPath: String) throws -> Gemma4Eagle3Config {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        let j = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let tlc = j["transformer_layer_config"] as? [String: Any] ?? [:]
        func i(_ d: [String: Any], _ k: String, _ def: Int) -> Int { (d[k] as? Int) ?? def }
        func f(_ d: [String: Any], _ k: String, _ def: Float) -> Float {
            if let v = d[k] as? Double { return Float(v) }; if let v = d[k] as? Int { return Float(v) }; return def
        }
        let hidden = i(tlc, "hidden_size", 5376)
        let heads = i(tlc, "num_attention_heads", 32)
        let hd = (tlc["head_dim"] as? Int) ?? (hidden / heads)
        let caps = (j["eagle_aux_hidden_state_layer_ids"] as? [Any])?.compactMap { ($0 as? Int) } ?? [2, 30, 57]
        return Gemma4Eagle3Config(
            hiddenSize: hidden,
            intermediateSize: i(tlc, "intermediate_size", 21504),
            numHiddenLayers: i(tlc, "num_hidden_layers", 1),
            numAttentionHeads: heads,
            numKeyValueHeads: i(tlc, "num_key_value_heads", 16),
            headDim: hd,
            rmsNormEps: f(tlc, "rms_norm_eps", 1e-6),
            ropeTheta: f(tlc, "rope_theta", 10000.0),
            ropeTraditional: (tlc["rope_traditional"] as? Bool) ?? false,
            attentionBias: (tlc["attention_bias"] as? Bool) ?? false,
            draftVocabSize: i(j, "draft_vocab_size", 32000),
            targetVocabSize: i(tlc, "vocab_size", 262144),
            targetHiddenSize: (j["target_hidden_size"] as? Int) ?? hidden,
            normBeforeResidual: (j["norm_before_residual"] as? Bool) ?? false,
            normBeforeFc: (j["norm_before_fc"] as? Bool) ?? false,
            captureLayerIds: caps.sorted())
    }
}

// MARK: - Llama-style attention/MLP (the drafter's OWN modules — not Gemma4's)

private class Eagle3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: Gemma4Eagle3Config) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

private class Eagle3Attention: Module {
    let nHeads: Int, nKVHeads: Int, headDim: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: RoPE

    /// `inputSize` overrides the q/k/v input dim (the EAGLE first layer feeds 2*hidden).
    init(_ c: Gemma4Eagle3Config, inputSize: Int? = nil) {
        nHeads = c.numAttentionHeads; nKVHeads = c.numKeyValueHeads; headDim = c.headDim
        scale = pow(Float(headDim), -0.5)
        let inDim = inputSize ?? c.hiddenSize
        _qProj.wrappedValue = Linear(inDim, nHeads * headDim, bias: c.attentionBias)
        _kProj.wrappedValue = Linear(inDim, nKVHeads * headDim, bias: c.attentionBias)
        _vProj.wrappedValue = Linear(inDim, nKVHeads * headDim, bias: c.attentionBias)
        _oProj.wrappedValue = Linear(nHeads * headDim, c.hiddenSize, bias: c.attentionBias)
        rope = RoPE(dimensions: headDim, traditional: c.ropeTraditional, base: c.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache?, positionOffset: Int) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        q = rope(q, offset: positionOffset)
        k = rope(k, offset: positionOffset)
        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
            .transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(out)
    }
}

// EAGLE first layer: attention sees concat(token_embed, fused_hidden) (input 2*hidden).
private class Eagle3FirstLayer: Module {
    let normBeforeResidual: Bool
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Eagle3Attention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Eagle3MLP

    init(_ c: Gemma4Eagle3Config) {
        normBeforeResidual = c.normBeforeResidual
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _selfAttn.wrappedValue = Eagle3Attention(c, inputSize: 2 * c.hiddenSize)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = Eagle3MLP(c)
    }

    func callAsFunction(embeds: MLXArray, hidden: MLXArray,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache?, positionOffset: Int) -> MLXArray {
        let e = inputLayerNorm(embeds)
        let hNormed = hiddenNorm(hidden)
        let residual = normBeforeResidual ? hNormed : hidden
        var h = concatenated([e, hNormed], axis: -1)
        h = selfAttn(h, mask: mask, cache: cache, positionOffset: positionOffset)
        h = residual + h
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// Subsequent layers (num_hidden_layers > 1; the 31B speculator has 1, so usually unused).
private class Eagle3DecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Eagle3Attention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Eagle3MLP
    init(_ c: Gemma4Eagle3Config) {
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _selfAttn.wrappedValue = Eagle3Attention(c)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = Eagle3MLP(c)
    }
    func callAsFunction(_ hidden: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache?, positionOffset: Int) -> MLXArray {
        let h = hidden + selfAttn(inputLayerNorm(hidden), mask: mask, cache: cache, positionOffset: positionOffset)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - EAGLE3 drafter

public final class Gemma4Eagle3Drafter: Module {
    public let config: Gemma4Eagle3Config

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "input_norm") var inputNorm: RMSNorm?
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    fileprivate let first: Eagle3FirstLayer
    fileprivate let rest: [Eagle3DecoderLayer]
    /// draft->target vocab offset table (hot id -> full id = id + d2t[id]); empty if no hot vocab.
    public private(set) var d2t: MLXArray?

    public init(_ c: Gemma4Eagle3Config) {
        config = c
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.targetVocabSize, dimensions: c.hiddenSize)
        _fc.wrappedValue = Linear(3 * c.targetHiddenSize, c.hiddenSize, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _inputNorm.wrappedValue = c.normBeforeFc ? RMSNorm(dimensions: 3 * c.targetHiddenSize, eps: c.rmsNormEps) : nil
        let usesDraftVocab = c.draftVocabSize != c.targetVocabSize
        _lmHead.wrappedValue = usesDraftVocab ? Linear(c.hiddenSize, c.draftVocabSize, bias: false) : nil
        first = Eagle3FirstLayer(c)
        rest = (1 ..< Swift.max(1, c.numHiddenLayers)).map { _ in Eagle3DecoderLayer(c) }
        super.init()
    }

    /// fc-fuse the concatenated target hidden (3*targetHidden -> hidden). If already hidden-sized
    /// (e.g. pre-fused), pass through. Mirrors `_prepare_target_hidden`.
    public func prepareTargetHidden(_ hidden: MLXArray) -> MLXArray {
        if hidden.dim(-1) == config.hiddenSize { return hidden }
        var h = hidden
        if let inputNorm { h = inputNorm(h) }
        return fc(h)
    }

    /// Project a hidden state to the (hot) draft vocab logits. Mirrors `_logits`.
    public func logits(_ hidden: MLXArray) -> MLXArray {
        let h = norm(hidden)
        if let lmHead { return lmHead(h) }
        return embedTokens.asLinear(h)
    }

    /// Map hot draft ids -> full target vocab ids (`id + d2t[id]`). Mirrors `_draft_to_target`.
    public func draftToTarget(_ draftIds: MLXArray) -> MLXArray {
        let ids = draftIds.asType(.int32)
        guard let d2t else { return ids }
        return ids + d2t[ids]
    }

    /// One drafter forward over `tokens` with the given (fc-fused or raw) target `hidden`.
    /// Returns the post-layers hidden. Mirrors `_forward_tokens` (without the cache-position
    /// bookkeeping, which the generator owns). `positionOffset` is the RoPE/KV offset.
    public func forwardTokens(_ tokens: MLXArray, hidden: MLXArray, cache: [KVCache]?,
                              positionOffset: Int) -> MLXArray {
        let h0 = prepareTargetHidden(hidden)
        let embeds = embedTokens(tokens.asType(.int32))
        var h = h0
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.causal
        let c0 = cache?[0]
        h = first(embeds: embeds, hidden: h, mask: mask, cache: c0, positionOffset: positionOffset)
        for (i, layer) in rest.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i + 1], positionOffset: positionOffset)
        }
        return h
    }

    public func newCache() -> [KVCache] { (0 ..< Swift.max(1, config.numHiddenLayers)).map { _ in KVCacheSimple() } }

    /// Last-position slice [1,1,H] of a [1,L,H] hidden tensor.
    fileprivate static func lastPos(_ a: MLXArray) -> MLXArray { a[0..., (a.dim(1) - 1)..., 0...] }

    // MARK: Loading

    /// Load the drafter from a directory of bf16 safetensors + config.json.
    /// `bindEmbed` optionally shares the target's embedding (saves ~1.4GB if shapes match).
    public static func load(directory: String, bindEmbed: Embedding? = nil) throws -> Gemma4Eagle3Drafter {
        let cfg = try Gemma4Eagle3Config.from(configPath: directory + "/config.json")
        let drafter = Gemma4Eagle3Drafter(cfg)

        // Gather safetensors weights from the directory.
        var weights: [String: MLXArray] = [:]
        let fm = FileManager.default
        for f in (try? fm.contentsOfDirectory(atPath: directory)) ?? [] where f.hasSuffix(".safetensors") {
            let arrs = try MLX.loadArrays(url: URL(fileURLWithPath: directory + "/" + f))
            for (k, v) in arrs { weights[k] = v }
        }
        // d2t / t2d are buffers, not module params — pull d2t out before update().
        if let d2t = weights["d2t"] { drafter.d2t = d2t.asType(.int32); weights["d2t"] = nil }
        weights["t2d"] = nil

        // The drafter ships bf16; keep dtype as-is. Map layer keys: HF stores the single
        // EAGLE layer as `layers.0.*`; our module names it `first` (+ `rest` for any extras).
        var mapped: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.hasPrefix("layers.0.") { mapped["first." + String(k.dropFirst("layers.0.".count))] = v }
            else if k.hasPrefix("layers.") {
                // layers.N.* (N>=1) -> rest.(N-1).*
                let rest = String(k.dropFirst("layers.".count))
                if let dot = rest.firstIndex(of: "."), let n = Int(rest[..<dot]), n >= 1 {
                    mapped["rest.\(n - 1)." + String(rest[rest.index(after: dot)...])] = v
                } else { mapped[k] = v }
            } else { mapped[k] = v }
        }
        try drafter.update(parameters: ModuleParameters.unflattened(mapped), verify: [.all])
        if let bindEmbed, bindEmbed.weight.shape == drafter.embedTokens.weight.shape {
            drafter.embedTokens = bindEmbed
        }
        eval(drafter)
        return drafter
    }
}

// MARK: - Greedy speculative generator (drafter-side state machine)

/// Owns the EAGLE3 drafter's speculative state for a single greedy sequence: its KV cache, the
/// next RoPE/cache position, and the carried "seed" (the drafter's predicted next token plus the
/// hidden that produced it). Mirrors the drafter half of mlx-vlm's `Eagle3DraftModel`
/// (`prefill_from_target_hidden` / `draft_block` / `accept_verified_tokens`). The verifier
/// forward, the acceptance walk, and the verifier-cache rollback are owned by the driver (the P1
/// test today, the service in P2).
public final class Gemma4Eagle3Generator {
    public let drafter: Gemma4Eagle3Drafter
    private var cache: [KVCache]
    private var nextPosition: Int = 1
    private var seedToken: Int?
    private var seedHidden: MLXArray?
    /// In-block drafter forwards since the last accept — trimmed off before the accept re-forward.
    private var roundAppended: Int = 0

    public init(drafter: Gemma4Eagle3Drafter) {
        self.drafter = drafter
        self.cache = drafter.newCache()
    }

    public func reset() {
        cache = drafter.newCache()
        nextPosition = 1
        seedToken = nil
        seedHidden = nil
        roundAppended = 0
    }

    /// Greedy argmax over the (hot) draft vocab at the last position, mapped to the full vocab.
    private func draftArgmax(_ hiddenLast: MLXArray) -> Int {
        let hot = MLX.argMax(drafter.logits(hiddenLast)[0, -1, 0...], axis: -1)
        return drafter.draftToTarget(hot).item(Int.self)
    }

    private func setSeed(_ hiddenLast: MLXArray) {
        seedToken = draftArgmax(hiddenLast)
        seedHidden = hiddenLast
    }

    /// Prime the drafter KV with the prompt (shifted by one, `bonus` appended) using the verifier's
    /// captured+concatenated hidden states, then compute the first seed. `verifierHidden3x` is
    /// `[1, P, 3*targetHidden]` over the prompt positions; `bonus` is the verifier's first token.
    public func prefill(promptTokens: [Int], verifierHidden3x: MLXArray, bonus: Int) {
        guard promptTokens.count > 0 else { return }
        var shifted = Array(promptTokens.dropFirst())
        shifted.append(bonus)                       // length == promptTokens.count
        let toks = MLXArray(shifted.map { Int32($0) }).reshaped([1, shifted.count])
        nextPosition = 1
        let hid = verifierHidden3x[0..., 0 ..< shifted.count, 0...]
        let h = drafter.forwardTokens(toks, hidden: hid, cache: cache, positionOffset: nextPosition)
        nextPosition += shifted.count
        setSeed(Gemma4Eagle3Drafter.lastPos(h))
    }

    /// Produce `blockSize - 1` greedy draft tokens. The first is the carried seed; each subsequent
    /// token is the drafter's own next-token prediction, feeding its own hidden forward.
    public func draftBlock(blockSize: Int) -> [Int] {
        roundAppended = 0
        var tokens: [Int] = []
        guard var tok = seedToken, var hPrev = seedHidden else { return tokens }
        tokens.append(tok)
        seedToken = nil
        seedHidden = nil
        while tokens.count < blockSize - 1 {
            let tokArr = MLXArray([Int32(tok)]).reshaped([1, 1])
            hPrev = drafter.forwardTokens(tokArr, hidden: hPrev, cache: cache, positionOffset: nextPosition)
            nextPosition += 1
            roundAppended += 1
            tok = draftArgmax(Gemma4Eagle3Drafter.lastPos(hPrev))
            tokens.append(tok)
        }
        return tokens
    }

    /// After verification: trim the in-block drafter forwards, then re-forward the accepted tokens
    /// (plus the bonus) with the verifier's hidden to advance the drafter KV and compute the next
    /// seed. `verifyHidden3x` is `[1, blockSize, 3*targetHidden]`; `accepted` is the number of
    /// matched draft tokens; `newLastToken` is the bonus (verifier) token that follows them.
    public func accept(verifyHidden3x: MLXArray, draftTokens: [Int], accepted: Int, newLastToken: Int) {
        if roundAppended > 0 {
            for c in cache { _ = c.trim(roundAppended) }
            nextPosition -= roundAppended
        }
        var toks: [Int] = []
        var hiddenSlices: [MLXArray] = []
        if accepted > 0 {
            toks.append(contentsOf: draftTokens[0 ..< accepted])
            hiddenSlices.append(verifyHidden3x[0..., 0 ..< accepted, 0...])
        }
        toks.append(newLastToken)
        hiddenSlices.append(verifyHidden3x[0..., accepted ..< (accepted + 1), 0...])

        let tokArr = MLXArray(toks.map { Int32($0) }).reshaped([1, toks.count])
        let hid = hiddenSlices.count == 1 ? hiddenSlices[0] : concatenated(hiddenSlices, axis: 1)
        let h = drafter.forwardTokens(tokArr, hidden: hid, cache: cache, positionOffset: nextPosition)
        nextPosition += toks.count
        setSeed(Gemma4Eagle3Drafter.lastPos(h))
        roundAppended = 0
    }
}
