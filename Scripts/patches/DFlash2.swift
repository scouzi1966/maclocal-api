import Foundation
import MLX
import MLXNN

public struct DFlash2TargetOutput {
    public let hidden: MLXArray
    public let logits: MLXArray

    public init(hidden: MLXArray, logits: MLXArray) {
        self.hidden = hidden
        self.logits = logits
    }
}

public protocol DFlash2Target: AnyObject {
    var dflash2HiddenSize: Int { get }
    var dflash2LayerCount: Int { get }
    var dflash2VocabularySize: Int { get }
    func dflash2NewCache() -> [any KVCache]
    func dflash2Embed(_ tokenIDs: MLXArray) -> MLXArray
    func dflash2Project(_ hidden: MLXArray) -> MLXArray
    func dflash2Forward(
        _ tokenIDs: MLXArray,
        captureLayerIDs: [Int],
        cache: [any KVCache]
    ) -> DFlash2TargetOutput
    func dflash2CaptureCache(_ cache: [any KVCache]) -> Any
    func dflash2RestoreCache(_ snapshot: Any, into cache: [any KVCache])
}

public struct DFlash2GenerationStatistics: Sendable {
    public let draftedTokens: Int
    public let acceptedDraftTokens: Int
    public let emittedTokens: Int
    public let verificationCycles: Int
    public let draftSeconds: Double
    public let verificationSeconds: Double
    public let rollbackSeconds: Double

    public var meanAcceptanceLength: Double {
        verificationCycles > 0
            ? Double(acceptedDraftTokens) / Double(verificationCycles)
            : 0
    }
}

public struct DFlash2GenerationResult: Sendable {
    public let tokenIDs: [Int]
    public let statistics: DFlash2GenerationStatistics
}

public struct DFlash2DraftConfiguration: Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let vocabularySize: Int
    public let targetLayers: Int
    public let targetLayerIDs: [Int]
    public let blockSize: Int
    public let maskTokenID: Int
    public let convolutionKernelSize: Int
    public let convolutionGroupSize: Int
    public let selectorRank: Int
    public let selectorTopK: Int
    public let rmsNormEpsilon: Float
    public let ropeTheta: Float

    public static func load(directory: String) throws -> Self {
        let data = try Data(contentsOf: URL(fileURLWithPath: directory + "/config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              (root["architectures"] as? [String])?.contains("DFlash2DraftModel") == true,
              (root["is_causal"] as? Bool) == false,
              let dflash = root["dflash_config"] as? [String: Any]
        else {
            throw DFlash2Error.invalidConfiguration(
                "architectures must contain DFlash2DraftModel and is_causal must be false")
        }
        func integer(_ object: [String: Any], _ key: String) throws -> Int {
            guard let value = (object[key] as? NSNumber)?.intValue else {
                throw DFlash2Error.invalidConfiguration("missing \(key)")
            }
            return value
        }
        func floating(_ object: [String: Any], _ key: String) throws -> Float {
            guard let value = (object[key] as? NSNumber)?.floatValue else {
                throw DFlash2Error.invalidConfiguration("missing \(key)")
            }
            return value
        }
        let hidden = try integer(root, "hidden_size")
        let layers = try integer(root, "num_hidden_layers")
        let targetLayers = try integer(root, "num_target_layers")
        let targetIDs = (dflash["target_layer_ids"] as? [NSNumber])?.map(\.intValue) ?? []
        let groupSize = try integer(dflash, "conv_group_size")
        let kernelSize = try integer(dflash, "conv_kernel_size")
        guard hidden > 0, layers > 0, hidden.isMultiple(of: groupSize), kernelSize == 2,
              targetIDs.count == layers,
              targetIDs.allSatisfy({ $0 >= 0 && $0 < targetLayers }) else {
            throw DFlash2Error.invalidConfiguration("invalid hidden, convolution, or target-layer layout")
        }
        return Self(
            hiddenSize: hidden,
            intermediateSize: try integer(root, "intermediate_size"),
            hiddenLayers: layers,
            attentionHeads: try integer(root, "num_attention_heads"),
            keyValueHeads: try integer(root, "num_key_value_heads"),
            headDimension: try integer(root, "head_dim"),
            vocabularySize: try integer(root, "vocab_size"),
            targetLayers: targetLayers,
            targetLayerIDs: targetIDs,
            blockSize: try integer(dflash, "block_size"),
            maskTokenID: try integer(dflash, "mask_token_id"),
            convolutionKernelSize: kernelSize,
            convolutionGroupSize: groupSize,
            selectorRank: try integer(dflash, "selector_rank"),
            selectorTopK: try integer(dflash, "selector_top_k"),
            rmsNormEpsilon: try floating(root, "rms_norm_eps"),
            ropeTheta: try floating(root, "rope_theta")
        )
    }
}

public enum DFlash2Error: LocalizedError {
    case invalidConfiguration(String)
    case incompatibleTarget(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message): return "Invalid DFlash2 configuration: \(message)"
        case .incompatibleTarget(let message): return "Incompatible DFlash2 target: \(message)"
        }
    }
}

private final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: DFlash2DraftConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        down(silu(gate(value)) * up(value))
    }
}

private final class DFlash2GroupedDynamicCausalConv: Module {
    let groupSize: Int
    let groups: Int
    let kernelSize: Int
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear
    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray

    init(_ config: DFlash2DraftConfiguration) {
        groupSize = config.convolutionGroupSize
        groups = config.hiddenSize / config.convolutionGroupSize
        kernelSize = config.convolutionKernelSize
        _kernelProjection.wrappedValue = Linear(
            config.hiddenSize, 2 * kernelSize * groups, bias: false)
        _baseKernel.wrappedValue = MLXArray.zeros(
            [2, kernelSize, config.hiddenSize], dtype: .float32)
    }

    private func convolve(_ hidden: MLXArray, dynamic: MLXArray, phase: Int) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let blocks = hidden.reshaped(batch, length, groups, groupSize)
        var output = MLXArray.zeros(like: blocks)
        for offset in 0 ..< kernelSize {
            let values: MLXArray
            if offset == 0 {
                values = blocks
            } else {
                let padding = MLXArray.zeros(
                    [batch, offset, groups, groupSize], dtype: blocks.dtype)
                values = concatenated(
                    [padding, blocks[0..., 0 ..< (length - offset), 0..., 0...]], axis: 1)
            }
            let fixed = baseKernel[phase, offset, 0...].reshaped(1, 1, groups, groupSize)
            let generated = dynamic[0..., 0..., offset, 0..., .newAxis]
            output = output + (fixed + generated) * values
        }
        return output.reshaped(hidden.shape)
    }

    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let dynamic = kernelProjection(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return (convolve(hidden, dynamic: dynamic[0..., 0..., 0, 0..., 0...], phase: 0),
                dynamic[0..., 0..., 1, 0..., 0...])
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        convolve(hidden, dynamic: dynamic, phase: 1)
    }
}

private final class DFlash2Attention: Module {
    let heads: Int
    let keyValueHeads: Int
    let headDimension: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    let rope: RoPE

    init(_ config: DFlash2DraftConfiguration) {
        heads = config.attentionHeads
        keyValueHeads = config.keyValueHeads
        headDimension = config.headDimension
        scale = pow(Float(config.headDimension), -0.5)
        _query.wrappedValue = Linear(config.hiddenSize, heads * headDimension, bias: false)
        _key.wrappedValue = Linear(config.hiddenSize, keyValueHeads * headDimension, bias: false)
        _value.wrappedValue = Linear(config.hiddenSize, keyValueHeads * headDimension, bias: false)
        _output.wrappedValue = Linear(heads * headDimension, config.hiddenSize, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: config.rmsNormEpsilon)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: config.rmsNormEpsilon)
        rope = RoPE(dimensions: headDimension, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(_ hidden: MLXArray, targetContext: MLXArray) -> MLXArray {
        let batch = hidden.dim(0)
        let block = hidden.dim(1)
        let context = targetContext.dim(1)
        var q = query(hidden).reshaped(batch, block, heads, headDimension)
        q = queryNorm(q).transposed(0, 2, 1, 3)
        q = rope(q, offset: context)

        var contextK = key(targetContext).reshaped(batch, context, keyValueHeads, headDimension)
        contextK = keyNorm(contextK).transposed(0, 2, 1, 3)
        contextK = rope(contextK, offset: 0)
        let contextV = value(targetContext).reshaped(
            batch, context, keyValueHeads, headDimension).transposed(0, 2, 1, 3)

        var blockK = key(hidden).reshaped(batch, block, keyValueHeads, headDimension)
        blockK = keyNorm(blockK).transposed(0, 2, 1, 3)
        blockK = rope(blockK, offset: context)
        let blockV = value(hidden).reshaped(
            batch, block, keyValueHeads, headDimension).transposed(0, 2, 1, 3)

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: concatenated([contextK, blockK], axis: 2),
            values: concatenated([contextV, blockV], axis: 2),
            scale: scale,
            mask: .none)
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, block, -1))
    }
}

private final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlash2Attention
    @ModuleInfo var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2GroupedDynamicCausalConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2GroupedDynamicCausalConv

    init(_ config: DFlash2DraftConfiguration) {
        _attention.wrappedValue = DFlash2Attention(config)
        _mlp.wrappedValue = DFlash2MLP(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _attentionConv.wrappedValue = DFlash2GroupedDynamicCausalConv(config)
        _mlpConv.wrappedValue = DFlash2GroupedDynamicCausalConv(config)
    }

    func callAsFunction(_ hidden: MLXArray, targetContext: MLXArray) -> MLXArray {
        let (attentionInput, attentionKernel) = attentionConv.prepare(inputNorm(hidden))
        var result = hidden + attentionConv.finish(
            attention(attentionInput, targetContext: targetContext), dynamic: attentionKernel)
        let (mlpInput, mlpKernel) = mlpConv.prepare(postAttentionNorm(result))
        result = result + mlpConv.finish(mlp(mlpInput), dynamic: mlpKernel)
        return result
    }
}

public final class DFlash2DraftModel: Module {
    public let config: DFlash2DraftConfiguration
    @ModuleInfo private var layers: [DFlash2DecoderLayer]
    @ModuleInfo private var norm: RMSNorm
    @ModuleInfo private var fc: Linear
    @ModuleInfo(key: "hidden_norm") private var hiddenNorm: RMSNorm
    @ModuleInfo(key: "candidate_selector.hidden_projection") private var selectorHidden: Linear
    @ModuleInfo(key: "candidate_selector.predecessor_codebook") private var predecessor: Embedding
    @ModuleInfo(key: "candidate_selector.successor_codebook") private var successor: Embedding

    public init(_ config: DFlash2DraftConfiguration) {
        self.config = config
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in DFlash2DecoderLayer(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _fc.wrappedValue = Linear(
            config.targetLayerIDs.count * config.hiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _selectorHidden.wrappedValue = Linear(config.hiddenSize, config.selectorRank, bias: false)
        _predecessor.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.selectorRank)
        _successor.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.selectorRank)
    }

    public func callAsFunction(noiseEmbedding: MLXArray, targetHidden: MLXArray) -> MLXArray {
        let context = hiddenNorm(fc(targetHidden))
        var hidden = noiseEmbedding
        for layer in layers {
            hidden = layer(hidden, targetContext: context)
        }
        return norm(hidden)
    }

    /// DFlash2's selector greedily walks adjacent candidate pairs. The target
    /// verifier still decides every emitted token.
    public func select(hidden: MLXArray, logits: MLXArray, anchor: Int) -> [Int] {
        let topK = config.selectorTopK
        let candidates = argPartition(logits, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = selectorHidden(hidden)
        var previous = MLXArray([Int32(anchor)])
        var path: [Int] = []
        for position in 0 ..< hidden.dim(1) {
            let candidate = candidates[0..., position, 0...]
            let edge = MLX.sum(
                predecessor(previous)[0..., .newAxis, 0...]
                    * projected[0..., position, .newAxis, 0...]
                    * successor(candidate),
                axis: -1)
            let selected = argMax(unary[0..., position, 0...] + edge, axis: -1)
            previous = takeAlong(candidate, selected[.ellipsis, .newAxis], axis: -1)[.ellipsis, 0]
            path.append(previous.item(Int.self))
        }
        return path
    }

    public static func load(directory: String) throws -> DFlash2DraftModel {
        let config = try DFlash2DraftConfiguration.load(directory: directory)
        let model = DFlash2DraftModel(config)
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(atPath: directory)
            where file.hasSuffix(".safetensors") {
            for (key, value) in try MLX.loadArrays(
                url: URL(fileURLWithPath: directory + "/" + file)) {
                var mapped = key
                if mapped == "candidate_selector.predecessor_codebook" {
                    mapped += ".weight"
                } else if mapped == "candidate_selector.successor_codebook" {
                    mapped += ".weight"
                }
                weights[mapped] = value
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}

public final class DFlash2Generator {
    private let target: any DFlash2Target
    private let draft: DFlash2DraftModel
    private let blockSize: Int

    public init(target: any DFlash2Target, draft: DFlash2DraftModel, blockSize: Int) throws {
        guard target.dflash2HiddenSize == draft.config.hiddenSize,
              target.dflash2LayerCount == draft.config.targetLayers,
              target.dflash2VocabularySize == draft.config.vocabularySize else {
            throw DFlash2Error.incompatibleTarget(
                "hidden size, layer count, and vocabulary must match the drafter config")
        }
        guard blockSize >= 2 else {
            throw DFlash2Error.invalidConfiguration("runtime block size must be at least 2")
        }
        self.target = target
        self.draft = draft
        self.blockSize = min(blockSize, draft.config.blockSize)
    }

    private func array(_ ids: [Int]) -> MLXArray {
        MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
    }

    private func greedy(_ logits: MLXArray, position: Int) -> Int {
        argMax(logits[0, position, 0...], axis: -1).item(Int.self)
    }

    public func generate(
        promptIDs: [Int],
        maxTokens: Int,
        stopTokenIDs: Set<Int> = [],
        shouldStop: () -> Bool = { false },
        onToken: ((Int) -> Bool)? = nil
    ) -> DFlash2GenerationResult {
        guard !promptIDs.isEmpty, maxTokens > 0 else {
            return DFlash2GenerationResult(
                tokenIDs: [],
                statistics: DFlash2GenerationStatistics(
                    draftedTokens: 0, acceptedDraftTokens: 0, emittedTokens: 0,
                    verificationCycles: 0, draftSeconds: 0,
                    verificationSeconds: 0, rollbackSeconds: 0))
        }

        let cache = target.dflash2NewCache()
        let prefill = target.dflash2Forward(
            array(promptIDs), captureLayerIDs: draft.config.targetLayerIDs, cache: cache)
        var context = prefill.hidden
        var staged = greedy(prefill.logits, position: prefill.logits.dim(1) - 1)
        var output: [Int] = []
        var draftedCount = 0
        var acceptedCount = 0
        var cycles = 0
        var draftTime = 0.0
        var verifyTime = 0.0
        var rollbackTime = 0.0
        var stopped = false

        func emit(_ token: Int) -> Bool {
            if shouldStop() { return false }
            output.append(token)
            if let onToken, !onToken(token) { return false }
            return output.count < maxTokens && !stopTokenIDs.contains(token)
        }

        while output.count < maxTokens, !stopped {
            guard emit(staged) else { break }
            let remaining = maxTokens - output.count
            if remaining <= 0 { break }

            let cycleBlock = min(blockSize, remaining + 1)
            let noiseIDs = [staged] + Array(
                repeating: draft.config.maskTokenID, count: cycleBlock - 1)
            let draftStart = Date.timeIntervalSinceReferenceDate
            let draftHidden = draft(
                noiseEmbedding: target.dflash2Embed(array(noiseIDs)),
                targetHidden: context)
            let proposalHidden = draftHidden[0..., 1..., 0...]
            let proposalLogits = target.dflash2Project(proposalHidden)
            let proposal = draft.select(
                hidden: proposalHidden, logits: proposalLogits, anchor: staged)
            eval(proposalLogits)
            draftTime += Date.timeIntervalSinceReferenceDate - draftStart
            draftedCount += proposal.count

            let candidate = [staged] + proposal
            let snapshot = target.dflash2CaptureCache(cache)
            let verifyStart = Date.timeIntervalSinceReferenceDate
            let verified = target.dflash2Forward(
                array(candidate), captureLayerIDs: draft.config.targetLayerIDs, cache: cache)
            eval(verified.logits, verified.hidden)
            verifyTime += Date.timeIntervalSinceReferenceDate - verifyStart
            cycles += 1

            var accepted = 0
            while accepted < proposal.count,
                  proposal[accepted] == greedy(verified.logits, position: accepted) {
                accepted += 1
            }
            acceptedCount += accepted
            let committed = Array(candidate.prefix(accepted + 1))

            let rollbackStart = Date.timeIntervalSinceReferenceDate
            target.dflash2RestoreCache(snapshot, into: cache)
            let replay = target.dflash2Forward(
                array(committed), captureLayerIDs: draft.config.targetLayerIDs, cache: cache)
            eval(replay.hidden)
            context = concatenated([context, replay.hidden], axis: 1)
            rollbackTime += Date.timeIntervalSinceReferenceDate - rollbackStart

            for token in proposal.prefix(accepted) {
                if !emit(token) {
                    stopped = true
                    break
                }
            }
            if stopped { break }
            staged = greedy(verified.logits, position: accepted)
        }

        return DFlash2GenerationResult(
            tokenIDs: output,
            statistics: DFlash2GenerationStatistics(
                draftedTokens: draftedCount,
                acceptedDraftTokens: acceptedCount,
                emittedTokens: output.count,
                verificationCycles: cycles,
                draftSeconds: draftTime,
                verificationSeconds: verifyTime,
                rollbackSeconds: rollbackTime))
    }
}
