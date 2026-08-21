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

/// A verifier cache snapshot must preserve both storage and cursor metadata.
/// Rotating caches update their backing arrays in place after wrapping, so
/// retaining array references or trimming by offset is not a rollback.
public enum DFlash2CacheSnapshot {
    public struct Layer {
        fileprivate let state: [MLXArray]
        fileprivate let metaState: [String]
    }

    public static func capture(_ cache: [any KVCache]) -> [Layer] {
        cache.map { entry in
            let state = entry.state.map { ($0 + MLXArray(0)).asType($0.dtype) }
            if !state.isEmpty { eval(state) }
            return Layer(state: state, metaState: entry.metaState)
        }
    }

    public static func restore(_ snapshot: [Layer], into cache: [any KVCache]) {
        precondition(snapshot.count == cache.count, "DFlash cache snapshot layer mismatch")
        for (index, layer) in snapshot.enumerated() {
            var entry = cache[index]
            if !layer.state.isEmpty {
                let state = layer.state.map { ($0 + MLXArray(0)).asType($0.dtype) }
                eval(state)
                entry.state = state
            }
            entry.metaState = layer.metaState
        }
    }
}

public struct DFlash2GenerationStatistics: Sendable {
    public let draftedTokens: Int
    public let acceptedDraftTokens: Int
    public let emittedTokens: Int
    public let verificationCycles: Int
    public let prefillSeconds: Double
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

public enum DFlashDraftArchitecture: String, Sendable {
    case dflash = "MuseGlimmerAssistantModel"
    case dflash2 = "DFlash2DraftModel"
}

public struct DFlash2DraftConfiguration: Sendable {
    public let architecture: DFlashDraftArchitecture
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
    public let maxPositionEmbeddings: Int
    public let layerTypes: [String]
    public let slidingWindow: Int?

    public static func load(
        directory: String,
        targetLayers resolvedTargetLayers: Int? = nil,
        vocabularySize resolvedVocabularySize: Int? = nil
    ) throws -> Self {
        let data = try Data(contentsOf: URL(fileURLWithPath: directory + "/config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DFlash2Error.invalidConfiguration("config.json must contain an object")
        }
        let architectures = root["architectures"] as? [String] ?? []
        let architecture: DFlashDraftArchitecture
        if architectures.contains(DFlashDraftArchitecture.dflash2.rawValue) {
            architecture = .dflash2
        } else if architectures.contains(DFlashDraftArchitecture.dflash.rawValue) {
            architecture = .dflash
        } else {
            throw DFlash2Error.invalidConfiguration(
                "unsupported draft architectures: \(architectures)")
        }
        if architecture == .dflash2, (root["is_causal"] as? Bool) != false {
            throw DFlash2Error.invalidConfiguration("DFlash2 requires is_causal=false")
        }
        let dflash = root["dflash_config"] as? [String: Any] ?? root
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
        let targetLayers = (root["num_target_layers"] as? NSNumber)?.intValue
            ?? resolvedTargetLayers
        let vocabularySize = (root["vocab_size"] as? NSNumber)?.intValue
            ?? resolvedVocabularySize
        guard let targetLayers, let vocabularySize else {
            throw DFlash2Error.invalidConfiguration(
                "draft config requires target num_hidden_layers and vocab_size")
        }
        let targetIDs = (dflash["target_layer_ids"] as? [NSNumber])?.map(\.intValue) ?? []
        let groupSize = (dflash["conv_group_size"] as? NSNumber)?.intValue ?? 0
        let kernelSize = (dflash["conv_kernel_size"] as? NSNumber)?.intValue ?? 0
        let layerTypes = root["layer_types"] as? [String] ?? []
        let slidingWindow = (root["sliding_window"] as? NSNumber)?.intValue
        let rope = root["rope_parameters"] as? [String: Any]
            ?? root["rope_scaling"] as? [String: Any]
            ?? root
        guard hidden > 0, layers > 0,
              targetIDs.count == layers,
              targetIDs == targetIDs.sorted(), Set(targetIDs).count == targetIDs.count,
              targetIDs.allSatisfy({ $0 >= 0 && $0 < targetLayers }),
              layerTypes.count == layers,
              layerTypes.allSatisfy({ $0 == "sliding_attention" || $0 == "full_attention" }),
              !layerTypes.contains("sliding_attention") || (slidingWindow ?? 0) > 0 else {
            throw DFlash2Error.invalidConfiguration("invalid hidden or target-layer layout")
        }
        if architecture == .dflash2,
           !(groupSize > 0 && hidden.isMultiple(of: groupSize) && kernelSize == 2) {
            throw DFlash2Error.invalidConfiguration("invalid DFlash2 convolution layout")
        }
        return Self(
            architecture: architecture,
            hiddenSize: hidden,
            intermediateSize: try integer(root, "intermediate_size"),
            hiddenLayers: layers,
            attentionHeads: try integer(root, "num_attention_heads"),
            keyValueHeads: try integer(root, "num_key_value_heads"),
            headDimension: try integer(root, "head_dim"),
            vocabularySize: vocabularySize,
            targetLayers: targetLayers,
            targetLayerIDs: targetIDs,
            blockSize: try integer(dflash, "block_size"),
            maskTokenID: try integer(dflash, "mask_token_id"),
            convolutionKernelSize: kernelSize,
            convolutionGroupSize: groupSize,
            selectorRank: (dflash["selector_rank"] as? NSNumber)?.intValue ?? 0,
            selectorTopK: (dflash["selector_top_k"] as? NSNumber)?.intValue ?? 0,
            rmsNormEpsilon: try floating(root, "rms_norm_eps"),
            ropeTheta: try floating(rope, "rope_theta"),
            maxPositionEmbeddings: try integer(root, "max_position_embeddings"),
            layerTypes: layerTypes,
            slidingWindow: slidingWindow
        )
    }
}

public protocol DFlashDraftingModel: AnyObject {
    var config: DFlash2DraftConfiguration { get }
    func callAsFunction(noiseEmbedding: MLXArray, targetHidden: MLXArray) -> MLXArray
    func select(hidden: MLXArray, logits: MLXArray, anchor: Int) -> [Int]
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
    let slidingWindow: Int?
    let isCausal: Bool
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    let rope: RoPE

    init(_ config: DFlash2DraftConfiguration, layerIndex: Int) {
        heads = config.attentionHeads
        keyValueHeads = config.keyValueHeads
        headDimension = config.headDimension
        scale = pow(Float(config.headDimension), -0.5)
        slidingWindow = config.layerTypes[layerIndex] == "sliding_attention"
            ? config.slidingWindow : nil
        isCausal = config.architecture == .dflash && slidingWindow != nil
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
        let fullContext = targetContext.dim(1)
        let contextStart: Int
        let visibleContext: MLXArray
        if let slidingWindow, fullContext > slidingWindow - 1 {
            contextStart = fullContext - (slidingWindow - 1)
            visibleContext = targetContext[0..., contextStart..., 0...]
        } else {
            contextStart = 0
            visibleContext = targetContext
        }
        let context = visibleContext.dim(1)
        var q = query(hidden).reshaped(batch, block, heads, headDimension)
        q = queryNorm(q).transposed(0, 2, 1, 3)
        q = rope(q, offset: fullContext)

        var contextK = key(visibleContext).reshaped(batch, context, keyValueHeads, headDimension)
        contextK = keyNorm(contextK).transposed(0, 2, 1, 3)
        contextK = rope(contextK, offset: contextStart)
        let contextV = value(visibleContext).reshaped(
            batch, context, keyValueHeads, headDimension).transposed(0, 2, 1, 3)

        var blockK = key(hidden).reshaped(batch, block, keyValueHeads, headDimension)
        blockK = keyNorm(blockK).transposed(0, 2, 1, 3)
        blockK = rope(blockK, offset: fullContext)
        let blockV = value(hidden).reshaped(
            batch, block, keyValueHeads, headDimension).transposed(0, 2, 1, 3)

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let slidingWindow {
            let queryIndices = MLXArray(
                Int32(fullContext) ..< Int32(fullContext + block))[0..., .newAxis]
            let keyIndices = MLXArray(
                Int32(contextStart) ..< Int32(fullContext + block))[.newAxis, 0...]
            let contextMask = (keyIndices .< Int32(fullContext))
                .&& ((queryIndices - keyIndices) .< Int32(slidingWindow))
            var blockMask = keyIndices .>= Int32(fullContext)
            if isCausal {
                blockMask = blockMask .&& (keyIndices .<= queryIndices)
            }
            mask = .array(contextMask .|| blockMask)
        } else {
            mask = .none
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: concatenated([contextK, blockK], axis: 2),
            values: concatenated([contextV, blockV], axis: 2),
            scale: scale,
            mask: mask)
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, block, -1))
    }
}

private final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlash2Attention
    @ModuleInfo var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(_ config: DFlash2DraftConfiguration, layerIndex: Int) {
        _attention.wrappedValue = DFlash2Attention(config, layerIndex: layerIndex)
        _mlp.wrappedValue = DFlash2MLP(config)
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
    }

    func callAsFunction(_ hidden: MLXArray, targetContext: MLXArray) -> MLXArray {
        var result = hidden + attention(inputNorm(hidden), targetContext: targetContext)
        result = result + mlp(postAttentionNorm(result))
        return result
    }
}

private final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlash2Attention
    @ModuleInfo var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2GroupedDynamicCausalConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2GroupedDynamicCausalConv

    init(_ config: DFlash2DraftConfiguration, layerIndex: Int) {
        _attention.wrappedValue = DFlash2Attention(config, layerIndex: layerIndex)
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

private final class DFlash2CandidateSelector: Module {
    let topK: Int
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear
    @ModuleInfo(key: "predecessor_codebook") var predecessor: Embedding
    @ModuleInfo(key: "successor_codebook") var successor: Embedding

    init(_ config: DFlash2DraftConfiguration) {
        topK = config.selectorTopK
        _hiddenProjection.wrappedValue = Linear(config.hiddenSize, config.selectorRank, bias: false)
        _predecessor.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.selectorRank)
        _successor.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.selectorRank)
    }

    func select(hidden: MLXArray, logits: MLXArray, anchor: Int) -> [Int] {
        let candidates = argPartition(logits, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)
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
}

public final class DFlashDraftModel: Module, DFlashDraftingModel {
    public let config: DFlash2DraftConfiguration
    @ModuleInfo private var layers: [DFlashDecoderLayer]
    @ModuleInfo private var norm: RMSNorm
    @ModuleInfo private var fc: Linear
    @ModuleInfo(key: "hidden_norm") private var hiddenNorm: RMSNorm

    public init(_ config: DFlash2DraftConfiguration) {
        precondition(config.architecture == .dflash)
        self.config = config
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            DFlashDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _fc.wrappedValue = Linear(
            config.targetLayerIDs.count * config.hiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
    }

    public func callAsFunction(
        noiseEmbedding: MLXArray, targetHidden: MLXArray
    ) -> MLXArray {
        let context = hiddenNorm(fc(targetHidden))
        var hidden = noiseEmbedding
        for layer in layers {
            hidden = layer(hidden, targetContext: context)
        }
        return norm(hidden)
    }

    public func select(hidden: MLXArray, logits: MLXArray, anchor: Int) -> [Int] {
        (0 ..< logits.dim(1)).map { position in
            argMax(logits[0, position, 0...], axis: -1).item(Int.self)
        }
    }
}

public final class DFlash2DraftModel: Module, DFlashDraftingModel {
    public let config: DFlash2DraftConfiguration
    @ModuleInfo private var layers: [DFlash2DecoderLayer]
    @ModuleInfo private var norm: RMSNorm
    @ModuleInfo private var fc: Linear
    @ModuleInfo(key: "hidden_norm") private var hiddenNorm: RMSNorm
    @ModuleInfo(key: "candidate_selector") private var candidateSelector: DFlash2CandidateSelector

    public init(_ config: DFlash2DraftConfiguration) {
        precondition(config.architecture == .dflash2)
        self.config = config
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            DFlash2DecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _fc.wrappedValue = Linear(
            config.targetLayerIDs.count * config.hiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _candidateSelector.wrappedValue = DFlash2CandidateSelector(config)
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
        candidateSelector.select(hidden: hidden, logits: logits, anchor: anchor)
    }

}

public enum DFlashDraftModelFactory {
    public static func load(
        directory: String,
        targetLayers: Int? = nil,
        vocabularySize: Int? = nil
    ) throws -> any DFlashDraftingModel {
        let config = try DFlash2DraftConfiguration.load(
            directory: directory,
            targetLayers: targetLayers,
            vocabularySize: vocabularySize)
        let weights = try loadWeights(directory: directory, architecture: config.architecture)
        switch config.architecture {
        case .dflash:
            let model = DFlashDraftModel(config)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
            eval(model)
            return model
        case .dflash2:
            let model = DFlash2DraftModel(config)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
            eval(model)
            return model
        }
    }

    private static func loadWeights(
        directory: String,
        architecture: DFlashDraftArchitecture
    ) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(atPath: directory)
            where file.hasSuffix(".safetensors") {
            for (key, value) in try MLX.loadArrays(
                url: URL(fileURLWithPath: directory + "/" + file)) {
                var mapped = key
                if mapped == "candidate_selector.predecessor_codebook"
                    || mapped == "candidate_selector.successor_codebook" {
                    mapped += ".weight"
                } else if architecture == .dflash, mapped == "encoder.fc.weight" {
                    mapped = "fc.weight"
                } else if architecture == .dflash,
                          mapped == "encoder.output_norm_enc.weight" {
                    mapped = "hidden_norm.weight"
                }
                weights[mapped] = value
            }
        }
        return weights
    }
}

public extension DFlash2DraftModel {
    static func load(directory: String) throws -> DFlash2DraftModel {
        let model = try DFlashDraftModelFactory.load(directory: directory)
        guard let model = model as? DFlash2DraftModel else {
            throw DFlash2Error.invalidConfiguration("expected DFlash2DraftModel")
        }
        return model
    }
}

public final class DFlash2Generator {
    private let target: any DFlash2Target
    private let draft: any DFlashDraftingModel
    private let blockSize: Int

    public init(
        target: any DFlash2Target,
        draft: any DFlashDraftingModel,
        blockSize: Int
    ) throws {
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

    private func elapsedSeconds(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    }

    public func generate(
        promptIDs: [Int],
        maxTokens: Int,
        stopTokenIDs: Set<Int> = [],
        shouldStop: () -> Bool = { false },
        onToken: ((Int) -> Bool)? = nil
    ) throws -> DFlash2GenerationResult {
        guard !promptIDs.isEmpty, maxTokens > 0 else {
            return DFlash2GenerationResult(
                tokenIDs: [],
                statistics: DFlash2GenerationStatistics(
                    draftedTokens: 0, acceptedDraftTokens: 0, emittedTokens: 0,
                    verificationCycles: 0, prefillSeconds: 0, draftSeconds: 0,
                    verificationSeconds: 0, rollbackSeconds: 0))
        }

        let cache = target.dflash2NewCache()
        if shouldStop() { throw CancellationError() }
        let prefillStart = DispatchTime.now().uptimeNanoseconds
        let prefill = target.dflash2Forward(
            array(promptIDs), captureLayerIDs: draft.config.targetLayerIDs, cache: cache)
        eval(prefill.hidden, prefill.logits)
        let prefillTime = elapsedSeconds(since: prefillStart)
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

        func emit(_ token: Int) throws -> Bool {
            if shouldStop() { throw CancellationError() }
            if stopTokenIDs.contains(token) { return false }
            output.append(token)
            if let onToken, !onToken(token) { throw CancellationError() }
            return output.count < maxTokens
        }

        while output.count < maxTokens, !stopped {
            guard try emit(staged) else { break }
            let remaining = maxTokens - output.count
            if remaining <= 0 { break }

            let cycleBlock = min(blockSize, remaining + 1)
            let noiseIDs = [staged] + Array(
                repeating: draft.config.maskTokenID, count: cycleBlock - 1)
            let draftStart = DispatchTime.now().uptimeNanoseconds
            let draftHidden = draft.callAsFunction(
                noiseEmbedding: target.dflash2Embed(array(noiseIDs)),
                targetHidden: context)
            let proposalHidden = draftHidden[0..., 1..., 0...]
            let proposalLogits = target.dflash2Project(proposalHidden)
            let proposal = draft.select(
                hidden: proposalHidden, logits: proposalLogits, anchor: staged)
            eval(proposalLogits)
            draftTime += elapsedSeconds(since: draftStart)
            draftedCount += proposal.count
            if shouldStop() { throw CancellationError() }

            let candidate = [staged] + proposal
            let snapshot = target.dflash2CaptureCache(cache)
            let verifyStart = DispatchTime.now().uptimeNanoseconds
            let verified = target.dflash2Forward(
                array(candidate), captureLayerIDs: draft.config.targetLayerIDs, cache: cache)
            eval(verified.logits, verified.hidden)
            verifyTime += elapsedSeconds(since: verifyStart)
            cycles += 1
            if shouldStop() { throw CancellationError() }

            var matched = 0
            while matched < proposal.count,
                  proposal[matched] == greedy(verified.logits, position: matched) {
                matched += 1
            }
            let terminalIndex = proposal.prefix(matched).firstIndex {
                stopTokenIDs.contains($0)
            }
            let accepted = terminalIndex.map { $0 + 1 } ?? matched
            acceptedCount += accepted
            let committed = Array(candidate.prefix(accepted + 1))

            let rollbackStart = DispatchTime.now().uptimeNanoseconds
            target.dflash2RestoreCache(snapshot, into: cache)
            let replay = target.dflash2Forward(
                array(committed), captureLayerIDs: draft.config.targetLayerIDs, cache: cache)
            eval(replay.hidden)
            context = concatenated([context, replay.hidden], axis: 1)
            rollbackTime += elapsedSeconds(since: rollbackStart)

            for token in proposal.prefix(accepted) {
                if try emit(token) == false {
                    stopped = true
                    break
                }
            }
            if stopped { break }
            staged = greedy(verified.logits, position: accepted)
        }

        if shouldStop() { throw CancellationError() }

        return DFlash2GenerationResult(
            tokenIDs: output,
            statistics: DFlash2GenerationStatistics(
                draftedTokens: draftedCount,
                acceptedDraftTokens: acceptedCount,
                emittedTokens: output.count,
                verificationCycles: cycles,
                prefillSeconds: prefillTime,
                draftSeconds: draftTime,
                verificationSeconds: verifyTime,
                rollbackSeconds: rollbackTime))
    }
}
