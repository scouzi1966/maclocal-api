# Implementation Guide

How to implement a new model architecture in AFM's MLX Swift backend.

## Patch System Overview

All vendor modifications go through `Scripts/patches/`. The patch script (`Scripts/apply-mlx-patches.sh`) copies complete Swift files from `Scripts/patches/` to vendor targets.

Three arrays in the script define the mapping:

```bash
PATCH_FILES=("NewModel.swift" "LLMModelFactory.swift" ...)
TARGET_PATHS=("Libraries/MLXLLM/Models/NewModel.swift" "Libraries/MLXLLM/LLMModelFactory.swift" ...)
NEW_FILES=("NewModel.swift")  # Files that don't exist upstream
```

- `PATCH_FILES` — filenames in `Scripts/patches/`
- `TARGET_PATHS` — relative paths under `vendor/mlx-swift-lm/`
- `NEW_FILES` — files that are new (no upstream original to back up)

## New Model File Structure

Create `Scripts/patches/<ModelName>.swift`. Follow this pattern (based on existing models):

```swift
import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct <ModelName>Configuration: Codable, Sendable {
    // Map config.json fields to Swift properties
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var vocabSize: Int
    var rmsNormEps: Float
    var ropeTheta: Float = 10000
    var tieWordEmbeddings: Bool = false
    var maxPositionEmbeddings: Int = 4096
    // Add model-specific fields...

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}

// MARK: - Attention

private class Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE  // or MLXFast.RoPE for performance

    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var oProj: Linear

    init(_ config: <ModelName>Configuration) {
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self.rope = RoPE(dimensions: headDim, base: config.ropeTheta)
        self._qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var q = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        if let cache {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
            (k, v) = cache.update(keys: k, values: v)
        } else {
            q = rope(q)
            k = rope(k)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - MLP

private class MLP: Module {
    @ModuleInfo var gate_proj: Linear
    @ModuleInfo var up_proj: Linear
    @ModuleInfo var down_proj: Linear

    init(_ config: <ModelName>Configuration) {
        self._gate_proj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._up_proj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._down_proj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down_proj(silu(gate_proj(x)) * up_proj(x))
    }
}

// MARK: - Transformer Block

private class TransformerBlock: Module {
    @ModuleInfo var selfAttn: Attention
    @ModuleInfo var mlp: MLP
    @ModuleInfo var inputLayernorm: RMSNorm
    @ModuleInfo var postAttentionLayernorm: RMSNorm

    init(_ config: <ModelName>Configuration) {
        self._selfAttn.wrappedValue = Attention(config)
        self._mlp.wrappedValue = MLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let r = selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        let h = x + r
        let out = mlp(postAttentionLayernorm(h))
        return h + out
    }
}

// MARK: - Model

public class <ModelName>Model: Module, LLMModel, KVCacheDimensionProvider {
    public let kvHeads: [Int]

    @ModuleInfo var embedTokens: Embedding
    @ModuleInfo var layers: [TransformerBlock]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo var lmHead: Linear?

    public init(_ config: <ModelName>Configuration) {
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in TransformerBlock(config) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var x = embedTokens(inputs)
        let mask = createAttentionMask(h: x, cache: cache)
        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask, cache: cache?[i])
        }
        x = norm(x)
        if let lmHead { return lmHead(x) }
        return embedTokens.asLinear(x)
    }
}
```

## Protocol Conformance

Models must conform to:
- `LLMModel` — provides `callAsFunction(_:cache:)` and generation interface
- `KVCacheDimensionProvider` — provides `kvHeads: [Int]` for KV cache allocation

## Weight Sanitization

If the model's weight names differ from the Swift property names, add a `sanitize()` method to the Configuration:

```swift
public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var result = [String: MLXArray]()
    for (key, value) in weights {
        let newKey = key
            .replacingOccurrences(of: "model.", with: "")
            .replacingOccurrences(of: "self_attn", with: "selfAttn")
            // ... other mappings
        result[newKey] = value
    }
    return result
}
```

## Registration Checklist

After creating the model file:

1. **Add to patch arrays** in `Scripts/apply-mlx-patches.sh`:
   ```bash
   PATCH_FILES=( ... "NewModel.swift" )
   TARGET_PATHS=( ... "Libraries/MLXLLM/Models/NewModel.swift" )
   NEW_FILES=( ... "NewModel.swift" )
   ```

2. **Register in LLMTypeRegistry** in `Scripts/patches/LLMModelFactory.swift`:
   ```swift
   "new_model_type": create(NewModelConfiguration.self, NewModelModel.init),
   ```

3. **Register in VLMTypeRegistry** (if VLM) in `Scripts/patches/VLMModelFactory.swift`

4. **Update ToolCallFormat.infer()** (if new tool call format) in `Scripts/patches/ToolCallFormat.swift`

5. **Apply patches**:
   ```bash
   ./Scripts/apply-mlx-patches.sh
   ```

6. **Build**:
   ```bash
   swift build
   ```

7. **Verify** — start server and test:
   ```bash
   MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache afm mlx -m <model-id> --port 9999
   ```
   ```bash
   curl http://localhost:9999/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
   ```

## Tips

- Use an existing model with similar architecture as your starting template
- For MoE models, look at `Qwen3_5MoE.swift` or `DeepseekV3.swift` as references
- For VLMs, look at `Qwen3VL.swift` or `Qwen3_5MoEVL.swift`
- Match Python mlx-lm layer names exactly — weight loading depends on name alignment
- Use `MLXFast.scaledDotProductAttention` for attention (not manual matmul)
- Use `MLXFast.rmsNorm` for performance-critical RMSNorm (weight must match input dtype)
- The `@ModuleInfo` property wrapper is required for parameters to be discovered by the module system
