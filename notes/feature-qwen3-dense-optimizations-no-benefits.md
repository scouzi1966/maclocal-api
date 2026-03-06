# feature/qwen3-dense-optimizations-no-benefits

## What Was Done

Ported three performance optimizations from the patched `Qwen3_5MoE.swift` (MoE architecture) to the dense `Qwen3.swift` architecture:

1. **Fused QKV Attention** — Combined 3 separate quantized matmuls (`wq`, `wk`, `wv`) into a single `quantizedMatmul` + `split`. Lazy initialization on first forward pass; falls back to separate calls for non-quantized models.

2. **Fused Gate+Up MLP with fusedSiluMul** — Combined `gate_proj` and `up_proj` into a single `quantizedMatmul`, then applied the custom `fusedSiluMul` Metal kernel (replaces slice+slice+silu+multiply with 1 GPU dispatch).

3. **MLXFast.rmsNorm for Q/K Norms** — Replaced `RMSNorm` module calls (4 graph ops: x*x, mean, rsqrt, mul) with `MLXFast.rmsNorm` C++ kernel (1 op). Pre-computed weight vectors cast to input dtype on first call.

Files changed:
- `Scripts/patches/Qwen3.swift` — new patch file with all three optimizations
- `Scripts/apply-mlx-patches.sh` — added Qwen3.swift to PATCH_FILES and TARGET_PATHS

## Measurements

Platform: M3 Ultra, 512GB, macOS 26. 500 tokens generated, 3 runs each, same prompt.

### Qwen3.5-2B-bf16 (unquantized, ~3.7GB)

| Build | Run 1 | Run 2 | Run 3 | Avg |
|---|---|---|---|---|
| Baseline v0.9.6 | 134.8 | 134.7 | 134.4 | **134.6 tok/s** |
| Patched | 134.3 | 134.2 | 134.2 | **134.2 tok/s** |

Delta: ~0% — Fusions require `QuantizedLinear`; unquantized model uses fallback paths.

### Qwen3.5-9B-MLX-4bit (quantized, ~4.9GB)

| Build | Run 1 | Run 2 | Run 3 | Avg |
|---|---|---|---|---|
| Baseline v0.9.6 | 113.9 | 113.4 | 113.6 | **113.7 tok/s** |
| Patched | 114.0 | 113.8 | 113.7 | **113.9 tok/s** |

Delta: ~0% — Within measurement noise.

### Qwen3.5-27B-8bit (quantized, ~26.8GB)

| Build | Run 1 | Run 2 | Run 3 | Avg |
|---|---|---|---|---|
| Baseline v0.9.6 | 22.1 | 22.4 | 22.4 | **22.3 tok/s** |
| Patched | 22.1 | 22.4 | 22.5 | **22.3 tok/s** |

Delta: 0% — Completely memory-bandwidth bound.

## Outcome

**No measurable benefit for dense Qwen3 models.** Not merged to main.

### Why It Doesn't Help

Dense models are **memory-bandwidth bound** on Apple Silicon. Every token requires reading all model weights:

- 9B-4bit: ~5GB read per token → ~8.8ms/tok at 113 tok/s
- 27B-8bit: ~27GB read per token → ~45ms/tok at 22 tok/s

The fusions reduce GPU kernel dispatches (~40 layers × ~4 fewer dispatches × 0.8μs = **~0.13ms saved per token**), which is <2% of the per-token time — invisible against the bandwidth wall.

### Why It Works for MoE

These same optimizations provide **27% speedup on Qwen3.5-35B-A3B-4bit** (95.7 → 128 tok/s) because MoE models only activate a fraction of experts per token. The active parameter read is ~4GB (vs 27GB for dense 27B), making per-token time ~7.8ms where dispatch overhead (~1ms) is a significant 13% fraction.

### Applicability

| Model Type | Active Params | Bandwidth Bound? | Dispatch Savings Matter? |
|---|---|---|---|
| Dense 2B+ | All params | Yes | No |
| MoE (small active) | ~4-10GB | Partially | Yes |
| MoE (large active) | 20GB+ | Yes | No |
