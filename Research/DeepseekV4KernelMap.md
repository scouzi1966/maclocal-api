# DeepSeek V4 Flash 0731 Kernel Map

This map tracks the native MLX/Metal implementation against the official
DeepSeek V4 Flash 0731 computation contracts. A path is not accepted merely
because it compiles: it needs Release-mode numerical parity, an A/B throughput
gain, metadata-based dispatch, and an automatic generic fallback.

## Current target

- Hardware: M3 Ultra, 512 GB unified memory
- Model: Vontra DeepSeek V4 Flash 0731 MXFP4 MLX conversion
- Production engine: `--mlx-kernels native`
- Plain-native baseline before validated defaults: 19.0 tok/s
- Validated native defaults: 23.1 tok/s over three deterministic 256-token runs
- Fused layer-tail candidate: 23.5-23.7 tok/s over three deterministic
  256-token runs; response SHA-256
  `5640c41f44fa7566a2b62e757167c8f399635df1c31d88d31d5132021594b03a`
- Compiled attention prefix checkpoint: 25.3-25.4 tok/s
- Architecture-gated 200-operation/100-GB command-buffer policy plus the
  checkpoint-native grouped MXFP8 attention-output projection: 26.66-26.69
  tok/s with the original BF16 output head
- The same path with the explicitly enabled affine-Q8 output head: 28.99-29.01
  tok/s over three deterministic Release runs; response SHA-256 starts with
  `16a6f3491b76`
- Target: 35-37 tok/s without model-ID dispatch or expert-weight duplication

The published canonical `antirez/ds4` M3 Ultra short-prompt results are 35.50
tok/s for its custom Q4_K GGUF and 36.86 tok/s for its asymmetric IQ2/Q2_K
GGUF. They are not measurements of the official MXFP4 checkpoint. DS4 is a
useful scheduling and Metal-kernel reference, but those figures are not an
apples-to-apples throughput baseline for this MXFP4 MLX storage ABI.

## Operation map

| DeepSeek contract | Call site and shapes | Existing MLX path | Optimized path | Fallback | Correctness | Performance | Decision |
|---|---|---|---|---|---|---|---|
| `hc_split_sinkhorn` | Twice per layer. Decode `mixes=[B,1,24]`, four lanes, 20 iterations. | Generic MLX split, sigmoid, softmax, row/column normalization, collapse. | One `MLXFast.metalKernel` computes split/Sinkhorn and collapse; a second kernel expands residual lanes. | `hcSplitSinkhornOps` and generic collapse/expand. Disable with `VMLX_DSV4_FUSED_HC4=0`. | Deterministic M=1 and M>1 parity tests; Release generation hash previously matched. | About +1.4% alone. | Keep enabled for supported HC=4 decode shapes. |
| Router score and top-k | Every routed layer. Decode logits `[B,1,E]`, top-k=6 for the tested checkpoint. | FP32 matmul, `sqrt(softplus)`, optional bias for selection, top-k, gather original scores, normalize and scale. | Metadata-gated fused selector. | Compiled generic selector. Disable with `VMLX_DSV4_FUSED_ROUTER=0`. | Selected experts and weights match the generic reference. | Included in the 23.1 tok/s validated defaults. | Keep enabled for supported decode geometry. |
| Gate/up activation preparation | Same routed activation feeds two independent MXFP4 projections. | Each `QuantizedSwitchLinear` independently prepares its activation. | Native custom pair kernel consumes one activation and separate gate/up expert banks without duplicating weights. | Independent native MLX projections for unsupported metadata. Disable with `VMLX_DSV4_NATIVE_MXFP4=0`. | Synthetic projection tests and deterministic generation parity pass. | Included in the 23.1 tok/s validated defaults. | Keep metadata-gated; never concatenate giant expert banks. |
| Routed gate/up projection | Six routes × `[4096 -> 2048]` in the tested conversion. | Native MLX `gatherQuantizedMM`, MXFP4 group size 32. | Metadata-gated MXFP4 pair-SwiGLU Metal kernel modeled on DS4's exact E2M1 decode contract. | Native MLX primitive for other modes, group sizes, shapes, and quants. | Release synthetic and generation parity pass. | Material part of the current native gain. | Keep guarded native kernel. |
| SwiGLU and route score | Routed gate/up output, score per selected expert. | FP32 clamp, SiLU, multiply, score multiply, cast back. | Fused into the MXFP4 pair projection kernel. | Generic MLX expression. | `testScoredSwiGLUDecodeMatchesReference` passes. | Removes intermediate dispatches and buffers. | Keep for validated metadata only. |
| Routed down projection and reduction | Six `[2048]` routed activations back to hidden size 4096. | Native MLX `gatherQuantizedMM` plus FP32 reduction. | Metadata-gated MXFP4 sum-six Metal kernel. | Native MLX path for unsupported route count, layout, shape, or quant. | Deterministic Release generation parity passes. | Included in the current native gain. | Keep guarded native kernel. |
| Shared expert MLP | One dense MXFP8 expert per layer. | Three native quantized projections plus generic SwiGLU. | Included with HC expansion and routed MoE in the compiled decode layer tail. | Uncompiled native MLX. Disable with `VMLX_DSV4_COMPILE_FFN=0`. | Exact generation parity and focused Release suite pass. | Extending the compiled tail over attention HC improved 23.1 to 23.5-23.7 tok/s. | Keep validated compiled tail. |
| Grouped attention output A projection | Eight independent MXFP8 groups project 4096 attention features to eight rank-1024 blocks. | Dequantize the full `wo_a` bank to BF16, reshape, then use grouped `einsum`. | Reshape the checkpoint's packed weights/scales by group and issue one batched MLX `quantizedMM`; no weight expansion. | Existing dequantized grouped projection for every other mode, group size, bit width, bias layout, or shape. Disable with `VMLX_DSV4_QUANTIZED_GROUPED_WOA=0`. | Promoted and control paths produced the same deterministic output hash; focused Release tests pass. | 28.09-28.25 versus 26.89-27.00 tok/s on the same Q8-head binary under the former scheduler. | Enabled by default only behind exact quantization-metadata checks. |
| Sparse attention | Gathered sliding/compressed/indexed KV positions. | MLX gather and attention implementation with hybrid mutable cache. | No custom kernel yet. | Existing MLX path. | Numerical trace exists for layer zero. | Aggregate profile below MoE cost. | Defer until Metal trace proves material. |
| Command scheduling | Roughly 160 command buffers per generated token in the pre-policy MLX Metal trace. Custom kernels expose a complete expert-bank allocation even when they read six experts, so MLX's byte accounting greatly overstates working bytes. | MLX Ultra defaults of 50 operations/50 MB per command buffer. | DeepSeek V4 on Ultra uses a measured 200-operation limit and a 100-GB byte ceiling so operation count, not backing-allocation accounting, remains the safety boundary. | Defaults remain unchanged for other architectures/hardware; explicit MLX environment limits win. | Three exact deterministic Release hashes; policy tests pass. | Raising only the byte ceiling moved the current Q8/grouped-`wo_a` path from 28.3 to about 29.0 tok/s; 100- and 300-operation sweeps were no better than 200. | Keep architecture/hardware-gated policy; fix MLX custom-kernel byte accounting upstream when practical. |
| Output head | BF16 `[129280,4096]` checkpoint matrix. | FP32-accumulating dense projection. | Optional one-time affine-Q8 cache (`VMLX_DSV4_Q8_LM_HEAD=1`). | Original BF16/FP32 projection remains the default. | Q8 runs are deterministic; the option is not a bitwise output-head replacement and remains explicit. A custom BF16-weight/FP32 Metal matvec matched the observed response but reached only 28.48-28.53 tok/s. | Q8 raises the optimized path to about 29.0 tok/s; the custom BF16 kernel lost to it. | Keep Q8 opt-in; reject the custom BF16 head. |
| Whole decode compile | Token model call plus mutable KV state. | Uncompiled canonical hybrid caches. | Experimental compiled closure. | Uncompiled path. | Canonical mutable cache is unsupported by the attempted whole-model closure; simple-cache experiments were invalid. | Rejected. | Do not enable without explicit cache inputs/outputs. |
| DSpARK M>1 | Proposal/verification shares attention, HC and MoE primitives. | Experimental model support and tests. | Reuse the same metadata-gated primitives with M>1 support. | Normal autoregressive path. | Capability tests exist; parity matrix incomplete. | Not qualified. | Must not fork kernel implementations. |

## Quantized ABI constraints

The official FP4 mathematical contract is FP8 activation with scales per 128
K values, packed E2M1 FP4 weights with scales per 32 K values, FP32
accumulation, and BF16/model-dtype output. The loaded MLX conversion is the
actual storage ABI: nibble order, scale representation, matrix orientation,
expert stride, and any load-time transformation must be read from its tensor
metadata rather than assumed from TileLang.

## Required measurements

Each accepted change must record three warmed Release runs with output hash,
prompt and generation timing, tokens per second, peak memory, and the disable
flag result. Metal traces must compare DeepSeek and Qwen using the same prompt
and output length and report dispatches per token, command-buffer gaps, and the
time split among MoE, mHC, attention, norm/residual, and LM head.

## References

- DeepSeek V4 Flash 0731 `inference/model.py`
- DeepSeek V4 Flash 0731 `inference/kernel.py`
- DeepSeek TileKernels `mhc`, `moe`, and `quant` directories
- MLX Swift `MLXFast.metalKernel`, compile, and native quantized operations
- Canonical DS4 runtime and Metal scheduler: `https://github.com/antirez/ds4`
- `kernelpool/ds4` `tp-fast-release` is retained only as a secondary comparison
  remote; it is not the canonical submodule source.
