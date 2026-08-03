# DeepSeek V4 Flash 0731 Performance

This note records the Release-build performance work for
`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX` on an Apple M3 Ultra with 512 GB of
unified memory. Measurements use the native `deepseek_v4` architecture path,
greedy decoding, and one warm server process. Model loading is excluded from
request wall time.

## Optimization

Two immutable weight conversions were being rebuilt in every decode step:

- Every attention layer dequantized its grouped `wo_a` projection before the
  output `einsum`. The official implementation treats this as an unpacked
  weight, so AFM now evaluates and retains it on first use.
- The unquantized BF16 language-model head was converted to FP32 for every
  token. AFM now evaluates and retains the FP32 weight on first use.

Both caches are enabled by default for DeepSeek V4. They are created after
checkpoint loading and quantization, on the first model forward pass. AFM does
not mutate a loaded MLX model during serving.

Diagnostic and constrained-memory opt-outs:

```bash
VMLX_DSV4_CACHE_WOA=0
VMLX_DSV4_CACHE_LM_HEAD=0
```

The optimized process retained about 4 GB more resident memory than the
cache-disabled control. This tradeoff is intentional for a model whose source
weights already require well over 100 GB of unified memory.

## Release A/B

The principal comparison used the same 256-token request and binary, changing
only the two environment flags above.

| Configuration | Decode | Request wall time | Average GPU | Median GPU |
|---|---:|---:|---:|---:|
| Caches disabled | 8.0 tok/s | 34.04 s | 55.8% | 55.7% |
| Caches enabled | 17.4 tok/s | 15.04 s | 80.4% | 81.4% |

The optimized run peaked at 89.5% GPU utilization and held 1371-1378 MHz with
nominal thermals. A separate three-request stability run produced 17.8, 17.9,
and 18.1 tok/s with identical response-content hashes and no resident-memory
growth. A forced 256-token decode sustained 17.9 tok/s. Multi-turn context
returned the expected answer at 18.5 tok/s.

Prompt processing improved from roughly 63 tok/s in the original path to
100-140 tok/s in the optimized runs.

Final validation after source cleanup used the Release server and the exact
Vontra 0731 checkpoint for two identical 256-token, non-streaming requests.
They completed at 19.08 and 19.13 tok/s (13.42 and 13.38 seconds), with prompt
processing at 134.6 and 133.2 tok/s. Both returned the same 256-token sequence
and `finish_reason: length`. AFM reported 151.9 GiB peak process memory.

With `AFM_PERF=1`, CPU-side detokenization and yield overhead was only
0.07-0.08 ms/token. Asynchronous GPU evaluation accounted for 76.5% of the
instrumented generation loop, so the remaining cost is model execution rather
than HTTP, response parsing, or token streaming.

## Correctness

Focused Release tests cover the official 0731 chat encoding, reasoning-prefix
encoding, routed-prefill geometry, and the scored limited-SwiGLU operation.
Deterministic cache-enabled and cache-disabled model runs produced identical
assistant text.

## Experimental ds4 Kernel Engine

AFM exposes an opt-in MLX kernel selector while keeping the existing MLX path
as the default:

```bash
afm mlx --mlx-kernels native ...
afm mlx --mlx-kernels ds4 ...
```

The equivalent AFMKit MLX provider configuration is `mlxKernels: "native"` or
`mlxKernels: "ds4"`. Selection is propagated through server and single-prompt
execution. Dispatch is based on checkpoint tensor geometry and quantization
metadata, never the repository or model ID. Unsupported shapes and quantizers
remain on the native implementation.

The ds4 option currently fuses selected-expert MXFP4 gate/up projection,
limited SwiGLU, and route scoring for the 0731 decode geometry. The reference
implementation is pinned as the `vendor/ds4` submodule on its
`tp-fast-release` line; AFM's Metal kernel remains checked into the reproducible
MLX patch set and is compiled by MLX at runtime.

Release validation on the exact Vontra 0731 checkpoint confirmed the
`ds4_gate_up_scored_swiglu` stage executed and produced correct text. It is not
the faster engine yet: an identical 64-token run took about 4.69 seconds after
model readiness with ds4 versus 3.65 seconds native, a roughly 29% regression.
The option therefore remains explicitly experimental and native remains the
default. This A/B control is retained for further kernel iteration rather than
presented as a production optimization.

## Remaining Cost

Opt-in synchronized stage profiling after the cache fix attributes decode time
approximately as follows:

- routed MoE: 44.5%
- attention: 21.4%
- mHC, normalization, and residual stages: about 26%

Within routed MoE, gate projection, up projection, activation, and down
projection each cost about 0.32-0.36 ms per layer under synchronized profiling.

The following experiments were rejected because they did not preserve both
correctness and performance:

- concatenating the giant routed gate/up expert banks;
- a custom no-copy dual MXFP4 gate/up Metal kernel;
- whole-model compilation and compiled mHC subgraphs;
- disabling activation QAT;
- parallel gate/up streams;
- caching an FP32 gate weight;
- changing scheduler limits.

Reusing the E4M3-prepared activation across the routed gate and up projections
was also exact, but measured only 17.74 tok/s versus 17.65 tok/s for the control
(about 0.5%, within run-to-run noise). It is not retained as a claimed
optimization. Forcing the existing fused gate/up cache for the complete MXFP4
expert bank regressed decode to 9.46 tok/s and prefill to 3.79 seconds because
the wider gather destroyed weight locality.

The next material optimization is the checkpoint's embedded DSpARK speculative
decoder. The 0731 package contains three `mtp.N` stages, a rank-256 Markov head,
a confidence head, and a five-token draft block. The published minimal
`generate.py` does not execute that path, but DeepSeek's DeepSpec runtime and
the independent ds4 implementation establish the required contract:

1. Capture mean-HC verifier states at layers 40, 41, and 42.
2. Produce a confidence-pruned draft with the embedded DSpARK stages.
3. Verify the anchor plus draft in one ordinary target forward pass.
4. Commit only the matching prefix plus the target's bonus/replacement token.
5. Roll every target cache back to the committed prefix.

This does not change the base model or its quantized weights. DSpARK is an
optional proposer; the ordinary target model remains authoritative. Dispatch
must therefore be based on checkpoint metadata and measured verifier cost, not
on repository/model names. MXFP4, affine, other supported quant modes, and
unquantized models continue through their existing target forward path.

On Apple Silicon, multi-row quantized verification has a hardware- and
quant-dependent cost curve. AFM must calibrate candidate widths in Release and
select the width that maximizes expected committed tokens per round. It must
fall back to ordinary autoregressive decoding when no width beats the measured
one-token path. A fixed five-token policy is not acceptable.

The implementation gates are:

- metadata and tensor-layout validation before allocating DSpARK state;
- one-stage numerical fixtures against the released reference implementation;
- exact greedy output equality against autoregressive generation;
- cache rollback tests for full, partial, and zero draft acceptance;
- Release A/B evidence for each supported quant descriptor;
- automatic fallback when acceptance or measured round cost makes speculation
  slower than ordinary decoding.

Current status: the checkpoint-backed DSpARK path is implemented behind
`AFM_DSPARK=1` and validates cache rollback/replay in synthetic compressed
attention tests. It remains opt-in because the real 0731 MXFP4 Release
benchmark regressed with a fixed four-token draft: 128 generated tokens took
31.09 seconds versus 17.51 seconds for the ordinary autoregressive path on the
same prompt and binary. The batched verifier also produced a semantically
equivalent but token-different continuation on the real quantized model, so it
is not yet safe to enable without per-quant calibration and acceptance checks.

## Profiling Limitation

Xcode 27 Beta 3's `Metal Shader Profile` attachment did not begin inference:
`xctrace` consumed CPU and about 31 GB RSS for several minutes while AFM waited.
The trace was stopped and is not used as performance evidence. Hardware
utilization above comes from 200 ms `mactop` samples taken only during Release
decode.
