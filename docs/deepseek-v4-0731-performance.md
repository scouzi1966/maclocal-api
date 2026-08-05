# DeepSeek V4 Flash 0731 Performance

The authoritative negative-results ledger is
[`deepseek-v4-0731-rejected-experiments.md`](deepseek-v4-0731-rejected-experiments.md).
Read it before starting a new kernel, graph, scheduler, or quantization
experiment so rejected variants are not repeated.

This note records the Release-build performance work for
`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX` on an Apple M3 Ultra with 512 GB of
unified memory. Measurements use the native `deepseek_v4` architecture path,
greedy decoding, and one warm server process. Model loading is excluded from
request wall time.

## Reusable Official-Checkpoint Conversion

The Release CLI includes `afm mlx-convert` for converting the official
DeepSeek V4 checkpoint to AFM's native MLX checkpoint layout. It reuses the
runtime's Swift `DeepseekV4Model.sanitize(weights:)` implementation instead of
duplicating architecture mapping in Python. Conversion is shard-streaming,
atomic per shard, and resumable through `.afm-mlx-conversion.json` in the
output directory.

```bash
afm mlx-convert \
  --source /path/to/DeepSeek-V4-Flash-0731 \
  --output /path/to/DeepSeek-V4-Flash-0731-AFM-MLX
```

The converted config records `afm_native_checkpoint: true` and explicit mixed
MXFP4/MXFP8 quantization metadata. That marker tells the loader not to run the
official-checkpoint sanitizer again. It changes model loading and deployment,
not the steady-state decode kernels; the conversion is therefore not expected
to close the remaining generation-throughput gap by itself.

An alternating Release one-shot A/B on the same machine measured 37.73 and
36.07 seconds directly from the official source checkpoint versus 26.50 and
26.56 seconds from the converted checkpoint. All four runs returned the exact
requested text. The conversion removed about 10 seconds, or 27%, from this
load-plus-short-generation path. An immediate post-conversion run was excluded
because its filesystem cache state was not comparable. These measurements are
not a claim of a steady-state decode improvement.

## Canonical DwarfStar Runtime Boundary

The canonical `antirez/ds4` Release server provides a hardware control for the
35-37 tok/s objective. On the same M3 Ultra, its published Q2 GGUF produced
38.11, 38.19, and 38.14 tok/s for three forced 256-token counting runs. A
natural-language Rayleigh-scattering prompt produced 37.79, 37.68, and 37.70
tok/s with byte-identical output across all three runs. Model loading was
excluded and no speculative decoder was active.

Artifacts:

- `/Volumes/edata/afm-captures/deepseek-v4-ds4-q2-format-boundary-20260804/20260804-195812`
- `/Volumes/edata/afm-captures/deepseek-v4-ds4-q2-natural-20260804/20260804-195900`

This proves the throughput target is real and not a counting-prompt artifact.
DwarfStar's Q2 package uses IQ2_XXS routed gate/up weights and Q2_K routed down
weights, so a second control used DwarfStar's published MXFP4 GGUF. The MXFP4
control reached 35.87, 35.96, and 36.20 tok/s for counting and 35.98, 36.01,
and 36.01 tok/s for natural language. All runs within each workload produced
byte-identical output.

Artifacts:

- `/Volumes/edata/afm-captures/deepseek-v4-ds4-mxfp4-format-boundary-20260804/20260804-202506`
- `/Volumes/edata/afm-captures/deepseek-v4-ds4-mxfp4-natural-20260804/20260804-202543`

The MXFP4 result proves that routed-expert MXFP4 can reach the target, but it
does not isolate the full gap to runtime alone. DwarfStar's package uses MXFP4
routed experts with Q8 attention, shared-expert, and output tensors, whereas
AFM's official conversion uses MXFP8 for most of those non-routed projections.
That difference selects different kernels and remains a controlled variable.
AFM's reusable `mlx-convert --profile dwarfstar-q8` control converted eligible
attention and shared-expert projections to generic MLX affine Q8 while retaining
the routed MXFP4 experts. With the same Release binary, staged MoE path, runtime
Q8 output head, prompt, and output hash, that checkpoint reached 24.37 tok/s
versus 29.97 tok/s for the MXFP8 checkpoint. Generic MLX affine Q8 is therefore
not the source of DwarfStar's advantage; its custom Q8 runtime remains a distinct
variable. Artifacts are in
`/Volumes/edata/afm-captures/deepseek-v4-dwarfstar-q8-control-benchmark/20260804-215709`
and
`/Volumes/edata/afm-captures/deepseek-v4-mxfp8-control-contemporary-baseline/20260804-215816`.

The paired `mlx-convert --profile dwarfstar-symmetric-q8` control stores the
same dense subset as signed symmetric Q8 blocks. The initial 28.32 tok/s run
was incorrectly compared with a promoted checkpoint using a different staged-
MoE policy. With staging synchronized it reached 29.15, 29.07, and 29.08 tok/s.
A later same-binary comparison measured 28.64 tok/s for symmetric Q8 against
28.18 tok/s for the promoted checkpoint. The typed C++ symmetric-Q8 primitive
then reduced Swift/MLX graph construction from 6.387 to 5.541 ms/token, but GPU
evaluation variance left end-to-end throughput effectively unchanged. The
profile remains an explicit research control because it has not produced a
material win, not because it is 6% slower. Artifacts are in
`/Volumes/edata/afm-captures/deepseek-v4-dwarfstar-symmetric-q8-staged-20260805/20260805-000619`,
`/Volumes/edata/afm-captures/deepseek-v4-symmetric-q8-cpp-primitive-20260805/20260805-002714`,
and
`/Volumes/edata/afm-captures/deepseek-v4-symmetric-q8-cpp-perf-20260805/20260805-002835`.

The next graph-boundary experiment kept the existing FP32 router GEMV but moved
top-6 route selection into the staged routed-MoE primitive. An activation-
asserted, same-binary comparison measured 29.49 tok/s against 30.10 tok/s for
the staged-MoE control, with identical output hashes. The 2.1% difference is
inside the agreed 3% neutral band, so selector fusion is not promoted. These
artifacts are the first benchmark pair to embed the Release binary SHA-256,
effective optimization environment, relevant source hashes, and required
runtime markers:

- `/Volumes/edata/afm-captures/deepseek-v4-staged-selector/20260804-223535`
- `/Volumes/edata/afm-captures/deepseek-v4-staged-selector-control/20260804-223623`

With GPU busy profiling enabled, DwarfStar committed approximately two command buffers
per generated token. AFM's Xcode trace measured roughly thirteen. Copying
DwarfStar's isolated MXFP4 arithmetic into MLX did not improve throughput; the
next native optimization must reduce graph/runtime boundaries or fuse broader
DeepSeek layer work while preserving MLX cache semantics.

The reusable benchmark harness is
`Scripts/benchmarks/benchmark_deepseek_v4_afm_ds4.py`. It launches DwarfStar
from its own source directory so runtime Metal sources resolve correctly, and
accepts repeatable `--ds4-env KEY=VALUE` diagnostics. Large models and captures
must stay on `/Volumes/edata`, never `/tmp` or the internal disk.

## Optimization

A runtime MXFP8 output-head experiment measured 29.3-29.5 server tok/s and
28.64-28.77 wall-clock tok/s, effectively identical to the retained affine-Q8
head baseline. It was removed rather than adding another unsupported runtime
switch.

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

AFM also applies measured MLX command-buffer limits of 200 operations and
400 MB for the `deepseek_v4` architecture on Ultra-class Apple Silicon. The
policy runs before the first MLX/Metal device access, is based on architecture
metadata and processor class rather than model ID, and leaves all other
architectures and hardware unchanged. Either explicit MLX environment override
disables the automatic policy:

```bash
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 afm mlx ...
```

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
limited SwiGLU, and route scoring for the 0731 decode geometry. The canonical
reference implementation is pinned from `antirez/ds4` `main` as the
`vendor/ds4` submodule. `kernelpool/ds4` `tp-fast-release` is retained only as
a secondary comparison remote. AFM's Metal kernel remains checked into the
reproducible MLX patch set and is compiled by MLX at runtime.

Release validation on the exact Vontra 0731 checkpoint confirmed the
`ds4_gate_up_scored_swiglu` stage executed and produced correct text. It is not
the faster engine yet: an identical 64-token run took about 4.69 seconds after
model readiness with ds4 versus 3.65 seconds native, a roughly 29% regression.
The option therefore remains explicitly experimental and native remains the
default. This A/B control is retained for further kernel iteration rather than
presented as a production optimization. DS4's published 35-37 tok/s M3 Ultra
measurements use its DeepSeek-specific GGUF runtime, quant layouts, and
long-lived Metal command batches. They are a scheduling target, not a direct
throughput comparison with the Vontra MXFP4 MLX checkpoint.

## Validated Native Decode Defaults

The native engine now enables the numerically validated DeepSeek decode paths
without hidden environment variables:

- fused metadata-gated MXFP4 routed gate/up/SwiGLU and sum-six down projection;
- fused router selection;
- compiled attention projection, mHC, and FFN layer tails.

Each feature retains an explicit `=0`/`false` diagnostic opt-out. Unsupported
quantization metadata, route geometry, or tensor shapes fall back to generic
MLX operations. Dispatch never depends on a model or repository ID.

Three warmed Release runs of an identical 256-token request measured 23.1
tok/s before extending the compiled FFN tail. Including the attention
hyper-connection residual expansion in that tail measured 23.5, 23.7, and
23.7 tok/s. All six runs produced byte-identical assistant content with SHA-256
`5640c41f44fa7566a2b62e757167c8f399635df1c31d88d31d5132021594b03a`.
Focused Release tests pass 10/10.

Compiling the stateless decode attention prefix (Q/KV projections,
normalization, RoPE, and KV activation QAT) while keeping cache mutation and
SDPA outside the compiled graph improved three subsequent Release runs to
25.4, 25.3, and 25.4 tok/s. The output SHA-256 remained identical and the
focused Release suite again passed 10/10. This is a 7-8% gain over the prior
23.5-23.7 tok/s checkpoint.

Coarsening MLX command-buffer limits to 200 operations and 400 MB then produced
26.42, 26.47, and 26.46 tok/s in a clean Release process without manually set
MLX environment variables. The corresponding 50/50 control measured
25.13-25.31 tok/s. All three automatic-policy runs returned identical text with
SHA-256
`54b9989d8acc60cff5f9ea1025853f8b2a65ff6ae1cde21de361de86d808b7da`.
The focused architecture and policy suite passed 16/16.

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
- effectively unbounded scheduler limits.

Scheduler sweeps plateaued at approximately 200/400 through 1000/1,000,000;
removing practical command-buffer limits regressed decode to 18.43 tok/s. A
fused HC-collapse plus RMSNorm candidate remained neutral at 26.46-26.47 tok/s
with the accepted scheduler policy and stays disabled.

Extending the compiled layer tail backward through the attention output
projection was also rejected. It preserved exact output but regressed three
Release runs to 23.6, 22.4, and 22.5 tok/s, indicating that this larger graph
increased repeated graph execution overhead rather than reducing scheduling
cost.

Reusing the E4M3-prepared activation across the routed gate and up projections
was also exact, but measured only 17.74 tok/s versus 17.65 tok/s for the control
(about 0.5%, within run-to-run noise). It is not retained as a claimed
optimization. Forcing the existing fused gate/up cache for the complete MXFP4
expert bank regressed decode to 9.46 tok/s and prefill to 3.79 seconds because
the wider gather destroyed weight locality.

The pre-policy Metal trace shows roughly 160 command-buffer submissions per
generated token, low compute occupancy, and CPU-to-GPU scheduling gaps. The
accepted 200/400 policy removes part of that overhead, but its 5% gain is far
short of the target. The canonical `antirez/ds4` runtime batches a token's
operations into long-lived command buffers and synchronizes only where
routing/readback requires it. AFM's next material native optimization is
therefore broader graph/command-buffer coarsening while retaining explicit
mutable hybrid-cache boundaries.

The checkpoint's embedded DSpARK speculative decoder is a separate possible
optimization. The 0731 package contains three `mtp.N` stages, a rank-256 Markov
head, a confidence head, and a five-token draft block. The published minimal
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
