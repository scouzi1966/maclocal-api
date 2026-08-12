# DeepSeek V4 Flash 0731 Rejected Experiments

This is the durable negative-results ledger for AFM's DeepSeek V4 Flash 0731
performance work. Read this file before changing the native MLX/Metal path.
An experiment listed here must not be repeated without a materially different
hypothesis, implementation, or measurement method.

The production target is metadata-gated `--mlx-kernels native` on an M3 Ultra
with 512 GB unified memory. Performance measurements are Release-only, use a
warm server, exclude model loading, and use the server's generation timing.
Artifacts live under `Research/benchmarks/deepseek-v4-*` and large captures
live under `/Volumes/edata/afm-captures`; never put captures in `/tmp` or on the
internal disk.

## Required Evidence for a Negative Finding

A result belongs in this ledger only when all of the following are recorded:

1. the Release binary path and SHA-256;
2. the model path and checkpoint format;
3. the exact prompt, sampling settings, token count, and warmup policy;
4. the effective `VMLX_DSV4_*` environment and an activation marker proving
   that the experimental path ran;
5. an interleaved or same-session control using the same binary;
6. at least three measured runs for a throughput conclusion;
7. the complete generated-text hash, not only a prefix comparison; and
8. a persistent artifact directory under `/Volumes/edata`.

Missing activation markers, stale binaries, Debug builds, unmatched controls,
and failed/skipped tests are invalid experiments rather than negative
findings. Throughput differences below 3% are recorded as neutral unless a
larger repeated sample establishes statistical separation. A negative result
must state which hypothesis it closes and which broader hypothesis remains
open.

## Current Baseline and Scope

- Converted official checkpoint, affine-Q8 head disabled: approximately
  27.7-28.1 tok/s for a forced 256-token autoregressive decode.
- Vontra MXFP4 checkpoint: approximately 27.9-28.5 tok/s under equivalent
  conditions.
- The converted official and Vontra checkpoints use the same steady-state
  native MoE kernel path. Conversion improves loading but did not improve
  steady-state decode.
- The optional affine-Q8 output head previously raised the complete optimized
  path to approximately 29.0 tok/s. That gain is outside the routed MoE kernel
  work discussed here.
- DSpARK can exceed 47 tok/s on a fully accepted counting workload, but its
  speed depends on draft acceptance and is not the autoregressive baseline.
- Canonical DwarfStar Q2 measured 38.11-38.19 tok/s for forced counting and
  37.68-37.79 tok/s for natural language on this machine. More importantly,
  canonical DwarfStar MXFP4 measured 35.87-36.20 tok/s for counting and
  35.98-36.01 tok/s for natural language. This proves routed-expert MXFP4 can
  close the target, but it is not a whole-checkpoint same-format comparison:
  DwarfStar uses Q8 attention, shared-expert, and output tensors while AFM uses
  MXFP8 for most of those roles. Runtime/dispatch remains a major difference,
  but the non-routed kernel mix is still an open controlled variable. DwarfStar used about two
  Metal command buffers per token. A direct MLX commit counter on the current
  symmetric-Q8/staged-MoE Release path measured 33.84 buffers/token; the older
  approximately 13-buffer trace was a different graph/policy checkpoint and
  must not be cited as the current count.
- A same-binary, exact-output Q8_0 checkpoint control reached 28.50, 28.48,
  and 28.38 tok/s. The symmetric-Q8 control reached 28.50, 28.54, and 28.52
  tok/s, while canonical DwarfStar reached 34.84, 35.01, and 35.04 tok/s.
  This closes checkpoint Q8 block representation and isolated Q8 arithmetic
  as explanations for the remaining 22.9% wall-throughput gap.

## Native Routed-MoE Experiments

| Experiment | Result | Decision and reason |
|---|---:|---|
| Concatenate complete gate/up expert banks | Severe regression | Duplicates a giant expert bank and destroys selected-expert locality. Never use whole-bank concatenation to save one dispatch. |
| Forced fused gate/up cache for complete MXFP4 bank | 9.46 tok/s; prefill 3.79 s | Rejected for the same locality and memory-traffic failure. |
| Early no-copy dual MXFP4 gate/up kernel | No promoted gain | The first custom implementation did not beat native MLX. The later metadata-gated pair kernel is a different implementation and is retained. |
| Parallel gate and up MLX streams | No promoted gain | Dispatch concurrency did not offset duplicated activation preparation and memory pressure. |
| Reuse E4M3-prepared activation between generic gate/up projections | 17.74 vs 17.65 tok/s control | Exact but within noise. Do not claim or preserve it as an optimization. |
| Disable routed activation QAT | About +0.4 tok/s in the current kernel study | Too small and changes the quantization contract. Diagnostic only, not a production optimization. |
| Route sorting / selected-expert reordering | Regression | Sorting overhead and changed access order did not improve effective locality. Keep checkpoint-selected route order. |
| Emit six down-projection route rows, then reduce separately | 28.76-28.78 vs about 29.0 tok/s | Exact but slower than the serial sum-six kernel. Candidate removed. |
| Cooperative down kernel: one SIMD group per route, threadgroup reduction | 27.7-27.8 tok/s vs 27.7-27.8 control | Correct and neutral. Route serialization was not the limiter; total work and reduced residency dominated. Tested with `VMLX_DSV4_COOPERATIVE_DOWN=1`; do not promote. Large results: `/Volumes/edata/afm-captures/deepseek-v4-cooperative-down-20260804`. |
| Generalize custom gate/up and down kernels to DSpARK multi-token verify | 33.04-33.42 vs 34.03-34.11 tok/s | Exact but slower. Keep generic MLX batched quantized matmul for verifier rows and specialize custom kernels for one-token decode. |
| `ROWS_PER_SIMD` sweep: 1, 2, 4 | No material gain over 2 | Launch-shape tuning did not close the gap. Retain 2 unless kernel arithmetic changes enough to justify a new occupancy sweep. |
| SIMD groups per threadgroup sweep: 1, 2, 4, 8 | No material gain over 2 | Threadgroup occupancy alone was not limiting. |
| Wide-lane and vector-load variants | Neutral or slower | Packing more scalar work per lane did not reduce effective kernel time. See `deepseek-v4-vector-loads-on-20260804` and `deepseek-v4-wide-lane-*`. |
| Replace constant E2M1 LUT with MLX-style half bit reconstruction | 27.7 tok/s over four runs | Numerically correct and neutral against the 27.7-27.8 baseline. Constant-table indexing is not the limiter. Large results: `/Volumes/edata/afm-captures/deepseek-v4-fp4-bitdecode-20260804`. Reverted. |
| Perform E2M1 multiplication in half, then accumulate products in FP32 | Control 27.7-27.8; gate/up 27.9-28.0; down 27.9; both 28.0 tok/s | Exact generated-text hash in the forced 256-token benchmark, but only about a 1.1% gain and it changes multiplication precision. Retained only as the opt-in `VMLX_DSV4_HALF_MULTIPLY` diagnostic; do not make it the default without broader numerical and quality validation. Large results: `/Volumes/edata/afm-captures/deepseek-v4-half-multiply-20260804`. |
| Load one E8M0 scale per adjacent lane pair and broadcast it with `simd_shuffle` | Control 27.80; gate/up 27.49; down 27.65; both 27.31 tok/s | Exact output hash but consistently slower. Scale reads are already cache-cheap; the shuffle and control overhead costs more than the duplicate load. Large results: `/Volumes/edata/afm-captures/deepseek-v4-pair-scale-load-20260804`. Reverted. |
| Stage common activations in threadgroup memory with eight SIMD groups | Control 28.05; gate/up 27.46; down 26.78; both 26.11 tok/s | Exact output hash but slower, especially for the 24 KB six-route down staging buffer. Barriers and reduced residency outweigh cached activation reads. The earlier SIMD-group sweep did not stage data; this experiment closes that separate hypothesis. Large results: `/Volumes/edata/afm-captures/deepseek-v4-shared-activation-20260804`. Reverted. |
| Hoist per-row weight and scale base addresses into fixed local arrays | 27.36, 27.37, 27.41 tok/s; mean 27.38 vs 28.05 control | Exact output hash, but about 2.4% slower. The additional local arrays likely increased register pressure and reduced residency; the compiler already handles the simple loop-invariant arithmetic more effectively. Large results: `/Volumes/edata/afm-captures/deepseek-v4-hoisted-row-bases-20260804`. Reverted. |
| Apply E8M0 scales with `metal::ldexp` instead of exponent reconstruction and multiply | Control 27.41; gate/up 27.49; down 27.40; both 27.43 tok/s | Exact output hash but within run noise. Metal's ordinary multiply path is already efficient; replacing it with exponent scaling does not change the bottleneck. Large results: `/Volumes/edata/afm-captures/deepseek-v4-ldexp-scale-20260804`. Reverted. |
| Stage only the 16-entry E2M1 decode LUT in threadgroup memory, following DS4's 64-byte LUT pattern | Control 27.12; gate/up 27.30; down 27.08; both 27.19 tok/s | Exact output hash. Gate/up improved only 0.67%, down regressed 0.15%, and the combined path improved 0.27%; these are not material production gains. This is distinct from the rejected 24 KB activation-staging experiment and rules out constant-LUT latency as a meaningful bottleneck. Large results: `/Volumes/edata/afm-captures/deepseek-v4-threadgroup-lut-20260804`. Reverted. |
| Repack separate MLX MXFP4 weights and scales into DwarfStar's interleaved 17-byte block, then port its fused gate/up and sum-six down arithmetic | 25.01 tok/s control vs 23.22 tok/s DwarfStar layout in an exact-hash 32-token smoke | Correct but about 7.2% slower. DwarfStar's unaligned 17-byte stride and runtime-specific memory access pattern do not map efficiently to the MLX custom-kernel path; the research prototype also retained duplicate packed storage because registered MLX module parameters cannot be mutated directly. This rejects the interleaved block layout, not DwarfStar kernels whose data-layout and graph-boundary assumptions match AFM. Large results: `/Volumes/edata/afm-captures/deepseek-v4-dwarfstar-blocks-20260804`. Reverted. |
| Persist the DwarfStar-style interleaved MXFP4 layout in the converted checkpoint and consume it directly from the typed staged primitive | Interleaved DwarfStar lanes 26.85, 26.83, 26.83 tok/s; legacy lanes 26.52, 26.48, 26.46; same-binary split-layout control 26.87, 26.72, 26.80 | All valid runs used `--prefill-step-size 1` and produced the exact `16a6f3491b76` hash. The persistent layout is neutral, not a material optimization. Without one-token prefill, generic `gatherQuantizedMM` reads the packed tensor as ordinary split MXFP4 and corrupts the prompt state; a one-token decode marker alone therefore does not prove end-to-end layout correctness. Direct real-weight Metal fixtures showed exact gate/up and down arithmetic, and all 1,799 unchanged source tensors were byte-verified. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-interleaved-lanes-valid-prefill1-20260805/20260805-014516`, `/Volumes/edata/afm-captures/deepseek-v4-interleaved-legacy-valid-prefill1-20260805/20260805-014615`, and `/Volumes/edata/afm-captures/deepseek-v4-standard-control-prefill1-20260805/20260805-014703`. Do not promote. |
| Store the fused routed SwiGLU intermediate as FP16 instead of inheriting the model's BF16 dtype, following DwarfStar's FP16 intermediate path | 27.44 tok/s control vs 27.41 tok/s FP16 over three forced 256-token runs | Exact output hash but neutral to slightly slower. Both formats use the same 16-bit memory footprint, and changing the intermediate dtype does not reduce the dominant MXFP4 projection work. Large results: `/Volumes/edata/afm-captures/deepseek-v4-fp16-routed-20260804`. Reverted. |
| Pre-cast the shared gate/up activation from BF16 to FP32 once, following DwarfStar's FP32 matvec input, to avoid repeated scalar conversion in every output-row work item | 27.48 tok/s control vs 27.30 tok/s FP32 over three forced 256-token runs | Exact output hash but about 0.7% slower. The additional cast and doubled activation traffic outweigh any reduction in per-load conversion instructions; those conversions are not the gate/up limiter. Large results: `/Volumes/edata/afm-captures/deepseek-v4-fp32-gate-input-20260804`. Reverted. |
| Repack each row into aligned 272-byte superblocks containing 16 E8M0 scales followed by 16 aligned 16-byte MXFP4 groups, preserving DwarfStar-style scale locality without its unaligned 17-byte stride | Aligned 26.86, 26.81, 26.83 wall tok/s and 29.9-30.0 server tok/s; same-binary split control 26.89, 26.81, 26.83 wall tok/s and 29.9-30.0 server tok/s | The full 165 GB converted package and aligned Metal gate/up and down kernels produced exact 64-token (`d2c0457690d2`) and 256-token (`16a6f3491b76`) hashes, but no measurable end-to-end gain. The 6.25% routed-weight expansion cancels any focused load-locality advantage in the complete model. Both packages used one-token prefill because generic batched quantized matmul cannot consume the packed ABI. Keep the converter/kernel only as an opt-in research path; do not promote or distribute this checkpoint layout. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-aligned-paired-20260805/20260805-021709` and `/Volumes/edata/afm-captures/deepseek-v4-standard-control-aligned-pair-20260805/20260805-021813`. |
| Port DwarfStar's byte-oriented low/high-nibble vector accumulation schedule while retaining AFM's split aligned weights and scales | Control 27.23; gate/up 27.22; down 27.29; both 27.18 tok/s over three forced 256-token runs | Exact output hash (`16a6f3491b76`) in every run. The focused rotating microbenchmark was also effectively neutral, and the end-to-end down-only result was only about +0.2%, below run noise; enabling both paths regressed. DwarfStar's arithmetic schedule is not the missing gain when isolated from its broader runtime. Large results: `/Volumes/edata/afm-captures/deepseek-v4-dwarfstar-arithmetic-20260804`. Reverted. |
| Pack gate and up SIMD reductions into one `float2` `simd_sum` per output row | 27.26 tok/s vs 27.23 immediate control over three forced 256-token runs | Exact output hash (`16a6f3491b76`) but only about +0.1%, within run noise. Independent scalar reductions are not a material part of the routed gate/up cost. Large results: `/Volumes/edata/afm-captures/deepseek-v4-paired-reduction-20260804`. Reverted. |
| Force full unrolling of the fixed routed-MXFP4 row loops, matching DwarfStar's `FOR_UNROLL` macro | 27.12 tok/s with 2 SIMD groups; 26.85 tok/s with DwarfStar-like 8 SIMD groups vs 27.23 control | Exact output hash (`16a6f3491b76`) but slower in both launch shapes. The Metal JIT's default treatment of these template-bounded loops is better than forced expansion; the 8-group interaction further reduces effective residency. Large results: `/Volumes/edata/afm-captures/deepseek-v4-explicit-row-unroll-20260804`. Reverted. |
| Physically pair gate/up MXFP4 words and E8M0 scales per row/group so both projections read one aligned region without duplicating weights | Rotating 2,000-sample microbenchmark: split 0.3606 ms median / 0.3785 mean; paired 0.3587 median / 0.3795 mean | Bit-exact, but the paired mean was 0.27% slower and the median difference was noise. Separate gate/up buffers are not causing a material transaction penalty, so changing the persistent checkpoint ABI would add complexity without throughput benefit. Harness: `Research/test-runners/benchmark_mxfp4_gate_up_layout.py`; large results: `/Volumes/edata/afm-captures/deepseek-v4-paired-gate-up-layout-20260804.txt`. Rejected before production integration. |
| Partially unroll the routed gate/up K-group loop four ways to expose independent weight loads | Focused gate/up mean improved about 1.4%, but five end-to-end runs regressed to 26.73-26.77 tok/s vs 27.20-27.26 control | Bit-exact output hash (`16a6f3491b76`). The synthetic kernel's small gain did not survive the full 43-layer graph, consistent with extra register pressure reducing effective residency. Large results: `/Volumes/edata/afm-captures/deepseek-v4-gate-up-unroll4-20260804`; focused harness: `/Volumes/edata/afm-captures/deepseek-v4-gate-up-partial-unroll-20260804.txt`. Reverted. |
| Use three output rows per SIMD as a midpoint between the 2-row baseline and register-heavy 4-row launch | Gate/up 27.52 vs 27.53 control; down 26.90 vs 27.53 control | Bit-exact output hash (`16a6f3491b76`). Gate/up was neutral despite a 7.3% focused-kernel mean improvement; the six-route down path regressed about 2.3%. Synthetic row reuse does not predict full-model residency and scheduling. Large results: `/Volumes/edata/afm-captures/deepseek-v4-gate-up-rows3-20260804` and `/Volumes/edata/afm-captures/deepseek-v4-down-rows3-20260804`. Reverted. |
| Replace the E4M3 activation-QAT binary search and dynamic `exp2` reconstruction with a direct IEEE-754 exponent/mantissa quantizer | 27.44 vs 27.45 tok/s control over three forced 256-token runs | Bit-exact output hash (`16a6f3491b76`) but neutral. The QAT conversion arithmetic is hidden by other GPU work and is not the routed-MoE gap. Large results: `/Volumes/edata/afm-captures/deepseek-v4-fast-e4m3-control-20260804` and `/Volumes/edata/afm-captures/deepseek-v4-fast-e4m3-direct-20260804`. Reverted. |
| Hide immutable expert banks inside the staged primitive instead of exposing them as graph inputs | 28.20, 28.17, 28.21 tok/s vs prior 28.34, 28.32, 28.29 | Bit-exact output hash (`16a6f3491b76`) and statistically neutral. The change adds graph-lifetime complexity without a material gain, so the validated explicit nine-input primitive remains the control. Large results: `/Volumes/edata/afm-captures/deepseek-v4-staged-moe-owned-weights/20260804-211604`. Reverted. |
| Combine the single staged primitive with DwarfStar's byte/nibble accumulation order on AFM's split MXFP4 layout | Mean 28.27 tok/s vs 28.23 control | Bit-exact output hash (`16a6f3491b76`) and neutral. DwarfStar's arithmetic does not become material when graph-boundary overhead is removed; its advantage remains in broader runtime execution. Large results: `/Volumes/edata/afm-captures/deepseek-v4-staged-dwarfstar-combined/20260804-212713` and matching control `deepseek-v4-staged-dwarfstar-combined-control/20260804-212622`. Reverted. |
| Enable Metal fast math only for the staged DeepSeek V4 JIT library, matching DwarfStar's default compile mode | Mean 28.21 tok/s vs 28.29 control | Bit-exact output hash (`16a6f3491b76`) and neutral. Strict JIT math is not the throughput gap for these projection kernels. Large results: `/Volumes/edata/afm-captures/deepseek-v4-staged-fast-math/20260804-213155` and matching control `deepseek-v4-staged-fast-math-control/20260804-213107`. Reverted. |
| Fuse the FP32 router's top-6 selection into the staged routed-MoE primitive while retaining the existing FP32 router GEMV | 29.49, 29.49, 29.47 tok/s; mean 29.49 vs same-binary control 30.15, 30.02, 30.13; mean 30.10 | Exact output hash (`16a6f3491b76`) but 2.1% slower, inside the agreed 3% neutral band. This run asserted `[DSV4Path] staged-selector active` and captured the Release binary hash, effective environment, and source hashes. Route selection is not a material graph-boundary bottleneck. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-staged-selector/20260804-223535` and `/Volumes/edata/afm-captures/deepseek-v4-staged-selector-control/20260804-223623`. Do not promote. |
| Pre-cast the staged routed-MoE activation from BF16 to FP32 once before the typed primitive | 26.73, 26.68, 26.67 wall tok/s; mean 26.69 vs same-binary control 26.80, 26.74, 26.77; mean 26.77 | The `[DSV4Path] routed-fp32-input active` marker and exact `16a6f3491b76` hash prove that the experimental Release path executed. The 0.3% regression is noise-level; repeated BF16-to-FP32 conversion in selected-expert dot products is not a material limiter. Reverted. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-routed-fp32-input-20260805/20260805-024520` and `/Volumes/edata/afm-captures/deepseek-v4-routed-fp32-input-control-20260805/20260805-024616`. |

The synchronized profile attributes roughly 44.5% of decode to routed MoE.
Within it, fused gate/up and the six-route down projection remain the primary
optimization target. The cooperative-down result specifically rules out the
simple hypothesis that serial route traversal is leaving most of the GPU idle.

## Graph, Compilation, and Scheduling Experiments

| Experiment | Result | Decision and reason |
|---|---:|---|
| Whole-model compile with mutable hybrid caches | Invalid/unsupported | The cache mutation contract was not represented as explicit closure I/O. Do not retry without a cache-functionalization design. |
| Whole-model or broad unsafe compile variants | Neutral or invalid | Broad compile boundaries did not improve the validated mutable-cache path. |
| Compiled mHC-only subgraphs | No promoted gain | Dispatch reduction was insufficient. |
| Fuse decode HC=4 split/collapse with the immediately following RMSNorm using the retained DwarfStar-style Metal kernel (`VMLX_DSV4_FUSED_HC_NORM=1`) | 27.81, 27.77, 27.87 tok/s; mean 27.82 vs synchronized same-source control mean 27.77 | Exact output hash (`16a6f3491b76`) in all runs. The roughly 0.2% difference is noise and far below the agreed 3% threshold. Keep opt-in only. Artifact: `/Volumes/edata/afm-captures/deepseek-v4-fused-hc-norm-current-20260805/20260805-073351`. |
| Extend compiled layer tail backward through attention output | 23.6, 22.4, 22.5 tok/s | Exact but much slower than the smaller accepted compile boundary. Repeated graph execution outweighed dispatch savings. |
| Fused HC collapse plus RMSNorm | 26.46-26.47 tok/s | Neutral under the accepted scheduler policy; stays disabled. |
| QKV fusion, combined attention/HC prefixes, and larger fused attention tails | No promoted gain | Retain only the separately validated compiled attention prefix and FFN tail. Artifact directories preserve the controls. |
| Scheduler 50 ops / 50 MB | 25.13-25.31 tok/s at that checkpoint | Slower control. |
| Scheduler 200 ops / high byte ceiling | Best measured policy | Retained. Current byte ceiling is intentionally high because custom kernels expose the whole expert allocation to MLX accounting while touching six experts. |
| Scheduler 300+, 400/800, 1000/1,000,000 | Plateau or regression | No gain over the retained 200-op policy. |
| Scheduler 200 ops with effectively unlimited byte accounting | 26.84, 26.76, 26.79 wall tok/s; mean 26.80 | Exact output and neutral to the retained scheduler. MLX's resource-size accounting is not forcing the current throughput limit. Artifact: `/Volumes/edata/afm-captures/deepseek-v4-current-scheduler-200-highmb-20260805/20260805-023844`. |
| Account only six selected expert slices when binding the 256-expert routed banks | 28.15, 28.18, 28.20 wall tok/s; mean 28.18 vs same-binary control 28.18, 28.19, 28.22; mean 28.20 | Exact output hash (`16a6f3491b76`) and neutral throughput. Accounted bytes fell from 1.53 TB to 177 GB for the profiled request, but command buffers increased from 1,135 to 1,170, proving another commit limit dominates. Reverted. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-selected-accounting-control-20260805/20260805-100755`, `/Volumes/edata/afm-captures/deepseek-v4-selected-accounting-enabled-20260805/20260805-100844`, and `/Volumes/edata/afm-captures/deepseek-v4-selected-accounting-command-buffer-20260805/20260805-100937`. |
| Put HC collapse/RMSNorm/router, routed MXFP4 experts, shared symmetric-Q8 expert, and HC expansion behind one typed MLX primitive | 16.63 tok/s vs same-binary control 28.59 tok/s for the 64-token smoke pair | Exact output hash (`d2c0457690d2`) and required `[DSV4Path] fused-hc-q8-tail active` marker, but 41.8% slower. The proof serialized collapse/RMSNorm and the 256-row router inside one 256-thread threadgroup; eliminating graph boundaries cannot compensate for destroying the optimized parallel dense path. Do not retry this single-threadgroup decomposition. A future broad primitive must retain parallel HC/router kernels and optimize command encoding independently. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-fused-hc-q8-tail-smoke-20260805/20260805-105407` and `/Volumes/edata/afm-captures/deepseek-v4-fused-hc-q8-tail-control-20260805/20260805-105507`. |
| Scheduler 1000 ops with effectively unlimited byte accounting | 26.21, 26.19, 26.13 wall tok/s; mean 26.18 | Exact output and about 2.4% slower than the paired current control. Larger command buffers do not improve the current graph. Artifact: `/Volumes/edata/afm-captures/deepseek-v4-current-scheduler-1000-20260805/20260805-023716`. |
| Effectively unbounded buffers | 18.43 tok/s | Severe regression. Never infer that fewer command buffers always means faster execution. |
| Active-task and launch-rank sweeps | No promoted gain | CPU submission was not the remaining dominant limiter after accepted scheduling changes. |

The older Xcode 27 System Trace measured approximately 86% GPU-active time and
about 13 command buffers per token on its then-current graph, with
sub-microsecond gaps. A fresh Beta 3 trace on the current graph stalled before
inference and produced no usable events. Low-overhead counters added at MLX's
Metal commit boundary measured 33.84 buffers and 4,586.65 encoded operations
per steady-state token by subtracting 64-token lifetime totals from 256-token
totals. Scheduler A/Bs above show that this fragmentation is real but is not,
by itself, the remaining throughput limiter.

Direct `MTLCaptureManager` capture is also rejected as the routine profiling
method for this checkpoint. A five-token capture copied approximately 89 GB of
model buffers into `native-5-token.gputrace`, while the installed command-line
tools exposed no per-kernel timing or occupancy export. Use Xcode 27 System
Trace for scheduling evidence and focused synthetic kernels for arithmetic
profiling unless Apple exposes a scriptable shader-profiler export.

## Attention, Shared MLP, and Output-Head Experiments

| Experiment | Result | Decision and reason |
|---|---:|---|
| Runtime MXFP8 output-head conversion | 29.3-29.5 server tok/s; 28.64-28.77 wall | Equivalent to retained affine-Q8 head; removed to avoid another unsupported switch. |
| Custom BF16-weight/FP32-accumulating Metal LM-head GEMV | 28.48-28.53 tok/s | Correct but slower than affine-Q8. Rejected. |
| FP32/BF16 head caching variants beyond retained caches | No promoted gain | Keep only validated immutable first-use caches and optional affine-Q8 head. |
| Q4 output head | No promoted production result | Retained only as research artifact. |
| Custom MXFP8 QMV | No promoted gain | Native MLX remained competitive or faster. |
| Convert attention and shared-expert projections to generic MLX affine Q8, retaining routed MXFP4 experts | 24.31, 24.40, 24.40 tok/s; mean 24.37 vs contemporaneous MXFP8 control 30.01, 30.04, 29.88; mean 29.97 | Exact output hash (`16a6f3491b76`) but an 18.7% regression. Generic MLX affine Q8 is not the mechanism behind DwarfStar's Q8 runtime performance. This rejects the generic MLX Q8 checkpoint profile for production, not DwarfStar's custom Q8_0 kernels. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-dwarfstar-q8-control-benchmark/20260804-215709` and `/Volumes/edata/afm-captures/deepseek-v4-mxfp8-control-contemporary-baseline/20260804-215816`. |
| Change MLX affine-Q8 decode scheduling from two SIMDgroups x four rows to four SIMDgroups x two rows | 24.66, 24.70, 24.71 tok/s vs 24.45, 24.41, 24.42 control | Invalid despite the roughly 1% apparent gain: every experimental run produced a different output hash, while all control hashes matched. The generic affine kernel has additional row/lane assumptions and cannot be retiled by changing those constants alone. Reverted. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-affine-q8-geometry-4x2-20260804/20260804-231908` and `/Volumes/edata/afm-captures/deepseek-v4-affine-q8-geometry-control-20260804/20260804-231624`. |
| Port DwarfStar's four-SIMDgroup/two-row Q8 matvec schedule while retaining MLX's unsigned affine Q8 storage | 24.71, 24.73, 24.74 tok/s vs 24.45, 24.41, 24.42 control | Exact output hash (`16a6f3491b76`) and only about 1.2% faster, inside the agreed 3% neutral band. The valid run asserted `[DSV4Path] dwarfstar-affine-q8 active` and recorded matching persistent/applied source hashes. DwarfStar's launch shape alone is not the gain; its symmetric interleaved Q8_0 format and custom dense runtime remain a coupled open variable. Reverted. Artifact: `/Volumes/edata/afm-captures/deepseek-v4-dwarfstar-affine-q8-custom-active-20260804/20260804-233413`. |
| Convert DwarfStar's advertised dense subset to paired signed symmetric Q8 storage and execute it with custom ordinary and grouped matvec kernels | 29.15, 29.07, 29.08 tok/s with staged MoE; later same-binary mean 28.64 vs 28.18 promoted control | Exact, stable output hash (`16a6f3491b76`) and a material improvement over generic affine Q8 (~24.4). The earlier 6% regression conclusion compared different staging policies and is withdrawn. A persistent typed C++ primitive reduced graph construction by 0.846 ms/token, but did not materially improve end-to-end throughput. Keep the converter profile experimental because the remaining gain is inside the agreed 3% neutral band. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-dwarfstar-symmetric-q8-staged-20260805/20260805-000619`, `/Volumes/edata/afm-captures/deepseek-v4-symmetric-q8-cpp-primitive-20260805/20260805-002714`, and `/Volumes/edata/afm-captures/deepseek-v4-symmetric-q8-cpp-perf-20260805/20260805-002835`. |
| Replace the signed symmetric-Q8 dense subset with DwarfStar-compatible interleaved Q8_0 blocks and custom ordinary/grouped kernels | Q8_0 28.50, 28.48, 28.38 tok/s; symmetric-Q8 control 28.50, 28.54, 28.52 tok/s; canonical DwarfStar 34.84, 35.01, 35.04 tok/s | After fixing a real multi-row launch bug (`B * L * heads` rows were incorrectly launched as `heads`), all nine 256-token runs produced the exact hash `16a6f3491b760b5b1ae04dedc1b8f76ff74c4688072344054b2d53d74f1a263d`. Q8_0 is performance-neutral against AFM's symmetric-Q8 representation and remains about 6.5 tok/s (22.9%) behind canonical DwarfStar. This closes checkpoint storage and isolated Q8_0 arithmetic as explanations for the gap; do not promote Q8_0 as a throughput optimization. The remaining difference is runtime scheduling and fused execution structure. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-q80-vs-ds4-256x3-20260805/20260805-122217` and `/Volumes/edata/afm-captures/deepseek-v4-symmetric-256x3-control-20260805/20260805-122501`. |
| Combine routed selector, routed MXFP4 gate/up and down, symmetric-Q8 shared gate/up and down, and final add in one typed MLX primitive and command encoder | 28.93, 28.89, 28.92 tok/s vs same-binary symmetric-Q8 control mean 28.64 | Exact output hash (`16a6f3491b76`) and required `[DSV4Path] staged-shared-q8 active` marker, but the roughly 1% change is inside the agreed 3% neutral band. Command-encoder boundaries around the shared expert are not the remaining limiter. Keep opt-in only. Artifact: `/Volumes/edata/afm-captures/deepseek-v4-fused-shared-q8-valid-20260805/20260805-005110`. |
| Decode FP4 nibbles through a DwarfStar-style threadgroup LUT while retaining AFM's existing routed tensor layout | 29.24, 29.22, 29.22 tok/s vs same-binary symmetric-Q8 control mean 28.64 | Exact output hash (`16a6f3491b76`) and required `[DSV4Path] threadgroup-fp4-lut active` marker. The roughly 2% gain is below the agreed 3% promotion threshold but is retained opt-in while the paired DwarfStar block-layout experiment is evaluated. Combining it with shared-Q8 fusion measured 29.11, 29.07, 29.07 and did not compose. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-threadgroup-lut-valid-20260805/20260805-010021` and `/Volumes/edata/afm-captures/deepseek-v4-threadgroup-lut-shared-q8-valid-20260805/20260805-010114`. |
| Shared-expert MXFP8 fusion variants | Neutral or slower | Keep the accepted compiled FFN tail rather than custom shared-expert fusion. |
| Fuse the BF16-rounded routed-plus-shared add directly into FFN HC=4 expansion | 27.43 tok/s vs 27.78 same-binary control | Exact output hash (`16a6f3491b76`) but 1.3% slower. Removing the 4096-element intermediate did not repay the extra custom-kernel boundary. Reverted. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-fused-ffn-expand-add-20260805/20260805-084722` and `/Volumes/edata/afm-captures/deepseek-v4-fused-ffn-expand-add-control-20260805/20260805-084811`. |
| Project only the six hash-selected gate rows during single-token decode instead of computing all 256 expert logits | 27.70, 27.59, 27.85 tok/s; mean 27.71 vs same-binary control 28.04, 28.01, 28.04; mean 28.03 | Exact output hash (`16a6f3491b76`) but 1.1% slower. The small dense FP32 projection is faster than the indexed gather-matmul at this shape. Reverted. Artifacts: `/Volumes/edata/afm-captures/deepseek-v4-selected-hash-gate-20260805/20260805-090901` and `/Volumes/edata/afm-captures/deepseek-v4-selected-hash-gate-control-20260805/20260805-090953`. |
| Dequantized grouped `wo_a` alternatives | Slower than packed grouped quantized MM | Retain the exact metadata-gated grouped quantized projection. |
| Disable grouped-`wo_a` activation QAT | Diagnostic only | Not promoted because it changes the checkpoint quantization contract. |

## Checkpoint and Speculative-Decoding Experiments

| Experiment | Result | Decision and reason |
|---|---:|---|
| Serve official source checkpoint directly instead of converted MLX checkpoint | 28.12 sky / 27.55 counting vs Vontra 28.46 / 27.90 | Same steady-state limit. Conversion removes loading/sanitization cost, not decode cost. |
| Fixed four-token DSpARK draft | 31.09 s vs 17.51 s autoregressive for 128 tokens | Rejected. Verification cost and low acceptance made it slower. |
| Fixed DSpARK widths without calibration | Workload-dependent | Never make a fixed draft width production default. Runtime calibration and autoregressive fallback are required. |
| Native custom MoE kernels for DSpARK verifier rows | 33.04-33.42 vs 34.03-34.11 tok/s | Rejected as noted above. The accepted Markov-head GEMV is separate and reaches 47 tok/s only when acceptance is high. |

## Measurement Mistakes to Avoid

- Never benchmark Debug. It has repeatedly produced misleadingly poor and
  irregular throughput.
- Do not close DeepSeek V4 MLX concurrency regressions solely because the old
  reshape crash path is gone. A Release live run on 2026-08-11 with four
  simultaneous `DeepSeek-V4-Flash-0731-AFM-MLX` requests showed the scheduler
  entering `DeepSeek V4 hybrid decode: B=4 row-split attention path`; per-request
  decode was still roughly 11-14 tok/s for the concurrent short workload while
  single-request natural-language decode remained about 27 tok/s. The admission
  window helps same-burst fairness and makes the behavior visible, but true
  production parity requires a dense DeepSeek V4 batch cache/attention path,
  not only scheduler tuning.
- Do not include model loading in decode throughput. Use one warmed Release
  server and report both server generation time and request wall time.
- Do not use a counting prompt alone as proof of general DSpARK performance;
  it reached 100% acceptance. Include at least one natural-language workload.
- Do not compare server token generation to UI animation cadence. Vesta's UI
  has previously made smooth generation appear slower, but direct AFM tests
  isolated the current kernel limit.
- Preserve model-output hashes or exact greedy text for every performance A/B.
- A matching hash is not sufficient when the expected hash was first produced
  by an unvalidated path. Require a semantic gate such as the expected prefix
  and forbidden structural tokens before accepting a hash as the oracle. The
  original executor benchmark consistently hashed a repeated BOS stream until
  safetensor alignment was fixed.
- DwarfStar executor safetensors must use layout contract version 3: 4096-byte
  shard payload alignment and 32-byte alignment for every real tensor start.
  Unaligned storage is a correctness bug, not a performance experiment. Never
  benchmark or draw model-equivalence conclusions from an unaligned package.
- Record the Release binary SHA-256, relevant source hashes, effective
  `VMLX_DSV4_*` environment, and required runtime activation markers. A run
  that requests an experimental path but lacks its marker is invalid, not a
  neutral result.
- Historical same-hash micro-results captured before provenance and activation
  assertions are supporting evidence only when the implementation cannot be
  proven active from its server log. Rerun them before using them to reject a
  materially different design.
- Change one factor per A/B. Model, head quantization, prompt, sampling,
  scheduler limits, and kernel selector must otherwise stay fixed.
- Use Release binary
  `.build/arm64-apple-macosx/release/afm`; never silently fall back to Debug.
- If `SwitchLayers.swift` changes, use `Scripts/swiftpm-reliable.sh` so stale
  native-driver products and `default.metallib` are invalidated correctly.

## When to Revisit a Rejected Idea

A rejected experiment may be reconsidered only when at least one premise has
changed: a new MLX quantized kernel ABI, native Metal FP4 instructions, a new
checkpoint layout, a materially different thread/work decomposition, or a
trace showing that the old bottleneck has moved. Record the new baseline,
control, exact environment, output hash, and artifact path in this ledger.
