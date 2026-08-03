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

The next optimization should target routed-expert dispatch without duplicating
the expert bank. It requires a correct MLX/Metal primitive that accepts two
packed weights in one dispatch; the tested custom implementation is not safe to
ship.

## Profiling Limitation

Xcode 27 Beta 3's `Metal Shader Profile` attachment did not begin inference:
`xctrace` consumed CPU and about 31 GB RSS for several minutes while AFM waited.
The trace was stopped and is not used as performance evidence. Hardware
utilization above comes from 200 ms `mactop` samples taken only during Release
decode.
