# LLM Context Benchmarks

AFM pins
[`ivanfioravanti/llm_context_benchmarks`](https://github.com/ivanfioravanti/llm_context_benchmarks)
as a test-only submodule and provides `Scripts/benchmark-context.sh` to run its
OpenAI-compatible benchmark against AFM.

The integration intentionally leaves the upstream project unchanged. AFM owns
server startup, process cleanup, run provenance, persistent artifact placement,
and artifact validation. The upstream project owns prompt fixtures, context
sweeps, client-side timing, hardware discovery, charts, and comparisons.

## Prepare

```bash
git submodule update --init vendor/llm-context-benchmarks
make build
```

The benchmark uses `uv` and Python 3.13 or newer. On the first run, the harness
creates the pinned upstream environment with `uv sync --frozen`. Later runs can
pass `--no-sync`.

## Run AFM MLX

```bash
Scripts/benchmark-context.sh \
  --model mlx-community/Qwen3.8-27B-4bit \
  --contexts 0.5,1,2,4,8,16 \
  --max-tokens 256 \
  --runs 3
```

AFM-specific startup controls are repeatable:

```bash
Scripts/benchmark-context.sh \
  --model mlx-community/Qwen3.8-27B-4bit \
  --afm-arg --mtp \
  --afm-arg --enable-prefix-caching \
  --warm-prefix
```

`--warm-prefix` changes the benchmark from unique cold-prefill prompts to a
cache-reuse sweep. It does not enable AFM prefix caching by itself, so pass both
options when measuring AFM's cache path.

## Run Apple Foundation Models

```bash
Scripts/benchmark-context.sh \
  --mode foundation \
  --contexts 0.5,1,2,3.6 \
  --max-tokens 128
```

The local system model has a smaller practical transcript limit than MLX or
PCC. Keep the local Foundation Models sweep at or below approximately 3.6k.

## Benchmark an Existing AFM Process

```bash
Scripts/benchmark-context.sh \
  --base-url http://127.0.0.1:9999/v1 \
  --model loaded-model \
  --no-sync
```

In this mode the harness never starts or stops AFM. If `--model` is omitted,
the upstream client asks `/v1/models` and uses the first reported model.

## Results

Each run is written below:

```text
test-reports/llm-context-benchmarks/<UTC timestamp>/
```

The directory contains:

- `afm-server.log` for a managed AFM process;
- `benchmark.log` with the complete upstream console output;
- `provenance.txt` with both repository commits and benchmark settings; and
- `output/benchmark_*/` with CSV, chart, table, hardware, response, and batch
  artifacts produced by the upstream benchmark.

The harness succeeds only when `benchmark_results.csv`,
`benchmark_chart.png`, `table.txt`, and `hardware_info.json` exist and are
non-empty.

## Interpretation

This is an end-to-end HTTP benchmark. It includes AFM request parsing,
streaming, model execution, and client-observed timing. It is not a replacement
for AFM's kernel microbenchmarks or model qualification suites.

Use cold prefill for comparable context-scaling measurements. Use warm-prefix
mode specifically to inspect prefix-cache behavior. Do not compare results from
different context fixtures, sampling settings, thermal states, or model
quantizations as if they were the same test.
