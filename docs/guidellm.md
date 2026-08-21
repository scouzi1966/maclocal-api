# vLLM Observability and GuideLLM

AFM's MLX and DwarfStar servers expose a true raw-prompt
`POST /v1/completions` endpoint and vLLM-compatible Prometheus families from
`GET /metrics`. The compatibility contract is pinned to:

- GuideLLM `97b3077c05a367599112fd7080082c2d32c14b7e`
- vLLM Playground `76276229092455f9ef66748731e4a615f4d80720`
- vLLM `9633933dd81228fbcae07969f20881ad0b7cb766`

AFM also retains its `afm:*` metric families. Both prefixes are rendered from
one immutable telemetry snapshot per scrape, so equivalent AFM and vLLM
samples cannot observe different request states.

## Start AFM

Build with the repository wrapper, then start an MLX model with explicit
concurrency:

```bash
Scripts/swiftpm-reliable.sh build -c release --product afm
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/release/afm mlx \
  --model Qwen/Qwen3-0.6B-MLX-4bit \
  --port 9999 \
  --concurrent 4
```

The DwarfStar path uses the same HTTP surface:

```bash
.build/release/afm mlx \
  --model /Volumes/edata/models/dwarfstar/model.gguf \
  --mlx-runtime dwarfstar \
  --port 9999 \
  --concurrent 4
```

`/v1/completions` is capability-gated. It is registered for MLX and
DwarfStar models that implement raw token generation. It is intentionally not
registered for Apple Foundation Models or gateway/proxy servers because those
paths do not provide native raw prompts and exact provider token accounting.

## Prometheus and Playground

Prometheus can scrape the server root directly:

```yaml
scrape_configs:
  - job_name: afm
    static_configs:
      - targets: ['host.docker.internal:9999']
```

Validate exposition with the standard Python parser and, when installed,
Prometheus `promtool`:

```bash
curl -fsS http://127.0.0.1:9999/metrics > /Volumes/edata/afm-metrics.prom
python - <<'PY'
from prometheus_client.parser import text_string_to_metric_families
list(text_string_to_metric_families(open('/Volumes/edata/afm-metrics.prom').read()))
PY
promtool check metrics --extended --lint=none < /Volumes/edata/afm-metrics.prom
```

Current `promtool check metrics` style lint exits 3 for colon-qualified metric
names, including official `vllm:*` names. It also reports abbreviated-unit
style diagnostics for vLLM's official `*_toks_per_s` names and AFM's
`snapshot_timestamp_ms`. The qualification harness requires the strict
`--extended --lint=none` parser/cardinality pass, runs the default linter as
well, and permits only those exact compatibility-style diagnostics. Any
HELP/TYPE, syntax, consistency, or unrelated lint problem fails qualification.

The pinned, unmodified vLLM Playground always scrapes `<server-root>/metrics`
and filters for `vllm:` names. Start it, select Remote mode, and enter
`http://127.0.0.1:9999` as the remote URL:

```bash
git clone https://github.com/micytao/vllm-playground.git \
  /Volumes/edata/vllm-playground
git -C /Volumes/edata/vllm-playground checkout \
  76276229092455f9ef66748731e4a615f4d80720
python -m venv /Volumes/edata/vllm-playground-venv
/Volumes/edata/vllm-playground-venv/bin/pip install \
  -e /Volumes/edata/vllm-playground
/Volumes/edata/vllm-playground-venv/bin/vllm-playground --port 7860
```

The checked-in Grafana dashboard uses `afm:*` queries and remains useful for
AFM-specific panels. Upstream vLLM dashboards and the Playground use the
parallel `vllm:*` families without a prefix rewrite.

## GuideLLM

Install the pinned source into a durable virtual environment:

```bash
git clone https://github.com/vllm-project/guidellm.git \
  /Volumes/edata/guidellm
git -C /Volumes/edata/guidellm checkout \
  97b3077c05a367599112fd7080082c2d32c14b7e
uv venv --python 3.12 /Volumes/edata/guidellm-venv
uv pip install --python /Volumes/edata/guidellm-venv/bin/python \
  -e /Volumes/edata/guidellm prometheus-client
```

Raw-completion smoke:

```bash
/Volumes/edata/guidellm-venv/bin/guidellm run \
  --backend kind=openai_http,target=http://127.0.0.1:9999,request_format=/v1/completions \
  --tokenizer kind=huggingface_auto,model=/path/to/model-or-tokenizer \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=2 \
  --data kind=synthetic_text,prompt_tokens=32,output_tokens=16
```

Concurrent chat qualification with JSON, CSV, and HTML artifacts:

```bash
OUT=/Volumes/edata/afm-guidellm-results
mkdir -p "$OUT"
/Volumes/edata/guidellm-venv/bin/guidellm run \
  --backend kind=openai_http,target=http://127.0.0.1:9999,request_format=/v1/chat/completions \
  --tokenizer kind=huggingface_auto,model=/path/to/model-or-tokenizer \
  --profile kind=concurrent,streams=4 \
  --constraint kind=max_requests,count=8 \
  --data kind=synthetic_text,prompt_tokens=32,output_tokens=16 \
  --output kind=json,path="$OUT/benchmarks.json" \
  --output kind=csv,path="$OUT/benchmarks.csv" \
  --output kind=html,path="$OUT/benchmarks.html" \
  --disable-console-interactive

Scripts/test-vllm-guidellm-compat.py guidellm-report \
  --json "$OUT/benchmarks.json" \
  --csv "$OUT/benchmarks.csv" \
  --html "$OUT/benchmarks.html" \
  --artifact-dir "$OUT/qualification" \
  --streaming
```

Pass `--streaming` for streaming runs so qualification requires positive TTFT
and inter-token latency. Omit it for non-streaming runs, where pinned GuideLLM
truthfully reports both stream-only measurements as zero.

Pinned GuideLLM can also omit its interval-derived token-throughput sample for
an exactly one-request synchronous smoke run. Qualification still requires
positive request latency, prompt/output tokens, and time per output token in
that case. Multi-request and concurrent runs must report positive aggregate
token throughput.

Run the direct HTTP, SSE, Prometheus, parity, and concurrency contract against
the same server:

```bash
/Volumes/edata/guidellm-venv/bin/python \
  Scripts/test-vllm-guidellm-compat.py http \
  --base-url http://127.0.0.1:9999 \
  --artifact-dir /Volumes/edata/afm-guidellm-results/http
```

## Compatibility Semantics

`ignore_eos: true` suppresses model EOS as a terminal condition. Explicit
caller stop strings, the model context bound, `max_tokens` or
`max_completion_tokens`, cancellation, and failures remain terminal. A
suppressed EOS token is not emitted and is not counted as an output token.
Absent or false preserves normal EOS behavior. `stop: null` alone does not
suppress EOS.

`stream_options.continuous_usage_stats` is accepted because pinned GuideLLM
sends it. AFM does not emit estimated usage on every token. When
`include_usage` is true, the stream contains exactly one final usage-only event
with exact provider counts, followed by `[DONE]`. When `include_usage` is
false, no usage event is emitted.

Streaming failures after HTTP headers produce one OpenAI error event and close
the stream. They do not emit generated success text, finish usage, or `[DONE]`.

Compatibility for Swift package consumers is source compatibility after a
rebuild. Existing protocol requirements remain unchanged, deprecated
`StatsAggregator` names remain available, and old initializer overloads still
compile. The shared deprecated facade retains an in-memory behavioral fallback
for standalone `AFMKitMLX` consumers; an application can still install the
Services-backed collector before the first meaningful facade operation. This is
not binary ABI, precompiled-module, witness-table, or hot-swap compatibility;
downstream binaries that include AFMKit modules must be rebuilt from source
against this version.

## Qualification Boundaries

| Runtime or mode | Raw completions | Metrics | GuideLLM claim |
|---|---|---|---|
| MLX serial/batch | Implemented | Provider-native | Qualified only by a retained run artifact |
| MLX MTP/EAGLE3 | Implemented | Includes speculative counters | Separate mode-specific run required |
| DwarfStar | Implemented | Provider-native | Separate runtime run required |
| DSpARK | Implemented | Includes speculative counters | Separate mode-specific run required |
| Foundation Models | Not registered | Ingress only | Not qualified |
| Gateway/proxy | Not registered | Ingress only | Not qualified |

The executable contract fails on protocol errors, unstable discovery,
non-atomic AFM/vLLM values, missing SSE termination, repeated or estimated
usage chunks, early EOS under fixed-output generation, and zero GuideLLM
latency, throughput, or token metrics. Streaming qualification additionally
requires positive TTFT and ITL; non-streaming qualification requires the
pinned tool's stream-only zero values. A runtime is not qualified merely
because the source supports it; retain the JSON summary and raw outputs from a
successful run.
