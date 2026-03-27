---
name: test-macafm
description: Run the maclocal-api (AFM/MLX) test suite — automated assertions and
  smart analysis. Use when asked to test, validate, regression-check, or benchmark
  AFM before release, after code changes, or for model onboarding.
---

# test-macafm

Run the maclocal-api test suite: automated pass/fail assertions and AI-judge smart analysis.

## Triggers

Use this skill when the user asks to:
- **Test** or **validate** the server (e.g., "run the tests", "test AFM", "validate the build")
- **Regression check** after code changes
- **Onboard a new model** (verify it works correctly with the server)
- **Release check** before tagging or pushing
- **Benchmark** or **profile** model performance

## First Questions to Ask

1. **Model** — Which model to test? (Ask if not specified. Default: whatever's loaded.)
2. **Tier** — smoke / standard / full? (Suggest based on context.)
3. **Binary path** — Default `.build/release/afm`. Ask if user has a custom build location.
4. **Port** — Default 9998. Ask if user's server is on a different port.
5. **Server running?** — Is the server already running, or should tests start it?

## Tier Decision Tree

| Tier | Time | When to use | What runs |
|------|------|-------------|-----------|
| **smoke** | ~2 min | Quick sanity check, any small model, CI | `test-assertions.sh --tier smoke` |
| **standard** | ~15 min | After feature changes, mid-size model | `test-assertions.sh --tier standard` |
| **full** | ~60 min | Release validation, production model | `test-assertions.sh --tier full` + `mlx-model-test.sh --smart` with `test-llm-comprehensive.txt` + promptfoo agentic evals |

**Quick guide:**
- "Just run a quick test" → smoke
- "Test before merging" → standard
- "Full release validation" or "onboard new model" → full
- User doesn't specify → suggest standard

## Execution Workflow

### 1. Build Check
```bash
# Ensure release build is current
swift build -c release
```

### 2. Start Server (if not running)
```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/release/afm mlx -m MODEL --port 9998 \
  --tool-call-parser afm_adaptive_xml \
  --enable-prefix-caching \
  --enable-grammar-constraints &
# Wait for server to be ready
until curl -sf http://127.0.0.1:9998/v1/models >/dev/null 2>&1; do sleep 1; done
```

**Recommended flags for testing:**
- `--tool-call-parser afm_adaptive_xml` — best tool call parser with JSON-in-XML fallback
- `--enable-prefix-caching` — 67-79% prompt token savings on repeated requests
- `--enable-grammar-constraints` — EBNF constrained decoding forces valid XML tool calls, improving success from 60% to 100% on realistic workloads

### 3. Run Automated Assertions
```bash
./Scripts/test-assertions.sh --tier TIER --model MODEL --port 9998
```

**Interpret results immediately.** If any FAIL, investigate before proceeding.

### 4. Run Smart Analysis (full tier only)

The smart analysis harness manages its own server (port 9877) — do NOT pass `--port`.
It uses `test-llm-comprehensive.txt` which has an `[all]` baseline prompt and `[@ label]`
template sections. The `--smart` flag accepts batch mode prefix and tool list.

```bash
AFM_BIN=.build/release/afm ./Scripts/mlx-model-test.sh \
  --model MODEL \
  --prompts Scripts/test-llm-comprehensive.txt \
  --smart 1:claude
```

**Smart analysis options:**
- `--smart claude` or `--smart codex` — batch mode 0 (one big swoop, may fail on large test suites)
- `--smart 1:claude` or `--smart 1:codex` — batch mode 1 (test-by-test, more reliable)
- `--smart 1:claude,codex` — run multiple AI judges
- `--tests 1,5,10` — run only specific test numbers (1-indexed)

**Note:** The `[all]` prompt runs for every test variant. With high `max_tokens` (e.g., 32768
on code tests), thinking models may generate very long reasoning for the baseline prompt.
Total run time for full suite: ~45-90 min depending on model speed.

### 5. Run GPU Shader Profile (full tier, or when investigating perf)

Generates an interactive HTML report with measured DRAM bandwidth, GPU utilization/power
timelines, and per-kernel Metal shader names from xctrace Shader Timeline.

**One-time setup** (creates custom Instruments template with Shader Timeline enabled):
```bash
python3 Scripts/create-shader-template.py
```

**Run the profile** (no server needed — uses single-prompt mode):
```bash
python3 Scripts/gpu-profile-report.py MODEL [max_tokens] [prompt]
# Default: 4096 tokens, built-in GPU analysis prompt
# Example: python3 Scripts/gpu-profile-report.py mlx-community/Qwen3.5-35B-A3B-4bit
```

This does everything automatically:
1. Warms up mactop (bandwidth monitor, no sudo)
2. Runs inference with `--gpu-profile --gpu-trace 15`
3. Collects 300ms bandwidth/GPU/power samples via PTY during inference
4. Extracts shader kernel names from the xctrace trace
5. Generates `/tmp/afm-gpu-profile.html` and opens in browser

**Or use individual flags** on any AFM invocation:
```bash
afm mlx -m MODEL --gpu-profile -s "prompt"           # Zero-overhead stats
afm mlx -m MODEL --gpu-profile-bw -s "prompt"        # + mactop bandwidth (~5s)
afm mlx -m MODEL --gpu-trace 10 -s "prompt"          # xctrace shader trace
```

**Live bandwidth monitor** (run in separate terminal during server requests):
```bash
./Scripts/gpu-profile.sh bandwidth
```

**What the report shows:**
- Device info (chip, memory, architecture)
- Prefill/decode tok/s with exact timing
- Memory breakdown (model weights vs KV cache)
- DRAM bandwidth timeline chart (measured via mactop)
- GPU utilization & power timeline chart
- Per-kernel Metal shader names (from Shader Timeline)
- Exact command line for reproducibility

**What to look for:**
- GPU utilization <100% during decode → CPU-GPU pipeline bubbles
- Bandwidth utilization >80% → memory-bound, kernel optimization won't help
- Bandwidth utilization <20% with MoE model → normal (only active experts read)
- Key kernels: `affine_qmv_fast` (decode bottleneck), `steel_gemm_fused` (prefill), `sdpa_vector` (attention)

### 5b. API-Based GPU Profiling (per-request, no CLI flags needed)

Clients can request GPU profiling data via the `X-AFM-Profile` HTTP header.
No server flags required — works on any running AFM server.

**Two levels:**
```bash
# Summary: GPU power, memory, bandwidth, tok/s
curl http://127.0.0.1:9999/v1/chat/completions \
  -H "X-AFM-Profile: true" \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}]}'

# Extended: summary + 300ms time-series samples (for charts/dashboards)
curl http://127.0.0.1:9999/v1/chat/completions \
  -H "X-AFM-Profile: extended" \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}]}'
```

**Response fields (`afm_profile`):**
- `gpu_power_avg_w` / `gpu_power_peak_w` — GPU power via native IOReport (no mactop)
- `memory_weights_gib` / `memory_kv_gib` / `memory_peak_gib` — memory breakdown in GiB
- `prefill_tok_s` / `decode_tok_s` — throughput
- `est_bandwidth_gbs` — DRAM bandwidth from IOReport power (calibrated at startup via MLX GPU stress)
- `chip` / `theoretical_bw_gbs` — hardware context
- `gpu_samples` — number of 300ms readings taken

**Extended adds (`afm_profile_extended`):**
- `summary` — same as `afm_profile`
- `samples[]` — per-300ms readings: `{t, bw_gbs, gpu_pct, gpu_power_w, dram_power_w}`

**How it works internally:**
- IOReport `Energy Model` + `GPU Stats` channels sampled every 300ms via DispatchSource timer
- DRAM bandwidth derived from DRAM power using chip-specific calibration constant
- Calibration runs once at startup: 1 GiB MLX GPU stress test (~2s, async, non-blocking)
- Per-request isolation: concurrent profiled requests are guarded (second request skips gracefully)
- Zero overhead when header not sent (one string lookup per request)
- Works for both streaming (SSE event before `[DONE]`) and non-streaming

**What to look for:**
- `gpu_power_peak_w` ~28W during decode on M3 Ultra (matches mactop)
- `est_bandwidth_gbs` ~170-180 GB/s for Qwen3.5-35B-A3B-4bit (21% of 800 GB/s theoretical)
- Short requests (<300ms): at least 1 sample (timer first-fires at 100ms)
- `afm_profile` absent from response when header not sent (no null pollution)

### 6. Run Promptfoo Agentic Evals (full tier, or when validating tool calling / structured output)

The promptfoo agentic eval suite tests AFM's tool-calling and structured-output across multiple server configurations and real-world agent framework schemas. It manages its own server lifecycle.

**Prerequisites:** `promptfoo` CLI must be installed (`npm install -g promptfoo`).

**Run the full suite:**
```bash
AFM_MODEL=MODEL \
AFM_BINARY=.build/arm64-apple-macosx/release/afm \
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh all
```

**Run individual suites:**
```bash
# Just structured output tests
./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh structured

# Just tool calling (all 3 parser profiles)
./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh toolcall

# Just grammar constraint validation (8 server phases)
./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh grammar-constraints

# Just one agent framework
./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh opencode
```

**Available modes:** `all`, `structured`, `structured-stress`, `toolcall`, `toolcall-quality`, `grammar-constraints`, `agentic`, `frameworks`, `opencode`, `pi`, `openclaw`, `hermes`, `default`, `adaptive-xml`, `adaptive-xml-grammar`

#### Suite Coverage (~137 test cases across 16 configs)

| Suite | Tests | Profiles | What it validates |
|-------|-------|----------|-------------------|
| **structured** | 6 | 1 (api json_schema) | `response_format=json_schema` strict compliance |
| **structured-stress** | 4 | 1 | Nested arrays, enums, nullable types in schema |
| **toolcall** | 7 | 3 (default, adaptive-xml, grammar) | Basic tool call parsing: weather, time, multi-tool |
| **toolcall-quality** | 6 | 3 | BFCL-inspired when-to-call decisions (should model use a tool?) |
| **grammar-constraints** | 17 | 8 server phases | Schema + tool enforcement across: no-grammar, grammar-enabled, adaptive-xml, concurrent, prefix-cache, mixed-strict, header downgrade/enforce |
| **agentic** | 4 | 3 | Multi-turn coding workflow tool chains |
| **frameworks** | 8 | 3 | Agent framework tool shapes (OpenCode, Pi, OpenClaw, Hermes) |
| **opencode** | 37 | 3 | OpenCode built-in tools (primary-source derived) |
| **pi** | 20 | 3 | Pi coding-agent tools |
| **openclaw** | 12 | 3 | OpenClaw tool coverage |
| **hermes** | 12 | 3 | Hermes agentic framework tools |

#### Server Profiles (managed automatically by the script)

| Profile | AFM flags | Purpose |
|---------|-----------|---------|
| `default` | (none) | Baseline: auto-detected tool call format |
| `adaptive-xml` | `--tool-call-parser afm_adaptive_xml` | Adaptive XML with JSON-in-XML fallback |
| `adaptive-xml-grammar` | `--tool-call-parser afm_adaptive_xml --enable-grammar-constraints` | Adaptive XML + EBNF grammar enforcement |
| `grammar-enabled` | `--enable-grammar-constraints` | Grammar without adaptive XML |
| `grammar-enabled-adaptive-xml` | Both flags | Regression guard: grammar + adaptive XML |
| `grammar-enabled-concurrent` | `--enable-grammar-constraints --concurrent 2` | Grammar under concurrency |
| `grammar-enabled-prefix-cache` | `--enable-grammar-constraints --enable-prefix-caching` | Grammar + prefix caching interaction |
| `grammar-enabled-concurrent-cache` | All three flags | Full feature stack |

#### Custom Provider & Judges

- **`providers/afm_provider.mjs`** — Custom promptfoo provider with two transports: `api` (OpenAI-compatible HTTP) and `cli-guided-json` (direct binary invocation). Supports extract modes: `content`, `tool_calls`, `normalized_message`, `full_response`. Captures `responseHeaders` for grammar header assertions.
- **`judges/assert-grammar-header.mjs`** — Validates `X-Grammar-Constraints` response header: expects `"downgraded"` when grammar not available, absent when grammar active.
- **`judges/classify-failures.mjs`** — Post-run AI-based failure classifier: categorizes each failure as `afm_bug` (server/protocol), `model_quality` (wrong tool/args), or `harness_bug` (false negative).

#### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AFM_MODEL` | `mlx-community/Qwen3.5-35B-A3B-4bit` | Model to test |
| `AFM_BINARY` | `.build/arm64-apple-macosx/release/afm` | Binary path |
| `AFM_PROMPTFOO_OUT_DIR` | `/Volumes/edata/promptfoo/data/maclocal-api/current` | Report output dir |
| `AFM_PROMPTFOO_PORT` | `9999` | Server port |
| `MACAFM_MLX_MODEL_CACHE` | (none) | Model cache dir |

#### Output

JSON reports per suite+profile in `$AFM_PROMPTFOO_OUT_DIR`:
- `structured-MODEL_SLUG.json`
- `toolcall-{default,adaptive-xml,adaptive-xml-grammar}-MODEL_SLUG.json`
- `grammar-{schema,tools}-{no-grammar,grammar-enabled,adaptive-xml,concurrent,prefix-cache}-MODEL_SLUG.json`
- `{agentic,frameworks,opencode,pi,openclaw,hermes}-{default,adaptive-xml,adaptive-xml-grammar}-MODEL_SLUG.json`

### 7. Review Reports
- Assertion report: `test-reports/assertions-report-*.html`
- Smart analysis: `test-reports/smart-analysis-{tool}-*.md`
- HTML report: `test-reports/mlx-model-report-*.html`
- GPU profile: `/tmp/afm-gpu-profile.html` (+ `/tmp/afm-metal.trace` for Instruments)
- JSONL data: `test-reports/assertions-report-*.jsonl`, `test-reports/mlx-model-report-*.jsonl`
- Promptfoo evals: `$AFM_PROMPTFOO_OUT_DIR/{suite}-{profile}-MODEL_SLUG.json` (default: `/Volumes/edata/promptfoo/data/maclocal-api/current/`)

### 7. Stop Server (if we started it)
```bash
kill %1  # or whatever the background job is
```

## Interpreting Results

### Assertion Test Failures

| Group | Common failures | What to check |
|-------|----------------|---------------|
| **Stop** | Stop string found in output | Check `MLXModelService.swift` stop buffer logic, streaming vs non-streaming paths |
| **Logprobs** | Schema invalid, logprob > 0 | Check `resolveLogprobs()` and `buildChoiceLogprobs()` |
| **Think** | `<think>` tags in content | Check `extractThinkContent()` and `extractThinkTags()` |
| **Tools** | No tool_calls, invalid JSON args | Check `extractToolCallsFallback()`, model's tool call format |
| **Cache** | cached_tokens always 0 | Check `enablePrefixCaching`, `findPrefixLength()`, `PromptCacheBox` |
| **Concurrent** | Non-200 responses | Check `SerialAccessContainer` locking, request queuing |
| **Error** | Wrong HTTP status codes | Check controller validation logic |
| **Kwargs** | Thinking not disabled by `enable_thinking: false` | Check `chat_template_kwargs` merging into `additionalContext` in `MLXModelService.swift` |
| **Perf** | Low tok/s, high TTFT | Check model quantization, Metal kernel performance |
| **OpenAI-compat** | Stream usage chunk missing, logprobs absent | Check `StreamingUsageChunk` encoding, empty choices on final chunk |
| **Guided JSON** | Schema validation failure, invalid JSON | Check `--guided-json` / `response_format` pipeline, grammar constraints |
| **Batch** | Garbage output, wrong answers at B>1 | Check BatchScheduler, KV cache isolation, mask generation |

### Smart Analysis False Positives

Known patterns where AI judges score incorrectly (see `references/interpreting-scores.md`):
- Stop sequences truncating output scored as "low quality" — truncation IS the expected behavior
- Empty content when stop fires on first visible token — correct behavior
- JSON mode not constraining thinking models — prompt injection, not grammar-constrained
- "Missing reasoning" when model doesn't support `<think>` — correct, not a bug
- Thinking model consuming entire `max_tokens` budget on reasoning with empty visible content — model behavior, not a server bug
- `[all]` baseline prompt scored low when it runs with a code/math test's high `max_tokens` and system prompt — irrelevant context for the baseline prompt

### Promptfoo Eval Failures

| Category | Typical pass rate | What failures mean |
|----------|-------------------|-------------------|
| **structured, structured-stress** | 100% | Server bug in `response_format` pipeline — investigate immediately |
| **toolcall** (all profiles) | 100% | Server bug in tool call parsing — investigate immediately |
| **toolcall-quality** | ~80% | Model chose wrong tool or missed when-to-call — model quality, not server |
| **grammar-schema / grammar-tools** (non-concurrent) | 100% | Grammar constraint enforcement broken — server bug |
| **grammar-schema / grammar-tools** (concurrent) | ~50-70% | Known race condition in `--concurrent 2` grammar path — not release blocker |
| **grammar-header / grammar-mixed** | 100% | `X-Grammar-Constraints` header or mixed-strict wiring broken — server bug |
| **agentic** | ~75-100% | Multi-turn failures are usually model quality; 0% pass = server bug |
| **frameworks** | 100% | Framework tool shapes must parse correctly — server bug if failing |
| **opencode** | ~70-80% | Complex 37-tool scenarios; model can't always pick correct tool — model quality |
| **pi** | ~80-90% | Model prompt injection resistance varies — model quality |
| **openclaw** | ~80-85% | Model quality on OpenClaw-specific schemas |
| **hermes** | ~90-100% | Hermes format failures on adaptive-xml profiles = parser difference, not bug |

**Key rule:** `structured`, `toolcall`, `grammar-*` (non-concurrent), `frameworks` suites should be **100% pass**. Any failure there is a server bug. Everything else has model-quality variance.

**Post-run failure classification** (optional): Run `judges/classify-failures.mjs` on any result JSON to get AI-based `afm_bug` vs `model_quality` vs `harness_bug` classification.

### When to Escalate

- **SDPA regression**: NaN or garbage in long-context tests → check MLX version, see MEMORY.md
- **Tool call format mismatch**: Unknown format → check `ToolCallFormat.infer()` and model's config.json
- **Build failure**: Vendor patch conflict → run `Scripts/apply-mlx-patches.sh --check`

## Concurrency Benchmark

Full-harness concurrency sweep that starts the server, runs warmup, tests all concurrency levels, collects GPU metrics via mactop, saves JSON results, and generates a comparison chart.

### Script
`Scripts/benchmarks/benchmark_afm_vs_mlxlm.py`

### Usage
```bash
# AFM-only concurrency sweep (recommended for quick benchmarks)
python3 Scripts/benchmarks/benchmark_afm_vs_mlxlm.py --afm-only

# Full AFM vs mlx-lm comparison (both servers, fair A/B)
python3 Scripts/benchmarks/benchmark_afm_vs_mlxlm.py

# Re-generate graph from existing results
python3 Scripts/benchmarks/benchmark_afm_vs_mlxlm.py --graph
python3 Scripts/benchmarks/benchmark_afm_vs_mlxlm.py --graph Scripts/benchmark-results/FILE.json
```

### What it does
1. Detects hardware (chip, memory)
2. Starts server(s) with `--concurrent N`
3. 60s GPU settle + multi-round warmup (JIT kernel compilation)
4. Sweeps concurrency levels: `[1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 50]`
5. At each level: fires N simultaneous streaming 4096-token requests, measures aggregate tok/s, per-request tok/s, GPU power/temp/usage via mactop
6. Saves JSON to `Scripts/benchmark-results/concurrency-benchmark-TIMESTAMP.json`
7. Generates PNG chart to `Scripts/benchmark-results/concurrency-benchmark-TIMESTAMP.png`

### Configuration (top of script)
| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_ID` | `mlx-community/Qwen3.5-35B-A3B-4bit` | Model to benchmark |
| `MAX_TOKENS` | 4096 | Tokens per request (forces long decode) |
| `MAX_CONCURRENT` | 50 | `--concurrent` flag value (must be >= max level) |
| `LEVELS` | `[1,2,4,8,12,16,20,24,32,40,50]` | Concurrency levels to test |
| `AFM_PORT` | 9999 | Port for AFM server |

### Reference results (March 18, v0.9.7, M3 Ultra 512GB, --concurrent 28)
```
  B   Agg t/s   Per-req   Wall    GPU%   GPU W
  1     118.7     118.7   34.5s    94%   28.5W
  2     193.9      97.0   42.2s    93%   41.6W
  4     298.4      74.6   54.9s    97%   62.7W
  8     407.3      50.9   80.5s    96%   75.5W
 12     493.4      41.1   99.6s    98%   83.4W
 16     573.9      35.9  114.2s    99%   88.2W
 20     581.6      29.1  140.8s    98%   79.1W
 24     629.6      27.4  149.6s    99%   83.2W
```

### Additional batch validation scripts
| Script | Purpose |
|--------|---------|
| `Scripts/feature-mlx-concurrent-batch/batch_stress_mactop.py` | Quick stress test at arbitrary concurrency (client-only, needs running server on port 9876) |
| `Scripts/feature-mlx-concurrent-batch/batch_stress_ioreg.py` | Same but uses ioreg for GPU stats (less accurate) |
| `Scripts/feature-mlx-concurrent-batch/validate_responses.py` | Known-answer correctness at B={1,2,4,8} |
| `Scripts/feature-mlx-concurrent-batch/validate_mixed_workload.py` | Mixed short+long workload batch validation |
| `Scripts/feature-mlx-concurrent-batch/validate_multiturn_prefix.py` | Multi-turn prefix cache under concurrency |

## Key File Reference

| File | Purpose |
|------|---------|
| `Scripts/benchmarks/benchmark_afm_vs_mlxlm.py` | Full concurrency benchmark harness (server lifecycle, warmup, sweep, GPU metrics, chart generation) |
| `Scripts/test-assertions.sh` | Automated pass/fail assertion tests (unit/smoke/standard/full tiers, includes `swift test`) |
| `Scripts/test-llm-comprehensive.txt` | Comprehensive smart analysis test suite (model-generic, `[@ label]` template mode, has `[all]` baseline) |
| `Scripts/test-Qwen3.5-35B-A3B-4bit.txt` | Model-specific test suite for Qwen3.5-35B-A3B-4bit (same tests as comprehensive, hardcoded model) |
| `Scripts/test-edge-cases.txt` | Legacy smart analysis test prompts (smaller set) |
| `Scripts/test-sampling-params.sh` | Sampling parameter tests (seed, temp, top_p, etc.) |
| `Scripts/test-structured-outputs.sh` | JSON schema / structured output tests |
| `Scripts/test-tool-call-parsers.py` | Unit tests for tool call parsing |
| `Scripts/mlx-model-test.sh` | Test harness: runs prompts, collects results, generates reports |
| `Scripts/test-chat-template-kwargs.sh` | Standalone chat_template_kwargs tests (includes --no-think CLI + precedence) |
| `Scripts/regression-test.sh` | Quick regression smoke test |
| `Scripts/feature-codex-optimize-api/test-openai-compat-evals.py` | OpenAI-python SDK compatibility evals (non-stream, stream, logprobs, vllm bench) |
| `Scripts/feature-codex-optimize-api/test-guided-json-evals.py` | Guided JSON / structured output evals (API, streaming, CLI, SDK parse, edge cases) |
| `Scripts/feature-mlx-concurrent-batch/validate_responses.py` | Batched generation correctness: known-answer questions at B={1,2,4,8} |
| `Scripts/feature-mlx-concurrent-batch/validate_mixed_workload.py` | Mixed short+long workload batch validation with GPU metrics |
| `Scripts/feature-mlx-concurrent-batch/validate_multiturn_prefix.py` | Multi-turn prefix cache validation under concurrency |
| `Scripts/gpu-profile-report.py` | Full GPU shader profiling harness: mactop BW + --gpu-profile + --gpu-trace + HTML report |
| `Scripts/gpu-profile.sh` | GPU profiling helpers: bandwidth monitor, capture, trace, power |
| `Scripts/create-shader-template.py` | One-time: patches Metal System Trace template for per-kernel shader names |
| `Tests/MacLocalAPITests/StreamingUsageChunkTests.swift` | Unit tests: streaming usage chunks, finish reasons, Foundation commonPrefixLength |
| `Tests/MacLocalAPITests/ConcurrentBatchTests.swift` | Unit tests: RequestSlot, StreamChunk, BatchScheduler internals |
| `Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh` | Promptfoo agentic eval orchestrator: 11 modes, 8 server profiles, 16 configs |
| `Scripts/feature-promptfoo-agentic/providers/afm_provider.mjs` | Custom promptfoo provider: api + cli-guided-json transports, 4 extract modes |
| `Scripts/feature-promptfoo-agentic/judges/assert-grammar-header.mjs` | Custom assertion: validates X-Grammar-Constraints response header |
| `Scripts/feature-promptfoo-agentic/judges/classify-failures.mjs` | AI-based failure classifier: afm_bug vs model_quality vs harness_bug |
| `Scripts/feature-promptfoo-agentic/promptfooconfig.*.yaml` | 16 promptfoo config files (~137 test cases total) |
| `Scripts/feature-promptfoo-agentic/datasets/` | 16 YAML dataset files across structured, toolcall, grammar, agentic directories |

## Validation Checklist

### Smoke Tier
- [ ] Server reachable, model loaded
- [ ] Basic completion returns content
- [ ] Stop sequences work (absent from output, correct finish_reason)
- [ ] Logprobs schema valid
- [ ] Think extraction works (if model supports it)
- [ ] Basic tool call works
- [ ] Error handling (empty messages, malformed JSON)

### Standard Tier (adds)
- [ ] All smoke checks
- [ ] Streaming stop sequence parity
- [ ] Streaming logprobs
- [ ] Prompt cache: cached_tokens=0 first, >0 second
- [ ] Concurrent requests (2 and 3 simultaneous)
- [ ] Multi-tool calls
- [ ] Additional stop edge cases
- [ ] chat_template_kwargs: `enable_thinking=false` disables thinking (if model supports it)
- [ ] chat_template_kwargs: streaming parity
- [ ] chat_template_kwargs: default behavior unaffected
- [ ] OpenAI-compat evals: `test-openai-compat-evals.py` (non-stream, stream, logprobs, usage chunk)
- [ ] Guided JSON evals: `test-guided-json-evals.py` (API schema, streaming schema, SDK parse)

### Full Tier (adds)
- [ ] All standard checks
- [ ] Performance: TTFT < 5s, tok/s > 1
- [ ] Long context (2K, 4K tokens) no crash/NaN
- [ ] Smart analysis: test-llm-comprehensive.txt with AI judge (`--smart 1:claude` or `--smart 1:codex`)
- [ ] Streaming parity (assembled content matches non-streaming)
- [ ] Cache timing improvement visible
- [ ] Batch correctness: `validate_responses.py` at B={1,2,4,8}
- [ ] Batch mixed workload: `validate_mixed_workload.py` (short+long decode, GPU metrics)
- [ ] Batch prefix cache: `validate_multiturn_prefix.py` (multi-turn conversations under concurrency)
- [ ] GPU shader profile: `gpu-profile-report.py` (bandwidth, power, kernel names, HTML report)
- [ ] API profile: `X-AFM-Profile: true` returns `afm_profile` with GPU power + bandwidth
- [ ] API profile: `X-AFM-Profile: extended` returns `afm_profile_extended` with samples array
- [ ] API profile: no header → no `afm_profile` fields in response (no null pollution)
- [ ] API profile: streaming → profile SSE event before `[DONE]`
- [ ] API profile: concurrent profiled requests → second skips gracefully
- [ ] Promptfoo structured: 100% pass (json_schema + stress)
- [ ] Promptfoo toolcall: 100% pass (all 3 profiles)
- [ ] Promptfoo grammar-constraints (non-concurrent): 100% pass
- [ ] Promptfoo frameworks: 100% pass (all 3 profiles)
- [ ] Promptfoo opencode/pi/openclaw/hermes: >70% pass (model quality variance expected)
- [ ] Promptfoo grammar-header: downgrade/enforce headers correct
