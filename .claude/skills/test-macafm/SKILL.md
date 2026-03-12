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
| **full** | ~60 min | Release validation, production model | `test-assertions.sh --tier full` + `mlx-model-test.sh --smart` with `test-llm-comprehensive.txt` |

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

### 5. Review Reports
- Assertion report: `test-reports/assertions-report-*.html`
- Smart analysis: `test-reports/smart-analysis-{tool}-*.md`
- HTML report: `test-reports/mlx-model-report-*.html`
- JSONL data: `test-reports/assertions-report-*.jsonl`, `test-reports/mlx-model-report-*.jsonl`

### 6. Stop Server (if we started it)
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

### Smart Analysis False Positives

Known patterns where AI judges score incorrectly (see `references/interpreting-scores.md`):
- Stop sequences truncating output scored as "low quality" — truncation IS the expected behavior
- Empty content when stop fires on first visible token — correct behavior
- JSON mode not constraining thinking models — prompt injection, not grammar-constrained
- "Missing reasoning" when model doesn't support `<think>` — correct, not a bug
- Thinking model consuming entire `max_tokens` budget on reasoning with empty visible content — model behavior, not a server bug
- `[all]` baseline prompt scored low when it runs with a code/math test's high `max_tokens` and system prompt — irrelevant context for the baseline prompt

### When to Escalate

- **SDPA regression**: NaN or garbage in long-context tests → check MLX version, see MEMORY.md
- **Tool call format mismatch**: Unknown format → check `ToolCallFormat.infer()` and model's config.json
- **Build failure**: Vendor patch conflict → run `Scripts/apply-mlx-patches.sh --check`

## Key File Reference

| File | Purpose |
|------|---------|
| `Scripts/test-assertions.sh` | Automated pass/fail assertion tests (smoke/standard/full tiers) |
| `Scripts/test-llm-comprehensive.txt` | Comprehensive smart analysis test suite (model-generic, `[@ label]` template mode, has `[all]` baseline) |
| `Scripts/test-Qwen3.5-35B-A3B-4bit.txt` | Model-specific test suite for Qwen3.5-35B-A3B-4bit (same tests as comprehensive, hardcoded model) |
| `Scripts/test-edge-cases.txt` | Legacy smart analysis test prompts (smaller set) |
| `Scripts/test-sampling-params.sh` | Sampling parameter tests (seed, temp, top_p, etc.) |
| `Scripts/test-structured-outputs.sh` | JSON schema / structured output tests |
| `Scripts/test-tool-call-parsers.py` | Unit tests for tool call parsing |
| `Scripts/mlx-model-test.sh` | Test harness: runs prompts, collects results, generates reports |
| `Scripts/test-chat-template-kwargs.sh` | Standalone chat_template_kwargs tests (includes --no-think CLI + precedence) |
| `Scripts/regression-test.sh` | Quick regression smoke test |

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

### Full Tier (adds)
- [ ] All standard checks
- [ ] Performance: TTFT < 5s, tok/s > 1
- [ ] Long context (2K, 4K tokens) no crash/NaN
- [ ] Smart analysis: test-llm-comprehensive.txt with AI judge (`--smart 1:claude` or `--smart 1:codex`)
- [ ] Streaming parity (assembled content matches non-streaming)
- [ ] Cache timing improvement visible
