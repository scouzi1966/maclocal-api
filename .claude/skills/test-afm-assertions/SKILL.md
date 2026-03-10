# test-afm-assertions

Run AFM assertion tests across one or more models with deterministic pass/fail validation. Tests tool calling (including XML format), stop sequences, logprobs, think extraction, streaming, prompt cache, error handling, and performance.

## Triggers

Use this skill when the user asks to:
- **Run assertion tests** against specific models
- **Validate tool calling** across Qwen3/Qwen3.5 models
- **Compare models** on the same assertion suite
- **Regression test** after code changes with multiple models
- **Validate XML tool call parsing** for Qwen3-Coder or Qwen3.5 models

## First Questions to Ask

1. **Which model(s)?** — Ask the user which model(s) to test. Show available models:
   ```bash
   MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache ./Scripts/list-models.sh
   ```
2. **Tier?** — unit (offline, ~5s), smoke (~2 min/model), standard (~5 min/model), or full (~15 min/model)?
   Default: standard. Use `unit` for offline-only Swift tests (no server needed).
3. **Forced parser?** — Should we also test with `--tool-call-parser qwen3_xml`?
   Suggest yes if testing Qwen3/Qwen3.5 models.
4. **Server already running?** — If yes, use single-model mode (`test-assertions.sh`).
   If no (or multiple models), use multi-model mode (`test-assertions-multi.sh`).

## Common Model Configurations

### Qwen3 XML Format Models (auto-detect xmlFunction)
| Model | Type | Size | Notes |
|-------|------|------|-------|
| `mlx-community/Qwen3.5-35B-A3B-4bit` | qwen3_5_moe | 19 GB | Primary test model |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | qwen3_5 (dense) | 5.6 GB | Fast, dense Qwen3.5 |
| `mlx-community/Qwen3-Coder-Next-4bit` | qwen3_5_moe | 42 GB | Coder variant, no thinking |
| `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | qwen3_moe | 16 GB | Coder MoE |

### Qwen3 JSON Format Models (auto-detect json/hermes)
| Model | Type | Size | Notes |
|-------|------|------|-------|
| `mlx-community/Qwen3-30B-A3B-4bit` | qwen3_moe | 16 GB | Uses hermes JSON, NOT XML |

### Key Differences to Watch
- **Thinking models** (Qwen3.5-35B, Qwen3.5-9B): Have `<think>` support, Section 4 and 10 tests run
- **Non-thinking models** (Qwen3-Coder-Next): No `<think>` support, those sections skip
- **XML vs JSON format**: Qwen3 (original) uses JSON/hermes; Qwen3.5 and Qwen3-Coder use xmlFunction
- **Dense vs MoE**: Dense models (Qwen3.5-9B) have different perf characteristics

## Execution Workflow

### Single Model (server already running)
```bash
./Scripts/test-assertions.sh --tier TIER --model MODEL --port PORT
```

### Multiple Models (manages its own server)
```bash
./Scripts/test-assertions-multi.sh \
  --models "model1,model2,model3" \
  --tier TIER \
  --also-forced-parser qwen3_xml
```

### With Forced Parser Only
```bash
./Scripts/test-assertions-multi.sh \
  --models "model1" \
  --parser qwen3_xml \
  --tier standard
```

## Test Sections

| Section | Group | Tier | What it tests |
|---------|-------|------|---------------|
| U | XMLParsing, NullableSchema | unit | **Swift unit tests** (102 tests: XML parsing, type coercion, EBNF grammar, nullable schemas) — no server required |
| 0 | Preflight | smoke | Server reachable, binary exists |
| 1 | Server | smoke | /v1/models, basic completion |
| 2 | Stop | smoke+ | Stop sequences (10 variants including streaming) |
| 3 | Logprobs | smoke+ | Schema validation, top_logprobs, streaming |
| 4 | Think | smoke | `<think>` extraction (skips if model lacks thinking) |
| 5 | Tools | smoke+ | Basic tool call, streaming, multi-tool, array/nullable params |
| 6 | Cache | standard | Prompt prefix caching (requires `--enable-prefix-caching`) |
| 7 | Concurrent | standard | 2 and 3 simultaneous requests |
| 8 | Error | standard | HTTP errors, CORS, json_object, max_tokens, developer role |
| 10 | Kwargs | standard | `chat_template_kwargs` enable_thinking control |
| 11 | XMLTools | standard | **XML tool call deep validation** (10 tests) |
| 12 | AdaptiveXML | standard | afm_adaptive_xml parser (14 tests: JSON-in-XML fallback, coercion, entity decoding, EBNF) |
| 13 | Grammar | standard | **Grammar constraint validation** (8 tests, requires `--grammar-constraints`) |
| 9 | Perf | full | TTFT, tok/s, long context (2K, 4K tokens) |

### Section 13: Grammar Constraint Validation
Only runs when `--grammar-constraints` is passed (server must have `--enable-grammar-constraints`).
Tests adapted from `Scripts/tests/test-tool-call-parsers.py` patterns (originally written when xgrammar was always active):

| Test | What it validates |
|------|-------------------|
| Calculator tool call (non-streaming) | Different tool than weather — validates tool selection under grammar |
| Calculator tool call (streaming) | Same via SSE streaming path |
| Two tools: grammar allows correct selection | Weather selected from weather+calc |
| Two tools: grammar selects calculate | Calc selected from weather+calc |
| Grammar enforces 3 required params | send_email with to/subject/body — grammar prevents missing params |
| Grammar constrains array param | Tags param must be array (grammar enforces json_array at generation time) |
| Grammar array param via streaming | Same via SSE streaming path |
| Complex schema: mixed types | string + int + array + object in single tool call |

### Section 11: XML Tool Call Deep Validation (Key Tests)
These are the most important tests for Qwen3/Qwen3.5 models:

| Test | What it validates |
|------|-------------------|
| Function name correctly extracted | XML `<function=name>` parsed to `function.name` |
| Parameter values are correct string types | `<parameter=key>value</parameter>` produces strings |
| Mixed-type params (string+bool+int) | Boolean/integer params survive XML round-trip |
| Nested object param | JSON objects inside XML parameters parse correctly |
| tool_choice=required | Model produces tool call when required |
| tool_choice={function: name} | Specific function is called |
| Tool call IDs are unique | Multiple tool calls get distinct IDs |
| Streaming XML tool calls | SSE chunks assemble into valid tool call |
| Streaming array param | Array params in streaming don't serialize as strings |
| OpenAI schema validation | Full schema: id, type, function.name, function.arguments |

## Interpreting Results

### Known Acceptable Failures
| Test | Why it can fail | Impact |
|------|----------------|--------|
| Stop on newline | Thinking models emit `\n` in reasoning before visible content | None — stop works on visible content |
| Prompt cache (×2) | Server started without `--enable-prefix-caching` | Performance only — no correctness impact |
| tool_choice=required | Chat template hint, not server-enforced. Some models ignore it for non-tool prompts | Known limitation |

### Real Failures to Investigate
| Symptom | Likely cause |
|---------|-------------|
| No tool_calls at all | Wrong tool call format detection. Check `model_type` in config.json |
| Arguments not valid JSON | XML parameter parsing bug. Check `extractToolCallsFallback()` |
| Array params as strings | PR #37 regression. Check `serializeToolCallArguments()` |
| Server error on nullable schema | PR #33 regression. Check Jinja template with `anyOf` |
| NaN/garbage in long context | SDPA regression. Check MLX version (pin to 0.30.3) |
| Streaming tool calls missing finish_reason | Check `MLXChatCompletionsController` streaming state machine |

### Comparing Auto-detect vs Forced Parser
When running with `--also-forced-parser qwen3_xml`:
- **Both pass**: Auto-detection works correctly for this model
- **Auto fails, forced passes**: Auto-detection chose wrong format. Check `inferToolCallFormat()`
- **Both fail**: Server-side tool call parsing bug, not a format detection issue
- **Auto passes, forced fails**: The forced template may be incompatible (e.g., forcing XML on a JSON model)

## Reports

- **Single model**: `test-reports/assertions-report-TIMESTAMP.html`
- **Multi-model**: `test-reports/multi-assertions-report-TIMESTAMP.html` (combined)
- **JSONL data**: Same path with `.jsonl` extension

## Quick Reference

```bash
# Offline unit tests only (no server needed, ~5 seconds)
./Scripts/test-assertions.sh --tier unit --model unused

# Fast smoke test on one model (server must be running)
./Scripts/test-assertions.sh --tier smoke --model MODEL --port 9998

# Standard test with grammar constraints (server must have --enable-grammar-constraints)
./Scripts/test-assertions.sh --tier standard --model MODEL --port 9998 --grammar-constraints

# Standard test across 3 Qwen models with forced parser comparison
./Scripts/test-assertions-multi.sh \
  --models "mlx-community/Qwen3.5-35B-A3B-4bit,mlx-community/Qwen3.5-9B-MLX-4bit,mlx-community/Qwen3-Coder-Next-4bit" \
  --tier standard \
  --also-forced-parser qwen3_xml

# Full validation with grammar constraints
./Scripts/test-assertions-multi.sh \
  --models "mlx-community/Qwen3.5-35B-A3B-4bit" \
  --tier full \
  --also-forced-parser qwen3_xml \
  --grammar-constraints

# Full validation of a single model
./Scripts/test-assertions-multi.sh \
  --models "mlx-community/Qwen3.5-35B-A3B-4bit" \
  --tier full \
  --also-forced-parser qwen3_xml
```
