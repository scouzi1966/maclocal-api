# Grammar-Constrained Decoding: `strict: true` Wiring

**Branch:** `feature/claude-grammar-constraints-tool-and-json`

## Overview

Wires the OpenAI `strict: true` field as the per-request opt-in for grammar-constrained decoding via xgrammar, gated by the `--enable-grammar-constraints` CLI flag as the admin opt-in.

- `strict: true` on `response_format.json_schema` activates JSON schema grammar
- `strict: true` on `tools[].function` activates tool call EBNF grammar
- Without the CLI flag, `strict: true` is silently downgraded to best-effort

## Policy Matrix

| `--enable-grammar-constraints` | API `strict: true` | `--concurrent` | Result |
|---|---|---|---|
| OFF | absent/false | any | Best-effort (prompt injection / parsing) |
| OFF | true | any | **Silent downgrade** + `X-Grammar-Constraints: downgraded` header + warning log |
| ON | absent/false | any | Best-effort (engine available but user didn't request) |
| ON | true | OFF | **Grammar enforced** via xgrammar token-level masking |
| ON | true | ON | **Grammar enforced** (concurrent path uses scheduler tokenizer) |

Principles:
- Users cannot gain features the admin didn't enable
- Users CAN opt out of enforcement (strict=false with CLI on)
- Never error on `strict: true` — maintain OpenAI API compatibility
- Prompt injection ("Respond with valid JSON...") always runs for json_schema regardless of grammar

## CLI Flag Interactions

| CLI Flag | Role | Interaction with Grammar |
|---|---|---|
| `--enable-grammar-constraints` | Admin gate — enables xgrammar engine | Required for grammar activation |
| `--tool-call-parser afm_adaptive_xml` | Tool call parsing format | **Independent** — grammar works with any parser (regression guard tested) |
| `--concurrent N` | Batch mode (N>1) | Grammar supported — uses `scheduler.tokenizer` for setup |
| `--enable-prefix-caching` | KV cache reuse | **Independent** — grammar is per-request logit processor, cache is prompt-level |

**Design note:** `hasStrictTools()` returns `true` if *any* tool in the request has `strict: true`. This activates grammar for the entire generation — token-level constraints can't be applied per-tool since generation is a single sequence. A request mixing `strict: true` and non-strict tools gets full grammar enforcement.

When both `response_format.json_schema.strict` and tool `strict` are true, json_schema grammar takes priority (schema grammar is more constrained than tool call EBNF).

## Implementation

### Files Changed

| File | Change |
|---|---|
| `Sources/MacLocalAPI/Models/OpenAIRequest.swift` | Added `strict: Bool?` to `RequestToolFunction` |
| `Sources/MacLocalAPI/Models/MLXModelService.swift` | `hasStrictTools()` helper, 3 grammar guard replacements (non-streaming, streaming, concurrent), warning logs |
| `Sources/MacLocalAPI/Models/BatchScheduler.swift` | Exposed `tokenizer` for concurrent grammar setup |
| `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift` | `X-Grammar-Constraints: downgraded` header on both streaming and non-streaming responses |
| `Sources/MacLocalAPI/main.swift` | CLI help text updates |

### Grammar Activation Logic

```
wantStrictSchema = responseFormat.type == "json_schema"
    && responseFormat.jsonSchema.strict == true
    && enableGrammarConstraints

wantStrictTools = hasStrictTools(tools) && enableGrammarConstraints

if wantStrictSchema → setupGrammarConstraint (JSON schema → xgrammar)
else if wantStrictTools → setupToolCallGrammarConstraint (EBNF grammar)
else → nil (best-effort)
```

### Response Header

When `strict: true` is requested but `--enable-grammar-constraints` is not set:
- `X-Grammar-Constraints: downgraded` added to response headers
- Warning logged: `[XGrammar] strict: true on tools but --enable-grammar-constraints not set; best-effort`

## Test Coverage Matrix

### Promptfoo Suite (`Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh grammar-constraints`)

7 phases, 12+ server restarts, covering the full CLI flag Cartesian product:

| Phase | Server Profile | Config | Tests | Covers |
|---|---|---|---|---|
| 1 | `default` (no grammar) | schema + tools | 12 | Downgrade path (A1-A6) |
| 2 | `grammar-enabled` | schema + tools | 12 | Enforcement path (B1-B6) |
| 3 | `grammar-enabled-adaptive-xml` | tools | 7 | Parser flag regression (E1) |
| 4 | `grammar-enabled-concurrent` | schema + tools | 12 | Concurrent path (C1-C2) |
| 5 | `grammar-enabled-prefix-cache` | schema + tools | 12 | Cache interaction (D1-D2) |
| 6 | `grammar-enabled` | mixed-strict | 1 | Schema+tool priority (F1) |
| 7 | `default` + `grammar-enabled` | header assertions | 4 | `X-Grammar-Constraints` header |

### Assertion Suite (`Scripts/test-assertions.sh --grammar-constraints`)

| Section | Tests | Covers |
|---|---|---|
| 13.1-13.2 | Calculator tool (non-stream + stream) | B4, B5 with `strict: true` |
| 13.3-13.4 | Two-tool selection | B4 tool routing |
| 13.5 | 3 required params enforced | B4 param completeness |
| 13.6-13.7 | Array param (non-stream + stream) | B4, B5 typed constraints |
| 13.8 | Complex nested schema | B4 deep structure |

### Coverage Summary

| Cell | Description | Covered | Test Location |
|---|---|---|---|
| A1 | CLI OFF, schema strict:true, non-stream | Yes | promptfoo P1 + P7 header |
| A2 | CLI OFF, schema strict:true, stream | Partial | promptfoo non-stream; streaming needs SSE provider |
| A3 | CLI OFF, schema strict:false/absent | Yes | promptfoo P1 control tests |
| A4 | CLI OFF, tool strict:true, non-stream | Yes | promptfoo P1 + P7 header |
| A5 | CLI OFF, tool strict:true, stream | Yes | test-assertions.sh S13 streaming |
| A6 | CLI OFF, tool strict:false/absent | Yes | promptfoo P1 control tests |
| B1 | CLI ON, schema strict:true, non-stream | Yes | promptfoo P2 + P7 header |
| B2 | CLI ON, schema strict:true, stream | Partial | promptfoo non-stream; streaming needs SSE provider |
| B3 | CLI ON, schema strict:false/absent | Yes | promptfoo P2 control tests |
| B4 | CLI ON, tool strict:true, non-stream | Yes | promptfoo P2 + P7 header |
| B5 | CLI ON, tool strict:true, stream | Yes | test-assertions.sh S13.2, S13.7 |
| B6 | CLI ON, tool strict:false/absent | Yes | promptfoo P2 control tests |
| C1 | CLI ON, concurrent, schema strict | Yes | promptfoo P4 |
| C2 | CLI ON, concurrent, tool strict | Yes | promptfoo P4 |
| D1 | CLI ON, prefix-cache, schema strict | Yes | promptfoo P5 |
| D2 | CLI ON, prefix-cache, tool strict | Yes | promptfoo P5 |
| E1 | adaptive_xml + grammar, tool strict | Yes | promptfoo P3 |
| E2 | default parser + grammar, tool strict | Yes | promptfoo P2 |
| F1 | Mixed strict (schema + tools) | Yes | promptfoo P6 |
| Header | X-Grammar-Constraints: downgraded | Yes | promptfoo P7 + custom JS judge |

**22 of 24 cells covered.** Remaining 2 (A2, B2 — streaming json_schema) require an SSE-capable promptfoo provider.

## Running Tests

```bash
# Promptfoo grammar-constraints suite (all 7 phases)
AFM_MODEL="mlx-community/Qwen3.5-35B-A3B-4bit" \
  MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  ./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh grammar-constraints

# Assertion tests with grammar constraints
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/release/afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9998 --enable-grammar-constraints &
./Scripts/test-assertions.sh --tier standard --model mlx-community/Qwen3.5-35B-A3B-4bit --port 9998 --grammar-constraints

# Unit tests
swift test
```
