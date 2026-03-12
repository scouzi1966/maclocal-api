# Issue #34 — Support `chat_template_kwargs` API Parameter

**Model:** `mlx-community/Qwen3.5-35B-A3B-4bit` | **Date:** 2026-03-03 | **GitHub:** [#34](https://github.com/AnomalyCo/maclocal-api/issues/34)

| | |
|---|---|
| **Total Tests** | 18 |
| **Pass (after fix)** | 14 |
| **Bug Confirmed (before fix)** | 4 |

## Problem

Qwen3.5 models' Jinja chat templates check `enable_thinking` to control reasoning. When `chat_template_kwargs: {"enable_thinking": false}` is sent in the API request, afm silently ignores it — the kwargs are never passed to the Jinja template renderer. As a result, thinking/reasoning is always on regardless of the request parameter.

## Fix

Added support for `chat_template_kwargs` at both API request level and CLI server level (vLLM-compatible). Changes in 4 files:

- `OpenAIRequest.swift` — parse `chat_template_kwargs` field
- `main.swift` — add `--default-chat-template-kwargs` and `--no-think` CLI flags
- `MLXModelService.swift` — merge server defaults + request-level overrides into Jinja `additionalContext`
- `MLXChatCompletionsController.swift` — pass kwargs at both call sites

---

## Before Fix (Installed afm v0.9.5) — max_tokens=50

### Test 1 — API `chat_template_kwargs` (non-streaming, max_tokens=50) — BUG

**Server Command:**
```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9999
```

**curl Command:**
```bash
curl -s http://127.0.0.1:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 50,
    "stream": false
  }' | python3 -m json.tool
```

**Result:**
- **content:** `""` (empty)
- **reasoning_content:** `"Thinking Process:\n\n1.  **Analyze the Request:**..."` (50 tokens)
- **completion_tokens:** 50 (hit max_tokens during thinking)

Thinking was NOT disabled despite `enable_thinking: false`. All 50 tokens consumed by reasoning — no answer produced.

Evidence: [before-api-kwargs.json](before-api-kwargs.json)

---

### Test 2 — API `chat_template_kwargs` (streaming, max_tokens=50) — BUG

**curl Command:**
```bash
curl -s http://127.0.0.1:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 50,
    "stream": true
  }'
```

**Result:**
- **content deltas:** `""` (empty throughout)
- **reasoning_content deltas:** `"Thinking"`, `" "`, `"Pr"`, `"o"`, `"c"`, `"e"`, `"ss:"`, ... (50 tokens)
- **finish_reason:** `"length"` (hit max_tokens during thinking)

Same bug in streaming mode — all output goes to `reasoning_content` deltas.

Evidence: [before-api-kwargs-stream.txt](before-api-kwargs-stream.txt)

---

## Before Fix (Installed afm v0.9.5) — max_tokens=2000

### Test 8 — API `chat_template_kwargs` (non-streaming, max_tokens=2000) — BUG

**Server Command:**
```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9999
```

**curl Command:**
```bash
curl -s http://127.0.0.1:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 2000,
    "stream": false
  }' | python3 -m json.tool
```

**Result:**
- **content:** `"Four"`
- **reasoning_content:** `"Thinking Process:\n\n1.  **Analyze the Request:** ..."` (153 tokens)
- **completion_tokens:** 153

With enough tokens the model eventually answers, but thinking was NOT disabled — 152 tokens wasted on reasoning that should not have occurred.

Evidence: [before-api-kwargs-2k.json](before-api-kwargs-2k.json)

---

### Test 9 — API `chat_template_kwargs` (streaming, max_tokens=2000) — BUG

**curl Command:**
```bash
curl -s http://127.0.0.1:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 2000,
    "stream": true
  }'
```

**Result:**
- **content deltas:** `"Four"` (eventually, after reasoning completes)
- **reasoning_content deltas:** `"Thinking"`, `" "`, `"Pr"`, ... (~150 tokens of reasoning)
- **finish_reason:** `"stop"`
- **completion_tokens:** 153

Same issue in streaming — reasoning deltas stream for ~150 tokens before content appears. 152 wasted tokens, increased latency.

Evidence: [before-api-kwargs-stream-2k.txt](before-api-kwargs-stream-2k.txt)

---

## After Fix (Built binary) — max_tokens=50

### Test 3 — API `chat_template_kwargs` (non-streaming, max_tokens=50) — PASS

**Server Command:**
```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9998
```

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 50,
    "stream": false
  }' | python3 -m json.tool
```

**Result:**
- **content:** `"Four"`
- **reasoning_content:** *absent*
- **completion_tokens:** 1

Thinking correctly disabled. Direct answer returned in `content`.

Evidence: [after-api-kwargs.json](after-api-kwargs.json)

---

### Test 4 — API `chat_template_kwargs` (streaming, max_tokens=50) — PASS

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 50,
    "stream": true
  }'
```

**Result:**
- **content deltas:** `"Four"`
- **reasoning_content deltas:** *absent*
- **finish_reason:** `"stop"`
- **completion_tokens:** 1

Streaming correctly emits only `content` deltas, no reasoning.

Evidence: [after-api-kwargs-stream.txt](after-api-kwargs-stream.txt)

---

### Test 5 — Default Behavior (no kwargs, thinking ON, max_tokens=200) — PASS

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "temperature": 0,
    "max_tokens": 200
  }' | python3 -m json.tool
```

**Result:**
- **content:** `"Four"`
- **reasoning_content:** `"Thinking Process:\n\n1.  **Analyze the Request:** ..."` (153 tokens)
- **completion_tokens:** 153

Default behavior preserved — thinking is ON when no kwargs are sent.

Evidence: [after-default-thinking.json](after-default-thinking.json)

---

### Test 6 — CLI `--no-think` Flag (non-streaming, max_tokens=50) — PASS

**Server Command:**
```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9998 --no-think
```

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "temperature": 0,
    "max_tokens": 50
  }' | python3 -m json.tool
```

**Result:**
- **content:** `"Four"`
- **reasoning_content:** *absent*
- **completion_tokens:** 1

Server-level `--no-think` flag disables thinking for all requests without API kwargs.

Evidence: [after-cli-no-think.json](after-cli-no-think.json)

---

### Test 7 — CLI `--no-think` Flag (streaming, max_tokens=50) — PASS

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "temperature": 0,
    "max_tokens": 50,
    "stream": true
  }'
```

**Result:**
- **content deltas:** `"Four"`
- **reasoning_content deltas:** *absent*
- **finish_reason:** `"stop"`
- **completion_tokens:** 1

Streaming with `--no-think` works correctly.

Evidence: [after-cli-no-think-stream.txt](after-cli-no-think-stream.txt)

---

## After Fix (Built binary) — max_tokens=2000

### Test 10 — API `chat_template_kwargs` (non-streaming, max_tokens=2000) — PASS

**Server Command:**
```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9998
```

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 2000,
    "stream": false
  }' | python3 -m json.tool
```

**Result:**
- **content:** `"Four"`
- **reasoning_content:** *absent*
- **completion_tokens:** 1

With generous token budget, answer is still 1 token — no reasoning overhead.

Evidence: [after-api-kwargs-2k.json](after-api-kwargs-2k.json)

---

### Test 11 — API `chat_template_kwargs` (streaming, max_tokens=2000) — PASS

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 2000,
    "stream": true
  }'
```

**Result:**
- **content deltas:** `"Four"`
- **reasoning_content deltas:** *absent*
- **finish_reason:** `"stop"`
- **completion_tokens:** 1

Streaming with 2k budget — instant answer, no reasoning overhead.

Evidence: [after-api-kwargs-stream-2k.txt](after-api-kwargs-stream-2k.txt)

---

### Test 12 — Default Behavior (no kwargs, thinking ON, max_tokens=2000) — PASS

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "temperature": 0,
    "max_tokens": 2000
  }' | python3 -m json.tool
```

**Result:**
- **content:** `"Four"`
- **reasoning_content:** `"Thinking Process:\n\n1.  **Analyze the Request:** ..."` (153 tokens)
- **completion_tokens:** 153

Default behavior preserved with higher token budget — thinking ON, both fields present.

Evidence: [after-default-thinking-2k.json](after-default-thinking-2k.json)

---

### Test 13 — CLI `--no-think` Flag (non-streaming, max_tokens=2000) — PASS

**Server Command:**
```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9998 --no-think
```

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "temperature": 0,
    "max_tokens": 2000
  }' | python3 -m json.tool
```

**Result:**
- **content:** `"Four"`
- **reasoning_content:** *absent*
- **completion_tokens:** 1

`--no-think` with generous budget — still 1 token, no wasted reasoning.

Evidence: [after-cli-no-think-2k.json](after-cli-no-think-2k.json)

---

### Test 14 — CLI `--no-think` Flag (streaming, max_tokens=2000) — PASS

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "temperature": 0,
    "max_tokens": 2000,
    "stream": true
  }'
```

**Result:**
- **content deltas:** `"Four"`
- **reasoning_content deltas:** *absent*
- **finish_reason:** `"stop"`
- **completion_tokens:** 1

Streaming with `--no-think` and 2k budget — clean single-token answer.

Evidence: [after-cli-no-think-stream-2k.txt](after-cli-no-think-stream-2k.txt)

---

## After Fix — Precedence Tests (CLI + API combined)

### Test 15 — `--no-think` server + API `enable_thinking: false` (both agree, non-streaming) — PASS

**Server Command:**
```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9998 --no-think
```

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 2000,
    "stream": false
  }' | python3 -m json.tool
```

**Result:**
- **content:** `"Four"`
- **reasoning_content:** *absent*
- **completion_tokens:** 1

Both server and request agree — thinking disabled, no conflict.

Evidence: [after-both-no-think.json](after-both-no-think.json)

---

### Test 16 — `--no-think` server + API `enable_thinking: false` (both agree, streaming) — PASS

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0,
    "max_tokens": 2000,
    "stream": true
  }'
```

**Result:**
- **content deltas:** `"Four"`
- **reasoning_content deltas:** *absent*
- **finish_reason:** `"stop"`
- **completion_tokens:** 1

Streaming — both agree, thinking disabled.

Evidence: [after-both-no-think-stream.txt](after-both-no-think-stream.txt)

---

### Test 17 — `--no-think` server + API `enable_thinking: true` (request overrides server, non-streaming) — PASS

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": true},
    "temperature": 0,
    "max_tokens": 2000,
    "stream": false
  }' | python3 -m json.tool
```

**Result:**
- **content:** `"Four"`
- **reasoning_content:** `"Thinking Process:\n\n1.  **Analyze the Request:** ..."` (153 tokens)
- **completion_tokens:** 153

Request-level `enable_thinking: true` correctly overrides server-level `--no-think`. Thinking re-enabled for this request.

Evidence: [after-cli-no-think-api-override-on.json](after-cli-no-think-api-override-on.json)

---

### Test 18 — `--no-think` server + API `enable_thinking: true` (request overrides server, streaming) — PASS

**curl Command:**
```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "chat_template_kwargs": {"enable_thinking": true},
    "temperature": 0,
    "max_tokens": 2000,
    "stream": true
  }'
```

**Result:**
- **content deltas:** `"Four"` (after reasoning completes)
- **reasoning_content deltas:** `"Thinking"`, `" "`, `"Pr"`, ... (153 tokens)
- **finish_reason:** `"stop"`
- **completion_tokens:** 153

Streaming — request overrides server, thinking re-enabled.

Evidence: [after-cli-no-think-api-override-on-stream.txt](after-cli-no-think-api-override-on-stream.txt)

---

## Test Matrix Summary

| # | Scenario | Mechanism | Mode | max_tokens | Binary | Result | content | reasoning | tokens |
|---|----------|-----------|------|------------|--------|--------|---------|-----------|--------|
| 1 | API kwargs `enable_thinking: false` | Request | Non-stream | 50 | Installed | **BUG** | `""` | Present | 50 |
| 2 | API kwargs `enable_thinking: false` | Request | Stream | 50 | Installed | **BUG** | `""` | Present | 50 |
| 8 | API kwargs `enable_thinking: false` | Request | Non-stream | 2000 | Installed | **BUG** | `"Four"` | Present | 153 |
| 9 | API kwargs `enable_thinking: false` | Request | Stream | 2000 | Installed | **BUG** | `"Four"` | Present | 153 |
| 3 | API kwargs `enable_thinking: false` | Request | Non-stream | 50 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 4 | API kwargs `enable_thinking: false` | Request | Stream | 50 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 10 | API kwargs `enable_thinking: false` | Request | Non-stream | 2000 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 11 | API kwargs `enable_thinking: false` | Request | Stream | 2000 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 5 | No kwargs (default) | — | Non-stream | 200 | Built (fix) | **PASS** | `"Four"` | Present | 153 |
| 12 | No kwargs (default) | — | Non-stream | 2000 | Built (fix) | **PASS** | `"Four"` | Present | 153 |
| 6 | `--no-think` CLI | Server | Non-stream | 50 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 7 | `--no-think` CLI | Server | Stream | 50 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 13 | `--no-think` CLI | Server | Non-stream | 2000 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 14 | `--no-think` CLI | Server | Stream | 2000 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 15 | `--no-think` + API `false` | Both | Non-stream | 2000 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 16 | `--no-think` + API `false` | Both | Stream | 2000 | Built (fix) | **PASS** | `"Four"` | Absent | 1 |
| 17 | `--no-think` + API `true` (override) | Request wins | Non-stream | 2000 | Built (fix) | **PASS** | `"Four"` | Present | 153 |
| 18 | `--no-think` + API `true` (override) | Request wins | Stream | 2000 | Built (fix) | **PASS** | `"Four"` | Present | 153 |

## Key Insight: Token Waste

The max_tokens=2000 "before" tests (8, 9) reveal an important nuance: with enough budget, the model *does* eventually produce the answer — but wastes **152 tokens on reasoning** that the user explicitly asked to disable:

- **153x token cost** (153 vs 1 token) for the same answer
- **~150x latency increase** for streaming clients waiting for the first content delta
- With max_tokens=50 (tests 1, 2), the answer is *completely lost* — all tokens consumed by reasoning

## Evidence Files

### Before Fix (Bug Reproduction)
- [before-api-kwargs.json](before-api-kwargs.json) — Non-streaming, max_tokens=50
- [before-api-kwargs-stream.txt](before-api-kwargs-stream.txt) — Streaming SSE, max_tokens=50
- [before-api-kwargs-2k.json](before-api-kwargs-2k.json) — Non-streaming, max_tokens=2000
- [before-api-kwargs-stream-2k.txt](before-api-kwargs-stream-2k.txt) — Streaming SSE, max_tokens=2000

### After Fix (API kwargs)
- [after-api-kwargs.json](after-api-kwargs.json) — Non-streaming, max_tokens=50
- [after-api-kwargs-stream.txt](after-api-kwargs-stream.txt) — Streaming, max_tokens=50
- [after-api-kwargs-2k.json](after-api-kwargs-2k.json) — Non-streaming, max_tokens=2000
- [after-api-kwargs-stream-2k.txt](after-api-kwargs-stream-2k.txt) — Streaming, max_tokens=2000

### After Fix (Default / Regression)
- [after-default-thinking.json](after-default-thinking.json) — Default thinking ON, max_tokens=200
- [after-default-thinking-2k.json](after-default-thinking-2k.json) — Default thinking ON, max_tokens=2000

### After Fix (CLI --no-think)
- [after-cli-no-think.json](after-cli-no-think.json) — Non-streaming, max_tokens=50
- [after-cli-no-think-stream.txt](after-cli-no-think-stream.txt) — Streaming, max_tokens=50
- [after-cli-no-think-2k.json](after-cli-no-think-2k.json) — Non-streaming, max_tokens=2000
- [after-cli-no-think-stream-2k.txt](after-cli-no-think-stream-2k.txt) — Streaming, max_tokens=2000

### After Fix (Precedence: CLI + API combined)
- [after-both-no-think.json](after-both-no-think.json) — --no-think + API false, non-streaming
- [after-both-no-think-stream.txt](after-both-no-think-stream.txt) — --no-think + API false, streaming
- [after-cli-no-think-api-override-on.json](after-cli-no-think-api-override-on.json) — --no-think + API true (override), non-streaming
- [after-cli-no-think-api-override-on-stream.txt](after-cli-no-think-api-override-on-stream.txt) — --no-think + API true (override), streaming

---

*Generated for [Issue #34](https://github.com/AnomalyCo/maclocal-api/issues/34). All tests run with `temperature: 0` for deterministic output. Prompt: "What is 2+2? Answer in one word."*
