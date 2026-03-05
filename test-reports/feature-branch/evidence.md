# Feature Branch Evidence: `--max-model-len` Context Length Enforcement

**Date:** 2026-03-04
**Branch:** `feature/max-model-len`
**Model:** `mlx-community/Qwen3-0.6B-4bit`
**Binary:** `.build/release/afm`

## Test Results Summary

### `test-max-model-len.sh` — 14/14 PASSED

| # | Group | Test | Result |
|---|-------|------|--------|
| 1 | Preflight | Binary exists | PASS |
| 2 | Preflight | Server starts with --max-model-len 512 | PASS |
| 3 | Models API | /v1/models reports context_window as integer | PASS |
| 4 | Models API | context_window matches --max-model-len (512) | PASS |
| 5 | Enforcement | Short prompt succeeds (under 512 tokens) | PASS |
| 6 | Enforcement | Long prompt returns HTTP 400 | PASS |
| 7 | Enforcement | Error has code=context_length_exceeded, type=invalid_request_error | PASS |
| 8 | Enforcement | Error message includes token count > 512 | PASS |
| 9 | Enforcement | Long prompt rejected in streaming mode | PASS |
| 10 | Enforcement | Prompt under limit succeeds (boundary test) | PASS |
| 11 | Backwards Compat | --max-kv-size alias: server starts | PASS |
| 12 | Backwards Compat | --max-kv-size 1024 reports context_window=1024 | PASS |
| 13 | Auto-detect | Server starts without --max-model-len | PASS |
| 14 | Auto-detect | Auto-detected context_window=40960 (from config.json) | PASS |

### Manual Verification: `/v1/models` context_window field

```
context_window = 40960 (type: int)
TEST: PASS
```

This validates the new test added to `test-assertions.sh` (Section 8) works correctly.

## Artifacts

- `max-model-len-20260304_210954.html` — HTML test report
- `max-model-len-20260304_210954.jsonl` — JSONL evidence log
- `evidence.md` — this file

## Features Validated

1. **CLI flag `--max-model-len`** — accepted, server starts correctly
2. **CLI alias `--max-kv-size`** — backwards-compatible alias works
3. **`/v1/models` API** — `context_window` field populated correctly (CLI override and auto-detect)
4. **Enforcement (non-streaming)** — returns HTTP 400 with OpenAI-compatible error shape
5. **Enforcement (streaming)** — SSE error chunk emitted with context_length_exceeded message
6. **Error format** — `type: invalid_request_error`, `code: context_length_exceeded`, message includes both limit and actual token count
7. **Auto-detection** — reads `max_position_embeddings` from model's `config.json` (Qwen3-0.6B: 40960)
8. **Boundary behavior** — prompts under the limit succeed normally
