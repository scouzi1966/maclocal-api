# Issue #38 — XML Tool Call Type Coercion Fix

**Issue:** https://github.com/scouzi1966/maclocal-api/issues/38
**PR:** https://github.com/scouzi1966/maclocal-api/pull/39
**Branch:** `fix/38-xml-tool-call-type-coercion`

## Bug

XML tool call format (`<parameter=timeout>30</parameter>`) serializes all parameter values as JSON strings. Numbers, booleans, and floats were sent as `"30"`, `"true"`, `"0.7"` instead of `30`, `true`, `0.7` — causing downstream validation errors like:

```
Invalid input: expected number, received string
```

Reported by user in OpenCode with Qwen3-Coder-Next-4bit (bash tool `timeout` parameter).

## Fix

Schema-aware type coercion across all three code paths:
1. `coerceArgumentTypes()` — non-streaming and streaming fallback
2. `jsonEncodeValue(schemaType:)` — streaming incremental XML emit
3. Both incremental emit sites look up parameter schema type before encoding

## Test Results

### Run 1: assertions-report-20260306_214821 (16 Section 11 tests)
- **Model:** mlx-community/Qwen3-Coder-Next-4bit
- **Result:** 51/56 passed (91%), 2 skipped
- **Section 11:** 15/15 PASS (all #38 tests green)
- **Known failures:** stop newline, streaming stop '5', tool_choice=none, prompt cache x2

### Run 2: assertions-report-20260306_215010 (18 Section 11 tests — added 2 edge cases)
- **Model:** mlx-community/Qwen3-Coder-Next-4bit
- **Result:** 53/58 passed (91%), 2 skipped
- **Section 11:** 17/17 PASS (all #38 tests green, including new edge cases)
- **Known failures:** same 5 pre-existing

## New Tests Added (8 total for #38)

| Test | Type | What it validates |
|------|------|-------------------|
| Integer param is JSON number | non-streaming | `max_results: 10` not `"10"` |
| Boolean param is JSON boolean | non-streaming | `enabled: true` not `"true"` |
| Number param is JSON number | non-streaming | `temperature: 0.7` not `"0.7"` |
| Mixed types correct | non-streaming | string stays string, int/bool coerced |
| Streaming integer is number | streaming | SSE-assembled `max_results` is int |
| Streaming boolean is boolean | streaming | SSE-assembled `enabled` is bool |
| Boolean false is JSON false | non-streaming | `enabled: false` not `"false"` |
| No-schema-type stays string | non-streaming | Param without `type` is not falsely coerced |
