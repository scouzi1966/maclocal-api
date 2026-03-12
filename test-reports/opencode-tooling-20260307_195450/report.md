# OpenCode Tooling Test Report

- **Date**: 2026-03-07 19:54 - 20:13
- **Model**: mlx-community/Qwen3-Coder-Next-4bit
- **Prompt**: Build a Python file organizer CLI tool (4 files: main.py, organizer.py, file_types.py, utils.py)
- **afm flags**: `--tool-call-parser llamacpp_tool_parser -V`
- **Iterations**: 3
- **afm build**: `feature/merge-41-inference-optimizations-llamacpp` branch with JSON-in-XML fallback + type coercion + tool_choice=none

## Summary

| Run | Tool Calls | Success | afm Bugs | Model Errors | OpenCode Guards | Lost |
|-----|-----------|---------|----------|-------------|-----------------|------|
| 1   | 7         | 6       | 0        | 1            | 0               | 0    |
| 2   | 11        | 7       | 0        | 0            | 4               | 0    |
| 3   | 12        | 8       | 0        | 0            | 4               | 0    |
| **Total** | **30** | **21** | **0** | **1** | **8** | **0** |

**Effective success rate** (excluding OpenCode guards): 21/22 = **95.5%**
**afm translation bugs**: **0**
**Tool calls lost**: **0** (vs ~22% lost without `llamacpp_tool_parser`)

## JSON-in-XML Fallback Activations

The `llamacpp_tool_parser` JSON-in-XML fallback activated **5 times** across 3 runs — these would have been **silently lost** without the new parser:

| # | Time | Tool | Args Parsed | Result |
|---|------|------|-------------|--------|
| 1 | 20:02:53 | write | 2 (content, filePath) | Success |
| 2 | 20:03:07 | write | 1 (content only) | Model error: filePath missing from JSON |
| 3 | 20:06:28 | bash | 2 (command, description) | Success |
| 4 | 20:12:30 | bash | 2 (command, description) | Success |
| 5 | 20:12:39 | bash | 2 (command, description) | Success |

**Key finding**: 5/30 tool calls (17%) used JSON-inside-XML format. All 5 were caught by the new fallback. Previously, all 5 would have been silently dropped.

## Parse Method Breakdown

| Method | Count | % |
|--------|-------|---|
| Incremental XML scanner | 25 | 83% |
| JSON-in-XML fallback | 5 | 17% |
| Lost (unparseable) | 0 | 0% |

## Errors by Category

### Model Generation Errors (1 total)

| # | Run | Tool | Error | Evidence |
|---|-----|------|-------|----------|
| 1 | 1 | write | filePath missing from JSON body | afm log line 235: `JSON-in-XML fallback parsed 'write' with 1 args` — model emitted JSON with only `content` key, no `filePath`. OpenCode returned: `expected string, received undefined` for `filePath`. |

**Root cause**: The model emitted a JSON tool call body `{"name":"write","arguments":{"content":"..."}}` without the `filePath` parameter. The JSON-in-XML fallback correctly parsed what was there — the model simply omitted the required parameter.

### OpenCode Guard Errors (8 total)

Runs 2 and 3 had 4 errors each — all "must read file before overwriting". These are OpenCode's safety guards because files already existed from run 1. Not afm bugs, not model errors — expected behavior when runs share a workdir.

### afm Translation Bugs (0)

No afm bugs found. All tool calls that the model generated correctly were translated accurately to the OpenCode client.

## Comparison: Before vs After `llamacpp_tool_parser`

| Metric | Before (qwen3_xml) | After (llamacpp_tool_parser) |
|--------|--------------------|-----------------------------|
| JSON-in-XML tool calls | Silently lost (~22%) | Caught by fallback (17%) |
| Tool calls lost | ~5-6 per 30 | 0 per 30 |
| afm translation bugs | Same | Same |
| Type coercion | Not applied | Applied (bool/int/number) |
| tool_choice=none | Not enforced | Enforced |

## Raw Logs

- afm: `Qwen3-Coder-Next-4bit-afm.log`
- OpenCode run 1: `Qwen3-Coder-Next-4bit-run1-opencode.json`
- OpenCode run 2: `Qwen3-Coder-Next-4bit-run2-opencode.json`
- OpenCode run 3: `Qwen3-Coder-Next-4bit-run3-opencode.json`
- OpenCode serve: `opencode-serve.log`
