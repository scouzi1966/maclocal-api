# OpenCode Tooling Test Report

- **Date**: 2026-03-07 15:33–16:03 UTC
- **Model**: mlx-community/Qwen3-Coder-Next-4bit
- **afm flags**: `--tool-call-parser qwen3_xml --enable-prefix-caching -V`
- **Prompt**: Simple Local File Organizer PRD (Python CLI project, 4 source files + tests)
- **Iterations**: 3 (10 min timeout each via expect)
- **OpenCode version**: 1.2.21
- **OpenCode mode**: `serve` on port 4096 + `run --attach` (required PTY via expect)

## Summary

| Run | Tool Calls | Successful | Invalid (afm) | Overwrite Err | Syntax Err | Fallback 0 |
|-----|-----------|------------|----------------|---------------|------------|------------|
| 1   | 38        | 37         | 1              | 0             | 0          | ~7         |
| 2   | 15        | 15         | 0              | 1             | 0          | ~7         |
| 3   | 22        | 20         | 2              | 2             | 1          | ~7         |
| **Total** | **75** | **72** | **3** | **3** | **1** | **~21** |

**Tool call success rate**: 96% (72/75)
**Fallback parse failures**: 21 (model generated JSON instead of XML — content lost silently)

## Errors by Category

### 1. afm Translation Bugs (model→client): 1 confirmed

| # | Run | Tool | What Happened | Classification |
|---|-----|------|---------------|----------------|
| 1 | 1 | write | Function name extracted as `write", "arguments": {"content":"...` — model emitted JSON-format `{"name": "write", "arguments": {...}}` inside `<tool_call>` tags. afm's `<function=NAME>` regex captured everything to `>`, including JSON content as part of the name. | **Hybrid: model error + afm parsing weakness** |

**Root cause**: The model switched from XML to JSON format mid-conversation. It emitted `<tool_call>{"name": "write", "arguments": {...}}</tool_call>` instead of `<tool_call><function=write><parameter=filePath>...</parameter></function></tool_call>`. afm's incremental scanner saw `<function=` was absent and fell through to emit whatever it had as the function name, which included the JSON payload.

**afm fix needed**: When `<function=` is not found in the body, afm should attempt JSON parsing as fallback before emitting a malformed function name.

### 2. Model Generation Errors: 23 total

| # | Run | Error Type | Count | Description |
|---|-----|-----------|-------|-------------|
| 1 | 1-3 | JSON inside XML tags | ~21 | Model emitted `<tool_call>{"name":"write","arguments":{...}}</tool_call>` — pure JSON wrapped in XML start/end tags. Neither incremental XML parser nor fallback regex can parse this. Content lost silently (fallback returns 0 tool calls). |
| 2 | 3 | `<parameter=file_path` without `<function=edit>` | 1 | Model emitted parameters without the `<function=NAME>` wrapper. afm emitted `edit\n<parameter=file_path` as the function name. |
| 3 | 3 | `read", "arguments": {"filePath":"..."}}\n</tool_call` | 1 | Same as #1 — JSON format with tool_call end tag leaked into name. |
| 4 | 3 | Escaped triple quotes `\\\"\\\"\\\"` | 1 | File content had Python `"""` docstrings. After afm's JSON serialization, written file had `\\\"\\\"\\\"` instead of `"""`, causing SyntaxError. **Needs investigation** — could be afm double-escaping or model generating pre-escaped content. |

### 3. OpenCode-side Errors (not afm bugs): 3

| # | Run | Error | Description |
|---|-----|-------|-------------|
| 1 | 2 | Overwrite without read | OpenCode requires `read` before `write` on existing files. Model didn't read first. |
| 2-3 | 3 | Overwrite without read (×2) | Same — model tried to overwrite files from previous run without reading them first. |

## Key Finding: JSON/XML Format Confusion

**The dominant error pattern is the model switching between JSON and XML tool call formats within a single conversation.** Out of ~96 total generation attempts (75 successful + 21 silent failures), 21 used JSON format inside `<tool_call>` tags instead of the XML `<function=NAME><parameter=KEY>` format.

### What the model generates (wrong):
```
<tool_call>
{"name": "write", "arguments": {"content": "...", "filePath": "..."}}
</tool_call>
```

### What afm expects (correct XML format):
```
<tool_call>
<function=write>
<parameter=filePath>/path/to/file</parameter>
<parameter=content>file contents</parameter>
</function>
</tool_call>
```

### Impact:
- 21 tool calls (22% of all attempts) were silently lost — `extractToolCallsFallback` returned 0 results
- These appeared to OpenCode as normal text content, not tool calls
- The model eventually re-attempted in XML format and succeeded

## Recommendations (report only — no fixes)

### afm changes worth investigating:
1. **JSON fallback in tool call body**: When `<function=` is absent from a `<tool_call>` body, try parsing as JSON `{"name":"...", "arguments":{...}}` before giving up
2. **Function name sanitization**: If extracted function name contains `"` or `{`, it's clearly malformed — reject and fall back
3. **Escaped quote investigation**: Determine if `\\\"` in file content comes from afm's JSON serialization or the model's output

### Model limitations (not fixable in afm):
- Qwen3-Coder-Next-4bit switches between JSON and XML formats unpredictably
- Longer conversations increase the chance of format switching
- The `--tool-call-parser qwen3_xml` flag correctly injects XML format in the chat template, but the model doesn't always comply

## Raw Logs

- afm: `Qwen3-Coder-Next-4bit-afm.log` (9225 lines)
- OpenCode Run 1: `Qwen3-Coder-Next-4bit-run1-opencode.json` (95 KB)
- OpenCode Run 2: `Qwen3-Coder-Next-4bit-run2-opencode.json` (54 KB)
- OpenCode Run 3: `Qwen3-Coder-Next-4bit-run3-opencode.json` (108 KB)

## Infrastructure Notes

- `opencode run` **hangs without a PTY** — it prints one INFO line and freezes. Must use `expect` or `script` to provide pseudo-TTY, or use `opencode serve` + `run --attach`
- `opencode.json` `model` field must be a **string** not an object
- The `npm` provider format (`@ai-sdk/openai-compatible`) requires runtime package resolution — may cause delays on first use
- All 3 runs used separate sessions on the same `opencode serve` instance (workdir `/tmp/opencode-serve-test`)
