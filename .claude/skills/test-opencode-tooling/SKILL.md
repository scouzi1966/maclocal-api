---
name: test-opencode-tooling
description: Use when testing tool call reliability between OpenCode and afm — captures streaming XML tool call errors, classifies them as afm translation bugs vs model generation errors, and produces a diagnostic report without fixing anything
---

# Test OpenCode Tooling

Automated loop that runs OpenCode tasks against afm, captures tool call errors from both sides, classifies each as an **afm bug** or **model error**, and generates a report. Does not fix anything.

## When to Use

- After changing tool call parsing code (XML, streaming, type coercion)
- Onboarding a new model to verify tool call reliability
- Investigating user-reported tool call failures with OpenCode
- Comparing tool call error rates across models

## First Questions to Ask

1. **Prompt/PRD** — Ask the user to paste the prompt text or provide a file path. This is the task OpenCode will execute (e.g., a PRD, coding task, or test scenario that exercises tool calls).
2. **Model(s)** — Which model(s) to test? Show available:
   ```bash
   MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache ./Scripts/list-models.sh
   ```
3. **afm start parameters** — Any extra flags beyond defaults? (e.g., `--tool-call-format`, `--enable-prefix-caching`, `--no-think`)
4. **Iterations** — How many times to run the same prompt per model? Default: 1. More runs help distinguish flaky model errors from deterministic afm bugs.
5. **Working directory** — Temp dir for OpenCode to work in. Default: create a fresh `/tmp/opencode-test-TIMESTAMP` per run.

## OpenCode CLI Gotchas

**CRITICAL**: `opencode run` hangs silently without a PTY. It prints one INFO line and freezes — no error, no output. You **must** use one of these approaches:

1. **`opencode serve` + `run --attach`** (recommended): Start a headless server, then attach `run` to it via `expect` for PTY
2. **`expect`** wrapper: Provides the pseudo-TTY that `opencode run` requires

Other gotchas:
- `opencode.json` `model` field must be a **string**, not an object — `"model": "ollama/model-id"` not `"model": {"default": "..."}`
- The `npm` provider format (`@ai-sdk/openai-compatible`) is required for custom baseURL — the `"api": "openai"` format does NOT accept `baseURL`
- OpenCode config is loaded from both `~/.config/opencode/opencode.json` (global) AND `$WORKDIR/opencode.json` (local) — local overrides global
- The workdir should be a **git repo** (`git init`) for OpenCode to function properly

## Execution Workflow

### 1. Setup

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_PORT=9877
OC_PORT=4096
REPORT_DIR="test-reports/opencode-tooling-${TIMESTAMP}"
mkdir -p "$REPORT_DIR"
```

Save the user's prompt to a file:
```bash
cat > "$REPORT_DIR/prompt.md" << 'PROMPT_EOF'
<paste user's prompt here>
PROMPT_EOF
```

### 2. Start OpenCode Serve

Create a workdir with git init and config pointing at afm:

```bash
OC_WORKDIR="/tmp/opencode-serve-${TIMESTAMP}"
mkdir -p "$OC_WORKDIR"
cd "$OC_WORKDIR" && git init -q && cd -
```

Write the OpenCode config. **Must use `npm` provider with `options.baseURL`**:
```bash
cat > "$OC_WORKDIR/opencode.json" << EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "afm-test",
      "options": {
        "baseURL": "http://localhost:${TEST_PORT}/v1"
      },
      "models": {
        "${MODEL}": {
          "name": "${MODEL}"
        }
      }
    }
  }
}
EOF
```

Start the headless server:
```bash
cd "$OC_WORKDIR"
opencode serve --port $OC_PORT --print-logs --log-level DEBUG \
  > "$REPORT_DIR/opencode-serve.log" 2>&1 &
OC_SERVE_PID=$!
cd -

# Wait for serve to be ready
until curl -sf http://127.0.0.1:${OC_PORT}/ >/dev/null 2>&1; do sleep 1; done
```

### 3. For Each Model

#### 3a. Start afm with verbose logging

```bash
AFM_DEBUG=1 MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/release/afm mlx -m "$MODEL" --port $TEST_PORT -V \
  $EXTRA_AFM_FLAGS \
  > "$REPORT_DIR/${MODEL_SLUG}-afm.log" 2>&1 &
AFM_PID=$!

# Wait for server ready
until curl -sf http://127.0.0.1:${TEST_PORT}/v1/models >/dev/null 2>&1; do sleep 1; done
```

Where `MODEL_SLUG` is the model ID with `/` replaced by `_`.

#### 3b. Run OpenCode via expect + attach (per iteration)

**`expect` provides the PTY that `opencode run` requires.** The `--attach` flag connects to the serve instance which already has the config and workdir.

```bash
/usr/bin/expect << EXPECT_EOF > "$REPORT_DIR/${MODEL_SLUG}-run${RUN}-opencode.json" 2>&1
set timeout 600
log_user 1

spawn opencode run --attach http://localhost:${OC_PORT} --print-logs --format json "${PROMPT}"
expect {
    timeout { puts "TIMEOUT"; exit 1 }
    eof { puts "EOF"; exit 0 }
}
EXPECT_EOF
```

The `--format json` flag outputs structured JSON events:
- `{"type":"tool_use",...}` — tool call with input/output/error
- `{"type":"text",...}` — assistant text content
- `{"type":"step_start",...}` / `{"type":"step_finish",...}` — generation boundaries

Each run creates a new session on the same serve instance. The timeout (600s = 10 min) should be enough for most PRDs — increase for complex tasks.

**IMPORTANT: Clean workdir between iterations.** Before each run, remove all generated files from the OpenCode workdir so that results from a previous iteration don't contaminate the next one (e.g., OpenCode's "must read file before overwriting" guard triggers on leftover files). The cleanest approach is to stop `opencode serve`, recreate the workdir from scratch (`rm -rf "$OC_WORKDIR" && mkdir -p "$OC_WORKDIR" && cd "$OC_WORKDIR" && git init -q && cd -`), copy the `opencode.json` config back, and restart `opencode serve`. This ensures each iteration starts with a pristine empty git repo.

```bash
# Between iterations: reset workdir
kill $OC_SERVE_PID 2>/dev/null; wait $OC_SERVE_PID 2>/dev/null
rm -rf "$OC_WORKDIR"
mkdir -p "$OC_WORKDIR"
cd "$OC_WORKDIR" && git init -q && cd -
# Re-copy opencode.json config (same as setup step)
cat > "$OC_WORKDIR/opencode.json" << EOF
{ ... same config as before ... }
EOF
cd "$OC_WORKDIR"
opencode serve --port $OC_PORT --print-logs --log-level DEBUG \
  >> "$REPORT_DIR/opencode-serve.log" 2>&1 &
OC_SERVE_PID=$!
cd -
until curl -sf http://127.0.0.1:${OC_PORT}/ >/dev/null 2>&1; do sleep 1; done
```

#### 3c. Stop afm (after all iterations for this model)

```bash
kill $AFM_PID 2>/dev/null
wait $AFM_PID 2>/dev/null
```

### 4. Stop OpenCode Serve

```bash
kill $OC_SERVE_PID 2>/dev/null
```

### 3. Analyze Logs

For each run, analyze both log files to extract and classify errors.

#### From afm logs (`-afm.log`), look for:

| Pattern | Classification |
|---------|---------------|
| `SKIP false </tool_call> end tag` | afm handled correctly (model emitted premature end tag) |
| `EMIT param[N]: key→...` with wrong value | Check if model sent wrong value (model error) or afm mangled it (afm bug) |
| `RECV </tool_call>` with `raw=` body | Raw model output — compare against what OpenCode received |
| `extractToolCallsFallback` activated | Incremental parser failed, fallback used — note if result was correct |
| `SEND tool_call fallback: found 0 tool calls` | **Critical** — tool call body couldn't be parsed at all. Usually means model emitted JSON instead of XML inside `<tool_call>` tags |
| `SEND tool_call name:` with JSON in name | afm extracted JSON payload as function name — model mixed formats |
| `coerceArgumentTypes` log entries | Type coercion activated — check if result matches schema |
| Malformed XML in raw body (e.g., `<function=X>` instead of `<parameter=X>`) | Model error — wrong XML tag |
| Duplicate `<parameter=key>` tags | Model error — model emitted same param twice |
| Missing `</function>` in body | Model error — incomplete XML generation |

#### From OpenCode output (`-opencode.json`), look for:

| Pattern | Classification |
|---------|---------------|
| `"tool":"invalid"` with mangled tool name | afm parsed function name wrong — cross-ref afm `SEND tool_call name:` log |
| `"invalid arguments"` with `undefined` values | Parameter was lost — cross-reference afm log to determine if afm dropped it or model never sent it |
| `"expected number, received string"` | Type coercion failed — afm bug if schema had `type: "integer"` |
| Tool name not in schema | Model hallucinated tool — model error |
| `"command" undefined` for bash tool | Cross-ref afm raw body: if `<parameter=command>` present → afm bug; if `<function=command>` → model error |
| `SyntaxError` with `\\\"` in written files | Possible afm double-escaping of quotes in tool call arguments |

#### Cross-referencing (the key step):

For each OpenCode error:
1. Find the corresponding tool call in afm's log (match by timestamp proximity)
2. Read the `raw=` body from afm's `RECV </tool_call>` log
3. Compare what the model generated vs what afm emitted vs what OpenCode received
4. Classify:
   - **afm schema→model bug**: afm sent wrong/incomplete tool schema to the model
   - **afm model→client bug**: Model output was correct but afm mangled it (dropped param, wrong type, truncated body)
   - **Model generation error**: Model produced invalid XML, wrong tags, missing params, hallucinated tools

### 4. Generate Report

Create `$REPORT_DIR/report.md`:

```markdown
# OpenCode Tooling Test Report
- Date: TIMESTAMP
- Model(s): ...
- Prompt: (first 200 chars)
- afm flags: ...
- Iterations per model: N

## Summary
| Model | Runs | Tool Calls | Errors | afm Bugs | Model Errors |
|-------|------|------------|--------|----------|--------------|

## Errors by Category

### afm Translation Bugs (model→client)
| # | Model | Run | Tool | Parameter | What Happened | afm Raw Body |
|---|-------|-----|------|-----------|---------------|-------------|

### afm Translation Bugs (schema→model)
| # | Model | Run | Tool | What Happened |
|---|-------|-----|------|---------------|

### Model Generation Errors
| # | Model | Run | Tool | Error Type | Raw Output |
|---|-------|-----|------|------------|------------|

## Raw Logs
- afm: [link to log file]
- OpenCode: [link to json file]
```

### 5. Present Results

Show the user:
- Summary table (pass rate per model)
- Each error with classification and evidence
- Recommendation: which errors are actionable afm bugs vs model limitations
- **Do NOT propose or implement fixes** — report only

## Error Classification Guide

### Definitely afm Bug
- Parameter present in raw model output but missing in OpenCode's received arguments
- Type mismatch when schema has explicit `type` and afm didn't coerce
- Tool call body truncated (false end tag not caught)
- Function name mangled or lost

### Definitely Model Error
- **JSON inside XML tags** (most common): Model emits `<tool_call>{"name":"write","arguments":{...}}</tool_call>` instead of XML `<function=write><parameter=...>` format. afm's fallback logs `found 0 tool calls` — content is silently lost. Qwen3-Coder-Next switches formats unpredictably, especially in longer conversations.
- `<function=X>` used instead of `<parameter=X>` (wrong XML tag)
- Tool name not in provided schema (hallucinated tool)
- Parameter never appears in raw model output
- Garbage characters in parameter values (e.g., trailing `}`)
- Incomplete XML (missing `</function>` or `</parameter>`)
- `<parameter=KEY>` without wrapping `<function=NAME>` — parameters emitted without function context

### Ambiguous (needs investigation)
- Empty parameter value — could be model sending empty or afm dropping content
- Duplicate parameters — model may emit twice, afm may deduplicate wrong
- Streaming assembly errors — compare raw chunks vs assembled result
- Escaped triple quotes (`\\\"\\\"\\\"`) in written files — could be afm double-escaping or model pre-escaping

## Common Mistakes

- **Not checking raw afm body**: Always cross-reference OpenCode errors against afm's `raw=` log. Without this, you can't classify.
- **Blaming afm for model errors**: Models frequently emit broken XML. Check the raw output first.
- **Blaming the model for afm bugs**: afm has had bugs dropping empty params, false end tags, type coercion failures. Don't assume the model is always wrong.
- **Running without `-V` flag**: Without verbose logging, you can't see raw model output or per-parameter emissions. Always use `-V`.
