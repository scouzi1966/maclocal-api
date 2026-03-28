---
name: test-afm-binary
description: Test a pre-built afm binary at any path — runs pre-flight safety checks, then any combination of unit tests, assertions, smart analysis, promptfoo evals, batch validation, OpenAI compat, GPU profiling. Use when user wants to validate a binary post-build, after code changes, or before release.
user_invocable: true
---

# Test AFM Binary

Test any pre-built `afm` binary with a menu of test suites. Validates the binary won't crash when relocated (pip/Homebrew install), then runs the selected tests.

## Usage

- `/test-afm-binary` — interactive: asks for binary path, model, and test selection
- `/test-afm-binary /path/to/afm` — test the binary at the given path
- `/test-afm-binary .build/arm64-apple-macosx/release/afm` — test the current build

## Instructions

### Step 1: Resolve Binary Path

Ask for the binary path if not provided as an argument. Default: `.build/arm64-apple-macosx/release/afm`.

```bash
BIN="${1:-.build/arm64-apple-macosx/release/afm}"
[ -x "$BIN" ] || BIN=".build/release/afm"
BIN_ABS="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
echo "Binary: $BIN_ABS"
```

If the binary doesn't exist or isn't executable, STOP and tell the user.

### Step 2: Pre-Flight Safety Checks (MANDATORY — always run)

These checks run before any test suite. They catch fatal distribution bugs that would crash every pip/Homebrew user. **If any check fails, STOP — do not proceed to testing.**

#### Check A: Binary version

```bash
REPORTED=$($BIN_ABS --version 2>&1)
echo "Version: $REPORTED"
```

If the version shows only a base version without a SHA suffix (e.g., `v0.9.8` instead of `v0.9.8-62395ab`), **warn** the user: this likely means the binary was built with an incremental `swift build` instead of `./Scripts/build-from-scratch.sh`. The SHA injection only happens in the build script. This is a warning, not a blocker — the binary may still be valid for testing.

#### Check B: Metallib present

```bash
BIN_DIR="$(dirname "$BIN_ABS")"

# Check for metallib in either location (SPM bundle or loose file)
if [ -f "$BIN_DIR/MacLocalAPI_MacLocalAPI.bundle/default.metallib" ]; then
  echo "PASS: Metallib in SPM bundle ($(du -h "$BIN_DIR/MacLocalAPI_MacLocalAPI.bundle/default.metallib" | cut -f1))"
elif [ -f "$BIN_DIR/default.metallib" ]; then
  echo "PASS: Loose metallib ($(du -h "$BIN_DIR/default.metallib" | cut -f1))"
else
  echo "FAIL: No metallib found next to binary"
  echo "The binary will crash on first inference without default.metallib"
fi
```

#### Check C: Relocated binary does NOT crash

```bash
TMPDIR=$(mktemp -d)
cp "$BIN_ABS" "$TMPDIR/"

# Copy metallib as loose file (pip wheel layout)
if [ -f "$BIN_DIR/MacLocalAPI_MacLocalAPI.bundle/default.metallib" ]; then
  cp "$BIN_DIR/MacLocalAPI_MacLocalAPI.bundle/default.metallib" "$TMPDIR/"
elif [ -f "$BIN_DIR/default.metallib" ]; then
  cp "$BIN_DIR/default.metallib" "$TMPDIR/"
fi

MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  "$TMPDIR/afm" mlx -m mlx-community/SmolLM3-3B-4bit -s "hello" --max-tokens 3 2>&1 | head -3
EXIT_CODE=${PIPESTATUS[0]}
rm -rf "$TMPDIR"

if [ "$EXIT_CODE" -ne 0 ]; then
  echo "FATAL: Relocated binary crashed (exit $EXIT_CODE)"
  echo "Bundle.module fatalError is still reachable — pip/Homebrew install will crash"
  echo "STOP. Fix MLXMetalLibrary.swift — it must NOT call Bundle.module"
else
  echo "PASS: Relocated binary runs without crash"
fi
```

**If this fails, STOP IMMEDIATELY. Do not run any tests. The binary is broken for distribution.**

#### Check D: No Bundle.module in source code

```bash
HITS=$(grep -r 'Bundle\.module' Sources/ --include='*.swift' | grep -v '^\s*//' | grep -v '// ' | wc -l | tr -d ' ')
if [ "$HITS" -gt 0 ]; then
  echo "FAIL: Found $HITS Bundle.module call(s) in source"
  grep -rn 'Bundle\.module' Sources/ --include='*.swift' | grep -v '//'
  echo "This WILL crash when installed via pip or Homebrew"
else
  echo "PASS: No Bundle.module calls in source"
fi
```

#### Present pre-flight results

| Check | What it catches | Result |
|-------|----------------|--------|
| A: Version | Incremental build (no SHA) | PASS/WARN/FAIL |
| B: Metallib | Missing Metal shaders → crash on inference | PASS/FAIL |
| C: Relocated binary | Bundle.module fatalError → crash on pip install | PASS/FAIL |
| D: No Bundle.module | Source code regression guard | PASS/FAIL |

**If B, C, or D fail, STOP. Do not proceed.**

### Step 3: Select Model

Show available models and let the user pick:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache ./Scripts/list-models.sh
```

Use AskUserQuestion with the model list. Default recommendation: `mlx-community/Qwen3.5-35B-A3B-4bit` (19 GB, MoE, best coverage).

For quick smoke tests, suggest `mlx-community/SmolLM3-3B-4bit` (1.6 GB, fast).

### Step 4: Select Tests

Use AskUserQuestion with **multiSelect: true**. Present these options:

| Option | Script | Server? | Port | Runtime | What it tests |
|--------|--------|---------|------|---------|---------------|
| **All** | (runs everything below) | — | — | ~3-4 hours | Complete validation |
| **Unit tests** | `swift test` | No | — | ~5s | 261 Swift unit tests (XML parsing, batch scheduler, KV cache, etc.) |
| **Assertions (smoke)** | `test-assertions.sh --tier smoke` | Yes | 9998 | ~2 min | Server reachable, basic completion, stop, logprobs, think, tools, errors |
| **Assertions (standard)** | `test-assertions.sh --tier standard` | Yes | 9998 | ~5 min | + streaming, cache, concurrent, kwargs, XML tools, adaptive XML, grammar, batch |
| **Assertions (full)** | `test-assertions.sh --tier full` | Yes | 9998 | ~15 min | + performance (TTFT, tok/s, long context 2K/4K tokens) |
| **Assertions + grammar + forced parser** | `test-assertions-multi.sh` | Managed | 9998 | ~30 min | Full tier × 2 (auto-detect + forced qwen3_xml) with grammar constraints |
| **Comprehensive smart analysis** | `mlx-model-test.sh --smart 1:claude` | Managed | 9877 | ~45-90 min | 91 test variants across samplers, stop, JSON, tools, code, math with AI judge |
| **Promptfoo agentic evals** | `run-promptfoo-agentic.sh all` | Managed | 9999 | ~60-120 min | 137 tests × 8 server profiles: structured, toolcall, grammar, agentic, frameworks |
| **Batch correctness** | `validate_responses.py` | Yes | 9999 | ~10-15 min | Known-answer correctness at B={1,2,4,8} |
| **Batch mixed workload** | `validate_mixed_workload.py` | Yes | 9999 | ~15-25 min | Short+long decode mix with GPU metrics |
| **Batch multiturn prefix** | `validate_multiturn_prefix.py` | Yes | 9999 | ~15-25 min | Multi-turn prefix cache under concurrency |
| **OpenAI compat evals** | `test-openai-compat-evals.py` | Managed | 9999 | ~5-10 min | OpenAI Python SDK compatibility (stream, logprobs, usage) |
| **Guided JSON evals** | `test-guided-json-evals.py` | Managed | 9999 | ~10-15 min | `response_format: json_schema` with real-world fixtures |
| **GPU profile** | `gpu-profile-report.py` | No (CLI) | — | ~30-60s | DRAM bandwidth, GPU power, shader kernel names, HTML report |

### Step 5: Run Selected Tests

For each selected test, set the correct environment and invoke. The binary path must be passed to every script.

**Environment (always set):**
```bash
export MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache
```

**Parallelism rules:**
- Unit tests (no server) → can run in parallel with anything
- Promptfoo (port 9999) → can run in parallel with assertions (port 9998)
- Batch validation (port 9999) → must NOT overlap with promptfoo
- Smart analysis (port 9877) → can run in parallel with assertions (port 9998)

**Per-test invocation:**

| Test | Command |
|------|---------|
| Unit tests | `swift test` |
| Assertions (any tier) | Start server: `MACAFM_MLX_MODEL_CACHE=... $BIN_ABS mlx -m MODEL --port 9998 --tool-call-parser afm_adaptive_xml --enable-prefix-caching --enable-grammar-constraints &` then `./Scripts/test-assertions.sh --tier TIER --model MODEL --port 9998 --bin "$BIN_ABS" --grammar-constraints` |
| Assertions + grammar + forced | `./Scripts/test-assertions-multi.sh --models "MODEL" --tier full --also-forced-parser qwen3_xml --grammar-constraints` with `AFM_BINARY="$BIN_ABS"` |
| Smart analysis | `AFM_BIN="$BIN_ABS" ./Scripts/mlx-model-test.sh --model MODEL --prompts Scripts/test-llm-comprehensive.txt --smart 1:claude` |
| Promptfoo | `AFM_MODEL=MODEL AFM_BINARY="$BIN_ABS" ./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh all` |
| Batch correctness | Start server: `$BIN_ABS mlx -m MODEL --port 9999 --concurrent 8 &` then `python3 Scripts/feature-mlx-concurrent-batch/validate_responses.py` |
| Batch mixed | Same server, then `python3 Scripts/feature-mlx-concurrent-batch/validate_mixed_workload.py` |
| Batch multiturn | Same server, then `python3 Scripts/feature-mlx-concurrent-batch/validate_multiturn_prefix.py` |
| OpenAI compat | `python3 Scripts/feature-codex-optimize-api/test-openai-compat-evals.py --start-server --model MODEL` with `AFM_BINARY="$BIN_ABS"` |
| Guided JSON | `python3 Scripts/feature-codex-optimize-api/test-guided-json-evals.py --start-server --model MODEL` with `AFM_BINARY="$BIN_ABS"` |
| GPU profile | `python3 Scripts/gpu-profile-report.py MODEL` with `AFM_BIN="$BIN_ABS"` |

**After each test completes**, present its results immediately. Don't wait for all tests to finish before showing anything.

### Step 6: Present Results Summary

After all selected tests complete, present a summary table:

| Suite | Pass | Total | Rate | Notes |
|-------|------|-------|------|-------|
| Pre-flight checks | N | 4 | — | — |
| Unit tests | N | N | — | — |
| Assertions (tier) | N | N | N% | — |
| ... | ... | ... | ... | — |

### Step 7: Open Promptfoo Web UI (if promptfoo tests were run)

After promptfoo evals complete, launch the interactive web interface:

```bash
promptfoo view -y &
# Opens browser at http://localhost:15500
# Shows all evaluations with interactive filtering, pass/fail drill-down, response comparison
# Results are persisted in ~/.promptfoo/promptfoo.db — all historical runs are visible
echo "Promptfoo UI running at http://localhost:15500 — press Ctrl+C to stop"
```

Leave the server running for the user to explore results. The web UI provides:
- Side-by-side comparison of outputs across server profiles (default vs adaptive-xml vs grammar)
- Drill-down into individual test failures with full request/response bodies
- Filtering by pass/fail status, test description, or provider
- Historical comparison with previous promptfoo runs

### Step 8: Archive Results

```bash
TODAY=$(date +%Y-%m-%d)
ARCHIVE_DIR="test-reports/binary-test/$TODAY"
mkdir -p "$ARCHIVE_DIR"

# Copy all reports generated during this session
cp test-reports/assertions-report-*.html test-reports/assertions-report-*.jsonl "$ARCHIVE_DIR/" 2>/dev/null
cp test-reports/multi-assertions-report-*.html test-reports/multi-assertions-report-*.jsonl "$ARCHIVE_DIR/" 2>/dev/null
cp test-reports/smart-analysis-*.md "$ARCHIVE_DIR/" 2>/dev/null
cp test-reports/mlx-model-report-*.html test-reports/mlx-model-report-*.jsonl "$ARCHIVE_DIR/" 2>/dev/null

# Copy promptfoo results
PROMPTFOO_DIR="${AFM_PROMPTFOO_OUT_DIR:-/Volumes/edata/promptfoo/data/maclocal-api/current}"
cp "$PROMPTFOO_DIR"/*-mlx-community_*.json "$ARCHIVE_DIR/" 2>/dev/null
```

Write a `SUMMARY.md` in the archive directory with: binary path, version, model tested, platform, date, and a pass/fail table for every test suite run.

## Interpreting Results

### Server-Critical Suites (must be 100% pass — failures = server bug)

| Suite | What it validates |
|-------|-------------------|
| Assertions: sections 0-8, 10-15 | Core server functionality |
| Promptfoo: structured, toolcall, grammar (non-concurrent), frameworks | API-level tool calling and structured output |
| OpenAI compat evals | SDK compatibility |
| Batch correctness | KV cache isolation under concurrency |

### Model-Quality Suites (failures expected — not server bugs)

| Suite | Typical pass rate | Why it varies |
|-------|-------------------|---------------|
| Promptfoo: opencode, pi, openclaw, hermes | 70-90% | Model can't always pick correct tool for complex scenarios |
| Promptfoo: toolcall-quality | ~80% | Model quality on when-to-call decisions |
| Promptfoo: grammar (concurrent) | 50-70% | Known race condition in `--concurrent 2` grammar path |
| Smart analysis | Varies | AI judge scoring variance, thinking model token budget |
| Batch multiturn prefix | ~85-90% | Model answer quality at high concurrency |

### When to Investigate

- **Any assertion failure** in sections 0-8 → server bug, investigate immediately
- **Relocated binary crash (Check C)** → Bundle.module regression, fix before doing anything else
- **All tool calls missing** → wrong tool call format detection, check `model_type` in config.json
- **NaN/garbage in long context** → SDPA regression, check MLX version (must be pinned to 0.30.3)
- **Streaming tool calls missing finish_reason** → check `MLXChatCompletionsController` state machine

## Quick Reference

```bash
# Smoke test the current build
/test-afm-binary .build/arm64-apple-macosx/release/afm

# Test a Homebrew-installed binary
/test-afm-binary $(brew --prefix afm-next)/bin/afm

# Test a pip-installed binary
/test-afm-binary $(python3 -c "import macafm_next; print(macafm_next.binary_path())")
```
