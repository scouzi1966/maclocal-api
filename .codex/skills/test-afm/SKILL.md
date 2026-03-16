---
name: test-afm
description: Run AFM validation workflows, from smoke checks through broader regression suites.
---

# Test AFM

Use this skill when the user asks to test, validate, regression-check, or benchmark AFM.

## First decisions

Determine:
- model to test
- test tier: `smoke`, `standard`, or `full`
- binary path, defaulting to the current repo build
- server port
- whether AFM is already running or the test should start it

## Tier guide

| Tier | Typical use | Main commands |
| --- | --- | --- |
| `smoke` | quick sanity check | `./Scripts/test-assertions.sh --tier smoke ...` |
| `standard` | feature regression | `./Scripts/test-assertions.sh --tier standard ...` |
| `full` | release validation or model onboarding | assertions plus smart analysis and compatibility bundles |

## Workflow

### 1. Ensure the build is current

```bash
swift build -c release
```

### 2. Start AFM if needed

Use the repo build and a caller-provided model. If a local model cache is needed, prefer an environment variable:

```bash
MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" \
  .build/arm64-apple-macosx/release/afm mlx -m "$MODEL_ID" --port 9998 &
```

Wait until the server is reachable before running tests.

### 3. Run automated assertions

```bash
./Scripts/test-assertions.sh --tier "$TIER" --model "$MODEL_ID" --port 9998
```

### 4. Run deeper checks when needed

Common follow-ups:
- `swift test`
- `Scripts/regression-test.sh`
- `python3 Scripts/feature-codex-optimize-api/test-openai-compat-evals.py ...`
- `python3 Scripts/feature-codex-optimize-api/test-guided-json-evals.py ...`
- `./Scripts/mlx-model-test.sh --model "$MODEL_ID" --prompts Scripts/test-llm-comprehensive.txt --smart 1:codex`

### 5. Review artifacts

Look in `test-reports/` for HTML, JSONL, Markdown, and eval outputs.

## Common failure buckets

- stop sequence handling
- streaming usage chunk or logprobs regressions
- thinking-content extraction versus visible content
- guided JSON / structured output failures
- prompt cache or concurrency regressions
- performance regressions such as poor TTFT or token throughput

Escalate to targeted script runs or unit tests when a tier result fails.
