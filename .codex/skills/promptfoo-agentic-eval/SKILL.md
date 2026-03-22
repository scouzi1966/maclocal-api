---
name: promptfoo-agentic-eval
description: Run and review the Promptfoo-based AFM agentic evaluation suite. Use when the user wants structured-output, tool-calling, grammar, guided-json, streaming, concurrency, or agentic QA coverage for AFM, and especially when they want help choosing harness options or interpreting failures.
---

# Promptfoo Agentic Eval

Use this skill when the user wants to run, expand, or interpret the Promptfoo
agentic suite for AFM.

This skill is for two linked goals:
- AFM functional validation
- model-quality evaluation for agentic use

Always distinguish:
- `afm_bug`
- `model_quality`
- `harness_bug`

Always report provenance for the suite you run:
- `afm_internal`
- `primary_source`
- `public_benchmark_inspired`
- `synthetic`

Never present a benchmark-inspired or synthetic suite as if it were a public
benchmark import.

## First questions to ask

Before running the suite, ask the user the minimum needed questions:

1. Which model should be tested?
2. Which scope should be run?
   - `structured`
   - `structured-stress`
   - `toolcall`
   - `toolcall-quality`
   - `agentic`
   - `frameworks`
   - `opencode`
   - `all`
   - one profile only: `default`, `adaptive-xml`, `adaptive-xml-grammar`
3. Is the goal:
   - AFM functional QA
   - model quality
   - both
4. Should the run stay serial/safe, or include concurrency cases?
5. Should you only review existing reports, or also execute the harness?
6. Should the run prefer:
   - primary-source-only cases
   - public-benchmark-inspired cases
   - synthetic representative cases
   - mixed

If the user does not specify, assume:
- model: the repo's current primary MLX model under test
- scope: `all`
- goal: `both`
- run mode: serial/safe first
- action: execute and then review
- provenance preference: mixed, but explicitly labeled

## Working directory

Run from:

```bash
cd /Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api
```

## Main assets

Read only what is needed:

- `Scripts/feature-promptfoo-agentic/README.md`
- `Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh`
- `Scripts/feature-promptfoo-agentic/providers/afm_provider.mjs`
- `Scripts/feature-promptfoo-agentic/matrix/functional-matrix.yaml`
- `Scripts/feature-promptfoo-agentic/matrix/failure-classification.yaml`
- `docs/roadmap/promptfoo-agentic-matrix.md`

Relevant suite configs and datasets:
- `Scripts/feature-promptfoo-agentic/promptfooconfig.structured.yaml`
- `Scripts/feature-promptfoo-agentic/promptfooconfig.structured-stress.yaml`
- `Scripts/feature-promptfoo-agentic/promptfooconfig.toolcall.yaml`
- `Scripts/feature-promptfoo-agentic/promptfooconfig.toolcall-quality.yaml`
- `Scripts/feature-promptfoo-agentic/promptfooconfig.agentic.yaml`
- `Scripts/feature-promptfoo-agentic/promptfooconfig.agentic-frameworks.yaml`
- `Scripts/feature-promptfoo-agentic/promptfooconfig.opencode.yaml`
- `Scripts/feature-promptfoo-agentic/datasets/agentic/opencode-primary-tools.yaml`

If reviewing failures, inspect:

- `test-reports/promptfoo-agentic/*.json`
- `test-reports/promptfoo-agentic/*.classified.json`
- `test-reports/promptfoo-agentic/*.classified.summary.md`
- `test-reports/promptfoo-agentic/server-*.log`

## Execution workflow

### 1. Run the harness

Use the wrapper unless the user explicitly wants a narrower manual run:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh
```

Allowed narrowed runs:

```bash
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh structured
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh structured-stress
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh toolcall
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh toolcall-quality
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh agentic
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh frameworks
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh opencode
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh default
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh adaptive-xml
Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh adaptive-xml-grammar
```

Known current suite sizes:
- `structured`: 6 cases (`afm_internal`)
- `structured-stress`: 4 cases (`public_benchmark_inspired`)
- `toolcall`: 7 cases (`afm_internal`)
- `toolcall-quality`: 6 cases (`public_benchmark_inspired`)
- `agentic`: 4 cases (`synthetic representative`)
- `frameworks`: 8 cases (`mixed / currently assumption-heavy`)
- `opencode`: 37 cases (`primary_source`)

If a requested run is under 20 cases, explicitly warn the user that it is a
small sample, not broad coverage.

### 2. Review results

Check:
- pass/fail counts
- whether failures are real or harness-related
- differences across parser profiles
- structured-output vs tool-calling behavior
- provenance of the suite and whether that limits how strong conclusions can be

### 3. Classify failures

If using the automated judge path:

```bash
AFM_JUDGE_MODEL="$MODEL_ID" \
AFM_JUDGE_BASE_URL=http://127.0.0.1:9999/v1 \
node Scripts/feature-promptfoo-agentic/judges/classify-failures.mjs <report.json>
```

If working interactively in Codex/Claude-style CLI, classify manually using the
rubric in:

- `docs/roadmap/promptfoo-agentic-matrix.md`
- `Scripts/feature-promptfoo-agentic/matrix/failure-classification.yaml`

## Classification rubric

### `afm_bug`

Use when AFM violates server/runtime/protocol invariants:
- malformed JSON or SSE
- broken `tool_calls` envelope
- wrong `tool_choice` semantics
- grammar-constrained output violates grammar/schema
- stream/non-stream deterministic mismatch
- parser corrupts an otherwise valid call
- timeout, truncation, duplicate emission, crash

### `model_quality`

Use when AFM output is valid but the model behavior is weak:
- wrong tool
- missing tool
- unnecessary tool
- wrong arguments
- poor multi-turn or refusal behavior

### `harness_bug`

Use when the test machinery is wrong:
- assertion false negative
- provider normalization issue
- Promptfoo config mismatch
- classification/judge pipeline issue

## Reporting format

When reporting results, give:

1. overall run status
2. total tests executed
3. pass/fail counts per suite/profile
4. suite provenance summary
   - how many cases came from `afm_internal`
   - `primary_source`
   - `public_benchmark_inspired`
   - `synthetic`
4. failure classification summary:
   - `afm_bug`
   - `model_quality`
   - `harness_bug`
5. remaining `not_yet_classified` count, if any
6. top next actions

Prefer concise summaries, but include concrete failing cases when they matter.

## Expansion guidance

When the user asks to extend the suite, prioritize:

1. stronger custom assertions
2. streaming and grammar-specific cases
3. primary-source-derived framework suites
4. public benchmark sampling:
   - BFCL
   - When2Call
   - StructEval
   - tau-bench-style multi-turn cases
5. real AFM use cases:
   - coding agents
   - OpenClaw/Hermes-style tool orchestration
   - structured output workflows

Prefer primary sources over secondary descriptions. If a suite is built from
secondary material or assumptions, say so explicitly and do not overstate its
authority.

Do not explode the matrix blindly. Use the layered matrix in
`docs/roadmap/promptfoo-agentic-matrix.md`.
