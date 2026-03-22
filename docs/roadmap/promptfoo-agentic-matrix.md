# Promptfoo Agentic Matrix for AFM

## Purpose

This document defines the test matrix and failure-classification model for AFM's
Promptfoo-based evaluation suite.

Goals:
- validate AFM functionality for MLX-served agentic workloads
- measure model quality on tool use and structured output
- separate AFM bugs from model-quality misses
- support AI-driven judging and summarization as a later automation/skill layer

AFM is intended to serve models for agentic use. Tool calling, guided
generation, grammar constraints, streaming, and concurrency are first-class
concerns.

## Failure Classes

Every failing case starts as `not_yet_classified`.

The post-run AI judge must convert each failure to one of:

- `1 = afm_bug`
- `2 = model_quality`
- `3 = harness_bug`

### 1 = afm_bug

Use `afm_bug` only when AFM violates a server/runtime invariant.

Examples:
- malformed JSON or malformed SSE event framing
- invalid `tool_calls` envelope or broken `index` / `id` handling
- grammar-constrained mode returns output that violates the grammar or schema
- `tool_choice=none` or `required` semantics are broken by the server path
- stream/non-stream deterministic mismatch for the same valid case
- parser profile changes a valid call into an invalid one
- server crash, timeout, truncation, duplicate emission, wrong finish reason
- guided-json mode returns invalid JSON or wrong response envelope

### 2 = model_quality

Use `model_quality` when AFM's protocol and parsing are correct, but the model
behavior is weak.

Examples:
- wrong tool selected
- tool should have been called but was not
- tool should not have been called but was
- arguments are structurally valid but semantically wrong
- refusal/ambiguity handling is poor
- multi-turn tool use loses context
- coding-agent style decision quality is weak

### 3 = harness_bug

Use `harness_bug` when the evaluation machinery is wrong, even if AFM and the
model behavior are fine.

Examples:
- assertion logic marks a correct AFM output as failed
- provider normalization bug
- Promptfoo config mismatch
- judge/reporting artifact bug

## Matrix Dimensions

The suite should avoid a full Cartesian product. Instead, use a layered matrix
with explicit interdependencies.

### Core Server Factors

- `stream`: `false`, `true`
- `concurrent`: `1`, `8`
- `prefix_caching`: `off`, `on`
- `guided_json`: `off`, `on`
- `tool_call_parser`: `default`, `afm_adaptive_xml`
- `grammar_constraints`: `off`, `on`
- `fix_tool_args`: `off`, `on`
- `no_think`: `off`, `on`

### Interdependency Rules

- `grammar_constraints=on` only when `tool_call_parser=afm_adaptive_xml`
- `guided_json=on` is tested separately from tool-calling cases
- quality-focused cases should start with `concurrent=1`
- `concurrent=8` is primarily for AFM functional and regression checks
- deterministic cases should use `temperature=0` and fixed `seed`

## Matrix Families

### A. AFM Functional Matrix

Purpose:
- validate feature plumbing
- isolate AFM bugs

Profiles:

1. `plain-control`
- no tools
- no guided-json
- stream off/on
- concurrent 1/8
- prefix caching off/on

2. `guided-json`
- guided-json on
- no-think default and forced
- stream off/on
- concurrent 1

3. `tool-default`
- default parser
- stream off/on
- concurrent 1/8

4. `tool-adaptive`
- `afm_adaptive_xml`
- `fix_tool_args` off/on
- stream off/on
- concurrent 1/8

5. `tool-adaptive-grammar`
- `afm_adaptive_xml`
- grammar on
- `fix_tool_args` off/on
- stream off/on
- concurrent 1/8

### B. Model Quality Matrix

Purpose:
- measure model reliability for agentic workloads

Profiles:
- serial first (`concurrent=1`)
- deterministic exact-match cases
- low-temperature realistic agent cases
- parser-specific quality comparisons only where they are expected to matter

### C. Multi-turn Agentic Matrix

Purpose:
- simulate real AFM serving scenarios for agents

Profiles:
- tool result continuation
- chained tool calls
- constrained follow-up tool use
- coding-agent workflows

## Public Benchmark Sources

These should be sampled and adapted, not imported wholesale.

### Tool Calling

- BFCL / Berkeley Function Calling Leaderboard
- When2Call
- xLAM / APIGen
- Glaive function-calling
- NESTFUL
- IFEval-FC

### Agentic / Multi-turn

- tau-bench
- DICE-BENCH
- OpenHands benchmark task shapes

### Structured Output

- StructEval
- llm-structured-output stress ideas

## Real-World AFM Use Cases to Mirror

- coding agents:
  - OpenCode
  - OpenHands
  - Pi-like repository assistants
- agent frameworks:
  - OpenClaw
  - Hermes Agent
- productivity / support:
  - multi-tool planning
  - customer-support style policy and action tasks

## Promptfoo Harness Requirements

Before the full suite is expanded, the harness must support:

- normalized OpenAI tool-call arguments
- exact function/argument assertions on parsed objects
- raw response artifact capture
- streaming transcript capture
- deterministic re-run support
- failure classification field in each test result

## Recommended Layout

Under `Scripts/feature-promptfoo-agentic/`:

- `datasets/functional/`
- `datasets/quality/`
- `datasets/agentic/`
- `matrix/`
- `asserts/`
- `judges/`

## Phased Rollout

### Phase 1

- solidify provider normalization
- add custom assertions
- add matrix metadata
- keep datasets small and hand-curated

### Phase 2

- import sampled BFCL / When2Call / StructEval-style cases
- add streaming-specific assertions
- add AI judge for `not_yet_classified`

Status:
- implemented as representative local cases in:
  - `datasets/quality/toolcall-bfcl-when2call.yaml`
  - `datasets/quality/structured-stress.yaml`
  - `datasets/agentic/coding-workflows.yaml`
  - `datasets/agentic/framework-tool-schemas.yaml`
  - `datasets/agentic/opencode-primary-tools.yaml`
- these are benchmark-inspired, hand-curated cases rather than wholesale
  dataset imports

Primary-source note:
- `opencode-primary-tools.yaml` is sourced directly from the official OpenCode
  built-in tools documentation and includes explicit provenance metadata on
  every case.

### Phase 3

- add multi-turn coding-agent and framework-style tasks
- add summarized reports grouped by:
  - AFM bug
  - model quality
  - flaky / needs review
