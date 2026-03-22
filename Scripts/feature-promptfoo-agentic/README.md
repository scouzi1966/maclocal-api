# Promptfoo Agentic Suite

This directory contains the first Promptfoo-based evaluation scaffold for AFM's
agentic and structured-output features.

Goals:
- evaluate AFM tool calling across parser and grammar configurations
- evaluate structured output across API `json_schema` and CLI `--guided-json`
- reuse AFM's existing test cases instead of creating a parallel test universe
- make it easy to expand into BFCL-style tool calling and larger schema suites

## Layout

- `promptfooconfig.structured.yaml`
  Structured-output suite. Includes API `response_format=json_schema` and CLI
  `--guided-json` providers.
- `promptfooconfig.toolcall.yaml`
  Tool-calling suite. Designed to compare default, `afm_adaptive_xml`, and
  `afm_adaptive_xml + --enable-grammar-constraints` server profiles.
- `promptfooconfig.toolcall-quality.yaml`
  BFCL- and When2Call-inspired tool-choice quality suite. Uses normalized
  message output so both tool calls and direct/follow-up responses can be
  asserted.
- `promptfooconfig.structured-stress.yaml`
  Harder structured-output suite with nested schemas, refs, enums, and CLI
  `--guided-json` stress cases.
- `promptfooconfig.agentic.yaml`
  Multi-turn coding-agent suite inspired by OpenCode, OpenHands, and Hermes
  agent workflows.
- `promptfooconfig.agentic-frameworks.yaml`
  Framework-schema suite built around real-world tool shapes from OpenCode,
  Pi, OpenClaw, and Hermes-style agents.
- `promptfooconfig.opencode.yaml`
  Primary-source-derived OpenCode suite based directly on the official built-in
  tools documentation.
- `providers/afm_provider.mjs`
  Custom Promptfoo provider for AFM. Supports:
  - HTTP OpenAI-style chat completions
  - direct CLI `--guided-json` invocation
- `datasets/structured-core.yaml`
  Starter structured-output cases derived from
  `Scripts/feature-codex-optimize-api/guided-json-cases.json`.
- `datasets/toolcall-core.yaml`
  Starter tool-call cases derived from `Scripts/test-assertions.sh` Section 12.
- `datasets/quality/toolcall-bfcl-when2call.yaml`
  Public-benchmark-inspired tool-choice quality cases.
- `datasets/quality/structured-stress.yaml`
  StructEval-style structured-output stress cases.
- `datasets/agentic/coding-workflows.yaml`
  Multi-turn coding-agent tool-use cases.
- `datasets/agentic/framework-tool-schemas.yaml`
  Agent-framework-specific tool schema cases.
- `datasets/agentic/opencode-primary-tools.yaml`
  OpenCode cases with explicit primary-source provenance.

## Environment

The configs expect these environment variables:

- `AFM_MODEL`
  Default model id to send to AFM.
- `AFM_BASE_URL_DEFAULT`
  Base URL for a normal AFM server, e.g. `http://127.0.0.1:9999/v1`.
- `AFM_BASE_URL_ADAPTIVE_XML`
  Base URL for AFM started with `--tool-call-parser afm_adaptive_xml`.
- `AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR`
  Base URL for AFM started with:
  `--tool-call-parser afm_adaptive_xml --enable-grammar-constraints`.

Optional for CLI structured-output checks:

- `AFM_BINARY`
  Path to the `afm` binary. Defaults to `.build/arm64-apple-macosx/release/afm`.
- `MACAFM_MLX_MODEL_CACHE`
  Local model cache root.
- `AFM_NO_THINK`
  Set to `1` to add `--no-think` in CLI guided-json mode.

## Suggested server matrix

Default server:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm mlx \
  -m mlx-community/Qwen3.5-35B-A3B-4bit \
  --port 9999
```

Adaptive XML server:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm mlx \
  -m mlx-community/Qwen3.5-35B-A3B-4bit \
  --tool-call-parser afm_adaptive_xml \
  --port 10000
```

Adaptive XML + grammar server:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/arm64-apple-macosx/release/afm mlx \
  -m mlx-community/Qwen3.5-35B-A3B-4bit \
  --tool-call-parser afm_adaptive_xml \
  --enable-grammar-constraints \
  --port 10001
```

## Running

Run commands from the repository root:

```bash
cd /Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api
```

One-command wrapper:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh
```

That wrapper:
- runs the structured core suite once against the default AFM profile
- runs the structured stress suite once against the default AFM profile
- runs the tool-calling core suite three times:
  - default parser
  - `afm_adaptive_xml`
  - `afm_adaptive_xml + grammar`
- runs the BFCL / When2Call-inspired tool-quality suite three times
- runs the multi-turn coding-agent suite three times
- runs the framework-schema suite three times
- runs the OpenCode primary-source suite three times
- writes JSON reports under `test-reports/promptfoo-agentic/`
- starts and stops one AFM server at a time

Useful variants:

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

Structured-output suite:

```bash
AFM_MODEL=mlx-community/Qwen3.5-35B-A3B-4bit \
AFM_BASE_URL_DEFAULT=http://127.0.0.1:9999/v1 \
promptfoo eval -c Scripts/feature-promptfoo-agentic/promptfooconfig.structured.yaml
```

Tool-calling suite:

```bash
AFM_MODEL=mlx-community/Qwen3.5-35B-A3B-4bit \
AFM_BASE_URL_DEFAULT=http://127.0.0.1:9999/v1 \
AFM_BASE_URL_ADAPTIVE_XML=http://127.0.0.1:10000/v1 \
AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR=http://127.0.0.1:10001/v1 \
promptfoo eval -c Scripts/feature-promptfoo-agentic/promptfooconfig.toolcall.yaml
```

Validate configs without running the full suite:

```bash
promptfoo validate config -c Scripts/feature-promptfoo-agentic/promptfooconfig.structured.yaml
promptfoo validate config -c Scripts/feature-promptfoo-agentic/promptfooconfig.structured-stress.yaml
promptfoo validate config -c Scripts/feature-promptfoo-agentic/promptfooconfig.toolcall.yaml
promptfoo validate config -c Scripts/feature-promptfoo-agentic/promptfooconfig.toolcall-quality.yaml
promptfoo validate config -c Scripts/feature-promptfoo-agentic/promptfooconfig.agentic.yaml
promptfoo validate config -c Scripts/feature-promptfoo-agentic/promptfooconfig.agentic-frameworks.yaml
promptfoo validate config -c Scripts/feature-promptfoo-agentic/promptfooconfig.opencode.yaml
```

## Current scope

This is the initial scaffold, not the final benchmark:

- enough coverage to validate provider/config shape
- enough cases to compare parser/grammar profiles
- structured and tool-call datasets are intentionally compact

Current expansions include:
- BFCL- and When2Call-inspired tool-call quality cases
- larger schema stress cases inspired by StructEval
- multi-turn coding-agent chains inspired by OpenCode, OpenHands, and Hermes
- framework-specific tool schemas inspired by OpenCode, Pi, OpenClaw, and Hermes
- primary-source-derived OpenCode built-in tool coverage

Still missing:
- streaming transcript assertions
- public benchmark imports beyond hand-curated representative cases
- latency and failure-bucket reporting

## Matrix and Classification

The broader design for the full suite lives in:

- [docs/roadmap/promptfoo-agentic-matrix.md](/Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api/docs/roadmap/promptfoo-agentic-matrix.md)

Starter matrix artifacts:

- `matrix/functional-matrix.yaml`
- `matrix/failure-classification.yaml`
- `datasets/quality/toolcall-quality-core.yaml`

## AI Classification

Post-run AI failure classification is available via:

- `judges/classify-failures.mjs`
- `judges/failure-classifier-prompt.md`

Run it from the repo root after a Promptfoo eval:

```bash
AFM_JUDGE_MODEL=mlx-community/Qwen3.5-35B-A3B-4bit \
AFM_JUDGE_BASE_URL=http://127.0.0.1:9999/v1 \
node Scripts/feature-promptfoo-agentic/judges/classify-failures.mjs \
  test-reports/promptfoo-agentic/toolcall-default-mlx-community_Qwen3.5-35B-A3B-4bit.json
```

That writes:
- `*.classified.json`
- `*.classified.summary.md`
