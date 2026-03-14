# Guided JSON Evals

Use `Scripts/feature-codex-optimize-api/test-guided-json-evals.py` to validate AFM structured outputs against OpenAI-style strict `json_schema` behavior.

Scope:

- API `response_format: {type: "json_schema", json_schema: {strict: true, ...}}`
- API streaming `json_schema` with final usage-chunk validation
- `openai-python` `beta.chat.completions.parse(...)` compatibility
- optional CLI `--guided-json` validation for the same schemas
- optional CLI `--guided-json --no-think` validation for the same schemas
- real-world cases instead of only toy objects
- conflict/refusal behavior when the prompt tells the model not to use JSON
- truncation behavior under intentionally low `max_tokens`
- schema-edge coverage including nullable fields, bounded arrays, `anyOf`, and unsupported-schema rejection

Fixtures live in [guided-json-cases.json](/Volumes/edata/codex/dev/git/maclocal-api/Scripts/feature-codex-optimize-api/guided-json-cases.json) and cover:

- person record extraction
- meeting action items
- support ticket triage
- article summarization with entities
- product catalog normalization with `$defs` / `$ref`
- streaming structured person extraction
- conflict-prompt structured output behavior
- truncation detection on a larger summary schema
- schema edges: nullable field, bounded enum array, `anyOf`, unsupported `patternProperties`

These cases are aligned to OpenAI structured-output expectations:

- strict schema mode
- required fields
- enums
- nested arrays and objects
- `additionalProperties: false`
- `$defs` / `$ref`

Reuse an existing server:

```bash
python3 Scripts/feature-codex-optimize-api/test-guided-json-evals.py \
  --base-url http://127.0.0.1:9999/v1 \
  --model mlx-community/Qwen3.5-35B-A3B-4bit
```

Start a temporary MLX server as part of the run:

```bash
python3 Scripts/feature-codex-optimize-api/test-guided-json-evals.py \
  --start-server \
  --server-mode mlx \
  --port 10000 \
  --base-url http://127.0.0.1:10000/v1 \
  --model mlx-community/Qwen3.5-35B-A3B-4bit
```

Start a temporary Foundation server and run the same API checks against the Foundation controller:

```bash
python3 Scripts/feature-codex-optimize-api/test-guided-json-evals.py \
  --start-server \
  --server-mode foundation \
  --port 10001 \
  --base-url http://127.0.0.1:10001/v1 \
  --model foundation
```

That starts the root `afm` server path, equivalent to:

```bash
afm --port 10001
```

Also run direct CLI `--guided-json` checks:

```bash
python3 Scripts/feature-codex-optimize-api/test-guided-json-evals.py \
  --base-url http://127.0.0.1:9999/v1 \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --run-cli
```

Also run the same CLI checks with `--no-think`:

```bash
python3 Scripts/feature-codex-optimize-api/test-guided-json-evals.py \
  --base-url http://127.0.0.1:9999/v1 \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --run-cli-no-think
```

Run both CLI variants together:

```bash
python3 Scripts/feature-codex-optimize-api/test-guided-json-evals.py \
  --base-url http://127.0.0.1:9999/v1 \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --run-cli \
  --run-cli-no-think
```

Reports are written to `Scripts/feature-codex-optimize-api/results/guided-json-evals-*.json`.

Reference material:

- OpenAI Structured Outputs guide: `platform.openai.com/docs/guides/structured-outputs`
- OpenAI Cookbook structured outputs intro: `cookbook.openai.com/examples/structured_outputs_intro`

This bundle is designed to catch regressions in schema conformance and SDK behavior, not benchmark throughput.
