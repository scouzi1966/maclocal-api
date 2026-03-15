# OpenAI Compat Evals

Use `Scripts/feature-codex-optimize-api/test-openai-compat-evals.py` to run a small reusable compatibility bundle against a local OpenAI-style endpoint.

It currently checks:

- `GET /v1/models`
- `openai-python` non-stream chat completions
- `openai-python` non-stream chat completions with `logprobs` and `top_logprobs`
- `openai-python` streaming chat completions with `stream_options.include_usage`
- `openai-python` streaming chat completions with `logprobs` and final usage chunk
- `vllm bench serve` smoke benchmark

Reuse an existing server:

```bash
python3 Scripts/feature-codex-optimize-api/test-openai-compat-evals.py \
  --base-url http://127.0.0.1:9999/v1 \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct
```

Start `afm` automatically for the eval:

```bash
python3 Scripts/feature-codex-optimize-api/test-openai-compat-evals.py \
  --start-server \
  --port 9999 \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --server-arg=--no-think
```

Reports are written to `Scripts/feature-codex-optimize-api/results/openai-compat-evals-*.json`.

This bundle is meant for fast SDK and wire-format validation, not full API conformance. Add targeted evals separately for tool calling, structured outputs, error payloads, and provider-specific SDK behavior.
