# Hermes (Nous Research agent framework)

[Hermes](https://github.com/NousResearch/Hermes-Function-Calling) is Nous Research's open agent framework. It works with any OpenAI-compatible local provider.

> Disambiguation: this page is about the Hermes **agent framework**, not the Nous-Hermes family of LLMs (those would be the "model"). afm also supports the `hermes` tool-call format used by Hermes-derived models, auto-detected when `model_type` matches.

## 1. Start the server

```bash
afm mlx -m mlx-community/Qwen3-Coder-Next-4bit \
  --port 9999 --enable-prefix-caching --concurrent 4
```

## 2. Configure Hermes

Set the OpenAI base URL to afm:

```bash
export OPENAI_API_BASE=http://localhost:9999/v1
export OPENAI_API_KEY=x
```

Or in Hermes config (e.g. YAML/JSON depending on your setup):

```yaml
provider:
  type: openai
  base_url: http://localhost:9999/v1
  api_key: x
  model: mlx-community/Qwen3-Coder-Next-4bit
```

## 3. Run Hermes

Hermes will issue chat completions with tool calling. afm's auto-detected tool format and harmony/think reasoning extraction work with Hermes' loop.

## Tips

- For Hermes' self-improvement loops (trajectory generation), enable structured output: pass `response_format: {"type":"json_schema","json_schema":{...,"strict":true}}` and turn on `--enable-grammar-constraints` server-side.
- Multi-step trajectories benefit from `--enable-prefix-caching`.
- Deterministic seeds: pass `seed: <int>` per request — afm threads it through end-to-end (logged via `--vv` for reproducibility).
