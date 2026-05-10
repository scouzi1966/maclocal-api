# Cline

[Cline](https://github.com/cline/cline) is a VSCode coding agent that supports any OpenAI-compatible provider.

## 1. Start the server

```bash
afm mlx -m mlx-community/Qwen3-Coder-Next-4bit \
  --port 9999 --enable-prefix-caching --concurrent 4
```

## 2. Configure Cline

In VSCode → Cline settings → API Provider:

- **API Provider**: OpenAI Compatible
- **Base URL**: `http://localhost:9999/v1`
- **API Key**: any value (afm ignores it)
- **Model ID**: the same model you started afm with, e.g. `mlx-community/Qwen3-Coder-Next-4bit`

## 3. Use it

Cline will issue streaming chat completions with tools. Tool format is auto-detected from `model_type`.

## Tips

- Cline's system prompt is large (~5–10k tokens) and stable across turns. Run with `--enable-prefix-caching` for instant turn-2+ prefill.
- For multiple Cline tabs, raise `--concurrent N` to match.
- Cline aborts mid-turn often (user edits, retries). Mid-stream cancel on client disconnect is on the [roadmap](../../docs/ROADMAP.md) (Tier 1.4); until landed, expect a few seconds of GPU idle after abort.
