# OpenCode

[OpenCode](https://opencode.ai/) is a terminal-based AI coding assistant.

## 1. Start the server

```bash
afm mlx -m mlx-community/Qwen3-Coder-Next-4bit \
  -t 1.0 --top-p 0.95 --max-tokens 8192 \
  --enable-prefix-caching
```

Recommended:
- **Model**: `mlx-community/Qwen3-Coder-Next-4bit` (best small coder model with XML tool calling)
- **Prefix caching**: OpenCode reuses a long system prompt — `--enable-prefix-caching` makes turn-2+ near-instant
- **Concurrency**: add `--concurrent 4` if you run multiple OpenCode sessions

## 2. Configure OpenCode

`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "afm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "afm (local)",
      "options": { "baseURL": "http://localhost:9999/v1" },
      "models": {
        "mlx-community/Qwen3-Coder-Next-4bit": {
          "name": "mlx-community/Qwen3-Coder-Next-4bit"
        }
      }
    }
  }
}
```

## 3. Use it

In OpenCode: `/connect` → scroll to `afm (local)` → select. When prompted for an API key, type any value (afm doesn't gate on it).

## Notes

- Tool format is auto-detected from `model_type` in the model's `config.json`. For Qwen3-Coder this is `xmlFunction`.
- For zero-parameter tool calls, three safety nets strip stray `</function` suffixes — see [issue #80](https://github.com/scouzi1966/maclocal-api/issues/80).
- If you want streaming usage suppressed, OpenCode does not currently send `stream_options.include_usage`; pass it through the request body if your client wrapper supports it.
