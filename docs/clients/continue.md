# Continue.dev

[Continue.dev](https://www.continue.dev/) is an open-source autopilot for VSCode and JetBrains with broad OpenAI-compatible support.

## 1. Start the server

```bash
afm mlx -m mlx-community/Qwen3-Coder-Next-4bit \
  --port 9999 --enable-prefix-caching
```

## 2. Configure Continue

`~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "afm (local)",
      "provider": "openai",
      "model": "mlx-community/Qwen3-Coder-Next-4bit",
      "apiBase": "http://localhost:9999/v1",
      "apiKey": "x"
    }
  ],
  "tabAutocompleteModel": {
    "title": "afm-autocomplete",
    "provider": "openai",
    "model": "mlx-community/Qwen3-Coder-Next-4bit",
    "apiBase": "http://localhost:9999/v1",
    "apiKey": "x"
  }
}
```

## 3. Use it

Continue's chat, edit, and tab-autocomplete will all run against afm.

## Tips

- For autocomplete latency, prefer a smaller model (e.g. `mlx-community/Qwen3-1.7B-4bit`) on a separate port.
- Continue sends large repo-context prompts; `--enable-prefix-caching` is a big win.
- Tool calling: Continue uses tools for some workflows (search, edit). The auto-detected tool format handles it.
