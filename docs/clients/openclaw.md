# OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is an open coding agent that targets local models via the `openai-completions` API mode.

## 1. Generate the provider config

afm prints a paste-ready config for you:

```bash
afm mlx -m mlx-community/Qwen3-Coder-Next-4bit --openclaw-config
```

This emits a JSON block with `baseUrl`, `api: "openai-completions"`, model metadata (vision / reasoning detection, context window, max tokens), and zero-cost pricing fields. Copy it into your OpenClaw provider config.

## 2. Start the server

```bash
afm mlx -m mlx-community/Qwen3-Coder-Next-4bit \
  --port 9999 --enable-prefix-caching
```

(`--openclaw-config` defaults to port 9999 unless you pass `-p`.)

## 3. Run OpenClaw

OpenClaw will issue requests against `http://localhost:9999/v1`. Tool calling, streaming, and reasoning extraction all work out of the box.

## Notes

- The generated config detects `<think>` reasoning support automatically.
- Vision capability is reported as true for VLM-class model IDs.
- For multi-session use, add `--concurrent N` (each OpenClaw conversation can hold its own slot).
