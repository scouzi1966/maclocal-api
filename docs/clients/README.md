# Agent client cookbooks

One-page recipes for connecting popular agentic clients to `afm`. Each page assumes you've already started a server (e.g. `afm mlx -m mlx-community/Qwen3-Coder-Next-4bit`) on `http://localhost:9999`.

| Client | Page | Notes |
|---|---|---|
| OpenCode | [opencode.md](opencode.md) | Terminal coding assistant — already wired in main README |
| OpenClaw | [openclaw.md](openclaw.md) | `afm mlx --openclaw-config` generates the JSON for you |
| Cline | [cline.md](cline.md) | VSCode coding agent; OpenAI-compatible provider |
| Continue.dev | [continue.md](continue.md) | VSCode/JetBrains assistant; OpenAI-compatible |
| Aider | [aider.md](aider.md) | Git-aware CLI; `OPENAI_API_BASE` env var |
| Cursor | [cursor.md](cursor.md) | Editor; needs `OpenAI Base URL` setting |
| Hermes | [hermes.md](hermes.md) | Nous Research agent framework; OpenAI-compatible |

See the main [README — Why AFM for agents](../../README.md#why-afm-for-agents) for the capability matrix.
