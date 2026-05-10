# Cursor

[Cursor](https://cursor.com/) is a coding editor with a built-in AI sidebar.

> **Caveat**: Cursor's "Custom OpenAI base URL" requires a publicly reachable HTTPS endpoint for some features (Composer, Chat). For local-only use, expose afm via a tunnel (Cloudflare Tunnel, Tailscale Funnel, ngrok) or use Cursor's local-models setting where available.

## 1. Start the server

```bash
afm mlx -m mlx-community/Qwen3-Coder-Next-4bit --port 9999
```

## 2. (Optional) expose via tunnel

```bash
# Cloudflare quick tunnel — replace with your tool of choice
cloudflared tunnel --url http://localhost:9999
```

Note the public HTTPS URL it prints.

## 3. Configure Cursor

Cursor settings → Models → "Add Model":

- **Model name**: `mlx-community/Qwen3-Coder-Next-4bit`
- **OpenAI Base URL**: `https://your-tunnel-url/v1` (or `http://localhost:9999/v1` if your build allows local)
- **API Key**: any value

## Tips

- Cursor sends large prompts (system + repo context); `--enable-prefix-caching` is a big win.
- For Composer multi-turn loops, `--concurrent 2` lets Cursor's parallel calls share the model.
