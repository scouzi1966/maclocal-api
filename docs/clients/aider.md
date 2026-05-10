# Aider

[Aider](https://aider.chat/) is a git-aware terminal pair-programmer.

## 1. Start the server

```bash
afm mlx -m mlx-community/Qwen3-Coder-Next-4bit \
  --port 9999 --enable-prefix-caching
```

## 2. Point Aider at afm

```bash
export OPENAI_API_BASE=http://localhost:9999/v1
export OPENAI_API_KEY=x
aider --model openai/mlx-community/Qwen3-Coder-Next-4bit
```

## 3. Use it

Aider commits/edits/reverts as usual. All inference is local.

## Tips

- Aider's repo-map prompt grows with codebase size; `--enable-prefix-caching` keeps subsequent turns fast.
- Aider doesn't stream tool-call deltas — it waits for the final assistant message — so streaming overhead is minimal either way.
- For very long sessions, add `--kv-bits 8` to halve the KV-cache memory footprint.
