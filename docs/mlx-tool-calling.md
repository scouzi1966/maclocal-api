# MLX Tool-Calling Modes

AFM's MLX backend has three tool-calling modes. Choose the mode based on whether you want MLX Python parity, production robustness, or raw model inspection.

## Default Native Mode

Use no parser flag:

```bash
afm mlx -m mlx-community/Qwen3.6-35B-A3B-4bit --no-think
```

AFM auto-detects the model's native tool-call format from `config.json`. For Qwen XML tool models, the default parser is `qwen3_xml`.

This is the recommended mode for benchmark comparisons with `mlx_lm.server`:

- no AFM adaptive repair by default
- valid native Qwen XML becomes OpenAI-compatible `tool_calls`
- malformed tool output is not broadly coerced by AFM repair logic
- generation stops after a structured tool call inside the MLX producer, so the stream finishes cleanly before the next request

## Repair Mode

Use explicit repair flags:

```bash
afm mlx -m mlx-community/Qwen3.6-35B-A3B-4bit \
  --no-think \
  --tool-call-parser afm_adaptive_xml \
  --fix-tool-args
```

Repair mode is for real agent clients where robustness is more important than strict parity. It can salvage malformed XML/JSON, coerce argument types, and remap argument names.

Do not use this mode for MLX Python parity benchmarks unless you are intentionally measuring AFM's repair layer.

## Raw Mode

Disable AFM tool extraction:

```bash
afm mlx -m mlx-community/Qwen3.6-35B-A3B-4bit \
  --no-think \
  --tool-call-parser none
```

Raw mode returns generated tool markup as assistant text and emits no server-extracted `tool_calls`. It is useful for inspecting what the model produced before parsing.

Raw mode is usually not suitable for agent benchmarks such as VulcanBench because the harness expects structured tool calls to execute tools.

## Benchmark Guidance

For VulcanBench or other agent benchmarks:

1. Use default native mode for AFM.
2. Use the same model weights as Python MLX.
3. Align important generation settings such as thinking, temperature, and max tokens.
4. Compare traces turn by turn before interpreting scores.
5. Treat repair mode results separately from parity results.

Tool-call parsing determines whether an agent loop can read files, edit files, run commands, and continue after observations. If tool extraction differs between servers, benchmark scores are not directly comparable.
