# Nemotron 3.5 Lightning

AFM supports `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-mxfp4`
through the MLX `nemotron_h` architecture path.

Validated behavior:

- streaming and non-streaming chat completions
- batch dispatch and concurrent generation
- XML and adaptive-XML tool calls
- strict JSON-schema structured output
- long prompts and deterministic seeded replay
- safe prefix-cache fallback for the hybrid Mamba/KV cache

The hybrid recurrent layers cannot be rewound like a conventional KV cache.
AFM therefore rejects unsafe partial recurrent-state reuse and performs a cold
prefill when necessary. The radix cache can identify matching prefixes and hold
snapshots, but AFM does not restore a hybrid recurrent snapshot unless the
architecture provides an exact, correctness-preserving replay boundary.

Release validation on Apple Silicon completed 182/182 comprehensive cases and
475/476 live assertions. The remaining live assertion was model behavior: the
model returned text instead of a required EBNF tool call. Promptfoo completed
388 cases with no transport or parser errors; its failures were semantic tool
selection or argument-quality mismatches rather than malformed protocol output.
