# DeepSeek V4 Flash 0731: official vs Vontra

Release build, native MLX kernels, temperature 0, thinking disabled, 256-token cap.
Each pair was run against an already-loaded AFM server on the same M3 Ultra.

| Checkpoint | Decode mode | Prompt | Tokens/s |
| --- | --- | --- | ---: |
| Official source | autoregressive | Why is the sky blue? | 28.12 |
| Vontra MXFP4 MLX | autoregressive | Why is the sky blue? | 28.46 |
| Official source | autoregressive | Count upward | 27.55 |
| Vontra MXFP4 MLX | autoregressive | Count upward | 27.90 |
| Official source | DSpARK | Why is the sky blue? | 17.6 |
| Official source | DSpARK | Count upward | 47.2 |

The official checkpoint now loads directly. AFM reinterprets packed FP4/FP8
weights without requantization and expands official 128x128 FP8 scale blocks
to MLX's per-row, per-32-value scale layout. Direct loading does not materially
change steady-state autoregressive throughput relative to Vontra's conversion.

DSpARK generated natural prose at 43.1% draft acceptance (3.17 output tokens
per verifier round), making it slower than autoregressive decode. Counting had
100% acceptance (5.95 output tokens per round), demonstrating a 47.2 tok/s
best case that is not representative of ordinary prompts.
