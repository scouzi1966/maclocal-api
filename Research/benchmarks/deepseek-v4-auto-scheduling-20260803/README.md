# DeepSeek V4 automatic Metal scheduling

Release validation of the architecture- and hardware-gated MLX scheduling
policy on an Apple M3 Ultra. No `MLX_MAX_OPS_PER_BUFFER` or
`MLX_MAX_MB_PER_BUFFER` variables were set when the server launched.

- Model: `Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`
- Engine: `--mlx-kernels native`
- Policy selected from logs: 200 operations, 400 MB
- Prompt: `Explain how virtual memory works, including page tables, TLBs, and page faults.`
- Decode: greedy, seed 42, 256 tokens, no thinking
- Runs: 26.42, 26.47, and 26.46 tok/s
- Control with MLX 50/50 defaults: 25.13-25.31 tok/s
- Response SHA-256: `54b9989d8acc60cff5f9ea1025853f8b2a65ff6ae1cde21de361de86d808b7da`

The three `run-N.json` files contain the complete OpenAI-compatible responses
and timing records.
