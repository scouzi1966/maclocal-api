# DeepSeek V4 constant MXFP4 lookup table

Release validation of the native DeepSeek V4 MXFP4 kernels after replacing
per-threadgroup FP4 lookup-table copies and barriers with constant-memory
lookups.

- Model: `Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`
- Engine: `--mlx-kernels native`
- Metal scheduling: 200 operations, 400 MB
- Prompt: `Explain how virtual memory works, including page tables, TLBs, and page faults.`
- Decode: greedy, seed 42, 256 tokens, no thinking
- Runs: 26.72, 26.64, and 26.71 tok/s
- Previous validated checkpoint: 26.42, 26.47, and 26.46 tok/s
- Response SHA-256: `54b9989d8acc60cff5f9ea1025853f8b2a65ff6ae1cde21de361de86d808b7da`
- Release tests: `DeepseekV40731EncodingTests` 10/10;
  `AFMMLXModelArchitectureTests` 16/16

The three `run-N.json` files contain the complete OpenAI-compatible responses
and timing records.
