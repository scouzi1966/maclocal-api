# DeepSeek V4 native default benchmark

Hardware: M3 Ultra, 512 GB unified memory. Build: Release. Model:
`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX`. Engine:
`--mlx-kernels native`, temperature 0, no thinking, 256 output tokens.

| Candidate | Run 1 | Run 2 | Run 3 | Deterministic content SHA-256 |
|---|---:|---:|---:|---|
| Validated native defaults | 23.1 tok/s | 23.1 tok/s | 23.1 tok/s | `5640c41f44fa7566a2b62e757167c8f399635df1c31d88d31d5132021594b03a` |
| Fused attention-HC/FFN layer tail | 23.5 tok/s | 23.7 tok/s | 23.7 tok/s | `5640c41f44fa7566a2b62e757167c8f399635df1c31d88d31d5132021594b03a` |

The fused tail keeps the mutable attention/KV operation outside compilation,
then compiles the attention residual expansion, FFN hyper-connection collapse,
MoE/shared expert work, and FFN expansion as one decode graph. The result is a
small but repeatable gain with byte-identical generated content. It does not
close the remaining scheduling gap to the 35-37 tok/s goal.
