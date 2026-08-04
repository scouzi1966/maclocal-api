# DeepSeek V4 compiled attention-prefix benchmark

Release binary, native MLX kernels, temperature 0, prefix cache disabled, and
three consecutive 256-token non-streaming requests against the same loaded
Vontra DeepSeek V4 Flash 0731 MXFP4 model.

| Run | Prefill | Decode | Output SHA-256 |
| --- | ---: | ---: | --- |
| 1 | 164.6 tok/s | 25.4 tok/s | `5640c41f44fa7566a2b62e757167c8f399635df1c31d88d31d5132021594b03a` |
| 2 | 164.1 tok/s | 25.3 tok/s | `5640c41f44fa7566a2b62e757167c8f399635df1c31d88d31d5132021594b03a` |
| 3 | 163.7 tok/s | 25.4 tok/s | `5640c41f44fa7566a2b62e757167c8f399635df1c31d88d31d5132021594b03a` |

The previous validated checkpoint measured 23.5, 23.7, and 23.7 tok/s. The
candidate therefore improves repeated decode by approximately 7-8% while
preserving deterministic output. `DeepseekV40731EncodingTests` passed 10/10 in
Release configuration.
