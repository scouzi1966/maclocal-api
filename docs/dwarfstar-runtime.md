# DwarfStar Runtime

AFM consumes [DwarfStar](https://github.com/antirez/ds4) as an unchanged Git
submodule. AFM does not patch DwarfStar's model loader, sampler, cache, or Metal
kernels. The `CDwarfStar` target is an AFM-owned interface adapter around
DwarfStar's public C API. Interface-level compatibility code is permitted in
AFM-owned targets; no such code is copied into or applied over `vendor/ds4`.
Supported build scripts fail when that submodule has tracked, staged, or
untracked changes.

## Model Selection

The DwarfStar runtime accepts a local native GGUF file whose
`general.architecture` metadata is `deepseek4`. With `--mlx-runtime auto`, AFM
selects DwarfStar for that file without relying on its filename. Directory-based
MLX or AFM safetensor checkpoints remain on the MLX runtime.

AFM does not project safetensor shards into DwarfStar's address space. Supporting
that representation would require changing DwarfStar's loader internals and is
outside the dependency boundary. Convert or obtain a DwarfStar-compatible GGUF
to use the DwarfStar runtime.

## Supported Capabilities

- Text generation and streaming
- Temperature, top-k, top-p, min-p, seed, stop sequences, and token limits
- Concurrent resident sessions and batched prefill/decode
- Prefix caching
- DeepSeek thinking modes and budgets
- DwarfStar DSpark speculative decoding
- OpenAI-compatible HTTP transport, cancellation, usage, and timing

## Known Limitations

- Text only; no image or multimodal encoder
- Tool requests are rejected because vanilla DwarfStar does not expose AFM's
  tool-prompt and tool-event integration
- Repetition and presence penalties are unavailable because the public sampler
  does not expose mutable logits and token-frequency state
- Token logprobs and top-logprobs are unavailable because candidate
  probabilities are not exposed by the public session API
- Strict JSON schema and grammar-constrained decoding are unavailable because
  the public sampler has no per-token logit-mask hook
- KV-cache quantization is unavailable through AFM because DwarfStar owns its
  cache representation
- MLX MTP and EAGLE options do not apply; DwarfStar uses DSpark
- Reasoning generation is supported, but reasoning and final-answer event
  separation is not yet complete in the AFM adapter

These limitations are reported as capability errors. They do not modify or
replace the corresponding MLX functionality.
