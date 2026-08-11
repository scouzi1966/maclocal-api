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
- DeepSeek DSML tool definitions, tool calls, and tool-result continuation
- Parallel tool calls through both AFMKit events and the macOS 27 Foundation
  Models executor channel
- DeepSeek thinking modes and budgets
- DwarfStar DSpark speculative decoding
- OpenAI-compatible HTTP transport, cancellation, usage, and timing

## Continuous Batching

`--concurrent <n>` creates that many resident DwarfStar sessions. Decode-ready
sessions are evaluated together with DwarfStar's native batch API. New prompts
are admitted round-robin in bounded prefill quanta; while decode work is active,
AFM uses the upstream mixed prefill/decode API so a long prompt does not stop
tokens from otherwise ready requests. A fresh session first establishes the
checkpoint required by DwarfStar before it joins a mixed scheduling epoch.

## Prefix Cache

`--enable-prefix-caching` enables both resident exact-prefix reuse and
DwarfStar's persistent disk KV store. The default disk location is
`~/Library/Caches/AFM/DwarfStarPrefixCache/<checkpoint-key>`, isolated by model
path, size, and modification time. Set `AFM_DWARFSTAR_PREFIX_CACHE` to place it
on another volume and `AFM_DWARFSTAR_PREFIX_CACHE_MB` to change the default
4096 MB budget.

Disk restore/store can synchronize GPU state, so AFM performs those operations
only while the engine is otherwise idle. Concurrent traffic continues to use
resident session reuse without introducing disk I/O into the decode hot path.

## Known Limitations

- Text only; no image or multimodal encoder
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

Tool calling is implemented in AFM-owned interface code, without modifying the
DwarfStar dependency. AFM renders tool schemas and prior calls using DeepSeek's
DSML contract, passes tool results back as transcript messages, and converts
generated DSML blocks into typed `AFMGenerationEvent.toolCall` events. DwarfStar
still owns tokenization, sampling, cache state, batching, and Metal execution.

These limitations are reported as capability errors. They do not modify or
replace the corresponding MLX functionality.
