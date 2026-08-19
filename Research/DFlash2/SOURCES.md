# DFlash 2 Sources

Inspected 2026-08-19. Primary and official sources are listed first.

## Primary Sources

1. [Inco AI, "DFlash 2: Keep Drafting Parallel"](https://inco.ai/blog/dflash2/),
   published 2026-08-18.
   - Confirms one-pass parallel drafting remains intact.
   - Defines the adjacent-candidate selector and block-local stateless two-tap
     dynamic convolution around every draft attention and MLP sublayer.
   - Reports upstream acceptance/latency results and links the engine releases.
   - oMLX example uses a Qwen 4-bit target, DFlash 2 draft quantization,
     runtime block size 5, and `verify_mode=dflash`.

2. [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
   revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
   - [config.json](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json)
   - [model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/README.md)
   - Safetensors header inspected with an HTTP range request; 81 tensors,
     3,848,817,896-byte payload.
   - Card declares base model `Qwen/Qwen3.8-27B`, block size 8/seven draft
     tokens, and lossless greedy/distribution-preserving sampling.

3. [incoai/Muse-Glimmer-30B-DFlash2](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2),
   revision `8336acb8dc9b8bf9c25f12d7785ee6df26703119`.
   - [config.json](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2/blob/main/config.json)
   - [model card](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2/blob/main/README.md)
   - Safetensors header inspected with an HTTP range request; 81 tensors,
     5,544,328,424-byte payload.
   - Card declares base model `meta-models/Muse-Glimmer-30B`, block size
     16/fifteen draft tokens, and derivation from Meta's assistant drafter.

4. [Qwen/Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
   - Target is top-level `qwen3_5`, text `qwen3_5_text`, hidden size 5120,
     64 layers, vocabulary 248320, and MTP metadata.

5. [meta-models/Muse-Glimmer-30B config](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/config.json).
   - Target is top-level `muse_glimmer`, text `muse_glimmer_text`, hidden size
     6656, 52 layers, vocabulary 202048, soft cap/output multiplier metadata.

6. [Chen, Liang, Liu, "DFlash: Block Diffusion for Flash Speculative Decoding"](https://arxiv.org/abs/2602.06036),
   arXiv v2, 2026-05-28, accepted at ICML 2026.
   - Defines original DFlash target feature capture, KV injection, one-pass
     block drafting, verification, training, and acceptance-length terminology.
   - Original paper does not define the DFlash 2 selector/convolution additions.

7. [oMLX 0.6.2 DFlash 2 release](https://github.com/z-lab/omlx-fork/releases/tag/0.6.2-dflash2),
   tag object `009e6e2645ea51d72520a5be05e6f8df7210e2e2`, commit
   `46225aebee34967d4f4bbb669bf02fc4e2de696a`.
   - Public oMLX integration adds a runtime block-size setting, passes sampling
     controls into DFlash, and documents single-stream operation/fallback.
   - It pins DFlash MLX implementation commit
     `415cc48d83846cfcd0d5b9da3c83e4f1478acda6`.
   - The pinned `z-lab/dflash-mlx` repository returned 404 during inspection.
     The implementation source is therefore unavailable through that pin; no
     production code will assume undocumented behavior from the binary release.
   - Release DMG SHA-256 from GitHub metadata:
     `94f56e14bfa8188d47e187f571bc61244a65010fb23d3601a28a2e22d5e5bd21`.

## Current Repository Sources

- `Sources/AFMKitCore/AFMCoreTypes.swift`
- `Sources/AFMKit/AFMEngine.swift`
- `Sources/AFMKitMLX/AFMMLXRuntime.swift`
- `Sources/AFMKitMLX/AFMMLXRuntimeAdapter.swift`
- `Sources/AFMKitMLX/AFMMLXSpeculativeDecoding.swift`
- `Sources/AFMKitMLX/AFMMLXSpeculativeRuntimeResourceResolver.swift`
- `Sources/AFMKitMLX/AFMMLXSpeculativeRuntimeSetupPlanner.swift`
- `Sources/AFMKitMLX/AFMMLXProvider.swift`
- `Sources/AFMKitMLX/Models/MLXModelService.swift`
- `Sources/AFMKitMLX/Models/BatchScheduler.swift`
- `Sources/AFMKitMLX/Models/StatsAggregator.swift`
- `Sources/AFMKitDwarfStar/AFMDwarfStarProvider.swift`
- `Sources/AFMKitDwarfStar/AFMDwarfStarScheduler.swift`
- `Sources/AFMOpenAICompat/OpenAIRequest.swift`
- `Sources/AFMServer/Controllers/MLXChatCompletionsController.swift`
- `Sources/AFMServer/Services/AFMLocalClient.swift`
- `Sources/AFMCLI/main.swift`
- `Scripts/apply-mlx-patches.sh`
- `Scripts/build-mlx-swift-lm-fork.sh`
- `Scripts/check-mlx-source-selection.sh`
- existing speculative, MTP, EAGLE3, DSpARK, Qwen 3.8, Muse, streaming,
  cancellation, prefix-cache, and batch tests under `Tests/MacLocalAPITests`.

## Source Handling Rules

- Repository IDs from model cards are test matrix inputs, never runtime
  architecture detectors.
- Blog/release speed numbers are attributed upstream and are not maclocal-api
  claims.
- No source changes will be made in oMLX, MLX, or other upstream worktrees.
- Dependency changes must be represented by maclocal-api patch inputs with an
  upstream revision/hash guard and clean-application test.

