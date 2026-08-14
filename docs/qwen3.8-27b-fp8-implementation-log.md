# Qwen3.8-27B-FP8 Implementation Log

Date: 2026-08-14
Branch: `codex/qwen3.8-27b-fp8`
Lock status: source-only edits completed under active test lock; no build, test, model load, download, conversion, or upload executed.

## Scope handled

- Reviewed AFM's current Qwen architecture support and MLX loader constraints.
- Inspected the exact official FP8 config and the authoritative MLX conversions.
- Confirmed Qwen3.8 retains the existing `qwen3_5` model contract implemented by AFM.
- Added only pre-download name routing plus deterministic config-backed tests.
- Documented why the raw Transformers FP8 checkpoint is not the AFM runtime artifact.

## Architecture findings

### Authoritative evidence inspected

- Requested checkpoint: [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
  - Inspected revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Official base checkpoint: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- Authoritative MLX conversions:
  - [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
    - Inspected revision: `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
  - [`mlx-community/Qwen3.8-27B-mxfp8`](https://huggingface.co/mlx-community/Qwen3.8-27B-mxfp8)
    - Inspected revision: `d48d163bcdf24acaf656474854ab88ea17d65bd1`
  - [`mlx-community/Qwen3.8-27B-bf16`](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16)
- Comparison baseline: `Qwen/Qwen3.6-27B`
- Current AFM implementation:
  - `Scripts/patches/Qwen3_5MoE.swift`
  - `Scripts/patches/Qwen3VL.swift`
  - `Scripts/patches/LLMModelFactory.swift`
  - `Scripts/patches/VLMModelFactory.swift`

### Exact published architecture

- `Qwen/Qwen3.8-27B-FP8` publishes `architectures: ["Qwen3_5ForConditionalGeneration"]`.
- Its top-level `model_type` is `qwen3_5`; the text tower is `qwen3_5_text`.
- It is multimodal: top-level config includes `text_config`, `vision_config`, and `image_token_id`.
- The text tower matches the public Qwen3.6-27B shape already implemented in AFM:
  - GatedDeltaNet / full-attention interleave
  - 64 layers
  - 5120 hidden size
  - 24 Q heads / 4 KV heads in full attention
  - 16 QK heads / 48 V heads in linear attention
  - 48 linear-attention layers and 16 full-attention layers
  - 262144 context
- The official FP8 checkpoint includes one MTP layer and an `mtp.safetensors` sidecar.
- Its Transformers FP8 scheme is E4M3, dynamically scaled, with a 128x128 weight block.
- The authoritative MLX 4-bit artifact was converted from `Qwen/Qwen3.8-27B` with `mlx-vlm` 0.6.8 and includes both language and vision weights. It is not a byte-for-byte conversion of the separate Transformers FP8 repository.

### Prompt, reasoning, and tool contract

- The published `chat_template.jinja` defaults `enable_thinking` to true.
- `enable_thinking=false` emits an empty `<think>\n\n</think>` section before the answer.
- Tool calls use the Qwen XML function format already selected by AFM for `qwen3_5`:
  `<tool_call><function=name><parameter=name>value</parameter></function></tool_call>`.
- The published generation defaults are `temperature: 1.0`, `top_k: 20`, and `top_p: 0.95`.

### AFM consequence

- Qwen3.8 routes through AFM's existing `Qwen3_5MoE` / `Qwen3VL` support. It does not require a new `qwen3_8` Swift architecture or model-factory registration.
- The source changes deliberately do not invent `qwen3_8` config aliases. Compatibility is established from the published config, not inferred from the repository name.

## Public artifact status

### `Qwen/Qwen3.8-27B-FP8`

- The exact repository is public and its config, tokenizer metadata, safetensors index, and MTP sidecar metadata are inspectable.
- The repo is a Transformers FP8 artifact, not an MLX-formatted artifact.

### MLX equivalent

- Suitable authoritative artifacts exist. The primary Release validation target is `mlx-community/Qwen3.8-27B-4bit` because it uses AFM's established affine 4-bit path and preserves the multimodal checkpoint.
- `mlx-community/Qwen3.8-27B-mxfp8` is the closer numeric analogue to FP8 and is the secondary validation target for MLX block-scaled 8-bit coverage.
- Both publish the same `qwen3_5` / `Qwen3_5ForConditionalGeneration` architecture and preserve the language and vision towers.
- Their safetensors indexes expose the 64-layer `language_model.model.layers.*` namespace and all 27 `vision_tower.blocks.*` groups. Neither index contains MTP tensors.

## FP8 viability

- Direct loading of the raw Hugging Face Transformers FP8 repo is **not the supported AFM MLX path**.
- AFM's loader expects MLX-formatted checkpoints and optional MLX quantization metadata.
- The raw Qwen FP8 repos publish Transformers-style `quantization_config` (`quant_method: "fp8"`, `fmt: "e4m3"`), which is not the same as an MLX snapshot.
- AFM's special packed-weight inference in `Scripts/patches/Load.swift` is targeted at MLX block-scaled tensor layouts (`mxfp4` / `mxfp8`) after weight loading, not generic Transformers FP8 repos.

## Source changes made

### Pre-download compatibility

- `Sources/AFMKitMLX/AFMMLXModelArchitecture.swift`
  - Added `qwen3.8-` / `qwen3.8_` to dual-mode repo-name heuristics.
  - Rationale: the exact official and MLX repositories are now verified as multimodal `qwen3_5` checkpoints, so the pre-config UI heuristic can route their names consistently with Qwen3.5/Qwen3.6.

- `Sources/AFMKitMLX/AFMMLXModelCatalog.swift`
  - Added `mlx-community/Qwen3.8-27B-4bit` as the primary curated text+vision checkpoint.
  - Added `mlx-community/Qwen3.8-27B-mxfp8` as the closer FP8-format comparison checkpoint.
  - Uses the published generation defaults (`temperature: 1.0`, `top_p: 0.95`) and a conservative 32768-token catalog limit pending long-context Release validation.

### Deterministic tests

- `Tests/MacLocalAPITests/AFMMLXModelArchitectureTests.swift`
  - Added the repo-name heuristic assertion.
  - Added a config-backed preflight test using the published Qwen3.8 model type, architecture, text/vision layout, and MLX MXFP8 metadata.
  - Added direct `Qwen3_5MoEVLConfiguration` decoding coverage using the published dense text and vision dimensions.

- `Tests/MacLocalAPITests/AFMMLXSpeculativeDecodingTests.swift`
  - Added MTP compatibility coverage using the actual published `qwen3_5` / `Qwen3_5ForConditionalGeneration` contract.

- `Tests/MacLocalAPITests/AFMMLXModelCatalogTests.swift`
  - Added catalog ordering, text+vision capability, generation preset, and runtime-configuration coverage for the authoritative 4-bit artifact.

## Deferred conversion plan

No custom conversion or `scouzi1966` upload is required because authoritative `mlx-community` artifacts already exist.

### Proposed target repo

- Proposed only as a fallback if the authoritative artifacts prove defective: `scouzi1966/Qwen3.8-27B-4bit`

If a VLM-preserving MLX conversion is produced and kept public:

- `scouzi1966/Qwen3.8-27B-4bit`
- optional sidecar sibling if MTP extraction is needed later:
  - `scouzi1966/Qwen3.8-27B-MTPLX`

### Locked-out procedure to run later

1. Capture the source:
   - `config.json`
   - `tokenizer_config.json`
   - `generation_config.json`
   - `model.safetensors.index.json`
2. Treat the source as multimodal and use the VLM conversion path rather than text-only `mlx-lm`.
3. Preserve the 27-layer vision tower, 64-layer text tower, tokenizer/processor metadata, and MTP sidecar separately.
4. Preserve source provenance in the destination model card and commit notes.
5. Re-run AFM preflight and release validation against the converted MLX snapshot, not the raw FP8 repo.

### Release-only validation commands to run after unlock

These are intentionally not executed under the lock.

```bash
git branch --show-current
```

```bash
Scripts/swiftpm-reliable.sh test -c release --filter AFMMLXModelArchitectureTests
```

```bash
Scripts/swiftpm-reliable.sh test -c release --filter AFMMLXSpeculativeDecodingTests
```

Primary model validation after unlock:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
./.build/release/afm mlx -m mlx-community/Qwen3.8-27B-4bit --port 9999
```

```bash
./Scripts/test-assertions.sh --tier smoke --model mlx-community/Qwen3.8-27B-4bit --port 9999
```

If a VLM conversion is involved:

```bash
./Scripts/test-assertions.sh --tier smoke --model mlx-community/Qwen3.8-27B-4bit --port 9999 --section vlm
```

Secondary MXFP8 validation after the 4-bit baseline passes:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
./.build/release/afm mlx -m mlx-community/Qwen3.8-27B-mxfp8 --port 9999
```

## Outstanding blockers

- The active lock prohibits the Release build/test/model-load steps needed to confirm runtime compatibility.
- The authoritative MLX repos were published on 2026-08-14; they still require actual AFM text, vision, streaming, tool-call, structured-output, concurrency, and prefix-cache validation.
- The `mlx-community` conversions do not publish `mtp.safetensors`, so MTP must be treated as unavailable unless a separately compatible sidecar is produced and validated later.
