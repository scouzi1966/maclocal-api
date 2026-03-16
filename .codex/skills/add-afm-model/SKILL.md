---
name: add-afm-model
description: Investigate and add support for a new Hugging Face MLX model in AFM.
user_invocable: true
---

# Add AFM Model

Use this skill when the user wants to check support for or add a new MLX model.

## Workflow

### 1. Parse the model identifier

Accept either:
- `org/model`
- `https://huggingface.co/org/model`

Normalize to `org/model`.

### 2. Inspect the model config

Fetch and inspect `config.json`.

If the model does not appear to be MLX-quantized, stop and tell the user to use an MLX version or convert it first.

Key fields to capture:
- `model_type`
- `architectures`
- MoE indicators such as `num_experts` or `num_local_experts`
- VLM indicators such as `vision_config` or `image_token_id`

### 3. Check whether AFM already supports it

Search `Scripts/patches/LLMModelFactory.swift` for the `model_type`.

If it is already registered, tell the user it is already supported and suggest validating it with the AFM test workflow.

### 4. Decide the implementation tier

- Registry-only addition:
  - existing Swift architecture already covers the model
- Port from Python:
  - Python `mlx-lm` supports it and Swift does not yet
- New architecture:
  - no existing implementation is available

### 5. Implement through the patch system

Do not edit `vendor/` directly.

Use:
- `Scripts/patches/LLMModelFactory.swift`
- `Scripts/patches/VLMModelFactory.swift` when vision is involved
- `Scripts/apply-mlx-patches.sh`

If needed, add a new Swift model file under `Scripts/patches/`.

### 6. Verify

Build and run a simple prompt through AFM. If the model is new, follow up with the AFM test workflow for a proper validation pass.

## Notes

- Prefer repo-relative instructions and environment variables such as `$MODEL_CACHE`
- Check whether tool-call formatting or VLM registration also needs updates
