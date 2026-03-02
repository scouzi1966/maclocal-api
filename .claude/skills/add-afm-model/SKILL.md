---
name: add-afm-model
description: Add support for a new HuggingFace MLX model to AFM. Use when user wants to add, onboard, or check compatibility of a model — handles everything from "already supported" to implementing new architectures.
user_invocable: true
---

# Add AFM Model

Investigate and add support for a HuggingFace MLX model to the AFM server.

## Usage

- `/add-afm-model <model-id>` — e.g., `/add-afm-model mlx-community/Qwen3-8B-4bit`
- `/add-afm-model <url>` — e.g., `/add-afm-model https://huggingface.co/mlx-community/Qwen3-8B-4bit`

## Instructions

### Step 1: Parse Input

Extract the HuggingFace model ID from the user's input:
- Full URL: `https://huggingface.co/org/model` → `org/model`
- Model ID: `org/model` → use as-is
- If ambiguous, ask the user.

### Step 2: Fetch config.json

Fetch `https://huggingface.co/<model-id>/resolve/main/config.json` using WebFetch.

**Not MLX check:** If config.json has no `quantization` or `quantization_config` field, the model is not MLX-quantized. Inform the user:
> "This model is not in MLX format. Look for an MLX-quantized version on huggingface.co/mlx-community, or quantize it yourself with `mlx_lm.convert`."

Stop here if not MLX.

### Step 3: Extract model_type

Read the `model_type` field from config.json. This is the key that maps to a Swift model implementation.

Also note: `architectures`, `num_experts`/`num_local_experts` (MoE indicator), `image_token_id`/`vision_config` (VLM indicator).

### Step 4: Check LLMTypeRegistry

Search `Scripts/patches/LLMModelFactory.swift` for the `model_type` string in the `LLMTypeRegistry.shared` dictionary (lines ~25-80).

**If found → Already supported.** Tell the user:
> "This model is already supported! Run it with:
> ```
> MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache afm mlx -m <model-id> --port 9999
> ```"

Suggest running `/test-macafm` for validation if this is a new model variant.

Stop here if already registered.

### Step 5: Check Existing Architectures

The model_type is NOT in the registry. Now determine if an existing Swift implementation can handle it.

1. List files in `vendor/mlx-swift-lm/Libraries/MLXLLM/Models/`
2. Search for the model's base architecture name (e.g., if model_type is `foo_moe`, check for `Foo.swift` or similar)
3. Read the model's config.json fields and compare against existing implementations — some architectures handle variants (e.g., DeepseekV3 handles `kimi_k2`, Qwen2 handles `acereason`)
4. Check if an existing model has a dense fallback (e.g., `numExperts == 0` path)

**If a compatible architecture exists → Registry-only fix.** Proceed to **Tier: Registry Addition**.

### Step 6: Check Python mlx-lm

Search for the model_type in the Python mlx-lm library:
- Fetch `https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models` (or search GitHub)
- Look for a Python file matching the model_type

**If Python implementation exists → Port to Swift.** Proceed to **Tier: Port from Python**.

**If no Python implementation → Implement from scratch.** Proceed to **Tier: New Architecture**.

---

## Tier: Registry Addition

The simplest case — architecture exists, just needs a type alias.

1. Read `references/implementation-guide.md` for the patch system details
2. Add the model_type to `LLMTypeRegistry.shared` in `Scripts/patches/LLMModelFactory.swift`, mapping to the correct Configuration/Model pair
3. If the model is a VLM (has `vision_config`/`image_token_id`), also add to `Scripts/patches/VLMModelFactory.swift`
4. If the model has a new tool call format, update `Scripts/patches/ToolCallFormat.swift` `infer()` method
5. Apply patches: `./Scripts/apply-mlx-patches.sh`
6. Build: `swift build` (or `/build-afm` for full rebuild)
7. Verify: start server and test with a simple prompt

## Tier: Port from Python

Port the Python mlx-lm implementation to Swift.

1. Read `references/implementation-guide.md` for the full implementation pattern
2. Read `references/model-investigation.md` for config.json field mapping
3. Fetch and study the Python implementation from mlx-lm
4. Find the closest existing Swift model to use as a template (check `vendor/mlx-swift-lm/Libraries/MLXLLM/Models/`)
5. Create the new Swift file in `Scripts/patches/<ModelName>.swift`
6. Implement: Configuration (Codable) → Attention → MLP → TransformerBlock → Model
7. Add to `PATCH_FILES`, `TARGET_PATHS`, `NEW_FILES` in `Scripts/apply-mlx-patches.sh`
8. Register in `LLMTypeRegistry` (and VLM if needed)
9. Add weight sanitization if needed (in the Configuration's `sanitize()`)
10. Apply, build, verify

## Tier: New Architecture

No existing implementation anywhere. Research and implement from scratch.

1. Read `references/implementation-guide.md` and `references/model-investigation.md`
2. Find architecture documentation: paper, blog post, or reference implementation
3. Study config.json thoroughly for all architecture-specific fields
4. Find the closest existing Swift model as a starting template
5. Follow the same implementation steps as "Port from Python" above
6. Pay special attention to: attention patterns, normalization layers, MoE routing, positional embeddings
7. Test extensively — new architectures often have subtle bugs

---

## Key Reminders

- **NEVER edit files in `vendor/` directly** — all changes go through `Scripts/patches/`
- Always check VLM registry too if the model has vision capabilities
- Use `MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache` to avoid re-downloading
- After adding a new model, suggest running `/test-macafm` for full validation
