# Model Investigation Reference

How to investigate a HuggingFace model's architecture and capabilities.

## Fetching Model Configuration

### config.json (primary)

```
https://huggingface.co/<model-id>/resolve/main/config.json
```

Key fields to extract:

| Field | Purpose |
|-------|---------|
| `model_type` | Registry lookup key (e.g., `qwen3`, `llama`, `deepseek_v3`) |
| `architectures` | Full class names (e.g., `["Qwen3ForCausalLM"]`) |
| `quantization` or `quantization_config` | Present = MLX format. Absent = not MLX |
| `num_experts` / `num_local_experts` | MoE model if > 0 |
| `num_experts_per_tok` / `top_k` | MoE top-k routing |
| `vision_config` / `image_token_id` | VLM (vision-language model) |
| `text_config` | Nested text model config (common in VLMs) |
| `hidden_size` | Embedding dimension |
| `intermediate_size` | FFN intermediate dimension |
| `num_hidden_layers` | Number of transformer layers |
| `num_attention_heads` | Number of attention heads |
| `num_key_value_heads` | GQA head count (< num_attention_heads = GQA) |
| `rms_norm_eps` | RMSNorm epsilon |
| `rope_theta` | RoPE base frequency |
| `rope_scaling` | RoPE scaling config (type, factor, etc.) |
| `vocab_size` | Vocabulary size |
| `max_position_embeddings` | Maximum context length |
| `tie_word_embeddings` | Whether input/output embeddings are shared |

### tokenizer_config.json (for tool calling / chat template)

```
https://huggingface.co/<model-id>/resolve/main/tokenizer_config.json
```

Key fields:

| Field | Purpose |
|-------|---------|
| `chat_template` | Jinja template — check for `<think>`, `<tool_call>`, tool format |
| `added_tokens_decoder` | Special tokens (tool call tags, think tags, etc.) |
| `eos_token` | End-of-sequence token |

### Chat Template Indicators

Look in `chat_template` for:
- `<think>` / `enable_thinking` → model supports reasoning extraction
- `<tool_call>` → tool calling support
- `<|tool_call_begin|>` → alternative tool call format
- `<function=` → xmlFunction tool call format (Qwen3-Coder style)

## VLM vs LLM Detection

A model is a VLM if config.json contains ANY of:
- `vision_config` (nested object)
- `image_token_id` (integer)
- `visual` (nested object, some architectures)
- `model_type` ends with `_vl` or contains `vision`

VLMs need registration in BOTH `LLMModelFactory.swift` AND `VLMModelFactory.swift`.

Note: Some VLMs (like Qwen3.5-35B-A3B) can be loaded via LLMModelFactory for text-only requests as a performance optimization. The VLM factory is the fallback for image inputs.

## Common model_type Aliases

Some model types reuse existing architectures:

| model_type | Maps to | Notes |
|-----------|---------|-------|
| `mistral` | Llama | Same architecture |
| `acereason` | Qwen2 | Qwen2 variant |
| `kimi_k2` | DeepseekV3 | Same architecture |
| `joyai_llm_flash` | DeepseekV3 | Same architecture |

When investigating a new model_type, check if `architectures[0]` shares a base with known models (e.g., `*ForCausalLM` suffix with a known prefix).

## MoE Architecture Detection

If `num_experts` > 0 or `num_local_experts` > 0:
- This is a Mixture-of-Experts model
- Check `num_experts_per_tok` / `top_k` for routing
- Check for `shared_expert_intermediate_size` (shared expert, DeepseekV3 style)
- Some dense models set `num_experts: 0` — check the Swift implementation for dense fallback paths

## Weight File Inspection

If architecture is unclear, check weight file keys:
```
https://huggingface.co/<model-id>/resolve/main/model.safetensors.index.json
```

The `weight_map` shows tensor names → helps identify layer structure, attention patterns, and MoE routing.
