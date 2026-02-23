# Claude Code Reference for MacLocal API

## Project Overview

AFM (Apple Foundation Models) is an OpenAI-compatible API server for local LLM inference on Apple Silicon. It supports two backends:
- **MLX** (`afm mlx`): Uses mlx-swift-lm for MLX-format models from Hugging Face
- **Apple Foundation Models** (`afm` without subcommand): Uses Apple's on-device models (macOS 26+)

The server exposes `/v1/chat/completions` and `/v1/models` endpoints compatible with OpenAI API clients.

## Project Structure

```
Sources/MacLocalAPI/
├── main.swift                          # CLI entry point (ArgumentParser)
├── Controllers/
│   ├── MLXChatCompletionsController.swift  # Streaming/non-streaming SSE handler
│   └── ...
├── Models/
│   ├── MLXModelService.swift           # Model loading, generation, prompt caching
│   ├── OpenAIRequest.swift             # Request types (ChatCompletionRequest, etc.)
│   ├── OpenAIResponse.swift            # Response types
│   └── ...
vendor/
├── mlx-swift-lm/                       # Git submodule — DO NOT modify directly
├── llama.cpp/                          # Git submodule
Scripts/
├── patches/                            # Our patches to vendor code (copied over originals)
├── apply-mlx-patches.sh                # Applies patches from Scripts/patches/ to vendor/
├── build-from-scratch.sh               # Full build: submodules + patches + webui + build
```

## Vendor Patch System

**NEVER modify files in `vendor/` directly.** All changes go through `Scripts/patches/`.

The patch script (`Scripts/apply-mlx-patches.sh`) copies complete Swift files from `Scripts/patches/` to vendor targets. Three arrays define the mapping:
- `PATCH_FILES=()` — filenames in `Scripts/patches/`
- `TARGET_PATHS=()` — relative paths under `vendor/mlx-swift-lm/`
- `NEW_FILES=()` — files that don't exist upstream

Commands: `--check` (verify), `--revert` (restore originals), no flag (apply).

## Build

```bash
swift build                              # Debug build
swift build -c release                   # Release build
./Scripts/build-from-scratch.sh          # Full build (submodules + patches + webui + clean + build)
```

## Running the Server

**IMPORTANT:** Always set the model cache directory to avoid re-downloading models:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache afm mlx -m <model-id> --port 9999
```

Debug logging:
```bash
AFM_DEBUG=1 MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache afm mlx -m <model-id> --port 9999
```

Debug logging shows `[KVCache]` hit/miss stats, tool call detection, and timing info.

## Key Features

### Sampling Parameters
All functional end-to-end: `temperature`, `top_p`, `repetition_penalty`, `top_k`, `min_p`, `presence_penalty`, `seed`.

Added via vendor patch in `Scripts/patches/Evaluate.swift`: `TopKProcessor`, `MinPProcessor`, `PresenceContext`, `CompositeLogitProcessor`.

Sampler chain order (following llama.cpp): penalties → top_k → min_p → temperature+sampling.

`frequency_penalty`: parsed but silently ignored (not implemented).

### Tool Calling
OpenAI-compatible `tools`, `tool_choice`, `tool_calls` implemented.

**Streaming tool call detection** uses token-level start/end tag matching (mlx-lm Python style):
- Tags derived from model's `ToolCallFormat` (e.g., `<tool_call>`/`</tool_call>`)
- Content outside tool calls streams normally; only tool call body is buffered
- Fallback regex parser (`extractToolCallsFallback`) handles edge cases
- `finish_reason: "tool_calls"` when tool calls are present

**Qwen3-Coder XML format**: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`. Vendor's `ToolCallProcessor` fails on this (regex without `dotMatchesLineSeparators`). Fixed with:
1. `inferToolCallFormat()` reads `model_type` from config.json
2. `extractToolCallsFallback()` post-generation regex parsing
3. Duplicate parameter workaround (keep first non-empty value)

### Logprobs
OpenAI-compatible `logprobs` and `top_logprobs` (0-20) implemented.

Uses `log(softmax(logits/temp))` after processor chain. `logSoftmax` not available in MLX Swift — use `log(softmax(x))`. Use `Swift.min()` to avoid MLX namespace collision.

### Prompt Caching
Server-level single-slot token-level prefix matching (llama.cpp style). `PromptCacheBox` stores KV cache + prompt token array. Reuses matching prefix, only processes suffix tokens. Multimodal inputs skip caching.

### Think/Reasoning Extraction
`<think>...</think>` tags extracted into `reasoning_content` field for streaming and non-streaming responses. Buffer holds 7-8 chars for tag boundary detection.

**Note:** Not all models support thinking. Check the model's `chat_template.jinja` for `<think>` / `enable_thinking` logic. For example, `Qwen3-Coder-Next` does NOT have thinking support (confirmed by Hugging Face model card and chat template).

### Stop Sequences
`stop` field implemented end-to-end. Buffer-based approach handles stop strings spanning chunk boundaries.

### Response Format
`response_format` supports `text`, `json_object`, `json_schema`. Current implementation uses prompt injection (not guaranteed valid JSON).

## Integrations

- **OpenCode** (https://github.com/anomalyco/opencode): Uses OpenAI-compatible API as local provider
- **OpenClaw** (https://github.com/openclaw/openclaw): Uses `openai-completions` API mode. `afm mlx -m <model> --openclaw-config` generates provider config.
- `max_completion_tokens` accepted alongside `max_tokens`
- `developer` role mapped to `system`

## Model-Specific Notes

### Qwen3-Coder-Next
- Tool call format: `xmlFunction` (auto-detected via `model_type` in config.json)
- Chat template wraps tool calls in `<tool_call>`/`</tool_call>` tags
- `<tool_call>` (id 151657) and `</tool_call>` (id 151658) are added tokens in the vocabulary
- **No thinking/reasoning support** — chat template has no `<think>` logic
- Known issue: sometimes emits duplicate `<parameter=key>` tags or JSON objects instead of strings for tool parameters

### Supported Tool Call Formats
Defined in `vendor/.../ToolCallFormat.swift`: `json`, `lfm2`, `xmlFunction`, `glm4`, `gemma`, `kimiK2`, `minimaxM2`. Auto-detected from `model_type` in config.json via `ToolCallFormat.infer()`.

## Architecture Notes

### MLXModelService
- `ModelContainer` uses `SerialAccessContainer` (async mutex) for thread-safe model access
- `container.perform {}` holds lock for entire generation — ensures single-sequence access
- `generateStreaming()` returns `(modelID, stream, promptTokens, toolCallStartTag, toolCallEndTag)`
- The inner `generateTask` (vendor code) runs token generation in a `Task` with synchronous `iterator.next()` loop

### MLXChatCompletionsController
- Handles both streaming (SSE) and non-streaming responses
- Streaming uses Vapor's `Response.Body.init(asyncStream:)` with `NIOAsyncWriter`
- Tool call detection state machine: `inToolCall` / `madeToolCall` / `currentToolText`
- Think extraction runs in the same streaming loop via `extractThinkTags()`
- `X-Accel-Buffering: no` header set for nginx proxy compatibility

### Vendor Code (mlx-swift-lm)
- `ToolCallProcessor`: processes chunks through format-specific parsers (inline mode for xmlFunction)
- `NaiveStreamingDetokenizer`: decodes accumulated tokens, returns suffix diff per token
- `TokenIterator.next()`: synchronous GPU computation per token
- `GenerateParameters`: sampling config passed to the model
