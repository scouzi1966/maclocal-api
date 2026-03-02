# MLX Inference and Tool Calling Reference

## Scope

Use this file when changing MLX generation behavior, streaming logic, tool calling, thinking extraction, media parsing, or prompt caching.

## Key Files

- `Controllers/MLXChatCompletionsController.swift`
- `Models/MLXModelService.swift`
- `Models/OpenAIRequest.swift`
- `Models/OpenAIResponse.swift`

## Non-Streaming Path

`MLXChatCompletionsController.chatCompletions`:

1. Decode request and validate.
2. Enforce `top_logprobs <= maxLogprobs`.
3. Resolve effective params from request + CLI defaults.
4. Call `MLXModelService.generate(...)`.
5. If tool calls exist, return `tool_calls` finish reason.
6. Otherwise return text content with optional `reasoning_content`.

The controller extracts `<think>...</think>` tags unless `--raw` disables extraction (WebUI still forces extraction path to keep UI behavior consistent).

## Streaming Path

`createStreamingResponse` drives token-by-token SSE output and merges:

- content deltas
- reasoning deltas
- logprobs
- tool call deltas

It preserves ordering guarantees:

- flushes pending textual content before emitting tool-call deltas.
- emits initial assistant delta.
- emits final usage/timing and `[DONE]`.

## Tool Calling Internals

Tool calling supports both:

- vendor parser tool-call chunks
- AFM token-level tag detection fallback

Critical behaviors:

- Detect start/end tags from model tool format (`toolCallStartTag`, `toolCallEndTag`).
- Stream regular text outside tool tags.
- Buffer only tool-call body while `inToolCall`.
- Parse and emit incremental function/argument fragments in OpenAI delta shape.
- Emit final `finish_reason=tool_calls` when tool calls were generated.

Argument repair:

- `--fix-tool-args` remaps model-emitted keys back to schema keys.
- mapping includes case and naming normalization behavior.

## Tool-Call Parser Overrides

`MLXModelService` can override chat template and parser when tools are present:

- `hermes`
- `llama3_json`
- `gemma`
- `mistral`
- `qwen3_xml`

The service embeds adapted vLLM templates and keeps `<tool_call>...</tool_call>` wrapping for reliable streaming detection.

## Structured Outputs and Response Format

`response_format` can influence prompt construction and output constraints.

For `json_schema`, the service can inject schema guidance to improve conformance.

## Multimodal Inputs

Message content parser supports `image_url` parts that can be:

- data URLs
- remote URLs (downloaded)
- file paths

Media routing:

- image MIME/extension -> image path
- video MIME/extension -> video path

Temporary files are created for transformed or downloaded media and cleaned up after generation.

## Prompt Prefix Caching

MLX service supports prefix caching:

- token-prefix matching between previous and current prompts
- reuses KV cache for shared prefix
- skips cache reuse for multimodal inputs

This is a major latency optimization for repeated prompts with shared context.
