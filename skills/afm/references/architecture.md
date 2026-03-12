# AFM Architecture Reference

## Scope

Use this file when changing request flow, route wiring, or cross-mode behavior.

## Top-Level Shape

- CLI entrypoint:
  - `Sources/MacLocalAPI/main.swift`
- Server/router setup:
  - `Sources/MacLocalAPI/Server.swift`
- Foundation path:
  - `Sources/MacLocalAPI/Controllers/ChatCompletionsController.swift`
  - `Sources/MacLocalAPI/Models/FoundationModelService.swift`
- MLX path:
  - `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift`
  - `Sources/MacLocalAPI/Models/MLXModelService.swift`
- Shared request/response schema:
  - `Sources/MacLocalAPI/Models/OpenAIRequest.swift`
  - `Sources/MacLocalAPI/Models/OpenAIResponse.swift`
- Gateway services:
  - `Sources/MacLocalAPI/Services/BackendDiscoveryService.swift`
  - `Sources/MacLocalAPI/Services/BackendProxyService.swift`

## Runtime Modes

- `afm`:
  - Foundation model server mode (default), optional WebUI and gateway.
- `afm mlx`:
  - MLX-backed inference, either one-shot prompt or long-running server.
- `afm vision`:
  - OCR/table extraction via Apple Vision.

`main.swift` manually dispatches `mlx` and `vision` subcommands to avoid root/subcommand flag conflicts while still exposing subcommands in help text.

## HTTP Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `OPTIONS /v1/chat/completions`
- `POST /models/load` and `POST /models/unload` (WebUI compatibility stubs)

Global behavior in `Server.swift`:

- CORS headers on responses.
- max body size raised to 100mb.
- custom payload-too-large middleware that returns OpenAI-style error JSON.

## Request Flow Summary

### Foundation Request Flow

1. Decode `ChatCompletionRequest`.
2. Route to proxy if requested model is not `"foundation"` and gateway can serve it.
3. Build `FoundationModelService`.
4. Merge stop sequences from CLI and request.
5. Execute streaming/non-streaming generation.
6. Return OpenAI-style response or error.

### MLX Request Flow

1. Decode `ChatCompletionRequest`.
2. Validate `top_logprobs` against server max.
3. Merge sampling params (request overrides CLI defaults).
4. Execute `MLXModelService.generate` or `generateStreaming`.
5. Return text or `tool_calls` response.
6. Extract `reasoning_content` from `<think>` unless raw mode disables it.

## Cross-Cutting Invariants

- Keep OpenAI chat completion response shape stable.
- Preserve streaming chunk format (`chat.completion.chunk`) and terminal `[DONE]`.
- Keep `content` key present in assistant message encoding even when null.
- Preserve `system_fingerprint` emission in both non-streaming and streaming responses.
- Keep stop-sequence merge behavior: CLI + API, deduped, order-preserving.
