# AFM API Contract Reference

## Scope

Use this file when changing request/response fields, response format semantics, or compatibility with OpenAI SDKs.

## Primary API Endpoints

- `POST /v1/chat/completions`
- `GET /v1/models`

Controllers:

- Foundation path: `Controllers/ChatCompletionsController.swift`
- MLX path: `Controllers/MLXChatCompletionsController.swift`

## Request Schema Anchors

`Models/OpenAIRequest.swift` defines `ChatCompletionRequest` and related types.

Supported request fields include:

- `model`
- `messages`
- `temperature`
- `max_tokens`, `max_completion_tokens` (`effectiveMaxTokens`)
- `top_p`, `top_k`, `min_p`
- `repetition_penalty`, `repeat_penalty` (`effectiveRepetitionPenalty`)
- `presence_penalty`, `frequency_penalty`
- `seed`
- `logprobs`, `top_logprobs`
- `stop`
- `stream`
- `tools`, `tool_choice`
- `response_format`

Message content supports:

- string text
- multipart content with `text` and `image_url`
- tool-call message fields (`tool_calls`, `tool_call_id`)

## Response Schema Anchors

`Models/OpenAIResponse.swift` defines:

- `ChatCompletionResponse`
- `ChatCompletionStreamResponse`
- `ResponseMessage`
- `ResponseToolCall` and tool-call deltas
- usage and logprobs payloads

Important compatibility details:

- `ResponseMessage.encode` always encodes `content` (nullable), because some SDKs require the key even when null.
- `system_fingerprint` is emitted for both foundation and mlx-backed responses.
- Tool-call responses must set `finish_reason = "tool_calls"`.

## Streaming Contract

SSE headers used by both controllers include:

- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `Connection: keep-alive`
- `X-Accel-Buffering: no`

Streaming body format:

- emits `data: {json}\n\n` chunks
- terminates with `data: [DONE]\n\n`

## Structured Output Contract

`response_format` supports:

- `type = "text"`
- `type = "json_object"`
- `type = "json_schema"`

Foundation path:

- strict json schema mode is checked via `response_format.type == "json_schema"` and `json_schema.strict == true`
- schema converts to Apple generation schema via `JSONSchemaConverter`

MLX path:

- structured output guidance can be injected via prompt/tooling path in `MLXModelService`

## Error Contract

Controllers return OpenAI-style error JSON (`OpenAIError`) with meaningful `type` and HTTP status.

Examples:

- invalid request fields -> `400`
- unknown model -> `404` (`model_not_found` path)
- payload too large -> `413` with clear message
- backend/proxy failures -> mapped backend-style error message
