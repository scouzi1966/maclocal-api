import Vapor
import Foundation

/// Serves a hand-curated OpenAPI 3.1 spec describing the AFM HTTP surface
/// at `GET /openapi.json` and a Scalar-rendered HTML browser at `GET /docs`.
/// (T1.7)
///
/// The spec covers the agent-relevant endpoints: chat completions (incl.
/// cancellation, request-id headers, stream_options, parallel_tool_calls),
/// embeddings, audio transcribe/synthesize, vision OCR, batch completions,
/// files, tokenize / count_tokens, models, health, and metrics. It is
/// hand-maintained — keep it in sync when adding new endpoints.
///
/// `/docs` returns a minimal HTML shell that loads Scalar's API reference
/// renderer from a CDN. The page works offline only if the CDN is reachable;
/// agents that need a fully offline UI should consume `/openapi.json` directly.
struct OpenAPIController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        routes.get("openapi.json", use: openAPI)
        routes.get("docs", use: docs)
    }

    func openAPI(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.body = .init(string: Self.specJSON)
        return response
    }

    func docs(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "text/html; charset=utf-8")
        response.body = .init(string: Self.docsHTML)
        return response
    }

    /// Minimal Scalar-loader page. Points at `/openapi.json` on the same origin.
    static let docsHTML: String = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>afm — API reference</title>
        <style>body { margin: 0 }</style>
      </head>
      <body>
        <script id="api-reference" data-url="/openapi.json"></script>
        <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
      </body>
    </html>
    """

    /// Hand-curated OpenAPI 3.1 spec. Intentionally kept compact — describes
    /// the surface agents actually call rather than every property of every
    /// payload. JSON-encoded inline so we don't fight SPM resource bundling.
    static let specJSON: String = #"""
    {
      "openapi": "3.1.0",
      "info": {
        "title": "afm — local OpenAI-compatible API",
        "version": "0.9.13",
        "description": "OpenAI-compatible local LLM server for Apple Silicon. Backs both Apple Foundation Models and HuggingFace MLX models. See docs/clients/ for per-agent recipes.",
        "license": { "name": "MIT" }
      },
      "servers": [
        { "url": "http://localhost:9999", "description": "Default local instance" }
      ],
      "tags": [
        { "name": "chat", "description": "Chat completions" },
        { "name": "embeddings", "description": "Text embeddings" },
        { "name": "audio", "description": "Speech transcription and synthesis" },
        { "name": "vision", "description": "Apple Vision OCR" },
        { "name": "batch", "description": "Batch completions API" },
        { "name": "tokenize", "description": "Tokenizer access for context-budgeting" },
        { "name": "models", "description": "Model discovery" },
        { "name": "health", "description": "Liveness and observability" }
      ],
      "components": {
        "parameters": {
          "RequestIdHeader": {
            "name": "X-Request-ID",
            "in": "header",
            "required": false,
            "description": "Optional inbound correlation id; if absent the server mints `req_<uuid12>`. Echoed on every response and surfaced in `error.request_id`.",
            "schema": { "type": "string" }
          }
        },
        "headers": {
          "XRequestID": {
            "description": "Server-assigned or echoed request correlation id.",
            "schema": { "type": "string" }
          }
        },
        "schemas": {
          "ChatCompletionRequest": {
            "type": "object",
            "required": ["messages"],
            "properties": {
              "model": { "type": "string" },
              "messages": { "type": "array", "items": { "$ref": "#/components/schemas/Message" } },
              "temperature": { "type": "number" },
              "top_p": { "type": "number" },
              "top_k": { "type": "integer" },
              "min_p": { "type": "number" },
              "max_tokens": { "type": "integer" },
              "max_completion_tokens": { "type": "integer" },
              "presence_penalty": { "type": "number" },
              "repetition_penalty": { "type": "number" },
              "seed": { "type": "integer" },
              "logprobs": { "type": "boolean" },
              "top_logprobs": { "type": "integer", "minimum": 0, "maximum": 20 },
              "stop": { "type": "array", "items": { "type": "string" } },
              "stream": { "type": "boolean" },
              "stream_options": {
                "type": "object",
                "properties": {
                  "include_usage": { "type": "boolean", "description": "When false, the final SSE chunk does not carry a usage block. Default true." }
                }
              },
              "tools": { "type": "array", "items": { "$ref": "#/components/schemas/Tool" } },
              "tool_choice": { "description": "auto | none | required | { type: 'function', function: { name } }" },
              "parallel_tool_calls": { "type": "boolean", "description": "When false, the server emits at most one tool call per assistant turn." },
              "response_format": { "$ref": "#/components/schemas/ResponseFormat" }
            }
          },
          "Message": {
            "type": "object",
            "required": ["role"],
            "properties": {
              "role": { "type": "string", "enum": ["system", "developer", "user", "assistant", "tool"] },
              "content": { "description": "string or array of content parts (text | image_url | input_audio)" },
              "name": { "type": "string" },
              "tool_calls": { "type": "array" },
              "tool_call_id": { "type": "string" }
            }
          },
          "Tool": {
            "type": "object",
            "required": ["type", "function"],
            "properties": {
              "type": { "type": "string", "const": "function" },
              "function": {
                "type": "object",
                "required": ["name"],
                "properties": {
                  "name": { "type": "string" },
                  "description": { "type": "string" },
                  "parameters": { "type": "object" },
                  "strict": { "type": "boolean" }
                }
              }
            }
          },
          "ResponseFormat": {
            "type": "object",
            "required": ["type"],
            "properties": {
              "type": { "type": "string", "enum": ["text", "json_object", "json_schema"] },
              "json_schema": {
                "type": "object",
                "properties": {
                  "name": { "type": "string" },
                  "description": { "type": "string" },
                  "schema": { "type": "object" },
                  "strict": { "type": "boolean" }
                }
              }
            }
          },
          "ChatCompletionResponse": {
            "type": "object",
            "properties": {
              "id": { "type": "string" },
              "object": { "type": "string", "const": "chat.completion" },
              "model": { "type": "string" },
              "choices": { "type": "array" },
              "usage": { "type": "object" }
            }
          },
          "OpenAIError": {
            "type": "object",
            "properties": {
              "error": {
                "type": "object",
                "properties": {
                  "message": { "type": "string" },
                  "type": { "type": "string" },
                  "code": { "type": "string", "nullable": true },
                  "request_id": { "type": "string", "nullable": true }
                }
              }
            }
          },
          "TokenizeRequest": {
            "type": "object",
            "properties": {
              "model": { "type": "string" },
              "text": { "type": "string", "description": "OpenAI/Anthropic style." },
              "prompt": { "type": "string", "description": "vLLM style alias for text." }
            }
          },
          "TokenizeResponse": {
            "type": "object",
            "required": ["tokens", "count", "model"],
            "properties": {
              "tokens": { "type": "array", "items": { "type": "integer" } },
              "count": { "type": "integer" },
              "model": { "type": "string" },
              "max_model_len": { "type": "integer", "nullable": true }
            }
          },
          "CountTokensResponse": {
            "type": "object",
            "required": ["input_tokens", "model"],
            "properties": {
              "input_tokens": { "type": "integer" },
              "model": { "type": "string" }
            }
          },
          "CancellationResponse": {
            "type": "object",
            "required": ["id", "object", "cancelled"],
            "properties": {
              "id": { "type": "string" },
              "object": { "type": "string", "const": "chat.completion.cancellation" },
              "cancelled": { "type": "boolean" }
            }
          }
        }
      },
      "paths": {
        "/v1/chat/completions": {
          "post": {
            "tags": ["chat"],
            "summary": "Create a chat completion",
            "parameters": [{ "$ref": "#/components/parameters/RequestIdHeader" }],
            "requestBody": {
              "required": true,
              "content": {
                "application/json": { "schema": { "$ref": "#/components/schemas/ChatCompletionRequest" } }
              }
            },
            "responses": {
              "200": {
                "description": "Chat completion (or SSE stream when `stream:true`).",
                "headers": { "X-Request-ID": { "$ref": "#/components/headers/XRequestID" } },
                "content": {
                  "application/json": { "schema": { "$ref": "#/components/schemas/ChatCompletionResponse" } },
                  "text/event-stream": { "schema": { "type": "string", "description": "SSE: data: <ChatCompletionStreamResponse>\\n\\n ... data: [DONE]" } }
                }
              },
              "400": { "description": "Bad request", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OpenAIError" } } } },
              "503": { "description": "Server full / queue saturated; carries Retry-After: 2", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OpenAIError" } } } }
            }
          }
        },
        "/v1/chat/completions/{id}/cancel": {
          "post": {
            "tags": ["chat"],
            "summary": "Cancel an inflight chat completion (T1.5)",
            "parameters": [
              { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }
            ],
            "responses": {
              "200": { "description": "Cancellation triggered", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CancellationResponse" } } } },
              "404": { "description": "Unknown id (already completed or never registered)", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CancellationResponse" } } } }
            }
          }
        },
        "/v1/tokenize": {
          "post": {
            "tags": ["tokenize"],
            "summary": "Tokenize text (vLLM-compatible)",
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/TokenizeRequest" } } } },
            "responses": {
              "200": { "description": "Tokens + count", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/TokenizeResponse" } } } },
              "422": { "description": "No MLX model loaded (Foundation backend has no public tokenizer)", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OpenAIError" } } } }
            }
          }
        },
        "/v1/count_tokens": {
          "post": {
            "tags": ["tokenize"],
            "summary": "Count tokens (Anthropic-compatible)",
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/TokenizeRequest" } } } },
            "responses": {
              "200": { "description": "Token count", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/CountTokensResponse" } } } },
              "422": { "description": "No MLX model loaded", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/OpenAIError" } } } }
            }
          }
        },
        "/v1/embeddings": {
          "post": {
            "tags": ["embeddings"],
            "summary": "Create embeddings",
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "type": "object" } } } },
            "responses": { "200": { "description": "Embeddings vector(s)" } }
          }
        },
        "/v1/audio/transcriptions": {
          "post": {
            "tags": ["audio"],
            "summary": "Transcribe audio (Apple Speech)",
            "requestBody": { "required": true, "content": { "multipart/form-data": { "schema": { "type": "object" } } } },
            "responses": { "200": { "description": "Transcription result" } }
          }
        },
        "/v1/audio/speech": {
          "post": {
            "tags": ["audio"],
            "summary": "Synthesize speech (Apple AVSpeechSynthesizer)",
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "type": "object" } } } },
            "responses": { "200": { "description": "Audio bytes" } }
          }
        },
        "/v1/ocr": {
          "post": {
            "tags": ["vision"],
            "summary": "Apple Vision OCR over image input",
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "type": "object" } } } },
            "responses": { "200": { "description": "Extracted text" } }
          }
        },
        "/v1/batch/completions": {
          "post": {
            "tags": ["batch"],
            "summary": "SSE-multiplexed batch completions",
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "type": "object" } } } },
            "responses": { "200": { "description": "Batch SSE stream" } }
          }
        },
        "/v1/files": {
          "post": {
            "tags": ["batch"],
            "summary": "Upload a file (used by batch API)",
            "responses": { "200": { "description": "File object" } }
          }
        },
        "/v1/models": {
          "get": {
            "tags": ["models"],
            "summary": "List available models",
            "responses": { "200": { "description": "Model list" } }
          }
        },
        "/health": {
          "get": {
            "tags": ["health"],
            "summary": "Liveness probe",
            "responses": { "200": { "description": "OK" } }
          }
        },
        "/metrics": {
          "get": {
            "tags": ["health"],
            "summary": "Prometheus metrics (vLLM-namespaced)",
            "responses": { "200": { "description": "text/plain Prometheus exposition format" } }
          }
        }
      }
    }
    """#
}
