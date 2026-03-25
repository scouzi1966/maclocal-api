# Batch Dispatch API Design

## Overview

Add two batch completion endpoints to AFM: an OpenAI-compatible Batch API (`/v1/batches` + `/v1/files`) for transparent Python client compatibility, and a custom SSE multiplex endpoint (`/v1/batch/completions`) for low-latency real-time use cases. Both use the existing BatchScheduler for concurrent GPU-batched inference with optional prefix caching.

Key difference from OpenAI's Batch API: requests are dispatched immediately (not queued for later), processed concurrently via dense batched decoding, and results are available in seconds rather than hours.

## Requirements

1. **OpenAI client transparency**: `openai.batches.create()` must work unchanged against AFM
2. **Immediate dispatch**: All requests start processing as soon as the batch is submitted
3. **Concurrent execution**: Requests within a batch run simultaneously via BatchScheduler's dense batched decode (`model([B, 1])`)
4. **Prefix caching**: RadixTreeCache is auto-enabled for batch workloads to share KV cache prefixes across requests
5. **Auto-promotion**: Server auto-promotes from serial to batch mode when a batch request arrives (no `--concurrent` flag required)
6. **Interleaving**: Regular `/v1/chat/completions` requests can run alongside batch requests in the same scheduler
7. **Per-request streaming**: Each request in a batch can independently choose `stream: true/false`
8. **Non-streaming through BatchScheduler**: Non-streaming requests must use BatchScheduler when `--concurrent` is active (currently they bypass it via `container.perform {}`)

## Endpoint A: OpenAI-Compatible Batch API

### File Endpoints

#### `POST /v1/files`

Upload a JSONL file for batch processing.

**Request:** `multipart/form-data` with fields:
- `file`: JSONL file content
- `purpose`: must be `"batch"` (other purposes rejected)

**Response:**
```json
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 12345,
  "created_at": 1711471533,
  "filename": "requests.jsonl",
  "purpose": "batch"
}
```

#### `GET /v1/files/{file_id}`

Returns file metadata.

#### `GET /v1/files/{file_id}/content`

Returns raw file content (JSONL). Used to download batch results.

#### `DELETE /v1/files/{file_id}`

Removes file from in-memory store.

### Batch Endpoints

#### `POST /v1/batches`

Create and immediately dispatch a batch.

**Request:**
```json
{
  "input_file_id": "file-abc123",
  "endpoint": "/v1/chat/completions",
  "completion_window": "24h"
}
```

- `endpoint`: must be `/v1/chat/completions` (only supported endpoint)
- `completion_window`: accepted but ignored (dispatch is immediate)

**Input JSONL format** (one request per line):
```jsonl
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "qwen3-coder", "messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "World"}]}}
```

**Response:**
```json
{
  "id": "batch_abc123",
  "object": "batch",
  "endpoint": "/v1/chat/completions",
  "input_file_id": "file-abc123",
  "completion_window": "24h",
  "status": "in_progress",
  "created_at": 1711471533,
  "request_counts": {
    "total": 2,
    "completed": 0,
    "failed": 0
  }
}
```

**Dispatch flow:**
1. Parse input JSONL, validate each request
2. Call `ensureBatchMode()` to auto-promote if needed
3. Reserve slots for all requests (or return 503 if insufficient capacity)
4. Submit each request to `BatchScheduler.submit()`
5. Collect results asynchronously into output JSONL
6. Return batch object with `status: "in_progress"`

#### `GET /v1/batches/{batch_id}`

Poll batch status.

**Response** (while processing):
```json
{
  "id": "batch_abc123",
  "object": "batch",
  "status": "in_progress",
  "request_counts": {"total": 2, "completed": 1, "failed": 0}
}
```

**Response** (completed):
```json
{
  "id": "batch_abc123",
  "object": "batch",
  "status": "completed",
  "output_file_id": "file-xyz789",
  "request_counts": {"total": 2, "completed": 2, "failed": 0},
  "completed_at": 1711471540
}
```

**Output JSONL format:**
```jsonl
{"id": "batch_req_001", "custom_id": "req-1", "response": {"status_code": 200, "request_id": "req-abc", "body": {"id": "chatcmpl-abc", "object": "chat.completion", "created": 1711471535, "model": "qwen3-coder", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}}, "error": null}
{"id": "batch_req_002", "custom_id": "req-2", "response": {"status_code": 200, "request_id": "req-def", "body": {"id": "chatcmpl-def", "object": "chat.completion", "created": 1711471536, "model": "qwen3-coder", "choices": [{"index": 0, "message": {"role": "assistant", "content": "World!"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}}}, "error": null}
```

#### `GET /v1/batches`

List all batches. Returns array of batch objects.

#### `POST /v1/batches/{batch_id}/cancel`

Cancel a running batch. Signals all in-progress slots to stop.

### Batch Statuses

- `validating` → brief, during JSONL parsing
- `in_progress` → requests dispatched and generating
- `completed` → all requests finished (success or per-request error)
- `failed` → batch-level error (e.g., model not loaded)
- `cancelling` → cancel requested, waiting for active slots to drain
- `cancelled` → all slots stopped

## Endpoint B: Custom SSE Multiplex

### `POST /v1/batch/completions`

Low-latency endpoint for real-time batch processing with multiplexed streaming.

**Request:**
```json
{
  "requests": [
    {
      "custom_id": "req-1",
      "body": {
        "model": "qwen3-coder",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "temperature": 0.7
      }
    },
    {
      "custom_id": "req-2",
      "body": {
        "messages": [{"role": "user", "content": "World"}],
        "stream": false
      }
    }
  ]
}
```

**Response:** `text/event-stream` (SSE) with multiplexed events:

```
data: {"custom_id":"req-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"custom_id":"req-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"custom_id":"req-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"custom_id":"req-2","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"World!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11}}

data: {"custom_id":"req-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

data: [DONE]
```

**Per-request streaming behavior:**
- `stream: true`: emit `chat.completion.chunk` events with deltas as tokens are generated
- `stream: false`: buffer all tokens internally, emit a single `chat.completion` event with the full response when generation completes

**Error handling:** Per-request errors emit an error event and don't kill other requests:
```
data: {"custom_id":"req-3","object":"batch.error","error":{"message":"Context length exceeded","type":"invalid_request_error"}}
```

**Validation:**
- Maximum batch size: 64 requests
- Duplicate `custom_id` values rejected with 400
- Each `body` validated as a standard `ChatCompletionRequest`

## Auto-Promotion

When the server runs in serial mode (`maxConcurrent = 0`) and a batch request arrives:

1. **Promote:** `MLXModelService.ensureBatchMode(concurrency:)` creates a BatchScheduler with `maxConcurrent = max(requestCount, 8)` and enables RadixTreeCache. Reuses existing `initScheduler()` internals.
2. **Reference counting:** `activeBatchCount` (atomic `Int`) tracks in-flight batch operations. Incremented on batch start, decremented on batch completion.
3. **Teardown:** When `activeBatchCount` hits 0 AND the server was originally serial, schedule teardown after a 5-second grace period. If a new batch arrives within the grace period, cancel teardown.
4. **Already in batch mode:** If `--concurrent` was set at startup, skip promotion/teardown. Requests join the existing scheduler.

## Non-Streaming Through BatchScheduler

**Current behavior (bug):** Non-streaming requests in `--concurrent` mode use `container.perform {}` (serial mutex), bypassing BatchScheduler entirely. They hold the model lock exclusively, blocking all other requests.

**Fix:** When `maxConcurrent >= 2` (batch mode active), non-streaming `generate()` calls `generateStreaming()` internally, submits to BatchScheduler, and collects the stream into a complete response. This:
- Enables non-streaming requests to participate in batched decode
- Allows them to interleave with streaming requests
- Shares GPU time fairly across all in-flight requests

The controller-level change is minimal: call `generateStreaming()` + collect instead of `generate()` when a scheduler is active.

## In-Memory Storage

### FileStore

Thread-safe dictionary `[String: StoredFile]` for JSONL files:

```
StoredFile:
  id: String          // "file-<uuid>"
  bytes: Int
  filename: String
  purpose: String     // "batch"
  data: Data          // raw JSONL content
  createdAt: Date
```

- Auto-eviction: files older than 1 hour are purged (checked lazily on access)
- No disk persistence: this is a local inference server, not cloud storage

### BatchStore

Thread-safe dictionary `[String: BatchState]` for batch lifecycle:

```
BatchState:
  id: String          // "batch-<uuid>"
  inputFileId: String
  status: BatchStatus
  requestCounts: RequestCounts  // total, completed, failed
  results: [BatchResultEntry]   // collected as requests complete
  outputFileId: String?         // set when all requests done
  createdAt: Date
  completedAt: Date?
  error: BatchError?            // batch-level error
```

Both stores live in a single `BatchStore` actor for thread safety.

## Files Changed

| File | Change |
|------|--------|
| `Controllers/BatchCompletionsController.swift` | **New** — SSE multiplex endpoint (`/v1/batch/completions`) |
| `Controllers/BatchAPIController.swift` | **New** — OpenAI-compatible `/v1/batches` + `/v1/files` endpoints |
| `Models/BatchStore.swift` | **New** — in-memory file store + batch state tracking (actor) |
| `Models/OpenAIRequest.swift` | Add `BatchCompletionRequest`, `BatchRequestItem`, `BatchCreateRequest`, `BatchInputLine` |
| `Models/OpenAIResponse.swift` | Add `BatchSSEEvent`, `BatchError`, `BatchObject`, `FileObject`, `BatchResultLine` |
| `Models/MLXModelService.swift` | Add `ensureBatchMode(concurrency:)`, `activeBatchCount`, grace-period teardown |
| `Controllers/MLXChatServing.swift` | Add `ensureBatchMode(concurrency:)` to protocol |
| `Controllers/MLXChatCompletionsController.swift` | Non-streaming requests route through `generateStreaming()` + collect when scheduler active |
| `Server.swift` | Register batch + file routes |

## Testing Strategy

1. **Unit tests:** BatchStore actor (file CRUD, batch state transitions, auto-eviction, concurrent access)
2. **Integration tests:** Submit batch JSONL, poll until complete, verify output JSONL matches expected responses
3. **SSE multiplex tests:** Submit batch via `/v1/batch/completions`, verify interleaved chunks have correct `custom_id` tags
4. **Auto-promotion tests:** Verify serial→batch promotion on first batch, teardown after grace period
5. **Interleaving tests:** Submit batch + regular completion simultaneously, verify both complete
6. **Python client test:** Use `openai` Python package against the server, verify standard `client.batches` workflow works end-to-end
7. **Non-streaming fix test:** Submit non-streaming request with `--concurrent`, verify it goes through BatchScheduler (check batched decode logs)

## Out of Scope

- Disk-persisted file/batch storage
- `completion_window` enforcement (always immediate)
- Rate limiting per batch
- Batch priority scheduling
- `/v1/embeddings` as batch endpoint (only `/v1/chat/completions` supported)
