# Batched Tooling Phase 1

## Goal

Phase 1 makes AFM's streaming tool-call path shared between serial and concurrent generation.

The target for this phase was:

- one shared tool-call streaming runtime
- per-slot tool-call state in the batch scheduler
- batch-side emission of tool-call deltas and completed tool calls
- controller behavior that stays OpenAI-compatible
- test coverage for runtime behavior, scheduler extraction, and concurrent request isolation

This phase does **not** try to batch external tool execution, unify every non-streaming code path, or make grammar/structured-output state batch-native.

## What Changed

### Shared runtime

A shared streaming parser/runtime now lives in:

- `Sources/MacLocalAPI/Models/ToolCallStreamingRuntime.swift`

It owns:

- tool-call start/end tag detection
- incremental tool-call delta extraction
- placeholder-to-final replacement flow
- adaptive XML JSON fallback handling
- final salvage for incomplete tool calls at stream end
- key remapping and `fix-tool-args` integration through injected closures

### Controller refactor

The streaming controller now uses the shared runtime instead of its previous inline state machine:

- `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift`

The controller now focuses on:

- request handling
- SSE serialization
- OpenAI wire-format output

A small protocol seam was added so controller streaming behavior can be tested without the full model service:

- `Sources/MacLocalAPI/Controllers/MLXChatServing.swift`

### Batch scheduler support

The batch scheduler now supports per-slot tool-call runtime state:

- `Sources/MacLocalAPI/Models/BatchScheduler.swift`

Phase 1 additions there:

- per-slot `ToolCallStreamingRuntime`
- per-request tool runtime configuration passed at submit time
- text chunk routing through the shared runtime
- batch-side emission of:
  - `StreamDeltaToolCall`
  - completed `ResponseToolCall`
- helper extraction methods:
  - `deltaToolCallsToEmit(from:)`
  - `completedToolCallsToEmit(from:)`
  - `streamChunksToEmit(from:)`

### Stream chunk model

`StreamChunk` now supports incremental tool-call deltas:

- `Sources/MacLocalAPI/Models/MLXModelService.swift`

Added field:

- `toolCallDeltas: [StreamDeltaToolCall]?`

### Metallib bootstrap for tests

`swift test` was failing because MLX metallib setup only happened in `main.swift`.

That logic is now shared through:

- `Sources/MacLocalAPI/Utils/MLXMetalLibrary.swift`

It is used by:

- `Sources/MacLocalAPI/main.swift`
- MLX-using test suites

This preserves distributed binary behavior while allowing `swift test` to initialize MLX correctly.

## Test Coverage

### Runtime tests

- `Tests/MacLocalAPITests/ToolCallStreamingRuntimeTests.swift`

Coverage includes:

- incremental XML tool call parsing
- adaptive XML fallback
- salvage of incomplete tool calls
- text preservation around tool-call spans

### Scheduler/helper tests

- `Tests/MacLocalAPITests/ConcurrentBatchTests.swift`

Coverage includes:

- completed tool-call extraction
- incremental delta extraction
- delta-before-final ordering
- per-event-list isolation expectations

### Controller streaming tests

- `Tests/MacLocalAPITests/MLXChatCompletionsControllerStreamingTests.swift`

Coverage includes:

- raw streamed tool-call text -> SSE tool-call deltas
- completed batch tool calls -> SSE output
- batch-side tool-call deltas -> SSE output
- concurrent request isolation at the route level

### Existing parser tests

Existing tool parser coverage remains active in:

- `Tests/MacLocalAPITests/XMLToolCallParsingTests.swift`

## Validation

Phase 1 validation passed with:

```bash
swift test
```

This includes the previously failing MLX-based test suites after the metallib bootstrap was shared with the test runner.

## What Phase 1 Does Not Do

Still out of scope:

- external tool execution batching
- batch-native grammar matcher state
- batch-native structured-output state unification
- full non-streaming tooling unification
- scheduler-level fake-model end-to-end tests using real batched decode

## Recommended Phase 2

The next useful step is to extend the same unification approach into:

- non-streaming tool-call handling parity
- batch-aware grammar / structured-output runtime state
- broader serial vs concurrent equivalence testing across parser modes

Phase 1 is intentionally the smallest useful engine-level step:

- shared runtime
- batch slot integration
- SSE correctness
- concurrency isolation
