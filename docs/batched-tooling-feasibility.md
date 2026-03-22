# Batched Tooling Feasibility

Date: 2026-03-22

## Summary

AFM already has:
- batched token generation for concurrent streaming requests
- tool-aware chat template overrides
- parser selection and fallback extraction
- argument remapping and coercion
- grammar-constrained tool-call generation for `afm_adaptive_xml`

So batch-native tooling is feasible.

However, AFM does **not** currently have a true batch-native tooling subsystem.

Today:
- `BatchScheduler` batches model decode
- tool-call parsing and OpenAI `tool_calls` assembly are still largely handled per request above the batch layer

This means AFM can serve multiple tool-calling requests concurrently, but the tooling stack itself is not yet integrated into the batch engine.

## Current architecture

### What is already tool-call aware

In
[Sources/MacLocalAPI/Models/MLXModelService.swift](/Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api/Sources/MacLocalAPI/Models/MLXModelService.swift),
AFM already supports:

- tool-aware chat template overrides
- `--tool-call-parser`
- `afm_adaptive_xml`
- fallback extraction for multiple tool-call formats
- argument remapping
- argument type coercion
- `--fix-tool-args`
- grammar-constrained XML tool calling
- streaming OpenAI-style tool-call emission
- non-streaming OpenAI-style `tool_calls`

In
[Scripts/patches/Evaluate.swift](/Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api/Scripts/patches/Evaluate.swift),
the serial generation path also has a `ToolCallProcessor` in the vendor stream pipeline.

### What the batch path does today

In
[Sources/MacLocalAPI/Models/BatchScheduler.swift](/Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api/Sources/MacLocalAPI/Models/BatchScheduler.swift),
the scheduler:

- batches decode steps across active slots
- maintains per-slot sampler, processor, and detokenizer state
- yields `StreamChunk` text back to the caller

But the scheduler does **not**:

- run a `ToolCallProcessor`
- maintain a per-slot tool-call parser state machine
- emit parsed tool calls from inside the batch loop

Instead, the controller side still watches streamed text for tool-call tags and reconstructs tool calls per request.

In
[Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift](/Volumes/edata/codex/dev/git/maclocal-api/NEXT/maclocal-api/Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift),
the streaming response path performs:

- tool-call tag detection
- per-request tool-call buffering
- fallback parsing
- incremental SSE tool-call emission

This works, but it is layered **on top** of batched text generation rather than integrated into it.

## What “batched tooling” should mean

The useful target is **not** “batch external tool execution.”

Like vLLM and SGLang, AFM should still:
- generate tool calls
- return them to the caller
- let the caller execute tools
- resume generation after tool results arrive

So for AFM, “batched tooling” should mean:

- batch-native tool-call generation
- per-slot tool-call parsing state
- per-slot structured/grammar constraint state
- serial and concurrent parity in tool-call behavior

## Feasibility assessment

### Feasible parts

This is feasible because the core pieces already exist:

- tool-aware prompt/template logic
- parser and fallback logic
- grammar setup for XML tool calls
- a working batched decode scheduler with per-slot state

Adding per-slot tooling state to the scheduler is conceptually straightforward.

### Hard parts

#### 1. Serial vs concurrent divergence

AFM currently has two different layers of tool-call handling:

- serial path: more direct generation/piece handling
- concurrent path: batched decode plus controller-side parsing

If batch-native tooling is added without first unifying the semantics, the two paths will drift further.

#### 2. Streaming complexity

Incremental tool-call emission is stateful. Each batch slot would need:

- tool-call boundary detection
- current tool-call buffer
- incremental emitted tool metadata
- fallback parse state

This is manageable, but it belongs in per-slot scheduler state, not in the controller.

#### 3. Grammar matcher lifecycle

When grammar constraints are active, each request needs independent matcher state.
If structured output and tool-call constraints become more batch-native, the scheduler must carry per-slot grammar state too.

#### 4. Testing burden

Any refactor here must validate:

- serial vs concurrent equivalence
- stream vs non-stream
- default parser vs `afm_adaptive_xml`
- grammar off vs on
- `tool_choice` modes

## Recommended implementation order

### Phase 1: Shared tooling runtime abstraction

Create a shared abstraction, for example `ToolCallRuntime`, that owns:

- parser mode
- tool-call boundary detection
- fallback extraction
- argument remapping
- type coercion
- optional fixups

Use this in the serial path first, with no intended behavior change.

Goal:
- remove duplicated or controller-heavy tool-call logic
- establish one source of truth for tool-call handling

### Phase 2: Per-slot tooling state in BatchScheduler

Add one `ToolCallRuntime`-like instance per slot.

Each slot should carry:

- parser format
- current tool-call buffer
- incremental emission state
- optional grammar state
- final collected tool calls

At that point, tool-call parsing becomes batch-compatible because it is still isolated by slot, but it lives inside the batch engine.

### Phase 3: Thin controller

Move streaming tool-call extraction out of the controller.

Controller responsibilities should shrink to:

- HTTP/SSE framing
- OpenAI response serialization
- finish-reason wiring

Scheduler responsibilities should include:

- per-slot decode
- per-slot tool-call extraction
- per-slot structured/tool grammar progression

### Phase 4: Structured-output/tooling unification

Once per-slot runtime state exists, unify:

- guided/structured constraints
- XML tool-call grammar constraints
- slot-local matcher lifecycle

This will be especially valuable if grammar constraints are expanded to structured output paths.

## Recommendation

AFM should not attempt to implement “batched tooling” by adding more controller-side logic around batched decode.

The better plan is:
1. unify serial and concurrent tooling semantics
2. introduce a shared per-request tooling runtime
3. move per-request tool parsing into per-slot batch state
4. keep tool execution outside the engine

## Verdict

- Feasible: yes
- Small patch: no
- Best first step: unify serial and concurrent tool-call handling behind a shared runtime abstraction
- Long-term goal: batch-native per-slot tooling state inside `BatchScheduler`
