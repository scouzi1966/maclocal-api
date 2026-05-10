# Roadmap: Make AFM the best local server for agentic clients

## Context

`afm` (this repo) is an OpenAI-compatible local LLM server on Apple Silicon backed by both Apple Foundation Models and HuggingFace MLX models. The user wants AFM to be the obvious pick for agentic clients — OpenCode and OpenClaw (already targeted), Cline / Continue.dev / Aider / Cursor (broader OpenAI-compat ecosystem), Hermes (Nous Research's agent framework), and the OpenAI Responses-API world. The four big differentiators — MCP server mode, persistent KV-cache handles, hybrid Foundation+MLX routing, and `/v1/responses` — are explicitly in scope.

A read-only audit found AFM already implements most of an agent's wishlist (tool calling across 7+ formats with auto-detection, structured output with xgrammar, deterministic seeds, logprobs, Prometheus `/metrics`, prefix caching, concurrent batch decode, gateway-mode normalization), but is missing several high-leverage agent semantics (mid-stream cancel, tokenize endpoints, request-id correlation, parallel_tool_calls flag, schema-validated tool args, persistent KV handles, MCP, hybrid routing, Responses API). The intended outcome is a tiered, multi-PR roadmap that turns AFM's existing surface into a marketable agent platform and adds what's missing in priority order.

## Strategy

Land in four tiers, smallest-and-most-leveraged first. Tiers 0–1 are a week's work and unblock Tier 2; Tier 3 is the moat (Apple-Silicon-native features no other local server can match).

## Tier 0 — Promotion only (no code)

| # | Item | Effort |
|---|---|---|
| T0.1 | Agent-readiness matrix in `README.md` covering: 7 tool-call formats auto-detected via `inferToolCallFormat()` (`Sources/MacLocalAPI/Models/MLXModelService.swift:3598`); `afm_adaptive_xml` parser (JSON-in-XML fallback, nullable flatten, EBNF); `tool_choice` (auto/none/required/named); streaming `StreamDeltaToolCall`; `<think>` + harmony channel reasoning extraction; deterministic `seed` and `logprobs/top_logprobs`; strict `json_schema` + xgrammar EBNF + `--guided-json`; radix-tree prefix cache; 4/8-bit kv quant; streaming kv-evict; fair-queue `--concurrent`; vLLM-namespaced Prometheus `/metrics`; `Retry-After: 2` on 503; gateway proxy with `reasoning → reasoning_content` normalization. | S |
| T0.2 | Per-client cookbook pages under `docs/clients/` (one file each for OpenCode, OpenClaw, Cline, Continue, Aider, Cursor, Hermes): base URL, model id, tool-format note, streaming, concurrency, KV flags. | S |

## Tier 1 — Quick wins (S effort, max agent leverage per LOC)

| # | Item | Surface | Files | Effort |
|---|---|---|---|---|
| T1.1 | Echo `X-Request-ID` / `OpenAI-Request-ID`. Read inbound, else mint `req_<uuid12>`; echo on every response and SSE comment line; surface in `error.request_id`. | HTTP header | `Sources/MacLocalAPI/Server.swift` (middleware); `Controllers/MLXChatCompletionsController.swift` already plumbs `requestId` | S |
| T1.2 | Honor `stream_options.include_usage`. Today usage is always sent; gate it. | Request body | `Models/OpenAIRequest.swift` (new struct); chat controllers' final-chunk emitter | S |
| T1.3 | Honor `parallel_tool_calls: false`. Emit at most one tool-call/turn when set. | Request body | `Models/OpenAIRequest.swift`; `Models/ToolCallStreamingRuntime.swift`; both chat controllers | S |
| T1.4 | Mid-stream cancel on client disconnect. Wire Vapor's connection-closed signal to the streaming Task; scheduler already checks `Task.isCancelled` (`Models/BatchScheduler.swift:359, 366, 368, 652`). | invisible | `Controllers/MLXChatCompletionsController.swift:503` (`createStreamingResponse`); `ChatCompletionsController.swift` | S |
| T1.5 | `POST /v1/chat/completions/{id}/cancel`. In-mem `requestId → Task` map populated by T1.1. | New endpoint | new `Controllers/CancelController.swift`; `Server.swift` route | S, deps T1.1, T1.4 |

> **Status note (PR-2 landed at `8bc58d0`)** — T1.4/T1.5 shipped with cooperative cancellation via an `InflightRequestRegistry` actor (`Sources/MacLocalAPI/Models/InflightRequestRegistry.swift`). The cancel endpoint and (future) client-disconnect path both call `bodyTask.cancel()`, which propagates through the `AsyncThrowingStream` iterator's deinit and `onTermination` to the underlying generator on the **serial** path (`MLXModelService.swift:2423`). **Caveat for concurrent mode** (`--concurrent N`): cancelling the controller's `bodyTask` stops sending bytes to the dead client immediately, but the BatchScheduler slot keeps generating to natural completion (output is dropped silently). True slot-level cancel needs a `requestId → slotUUID` mapping wired through `service.cancelBatchSlots(ids:)` (`Controllers/MLXChatServing.swift:47`). Tracked as a follow-up; not a blocker.

| T1.6 | `POST /v1/tokenize` and `POST /v1/count_tokens` (vLLM/Anthropic style). Pre-flight context-budget for Aider/Cline. Foundation path returns approximate-or-error. | New endpoints | new `Controllers/TokenizeController.swift`; reuse tokenizer from `Models/MLXModelService.swift` | S |
| T1.7 | Static `GET /openapi.json` + `/docs` (Scalar UI). Self-discovery for Cursor/Continue/Hermes. | New endpoint | `Resources/openapi.json`; `Server.swift` route | S |

## Tier 2 — Core agent semantics (M effort, must-have)

| # | Item | Why | Effort / Deps |
|---|---|---|---|
| T2.1 | **Tool-arg schema validation feedback.** Validate model-emitted args against the client's JSON schema; on mismatch, return `tool_call_validation_error` choice. Optional auto-repair via one xgrammar-constrained retry. Modes via request flag `tool_validation: off/warn/repair/strict`. Files: `Models/JSONSchemaConverter.swift`, `Models/ToolCallStreamingRuntime.swift`, `Models/XGrammarService.swift`. | Removes the universal "model returned bad JSON" footgun all agent loops have. | M, deps T1.1 |
| T2.2 | **Per-request LoRA adapter.** Hot-swap adapter per request (`request.adapter: string`); add `--adapter-registry path.json` for named adapters. Files: `main.swift`, `Models/MLXModelService.swift`, `Models/OpenAIRequest.swift`. Expose `afm:adapter_active` in metrics. | One server, multiple personas (codegen vs chat). | M |
| T2.3 | **`--<client>-config` parity.** Clone the `--openclaw-config` codepath at `main.swift:854` (`printOpenClawConfig`) for `--cline-config`, `--continue-config`, `--aider-config`, `--cursor-config`, `--opencode-config`, `--hermes-config`. Each prints the exact config stanza that client expects. Templates in `Resources/client-templates/`. | Single-command onboarding is the biggest adoption lever. | M |
| T2.4 | **OpenAI-conformance test pack.** Golden-file suite under `Tests/AgentConformanceTests/` hitting every endpoint at fixed seed; CI fails on schema drift across macOS-14/15/26. Reuse existing `Scripts/test-all-features.sh` plumbing. | Locks in the surface for the agent ecosystem. | M |
| T2.5 | **Per-message `cache_control: {type: "ephemeral"}` (Anthropic-style breakpoints).** Pin that prefix in radix cache for session TTL. Closes the gap with Anthropic prompt caching that Hermes/Aider rely on. Files: `Models/RadixTreeCache.swift`, `Models/MLXCacheResolver.swift`. | Anthropic-style breakpoints are de facto industry standard now. | M |

## Tier 3 — Differentiators (L effort, the moat)

| # | Item | Detail |
|---|---|---|
| **T3.1** | **Persistent KV-cache session handles.** `POST /v1/sessions` returns `session_id`; subsequent `chat/completions` pass `session_id` and only the *delta* messages — radix cache holds the prefix across turns. LRU + explicit `DELETE /v1/sessions/{id}`. Files: new `Controllers/SessionsController.swift`, new `Models/SessionStore.swift`, hooks in `Models/RadixTreeCache.swift` and `Models/MLXModelService.swift`. **Why first**: agents like Aider/Continue/Cline run multi-turn loops where the system prompt + repo map is 50–200k tokens; re-prefilling each turn dominates wall-clock on Apple Silicon. Single biggest user-visible win available. Effort: L. |
| **T3.2** | **MCP server mode.** `afm --mcp` (and co-served at `/mcp/sse` + `/mcp/stream`) speaks the Model Context Protocol so Claude Code, Cursor, Cline can call AFM as an MCP tool provider — exposes `chat`, `embed`, `tokenize`, `transcribe`, `tts`, `ocr`, `vision` as MCP tools; gateway-discovered backends as MCP resources. Spec: modelcontextprotocol.io 2025-06. Files: new `Sources/MacLocalAPI/MCP/` (`MCPServer.swift`, `MCPTools.swift`, `MCPResources.swift`); `main.swift` flag. Effort: L, deps T1.6. |
| **T3.3** | **`/v1/responses` (OpenAI Responses API).** `POST /v1/responses`, `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`; streaming `response.created` / `response.output_text.delta` / `response.completed` events; `previous_response_id` chaining; server-stored `input` & `output`. Files: new `Controllers/ResponsesController.swift`, new `Models/ResponseStore.swift` (sqlite or in-mem), translation to/from existing chat path; reuse `ToolCallStreamingRuntime`. Effort: L, deps T3.1 (chaining via session handles), T1.5 (cancel). |
| **T3.4** | **Hybrid Foundation+MLX auto-routing.** `model: "auto"` (or `--router policy.json`) picks Foundation Models for short low-latency turns (<1k ctx, no tools, macOS 26+) and MLX otherwise; transparent fallback if Foundation refuses (guardrails / unsupported). Decision exposed in `system_fingerprint` + `X-AFM-Router` header. Files: new `Models/HybridRouter.swift`; hooks in `Controllers/MLXChatCompletionsController.swift` and `ChatCompletionsController.swift`; telemetry to `Models/StatsAggregator.swift`. Effort: L. The Apple-Silicon-native moat — no other server can do this. |

## Sequencing

1. **PR-1**: T0.1 + T0.2 + T1.1 + T1.2 + T1.3 (one stacked PR — README + 3 small request-shape changes that all touch `OpenAIRequest.swift` together).
2. **PR-2**: T1.4 + T1.5 (cancel pair — same critical path).
3. **PR-3**: T1.6 + T1.7 (new read-only endpoints).
4. **PR-4..6**: T2.1, T2.3, T2.4 in parallel (each its own PR).
5. **PR-7**: T2.2 + T2.5 (cache + adapter, both touch `MLXModelService.swift`).
6. **PR-8**: T3.1 (sessions) — unlocks T3.3.
7. **PR-9 / PR-10**: T3.2 (MCP) and T3.4 (hybrid router) in parallel.
8. **PR-11**: T3.3 (Responses API) last.

## Critical files

- `Sources/MacLocalAPI/Server.swift` — route registration, middleware
- `Sources/MacLocalAPI/main.swift` — CLI flags, `printOpenClawConfig` (`:854`)
- `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift` — streaming path
- `Sources/MacLocalAPI/Controllers/ChatCompletionsController.swift` — Foundation path
- `Sources/MacLocalAPI/Models/OpenAIRequest.swift` / `OpenAIResponse.swift` — wire shape
- `Sources/MacLocalAPI/Models/ToolCallStreamingRuntime.swift` — tool-call surface
- `Sources/MacLocalAPI/Models/RadixTreeCache.swift` — prefix cache
- `Sources/MacLocalAPI/Models/MLXCacheResolver.swift` — cache resolution
- `Sources/MacLocalAPI/Models/BatchScheduler.swift` — concurrent decode + cancel checks
- `Sources/MacLocalAPI/Models/MLXModelService.swift` — model load, tokenizer, format inference (`:3598`)

## Functions to reuse

- `inferToolCallFormat()` — `Sources/MacLocalAPI/Models/MLXModelService.swift:3598` (auto-detect from `model_type`)
- `printOpenClawConfig()` — `Sources/MacLocalAPI/main.swift:854` (template for Tier 2.3 client configs)
- `extractHarmonyContent` / `extractHarmonyChannels` — `Controllers/MLXChatCompletionsController.swift` (already harmonized output for gpt-oss; `/v1/responses` reuses)
- `XGrammarService` — `Models/XGrammarService.swift` (constrained retry for T2.1 tool repair)
- vLLM-namespace `/metrics` — `Controllers/MetricsController.swift` (extend with `afm:adapter_active`, `afm:router_decisions_total`, session counts)
- Existing `requestId` plumbing in chat controllers (lines 100/207/267/523) — surface as response header in T1.1

## Verification

| Tier | How to verify |
|---|---|
| T0 | README diff renders cleanly; every row links to a working file path. |
| T1.1 | `curl -D-` shows `x-request-id`; pass inbound id, see it echoed. |
| T1.2 | OpenAI Python SDK strict mode passes; `include_usage:false` → no usage in last chunk. |
| T1.3 | Force a two-tool-call response; flag false → only first surfaces. |
| T1.4 | Start long generation, kill curl; `afm:num_requests_running` drops within 250 ms. |
| T1.5 | Mid-turn POST cancel → stream ends with `finish_reason:"cancelled"`. |
| T1.6 | `/v1/tokenize` count matches usage from a 1-token completion. |
| T1.7 | `npx @scalar/cli` renders `/openapi.json` without error. |
| T2.1 | Assertion suite: schema requires int, force-string output → repair → valid. |
| T2.2 | Two parallel requests with different adapters → distinct stylistic output; `afm:adapter_active` reflects both. |
| T2.3 | Each generated `--<client>-config` smoke-tested against the real client. |
| T2.4 | GH Actions matrix green on macOS-14/15/26. |
| T2.5 | Repeat request with same prefix → `cached_tokens > 0`. |
| T3.1 | Bench: 8 turns × 80k-token prefix → ≥5× TTFT speedup vs no-session. Add to `skills/test-afm-assertions`. |
| T3.2 | Register AFM in Claude Code `mcp_servers.json`; call `afm.chat` MCP tool; see streamed tokens. |
| T3.3 | OpenAI SDK `client.responses.create()` works against AFM, both single-shot and `previous_response_id` chaining. |
| T3.4 | Assertion pack: short prompt → Foundation (<200 ms TTFT); long/tool prompt → MLX; refusal → MLX fallback within one retry; `/metrics` exposes `afm:router_decisions_total{backend=}`. |

## Notes / open questions

- **"Hermes" disambiguation**: interpreted as Nous Research's Hermes Agent Framework (an OpenAI-compat consumer), not the Hermes tool-call parser already supported in `vendor/.../ToolCallFormat.swift`. If you meant the latter, T0.2's Hermes cookbook page collapses into existing tool-format docs.
- **Foundation Models limits**: Tier 3.4's auto-router needs a probe-and-cache pass for Foundation availability per `os.major`; macOS pre-26 always routes to MLX.
- **MCP transport**: prefer the newer Streamable HTTP transport over stdio for a server-mode MCP, since AFM is already an HTTP server.
