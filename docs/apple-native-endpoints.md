# Apple-native endpoints — routing convention (audit for #133)

This is the audit called for in #133: confirm that the Apple-native HTTP capabilities
(Vision, Speech, Embeddings) are integrated **consistently** on the main server, and define
the convention that #132 (embeddings on the main server) lands on. Output: the agreed pattern
+ a short list of remaining drift.

## The capabilities and where they live (after #132)

| Capability | Routes (main server `:9999`) | Controller |
|---|---|---|
| Vision OCR | `POST /v1/vision/ocr` (+ classify / barcode / saliency / tables) | `VisionAPIController` |
| Speech | `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, `GET /v1/audio/voices` | `SpeechAPIController` |
| Embeddings | `POST /v1/embeddings` | `EmbeddingsController` (unified, lazy Apple-NL) — **new in #132** |

All three are registered on the main `Server.swift` router. `afm embed` remains a dedicated
standalone server on `:9998` for embeddings-only deployments.

## The convention (what "consistent" means)

1. **Error envelope** — every endpoint returns the OpenAI error shape via `OpenAIError`
   (`{ "error": { "message", "type", "code"?, "request_id"? } }`). The `type` string is
   **domain-specific** and stable: `vision_unavailable`, `speech_unavailable` / `speech_error`,
   `embedding_error`, plus shared `internal_error`. Treat these as a public contract — do not
   rename existing ones (clients may match on `type`).
2. **Availability → 503** — when the underlying Apple framework/OS isn't available, return
   **`503 Service Unavailable`** with the domain `*_unavailable` type (Speech gates macOS 13,
   Vision gates macOS 26, Embeddings surfaces `backendUnavailable` lazily). A client can treat
   503 + `*_unavailable` uniformly as "this capability isn't available here."
3. **CORS** — wildcard `Access-Control-Allow-Origin: *`, reflect the preflight
   `Access-Control-Request-Headers` (so SDK-specific `x-stainless-*` etc. pass), `Vary` on the
   requested-headers list, and an `OPTIONS` handler per route group.
4. **`model` field** — chat/embeddings route on `model` (embeddings validates it against the
   loaded/known ids and 404s on unknown). Vision/Speech are single-capability endpoints and
   **ignore `model`** by design (there is one Apple framework behind each).
5. **`/v1/models` advertising** — the main `/v1/models` lists the chat model **and** the Apple
   NL embedding models (`capabilities: ["embeddings"]`). The embeddings controller does **not**
   register its own `/v1/models` on the unified server (the main server owns that route);
   `afm embed` still serves its own `/v1/models` standalone.
6. **Capability discovery** — `--help-json` (PR #131) and `OpenAPIController` should list every
   Apple-native endpoint.

## Drift found (status)

| Item | Status |
|---|---|
| `/v1/models` didn't surface embeddings | **Fixed (#132/#133)** — embedding models now advertised on the main server |
| Embeddings only on `:9998`, not the unified server | **Fixed (#132)** — `POST /v1/embeddings` on `:9999`, lazy Apple-NL load |
| Each controller hand-rolls CORS (Vision/Speech/Embeddings) | **Open (low priority)** — behavior is consistent; could be factored into one shared middleware. Not done here to avoid behavior risk. |
| Error `type` strings differ per domain | **Intentional** — domain-specific, stable; documented as contract above. No rename. |
| Availability gating differs per OS floor | **Consistent enough** — all converge on `503 + *_unavailable`. Documented. |
| `/v1/models` capability advertising for Vision/Speech | **Partial** — chat row already carries `"vision"`; Speech isn't separately advertised. Acceptable (capabilities are endpoint-discoverable via OpenAPI); revisit if a client needs it. |

## Follow-ups (not blocking)
- Factor the three hand-rolled CORS implementations into a shared `AppleNativeCORSMiddleware`.
- Optionally advertise Vision/Speech as capability rows in `/v1/models`.
