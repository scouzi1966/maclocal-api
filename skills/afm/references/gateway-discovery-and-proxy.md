# Gateway Discovery and Proxy Reference

## Scope

Use this file when changing `afm -g` behavior, backend probing, proxy normalization, or multi-backend model routing.

## Key Files

- `Sources/MacLocalAPI/Services/BackendConfiguration.swift`
- `Sources/MacLocalAPI/Services/BackendDiscoveryService.swift`
- `Sources/MacLocalAPI/Services/BackendProxyService.swift`
- `Sources/MacLocalAPI/Server.swift`

## Known Backends

Default known backends include:

- Ollama (`11434`)
- LM Studio (`1234`)
- Jan (`1337`)
- mlx-lm (`8080`)
- llama.cpp (`8081`)

Blacklisted ports can be excluded from probing (for example Jan internal port `3570`).

## Discovery Strategy

`BackendDiscoveryService.startPeriodicScanning()` runs in two phases:

1. fast known-backend probe on startup
2. background open-port scan + periodic full rescans

`refreshIfStale()` enables on-demand rescan from `/v1/models`.

Port scanning uses `lsof` to find listening TCP ports, then filters by allowed ranges before HTTP probing.

## Model Registry in Gateway Mode

`/v1/models` merges:

- local foundation entry
- discovered backend models

Capabilities are cached/probed and exposed in response model metadata.

## Proxy Behavior

`BackendProxyService` handles both non-streaming and streaming proxy requests:

- rewrites request model id to backend original id
- can strip history when model switches (clean context across models)
- injects `Authorization: Bearer afmapikeyunsafe` for compatibility-only backends

Important: `afmapikeyunsafe` is not a security control; it is a passphrase for backends that require any key.

## Response Normalization

Proxy normalization includes:

- non-streaming: map `reasoning` -> `reasoning_content`
- streaming: normalize delta reasoning field similarly
- extract `<think>` tags into `reasoning_content` where needed
- inject `timings` in final chunk for backends that do not provide timings

## Error Handling

Streaming proxy maps backend status errors to visible SSE error content, including:

- 401 auth-required guidance
- 403 access denied
- 404 model not found
- 5xx backend error

Keep this behavior user-visible in stream responses instead of silent failures.
