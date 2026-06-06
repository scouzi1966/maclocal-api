# Embeddings API

`POST /v1/embeddings` exposes Apple's NaturalLanguage contextual embeddings over an OpenAI-compatible HTTP endpoint.

It runs as a dedicated server started by `afm embed`, separate from the chat/MLX server. It is intended for:
- local-first RAG, semantic search, and clustering workflows that already speak the OpenAI embeddings interface
- on-device embedding using Apple's own models — no Hugging Face model downloads

The backend is Apple's `NLContextualEmbedding`. There is no MLX or sentence-transformers backend in this release.

Embedding runs entirely on-device. The contextual embedding assets are OS-provided; the framework may download them from Apple on first use if they are not already present (`afm embed` requests them at startup), after which inference is local and offline.

## Requirements

- macOS with Apple's NaturalLanguage contextual embedding support
- Apple Silicon
- Network access on first run if the contextual embedding assets are not already installed (they are downloaded once, then cached by the OS)

## Running the Server

```bash
afm embed                          # default model on port 9998
afm embed -m apple-nl-contextual-multi --port 9998
afm embed --list-models            # print shipped model ids and exit
```

CLI options:
- `-m, --model`: embedding model id (default: `apple-nl-contextual-en`)
- `-p, --port`: server port (default: `9998`)
- `-H, --hostname`: bind address (default: `127.0.0.1`)
- `-v, --verbose`: enable verbose logging
- `-V, --very-verbose`: log full requests/responses
- `--list-models`: list available embedding model ids and exit

The server loads a single model at startup. Native dimension and maximum input length are read from the Apple framework at load time, so they track OS updates rather than being hard-coded.

## Shipped Models

| Model id | Coverage |
|----------|----------|
| `apple-nl-contextual-en` | English (default) |
| `apple-nl-contextual-multi` | Latin-script multilingual |

Apple's multilingual contextual model is Latin-script only; non-Latin scripts are out of scope for this backend.

`GET /v1/models` advertises **only** the model the running server actually loaded — not the full shipped list — so a client cannot discover an id the server can't serve.

## Request

```json
{
  "model": "apple-nl-contextual-en",
  "input": "The quick brown fox",
  "encoding_format": "float",
  "dimensions": 256
}
```

Fields:
- `input` (required): one of
  - a string — `"hello"`
  - an array of strings — `["hello", "world"]`
  - a token-id array — `[15, 234, 91]`
  - an array of token-id arrays — `[[15, 234], [91, 7]]`
- `model` (optional): must match the loaded model id; any other id returns `404`. Defaults to the loaded model when omitted.
- `encoding_format` (optional): `float` (default) or `base64`
- `dimensions` (optional): Matryoshka-style truncation. Must be between `1` and the model's native dimension. The vector is truncated to `dimensions` and L2-renormalized.
- `user` (optional): accepted and ignored, for OpenAI client compatibility.

All output vectors are L2-normalized. When `dimensions` is omitted the full native vector is returned (normalized).

### Token-id input

Pre-tokenized input is accepted as integer arrays. Empty token-id arrays (e.g. `[[]]` or `[[1,2],[]]`) are rejected with `400` before reaching the backend.

```bash
curl http://localhost:9998/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": [[15, 234, 91]], "model": "apple-nl-contextual-en"}'
```

### Base64 output

With `"encoding_format": "base64"`, each embedding is returned as a base64 string of little-endian `float32` values instead of a JSON array.

## Response Shape

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, "..."]
    }
  ],
  "model": "apple-nl-contextual-en",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

Notable fields:
- `data`: one entry per input, in request order, each with its `index` and `embedding`
- `embedding`: a float array, or a base64 string when `encoding_format` is `base64`
- `usage.prompt_tokens` / `usage.total_tokens`: summed token counts across all inputs

### Truncation header

Apple's `NLContextualEmbedding` silently truncates inputs longer than the model's maximum sequence length. When any input is truncated, the response includes:

```
X-Embedding-Truncated: <number of truncated inputs>
```

The header is exposed to browsers via `Access-Control-Expose-Headers`. It is omitted when nothing was truncated.

## CORS

`POST /v1/embeddings` and `GET /v1/models` both answer CORS preflight (`OPTIONS`). Responses set `Access-Control-Allow-Origin: *` and reflect the browser's `Access-Control-Request-Headers` (falling back to `Content-Type, Authorization, OpenAI-Organization, OpenAI-Project`). Preflight responses set `Vary: Access-Control-Request-Headers` so intermediary caches don't replay a preflight computed for one client's header set against another.

## Error Semantics

The endpoint maps failures to HTTP statuses:
- `400 Bad Request`: malformed JSON, missing or empty input, empty token-id array, invalid `dimensions` (outside `1…native`), unknown `encoding_format`, tokenization failure
- `404 Not Found`: requested `model` is not the loaded model
- `413 Payload Too Large`: request body exceeds 1 MiB
- `500 Internal Server Error`: backend dimension drift or other internal failure
- `503 Service Unavailable`: embedding backend or assets unavailable

Other 4xx `AbortError`s (e.g. `415 Unsupported Media Type`) pass through with their original status. Errors use the OpenAI-compatible envelope:

```json
{
  "error": {
    "message": "Embedding model not found: text-embedding-3-small",
    "type": "embedding_error"
  }
}
```

## Health Check

```bash
curl http://localhost:9998/health
```

Reports server status and the build version.

## Deliberately Out of Scope

- MLX / sentence-transformers embedding backend (BGE, etc.)
- Non-Latin script coverage for the multilingual backend
- Matryoshka beyond the truncate-and-renormalize behavior above
- `/v1/batch/embeddings`

## Validation

Covered by the embeddings XCTest suite (controller + registry tests):

```bash
swift test --disable-sandbox
```
