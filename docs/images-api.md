# Images API

AFM exposes OpenAI-compatible image generation and image editing backed by
FLUX.2 Klein on the same pinned MLX runtime used for text inference. The image
model loads lazily on the first image request, so normal chat startup and
memory use are unchanged.

## Model setup

Place the Hugging Face snapshot below the configured MLX model cache:

```text
$MACAFM_MLX_MODEL_CACHE/mlx-community/FLUX.2-klein-4B-bf16/
├── transformer/
├── text_encoder/
├── tokenizer/
└── vae/
```

The default runtime quantizes the transformer to int4 after loading the bf16
snapshot. Set `MACAFM_IMAGE_QUANT=bf16`, `int8`, or `int4` to select the memory
tier. Set `MACAFM_IMAGE_MODEL` to use a different compatible cache-relative
model ID.

Requests must omit `model` or name that configured image model exactly. To
support a client that cannot select a separate image model, set
`MACAFM_IMAGE_MODEL_ALIASES` to an explicit comma-separated allowlist, for
example `Qwen3.8-27B,dall-e-3`. Unknown model IDs are rejected with
`code=unsupported_model`; AFM never silently reroutes them.

## Generate an image

```bash
curl http://127.0.0.1:9999/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/FLUX.2-klein-4B-bf16",
    "prompt": "a red fox reading beside a campfire",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json",
    "seed": 42
  }'
```

The response follows the OpenAI Images shape:

```json
{"created": 1788091200, "data": [{"b64_json": "iVBORw0KGgo..."}]}
```

## Edit an image

```bash
curl http://127.0.0.1:9999/v1/images/edits \
  -F 'model=mlx-community/FLUX.2-klein-4B-bf16' \
  -F 'prompt=turn the daytime scene into moonlight' \
  -F 'image=@input.png;type=image/png' \
  -F 'size=1024x1024' \
  -F 'response_format=b64_json'
```

When an OpenAI client sends an alias explicitly listed in
`MACAFM_IMAGE_MODEL_ALIASES`, AFM routes the request to `MACAFM_IMAGE_MODEL`.
Because that opt-in routing emulates rather than exactly honors the requested
`model`, successful responses include:

```text
X-AFM-Compatibility: emulated
X-AFM-Emulated-Parameters: model
```

## Compatibility and errors

AFM does not silently ignore behavior-changing OpenAI Images controls. A known
but unavailable control returns HTTP 400 with an OpenAI-compatible,
parameter-specific error:

```json
{
  "error": {
    "message": "Parameter 'background' is not supported by the configured AFM image model",
    "type": "invalid_request_error",
    "code": "unsupported_parameter",
    "param": "background",
    "request_id": "req_..."
  }
}
```

The same request ID is returned in `X-Request-ID` and `OpenAI-Request-ID`.
Malformed values use `code=invalid_request_error`; an incorrect request media
type returns HTTP 415 with `code=unsupported_media_type`; unknown fields are
rejected with `code=unknown_parameter` so misspelled controls cannot disappear
silently. Unsupported HTTP methods return 405 and `Allow: POST, OPTIONS`.
Permanent capability gaps are deliberately reported as non-retryable client
errors rather than 501 server errors.

Generation JSON is limited to 1 MB. Edit multipart requests are limited to 25
MB, with one non-empty PNG, JPEG, or WebP input of at most 20 MB. Prompts are
limited to 32,000 characters. Explicit sizes must be 64–2048 pixels in each
dimension, divisible by 16, and between 1:3 and 3:1 aspect ratio. These checks
prevent the provider from silently rounding dimensions or failing after model
loading. A missing model snapshot returns retryable HTTP 503; invalid image
bytes return HTTP 400 with `param=image`; unexpected provider failures return a
correlated OpenAI-shaped HTTP 500.

| Control | AFM behavior |
| --- | --- |
| `prompt`, `size`, `n` (1-4), `response_format=b64_json`, `output_format=png`, `seed` | Exact |
| exact configured `model` | Exact |
| alias listed in `MACAFM_IMAGE_MODEL_ALIASES` | Emulated routing with response headers |
| `stream=false`, `user` | Accepted compatibility values |
| `background`, `moderation`, `output_compression`, `partial_images`, `quality`, `style` | Rejected with `unsupported_parameter` |
| edit `input_fidelity`, `mask`, and `image[]` multi-input syntax | Rejected with `unsupported_parameter` |
| `stream=true`, `response_format=url`, `output_format=jpeg/webp`, `n` 5-10 | Rejected with `unsupported_parameter` |

The current endpoint returns PNG data through `b64_json`. URL-hosted output,
streaming partial images, masks, and variations are not implemented.
