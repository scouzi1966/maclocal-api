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

`model` remains accepted when an OpenAI client sends its active chat model ID;
AFM routes these image-only endpoints to `MACAFM_IMAGE_MODEL`. This lets one
server expose chat and image surfaces without requiring the client to manage a
second base URL.

The current endpoint returns PNG data through `b64_json`. URL-hosted output,
streaming partial images, masks, and variations are not implemented.
