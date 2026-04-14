# Vision OCR API

`POST /v1/vision/ocr` exposes Apple Vision OCR over HTTP in the same local server as the OpenAI-compatible chat endpoints.

It is intended for:
- direct OCR over HTTP without shelling out to `afm vision`
- clients that already produce OpenAI-style `image_url` content parts
- local document extraction workflows that need structured text, table output, and page-level metadata

## Requirements

- macOS 26.0 or later
- Apple Silicon
- Apple Vision available on the host machine

## Supported Inputs

The endpoint accepts one or more OCR inputs in a single request.

### Local file path

```json
{
  "file": "/tmp/invoice.pdf"
}
```

### Base64 payload

```json
{
  "data": "iVBORw0KGgoAAAANSUhEUgAA...",
  "filename": "scan.png",
  "media_type": "image/png"
}
```

### Data URL

```json
{
  "data": "data:application/pdf;base64,JVBERi0xLjcK..."
}
```

### OpenAI-style image input

```json
{
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

### OpenAI-style chat message parts

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Read this document" },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:application/pdf;base64,..."
          }
        }
      ]
    }
  ]
}
```

### Multipart upload

Use a multipart `file` field. Optional OCR controls can be supplied as form fields alongside the upload.

```bash
curl http://localhost:9999/v1/vision/ocr \
  -F "file=@/tmp/invoice.pdf" \
  -F "recognition_level=accurate" \
  -F "languages=en-US"
```

## OCR Options

```json
{
  "recognition_level": "accurate",
  "uses_language_correction": true,
  "languages": ["en-US"],
  "max_pages": 10,
  "table": false,
  "debug": false
}
```

Supported options:
- `recognition_level`: `accurate` or `fast`
- `uses_language_correction`: enables Vision language correction
- `languages`: preferred OCR language tags
- `max_pages`: page cap for multi-page PDFs
- `table`: returns structured table extraction in the document payload
- `debug`: returns raw Vision detection output instead of OCR documents

Current guardrails:
- max input size: 25 MB
- max pages per document: 50 by default
- max image dimension: 4096 px on either side
- supported formats: `png`, `jpg`, `jpeg`, `heic`, `pdf`

## Response Shape

Successful responses return a `vision.ocr` object:

```json
{
  "object": "vision.ocr",
  "mode": "text",
  "documents": [
    {
      "file": "/tmp/invoice.pdf",
      "source_type": "file",
      "text": "Page 1...\n\nPage 2...",
      "full_text": "Page 1...\n\nPage 2...",
      "page_count": 2,
      "document_hints": ["invoice", "multi_page", "table_like"],
      "pages": [
        {
          "page_number": 1,
          "text": "Page 1...",
          "width": 1024,
          "height": 768,
          "text_blocks": [],
          "tables": []
        }
      ],
      "text_blocks": [],
      "tables": []
    }
  ],
  "combined_text": "Page 1...\n\nPage 2...",
  "document_hints": ["invoice", "multi_page", "table_like"]
}
```

Notable fields:
- `mode`: `text`, `table`, or `debug`
- `documents`: one entry per resolved OCR input
- `pages`: per-page text, dimensions, blocks, and tables
- `text_blocks`: flattened OCR text blocks with confidence and bounding boxes
- `tables`: structured tables with headers, rows, row objects, CSV, and bounding boxes
- `combined_text`: all OCR text joined across all documents
- `document_hints`: inferred hints such as `invoice`, `multi_page`, or `table_like`

## Error Semantics

The endpoint maps OCR failures to HTTP statuses:
- `400 Bad Request`: missing input, unsupported format, invalid base64 or data URL
- `404 Not Found`: local file path does not exist
- `413 Payload Too Large`: request exceeds the OCR file-size limit
- `422 Unprocessable Entity`: page-limit exceeded, image too large, unreadable image, no text found, no tables found, segmentation failure
- `503 Service Unavailable`: Apple Vision OCR is unavailable on the current platform

Errors use the same JSON envelope style as the rest of the OpenAI-compatible API:

```json
{
  "error": {
    "message": "The specified file was not found",
    "type": "invalid_request_error"
  }
}
```

## Foundation Chat Integration

Foundation chat requests can auto-run Apple Vision OCR before prompting the model.

This only happens when all of the following are true:
- the request includes image content in `messages[].content[]`
- the request includes the built-in tool named `apple_vision_ocr`
- `tool_choice` is omitted, `auto`, `required`, or explicitly selects `apple_vision_ocr`

When that path is taken, OCR text is injected into the message content before the Foundation model sees the prompt.

## Validation

Validated locally with:

```bash
swift test --disable-sandbox
```

Result at the time of implementation:
- `293 tests in 19 suites passed`
