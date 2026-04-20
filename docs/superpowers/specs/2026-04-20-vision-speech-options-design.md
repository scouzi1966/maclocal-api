# Vision & Speech Feature Expansion

**Date:** 2026-04-20
**Status:** Approved
**Approach:** Layered — OpenAI-compatible endpoint shapes with Apple-specific extras as optional parameters

## Overview

Expand AFM's vision and speech capabilities across three areas:
1. **TTS** — new `/v1/audio/speech` endpoint and `afm speech synthesize` CLI
2. **Transcription enhancements** — OpenAI parameter parity on existing `/v1/audio/transcriptions`
3. **Vision enhancements** — new modes (barcode, classify, saliency) and parameters on `/v1/vision/ocr`

All additions follow the same pattern: match OpenAI's API shape where one exists, expose Apple-specific capabilities as additional optional parameters.

### CLI Flag Consistency

All three features use `--format` for output format selection:
- `afm speech synthesize ... --format wav` (audio format)
- `afm speech -f audio.wav --format verbose_json` (transcription output format)
- `afm vision -f image.png --format text` (vision output format)

### Parameter Applicability

Mode-specific parameters are silently ignored when used with inapplicable modes:
- `max_labels` — only applies to `classify` mode
- `saliency_type`, `include_heat_map` — only apply to `saliency` mode
- `auto_crop` — applies to `text`, `table`, `barcode`, and `auto` modes

### Error Handling

All endpoints return HTTP 400 for invalid parameters and HTTP 413 for oversized inputs. Mode-specific "not found" cases (no text, no barcodes, no salient regions) return HTTP 200 with empty result arrays, not errors — the request succeeded, there was just nothing to find.

---

## 1. TTS — `/v1/audio/speech`

### API

**Endpoint:** `POST /v1/audio/speech`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `tts-1` | Accepted but ignored (single on-device engine). Accepts `tts-1`/`tts-1-hd`. |
| `input` | string | required | Text to synthesize. Max 4096 characters. |
| `voice` | string | `alloy` | OpenAI voice name (`alloy`, `echo`, `fable`, `nova`, `onyx`, `shimmer`). Mapped to Apple voices by gender/tone. |
| `response_format` | string | `aac` | `aac` (default), `wav`, `caf`. Apple does not support MP3 encoding natively. |
| `speed` | float | `1.0` | `0.25` to `4.0`. Mapped to `AVSpeechUtterance.rate`. Clamped to Apple's supported range. |
| `locale` | string | `en-US` | Determines voice pool if `voice`/`apple_voice` not specified. |
| `apple_voice` | string | null | Exact Apple voice identifier (e.g., `com.apple.voice.compact.en-US.Samantha`). Overrides `voice`. |

**Response:** Raw audio bytes with `Content-Type` header (`audio/aac`, `audio/wav`, `audio/x-caf`). Streams as it synthesizes.

**Input limit:** Max 4096 characters. Returns HTTP 400 if exceeded.

### Voice Listing

**Endpoint:** `GET /v1/audio/voices`

Returns available on-device voices:
```json
{
  "voices": [
    {
      "id": "com.apple.voice.compact.en-US.Samantha",
      "name": "Samantha",
      "locale": "en-US",
      "gender": "female",
      "quality": "compact"
    }
  ]
}
```

Optional query parameter: `locale` to filter (e.g., `?locale=en`).

**Note:** This endpoint is AFM-specific (no OpenAI equivalent). Useful for discovering available Apple voice identifiers.

### CLI

```bash
afm speech synthesize "Hello world"                    # Default voice, stdout
afm speech synthesize "Hello world" -o output.mp3      # Save to file
afm speech synthesize "Hello world" --voice nova       # OpenAI voice name
afm speech synthesize "Hello world" --locale ja-JP     # Japanese voice
afm speech --list-voices                               # List available voices
afm speech --list-voices --locale en                   # Filter by locale
```

### Implementation

- `SpeechSynthesisService` wrapping `AVSpeechSynthesizer`
- Delegate writes audio buffers to a pipe/stream
- AAC encoding via `AVAudioConverter` or `AudioToolbox` `AudioConverterFillComplexBuffer`
- WAV is raw PCM with header (trivial). CAF via `ExtAudioFile`.
- `SpeechAPIController` extended with new routes
- `SpeechCommand` gets `synthesize` subcommand and `--list-voices` flag

### Voice Mapping

OpenAI voice names mapped to Apple voices by gender/tone. Fallback: system default voice for locale.

| OpenAI Voice | Gender | Apple Voice (en-US) | Fallback Strategy |
|-------------|--------|--------------------|--------------------|
| `alloy` | neutral | Samantha | First available female voice |
| `echo` | male | Daniel | First available male voice |
| `fable` | male | Tom | Second available male voice |
| `nova` | female | Karen | Second available female voice |
| `onyx` | male | Alex | Third available male voice |
| `shimmer` | female | Ava | Third available female voice |

For non-en-US locales: filter voices by locale, pick by gender order. Prefer `enhanced` > `premium` > `compact` quality. If the mapped voice is not downloaded, fall back to the next available voice matching gender, then to any voice for the locale.

### CLI Subcommand Structure

```
afm speech                          # Error: requires subcommand
afm speech synthesize "text"        # TTS
afm speech transcribe -f file.wav   # Transcription (new name)
afm speech -f file.wav              # Transcription (legacy shortcut)
afm speech --list-voices            # Voice listing
```

`transcribe` is the new explicit subcommand. Bare `afm speech -f` kept as alias for backwards compat.

### Limitations

- Quality depends on which voices the user has downloaded (compact vs premium vs enhanced)
- No voice cloning or custom voices
- `AVSpeechSynthesizer` rate range is narrower than OpenAI's 0.25-4.0 — clamp to `AVSpeechUtteranceMinimumSpeechRate`...`AVSpeechUtteranceMaximumSpeechRate` and document
- OpenAI voice name mapping is approximate — see voice mapping table above
- No native MP3 encoding — AAC is the default compressed format

---

## 2. Transcription Enhancements

### New Parameters on `/v1/audio/transcriptions`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `whisper-1` | Accepted but ignored (single on-device engine). For OpenAI client compat. |
| `response_format` | string | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `language` | string | null | ISO-639-1 code. Maps to Apple locale. Alias for existing `locale`. |
| `timestamp_granularities` | array | `["segment"]` | `["word"]`, `["segment"]`, or both. Only honored with `verbose_json`. |

Existing Apple-specific params retained: `recognition_level`, `uses_language_correction`, `locale` (alias for `language`).

### Response Formats

**`json`** (current behavior):
```json
{"text": "Hello, how are you?"}
```

**`text`**: Plain string body.

**`verbose_json`**:
```json
{
  "text": "Hello, how are you?",
  "language": "en",
  "duration": 2.34,
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.42},
    {"word": "how", "start": 0.52, "end": 0.71}
  ],
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.34, "text": "Hello, how are you?", "confidence": 0.94}
  ]
}
```

**`srt`/`vtt`**: Generated from segment timestamps. Standard subtitle formats.

### CLI Additions

```bash
afm speech -f audio.wav --format verbose_json
afm speech -f audio.wav --format srt
afm speech -f audio.wav --language ja
afm speech -f audio.wav --timestamps word,segment
```

### Implementation

- `SFSpeechRecognitionResult` already provides `segments` with `timestamp`, `duration`, `substring`, and `confidence`
- Top-level `duration` in `verbose_json` is computed from the last segment's `timestamp + duration`
- Format conversion (SRT/VTT) is string templating from segment data
- `SpeechService` returns richer result struct; controller formats per `response_format`

---

## 3. Vision Enhancements

### New Parameters on `/v1/vision/ocr`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | `text` | `text`, `table`, `barcode`, `classify`, `saliency`, `auto` |
| `detail` | string | `high` | `high` (maps to `accurate`), `low` (maps to `fast`). Alias for `recognition_level`. Only applies to `text`/`table` modes. |
| `auto_crop` | bool | `false` | Run document segmentation before OCR. |
| `response_format` | string | `json` | `json`, `text`, `verbose_json`. |
| `max_labels` | int | `5` | For classify mode: number of classification labels to return. |
| `saliency_type` | string | `attention` | `attention` or `objectness`. For saliency mode. |
| `include_heat_map` | bool | `false` | For saliency mode: include base64 PNG heat map. |

Existing params retained: `recognition_level` (alias for `detail`), `verbose` (alias for `response_format: verbose_json`), `table` (alias for `mode: table`), `uses_language_correction`, `languages`, `max_pages`.

### Mode: `barcode`

Uses `VNDetectBarcodesRequest`. Supports QR, EAN-13, Code 128, Data Matrix, PDF417, Aztec, and others.

```json
{
  "results": [
    {
      "type": "QR",
      "payload": "https://example.com",
      "bounding_box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.3},
      "confidence": 0.99
    }
  ]
}
```

### Mode: `classify`

Image classification using `VNClassifyImageRequest` — returns ranked labels from Apple's ~1,000-label taxonomy with confidence scores, plus attention-based salient regions for context. This is classification, not natural-language classifying.

```json
{
  "labels": [
    {"label": "document", "confidence": 0.92},
    {"label": "text", "confidence": 0.87},
    {"label": "indoor", "confidence": 0.65}
  ],
  "salient_regions": [
    {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.8}
  ]
}
```

### Mode: `saliency`

Two Apple saliency modes:
- `attention` — where a human would look first (typically single region)
- `objectness` — where "objects" are (can return multiple regions)

```json
{
  "regions": [
    {
      "type": "attention",
      "bounding_box": {"x": 0.15, "y": 0.2, "width": 0.5, "height": 0.6}
    }
  ],
  "heat_map": "<base64 PNG if include_heat_map=true>"
}
```

### Mode: `auto`

Runs text + barcode + classify concurrently. Returns all three result sets combined. Response includes a `modes_run` array indicating which modes produced results. Saliency excluded from auto (it's a preprocessing/analysis tool, not extraction).

### Auto-Crop

`VNDetectDocumentSegmentationRequest` detects and crops the document region before passing to OCR pipeline. Two-pass, adds ~50ms.

### CLI Additions

```bash
afm vision -f photo.jpg --mode classify                          # Top 5 labels
afm vision -f photo.jpg --mode classify --max-labels 10          # More labels
afm vision -f photo.jpg --mode saliency
afm vision -f photo.jpg --mode saliency --saliency-type objectness
afm vision -f photo.jpg --mode saliency --heat-map
afm vision -f photo.jpg --mode barcode
afm vision -f photo.jpg --mode auto
afm vision -f photo.jpg --auto-crop
afm vision -f doc.png --format text
afm vision -f doc.png --detail low
```

### Implementation

- `VisionService` extended with new request types for barcode, classify, saliency
- Each mode is a separate method returning a typed result
- `auto` mode runs text + barcode + classify concurrently via `TaskGroup` (text omitted on pre-macOS 26)
- Auto-crop is a preprocessing step that feeds into text, table, barcode, and auto modes
- `VisionAPIController` routes to the correct service method based on `mode`
- `VisionCommand` gets `--mode`, `--detail`, `--auto-crop`, `--max-labels`, `--saliency-type`, `--heat-map` flags

---

## Backwards Compatibility

All new parameters are optional with defaults matching current behavior:
- `mode` defaults to `text` (current behavior)
- `detail` defaults to `high` (current `accurate`)
- `response_format` defaults to `json` (current behavior)
- Existing params (`table`, `verbose`, `recognition_level`, `locale`) kept as aliases

No breaking changes to existing API or CLI.

---

## macOS Requirements

| Feature | Minimum macOS |
|---------|---------------|
| Vision OCR (existing) | 26.0 |
| Vision barcode/classify/saliency | 13.0+ (Vision framework) |
| Speech transcription (existing) | 13.0 |
| Speech TTS | 13.0+ (AVSpeechSynthesizer) |

Note: Barcode, classify, and saliency use older Vision APIs that don't require macOS 26. Only OCR text/table extraction requires macOS 26 (Apple Intelligence).

### macOS Version Gating Architecture

The existing `VisionService` is gated behind `@available(macOS 26.0, *)`. Since barcode/classify/saliency work on macOS 13+, the service needs per-mode availability checks:

- `text`, `table`, `auto_crop` — require macOS 26.0 (`VNRecognizeTextRequest` with Apple Intelligence)
- `barcode`, `classify`, `saliency` — require macOS 13.0+
- `auto` mode — runs all available modes for the current macOS version; text is omitted on pre-26

Implementation: `VisionService` methods check `#available(macOS 26.0, *)` internally rather than gating the entire class. The controller returns HTTP 501 with a message if a mode requires a newer macOS than the host.
