---
name: afm
description: Maintain and extend AFM (maclocal-api), a Swift OpenAI-compatible local LLM server and CLI for Apple Foundation Models, MLX models, API gateway proxying, and Vision OCR. Use when working on AFM CLI commands (`afm`, `afm mlx`, `afm vision`), OpenAI `/v1/chat/completions` and `/v1/models` behavior, streaming SSE, tool-calling, structured outputs, reasoning extraction, AFMKit integration, WebUI packaging, or AFM build/test/release scripts.
---

# AFM Skill

AFM (maclocal-api) is a local OpenAI-compatible API server and CLI for:

- Apple Foundation Models (`afm`)
- MLX models from Hugging Face (`afm mlx`)
- Local backend gateway/proxy mode (`afm -g`)
- Vision OCR/table extraction (`afm vision`)

## 1. Overview and Triggers

Use this skill when working on:

- CLI flags, argument dispatch, or mode routing (`afm`, `afm mlx`, `afm vision`)
- OpenAI-compatible routes (`/v1/chat/completions`, `/v1/models`)
- Streaming SSE behavior and response compatibility
- MLX tool calling, parser overrides, reasoning extraction, structured outputs
- Gateway backend discovery/proxying (Ollama, LM Studio, Jan, local OpenAI endpoints)
- Vision OCR and table extraction
- AFMKit provider integration, build scripts, regression tests

## 2. Key File Reference

| Purpose | File Path |
|---------|-----------|
| CLI modes, flags, single-prompt and stdin behavior | `Sources/MacLocalAPI/main.swift` |
| Vision command | `Sources/MacLocalAPI/VisionCommand.swift` |
| Vapor server + route registration | `Sources/MacLocalAPI/Server.swift` |
| Foundation chat completions controller | `Sources/MacLocalAPI/Controllers/ChatCompletionsController.swift` |
| MLX chat completions controller | `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift` |
| Foundation model service | `Sources/MacLocalAPI/Models/FoundationModelService.swift` |
| MLX model service | `Sources/MacLocalAPI/Models/MLXModelService.swift` |
| OpenAI request schema | `Sources/MacLocalAPI/Models/OpenAIRequest.swift` |
| OpenAI response schema | `Sources/MacLocalAPI/Models/OpenAIResponse.swift` |
| Backend discovery | `Sources/MacLocalAPI/Services/BackendDiscoveryService.swift` |
| Backend proxying | `Sources/MacLocalAPI/Services/BackendProxyService.swift` |
| Backend definitions/capabilities | `Sources/MacLocalAPI/Models/BackendConfiguration.swift` |
| Provider boundary check | `Scripts/check-afmkit-consumer-boundary.sh` |
| Full bootstrap build | `Scripts/build-from-scratch.sh` |

## 3. Core Rules

- Work from `maclocal-api/` project root.
- Keep OpenAI compatibility first: request fields, response shape, SSE chunk format.
- Keep Foundation and MLX behavior aligned where practical.
- Treat AFMKit as the only provider source of truth. Use
  `MACLOCAL_AFMKIT_WORKSPACE_PATH` for paired development, then bump the exact
  AFMKit version after its changes are released.
- Preserve CORS and streaming headers.
- Keep both streaming and non-streaming paths correct for every new feature.

## 4. Quick Start Commands

```bash
# Build against the exact AFMKit release
make build

# Debug build
make debug

# Full bootstrap build
./Scripts/build-from-scratch.sh

# Provider boundary
./Scripts/check-afmkit-consumer-boundary.sh

# Run Foundation server
./.build/debug/afm -p 9999 -v

# Run MLX server
./.build/debug/afm mlx -m mlx-community/Qwen3-0.6B-4bit -p 9999 -v

# Single prompt
./.build/debug/afm mlx -m mlx-community/Qwen3-0.6B-4bit -s "hello"

# Vision OCR
./.build/debug/afm vision -f media/ocr.png
```

## 5. Primary Workflow

1. Identify target mode and load only relevant reference files.
2. Update shared request/response models first when adding API-visible behavior.
3. Thread parameters through CLI -> server/controller -> service -> response.
4. Validate non-streaming and streaming paths.
5. Run targeted checks, then broader regression scripts.
6. Update README examples if user-visible behavior changed.

## 6. Reference Links

| Reference | When to Use |
|-----------|-------------|
| `references/architecture.md` | Route wiring, lifecycle, cross-mode request flow |
| `references/cli-and-modes.md` | Flag behavior, command dispatch, mode defaults |
| `references/api-contract.md` | OpenAI request/response compatibility details |
| `references/mlx-inference-and-tool-calling.md` | MLX streaming, tool calls, reasoning extraction, multimodal |
| `references/gateway-discovery-and-proxy.md` | Gateway scans, backend capabilities, proxy normalization |
| `references/patch-build-test.md` | Vendor patch process, build scripts, regression commands |

## 7. Validation Checklist

- CLI parse/help:
  - `./.build/debug/afm --help`
  - `./.build/debug/afm mlx --help`
  - `./.build/debug/afm vision --help`
- Server health/models:
  - `curl http://127.0.0.1:9999/health`
  - `curl http://127.0.0.1:9999/v1/models`
- Regression scripts:
  - `./test-all-features.sh`
  - `./Scripts/afm-cli-tests.sh`
  - `./test-streaming.sh`
  - `./test-go.sh`
  - `./test-metrics.sh`
