If you find this useful, please ⭐ the repo! &nbsp; Also check out [Vesta AI Explorer](https://kruks.ai/)! — my full-featured native macOS AI app.

## Install

|  | Stable (v0.9.12) | Nightly (afm-next) |
|---|---|---|
| **Homebrew** | `brew install scouzi1966/afm/afm` | `brew install scouzi1966/afm/afm-next` |
| **pip** | `pip install macafm` | `pip install --extra-index-url https://kruks.ai/afm/wheels/simple/ macafm-next` |
| **Release notes** | [v0.9.12](https://github.com/scouzi1966/maclocal-api/releases/tag/v0.9.12) | [v0.9.13-next](https://github.com/scouzi1966/maclocal-api/releases/tag/nightly-20260613-5aad36d) |

### 🔨 Build from source — one command

Clone and build everything (submodules, patches, WebUI, release binary) with a single script. It checks your toolchain, auto-installs what it can (Node via Homebrew), and tells you what to install manually (Xcode Command Line Tools) — no AI agent or project knowledge required. The script initializes submodules for you, so a plain `git clone` is all you need.

```bash
git clone https://github.com/scouzi1966/maclocal-api.git
cd maclocal-api
./build.sh
```

That's it. The `afm` binary lands in `.build/release/afm`. Add `--install` to also install it to `/usr/local/bin` (on your `PATH` by default; uses `sudo` if needed):

```bash
./build.sh --install
```

Run `./build.sh --help` for all options (`--debug`, `--skip-webui`, `--yes` for non-interactive/CI).

> [!TIP]
> **Switching between stable and nightly:**
> ```bash
> brew unlink afm && brew install scouzi1966/afm/afm-next   # switch to nightly
> brew unlink afm-next && brew link afm                      # switch back to stable
> ASSUMES you did a brew install scouzi1966/afm/afm previously
> ```

### Install a previous version

Older stable releases are kept as pinned formulae in the Homebrew tap and as version-pinned wheels on PyPI. Useful for reproducing an issue against a specific build or rolling back without waiting for a new release.

**Homebrew (pinned stable formulae):** `afm@<version>` — available for `0.9.0`, `0.9.1`, `0.9.3`–`0.9.10`.

```bash
brew install scouzi1966/afm/afm@0.9.10      # install v0.9.10
brew uninstall afm                          # if current afm is already installed
brew link afm@0.9.10                        # expose `afm` on PATH
afm --version                               # → v0.9.10
```

**Homebrew (pinned nightly formulae):** `afm-next@<full-version>` — e.g. `afm-next@0.9.11-next.9c3225e.20260418`. Lists of available pinned nightlies are at [github.com/scouzi1966/homebrew-afm](https://github.com/scouzi1966/homebrew-afm).

```bash
brew install scouzi1966/afm/afm-next@0.9.11-next.9c3225e.20260418
```

**pip (version-pinned wheels):** any published release.

```bash
pip install macafm==0.9.10                  # previous stable
pip install --extra-index-url https://kruks.ai/afm/wheels/simple/ \
  macafm-next==0.9.11.dev20260418           # pinned nightly
```

> [!NOTE]
>
> 31 Mar, 2026. AFM was pinned to an older version of https://github.com/huggingface/swift-huggingface. I have now pinned to the latest which uses hub for model cache. The older version downloaded models to the ~/Documents/Huggingface folder which was causing some pain with iCloud sync. They are now stored under ~/.cache which is not in iCloud scope. the TLDR is that models will be re-downloaded again. You can manually delete the older models located in ~/Documents/Huggingface to regain some valuable space available (spring cleaning!). Please report any issues.
> 
> **Attention M-series Mac AI enthusiasts!** You don't need to be a Swift developer to explore. Vibe coding really allows anyone to participate in this project. A lot of the hype is real! It does work.
>
> [Fork this repo](https://github.com/scouzi1966/maclocal-api/fork) first, then clone your fork to submit PRs:
>
> ```bash
> git clone https://github.com/<your-username>/maclocal-api.git   
> cd maclocal-api
> claude
> /build-afm
> ```
>
> To just experiment locally
> 
> ```bash
> git clone https://github.com/scouzi1966/maclocal-api.git   
> cd maclocal-api
> claude
> /build-afm
> ```
>
> /build-afm is an AI skill that builds for the first time so that you can start coding
>
> Start vibe coding! I will add support for skills with more coding agents in the future.

# afm — Run Any MLX LLM on Your Mac, 100% Local

Extensive testing of Qwen3.5-35B-A3B with afm. Uses an experimental technique with Claude and Codex as judges for evaluation scoring. Click the link below to view test results.

### [afm-next Nightly Test Report — Qwen3.5-35B-A3B Focus](https://kruks.ai/macafm/)

Run open-source MLX models **or** Apple's on-device Foundation Model through an OpenAI-compatible API. Built entirely in Swift for maximum Metal GPU performance. No Python runtime, no cloud, no API keys.

## What's new in afm-next

> [!IMPORTANT]
> The nightly build is the future stable release. It includes everything in v0.9.12 plus:
> - No new features yet — nightly is currently in sync with the stable release

> [!TIP]
> 🙏 **Huge thanks to [@jesserobbins](https://github.com/jesserobbins)** — first-time contributor, landed two substantial features in this cycle (Vision OCR + Speech transcription). Both PRs brought afm's Apple-native capabilities from the CLI into first-class HTTP APIs. Contributions of this size and quality from a new contributor are rare and appreciated.

## Quick Start

```bash
# Run any MLX model with WebUI
afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit -w

# Or any smaller model
afm mlx -m mlx-community/gemma-3-4b-it-8bit -w

# Chat from the terminal (auto-downloads from Hugging Face)
afm mlx -m Qwen3-0.6B-4bit -s "Explain quantum computing"

# Interactive model picker (lists your downloaded models)
MACAFM_MLX_MODEL_CACHE=/path/to/models afm mlx -w

# Apple's on-device Foundation Model with WebUI
afm -w
```

## Why AFM for agents

afm is built for agentic clients — OpenCode, OpenClaw, Cline, Continue.dev, Aider, Cursor, Hermes — that drive multi-turn tool-using LLM loops against a local OpenAI-compatible endpoint. The capabilities below are already in the box:

| Capability | What it gets you | Where it lives |
|---|---|---|
| **7+ tool-call formats, auto-detected** | json, lfm2, xmlFunction (Qwen3-Coder), glm4, gemma, kimiK2, minimaxM2 picked from `model_type` in `config.json` — no per-model tuning | `MLXModelService.swift:inferToolCallFormat` |
| **`afm_adaptive_xml` parser** | JSON-in-XML fallback, type coercion, nullable schema flatten, fuzzy tool-name match — survives the malformed XML real models emit | `Models/ToolCallStreamingRuntime.swift` |
| **`tool_choice`: auto / none / required / named function** | Standard OpenAI semantics; named-function forcing routed end-to-end | `Models/OpenAIRequest.swift:ToolChoice` |
| **Streaming tool-call deltas** | Token-level start/end tag detection; content outside tool calls streams normally | `Controllers/MLXChatCompletionsController.swift` |
| **`<think>` + harmony channel reasoning extraction** | Routes Qwen/DeepSeek `<think>…</think>` and gpt-oss `<\|channel\|>analysis…` into `reasoning_content` so the WebUI/agent can show it separately | `Controllers/MLXChatCompletionsController.swift:extractThinkTags / extractHarmonyChannels` |
| **Strict `json_schema` + xgrammar EBNF** | Guaranteed-valid JSON via token-level grammar enforcement when `--enable-grammar-constraints` is on | `Models/XGrammarService.swift` |
| **`--guided-json` server default** | One CLI flag pins a schema across every chat request that omits its own `response_format` (Foundation + MLX backends) | `Sources/MacLocalAPI/main.swift` |
| **Deterministic `seed`, `logprobs`, `top_logprobs`** | All sampling controls (temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, seed, logprobs+top_logprobs up to 20) plumbed end-to-end | `Models/OpenAIRequest.swift` + `Scripts/patches/Evaluate.swift` |
| **Radix-tree prefix KV cache** | `--enable-prefix-caching` reuses KV across turns — agent loops with stable system prompts get prefill for free | `Models/RadixTreeCache.swift` |
| **4/8-bit KV quantization** | `--kv-bits 4|8` cuts memory ~2-4× on long-context turns | `Sources/MacLocalAPI/main.swift` |
| **Concurrent batch decode** | `--concurrent N` runs N requests through one model with fair queueing; vLLM-style metrics expose queue depth | `Models/BatchScheduler.swift` |
| **vLLM-namespaced Prometheus `/metrics`** | `afm:max_concurrent_slots`, `afm:num_requests_running`, `afm:num_requests_waiting`, plus per-request token/timing histograms | `Controllers/MetricsController.swift` |
| **`Retry-After: 2` on 503** | Tells well-behaved agents (LangChain, OpenAI SDK) when to retry — no thundering herd | `Controllers/MLXChatCompletionsController.swift` |
| **Multi-backend gateway mode** | `--gateway` discovers Ollama / LM Studio / Jan on the same machine and proxies them under one OpenAI surface, normalizing `reasoning` → `reasoning_content` | `Models/BackendDiscoveryService.swift` + `BackendProxyService.swift` |
| **`X-Request-ID` / `OpenAI-Request-ID` echo** | Inbound IDs are honored; otherwise minted as `req_<uuid12>`. Echoed on every response and inside `error.request_id` for retry correlation | `Server.swift:RequestIDMiddleware` |
| **`stream_options.include_usage` honored** | Suppress the final usage chunk when the client doesn't want it (matches OpenAI strict mode) | `Models/OpenAIRequest.swift:StreamOptions` |
| **`parallel_tool_calls: false` honored** | Truncate to a single tool call per turn for agents that want serial execution | `Controllers/MLXChatCompletionsController.swift:finalizeAssistantTurn` |
| **Speech (transcribe + TTS) and Vision OCR** | `/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/ocr` — agents can hand off audio/image inputs without a separate service | `Controllers/SpeechAPIController.swift`, `VisionAPIController.swift` |
| **On-device embeddings for RAG** | `/v1/embeddings` from Apple's NaturalLanguage model — OpenAI-compatible vectors for retrieval/semantic search. Runs as a dedicated `afm embed` server (:9998), separate from the chat endpoint | `Controllers/EmbeddingsController.swift` |
| **Per-client config generators** | `afm mlx -m <model> --openclaw-config` prints a paste-ready provider config; cookbook recipes in [`docs/clients/`](docs/clients/) cover OpenCode, OpenClaw, Cline, Continue.dev, Aider, Cursor, Hermes | `Sources/MacLocalAPI/main.swift:printOpenClawConfig` |

See [`docs/clients/`](docs/clients/) for one-page recipes per agent.

## Use with OpenCode

[OpenCode](https://opencode.ai/) is a terminal-based AI coding assistant. Connect it to afm for a fully local coding experience — no cloud, no API keys. No Internet required (other than initially download the model of course!)

**1. Configure OpenCode** (`~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "macafm (local)",
      "options": {
        "baseURL": "http://localhost:9999/v1"
      },
      "models": {
        "mlx-community/Qwen3-Coder-Next-4bit": {
          "name": "mlx-community/Qwen3-Coder-Next-4bit"
        }
      }
    }
  }
}
```

**2. Start afm with a coding model:**
```bash
afm mlx -m mlx-community/Qwen3-Coder-Next-4bit -t 1.0 --top-p 0.95 --max-tokens 8192
```

**3. Launch OpenCode** and type `/connect`. Scroll down to the very bottom of the provider list — `macafm (local)` will likely be the last entry. Select it, and when prompted for an API key, enter any value (e.g. `x`) — tokenized access is not yet implemented in afm so the key is ignored. All inference runs locally on your Mac's GPU.

---

## 28+ MLX Models Tested

![MLX Models](test-reports/MLX-Models.png)

28 models tested and verified including Qwen3, Gemma 3/3n, GLM-4/5, DeepSeek V3, LFM2, SmolLM3, Llama 3.2, MiniMax M2.5, Nemotron, and more. See [test reports](test-reports/).

---

[![Swift](https://img.shields.io/badge/Swift-6.2+-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-26+-blue.svg)](https://developer.apple.com/macos/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=scouzi1966/maclocal-api&type=Date)](https://star-history.com/#scouzi1966/maclocal-api&Date)

## Related Projects

- [Vesta AI Explorer](https://kruks.ai/) — full-featured native macOS AI chat app
- [AFMTrainer](https://github.com/scouzi1966/AFMTrainer) — LoRA fine-tuning wrapper for Apple's toolkit (Mac M-series & Linux CUDA)
- [Apple Foundation Model Adapters](https://developer.apple.com/apple-intelligence/foundation-models-adapter/) — Apple's adapter training toolkit

## 🌟 Features

- **🔗 OpenAI API Compatible** - Works with existing OpenAI client libraries and applications
- **🧠 MLX Local Models** - Run any Hugging Face MLX model locally (Qwen, Gemma, Llama, DeepSeek, GLM, and 28+ tested models)
- **🌐 API Gateway** - Auto-discovers and proxies Ollama, LM Studio, Jan, and other local backends into a single API
- **⚡ LoRA adapter support** - Supports fine-tuning with LoRA adapters using Apple's tuning Toolkit
- **📱 Apple Foundation Models** - Uses Apple's on-device 3B parameter language model
- **👁️ Vision OCR** - Extract text from images and PDFs using Apple Vision via CLI and HTTP (`afm vision`, `/v1/vision/ocr`)
- **🔢 Embeddings** - OpenAI-compatible embeddings from Apple's NaturalLanguage model, on-device, via a dedicated server (`afm embed`, `/v1/embeddings`)
- **🖥️ Built-in WebUI** - Chat interface with model selection (`afm -w`)
- **🔒 Privacy-First** - All processing happens locally on your device
- **⚡ Fast & Lightweight** - No network calls, no API keys required
- **🛠️ Easy Integration** - Drop-in replacement for OpenAI API endpoints
- **📊 Token Usage Tracking** - Provides accurate token consumption metrics

## 📋 Requirements

- **macOS 26 (Tahoe) or later
- **Apple Silicon Mac** (M1/M2/M3/M4 series)
- **Apple Intelligence enabled** in System Settings
- **Xcode 26 (for building from source)

## 🚀 Quick Start

### Installation

#### Option 1: Homebrew (Recommended)

```bash
# Add the tap
brew tap scouzi1966/afm

# Install AFM
brew install afm

# Verify installation
afm --version
```
#### Option 2: pip (PyPI)

```bash
# Install from PyPI
pip install macafm

# Verify installation
afm --version
```

#### Option 3: Build from Source

```bash
# Clone the repository (build.sh initializes submodules for you)
git clone https://github.com/scouzi1966/maclocal-api.git
cd maclocal-api

# Build everything from scratch (checks/installs deps + patches + webui + release build)
./build.sh

# Or skip webui if you don't have Node.js
./build.sh --skip-webui

# Or use make (patches + release build, no webui)
make

# Run
./.build/release/afm --version
```

### Running

```bash
# API server only (Apple Foundation Model on port 9999)
afm

# API server with WebUI chat interface
afm -w

# WebUI + API gateway (auto-discovers Ollama, LM Studio, Jan, etc.)
afm -w -g

# Custom port with verbose logging
afm -p 8080 -v

# Show help
afm -h
```

### MLX Local Models

Run open-source models locally on Apple Silicon using MLX:

```bash
# Run a model with single prompt
afm mlx -m mlx-community/Qwen2.5-0.5B-Instruct-4bit -s "Explain gravity"

# Start MLX model with WebUI
afm mlx -m mlx-community/gemma-3-4b-it-8bit -w

# Interactive model picker (lists downloaded models)
afm mlx -w

# MLX model as API server
afm mlx -m mlx-community/Llama-3.2-1B-Instruct-4bit -p 8080

# Pipe mode
cat essay.txt | afm mlx -m mlx-community/Qwen3-0.6B-4bit -i "Summarize this"

# MLX help
afm mlx --help
```

Models are downloaded from Hugging Face on first use and cached locally. Any model from the [mlx-community](https://huggingface.co/mlx-community) collection is supported.

## 📡 API Endpoints

### Chat Completions
**POST** `/v1/chat/completions`

Compatible with OpenAI's chat completions API.

```bash
curl -X POST http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "foundation",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### List Models
**GET** `/v1/models`

Returns available Foundation Models.

```bash
curl http://localhost:9999/v1/models
```

### Vision OCR
**POST** `/v1/vision/ocr`

Runs Apple Vision OCR against local files, uploads, base64 payloads, `data:` URLs, and OpenAI-style image inputs.

```bash
curl -X POST http://localhost:9999/v1/vision/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "file": "/tmp/invoice.pdf",
    "recognition_level": "accurate",
    "languages": ["en-US"],
    "max_pages": 10
  }'
```

The endpoint returns structured JSON with per-document text, per-page text, text blocks, detected tables, document hints, and a top-level `combined_text` field. See [docs/vision-ocr-api.md](docs/vision-ocr-api.md) for request formats, options, and response details.

### Embeddings
**POST** `/v1/embeddings`

Serves OpenAI-compatible embeddings backed by Apple's NaturalLanguage contextual model, fully on-device. Started with `afm embed` (default port `9998`), separate from the chat server.

```bash
afm embed                       # start the embeddings server on port 9998

curl -X POST http://localhost:9998/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "apple-nl-contextual-en",
    "input": "The quick brown fox"
  }'
```

Accepts a string, an array of strings, or pre-tokenized ids; supports `float`/`base64` output and Matryoshka-style `dimensions` truncation. See [docs/embeddings-api.md](docs/embeddings-api.md) for models, request fields, response shape, and error semantics.

### Health Check
**GET** `/health`

Server health status endpoint.

```bash
curl http://localhost:9999/health
```

## 💻 Usage Examples

### Python with OpenAI Library

```python
from openai import OpenAI

# Point to your local MacLocalAPI server
client = OpenAI(
    api_key="not-needed-for-local",
    base_url="http://localhost:9999/v1"
)

response = client.chat.completions.create(
    model="foundation",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
)

print(response.choices[0].message.content)
```

### Vision OCR from OpenAI-Compatible Clients

The OCR endpoint also accepts OpenAI-style multimodal payloads. This is useful when your client already sends `messages[].content[]` parts with `image_url`.

```bash
curl -X POST http://localhost:9999/v1/vision/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Extract the invoice text"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:application/pdf;base64,..."
          }
        }
      ]
    }],
    "recognition_level": "accurate",
    "languages": ["en-US"]
  }'
```

Foundation chat requests can also auto-run Apple Vision OCR before prompting the model when:
- the request includes image content
- the request includes the built-in `apple_vision_ocr` tool
- `tool_choice` is `auto`, `required`, omitted, or explicitly selects that tool

### JavaScript/Node.js

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: 'not-needed-for-local',
  baseURL: 'http://localhost:9999/v1',
});

const completion = await openai.chat.completions.create({
  messages: [{ role: 'user', content: 'Write a haiku about programming' }],
  model: 'foundation',
});

console.log(completion.choices[0].message.content);
```

### curl Examples

```bash
# Basic chat completion
curl -X POST http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "foundation",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'

# With temperature control
curl -X POST http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "foundation",
    "messages": [{"role": "user", "content": "Be creative!"}],
    "temperature": 0.8
  }'
```

### Single Prompt & Pipe Examples

```bash
# Single prompt mode
afm -s "Explain quantum computing"

# Piped input from other commands
echo "What is the meaning of life?" | afm
cat file.txt | afm
git log --oneline | head -5 | afm

# Custom instructions with pipe
echo "Review this code" | afm -i "You are a senior software engineer"
```

## 🏗️ Architecture

```
MacLocalAPI/
├── Package.swift                    # Swift Package Manager config
├── Sources/MacLocalAPI/
│   ├── main.swift                   # CLI entry point & ArgumentParser
│   ├── Server.swift                 # Vapor web server configuration
│   ├── Controllers/
│   │   └── ChatCompletionsController.swift  # OpenAI API endpoints
│   └── Models/
│       ├── FoundationModelService.swift     # Apple Foundation Models wrapper
│       ├── OpenAIRequest.swift              # Request data models
│       └── OpenAIResponse.swift             # Response data models
└── README.md
```

## 🔧 Configuration

### Command Line Options

```
OVERVIEW: macOS server that exposes Apple's Foundation Models through
OpenAI-compatible API

Use -w to enable the WebUI, -g to enable API gateway mode (auto-discovers and
proxies to Ollama, LM Studio, Jan, and other local LLM backends).

USAGE: afm <options>
       afm mlx [<options>]      Run local MLX models from Hugging Face
       afm vision <image>       OCR text extraction from images/PDFs

OPTIONS:
  -s, --single-prompt <single-prompt>
                          Run a single prompt without starting the server
  -i, --instructions <instructions>
                          Custom instructions for the AI assistant (default:
                          You are a helpful assistant)
  -v, --verbose           Enable verbose logging
  --no-streaming          Disable streaming responses (streaming is enabled by
                          default)
  -a, --adapter <adapter> Path to a .fmadapter file for LoRA adapter fine-tuning
  -p, --port <port>       Port to run the server on (default: 9999)
  -H, --hostname <hostname>
                          Hostname to bind server to (default: 127.0.0.1)
  -t, --temperature <temperature>
                          Temperature for response generation (0.0-1.0)
  -r, --randomness <randomness>
                          Sampling mode: 'greedy', 'random',
                          'random:top-p=<0.0-1.0>', 'random:top-k=<int>', with
                          optional ':seed=<int>'
  -P, --permissive-guardrails
                          Permissive guardrails for unsafe or inappropriate
                          responses
  -w, --webui             Enable webui and open in default browser
  -g, --gateway           Enable API gateway mode: discover and proxy to local
                          LLM backends (Ollama, LM Studio, Jan, etc.)
  --prewarm <prewarm>     Pre-warm the model on server startup for faster first
                          response (y/n, default: y)
  --version               Show the version.
  -h, --help              Show help information.

Note: afm also accepts piped input from other commands, equivalent to using -s
with the piped content as the prompt.
```

### Environment Variables

The server respects standard logging environment variables:
- `LOG_LEVEL` - Set logging level (trace, debug, info, notice, warning, error, critical)

## ⚠️ Limitations & Notes

- **Model Scope**: Apple Foundation Model is a 3B parameter model (optimized for on-device performance)
- **macOS 26+ Only**: Requires the latest macOS with Foundation Models framework
- **Apple Intelligence Required**: Must be enabled in System Settings
- **Token Estimation**: Uses word-based approximation for token counting (Foundation model only; proxied backends report real counts)

## 🔍 Troubleshooting

### "Foundation Models framework is not available"
1. Ensure you're running **macOS 26 or later
2. Enable **Apple Intelligence** in System Settings → Apple Intelligence & Siri
3. Verify you're on an **Apple Silicon Mac**
4. Restart the application after enabling Apple Intelligence

### Server Won't Start
1. Check if the port is already in use: `lsof -i :9999`
2. Try a different port: `afm -p 8080`
3. Enable verbose logging: `afm -v`

### Build Issues
1. Ensure you have **Xcode 26 installed
2. Update Swift toolchain: `xcode-select --install`
3. Clean and rebuild: `swift package clean && swift build -c release`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone the repo (build.sh initializes submodules for you)
git clone https://github.com/scouzi1966/maclocal-api.git
cd maclocal-api

# Full build from scratch (submodules + patches + webui + release)
./build.sh

# Or for debug builds during development
./build.sh --debug --skip-webui

# Run with verbose logging
./.build/debug/afm -w -g -v
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Apple for the Foundation Models framework
- The Vapor Swift web framework team
- OpenAI for the API specification standard
- The Swift community for excellent tooling

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/scouzi1966/maclocal-api/issues)
3. Create a new issue with detailed information about your problem

## 🗺️ Roadmap

- [x] Streaming response support
- [x] MLX local model support (28+ models tested)
- [x] Multiple model support (API gateway mode)
- [x] Web UI for testing (llama.cpp WebUI integration)
- [x] Vision OCR subcommand
- [x] Function/tool calling (OpenAI-compatible, multiple formats)
- [ ] Performance optimizations
- [ ] [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) integration for automated tool calling validation
- [ ] Docker containerization (when supported)

---

**Made with ❤️ for the Apple Silicon community**

*Bringing the power of local AI to your fingertips.*
