If you find this useful, please ⭐ the repo!

## Latest app release --> https://github.com/scouzi1966/maclocal-api/releases/tag/v0.9.1

> [!TIP]
> ## What's new in v0.9.1 --> afm -w or afm --webui starts the API server AND a chat web interface!. It integrates llama.cpp webui.
> ## afm -w is all you need!

> [!TIP]
> ### TLDR Chose ONE of 2 methods to install
>
> ### TLDR install with Homebrew
> ```bash
> brew tap scouzi1966/afm
> brew install afm
>
> single command
> brew install scouzi1966/afm/afm
> ```
>
> > ### OR NEW METHOD WITH PIP! 
> ```bash
> pip install macafm
> ```
> To start a webchat:
>
> afm -w

> [!TIP]
>
> ### TLDR install with pip
> ```bash
> pip install macafm
> ```

# MacLocalAPI is the repo for the afm command on macOS 26 Tahoe. The afm command (cli) allows one to access the on-device Apple LLM Foundation model from the command line in a single prompt or in API mode. It allows integration with other OS command line tools using standard Unix pipes.

# Additionally, it contains a built-in server that serves the on-device Foundation Model with the OpenAI standard SDK through an API. You can use the model with another front end such as Open WebUI. By default, launching the simple 'afm' command starts a server on port 9999 immediately! Simple, fast.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=scouzi1966/maclocal-api&type=Date)](https://star-history.com/#scouzi1966/maclocal-api&Date)

# As easy to integrate with Open-webui as Ollama

[![Swift](https://img.shields.io/badge/Swift-6.2+-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-26+-blue.svg)](https://developer.apple.com/macos/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Note: afm command supports trained adapters using Apple's Toolkit: https://developer.apple.com/apple-intelligence/foundation-models-adapter/

I have also created a wrapper tool to make the fine-tuning AFM easier on both Macs M series and Linux with CUDA using Apple's provided LoRA toolkit.

Get it here: https://github.com/scouzi1966/AFMTrainer

You can also explore a pure and private MacOS chat experience (non-cli) here: https://github.com/scouzi1966/vesta-mac-dist

# The TLDR quick installation of the afm command on MacOS 26 Tahoe:

Chose ONE of 2 methods to install (Homebrew or pip):

### Method 1: Homebrew
```bash
# Add the tap (first time only)
brew tap scouzi1966/afm

# Install or upgrade AFM
brew install afm
# OR upgrade existing:
brew upgrade afm

# Verify installation
afm --version  # Should show latest release

# Brew workaround If you are having issues upgrading, Try the following:
brew uninstall afm
brew untap scouzi1966/afm
# Then try again
```

### Method 2: pip
```bash
pip install macafm

# Verify installation
afm --version
```

**HOW TO USE afm:**
```bash

# Start the API server only (Apple Foundation Model on port 9999)
afm

# Start the API server with WebUI chat interface
afm -w

# Start with WebUI and API gateway (auto-discovers Ollama, LM Studio, Jan, etc.)
afm -w -g

# Start on a custom port with a trained LoRA adapter
afm -a ./my_adapter.fmadapter -p 9998

# Use in single prompt mode
afm -i "you are a pirate, you only answer in pirate jargon" -s "Write a story about Einstein"

# Use in single prompt mode with adapter
afm -s "Write a story about Einstein" -a ./my_adapter.fmadapter

# Use in pipe mode
ls -ltr | afm -i "list the files only of ls output"
```

A very simple to use macOS server application that exposes Apple's Foundation Models through OpenAI-compatible API endpoints. Run Apple Intelligence locally with full OpenAI API compatibility. For use with Python, JS or even open-webui (https://github.com/open-webui/open-webui).

With the same command, it also supports single mode to interact the model without starting the server. In this mode, you can pipe with any other command line based utilities. 

As a bonus, both modes allows the use of using a LoRA adapter, trained with Apple's toolkit. This allows to quickly test them without having to integrate them in your app or involve xCode.

The magic command is afm

## 🌟 Features

- **🔗 OpenAI API Compatible** - Works with existing OpenAI client libraries and applications
- **⚡ LoRA adapter support** - Supports fine-tuning with LoRA adapters using Apple's tuning Toolkit
- **📱 Apple Foundation Models** - Uses Apple's on-device 3B parameter language model
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

#### Option 3: Install from tarball

```bash
# Download the release
curl -L -o afm-v0.5.5-arm64.tar.gz https://github.com/scouzi1966/maclocal-api/releases/download/v0.5.5/afm-v0.5.5-arm64.tar.gz

# Extract and install
tar -xzf afm-v0.5.5-arm64.tar.gz
sudo cp afm /usr/local/bin/

# Verify installation
afm --version  # Should show v0.5.5
```

#### Option 4: Build from Source

```bash
# Clone the repository
git clone https://github.com/scouzi1966/maclocal-api.git
cd maclocal-api

# Build the project
swift build -c release
```

### Running the Server

```bash
# Start server on default port 9999 (Homebrew install)
afm

# Or if built from source
./.build/release/afm

# Custom port with verbose logging
afm --port 8080 --verbose

# Show help
afm --help
```

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

- **Model Scope**: Uses Apple's 3B parameter Foundation Model (optimized for on-device performance)
- **No Streaming**: Current implementation doesn't support streaming responses
- **macOS 26+ Only**: Requires the latest macOS with Foundation Models framework
- **Apple Intelligence Required**: Must be enabled in System Settings
- **Token Estimation**: Uses word-based approximation for token counting

## 🔍 Troubleshooting

### "Foundation Models framework is not available"
1. Ensure you're running **macOS 26 or later
2. Enable **Apple Intelligence** in System Settings → Apple Intelligence & Siri
3. Verify you're on an **Apple Silicon Mac**
4. Restart the application after enabling Apple Intelligence

### Server Won't Start
1. Check if the port is already in use: `lsof -i :9999`
2. Try a different port: `./MacLocalAPI --port 8080`
3. Enable verbose logging: `./MacLocalAPI --verbose`

### Build Issues
1. Ensure you have **Xcode 26 installed
2. Update Swift toolchain: `xcode-select --install`
3. Clean and rebuild: `swift package clean && swift build -c release`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone the repo
git clone https://github.com/scouzi1966/maclocal-api.git
cd maclocal-api

# Build for development
swift build

# Run tests (when available)
swift test

# Run with development flags
./.build/debug/MacLocalAPI --verbose
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

- [ ] Streaming response support
- [ ] Function calling implementation  
- [ ] Multiple model support
- [ ] Performance optimizations
- [ ] Docker containerization (when supported)
- [ ] Web UI for testing

---

**Made with ❤️ for the Apple Silicon community**

*Bringing the power of local AI to your fingertips.*
