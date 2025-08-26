# MacLocalAPI

[![Swift](https://img.shields.io/badge/Swift-6.2+-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-26+-blue.svg)](https://developer.apple.com/macos/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A macOS server application that exposes Apple's Foundation Models through OpenAI-compatible API endpoints. Run Apple Intelligence locally with full OpenAI API compatibility.

## 🌟 Features

- **🔗 OpenAI API Compatible** - Works with existing OpenAI client libraries and applications
- **⚡ LoRA adapater support** - Supports fine-tuning with LoRA adapaters using Apple's tuning Toolkit
- **📱 Apple Foundation Models** - Uses Apple's on-device 3B parameter language model
- **🔒 Privacy-First** - All processing happens locally on your device
- **⚡ Fast & Lightweight** - No network calls, no API keys required
- **🛠️ Easy Integration** - Drop-in replacement for OpenAI API endpoints
- **📊 Token Usage Tracking** - Provides accurate token consumption metrics

## 📋 Requirements

- **macOS 26 (Tahoe) Beta 5** or later
- **Apple Silicon Mac** (M1/M2/M3/M4 series)
- **Apple Intelligence enabled** in System Settings
- **Xcode 26 beta 5** or later (for building from source)

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
#### Option 2: Install from tarball

```bash
# Download the release
curl -L -o afm-v0.5.5-arm64.tar.gz https://github.com/scouzi1966/maclocal-api/releases/download/v0.5.5/afm-v0.5.5-arm64.tar.gz

# Extract and install
tar -xzf afm-v0.5.5-arm64.tar.gz
sudo cp afm /usr/local/bin/

# Verify installation
afm --version  # Should show v0.5.5
```

#### Option 3: Build from Source

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

```bash
OVERVIEW: macOS server that exposes Apple's Foundation Models through OpenAI-compatible API

USAGE: afm [--port <port>] [--verbose] [--no-streaming] [--instructions <instructions>] [--single-prompt <single-prompt>]

OPTIONS:
  -p, --port <port>       Port to run the server on (default: 9999)
  -v, --verbose           Enable verbose logging
  --no-streaming          Disable streaming responses (streaming is enabled by default)
  -i, --instructions <instructions>
                          Custom instructions for the AI assistant (default: You are a helpful assistant)
  -s, --single-prompt <single-prompt>
                          Run a single prompt without starting the server
  --version               Show the version.
  -h, --help              Show help information.

Note: afm also accepts piped input from other commands, equivalent to using -s with the piped content as the prompt.

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
1. Ensure you're running **macOS 26 Beta 5** or later
2. Enable **Apple Intelligence** in System Settings → Apple Intelligence & Siri
3. Verify you're on an **Apple Silicon Mac**
4. Restart the application after enabling Apple Intelligence

### Server Won't Start
1. Check if the port is already in use: `lsof -i :9999`
2. Try a different port: `./MacLocalAPI --port 8080`
3. Enable verbose logging: `./MacLocalAPI --verbose`

### Build Issues
1. Ensure you have **Xcode 26 beta 5** installed
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
