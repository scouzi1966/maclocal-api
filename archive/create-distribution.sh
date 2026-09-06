#!/bin/bash

# Create AFM Distribution Package
# This script creates a redistributable package with the afm binary

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📦 Creating AFM Distribution Package${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Get version from binary or default
VERSION=$(git describe --tags --always 2>/dev/null || echo "v0.4.0")
ARCH=$(uname -m)
DIST_NAME="afm-${VERSION}-${ARCH}"

echo -e "${BLUE}ℹ️  Package: ${DIST_NAME}${NC}"
echo -e "${BLUE}ℹ️  Architecture: ${ARCH}${NC}"
echo ""

# Build release if needed
if [[ ! -f ".build/release/afm" ]]; then
    echo -e "${YELLOW}⚠️  Release binary not found. Building...${NC}"
    swift build -c release
    echo ""
fi

# Create distribution directory
DIST_DIR="dist/${DIST_NAME}"
echo -e "${BLUE}📁 Creating distribution directory: ${DIST_DIR}${NC}"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Copy binary
echo -e "${BLUE}📋 Copying binary...${NC}"
cp .build/release/afm "$DIST_DIR/"

# Copy webui resources if available
if [[ -d "Resources/webui" ]]; then
    echo -e "${BLUE}🌐 Copying webui resources...${NC}"
    mkdir -p "$DIST_DIR/share/afm"
    cp -R Resources/webui "$DIST_DIR/share/afm/webui"
    WEBUI_INCLUDED=true
else
    echo -e "${YELLOW}ℹ️  WebUI not found (run 'make webui' to build it)${NC}"
    WEBUI_INCLUDED=false
fi

# Create portable install script
echo -e "${BLUE}📝 Creating portable install script...${NC}"
cat > "$DIST_DIR/install.sh" << 'EOF'
#!/bin/bash

# AFM Portable Installer
# This script installs the afm binary from this package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/usr/local/bin"
SHARE_DIR="/usr/local/share/afm"

echo "🚀 Installing AFM from portable package..."
echo ""

# Check if binary exists
if [[ ! -f "$SCRIPT_DIR/afm" ]]; then
    echo "❌ Error: afm binary not found in package"
    exit 1
fi

# Create install directory if needed
if [[ ! -d "$INSTALL_DIR" ]]; then
    echo "Creating $INSTALL_DIR directory..."
    sudo mkdir -p "$INSTALL_DIR"
fi

# Install binary
echo "📦 Installing afm to $INSTALL_DIR..."
if [[ -w "$INSTALL_DIR" ]]; then
    cp "$SCRIPT_DIR/afm" "$INSTALL_DIR/"
    chmod +x "$INSTALL_DIR/afm"
else
    sudo cp "$SCRIPT_DIR/afm" "$INSTALL_DIR/"
    sudo chmod +x "$INSTALL_DIR/afm"
fi

# Install webui if included
if [[ -d "$SCRIPT_DIR/share/afm/webui" ]]; then
    echo "🌐 Installing webui resources..."
    sudo mkdir -p "$SHARE_DIR/webui"
    sudo cp -r "$SCRIPT_DIR/share/afm/webui/"* "$SHARE_DIR/webui/"
    echo "   WebUI installed to $SHARE_DIR/webui/"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "Usage: afm --help"
echo "Start server: afm --port 9999"
echo "Start with webui: afm -w"
echo ""
echo "Note: Requires macOS 26+ and Apple Intelligence enabled"
EOF

chmod +x "$DIST_DIR/install.sh"

# Create README
echo -e "${BLUE}📖 Creating README...${NC}"
cat > "$DIST_DIR/README.md" << EOF
# AFM - Apple Foundation Models API

A high-performance server that exposes Apple's Foundation Models through an OpenAI-compatible API.

## Quick Start

1. **Install**: Run \`./install.sh\` (requires admin privileges)
2. **Start**: Run \`afm --port 9999\`
3. **Use**: Send requests to \`http://localhost:9999/v1/chat/completions\`

## Requirements

- **macOS**: 15.1+ (Sequoia) recommended
- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4)
- **Apple Intelligence**: Must be enabled in System Settings

## Usage

\`\`\`bash
# Show help
afm --help

# Start server on port 8080
afm --port 8080

# Enable verbose logging
afm --verbose

# Disable streaming responses
afm --no-streaming
\`\`\`

## API Endpoints

- **POST** \`/v1/chat/completions\` - Chat completions (OpenAI compatible)
- **GET** \`/v1/models\` - List available models
- **GET** \`/health\` - Health check

## Features

- ✅ OpenAI-compatible API
- ⚡ ChatGPT-style smooth streaming
- 🎛️ CLI controls for streaming
- 🛑 Proper CTRL-C shutdown
- 📝 Markdown-aware streaming (preserves code blocks)
- 🔧 Configurable ports and logging

## Example Request

\`\`\`bash
curl -X POST http://localhost:9999/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "foundation",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
\`\`\`

---

Package: ${DIST_NAME}
Architecture: ${ARCH}
Built: $(date)
EOF

# Create tarball
echo -e "${BLUE}📦 Creating tarball...${NC}"
cd dist
tar -czf "${DIST_NAME}.tar.gz" "${DIST_NAME}"
cd ..

# Cleanup
echo -e "${BLUE}🧹 Cleaning up...${NC}"
rm -rf "$DIST_DIR"

echo ""
echo -e "${GREEN}✅ Distribution package created: dist/${DIST_NAME}.tar.gz${NC}"
echo ""
echo -e "${BLUE}📋 Package contents:${NC}"
echo "  • afm binary ($(du -h .build/release/afm | cut -f1))"
echo "  • install.sh (portable installer)"
echo "  • README.md (documentation)"
if [[ "$WEBUI_INCLUDED" == "true" ]]; then
    echo "  • share/afm/webui/ (llama.cpp webui)"
fi
echo ""
echo -e "${BLUE}🚀 Usage:${NC}"
echo "  1. Extract: tar -xzf dist/${DIST_NAME}.tar.gz"
echo "  2. Install: cd ${DIST_NAME} && ./install.sh"
echo ""
echo -e "${GREEN}🎉 Ready for distribution!${NC}"
