#!/bin/bash

# Apple Foundation Models API (AFM) Installer
# This script builds and installs the afm CLI tool to your system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validate Swift version requirements
validate_swift_version() {
    echo -e "${BLUE}🔍 Validating Swift version requirements...${NC}"
    
    if ! command -v swift &> /dev/null; then
        echo -e "${RED}❌ Error: Swift compiler not found. Please install Xcode or Swift toolchain.${NC}"
        exit 1
    fi
    
    # Get Swift version info
    SWIFT_VERSION_OUTPUT=$(swift --version 2>&1)
    
    # Extract version numbers
    SWIFT_VERSION=$(echo "$SWIFT_VERSION_OUTPUT" | grep -E "Apple Swift version" | sed -E 's/.*Apple Swift version ([0-9]+\.[0-9]+).*/\1/')
    DRIVER_VERSION=$(echo "$SWIFT_VERSION_OUTPUT" | grep -E "swift-driver version:" | sed -E 's/swift-driver version: ([0-9.]+).*/\1/')
    TARGET_OS=$(echo "$SWIFT_VERSION_OUTPUT" | grep -E "Target:" | sed -E 's/.*Target: ([a-z0-9]+-[a-z]+-[a-z]+)([0-9]+\.[0-9]+).*/\2/')
    
    echo -e "${BLUE}Current Swift configuration:${NC}"
    echo "  Swift version: $SWIFT_VERSION"
    echo "  Driver version: $DRIVER_VERSION" 
    echo "  Target OS: $TARGET_OS"
    
    # Validate minimum requirements
    REQUIRED_SWIFT="6.2"
    REQUIRED_DRIVER="1.127.11.2"
    REQUIRED_OS="26.0"
    
    # Check Swift version (6.2+)
    if [[ "$(printf '%s\n%s\n' "$REQUIRED_SWIFT" "$SWIFT_VERSION" | sort -V | tail -n1)" != "$SWIFT_VERSION" ]]; then
        echo -e "${RED}❌ Error: Swift version $SWIFT_VERSION is below minimum required $REQUIRED_SWIFT${NC}"
        echo -e "${RED}Required: swift-driver version: $REQUIRED_DRIVER+ Apple Swift version $REQUIRED_SWIFT+ (swiftlang-6.2.0.16.14 clang-1700.3.16.4)${NC}"
        echo -e "${RED}Required: Target: arm64-apple-macosx$REQUIRED_OS+${NC}"
        exit 1
    fi
    
    # Check driver version (1.127.11.2+)
    if [[ "$(printf '%s\n%s\n' "$REQUIRED_DRIVER" "$DRIVER_VERSION" | sort -V | tail -n1)" != "$DRIVER_VERSION" ]]; then
        echo -e "${RED}❌ Error: swift-driver version $DRIVER_VERSION is below minimum required $REQUIRED_DRIVER${NC}"
        echo -e "${RED}Required: swift-driver version: $REQUIRED_DRIVER+ Apple Swift version $REQUIRED_SWIFT+ (swiftlang-6.2.0.16.14 clang-1700.3.16.4)${NC}"
        echo -e "${RED}Required: Target: arm64-apple-macosx$REQUIRED_OS+${NC}"
        exit 1
    fi
    
    # Check target OS version (26.0+)
    if [[ "$(printf '%s\n%s\n' "$REQUIRED_OS" "$TARGET_OS" | sort -V | tail -n1)" != "$TARGET_OS" ]]; then
        echo -e "${RED}❌ Error: Target OS version $TARGET_OS is below minimum required $REQUIRED_OS${NC}"
        echo -e "${RED}Required: swift-driver version: $REQUIRED_DRIVER+ Apple Swift version $REQUIRED_SWIFT+ (swiftlang-6.2.0.16.14 clang-1700.3.16.4)${NC}"
        echo -e "${RED}Required: Target: arm64-apple-macosx$REQUIRED_OS+${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Swift version validation passed${NC}"
    echo ""
}

echo -e "${BLUE}🚀 Apple Foundation Models API (AFM) Installer${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}❌ Error: This installer only works on macOS${NC}"
    exit 1
fi

# Check macOS version (Updated for realistic requirements)
MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
MACOS_MINOR=$(echo "$MACOS_VERSION" | cut -d. -f2)

echo -e "${BLUE}ℹ️  Detected macOS version: $MACOS_VERSION${NC}"

if [[ $MACOS_MAJOR -lt 15 || ($MACOS_MAJOR -eq 15 && $MACOS_MINOR -lt 1) ]]; then
    echo -e "${YELLOW}⚠️  Warning: AFM works best with macOS 15.1+ (Sequoia) for Apple Intelligence${NC}"
    echo -e "${YELLOW}   The binary will install but Apple Intelligence features require newer macOS${NC}"
    echo ""
fi

# Check if release binary exists
if [[ ! -f ".build/release/afm" ]]; then
    echo -e "${YELLOW}⚠️  Release binary not found. Building now...${NC}"
    echo ""
    
    # Validate Swift version requirements
    validate_swift_version
    
    echo -e "${BLUE}🔨 Building AFM in release mode...${NC}"
    if swift build -c release; then
        echo -e "${GREEN}✅ Build successful${NC}"
    else
        echo -e "${RED}❌ Build failed${NC}"
        exit 1
    fi
    echo ""
fi

# Determine installation directory
INSTALL_DIR="/usr/local/bin"

# Check if /usr/local/bin exists and is in PATH
if [[ ! -d "$INSTALL_DIR" ]]; then
    echo -e "${YELLOW}⚠️  Creating $INSTALL_DIR directory...${NC}"
    sudo mkdir -p "$INSTALL_DIR"
fi

# Check if directory is writable or if we need sudo
if [[ -w "$INSTALL_DIR" ]]; then
    NEED_SUDO=false
else
    NEED_SUDO=true
    echo -e "${YELLOW}ℹ️  Administrator privileges required to install to $INSTALL_DIR${NC}"
fi

# Install the binary
echo -e "${BLUE}📦 Installing afm to $INSTALL_DIR/afm...${NC}"

if [[ "$NEED_SUDO" == "true" ]]; then
    sudo cp .build/release/afm "$INSTALL_DIR/afm"
    sudo chmod +x "$INSTALL_DIR/afm"
else
    cp .build/release/afm "$INSTALL_DIR/afm"
    chmod +x "$INSTALL_DIR/afm"
fi

# Verify installation
if [[ -f "$INSTALL_DIR/afm" ]]; then
    echo -e "${GREEN}✅ Installation successful!${NC}"
    echo ""
    
    # Check if /usr/local/bin is in PATH
    if [[ ":$PATH:" == *":$INSTALL_DIR:"* ]]; then
        echo -e "${GREEN}✅ $INSTALL_DIR is in your PATH${NC}"
    else
        echo -e "${YELLOW}⚠️  $INSTALL_DIR is not in your PATH${NC}"
        echo -e "${YELLOW}   Add this line to your shell profile (~/.zshrc, ~/.bashrc):${NC}"
        echo -e "${YELLOW}   export PATH=\"$INSTALL_DIR:\$PATH\"${NC}"
        echo ""
    fi
    
    # Show version and usage
    echo -e "${BLUE}🎉 AFM is now installed!${NC}"
    echo ""
    echo -e "${BLUE}Usage examples:${NC}"
    echo "  afm --help                    # Show help"
    echo "  afm --port 8080               # Start on port 8080"
    echo "  afm --verbose                 # Enable verbose logging"  
    echo "  afm --no-streaming            # Disable streaming responses"
    echo ""
    echo -e "${BLUE}API endpoints will be available at:${NC}"
    echo "  POST http://localhost:9999/v1/chat/completions"
    echo "  GET  http://localhost:9999/v1/models"
    echo "  GET  http://localhost:9999/health"
    echo ""
    echo -e "${YELLOW}📖 Requirements:${NC}"
    echo "  • macOS 15.1+ (Sequoia) recommended"
    echo "  • Apple Intelligence enabled in System Settings"
    echo "  • Compatible Apple Silicon Mac (M1/M2/M3/M4)"
    echo ""
    
    # Test installation
    if command -v afm &> /dev/null; then
        AFM_VERSION=$(afm --version 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✅ Installation verified - afm command available${NC}"
        if [[ "$AFM_VERSION" != "unknown" ]]; then
            echo -e "${BLUE}ℹ️  Version: $AFM_VERSION${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  afm command not immediately available${NC}"
        echo -e "${YELLOW}   You may need to restart your terminal or run: source ~/.zshrc${NC}"
    fi
    
else
    echo -e "${RED}❌ Installation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎊 Installation complete! Happy coding with Apple Foundation Models!${NC}"
