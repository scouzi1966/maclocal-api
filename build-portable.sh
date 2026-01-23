#!/bin/bash

# Build Portable AFM Executable
# This script creates a standalone, distributable afm binary

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

echo -e "${BLUE}🔨 Building Portable AFM Executable${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""

# Validate Swift version before building
validate_swift_version

# Generate version from git
get_version() {
    # Try git describe first (handles both exact tags and tags with distance)
    if git describe --tags --always --dirty 2>/dev/null; then
        return
    else
        # Fallback if no tags exist
        echo "0.0.0-$(git rev-parse --short HEAD)-$(date +%Y%m%d)"
    fi
}

BUILD_VERSION=$(get_version | tr -d '\n')
echo -e "${BLUE}📋 Build version: $BUILD_VERSION${NC}"
echo ""

# Generate BuildInfo.swift with version
echo -e "${BLUE}📝 Generating version file...${NC}"
cat > Sources/MacLocalAPI/BuildInfo.swift << EOF
// BuildInfo.swift
// Auto-generated build information - DO NOT EDIT MANUALLY

struct BuildInfo {
    static let version: String? = "$BUILD_VERSION"
}
EOF

# Clean previous builds
echo -e "${BLUE}🧹 Cleaning previous builds...${NC}"
swift package clean
rm -rf .build

# Build with optimizations
echo -e "${BLUE}⚡ Building release binary with optimizations...${NC}"
swift build -c release \
    --product afm \
    -Xswiftc -O \
    -Xswiftc -whole-module-optimization \
    -Xswiftc -cross-module-optimization

echo ""
echo -e "${GREEN}✅ Build completed${NC}"

# Get binary info
BINARY_PATH=".build/release/afm"
BINARY_SIZE=$(ls -lh "$BINARY_PATH" | awk '{print $5}')
BINARY_ARCH=$(file "$BINARY_PATH" | cut -d' ' -f3-)

echo -e "${BLUE}📊 Binary Information:${NC}"
echo "  • Path: $BINARY_PATH"
echo "  • Size: $BINARY_SIZE"
echo "  • Type: $BINARY_ARCH"
echo ""

# Strip debug symbols for smaller size
echo -e "${BLUE}🪚 Stripping debug symbols...${NC}"
strip "$BINARY_PATH"

NEW_SIZE=$(ls -lh "$BINARY_PATH" | awk '{print $5}')
echo -e "${GREEN}✅ Stripped binary size: $NEW_SIZE${NC}"
echo ""

# Show dependencies
echo -e "${BLUE}🔗 Dynamic library dependencies:${NC}"
otool -L "$BINARY_PATH" | grep -v "$BINARY_PATH:"
echo ""

# Test the binary
echo -e "${BLUE}🧪 Testing binary...${NC}"
if "$BINARY_PATH" --version &>/dev/null; then
    VERSION=$("$BINARY_PATH" --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✅ Binary test passed${NC}"
    if [[ "$VERSION" != "unknown" ]]; then
        echo -e "${BLUE}ℹ️  Version: $VERSION${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Binary test failed (may require runtime dependencies)${NC}"
fi
echo ""

# Test portability by copying to temp location
echo -e "${BLUE}🚚 Testing portability...${NC}"
TEMP_BINARY="/tmp/afm-test-$$"
cp "$BINARY_PATH" "$TEMP_BINARY"

if "$TEMP_BINARY" --help >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Portability test passed - binary works from different location${NC}"
    rm "$TEMP_BINARY"
else
    echo -e "${YELLOW}⚠️  Portability test failed${NC}"
    rm -f "$TEMP_BINARY"
fi
echo ""

# Create a simple wrapper script for convenience
echo -e "${BLUE}📝 Creating convenience script...${NC}"
cat > ".build/afm-portable" << 'EOF'
#!/bin/bash

# AFM Portable Wrapper
# This script runs afm from its build location

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/release/afm"

if [[ ! -f "$BINARY" ]]; then
    echo "❌ Error: afm binary not found at $BINARY"
    echo "   Run: swift build -c release"
    exit 1
fi

# Execute afm with all arguments
exec "$BINARY" "$@"
EOF

chmod +x ".build/afm-portable"
echo -e "${GREEN}✅ Convenience script created: .build/afm-portable${NC}"
echo ""

# Copy webui resources if they exist
if [ -d "Resources/webui" ]; then
    echo -e "${BLUE}🌐 Copying webui resources...${NC}"
    mkdir -p ".build/release/Resources/webui"
    cp -r Resources/webui/* ".build/release/Resources/webui/"
    echo -e "${GREEN}✅ WebUI resources copied to .build/release/Resources/webui/${NC}"
    WEBUI_INCLUDED=true
else
    echo -e "${YELLOW}ℹ️  WebUI not found. Run 'make webui' to build it.${NC}"
    WEBUI_INCLUDED=false
fi
echo ""

# Final summary
echo -e "${GREEN}🎉 Portable AFM build complete!${NC}"
echo ""
echo -e "${BLUE}📦 Usage options:${NC}"
echo "  1. Direct: ./.build/release/afm --port 9999"
echo "  2. Wrapper: ./.build/afm-portable --port 9999"
echo "  3. Copy anywhere: cp ./.build/release/afm /usr/local/bin/"
if [ "$WEBUI_INCLUDED" = true ]; then
    echo "  4. With WebUI: ./.build/release/afm -w"
fi
echo ""
echo -e "${BLUE}📋 Distribution:${NC}"
echo "  • The binary at ./.build/release/afm is now portable"
echo "  • Copy it anywhere on macOS and it should work"
echo "  • Requires macOS with Swift runtime (10.14.4+)"
if [ "$WEBUI_INCLUDED" = true ]; then
    echo "  • WebUI included in Resources/webui/"
fi
echo ""
echo -e "${YELLOW}💡 Tip: Run './create-distribution.sh' to create a distribution package${NC}"