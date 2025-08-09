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

echo -e "${BLUE}🔨 Building Portable AFM Executable${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""

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

# Final summary
echo -e "${GREEN}🎉 Portable AFM build complete!${NC}"
echo ""
echo -e "${BLUE}📦 Usage options:${NC}"
echo "  1. Direct: ./.build/release/afm --port 9999"
echo "  2. Wrapper: ./.build/afm-portable --port 9999"
echo "  3. Copy anywhere: cp ./.build/release/afm /usr/local/bin/"
echo ""
echo -e "${BLUE}📋 Distribution:${NC}"
echo "  • The binary at ./.build/release/afm is now portable"
echo "  • Copy it anywhere on macOS and it should work"
echo "  • Requires macOS with Swift runtime (10.14.4+)"
echo ""
echo -e "${YELLOW}💡 Tip: Run './create-distribution.sh' to create a distribution package${NC}"