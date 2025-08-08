#!/bin/bash

VERSION=${1:-"0.2.0"}
ARCH="arm64"

echo "Building afm (Apple Foundation Models API) v$VERSION for $ARCH..."

# Clean previous builds
rm -rf release-v$VERSION
rm -f maclocal-api-v$VERSION-$ARCH.tar.gz

# Build release binary
echo "Building release binary..."
swift build --configuration release

# Create release directory
mkdir -p release-v$VERSION

# Copy binary with new name
cp .build/arm64-apple-macosx/release/afm release-v$VERSION/

# Copy scripts
cp test-streaming.sh test-metrics.sh release-v$VERSION/

# Copy documentation
cp README.md LICENSE release-v$VERSION/ 2>/dev/null || true

# Create install guide
cat > release-v$VERSION/INSTALL.md << EOF
# afm (Apple Foundation Models API) v$VERSION - Installation Guide

## Quick Start
1. Extract: \`tar -xzf afm-v$VERSION-$ARCH.tar.gz\`
2. Run: \`chmod +x afm && ./afm\`
3. Test: \`./test-streaming.sh\`

## Requirements
- macOS 26+ with Apple Intelligence enabled
- Apple Silicon (M1, M2, M3, M4)

For more info: https://github.com/scouzi1966/maclocal-api
EOF

# Create archive
tar -czf afm-v$VERSION-$ARCH.tar.gz -C release-v$VERSION .

echo "✅ Release package created: afm-v$VERSION-$ARCH.tar.gz"
echo "📦 Contents:"
ls -la release-v$VERSION/

echo ""
echo "To upload to GitHub release:"
echo "gh release upload v$VERSION afm-v$VERSION-$ARCH.tar.gz"