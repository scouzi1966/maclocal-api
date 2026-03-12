#!/bin/bash

# Validate Swift version requirements
validate_swift_version() {
    echo "🔍 Validating Swift version requirements..."
    
    if ! command -v swift &> /dev/null; then
        echo "❌ Error: Swift compiler not found. Please install Xcode or Swift toolchain."
        exit 1
    fi
    
    # Get Swift version info
    SWIFT_VERSION_OUTPUT=$(swift --version 2>&1)
    
    # Extract version numbers
    SWIFT_VERSION=$(echo "$SWIFT_VERSION_OUTPUT" | grep -E "Apple Swift version" | sed -E 's/.*Apple Swift version ([0-9]+\.[0-9]+).*/\1/')
    DRIVER_VERSION=$(echo "$SWIFT_VERSION_OUTPUT" | grep -E "swift-driver version:" | sed -E 's/swift-driver version: ([0-9.]+).*/\1/')
    TARGET_OS=$(echo "$SWIFT_VERSION_OUTPUT" | grep -E "Target:" | sed -E 's/.*Target: ([a-z0-9]+-[a-z]+-[a-z]+)([0-9]+\.[0-9]+).*/\2/')
    
    echo "Current Swift configuration:"
    echo "  Swift version: $SWIFT_VERSION"
    echo "  Driver version: $DRIVER_VERSION" 
    echo "  Target OS: $TARGET_OS"
    
    # Validate minimum requirements
    REQUIRED_SWIFT="6.2"
    REQUIRED_DRIVER="1.127.11.2"
    REQUIRED_OS="26.0"
    
    # Check Swift version (6.2+)
    if [[ "$(printf '%s\n%s\n' "$REQUIRED_SWIFT" "$SWIFT_VERSION" | sort -V | tail -n1)" != "$SWIFT_VERSION" ]]; then
        echo "❌ Error: Swift version $SWIFT_VERSION is below minimum required $REQUIRED_SWIFT"
        echo "Required: swift-driver version: $REQUIRED_DRIVER+ Apple Swift version $REQUIRED_SWIFT+ (swiftlang-6.2.0.16.14 clang-1700.3.16.4)"
        echo "Required: Target: arm64-apple-macosx$REQUIRED_OS+"
        exit 1
    fi
    
    # Check driver version (1.127.11.2+)
    if [[ "$(printf '%s\n%s\n' "$REQUIRED_DRIVER" "$DRIVER_VERSION" | sort -V | tail -n1)" != "$DRIVER_VERSION" ]]; then
        echo "❌ Error: swift-driver version $DRIVER_VERSION is below minimum required $REQUIRED_DRIVER"
        echo "Required: swift-driver version: $REQUIRED_DRIVER+ Apple Swift version $REQUIRED_SWIFT+ (swiftlang-6.2.0.16.14 clang-1700.3.16.4)"
        echo "Required: Target: arm64-apple-macosx$REQUIRED_OS+"
        exit 1
    fi
    
    # Check target OS version (26.0+)
    if [[ "$(printf '%s\n%s\n' "$REQUIRED_OS" "$TARGET_OS" | sort -V | tail -n1)" != "$TARGET_OS" ]]; then
        echo "❌ Error: Target OS version $TARGET_OS is below minimum required $REQUIRED_OS"
        echo "Required: swift-driver version: $REQUIRED_DRIVER+ Apple Swift version $REQUIRED_SWIFT+ (swiftlang-6.2.0.16.14 clang-1700.3.16.4)"
        echo "Required: Target: arm64-apple-macosx$REQUIRED_OS+"
        exit 1
    fi
    
    echo "✅ Swift version validation passed"
    echo ""
}

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

# Use provided version or auto-generate from git
if [ -n "$1" ]; then
    VERSION="$1"
else
    VERSION=$(get_version | tr -d '\n')
fi

ARCH="arm64"

echo "Building afm (Apple Foundation Models API) v$VERSION for $ARCH..."

# Validate Swift version before building
validate_swift_version

# Clean previous builds
rm -rf release-v$VERSION
rm -f afm-v$VERSION-$ARCH.tar.gz

# Generate BuildInfo.swift with version
echo "Generating version file..."
cat > Sources/MacLocalAPI/BuildInfo.swift << EOF
// BuildInfo.swift
// Auto-generated build information - DO NOT EDIT MANUALLY

struct BuildInfo {
    static let version: String? = "$VERSION"
}
EOF

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