#!/bin/bash

# MacLocalAPI Installation Script
# Automates the build and installation process

set -e

echo "🚀 MacLocalAPI v0.1 Installation Script"
echo "======================================="
echo

# Check macOS version
if [[ $(sw_vers -productVersion | cut -d '.' -f1) -lt 26 ]]; then
    echo "❌ Error: macOS 26 (Tahoe) or later is required"
    echo "   Current version: $(sw_vers -productVersion)"
    exit 1
fi

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ Error: Apple Silicon Mac required"
    echo "   Current architecture: $(uname -m)"
    exit 1
fi

echo "✅ System requirements met"
echo

# Build the project
echo "🔨 Building MacLocalAPI..."
swift build -c release

if [[ $? -eq 0 ]]; then
    echo "✅ Build successful!"
    echo
    echo "📦 Installation complete!"
    echo
    echo "🚀 To start the server:"
    echo "   ./.build/release/MacLocalAPI --port 9999"
    echo
    echo "📖 For usage examples, see:"
    echo "   https://github.com/scouzi1966/maclocal-api#-usage-examples"
    echo
    echo "❗ Note: Ensure Apple Intelligence is enabled in System Settings"
else
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi