# AFM - Apple Foundation Models API
# Makefile for building and distributing the portable CLI

.PHONY: build clean install uninstall portable dist test help submodules submodule-status webui build-with-webui

# Default target
all: build

# Build the release binary (portable by default)
build:
	@echo "🔨 Building AFM..."
	@swift build -c release \
		--product afm \
		-Xswiftc -O \
		-Xswiftc -whole-module-optimization \
		-Xswiftc -cross-module-optimization
	@strip .build/release/afm
	@echo "✅ Build complete: .build/release/afm"
	@echo "📊 Size: $$(ls -lh .build/release/afm | awk '{print $$5}')"

# Build with enhanced portability optimizations
portable:
	@./build-portable.sh

# Initialize git submodules (pinned to specific commit for reproducibility)
# NOTE: llama.cpp is pinned to a specific commit - do not use --remote flag
submodules:
	@echo "📦 Initializing git submodules (pinned version)..."
	@git submodule update --init
	@echo "✅ Submodules initialized (llama.cpp @ $$(cd vendor/llama.cpp && git rev-parse --short HEAD))"

# Show pinned submodule versions
submodule-status:
	@echo "📌 Pinned submodule versions:"
	@git submodule status

# Build the webui from llama.cpp
webui: submodules
	@echo "🌐 Building webui..."
	@if [ ! -d "vendor/llama.cpp/tools/server/webui" ]; then \
		echo "❌ Error: webui source not found. Run 'make submodules' first."; \
		exit 1; \
	fi
	@cd vendor/llama.cpp/tools/server/webui && npm install && npm run build
	@mkdir -p Resources/webui
	@cp vendor/llama.cpp/tools/server/public/index.html.gz Resources/webui/
	@echo "✅ WebUI built: Resources/webui/index.html.gz"

# Build with webui included
build-with-webui: webui build
	@echo "✅ Build with webui complete"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@swift package clean
	@rm -rf .build
	@rm -f dist/*.tar.gz
	@echo "✅ Clean complete"

# Install to system (requires sudo)
install: build
	@echo "📦 Installing AFM to /usr/local/bin..."
	@sudo cp .build/release/afm /usr/local/bin/afm
	@sudo chmod +x /usr/local/bin/afm
	@echo "✅ AFM installed to /usr/local/bin/afm"

# Uninstall from system
uninstall:
	@echo "🗑️  Uninstalling AFM..."
	@sudo rm -f /usr/local/bin/afm
	@echo "✅ AFM uninstalled"

# Create distribution package
dist: portable
	@./create-distribution.sh

# Test the binary
test: build
	@echo "🧪 Testing AFM binary..."
	@./.build/release/afm --help > /dev/null && echo "✅ Binary test passed" || echo "❌ Binary test failed"
	@cp .build/release/afm /tmp/afm-test-$$$$ && \
		/tmp/afm-test-$$$$ --version > /dev/null 2>&1 && \
		echo "✅ Portability test passed" || echo "⚠️  Portability test failed"; \
		rm -f /tmp/afm-test-$$$$

# Development build (debug)
debug:
	@echo "🐛 Building debug version..."
	@swift build
	@echo "✅ Debug build complete: .build/debug/afm"

# Run the server (development)
run: debug
	@echo "🚀 Starting AFM server..."
	@./.build/debug/afm --port 9999

# Show help
help:
	@echo "AFM - Apple Foundation Models API"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  build           - Build release binary (default, portable)"
	@echo "  portable        - Build with enhanced portability"
	@echo "  clean           - Clean build artifacts"
	@echo "  install         - Install to /usr/local/bin (requires sudo)"
	@echo "  uninstall       - Remove from /usr/local/bin"
	@echo "  dist            - Create distribution package"
	@echo "  test            - Test the binary and portability"
	@echo "  debug           - Build debug version"
	@echo "  run             - Build and run debug server"
	@echo "  submodules      - Initialize git submodules"
	@echo "  webui           - Build webui from llama.cpp (requires Node.js)"
	@echo "  build-with-webui - Build with webui included"
	@echo "  help            - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make build              # Build portable executable"
	@echo "  make build-with-webui   # Build with webui support"
	@echo "  make install            # Build and install to system"
	@echo "  make dist               # Create distribution package"
	@echo "  make test               # Test binary works"
	@echo ""
	@echo "Output: .build/release/afm (portable executable)"