# AFM - Apple Foundation Models API
# Makefile for building and distributing the portable CLI

.PHONY: build clean install uninstall portable dist test help submodules submodule-status webui verify-webui build-with-webui patch patch-check patch-revert verify-afmkit-consumer-boundary

PATCH_SH := Scripts/apply-mlx-patches.sh

# Default target
all: build

# Legacy dependency maintenance only. Normal builds consume immutable AFMKit
# and AFM-compatible MLX packages and must not mutate vendor submodules.
patch:
	@echo "Applying the legacy mlx-swift-lm patch stack..."
	@bash $(PATCH_SH)

patch-check:
	@bash $(PATCH_SH) --check

patch-revert:
	@bash $(PATCH_SH) --revert

verify-afmkit-consumer-boundary:
	@Scripts/check-afmkit-consumer-boundary.sh

# Build the release binary (portable by default)
build: verify-afmkit-consumer-boundary
	@echo "🔨 Building AFM..."
	@Scripts/swiftpm-reliable.sh build -c release \
		--product afm \
		-Xswiftc -disable-upcoming-feature \
		-Xswiftc MemberImportVisibility
	@AFM_BIN="$$(Scripts/find-afm-binary.sh release)"; \
		strip "$$AFM_BIN"; \
		echo "✅ Build complete: $$AFM_BIN"; \
		echo "📊 Size: $$(ls -lh "$$AFM_BIN" | awk '{print $$5}')"

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
	@Scripts/verify-webui.sh Resources/webui/index.html.gz
	@echo "✅ WebUI built: Resources/webui/index.html.gz"

verify-webui:
	@Scripts/verify-webui.sh Resources/webui/index.html.gz

# Build with webui included
build-with-webui: webui build
	@echo "✅ Build with webui complete"

# Clean build artifacts without modifying dependency worktrees.
clean:
	@echo "🧹 Cleaning build artifacts..."
	@swift package clean
	@rm -rf .build
	@rm -f dist/*.tar.gz
	@echo "✅ Clean complete"

# Install to system (requires sudo)
install: verify-afmkit-consumer-boundary
	@INSTALL_PREFIX="$(or $(INSTALL_PREFIX),/usr/local)" \
		./build.sh --stable --yes --no-clean --skip-webui --install

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
	@AFM_BIN="$$(Scripts/find-afm-binary.sh release)"; \
		"$$AFM_BIN" --help > /dev/null && echo "✅ Binary test passed" || echo "❌ Binary test failed"
	@AFM_BIN="$$(Scripts/find-afm-binary.sh release)"; TEST_BIN=".build/afm-portability-test-$$$$"; \
		cp "$$AFM_BIN" "$$TEST_BIN" && \
		"$$TEST_BIN" --version > /dev/null 2>&1 && \
		echo "✅ Portability test passed" || echo "⚠️  Portability test failed"; \
		rm -f "$$TEST_BIN"

# Development build (debug)
debug: verify-afmkit-consumer-boundary
	@echo "🐛 Building debug version..."
	@Scripts/swiftpm-reliable.sh build
	@echo "✅ Debug build complete: $$(Scripts/find-afm-binary.sh debug)"

# Run the server (development)
run: debug
	@echo "🚀 Starting AFM server..."
	@"$$(Scripts/find-afm-binary.sh debug)" --port 9999

# Show help
help:
	@echo "AFM - Apple Foundation Models API"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  build           - Build release binary from immutable dependencies"
	@echo "  portable        - Build with enhanced portability"
	@echo "  clean           - Clean build artifacts without changing vendor worktrees"
	@echo "  patch           - Apply legacy vendor patches explicitly"
	@echo "  patch-check     - Verify vendor patch status"
	@echo "  patch-revert    - Revert explicitly applied vendor patches"
	@echo "  verify-afmkit-consumer-boundary - Check immutable dependency and packaging ownership"
	@echo "  install         - Install to /usr/local/bin (requires sudo)"
	@echo "  uninstall       - Remove from /usr/local/bin"
	@echo "  dist            - Create distribution package"
	@echo "  test            - Test the binary and portability"
	@echo "  debug           - Build debug version"
	@echo "  run             - Build and run debug server"
	@echo "  submodules      - Initialize git submodules"
	@echo "  webui           - Build webui from llama.cpp (requires Node.js)"
	@echo "  verify-webui    - Validate the packaged WebUI gzip and HTML payload"
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
	@echo "Output: $$(Scripts/find-afm-binary.sh release 2>/dev/null || echo '.build/<toolchain release path>/afm')"
