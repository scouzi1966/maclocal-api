#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AFMKIT_ROOT="${AFMKIT_EXAMPLE_PATH:-$ROOT/.build/checkouts/AFMKit}"
CONSUMER_SOURCE="$ROOT/.build/core-only-consumer-source"
CONSUMER_SCRATCH="$ROOT/.build/core-only-consumer-gate"
CONSUMER_LOG="$ROOT/.build/core-only-consumer-gate.log"

OPTIONAL_MODULE_PATTERN='-module-name (AFMKit|AFMKitMLX|AFMKitFoundationModels|AFMKitFoundationModels27|AFMKitServices|AFMOpenAICompat|AFMServer|AFMCLI|AFMXGrammar|Vapor|NIO[A-Za-z]*|CXGrammar|XGrammar|MLX|MLXLLM|MLXVLM|MLXLMCommon|Tokenizers|Hub|HuggingFace)([[:space:]]|$)'

export SWIFTPM_MODULECACHE_OVERRIDE="$ROOT/.build/core-only-swiftpm-module-cache"
export CLANG_MODULE_CACHE_PATH="$ROOT/.build/core-only-clang-module-cache"
export XDG_CACHE_HOME="$ROOT/.build/core-only-xdg-cache"

if grep -Eq '\.package\([^)]*branch:' "$ROOT/Package.swift"; then
    echo "Package.swift contains an unpinned branch dependency." >&2
    exit 1
fi

if [[ ! -f "$AFMKIT_ROOT/Package.swift" ]]; then
    "$ROOT/Scripts/resolve-release-dependencies.sh"
fi

expected_revision="$(python3 - "$ROOT/Package.resolved" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1]))
print(next(pin for pin in lock["pins"] if pin["identity"] == "afmkit")["state"]["revision"])
PY
)"
actual_revision="$(git -C "$AFMKIT_ROOT" rev-parse HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
    echo "AFMKit consumer fixture is not at the locked revision." >&2
    echo "expected=$expected_revision actual=$actual_revision" >&2
    exit 1
fi

lock_hash="$(shasum -a 256 "$ROOT/Package.resolved" | cut -d' ' -f1)"
rm -rf "$CONSUMER_SOURCE" "$CONSUMER_SCRATCH"
mkdir -p "$CONSUMER_SOURCE"
cp "$ROOT/Examples/AFMKitCoreOnlyConsumer/Package.swift" "$CONSUMER_SOURCE/"
cp -R "$ROOT/Examples/AFMKitCoreOnlyConsumer/Sources" "$CONSUMER_SOURCE/"

AFMKIT_EXAMPLE_PATH="$AFMKIT_ROOT" swift build \
    --package-path "$CONSUMER_SOURCE" \
    --scratch-path "$CONSUMER_SCRATCH" \
    -v 2>&1 | tee "$CONSUMER_LOG"

if grep -E -- "$OPTIONAL_MODULE_PATTERN" "$CONSUMER_LOG"; then
    echo "AFMKitCore-only consumer unexpectedly compiled an optional implementation target." >&2
    exit 1
fi

if ! grep -E -- '-module-name AFMKitCoreOnlyConsumer' "$CONSUMER_LOG" >/dev/null; then
    echo "AFMKitCore-only consumer was not compiled by the graph gate." >&2
    exit 1
fi

if ! grep -E -- '-module-name AFMKitCore' "$CONSUMER_LOG" >/dev/null; then
    echo "The independent AFMKitCore product was not compiled by the consumer gate." >&2
    exit 1
fi

if [[ "$(shasum -a 256 "$ROOT/Package.resolved" | cut -d' ' -f1)" != "$lock_hash" ]]; then
    echo "Independent consumer validation changed the tracked release lock." >&2
    exit 1
fi

echo "Independent AFMKitCore consumer resolved without optional implementation targets."
