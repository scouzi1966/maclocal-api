#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_MANIFEST="$ROOT_DIR/vendor/mlx-swift-lm/Package.swift"

if [[ ! -f "$VENDOR_MANIFEST" ]]; then
    echo "mlx-swift-lm vendor is not initialized; URL fallback is expected."
    exit 0
fi

PACKAGE_JSON="$(cd "$ROOT_DIR" && swift package dump-package)"
if ! grep -Fq '"path" : "vendor/mlx-swift-lm"' <<<"$PACKAGE_JSON" &&
   ! grep -Fq "\"path\" : \"$ROOT_DIR/vendor/mlx-swift-lm\"" <<<"$PACKAGE_JSON"; then
    echo "error: initialized vendor/mlx-swift-lm is not the resolved package source" >&2
    exit 1
fi

echo "mlx-swift-lm source selection: vendor/mlx-swift-lm"
