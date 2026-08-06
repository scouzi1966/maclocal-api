#!/usr/bin/env bash

# Applies AFM's DeepSeek V4 fused MXFP4 MoE primitive to the resolved
# mlx-swift checkout. SwiftPM checkouts are disposable, so the authoritative
# patches live in this repository and are reapplied after every resolve.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKOUT="${MLX_SWIFT_CHECKOUT:-$ROOT_DIR/.build/checkouts/mlx-swift}"
PATCH_DIR="$ROOT_DIR/Scripts/patches/mlx-swift-deepseek-v4"
MODE="${1:-apply}"

if [[ "$MODE" != "apply" && "$MODE" != "--check" ]]; then
    echo "usage: $0 [--check]" >&2
    exit 2
fi

apply_one() {
    local label="$1"
    local repository="$2"
    local patch="$3"

    if [[ ! -d "$repository/.git" && ! -f "$repository/.git" ]]; then
        echo "[mlx-deepseek-v4] Missing $label repository: $repository" >&2
        exit 1
    fi
    if [[ ! -s "$patch" ]]; then
        echo "[mlx-deepseek-v4] Missing $label patch: $patch" >&2
        exit 1
    fi

    if git -C "$repository" apply --reverse --check "$patch" >/dev/null 2>&1; then
        echo "[mlx-deepseek-v4] $label patch is applied." >&2
        return 0
    fi

    if ! git -C "$repository" apply --check "$patch" >/dev/null 2>&1; then
        echo "[mlx-deepseek-v4] $label source does not match either the clean or patched state." >&2
        echo "[mlx-deepseek-v4] Refusing a partial patch; resolve the mlx-swift version drift first." >&2
        exit 1
    fi

    if [[ "$MODE" == "--check" ]]; then
        echo "[mlx-deepseek-v4] $label patch is not applied." >&2
        exit 1
    fi

    git -C "$repository" apply "$patch"
    echo "[mlx-deepseek-v4] Applied $label patch." >&2
}

apply_one \
    "mlx-swift" \
    "$CHECKOUT" \
    "$PATCH_DIR/mlx-swift.patch"
apply_one \
    "MLX core" \
    "$CHECKOUT/Source/Cmlx/mlx" \
    "$PATCH_DIR/mlx-core.patch"
apply_one \
    "MLX C API" \
    "$CHECKOUT/Source/Cmlx/mlx-c" \
    "$PATCH_DIR/mlx-c.patch"
