#!/usr/bin/env bash

# Apply AFM-owned adaptations to the canonical, pinned DwarfStar submodule.
# Usage: Scripts/apply-ds4-patches.sh [--check|--revert]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS4_DIR="$ROOT_DIR/vendor/ds4"
PATCH_DIR="$ROOT_DIR/Scripts/patches/ds4"
EXPECTED_REVISION="b7e9f0091139999b6c070a57590c447c5741da5c"
MODE="${1:-apply}"

case "$MODE" in
    apply|--check|--revert) ;;
    *)
        echo "Usage: $0 [--check|--revert]" >&2
        exit 2
        ;;
esac

if [[ ! -f "$DS4_DIR/ds4.c" || ! -f "$DS4_DIR/ds4.h" ]]; then
    echo "[ds4-patches] DwarfStar submodule is not initialized: $DS4_DIR" >&2
    echo "[ds4-patches] Run: git submodule update --init vendor/ds4" >&2
    exit 1
fi

actual_revision="$(git -C "$DS4_DIR" rev-parse HEAD)"
if [[ "$actual_revision" != "$EXPECTED_REVISION" ]]; then
    echo "[ds4-patches] Unsupported DwarfStar revision: $actual_revision" >&2
    echo "[ds4-patches] Expected pinned upstream revision: $EXPECTED_REVISION" >&2
    exit 1
fi

patches=("$PATCH_DIR"/*.patch)
if [[ ! -f "${patches[0]}" ]]; then
    echo "[ds4-patches] No DwarfStar patches found in $PATCH_DIR" >&2
    exit 1
fi

is_applied() {
    git -C "$DS4_DIR" apply --reverse --check "$1" >/dev/null 2>&1
}

can_apply() {
    git -C "$DS4_DIR" apply --check "$1" >/dev/null 2>&1
}

case "$MODE" in
    apply)
        for patch in "${patches[@]}"; do
            if is_applied "$patch"; then
                echo "[ds4-patches] Already applied: $(basename "$patch")" >&2
            elif can_apply "$patch"; then
                git -C "$DS4_DIR" apply "$patch"
                echo "[ds4-patches] Applied: $(basename "$patch")" >&2
            else
                echo "[ds4-patches] Patch does not match the pinned source: $patch" >&2
                exit 1
            fi
        done
        ;;
    --check)
        for patch in "${patches[@]}"; do
            if ! is_applied "$patch"; then
                echo "[ds4-patches] Missing or partially applied: $(basename "$patch")" >&2
                exit 1
            fi
        done
        echo "[ds4-patches] All DwarfStar patches are applied." >&2
        ;;
    --revert)
        for ((index=${#patches[@]} - 1; index >= 0; index--)); do
            patch="${patches[$index]}"
            if is_applied "$patch"; then
                git -C "$DS4_DIR" apply --reverse "$patch"
                echo "[ds4-patches] Reverted: $(basename "$patch")" >&2
            elif can_apply "$patch"; then
                echo "[ds4-patches] Already reverted: $(basename "$patch")" >&2
            else
                echo "[ds4-patches] Cannot safely revert: $patch" >&2
                exit 1
            fi
        done
        ;;
esac
