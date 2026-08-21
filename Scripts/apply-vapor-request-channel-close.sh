#!/usr/bin/env bash

# Exposes Vapor's underlying NIO channel-close future on server-backed
# requests. SwiftPM checkouts are disposable, so the authoritative patch lives
# in this repository and is reapplied after every resolve.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKOUT="${VAPOR_CHECKOUT:-$ROOT_DIR/.build/checkouts/vapor}"
PATCH="$ROOT_DIR/Scripts/patches/vapor-request-channel-close/vapor.patch"
MODE="${1:-apply}"

if [[ "$MODE" != "apply" && "$MODE" != "--check" ]]; then
    echo "usage: $0 [--check]" >&2
    exit 2
fi

if [[ ! -d "$CHECKOUT/.git" && ! -f "$CHECKOUT/.git" ]]; then
    echo "[vapor-channel-close] Missing Vapor repository: $CHECKOUT" >&2
    exit 1
fi
if [[ ! -s "$PATCH" ]]; then
    echo "[vapor-channel-close] Missing patch: $PATCH" >&2
    exit 1
fi

if git -C "$CHECKOUT" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "[vapor-channel-close] Patch is applied." >&2
    exit 0
fi

if ! git -C "$CHECKOUT" apply --check "$PATCH" >/dev/null 2>&1; then
    echo "[vapor-channel-close] Source does not match either the clean or patched state." >&2
    echo "[vapor-channel-close] Refusing a partial patch; resolve Vapor version drift first." >&2
    exit 1
fi

if [[ "$MODE" == "--check" ]]; then
    echo "[vapor-channel-close] Patch is not applied." >&2
    exit 1
fi

git -C "$CHECKOUT" apply "$PATCH"
echo "[vapor-channel-close] Applied patch." >&2
