#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
SCRATCH_DIR="$REPO_DIR/.build"

if [ "${1:-}" = "--record" ]; then
    export AFM_RECORD_TUI_SNAPSHOTS=1
    shift
fi

if [ "$#" -ne 0 ]; then
    echo "usage: Scripts/test-tui.sh [--record]" >&2
    exit 64
fi

cd "$REPO_DIR"
exec Scripts/swiftpm-reliable.sh test \
    --package-path Tests/TUIHarness \
    --scratch-path "$SCRATCH_DIR"
