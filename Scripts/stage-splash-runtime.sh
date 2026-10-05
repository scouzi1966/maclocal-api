#!/usr/bin/env bash
# Application packaging delegates the release identity and staging to AFMKit.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AFMKIT_ROOT="${MACLOCAL_AFMKIT_PATH:-$ROOT_DIR/.build/checkouts/AFMKit}"
DESTINATION="${1:?Usage: stage-splash-runtime.sh <binary-directory>}"
if [[ ! -f "$AFMKIT_ROOT/Scripts/stage-splash-runtime.py" ]]; then
    echo "Pinned AFMKit Splash staging tool missing; resolve the AFMKit dependency first." >&2
    exit 1
fi
exec python3 "$AFMKIT_ROOT/Scripts/stage-splash-runtime.py" \
    --destination "$DESTINATION/splash-runtime" --cache "$ROOT_DIR/.build-splash-downloads"
