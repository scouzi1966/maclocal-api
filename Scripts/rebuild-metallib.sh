#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MACLOCAL_AFMKIT_PATH:-}" ]]; then
  cat >&2 <<'ERROR'
AFMKit owns default.metallib and its rebuild workflow.

Set MACLOCAL_AFMKIT_PATH to a writable AFMKit checkout, then rerun this
command. Normal maclocal-api builds consume AFMKit's immutable resource and do
not rebuild or patch resolved package checkouts.
ERROR
  exit 2
fi

AFMKIT_REBUILD="$MACLOCAL_AFMKIT_PATH/Scripts/rebuild-mlx-metallib.sh"
if [[ ! -x "$AFMKIT_REBUILD" ]]; then
  echo "AFMKit rebuild tool is missing or not executable: $AFMKIT_REBUILD" >&2
  exit 1
fi

exec "$AFMKIT_REBUILD" "$@"
