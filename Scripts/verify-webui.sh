#!/usr/bin/env bash

# Validate the generated WebUI artifact used by -w/--webui.

set -euo pipefail

WEBUI_PATH="${1:-Resources/webui/index.html.gz}"

if [[ ! -s "$WEBUI_PATH" ]]; then
    echo "[ERROR] Required WebUI artifact is missing or empty: $WEBUI_PATH" >&2
    exit 1
fi

if ! gzip -t "$WEBUI_PATH"; then
    echo "[ERROR] WebUI artifact is not a valid gzip stream: $WEBUI_PATH" >&2
    exit 1
fi

# Do not accept an arbitrary valid gzip file. The decompressed payload must be
# the static HTML entry point that AFM serves at GET /.
if ! gzip -cd "$WEBUI_PATH" | awk '
    BEGIN { IGNORECASE = 1; found = 0 }
    /<!doctype html|<html/ { found = 1 }
    END { exit(found ? 0 : 1) }
'; then
    echo "[ERROR] WebUI gzip does not contain an HTML document: $WEBUI_PATH" >&2
    exit 1
fi

echo "[webui] verified: $WEBUI_PATH"
