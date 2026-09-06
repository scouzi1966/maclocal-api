#!/usr/bin/env bash

# Validate the generated llama.cpp WebUI directory used by -w/--webui.

set -euo pipefail

WEBUI_DIR="${1:-Resources/webui}"

if [[ ! -d "$WEBUI_DIR" ]]; then
    echo "[ERROR] Required WebUI directory is missing: $WEBUI_DIR" >&2
    exit 1
fi

required=(
    "index.html"
    "manifest.webmanifest"
    "sw.js"
    "build.json"
    "_app/version.json"
)

for relative in "${required[@]}"; do
    if [[ ! -s "$WEBUI_DIR/$relative" ]]; then
        echo "[ERROR] Required WebUI payload is missing or empty: $relative" >&2
        exit 1
    fi
done

javascript_bundle_count="$(find "$WEBUI_DIR/_app/immutable" -type f -name 'bundle*.js' | wc -l | tr -d '[:space:]')"
if [[ "$javascript_bundle_count" != "1" ]]; then
    echo "[ERROR] Expected one WebUI JavaScript bundle, found $javascript_bundle_count" >&2
    exit 1
fi
stylesheet_bundle_count="$(find "$WEBUI_DIR/_app/immutable" -type f -name 'bundle*.css' | wc -l | tr -d '[:space:]')"
if [[ "$stylesheet_bundle_count" != "1" ]]; then
    echo "[ERROR] Expected one WebUI stylesheet bundle, found $stylesheet_bundle_count" >&2
    exit 1
fi
workbox_runtime_count="$(find "$WEBUI_DIR" -maxdepth 1 -type f -name 'workbox-*.js' | wc -l | tr -d '[:space:]')"
if [[ "$workbox_runtime_count" != "1" ]]; then
    echo "[ERROR] Expected one WebUI service-worker runtime, found $workbox_runtime_count" >&2
    exit 1
fi
if [[ -n "$(find "$WEBUI_DIR" -type l -print -quit)" ]]; then
    echo "[ERROR] WebUI payload must not contain symbolic links" >&2
    exit 1
fi

if ! awk '
    BEGIN { IGNORECASE = 1; found = 0 }
    /<!doctype html|<html/ { found = 1 }
    END { exit(found ? 0 : 1) }
' "$WEBUI_DIR/index.html"; then
    echo "[ERROR] WebUI index.html is not an HTML document" >&2
    exit 1
fi

echo "[webui] verified: $WEBUI_DIR"
