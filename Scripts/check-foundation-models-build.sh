#!/usr/bin/env bash
# Reject binaries that were compiled without the real Apple Foundation Models
# provider. Version/help smoke tests alone cannot distinguish those binaries
# from a complete build because MLX remains functional.
set -euo pipefail

BINARY="${1:-}"
if [[ -z "$BINARY" || ! -x "$BINARY" ]]; then
  echo "[foundation-build] Missing executable: ${BINARY:-<not provided>}" >&2
  exit 2
fi

if ! CAPABILITIES_JSON="$("$BINARY" --help-json 2>/dev/null)"; then
  echo "[foundation-build] Could not read build capabilities from $BINARY" >&2
  exit 1
fi

if ! AFM_CAPABILITIES_JSON="$CAPABILITIES_JSON" python3 - <<'PY'
import json
import os

try:
    document = json.loads(os.environ["AFM_CAPABILITIES_JSON"])
except (KeyError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid --help-json output: {error}")

capabilities = document.get("build_capabilities")
if not isinstance(capabilities, dict):
    raise SystemExit("build_capabilities is missing")
if capabilities.get("foundation_models_compiled") is not True:
    raise SystemExit("foundation_models_compiled is not true")
if capabilities.get("minimum_swift_compiler") != "6.4":
    raise SystemExit("minimum_swift_compiler is not 6.4")
PY
then
  echo "[foundation-build] $BINARY does not contain the required Apple Foundation Models provider." >&2
  echo "[foundation-build] Rebuild with Swift 6.4 or newer; refusing to package a degraded MLX-only executable." >&2
  exit 1
fi

echo "[foundation-build] Apple Foundation Models provider is compiled into $BINARY"
