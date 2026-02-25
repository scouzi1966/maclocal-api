#!/bin/bash
# List all cached models in MACAFM_MLX_MODEL_CACHE as org/model
#
# Usage:
#   ./Scripts/list-models.sh
#   MACAFM_MLX_MODEL_CACHE=/path/to/cache ./Scripts/list-models.sh

CACHE_DIR="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"

echo "MACAFM_MLX_MODEL_CACHE=$CACHE_DIR"
echo ""

if [ ! -d "$CACHE_DIR" ]; then
  echo "Error: Cache directory not found: $CACHE_DIR" >&2
  echo "Set MACAFM_MLX_MODEL_CACHE to your cache path." >&2
  exit 1
fi

# Skip non-model directories
SKIP="hub|gguf|xet"

models=()
for org_dir in "$CACHE_DIR"/*/; do
  org=$(basename "$org_dir")
  echo "$org" | grep -qE "^($SKIP)$" && continue
  for model_dir in "$org_dir"*/; do
    [ -d "$model_dir" ] || continue
    model=$(basename "$model_dir")
    models+=("$org/$model")
  done
done

printf '%s\n' "${models[@]}" | sort

echo ""
echo "${#models[@]} models found in $CACHE_DIR"
