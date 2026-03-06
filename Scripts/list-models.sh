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
SKIP="hub|gguf|xet|models"

models=()
sizes=()
for org_dir in "$CACHE_DIR"/*/; do
  org=$(basename "$org_dir")
  echo "$org" | grep -qE "^($SKIP)$|^models--" && continue
  for model_dir in "$org_dir"*/; do
    [ -d "$model_dir" ] || continue
    model=$(basename "$model_dir")
    # Get total size in bytes, convert to GB
    bytes=$(du -sk "$model_dir" 2>/dev/null | awk '{print $1}')
    gb=$(awk "BEGIN {printf \"%.1f\", $bytes / 1048576}")
    models+=("$org/$model")
    sizes+=("$gb")
  done
done

# Sort by name and display with size
paste <(printf '%s\n' "${models[@]}") <(printf '%s\n' "${sizes[@]}") | sort -t$'\t' -k1 | while IFS=$'\t' read -r name size; do
  printf "%-60s %6s GB\n" "$name" "$size"
done

echo ""
echo "${#models[@]} models found in $CACHE_DIR"
