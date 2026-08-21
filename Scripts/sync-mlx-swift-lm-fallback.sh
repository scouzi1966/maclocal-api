#!/usr/bin/env bash
# Synchronize the URL-consumer mlx-swift-lm snapshot from the patched vendor.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${MLX_LM_DIR:-$REPO_ROOT/vendor/mlx-swift-lm}"
DESTINATION="$REPO_ROOT/Dependencies/mlx-swift-lm"
EXPECTED_REVISION="$(tr -d '[:space:]' < "$REPO_ROOT/Scripts/mlx-swift-lm-upstream-revision")"

if [ ! -f "$SOURCE/Package.swift" ]; then
  echo "ERROR: mlx-swift-lm source is not initialized: $SOURCE" >&2
  exit 1
fi

ACTUAL_REVISION="$(git -C "$SOURCE" rev-parse HEAD)"
if [ "$ACTUAL_REVISION" != "$EXPECTED_REVISION" ]; then
  echo "ERROR: expected vanilla mlx-swift-lm $EXPECTED_REVISION, found $ACTUAL_REVISION" >&2
  exit 1
fi

MLX_LM_DIR="$SOURCE" "$REPO_ROOT/Scripts/apply-mlx-patches.sh" --check

mkdir -p "$DESTINATION"
rsync -a --delete \
  --exclude='.git' \
  --exclude='.build' \
  "$SOURCE/" "$DESTINATION/"

while IFS= read -r -d '' text_file; do
  perl -0pi -e 's/[ \t]+(?=\n)//g; s/\n+\z/\n/' "$text_file"
done < <(find "$DESTINATION" -type f \( -name '*.md' -o -name '.gitignore' \) -print0)

cat > "$DESTINATION/AFM-PATCH-SOURCE.md" <<EOF
# Generated mlx-swift-lm dependency

This directory is generated from upstream mlx-swift-lm revision
\`$EXPECTED_REVISION\` with \`Scripts/patches/\` applied.

Do not edit it directly. Run \`Scripts/sync-mlx-swift-lm-fallback.sh\` after
changing the upstream revision or repository-owned patch set.
EOF

echo "Synchronized $DESTINATION from patched upstream $EXPECTED_REVISION"
