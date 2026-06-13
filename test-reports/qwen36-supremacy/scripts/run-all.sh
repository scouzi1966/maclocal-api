#!/bin/bash
# Run every engine sequentially (one on the GPU at a time), then aggregate the dataset.
# Each runner kills the previous engine first. A failing engine (e.g. ollama) is logged, not fatal.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENGINES="${ENGINES:-afm mlx_vlm llamacpp lmstudio omlx ollama rapidmlx}"
for e in $ENGINES; do
  echo "=================== $e ==================="
  bash "./run-$e.sh" || echo ">>> $e exited non-zero (see results/$e-*.log)"
done
echo "=================== aggregate ==================="
python3 ./aggregate.py
