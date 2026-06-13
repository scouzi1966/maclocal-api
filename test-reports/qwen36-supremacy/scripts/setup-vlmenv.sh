#!/bin/bash
# One-time: create the isolated mlx-vlm env used by run-mlx_vlm.sh / run-reprobe.
# Default location is /tmp/bench/vlmenv (ephemeral). Set VLM_ENV_DIR for a persistent path,
# then export VLM_PYTHON=$VLM_ENV_DIR/bin/python before running the benchmarks.
set -euo pipefail
VLM_ENV_DIR="${VLM_ENV_DIR:-/tmp/bench/vlmenv}"
uv venv --python 3.12 "$VLM_ENV_DIR"
uv pip install --python "$VLM_ENV_DIR/bin/python" mlx-vlm
echo "mlx-vlm env ready: $VLM_ENV_DIR"
"$VLM_ENV_DIR/bin/python" -c "import mlx_vlm; print('mlx_vlm', mlx_vlm.__version__)"
echo "export VLM_PYTHON=$VLM_ENV_DIR/bin/python   # then run the benchmarks"
