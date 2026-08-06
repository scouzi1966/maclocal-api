#!/usr/bin/env bash

# Teach the pinned mlx-swift safetensors reader that E8M0 scale tensors are
# byte payloads. DeepSeek V4's official checkpoint stores MXFP scales with the
# F8_E8M0 metadata tag, while MLX consumes the identical bytes as uint8.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKOUT="${MLX_SWIFT_CHECKOUT:-$ROOT_DIR/.build/checkouts/mlx-swift}"
TARGET="$CHECKOUT/Source/Cmlx/mlx/mlx/io/safetensors.cpp"
MARKER="AFM_PATCH_F8_E8M0_SAFETENSORS"

if [[ ! -f "$TARGET" ]]; then
    echo "[mlx-fp8-loader] mlx-swift checkout is missing: $TARGET" >&2
    exit 1
fi

if grep -Fq "$MARKER" "$TARGET"; then
    echo "[mlx-fp8-loader] Official FP8 loader support is already applied." >&2
    exit 0
fi

chmod u+w "$TARGET"

python3 - "$TARGET" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()

define_needle = '#define ST_F8_E4M3 "F8_E4M3"\n'
define_replacement = (
    define_needle
    + '#define ST_F8_E8M0 "F8_E8M0" // AFM_PATCH_F8_E8M0_SAFETENSORS\n'
)
decode_needle = '} else if (str == ST_F8_E4M3) {\n    return uint8;\n'
decode_replacement = (
    '} else if (str == ST_F8_E4M3 || str == ST_F8_E8M0) {\n'
    '    // Both formats are packed byte payloads. Their interpretation is\n'
    '    // supplied by the model quantization metadata at matmul time.\n'
    '    return uint8;\n'
)

if define_needle not in source or decode_needle not in source:
    raise SystemExit("mlx-swift safetensors source no longer matches the supported patch context")

source = source.replace(define_needle, define_replacement, 1)
source = source.replace(decode_needle, decode_replacement, 1)
path.write_text(source)
PY

echo "[mlx-fp8-loader] Added official F8_E8M0 safetensors support." >&2
