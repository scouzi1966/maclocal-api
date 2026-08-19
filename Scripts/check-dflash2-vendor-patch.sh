#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor/mlx-swift-lm"

if [ ! -f "$VENDOR/Package.swift" ]; then
  echo "error: initialize vendor/mlx-swift-lm before checking DFlash2 patches" >&2
  exit 1
fi

"$ROOT/Scripts/apply-mlx-patches.sh" --check

test -f "$VENDOR/Libraries/MLXLMCommon/DFlash2.swift"
grep -Fq 'public final class DFlash2DraftModel' \
  "$VENDOR/Libraries/MLXLMCommon/DFlash2.swift"
grep -Fq 'extension Qwen3_5MoEModel: DFlash2Target' \
  "$VENDOR/Libraries/MLXLLM/Models/Qwen3_5MoE.swift"
grep -Fq 'extension MuseGlimmer: DFlash2Target' \
  "$VENDOR/Libraries/MLXVLM/Models/MuseGlimmer.swift"

echo "DFlash2 vendor patch application: OK"
