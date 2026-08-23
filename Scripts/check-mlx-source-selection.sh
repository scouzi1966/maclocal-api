#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${MACLOCAL_AFMKIT_PATH:-}" ]]; then
    AFMKIT_ROOT="$(cd "$MACLOCAL_AFMKIT_PATH" && pwd)"
else
    AFMKIT_ROOT="$ROOT_DIR/.build/checkouts/AFMKit"
fi

for manifest in \
    "$AFMKIT_ROOT/vendor/MLX/mlx-swift/Package.swift" \
    "$AFMKIT_ROOT/vendor/MLX/mlx-swift-lm/Package.swift"; do
    [[ -f "$manifest" ]] || {
        echo "error: AFMKit does not contain its vendored MLX package: $manifest" >&2
        exit 1
    }
done

swift package --package-path "$AFMKIT_ROOT" dump-package | python3 -c '
import json
import sys

package = json.load(sys.stdin)
if any(dependency.get("fileSystem") for dependency in package.get("dependencies", [])):
    raise SystemExit("error: AFMKit must not expose local package dependencies")
targets = {target.get("name"): target.get("path", "") for target in package.get("targets", [])}
expected = {
    "Cmlx": "vendor/MLX/mlx-swift/Source/Cmlx",
    "MLX": "vendor/MLX/mlx-swift/Source/MLX",
    "MLXLMCommon": "vendor/MLX/mlx-swift-lm/Libraries/MLXLMCommon",
    "MLXLLM": "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM",
    "MLXVLM": "vendor/MLX/mlx-swift-lm/Libraries/MLXVLM",
}
for name, suffix in expected.items():
    path = targets.get(name, "")
    if not path.endswith(suffix):
        raise SystemExit(f"error: AFMKit target {name} does not resolve {suffix}")
'

echo "MLX source selection: AFMKit/vendor/MLX"
