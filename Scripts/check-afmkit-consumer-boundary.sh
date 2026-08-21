#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

fail() {
  echo "[afmkit-boundary] $*" >&2
  exit 1
}

for shadow_target in AFMKitCore AFMKitMLX AFMKitDwarfStar; do
  [[ ! -d "Sources/$shadow_target" ]] || \
    fail "consumer shadow target still exists: Sources/$shadow_target"
done

grep -Fq 'revision: "dfeab23e95ea1979432958e3f9b002beb5685191"' Package.swift || \
  fail "Package.swift does not pin the immutable AFMKit checkpoint"
grep -Fq 'exact: "0.31.6-afm.3"' Package.swift || \
  fail "Package.swift does not pin mlx-swift-lm 0.31.6-afm.3"
grep -Fq 'exact: "0.31.6-afm.1"' Package.swift || \
  fail "Package.swift does not pin mlx-swift-afm 0.31.6-afm.1"

python3 - <<'PY'
import json
from pathlib import Path

expected = {
    "afmkit": "dfeab23e95ea1979432958e3f9b002beb5685191",
    "mlx-swift-afm": "6000b7b26b70be2713c74e9ec2adeb89be07b9e5",
    "mlx-swift-lm": "e0d7fa71bc5e422a416f191c297264f698391561",
}
pins = {
    pin["identity"]: pin["state"].get("revision")
    for pin in json.loads(Path("Package.resolved").read_text())["pins"]
}
for identity, revision in expected.items():
    if pins.get(identity) != revision:
        raise SystemExit(
            f"[afmkit-boundary] {identity} resolved to {pins.get(identity)!r}, "
            f"expected {revision}"
        )
PY

if grep -Eq '^(build|debug):.*(PATCH_STAMP|patch)' Makefile; then
  fail "normal Make targets must not depend on the legacy patch stack"
fi
if grep -Fq '$(PATCH_STAMP)' Makefile; then
  fail "Makefile still uses a vendor patch stamp"
fi

if grep -ERn \
  'MacLocalAPI_AFMKit(MLX|DwarfStar)\.bundle' \
  .github/workflows Scripts/build-nightly-wheel.sh Scripts/create-tarball.sh \
  Scripts/generate-tap-versioned.sh Scripts/publish-next.sh Scripts/publish-stable.sh \
  Scripts/verify-native-wheel.sh >/dev/null; then
  fail "release packaging still references a maclocal-api-owned provider bundle"
fi

for workflow in .github/workflows/nightly.yml .github/workflows/release.yml; do
  grep -Fq 'AFMKit_AFMKitMLX.bundle' "$workflow" || \
    fail "$workflow does not package the AFMKit MLX resource bundle"
  grep -Fq 'AFMKit_AFMKitDwarfStar.bundle' "$workflow" || \
    fail "$workflow does not package the AFMKit DwarfStar resource bundle"
done

if grep -ERn \
  '\b(MLXModelService|BatchScheduler|RequestSlot|RadixTreeCache|ToolCallStreamingRuntime)\b' \
  Sources/AFMServer \
  | grep -Ev '^[^:]+:[0-9]+:[[:space:]]*//' >/dev/null; then
  fail "AFMServer reaches into AFMKitMLX implementation types"
fi

echo "[afmkit-boundary] immutable graph, facades, and release ownership verified"
