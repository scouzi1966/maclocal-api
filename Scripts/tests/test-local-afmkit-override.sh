#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AFMKIT_PATH="${AFMKIT_OVERRIDE_TEST_PATH:-$ROOT_DIR/.build/checkouts/AFMKit}"

fail() {
  echo "[local-afmkit-test] $*" >&2
  exit 1
}

if [[ ! -f "$AFMKIT_PATH/Package.swift" ]]; then
  "$ROOT_DIR/Scripts/resolve-release-dependencies.sh"
fi

expected_revision="$(python3 - "$ROOT_DIR/Package.resolved" <<'PY'
import json
import sys
lock = json.load(open(sys.argv[1]))
print(next(pin for pin in lock["pins"] if pin["identity"] == "afmkit")["state"]["revision"])
PY
)"
actual_revision="$(git -C "$AFMKIT_PATH" rev-parse HEAD)"
[[ "$actual_revision" == "$expected_revision" ]] || \
  fail "fixture must be at the locked AFMKit revision"

manifest_hash="$(shasum -a 256 "$ROOT_DIR/Package.swift" | cut -d' ' -f1)"
lock_hash="$(shasum -a 256 "$ROOT_DIR/Package.resolved" | cut -d' ' -f1)"

for invocation in 1 2; do
  echo "[local-afmkit-test] exact-head invocation $invocation"
  MACLOCAL_AFMKIT_PATH="$AFMKIT_PATH" \
    "$ROOT_DIR/Scripts/swiftpm-reliable.sh" build \
      --target AFMKitFoundationModels

  [[ "$(shasum -a 256 "$ROOT_DIR/Package.swift" | cut -d' ' -f1)" == "$manifest_hash" ]] || \
    fail "local invocation changed the tracked manifest"
  [[ "$(shasum -a 256 "$ROOT_DIR/Package.resolved" | cut -d' ' -f1)" == "$lock_hash" ]] || \
    fail "local invocation changed the tracked release lock"
done

workspace="$ROOT_DIR/.build-local-afmkit-workspace/package"
MACLOCAL_AFMKIT_WORKSPACE_PATH="$AFMKIT_PATH" \
  swift package --package-path "$workspace" --scratch-path "$ROOT_DIR/.build" \
    show-dependencies --format json > "$ROOT_DIR/.build/local-afmkit-dependencies.json"
python3 - "$ROOT_DIR/.build/local-afmkit-dependencies.json" "$AFMKIT_PATH" <<'PY'
import json
import os
import sys

graph = json.load(open(sys.argv[1]))
expected = os.path.realpath(sys.argv[2])
afmkit = next(item for item in graph["dependencies"] if item["identity"] == "afmkit")
if os.path.realpath(afmkit["path"]) != expected:
    raise SystemExit("generated workspace did not consume the requested AFMKit checkout")
PY

echo "[local-afmkit-test] repeated local override preserved the tracked manifest and lock"
