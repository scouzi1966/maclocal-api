#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="$(mktemp -d)"
trap 'rm -rf "$WORK_ROOT"' EXIT

python3 - "$ROOT_DIR/Scripts/test-assertions.sh" <<'PY'
import json
import re
import sys

source = open(sys.argv[1], encoding="utf-8").read()
line = next(
    (line for line in source.splitlines() if "AFM_LINE_ONE" in line and "api_call" in line),
    None,
)
if line is None:
    raise SystemExit("newline stop assertion request was not found")

match = re.search(r"api_call '(.*)'\)$", line)
if match is None:
    raise SystemExit("newline stop assertion request could not be parsed")

request = json.loads(match.group(1))
if request.get("stop") != ["\n"]:
    raise SystemExit(
        f"newline stop fixture sent {request.get('stop')!r}, expected an actual newline"
    )
PY

if rg -q "find .*\\.xctest/Contents/MacOS" "$ROOT_DIR/Scripts/swiftpm-reliable.sh"; then
    echo "swiftpm-reliable must not mutate existing signed XCTest bundles" >&2
    exit 1
fi

scratch_path="$WORK_ROOT/scratch"
state_path="$WORK_ROOT/state"
stale_bundle="$scratch_path/out/Products/Debug/StaleTests.xctest/Contents/MacOS"
mkdir -p "$stale_bundle" "$scratch_path/checkouts/PreservedDependency"
touch "$stale_bundle/mlx.metallib" "$scratch_path/checkouts/PreservedDependency/Package.swift"

"$ROOT_DIR/Scripts/migrate-xctest-metallib-layout.sh" "$scratch_path" "$state_path"
[[ ! -e "$scratch_path/out" ]] || {
    echo "stale native-driver products were not removed" >&2
    exit 1
}
[[ -f "$scratch_path/checkouts/PreservedDependency/Package.swift" ]] || {
    echo "migration removed a preserved dependency checkout" >&2
    exit 1
}

# The per-scratch stamp makes this migration one-time. A later clean native
# product tree must remain untouched even if it contains a same-named resource.
mkdir -p "$stale_bundle"
touch "$stale_bundle/mlx.metallib"
"$ROOT_DIR/Scripts/migrate-xctest-metallib-layout.sh" "$scratch_path" "$state_path"
[[ -f "$stale_bundle/mlx.metallib" ]] || {
    echo "completed migration ran more than once" >&2
    exit 1
}

echo "assertion harness fixture checks passed"
