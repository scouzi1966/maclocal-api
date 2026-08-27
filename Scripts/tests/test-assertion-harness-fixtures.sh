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

pairwise = source.split("# Section 16: Pairwise Smoke", 1)[1]
if pairwise.count("2>&1 || echo 'ERROR')") < 8:
    raise SystemExit("pairwise curl failures are not fully converted into assertion results")
if "--max-time \"$REQUEST_TIMEOUT\"" not in pairwise:
    raise SystemExit("pairwise requests do not use the configurable assertion timeout")
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

if AFM_ASSERTIONS_REQUEST_TIMEOUT=invalid \
    "$ROOT_DIR/Scripts/test-assertions.sh" --tier unit >/dev/null 2>"$WORK_ROOT/timeout-error"; then
    echo "invalid assertion timeout was accepted" >&2
    exit 1
fi
grep -q "AFM_ASSERTIONS_REQUEST_TIMEOUT must be a positive integer" "$WORK_ROOT/timeout-error" || {
    echo "invalid assertion timeout did not produce the expected diagnostic" >&2
    exit 1
}

echo "assertion harness fixture checks passed"
