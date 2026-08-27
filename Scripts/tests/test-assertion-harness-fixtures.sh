#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

echo "assertion harness fixture checks passed"
