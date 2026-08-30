#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="$(mktemp -d)"
timeout_server_pid=""
cleanup() {
    if [[ -n "$timeout_server_pid" ]]; then
        kill "$timeout_server_pid" 2>/dev/null || true
        wait "$timeout_server_pid" 2>/dev/null || true
    fi
    rm -rf "$WORK_ROOT"
}
trap cleanup EXIT

help_output="$($ROOT_DIR/Scripts/test-assertions.sh --help)"
grep -q "AFM_RUN_FLUX_INTEGRATION=1" <<<"$help_output" || {
    echo "assertion harness help does not document the FLUX integration opt-in" >&2
    exit 1
}
grep -q "not required for normal /v1/images API operation" <<<"$help_output" || {
    echo "assertion harness help does not distinguish the test flag from runtime enablement" >&2
    exit 1
}

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
for result_var in ("R", "NS", "S", "R1", "R2"):
    if f') || {result_var}=""' not in pairwise:
        raise SystemExit(f"pairwise curl failures do not clear partial {result_var} output")
for parsed_var in ("NS_C", "S_C", "C1", "C2"):
    if f') || {parsed_var}=""' not in pairwise:
        raise SystemExit(f"pairwise JSON parsing does not tolerate empty {parsed_var} input")
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

# A server that accepts each request and then stalls exercises curl's timeout
# path, including the partial-output hazard of a streaming response. The
# assertion harness must record every pairwise failure and still write reports.
timeout_port_file="$WORK_ROOT/timeout-server-port"
/usr/bin/python3 "$ROOT_DIR/Scripts/tests/fixtures/slow-openai-server.py" "$timeout_port_file" &
timeout_server_pid=$!
for _ in $(seq 1 50); do
    [[ -s "$timeout_port_file" ]] && break
    sleep 0.1
done
[[ -s "$timeout_port_file" ]] || {
    echo "timeout fixture server did not start" >&2
    kill "$timeout_server_pid" 2>/dev/null || true
    exit 1
}
timeout_report_dir="$WORK_ROOT/timeout-reports"
timeout_status=0
AFM_ASSERTIONS_REQUEST_TIMEOUT=1 \
AFM_ASSERTIONS_REPORT_DIR="$timeout_report_dir" \
    "$ROOT_DIR/Scripts/test-assertions.sh" \
    --tier standard --section 16 --model fixture/slow-model \
    --port "$(<"$timeout_port_file")" --bin /usr/bin/true \
    >"$WORK_ROOT/timeout-run-output" 2>&1 || timeout_status=$?
kill "$timeout_server_pid" 2>/dev/null || true
wait "$timeout_server_pid" 2>/dev/null || true
timeout_server_pid=""
[[ "$timeout_status" -eq 6 ]] || {
    echo "timeout fixture expected six recorded failures, got status $timeout_status" >&2
    cat "$WORK_ROOT/timeout-run-output" >&2
    exit 1
}
timeout_jsonl=$(find "$timeout_report_dir" -maxdepth 1 -name 'assertions-report-*.jsonl' -print -quit)
timeout_html=$(find "$timeout_report_dir" -maxdepth 1 -name 'assertions-report-*.html' -print -quit)
[[ -n "$timeout_jsonl" && -n "$timeout_html" ]] || {
    echo "timeout fixture did not produce both JSONL and HTML reports" >&2
    exit 1
}
python3 - "$timeout_jsonl" <<'PY'
import json
import sys

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
pairwise = [record for record in records if record["group"] == "PairwiseSmoke"]
if len(pairwise) != 6 or any(record["status"] != "FAIL" for record in pairwise):
    raise SystemExit(f"expected six recorded pairwise failures, got {pairwise!r}")
PY

echo "assertion harness fixture checks passed"
