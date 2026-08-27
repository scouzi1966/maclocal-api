#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h:h}"
runner="$repo_root/Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh"
work_root="$(mktemp -d "${TMPDIR:-/tmp}/afm-promptfoo-args.XXXXXX")"

cleanup() {
  rm -rf "$work_root"
}
trap cleanup EXIT INT TERM

trace_launch() {
  local profile="$1"
  shift
  env \
    "$@" \
    AFM_BINARY=/usr/bin/false \
    AFM_PROMPTFOO_OUT_DIR="$work_root" \
    zsh -x "$runner" "$profile" 2>&1 || true
}

server_argv() {
  grep '/usr/bin/false mlx' | tail -1
}

occurrences() {
  local pattern="$1"
  awk -v pattern="$pattern" '{ total += gsub(pattern, "") } END { print total + 0 }'
}

unset_line="$(trace_launch default -u AFM_NO_THINK -u AFM_MTP | server_argv)"
[[ -n "$unset_line" ]]
[[ "$(print -r -- "$unset_line" | occurrences --no-think)" == "0" ]]
[[ "$(print -r -- "$unset_line" | occurrences --mtp)" == "0" ]]

zero_line="$(trace_launch adaptive-xml-grammar AFM_NO_THINK=0 AFM_MTP=0 | server_argv)"
[[ "$(print -r -- "$zero_line" | occurrences --no-think)" == "0" ]]
[[ "$(print -r -- "$zero_line" | occurrences --mtp)" == "0" ]]
[[ "$(print -r -- "$zero_line" | occurrences afm_adaptive_xml)" == "1" ]]
[[ "$(print -r -- "$zero_line" | occurrences --enable-grammar-constraints)" == "1" ]]

enabled_line="$(trace_launch adaptive-xml AFM_NO_THINK=1 AFM_MTP=1 | server_argv)"
[[ "$(print -r -- "$enabled_line" | occurrences --no-think)" == "1" ]]
[[ "$(print -r -- "$enabled_line" | occurrences --mtp)" == "1" ]]
[[ "$(print -r -- "$enabled_line" | occurrences afm_adaptive_xml)" == "1" ]]

set +e
invalid_output="$(AFM_NO_THINK=invalid "$runner" all 2>&1)"
invalid_status=$?
set -e
[[ "$invalid_status" == "1" ]]
[[ "$invalid_output" == "AFM_NO_THINK must be 0 or 1" ]]

echo "Promptfoo runner argument tests passed."
