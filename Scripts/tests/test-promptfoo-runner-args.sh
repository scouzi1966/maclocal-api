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
# Every requested profile first discovers capabilities using the native profile.
# The real CPU server fixtures verify supported forced profiles after discovery.
[[ "$(print -r -- "$zero_line" | occurrences afm_adaptive_xml)" == "0" ]]
[[ "$(print -r -- "$zero_line" | occurrences --enable-grammar-constraints)" == "0" ]]

enabled_line="$(trace_launch adaptive-xml AFM_NO_THINK=1 AFM_MTP=1 | server_argv)"
[[ "$(print -r -- "$enabled_line" | occurrences --no-think)" == "1" ]]
[[ "$(print -r -- "$enabled_line" | occurrences --mtp)" == "1" ]]
[[ "$(print -r -- "$enabled_line" | occurrences afm_adaptive_xml)" == "0" ]]

# Quoting must retain a support checkpoint path containing spaces as one value.
touch "$work_root/support checkpoint.gguf"
dspark_line="$(trace_launch default AFM_DSPARK_SUPPORT="$work_root/support checkpoint.gguf" AFM_PROMPTFOO_LOAD_TIMEOUT_SECONDS=900 | server_argv)"
[[ "$(print -r -- "$dspark_line" | occurrences --dspark-support)" == "1" ]]
[[ "$dspark_line" == *"support checkpoint.gguf"* ]]

mkdir "$work_root/mtp head"
mtp_line="$(trace_launch default AFM_MTP=1 AFM_MTP_MODEL="$work_root/mtp head" | server_argv)"
[[ "$(print -r -- "$mtp_line" | occurrences --mtp-model)" == "1" ]]
[[ "$mtp_line" == *"mtp head"* ]]
set +e
mtp_output="$(AFM_MTP=0 AFM_MTP_MODEL="$work_root/mtp head" "$runner" all 2>&1)"
mtp_status=$?
set -e
[[ "$mtp_status" == "1" ]]
[[ "$mtp_output" == "AFM_MTP_MODEL requires AFM_MTP=1 and an existing local model directory" ]]

for bad_timeout in 0 -1 invalid; do
  set +e
  timeout_output="$(AFM_PROMPTFOO_LOAD_TIMEOUT_SECONDS="$bad_timeout" "$runner" all 2>&1)"
  timeout_status=$?
  set -e
  [[ "$timeout_status" == "1" ]]
  [[ "$timeout_output" == "AFM_PROMPTFOO_LOAD_TIMEOUT_SECONDS must be a positive integer" ]]
done

set +e
missing_output="$(AFM_DSPARK_SUPPORT="$work_root/missing.gguf" "$runner" all 2>&1)"
missing_status=$?
set -e
[[ "$missing_status" == "1" ]]
[[ "$missing_output" == "AFM_DSPARK_SUPPORT must name an existing support GGUF file" ]]

set +e
invalid_output="$(AFM_NO_THINK=invalid "$runner" all 2>&1)"
invalid_status=$?
set -e
[[ "$invalid_status" == "1" ]]
[[ "$invalid_output" == "AFM_NO_THINK must be 0 or 1" ]]

echo "Promptfoo runner argument tests passed."
