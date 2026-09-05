#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
work_parent="${TMPDIR:-$repo_root/.build}"
mkdir -p "$work_parent"
work_root="$(mktemp -d "$work_parent/afm-judge-root.XXXXXX")"
trap 'rm -rf "$work_root"' EXIT
# Exercise the harness's actual configuration statements without model loading.
configuration="$(awk '/^RESULTS_FILE=/{copy=1} copy{print} copy && /^mkdir -p/{exit}' "$repo_root/Scripts/mlx-model-test.sh")"
[[ -n "$configuration" ]]
(
  unset AFM_TEST_WORK_ROOT
  AFM_RESULTS_FILE="$work_root/reports with spaces/results.jsonl"
  eval "$configuration"
  [[ "$TEST_WORK_ROOT" == "$work_root/reports with spaces/judge-work" ]]
  [[ -d "$TEST_WORK_ROOT" ]]
)
(
  AFM_RESULTS_FILE="$work_root/other/results.jsonl"
  AFM_TEST_WORK_ROOT="$work_root/explicit work"
  eval "$configuration"
  [[ "$TEST_WORK_ROOT" == "$AFM_TEST_WORK_ROOT" && -d "$TEST_WORK_ROOT" ]]
)
if grep -Eq 'mktemp.* /tmp/|2>/tmp/' "$repo_root/Scripts/mlx-model-test.sh"; then
  echo 'Judge scratch paths must not bypass the configured work root' >&2
  exit 1
fi
echo 'Comprehensive judge work-root tests passed.'
