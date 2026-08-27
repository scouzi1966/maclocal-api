#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h:h}"
work_root="$(mktemp -d "${TMPDIR:-/tmp}/afm-promptfoo-summary.XXXXXX")"
model="example/Test-Model"
slug="example_Test-Model"

cleanup() {
  rm -rf "$work_root"
}
trap cleanup EXIT INT TERM

write_report() {
  local filename="$1"
  local successes="$2"
  local failures="$3"
  local errors="$4"
  local description="$5"
  local success="$6"
  local failure_cause="${7:-}"
  local reason="${8:-}"
  local metadata='{}'
  local error_field=''
  if [[ -n "$failure_cause" ]]; then
    metadata="{\"failureCause\":\"${failure_cause}\"}"
  fi
  if [[ -n "$reason" ]]; then
    error_field=",\"error\":\"${reason}\""
  fi
  print -r -- "{\"results\":{\"stats\":{\"successes\":${successes},\"failures\":${failures},\"errors\":${errors}},\"results\":[{\"success\":${success}${error_field},\"testCase\":{\"description\":\"${description}\",\"metadata\":${metadata}}}]}}" > "$work_root/$filename-$slug.json"
}

write_report structured 1 0 0 native-pass true
write_report grammar-schema-concurrent 0 1 0 native-fail false
# Merely mentioning Promptfoo in an assertion reason is not a harness failure.
write_report opencode-default 0 1 0 behavior-fail false '' 'Expected output to mention promptfoo'
write_report opencode-adaptive-xml 0 1 0 forced-fail false
write_report pi-adaptive-xml-grammar 0 0 1 forced-error false
write_report toolcall-default 0 1 0 unresolved-fail false
write_report structured-stress 0 1 0 harness-fail false 'test harness'
# Explicit stronger evidence overrides the otherwise forced-parser attribution.
write_report hermes-adaptive-xml 0 1 0 forced-harness-fail false 'test harness'

node "$repo_root/Scripts/feature-promptfoo-agentic/summarize-results.mjs" "$work_root" "$model" >/dev/null

summary="$work_root/promptfoo-summary-$slug.json"
[[ -f "$summary" ]]
jq -e '
  .categories.nativeProtocolConformance.cases == 4 and
  .categories.nativeProtocolConformance.successes == 1 and
  .categories.nativeProtocolConformance.failures == 3 and
  .categories.modelAgentBehaviorQuality.cases == 1 and
  .categories.modelAgentBehaviorQuality.failures == 1 and
  .categories.forcedParserCompatibility.cases == 3 and
  .categories.forcedParserCompatibility.failures == 2 and
  .categories.forcedParserCompatibility.errors == 1 and
  .failureTaxonomy.totalFailuresAndErrors == 7 and
  .failureTaxonomy.buckets["engine/runtime likely"].count == 1 and
  .failureTaxonomy.buckets["model behavior likely"].count == 1 and
  .failureTaxonomy.buckets["test harness"].count == 2 and
  .failureTaxonomy.buckets["forced-parser experiment"].count == 2 and
  .failureTaxonomy.buckets.unresolved.count == 1
' "$summary" >/dev/null

rg -q 'Native protocol conformance' "$work_root/promptfoo-summary-$slug.md"
rg -q 'Forced-parser compatibility experiments' "$work_root/promptfoo-summary-$slug.md"
for label in \
  'engine/runtime likely' \
  'model behavior likely' \
  'test harness' \
  'forced-parser experiment' \
  'unresolved'; do
  rg -q "$label" "$work_root/promptfoo-summary-$slug.md"
done

# A current-run cutoff must exclude stale reports rather than silently mixing
# them into a partial rerun. A future cutoff leaves no eligible reports and
# therefore fails closed.
set +e
node "$repo_root/Scripts/feature-promptfoo-agentic/summarize-results.mjs" \
  "$work_root" "$model" 9999999999999 >/dev/null 2>&1
stale_status=$?
set -e
[[ "$stale_status" == "1" ]]

echo "Promptfoo categorized summary tests passed."
