#!/bin/bash
# Shared report renderer; sourced by test-assertions.sh for normal and aborted runs.
generate_assertion_report() {
TEST_END_TIME=$(date +%s)
TOTAL_SECS=$((TEST_END_TIME - TEST_START_TIME))

EFFECTIVE_TOTAL=$((TOTAL - SKIP))
if [ $EFFECTIVE_TOTAL -gt 0 ]; then
  PCT=$(( PASS * 100 / EFFECTIVE_TOTAL ))
else
  PCT=0
fi

if [ $FAIL -eq 0 ]; then
  BAR_COLOR="#3fb950"
else
  BAR_COLOR="#f85149"
fi

DATE_STR=$(date '+%Y-%m-%d %H:%M:%S')

cat > "$REPORT_FILE" <<'HTMLHEAD'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AFM Assertion Test Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif; background: #0d1117; color: #e6edf3; padding: 2rem; }
  .header { text-align: center; margin-bottom: 2rem; padding: 2rem; background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%); border: 1px solid #30363d; border-radius: 12px; }
  .header h1 { font-size: 1.8rem; margin-bottom: 0.5rem; background: linear-gradient(90deg, #58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header .meta { color: #8b949e; font-size: 0.9rem; line-height: 1.6; }
  .summary { display: flex; gap: 1rem; justify-content: center; margin: 1.5rem 0; flex-wrap: wrap; }
  .stat { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1rem 1.5rem; text-align: center; min-width: 120px; }
  .stat .value { font-size: 2rem; font-weight: 700; }
  .stat .label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
  .stat.pass .value { color: #3fb950; }
  .stat.fail .value { color: #f85149; }
  .stat.skip .value { color: #d29922; }
  .stat.time .value { color: #58a6ff; }
  .stat.pct .value { color: #d2a8ff; }
  .progress-bar { width: 100%; height: 8px; background: #21262d; border-radius: 4px; margin: 1rem auto; max-width: 400px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 4px; }
  table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; }
  th { background: #161b22; color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #30363d; }
  td { padding: 0.75rem 1rem; border-bottom: 1px solid #21262d; vertical-align: top; }
  tr:hover { background: #161b22; }
  .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  .badge.pass { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .badge.fail { background: #2d1215; color: #f85149; border: 1px solid #da3633; }
  .badge.skip { background: #2d2400; color: #d29922; border: 1px solid #9e6a03; }
  .group-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px; font-size: 0.7rem; font-weight: 500; background: #1a1f2e; color: #8b949e; border: 1px solid #30363d; }
  .group-badge.Preflight { color: #8b949e; border-color: #484f58; }
  .group-badge.Lifecycle { color: #3fb950; border-color: #238636; }
  .group-badge.Stop { color: #f85149; border-color: #da3633; }
  .group-badge.Logprobs { color: #58a6ff; border-color: #1f6feb; }
  .group-badge.Think { color: #d2a8ff; border-color: #8957e5; }
  .group-badge.Tools { color: #ffa657; border-color: #d18616; }
  .group-badge.XMLTools { color: #f0883e; border-color: #bd561d; }
  .group-badge.Cache { color: #79c0ff; border-color: #388bfd; }
  .group-badge.Concurrent { color: #f778ba; border-color: #db61a2; }
  .group-badge.Error { color: #ff7b72; border-color: #da3633; }
  .group-badge.Kwargs { color: #a5d6ff; border-color: #58a6ff; }
  .group-badge.Perf { color: #3fb950; border-color: #238636; }
  .group-badge.Structured { color: #d2a8ff; border-color: #8957e5; }
  .group-badge.AdaptiveXML { color: #f0883e; border-color: #bd561d; }
  .group-badge.Grammar { color: #d2a8ff; border-color: #8957e5; }
  .group-badge.XMLParsing { color: #f0883e; border-color: #bd561d; }
  .group-badge.NullableSchema { color: #79c0ff; border-color: #388bfd; }
  .group-badge.UnitTest { color: #a5d6ff; border-color: #58a6ff; }
  .group-badge.Batch { color: #7ee787; border-color: #3fb950; }
  .tier-row td { background: #161b22; padding: 0.6rem 1rem; font-weight: 700; font-size: 0.9rem; border-bottom: 2px solid #30363d; border-top: 2px solid #30363d; }
  .tier-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.7rem; font-weight: 600; }
  .tier-badge.unit { background: #1a1a2e; color: #a5d6ff; border: 1px solid #58a6ff; }
  .tier-badge.smoke { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .tier-badge.standard { background: #0d1a30; color: #58a6ff; border: 1px solid #1f6feb; }
  .tier-badge.full { background: #2d1f00; color: #d29922; border: 1px solid #9e6a03; }
  .detail { font-family: 'SF Mono', 'Menlo', monospace; font-size: 0.8rem; color: #8b949e; white-space: pre-wrap; word-break: break-word; max-height: 100px; overflow-y: auto; background: #0d1117; padding: 0.5rem; border-radius: 6px; border: 1px solid #21262d; margin-top: 0.25rem; }
  .duration { color: #8b949e; font-family: 'SF Mono', monospace; font-size: 0.85rem; }
  .test-idx { color: #8b949e; font-family: 'SF Mono', monospace; font-size: 0.85rem; }
  .footer { text-align: center; margin-top: 2rem; color: #484f58; font-size: 0.8rem; }
</style>
</head>
<body>
HTMLHEAD

if [ -s "$TRANSPORT_FAILURE_FILE" ]; then
  printf '%s\n' '<p role="alert">Partial run: engine unavailable to an assertion request. Remaining assertions were not executed.</p>' >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" <<EOF
<div class="header">
  <h1>AFM Assertion Test Report</h1>
  <div class="meta">
    Model: <strong>$MODEL</strong> &middot; Tier: <strong>$TIER</strong> &middot; Grammar: <strong>$GRAMMAR_CONSTRAINTS</strong><br>
    Server: <code>$BASE_URL</code><br>
    Date: $DATE_STR
  </div>
</div>
<div class="summary">
  <div class="stat pass"><div class="value">$PASS</div><div class="label">Passed</div></div>
  <div class="stat fail"><div class="value">$FAIL</div><div class="label">Failed</div></div>
  <div class="stat skip"><div class="value">$SKIP</div><div class="label">Skipped</div></div>
  <div class="stat pct"><div class="value">${PCT}%</div><div class="label">Pass Rate</div></div>
  <div class="stat time"><div class="value">${TOTAL_SECS}s</div><div class="label">Total Time</div></div>
</div>
<div class="progress-bar"><div class="progress-fill" style="width:${PCT}%;background:${BAR_COLOR};"></div></div>
<table>
<thead>
<tr><th>#</th><th>Test</th><th>Group</th><th>Coverage</th><th>Status</th><th>Classification</th><th>Duration</th><th>Details</th></tr>
</thead>
<tbody>
EOF

# Emit all rows in execution order with coverage tier badges
for entry in "${RESULTS[@]}"; do
  IFS='|' read -r status group name expected actual duration tier test_idx classification <<< "$entry"
  tier="${tier:-smoke}"
  test_idx="${test_idx:-0}"
  classification="${classification:-needs triage}"

  if [ "$status" = "PASS" ]; then
    badge='<span class="badge pass">PASS</span>'
    detail_text="$expected"
  elif [ "$status" = "SKIP" ]; then
    badge='<span class="badge skip">SKIP</span>'
    detail_text="$expected"
  else
    badge='<span class="badge fail">FAIL</span>'
    detail_text="Expected: $expected\nActual: $actual"
  fi
  detail_text=$(echo "$detail_text" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
  name_esc=$(echo "$name" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
  classification_esc=$(echo "$classification" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')

  dur_s=""
  if [ -n "$duration" ] && [ "$duration" -gt 0 ] 2>/dev/null; then
    dur_s=$(python3 -c "print(f'{$duration/1000:.1f}s')" 2>/dev/null || echo "${duration}ms")
  fi

  # Coverage badges: show which tiers include this test
  # smoke tests run in smoke+standard+full, standard in standard+full, full in full only
  tier_badges=""
  case "$tier" in
    unit)     tier_badges='<span class="tier-badge unit">unit</span> <span class="tier-badge smoke">smoke</span> <span class="tier-badge standard">standard</span> <span class="tier-badge full">full</span>' ;;
    smoke)    tier_badges='<span class="tier-badge smoke">smoke</span> <span class="tier-badge standard">standard</span> <span class="tier-badge full">full</span>' ;;
    standard) tier_badges='<span class="tier-badge standard">standard</span> <span class="tier-badge full">full</span>' ;;
    full)     tier_badges='<span class="tier-badge full">full</span>' ;;
  esac

  cat >> "$REPORT_FILE" <<EOF
<tr>
  <td><span class="test-idx">${test_idx}</span></td>
  <td>$name_esc</td>
  <td><span class="group-badge $group">$group</span></td>
  <td>${tier_badges}</td>
  <td>$badge</td>
  <td>${classification_esc}</td>
  <td><span class="duration">$dur_s</span></td>
  <td><div class="detail">$detail_text</div></td>
</tr>
EOF
done

cat >> "$REPORT_FILE" <<EOF
</tbody>
</table>
<div class="footer">
  Generated by Scripts/test-assertions.sh (tier: $TIER) &mdash; $(date '+%Y-%m-%d %H:%M:%S')
</div>
</body>
</html>
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: $PASS/$EFFECTIVE_TOTAL passed ($PCT%) | $SKIP skipped"
if [ $FAIL -gt 0 ]; then
  echo "  ❌ $FAIL FAILED"
fi
echo "  Report: $REPORT_FILE"
echo "  JSONL:  $JSONL_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

}
