#!/bin/bash
set -uo pipefail

# AFM MLX CLI Piping Tests
# Tests single-prompt mode (-s) and stdin piping with various unix command combinations.
# Usage: ./Scripts/afm-cli-tests.sh [model] [afm-binary]
# Generates an HTML report in test-reports/

export MACAFM_MLX_MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
AFM="${2:-$(dirname "$0")/../.build/release/afm}"
M="${1:-gemma-3-4b-it-8bit}"
CD="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="$CD/test-reports"
REPORT_FILE="$REPORT_DIR/cli-pipe-report-${TIMESTAMP}.html"
mkdir -p "$REPORT_DIR"

PASS=0
FAIL=0
TOTAL=15
TEST_START=$(date +%s)

# Arrays to collect results for HTML
declare -a TEST_NUMS TEST_DESCS TEST_TYPES TEST_STATUSES TEST_OUTPUTS TEST_DURATIONS

run_test() {
  local num=$1
  local desc="$2"
  local type="$3"
  shift 3
  echo ""
  echo "=== Test $num: $desc ==="
  local t_start=$(date +%s)
  local output
  if output=$(bash -c "$*" 2>/dev/null | tr '\r' '\n' | sed '/^\[/d;/^MLX model:/d;/^Loading MLX/d;/^MLX GPU:/d;/^$/d'); then
    if [ -n "$output" ]; then
      echo "PASS"
      echo "$output" | head -3
      PASS=$((PASS + 1))
      TEST_STATUSES[$num]="PASS"
    else
      echo "FAIL (empty output)"
      FAIL=$((FAIL + 1))
      TEST_STATUSES[$num]="FAIL"
      output="(empty output)"
    fi
  else
    echo "FAIL (error)"
    FAIL=$((FAIL + 1))
    TEST_STATUSES[$num]="FAIL"
    output="(error - non-zero exit)"
  fi
  local t_end=$(date +%s)
  TEST_NUMS[$num]=$num
  TEST_DESCS[$num]="$desc"
  TEST_TYPES[$num]="$type"
  TEST_OUTPUTS[$num]="$output"
  TEST_DURATIONS[$num]=$(( t_end - t_start ))
}

echo "# AFM MLX CLI Piping Tests"
echo "# Model: $M"
echo "# Binary: $AFM"
echo "# $(date)"

# --- Piped input with -s prompt ---

run_test 1 "git log | afm -s (summarize commits)" "pipe+flag" \
  "cd $CD && git log --oneline -10 | $AFM mlx -m $M -s 'Summarize these git commits in one sentence'"

run_test 2 "ls | afm -s (analyze project files)" "pipe+flag" \
  "ls $CD/Sources/MacLocalAPI/Controllers/ | $AFM mlx -m $M -s 'What kind of project has these files? One sentence.'"

run_test 3 "ps aux | afm -s (process analysis)" "pipe+flag" \
  "ps aux | head -15 | $AFM mlx -m $M -s 'What type of system is this? One sentence.'"

run_test 4 "git diff --stat | afm -s (diff summary)" "pipe+flag" \
  "cd $CD && git diff HEAD~1 --stat | $AFM mlx -m $M -s 'Summarize what changed in this diff in one sentence'"

run_test 5 "echo error | afm -s (error diagnosis)" "pipe+flag" \
  "echo 'ERROR: connection refused on port 5432, FATAL: password auth failed for user postgres' | $AFM mlx -m $M -s 'Diagnose this error and give a one-line fix'"

run_test 6 "du | afm -s (disk usage analysis)" "pipe+flag" \
  "du -sh $CD/*/ 2>/dev/null | head -10 | $AFM mlx -m $M -s 'Which directories use the most space? One sentence.'"

run_test 7 "brew list | afm -s (package audit)" "pipe+flag" \
  "brew list --formula 2>/dev/null | head -20 | $AFM mlx -m $M -s 'Which of these are dev tools? One line.'"

run_test 8 "sysctl | afm -s (system profile)" "pipe+flag" \
  "sysctl -a 2>/dev/null | grep -E 'machdep.cpu|hw.memsize|hw.ncpu' | head -10 | $AFM mlx -m $M -s 'Describe this machines specs in one sentence'"

run_test 9 "echo python | afm -s (translate to Rust)" "pipe+flag" \
  "echo 'def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)' | $AFM mlx -m $M -s 'Translate this Python to Rust. Just the code.'"

run_test 10 "echo JSON | afm -s (config validation)" "pipe+flag" \
  "echo '{\"port\": 9999, \"host\": \"0.0.0.0\", \"max_connections\": -5, \"timeout\": 0}' | $AFM mlx -m $M -s 'Find problems with this JSON server config. One sentence.'"

# --- Pure stdin (no -s, model receives piped content directly) ---

run_test 11 "cat source | afm (stdin code review)" "stdin" \
  "head -30 $CD/Sources/MacLocalAPI/Models/OpenAIRequest.swift | $AFM mlx -m $M"

run_test 12 "cat Package.swift | afm (stdin dependency analysis)" "stdin" \
  "head -50 $CD/Package.swift | $AFM mlx -m $M"

# --- Output piping (afm output to other commands) ---

run_test 13 "afm -s | wc -w (count output words)" "output-pipe" \
  "$AFM mlx -m $M -s 'Write a haiku about coding' | wc -w"

run_test 14 "afm -s (generate python one-liner)" "output-pipe" \
  "$AFM mlx -m $M -s 'Write a Python one-liner that prints the sum of 1 to 100. Just the code, nothing else.'"

# --- Chained: afm | afm ---

run_test 15 "afm | afm (chained: generate then critique)" "chained" \
  "$AFM mlx -m $M -s 'Write a SQL query to find duplicate emails in a users table. Just the query.' | $AFM mlx -m $M -s 'Is this SQL correct? YES or NO with one sentence reason.'"

TEST_END=$(date +%s)
TOTAL_DURATION=$(( TEST_END - TEST_START ))

echo ""
echo "==========================================="
echo "RESULTS: $PASS passed, $FAIL failed out of $TOTAL"
echo "Total time: ${TOTAL_DURATION}s"
echo "==========================================="

# --- Generate HTML Report ---

escape_html() {
  sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g'
}

PASS_PCT=$(( PASS * 100 / TOTAL ))

cat > "$REPORT_FILE" << 'HTMLHEAD'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AFM CLI Piping Test Report</title>
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
  .stat.time .value { color: #58a6ff; }
  .stat.pct .value { color: #d2a8ff; }
  .progress-bar { width: 100%; height: 8px; background: #21262d; border-radius: 4px; margin: 1rem auto; max-width: 400px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
  table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; }
  th { background: #161b22; color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #30363d; }
  td { padding: 0.75rem 1rem; border-bottom: 1px solid #21262d; vertical-align: top; }
  tr:hover { background: #161b22; }
  .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  .badge.pass { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .badge.fail { background: #2d1215; color: #f85149; border: 1px solid #da3633; }
  .type-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px; font-size: 0.7rem; font-weight: 500; background: #1a1f2e; color: #8b949e; border: 1px solid #30363d; }
  .type-badge.pipe-flag { color: #58a6ff; border-color: #1f6feb; background: #0c2d6b33; }
  .type-badge.stdin { color: #d2a8ff; border-color: #8957e5; background: #3b1f7233; }
  .type-badge.output-pipe { color: #f0883e; border-color: #d18616; background: #4b2e0433; }
  .type-badge.chained { color: #3fb950; border-color: #238636; background: #0d281833; }
  .output { font-family: 'SF Mono', 'Menlo', monospace; font-size: 0.8rem; color: #8b949e; white-space: pre-wrap; word-break: break-word; max-height: 150px; overflow-y: auto; background: #0d1117; padding: 0.5rem; border-radius: 6px; border: 1px solid #21262d; margin-top: 0.25rem; }
  .duration { color: #8b949e; font-family: 'SF Mono', monospace; font-size: 0.85rem; }
  .footer { text-align: center; margin-top: 2rem; color: #484f58; font-size: 0.8rem; }
</style>
</head>
<body>
HTMLHEAD

# Header section
cat >> "$REPORT_FILE" << EOF
<div class="header">
  <h1>AFM MLX CLI Piping Test Report</h1>
  <div class="meta">
    Model: <strong>$M</strong><br>
    Binary: <code>$AFM</code><br>
    Date: $(date '+%Y-%m-%d %H:%M:%S')
  </div>
</div>
EOF

# Summary stats
if [ "$PASS" -eq "$TOTAL" ]; then
  BAR_COLOR="#3fb950"
elif [ "$FAIL" -gt $(( TOTAL / 2 )) ]; then
  BAR_COLOR="#f85149"
else
  BAR_COLOR="#d29922"
fi

cat >> "$REPORT_FILE" << EOF
<div class="summary">
  <div class="stat pass"><div class="value">$PASS</div><div class="label">Passed</div></div>
  <div class="stat fail"><div class="value">$FAIL</div><div class="label">Failed</div></div>
  <div class="stat pct"><div class="value">${PASS_PCT}%</div><div class="label">Pass Rate</div></div>
  <div class="stat time"><div class="value">${TOTAL_DURATION}s</div><div class="label">Total Time</div></div>
</div>
<div class="progress-bar"><div class="progress-fill" style="width:${PASS_PCT}%;background:${BAR_COLOR};"></div></div>
EOF

# Table header
cat >> "$REPORT_FILE" << 'TABLEHEAD'
<table>
<thead>
<tr><th>#</th><th>Test</th><th>Type</th><th>Status</th><th>Duration</th><th>Output</th></tr>
</thead>
<tbody>
TABLEHEAD

# Table rows
for i in $(seq 1 $TOTAL); do
  status="${TEST_STATUSES[$i]:-FAIL}"
  desc=$(echo "${TEST_DESCS[$i]:-}" | escape_html)
  raw_type="${TEST_TYPES[$i]:-}"
  output=$(echo "${TEST_OUTPUTS[$i]:-}" | escape_html | head -10)
  duration="${TEST_DURATIONS[$i]:-0}"

  if [ "$status" = "PASS" ]; then
    badge_class="pass"
  else
    badge_class="fail"
  fi

  case "$raw_type" in
    pipe+flag)   type_class="pipe-flag"; type_label="pipe + -s" ;;
    stdin)       type_class="stdin"; type_label="stdin" ;;
    output-pipe) type_class="output-pipe"; type_label="output pipe" ;;
    chained)     type_class="chained"; type_label="chained" ;;
    *)           type_class=""; type_label="$raw_type" ;;
  esac

  cat >> "$REPORT_FILE" << EOF
<tr>
  <td><strong>$i</strong></td>
  <td>$desc</td>
  <td><span class="type-badge $type_class">$type_label</span></td>
  <td><span class="badge $badge_class">$status</span></td>
  <td><span class="duration">${duration}s</span></td>
  <td><div class="output">$output</div></td>
</tr>
EOF
done

# Close table and footer
cat >> "$REPORT_FILE" << EOF
</tbody>
</table>
<div class="footer">
  Generated by <strong>afm-cli-tests.sh</strong> &mdash; $(date '+%Y-%m-%d %H:%M:%S')
</div>
</body>
</html>
EOF

echo ""
echo "HTML report: $REPORT_FILE"
