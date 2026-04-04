#!/bin/bash
# Multi-model assertion test runner for AFM MLX server.
# Runs test-assertions.sh against multiple models, optionally with forced parser,
# and generates a combined HTML report.
#
# Usage:
#   ./Scripts/test-assertions-multi.sh --models "model1,model2" [--tier standard] [--port 9998]
#   ./Scripts/test-assertions-multi.sh --models "org/model" --parser qwen3_xml
#   ./Scripts/test-assertions-multi.sh --models "m1,m2" --also-forced-parser qwen3_xml
#
# Options:
#   --models MODEL1,MODEL2,...   Comma-separated model IDs to test
#   --tier unit|smoke|standard|full   Test tier (default: standard)
#   --port PORT                  Base port (default: 9998)
#   --bin PATH                   Path to afm binary (default: .build/release/afm)
#   --parser PARSER              Force all tests to use this --tool-call-parser
#   --also-forced-parser PARSER  After auto-detect tests, re-test each model with forced parser
#   --keep-server                Don't stop the last server after tests
#
# Environment:
#   MACAFM_MLX_MODEL_CACHE   Model cache directory (default: /Volumes/edata/models/vesta-test-cache)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/test-reports"
CACHE_DIR="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"

MODELS=""
TIER="standard"
PORT=9998
BIN=".build/release/afm"
FORCED_PARSER=""
ALSO_FORCED_PARSER=""
KEEP_SERVER=false
GRAMMAR_CONSTRAINTS=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --models) MODELS="$2"; shift 2 ;;
    --tier) TIER="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --parser) FORCED_PARSER="$2"; shift 2 ;;
    --also-forced-parser) ALSO_FORCED_PARSER="$2"; shift 2 ;;
    --keep-server) KEEP_SERVER=true; shift ;;
    --grammar-constraints) GRAMMAR_CONSTRAINTS=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -z "$MODELS" ]; then
  echo "ERROR: --models is required"
  echo "Usage: $0 --models 'model1,model2' [--tier standard] [--also-forced-parser qwen3_xml]"
  exit 1
fi

mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
COMBINED_REPORT="$REPORT_DIR/multi-assertions-report-${TIMESTAMP}.html"
SUMMARY_JSONL="$REPORT_DIR/multi-assertions-report-${TIMESTAMP}.jsonl"

# Split models into array
IFS=',' read -ra MODEL_LIST <<< "$MODELS"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AFM Multi-Model Assertion Tests"
echo "  Models: ${#MODEL_LIST[@]} | Tier: $TIER | Port: $PORT"
if [ -n "$ALSO_FORCED_PARSER" ]; then
  echo "  Also testing with forced parser: $ALSO_FORCED_PARSER"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SERVER_PID=""

stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 2
  fi
}

start_server() {
  local model="$1"
  local extra_args="${2:-}"

  stop_server

  echo "  Starting server: $model $extra_args"
  MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" $BIN mlx -m "$model" --port "$PORT" $extra_args &
  SERVER_PID=$!

  # Wait for server to be ready
  local deadline=$((SECONDS + 120))
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if [ $SECONDS -ge $deadline ]; then
      echo "  ERROR: Server failed to start within 120s"
      return 1
    fi
    sleep 1
  done
  echo "  Server ready (PID $SERVER_PID)"
}

trap 'stop_server' EXIT

# Collect results: array of (model, parser_mode, report_file, pass, fail, skip, total)
declare -a RUN_RESULTS=()

run_one_config() {
  local model="$1"
  local parser_mode="$2"  # "auto" or parser name
  local extra_server_args="$3"

  echo ""
  echo "╔══════════════════════════════════════════════════════════"
  echo "║ Model: $model"
  echo "║ Parser: $parser_mode"
  echo "╚══════════════════════════════════════════════════════════"

  if ! start_server "$model" "$extra_server_args"; then
    RUN_RESULTS+=("$model|$parser_mode|FAIL|0|1|0|1|server_start_failed")
    return
  fi

  # Run assertions — capture output
  local run_output
  local grammar_flag=""
  if [ "$GRAMMAR_CONSTRAINTS" = true ]; then
    grammar_flag="--grammar-constraints"
  fi
  run_output=$("$SCRIPT_DIR/test-assertions.sh" --tier "$TIER" --model "$model" --port "$PORT" --bin "$BIN" $grammar_flag 2>&1) || true
  echo "$run_output"

  # Extract the report file path
  local report_file
  report_file=$(echo "$run_output" | grep "Report:" | tail -1 | awk '{print $NF}')
  local jsonl_file
  jsonl_file=$(echo "$run_output" | grep "JSONL:" | tail -1 | awk '{print $NF}')

  # Extract pass/fail counts (strip ANSI codes first)
  local clean_output
  clean_output=$(echo "$run_output" | sed $'s/\x1b\\[[0-9;]*m//g')
  local results_line
  results_line=$(echo "$clean_output" | grep "Results:" | tail -1)
  local pass_count fail_count skip_count total_count
  # Parse "Results: 46/51 passed (90%) | 2 skipped"
  pass_count=$(echo "$results_line" | sed -n 's/.*Results: *\([0-9]*\)\/.*/\1/p')
  total_count=$(echo "$results_line" | sed -n 's/.*Results: *[0-9]*\/\([0-9]*\) .*/\1/p')
  fail_count=$(echo "$clean_output" | grep -cE "^  ❌" || echo "0")
  skip_count=$(echo "$clean_output" | grep -cE "SKIP" || echo "0")

  # Append JSONL to combined file with model/parser tag
  if [ -f "$jsonl_file" ]; then
    while IFS= read -r line; do
      echo "{\"_run\":{\"model\":\"$model\",\"parser\":\"$parser_mode\"},\"data\":$line}" >> "$SUMMARY_JSONL"
    done < "$jsonl_file"
  fi

  RUN_RESULTS+=("$model|$parser_mode|$report_file|${pass_count:-0}|${fail_count:-0}|${skip_count:-0}|${total_count:-0}")
}

# Build grammar flag for server start
GRAMMAR_SERVER_FLAG=""
if [ "$GRAMMAR_CONSTRAINTS" = true ]; then
  GRAMMAR_SERVER_FLAG="--enable-grammar-constraints"
fi

# Run all configurations
for model in "${MODEL_LIST[@]}"; do
  model=$(echo "$model" | xargs)  # trim whitespace

  if [ -n "$FORCED_PARSER" ]; then
    run_one_config "$model" "$FORCED_PARSER" "--tool-call-parser $FORCED_PARSER --enable-prefix-caching $GRAMMAR_SERVER_FLAG"
  else
    # Auto-detect
    run_one_config "$model" "auto" "--enable-prefix-caching $GRAMMAR_SERVER_FLAG"

    # Also forced parser if requested
    if [ -n "$ALSO_FORCED_PARSER" ]; then
      run_one_config "$model" "$ALSO_FORCED_PARSER" "--tool-call-parser $ALSO_FORCED_PARSER $GRAMMAR_SERVER_FLAG"
    fi
  fi
done

if ! $KEEP_SERVER; then
  stop_server
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Generate Combined HTML Report
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_TESTS=0

# Build table rows
TABLE_ROWS=""
for result in "${RUN_RESULTS[@]}"; do
  IFS='|' read -r r_model r_parser r_report r_pass r_fail r_skip r_total <<< "$result"
  TOTAL_PASS=$((TOTAL_PASS + r_pass))
  TOTAL_FAIL=$((TOTAL_FAIL + r_fail))
  TOTAL_TESTS=$((TOTAL_TESTS + r_total))

  if [ "$r_fail" = "0" ]; then
    status_class="pass"
    status_icon="✅"
  else
    status_class="fail"
    status_icon="❌"
  fi

  # Link to individual report if it exists
  local_report=""
  if [ -f "$r_report" ]; then
    local_report="<a href=\"$(basename "$r_report")\">[details]</a>"
  fi

  TABLE_ROWS+="<tr class=\"$status_class\">
    <td class=\"model\">$r_model</td>
    <td>$r_parser</td>
    <td class=\"pass-count\">$r_pass</td>
    <td class=\"fail-count\">$r_fail</td>
    <td>$r_skip</td>
    <td>$r_total</td>
    <td>$status_icon $local_report</td>
  </tr>"
done

# Parse individual report details for the combined view
DETAIL_SECTIONS=""
for result in "${RUN_RESULTS[@]}"; do
  IFS='|' read -r r_model r_parser r_report r_pass r_fail r_skip r_total <<< "$result"

  if [ -f "$r_report" ]; then
    # Extract the test results table from the individual HTML report
    results_body=$(sed -n '/<tbody>/,/<\/tbody>/p' "$r_report" 2>/dev/null || echo "")
  else
    results_body="<tr><td colspan='5'>Report not available</td></tr>"
  fi

  DETAIL_SECTIONS+="
  <div class=\"model-section\">
    <h2>$r_model <span class=\"parser-badge\">$r_parser</span></h2>
    <div class=\"summary-bar\">
      <span class=\"pass-count\">$r_pass passed</span> /
      <span class=\"fail-count\">$r_fail failed</span> /
      $r_total total
    </div>
    <table class=\"results-table\">
      <thead><tr><th>#</th><th>Test</th><th>Group</th><th>Coverage</th><th>Status</th><th>Duration</th><th>Details</th></tr></thead>
      $results_body
    </table>
  </div>"
done

OVERALL_PCT=0
if [ $TOTAL_TESTS -gt 0 ]; then
  OVERALL_PCT=$((TOTAL_PASS * 100 / TOTAL_TESTS))
fi

cat > "$COMBINED_REPORT" <<HTMLEOF
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AFM Multi-Model Assertion Report — $(date '+%Y-%m-%d %H:%M')</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 24px; }
  h1 { color: #58a6ff; margin-bottom: 8px; font-size: 1.5em; }
  .meta { color: #8b949e; margin-bottom: 24px; font-size: 0.9em; }
  .overall { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
             padding: 16px 24px; margin-bottom: 24px; }
  .overall .pct { font-size: 2em; font-weight: bold; }
  .overall .pct.good { color: #3fb950; }
  .overall .pct.warn { color: #d29922; }
  .overall .pct.bad { color: #f85149; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }
  th { background: #161b22; color: #8b949e; text-align: left; padding: 8px 12px;
       border-bottom: 2px solid #30363d; font-size: 0.85em; text-transform: uppercase; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 0.9em; }
  tr:hover { background: #161b22; }
  tr.pass td:first-child { border-left: 3px solid #3fb950; }
  tr.fail td:first-child { border-left: 3px solid #f85149; }
  .model { font-family: 'SF Mono', monospace; font-weight: 600; }
  .pass-count { color: #3fb950; font-weight: bold; }
  .fail-count { color: #f85149; font-weight: bold; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .model-section { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                   padding: 16px 24px; margin-bottom: 20px; }
  .model-section h2 { color: #c9d1d9; font-size: 1.2em; margin-bottom: 8px; }
  .parser-badge { background: #1f6feb33; color: #58a6ff; padding: 2px 8px;
                  border-radius: 4px; font-size: 0.7em; font-weight: normal; }
  .summary-bar { color: #8b949e; margin-bottom: 12px; font-size: 0.9em; }
  .results-table td.PASS { color: #3fb950; }
  .results-table td.FAIL { color: #f85149; }
  .results-table td.SKIP { color: #d29922; }
  .group-badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
                 border: 1px solid; font-size: 0.8em; font-weight: 600; }
  .tier-row td { background: #1c2129; padding: 6px 12px; font-weight: 700; font-size: 0.9em;
                 border-bottom: 2px solid #30363d; border-top: 2px solid #30363d; }
  .tier-badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.7em; font-weight: 600; }
  .tier-badge.unit { background: #1a1a2e; color: #a5d6ff; border: 1px solid #58a6ff; }
  .tier-badge.smoke { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .tier-badge.standard { background: #0d1a30; color: #58a6ff; border: 1px solid #1f6feb; }
  .tier-badge.full { background: #2d1f00; color: #d29922; border: 1px solid #9e6a03; }
  .badge { display: inline-block; padding: 1px 8px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }
  .badge.pass { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .badge.fail { background: #2d1215; color: #f85149; border: 1px solid #da3633; }
  .badge.skip { background: #2d2400; color: #d29922; border: 1px solid #9e6a03; }
  .test-idx { color: #8b949e; font-family: 'SF Mono', monospace; font-size: 0.85em; }
</style>
</head>
<body>
<h1>AFM Multi-Model Assertion Report</h1>
<div class="meta">
  Generated: $(date '+%Y-%m-%d %H:%M:%S') | Tier: $TIER | Models: ${#MODEL_LIST[@]} configurations<br>
  Binary: $BIN | Cache: $CACHE_DIR
</div>

<div class="overall">
  <span class="pct $([ $OVERALL_PCT -ge 95 ] && echo good || ([ $OVERALL_PCT -ge 80 ] && echo warn || echo bad))">$OVERALL_PCT%</span>
  &nbsp; $TOTAL_PASS / $TOTAL_TESTS passed across all models
  $([ $TOTAL_FAIL -gt 0 ] && echo "— <span class=\"fail-count\">$TOTAL_FAIL failures</span>" || echo "")
</div>

<h2 style="color:#c9d1d9; margin-bottom: 12px;">Summary</h2>
<table>
  <thead>
    <tr><th>Model</th><th>Parser</th><th>Pass</th><th>Fail</th><th>Skip</th><th>Total</th><th>Status</th></tr>
  </thead>
  <tbody>
    $TABLE_ROWS
  </tbody>
</table>

<h2 style="color:#c9d1d9; margin-bottom: 12px;">Detailed Results</h2>
$DETAIL_SECTIONS

</body>
</html>
HTMLEOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Multi-Model Results: $TOTAL_PASS/$TOTAL_TESTS passed ($OVERALL_PCT%)"
echo "  Combined Report: $COMBINED_REPORT"
echo "  JSONL: $SUMMARY_JSONL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
