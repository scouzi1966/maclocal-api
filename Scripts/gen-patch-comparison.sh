#!/bin/bash
# Generates an MLX patch comparison report (HTML + CSV).
# Compares maclocal-api patches against upstream mlx-swift-lm at three refs:
#   1. Pinned tag (matching vendor submodule)
#   2. Latest semver tag
#   3. Main branch HEAD
#
# Usage:
#   ./Scripts/gen-patch-comparison.sh [--output-dir DIR] [--tmp-dir DIR]
#
# Requires: gh (GitHub CLI), git, diff

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PATCHES_DIR="$SCRIPT_DIR/patches"

# Defaults
OUTPUT_DIR="$PROJECT_ROOT/test-reports"
TMP_DIR="/tmp/mlx-upstream"
TODAY=$(date +%Y%m%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $1"; }

# --- CLI args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --tmp-dir)    TMP_DIR="$2"; shift 2 ;;
    -h|--help)
      cat << 'HELPEOF'
---
name: gen-patch-comparison
description: Generate an HTML + CSV report comparing maclocal-api vendor patches against upstream mlx-swift-lm at three git refs (pinned tag, latest tag, main branch HEAD). Detects upstream-only files not covered by patches. Use when auditing patch drift, planning an upstream merge or upgrade, or checking for new upstream models/features to adopt.
tags: [mlx, patches, vendor, upstream, comparison, report, audit, upgrade, merge, drift]
inputs:
  --output-dir DIR    Directory for generated reports (default: test-reports/)
  --tmp-dir DIR       Directory for upstream repo clones (default: /tmp/mlx-upstream)
outputs:
  - mlx-patch-comparison-TIMESTAMP.html   Interactive HTML report
  - mlx-patch-comparison-TIMESTAMP.csv    Machine-readable CSV
requires: [gh, git, diff]
triggers:
  - patch comparison report
  - upstream drift analysis
  - vendor upgrade audit
  - check for new upstream models
  - mlx-swift-lm merge planning
examples:
  - ./Scripts/gen-patch-comparison.sh
  - ./Scripts/gen-patch-comparison.sh --output-dir /tmp/reports
---

Usage: gen-patch-comparison.sh [--output-dir DIR] [--tmp-dir DIR]

  Generate MLX patch comparison report (HTML + CSV).
  Compares maclocal-api patches against upstream mlx-swift-lm at three refs:
    1. Pinned tag (matching vendor submodule, e.g. 2.30.3)
    2. Latest semver tag (e.g. 2.30.6)
    3. Main branch HEAD

  Reports include:
    - Per-file diff stats and conflict risk classification
    - Upstream-only files not covered by maclocal-api patches
    - mlx-swift framework changelog between refs
    - maclocal-api commit SHA and all upstream ref SHAs

Options:
  --output-dir DIR    Output directory for reports (default: test-reports/)
  --tmp-dir DIR       Temp directory for upstream clones (default: /tmp/mlx-upstream)
  -h, --help          Show this help message

Requires: gh (GitHub CLI), git, diff
HELPEOF
      exit 0 ;;
    *) log_error "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR" "$TMP_DIR"

# ============================================================
# Step 1: Capture maclocal-api state
# ============================================================
log_step "Capturing maclocal-api state"

MACLOCAL_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)
MACLOCAL_SHA_LONG=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
MACLOCAL_BRANCH=$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD)
MACLOCAL_DATE=$(git -C "$PROJECT_ROOT" log -1 --format=%ci | cut -d' ' -f1)

# Get submodule pin info
SUBMODULE_SHA=$(git -C "$PROJECT_ROOT" submodule status vendor/mlx-swift-lm | awk '{print $1}' | tr -d '+')
SUBMODULE_DESC=$(git -C "$PROJECT_ROOT/vendor/mlx-swift-lm" describe --tags 2>/dev/null || echo "$SUBMODULE_SHA")

# Extract pinned mlx-swift version from Package.swift
MLX_SWIFT_PIN=$(grep -o 'exact: "[^"]*"' "$PROJECT_ROOT/vendor/mlx-swift-lm/Package.swift" 2>/dev/null | head -1 | grep -o '"[^"]*"' | tr -d '"' || echo "unknown")

log_info "maclocal-api: $MACLOCAL_SHA ($MACLOCAL_BRANCH) from $MACLOCAL_DATE"
log_info "Submodule pin: $SUBMODULE_DESC (sha: ${SUBMODULE_SHA:0:7})"
log_info "mlx-swift pin: $MLX_SWIFT_PIN"

# ============================================================
# Step 2: Clone or update upstream repos
# ============================================================
log_step "Updating upstream repos"

clone_or_update() {
  local repo="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    log_info "Updating $repo"
    git -C "$dir" fetch --tags --quiet origin 2>/dev/null || true
    git -C "$dir" fetch --quiet origin main 2>/dev/null || true
  else
    log_info "Cloning $repo"
    gh repo clone "$repo" "$dir" -- --quiet 2>/dev/null
  fi
}

MLX_LM_UPSTREAM="$TMP_DIR/mlx-swift-lm"
MLX_SWIFT_UPSTREAM="$TMP_DIR/mlx-swift"

clone_or_update "ml-explore/mlx-swift-lm" "$MLX_LM_UPSTREAM"
clone_or_update "ml-explore/mlx-swift" "$MLX_SWIFT_UPSTREAM"

# ============================================================
# Step 3: Resolve three upstream refs per repo
# ============================================================
log_step "Resolving upstream refs"

# Extract base tag from submodule describe (e.g. "2.30.3" from "2.30.3-7-g2c70054")
PINNED_TAG=$(echo "$SUBMODULE_DESC" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+' || echo "")
if [ -z "$PINNED_TAG" ]; then
  log_warn "Could not parse pinned tag from '$SUBMODULE_DESC', using submodule SHA"
  PINNED_TAG="$SUBMODULE_SHA"
fi

# Latest semver tag for mlx-swift-lm
LM_LATEST_TAG=$(git -C "$MLX_LM_UPSTREAM" tag --sort=-v:refname | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
LM_MAIN_SHA=$(git -C "$MLX_LM_UPSTREAM" rev-parse --short origin/main)
LM_MAIN_SHA_LONG=$(git -C "$MLX_LM_UPSTREAM" rev-parse origin/main)

# mlx-swift refs
MLX_SWIFT_LATEST_TAG=$(git -C "$MLX_SWIFT_UPSTREAM" tag --sort=-v:refname | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
MLX_SWIFT_MAIN_SHA=$(git -C "$MLX_SWIFT_UPSTREAM" rev-parse --short origin/main)

# Commit counts between refs
LM_PINNED_TO_LATEST=$(git -C "$MLX_LM_UPSTREAM" rev-list --count "$PINNED_TAG..$LM_LATEST_TAG" 2>/dev/null || echo "?")
LM_LATEST_TO_MAIN=$(git -C "$MLX_LM_UPSTREAM" rev-list --count "$LM_LATEST_TAG..origin/main" 2>/dev/null || echo "?")
MLX_PINNED_TO_LATEST=$(git -C "$MLX_SWIFT_UPSTREAM" rev-list --count "$MLX_SWIFT_PIN..$MLX_SWIFT_LATEST_TAG" 2>/dev/null || echo "?")
MLX_LATEST_TO_MAIN=$(git -C "$MLX_SWIFT_UPSTREAM" rev-list --count "$MLX_SWIFT_LATEST_TAG..origin/main" 2>/dev/null || echo "?")

log_info "mlx-swift-lm: pinned=$PINNED_TAG, latest=$LM_LATEST_TAG, main=$LM_MAIN_SHA"
log_info "  pinned→latest: $LM_PINNED_TO_LATEST commits, latest→main: $LM_LATEST_TO_MAIN commits"
log_info "mlx-swift: pin=$MLX_SWIFT_PIN, latest=$MLX_SWIFT_LATEST_TAG, main=$MLX_SWIFT_MAIN_SHA"
log_info "  pinned→latest: $MLX_PINNED_TO_LATEST commits, latest→main: $MLX_LATEST_TO_MAIN commits"

# ============================================================
# Step 4: Read patch arrays from apply-mlx-patches.sh
# ============================================================
log_step "Reading patch configuration"

# Source the arrays by extracting them
eval "$(grep -E '^(PATCH_FILES|TARGET_PATHS|NEW_FILES)=' "$SCRIPT_DIR/apply-mlx-patches.sh")"

NUM_PATCHES=${#PATCH_FILES[@]}
log_info "Found $NUM_PATCHES patch files"

is_new_file() {
  local filename="$1"
  for nf in "${NEW_FILES[@]}"; do
    [[ "$nf" == "$filename" ]] && return 0
  done
  return 1
}

# ============================================================
# Feature descriptions (qualitative — update when patches change)
# bash 3 compatible: parallel arrays instead of associative array
# ============================================================
get_feature_desc() {
  local name="$1"
  case "$name" in
    Qwen3_5MoE.swift)      echo "Fused QKV/GDN/gate+up projections; fusedSiluMul Metal kernel; MLXFast.rmsNorm; adaptive sanitize" ;;
    Evaluate.swift)         echo "TopK/MinP/Presence/Composite processors; seed; logprobs; AFM_PERF; clearCache; shouldQuantizeCache" ;;
    KVCache.swift)          echo "2x growth strategy; Updatable protocol; MXFP4 mode parameter" ;;
    SwitchLayers.swift)     echo "fusedSiluMul Metal kernel; SwitchGLU gate+up fusion; memory headroom check" ;;
    LLMModelFactory.swift)  echo "qwen3_5_moe/kimi_k2/kimi_k25/joyai_llm_flash/glm_moe_dsa/minimax_m2; sanitizeJSON(); toolCallFormat removal" ;;
    ToolCallFormat.swift)   echo "Qwen3 model type detection (qwen3_next, qwen3_coder)" ;;
    Tokenizer.swift)        echo "TikToken support; .jinja fallback; O(n) window trim" ;;
    Load.swift)             echo ".jinja/.tiktoken downloads; MXFP4 fix; relaxed weight verification" ;;
    VLMModelFactory.swift)  echo "Qwen3.5 MoE VL registration" ;;
    Qwen3VL.swift)          echo "Qwen3 VL patches" ;;
    Qwen3Next.swift)        echo "GatedDeltaNet model architecture" ;;
    GatedDelta.swift)       echo "GatedDeltaNet attention layer" ;;
    DeepseekV3.swift)       echo "Bug fixes; Kimi K2 reuse" ;;
    MiniMaxM2.swift)        echo "MiniMax M2 model architecture" ;;
    NemotronH.swift)        echo "NemotronH hybrid Mamba+Attention model" ;;
    GLM4MoeLite.swift)      echo "GLM-4 MoE Lite model" ;;
    GLM5MoeDsa.swift)       echo "GLM-5 MoE DSA model" ;;
    KimiK25.swift)          echo "Kimi K2.5 model architecture" ;;
    Qwen3_5MoEVL.swift)     echo "Qwen3.5 MoE VLM wrapper" ;;
    SamplerTests.swift)     echo "Unit tests for sampling processors" ;;
    *)                      echo "" ;;
  esac
}

# ============================================================
# Step 5: Per-file comparison
# ============================================================
log_step "Comparing files"

# Arrays to collect results
declare -a R_FILE R_TARGET R_TYPE R_MACLOCAL_LINES
declare -a R_PINNED_LINES R_PINNED_STATUS R_PINNED_ADDED R_PINNED_REMOVED
declare -a R_LATEST_LINES R_LATEST_STATUS R_LATEST_ADDED R_LATEST_REMOVED
declare -a R_MAIN_LINES R_MAIN_STATUS R_MAIN_ADDED R_MAIN_REMOVED
declare -a R_RISK R_FEATURES

get_file_at_ref() {
  local repo_dir="$1" ref="$2" path="$3"
  git -C "$repo_dir" show "$ref:$path" 2>/dev/null || echo ""
}

count_lines() {
  local content="$1"
  if [ -z "$content" ]; then
    echo "0"
  else
    echo "$content" | wc -l | tr -d ' '
  fi
}

diff_stats() {
  local file_a="$1" file_b="$2"
  # Returns "added removed" line counts
  if [ ! -s "$file_a" ] && [ ! -s "$file_b" ]; then
    echo "0 0"
  elif [ ! -s "$file_a" ]; then
    echo "$(wc -l < "$file_b" | tr -d ' ') 0"
  elif [ ! -s "$file_b" ]; then
    echo "0 $(wc -l < "$file_a" | tr -d ' ')"
  else
    local added removed
    added=$(diff "$file_b" "$file_a" 2>/dev/null | grep -c '^>' || true)
    removed=$(diff "$file_b" "$file_a" 2>/dev/null | grep -c '^<' || true)
    echo "$added $removed"
  fi
}

classify_status() {
  local maclocal_lines="$1" upstream_lines="$2" added="$3" removed="$4" is_new="$5"
  if [ "$upstream_lines" -eq 0 ]; then
    if [ "$is_new" = "true" ]; then
      echo "NEW"
    else
      echo "ABSENT"
    fi
  elif [ "$added" -eq 0 ] && [ "$removed" -eq 0 ]; then
    echo "IDENTICAL"
  else
    echo "CHANGED"
  fi
}

classify_risk() {
  local is_new="$1" pinned_lines="$2" latest_lines="$3" main_lines="$4"
  local upstream_pinned_to_latest_changed="$5" upstream_latest_to_main_changed="$6"
  local maclocal_added_vs_pinned="$7"

  # NEW files with no upstream equivalent at any ref: NONE
  if [ "$is_new" = "true" ] && [ "$pinned_lines" -eq 0 ] && [ "$latest_lines" -eq 0 ] && [ "$main_lines" -eq 0 ]; then
    echo "NONE"
    return
  fi
  # NEW file that now exists upstream: LOW
  if [ "$is_new" = "true" ]; then
    if [ "$latest_lines" -gt 0 ] || [ "$main_lines" -gt 0 ]; then
      echo "LOW"
    else
      echo "NONE"
    fi
    return
  fi

  # For MODIFIED files: risk depends on whether UPSTREAM changed between pinned and latest/main
  # upstream_pinned_to_latest_changed: lines changed in upstream between pinned tag and latest tag
  # upstream_latest_to_main_changed: lines changed in upstream between latest tag and main

  local total_upstream_change=$((upstream_pinned_to_latest_changed + upstream_latest_to_main_changed))

  if [ "$total_upstream_change" -eq 0 ]; then
    # Upstream didn't change this file at all — our patch applies identically
    echo "NONE"
    return
  fi

  # Upstream changed. Risk depends on magnitude relative to our changes
  if [ "$total_upstream_change" -lt 20 ]; then
    echo "LOW"
  elif [ "$total_upstream_change" -lt 100 ]; then
    if [ "$maclocal_added_vs_pinned" -gt 200 ]; then
      echo "MEDIUM"
    else
      echo "LOW"
    fi
  else
    # Large upstream changes + we have modifications = HIGH
    if [ "$maclocal_added_vs_pinned" -gt 100 ]; then
      echo "HIGH"
    else
      echo "MEDIUM"
    fi
  fi
}

SUMMARY_TOTAL=0
SUMMARY_NONE=0
SUMMARY_LOW=0
SUMMARY_MEDIUM=0
SUMMARY_HIGH=0
SUMMARY_NEW=0

for i in "${!PATCH_FILES[@]}"; do
  patch_name="${PATCH_FILES[$i]}"
  target_path="${TARGET_PATHS[$i]}"
  patch_file="$PATCHES_DIR/$patch_name"

  is_new="false"
  is_new_file "$patch_name" && is_new="true"

  # Target library (extract from path)
  target_lib=$(echo "$target_path" | sed 's|Libraries/||' | cut -d'/' -f1)

  # maclocal-api patch line count
  if [ -f "$patch_file" ]; then
    maclocal_lines=$(wc -l < "$patch_file" | tr -d ' ')
  else
    maclocal_lines=0
    log_warn "Patch file not found: $patch_name"
  fi

  # Get upstream files at each ref
  pinned_content=$(get_file_at_ref "$MLX_LM_UPSTREAM" "$PINNED_TAG" "$target_path")
  latest_content=$(get_file_at_ref "$MLX_LM_UPSTREAM" "$LM_LATEST_TAG" "$target_path")
  main_content=$(get_file_at_ref "$MLX_LM_UPSTREAM" "origin/main" "$target_path")

  pinned_lines=$(count_lines "$pinned_content")
  latest_lines=$(count_lines "$latest_content")
  main_lines=$(count_lines "$main_content")

  # Write temp files for diff
  tmp_patch="$TMP_DIR/_patch.tmp"
  tmp_pinned="$TMP_DIR/_pinned.tmp"
  tmp_latest="$TMP_DIR/_latest.tmp"
  tmp_main="$TMP_DIR/_main.tmp"

  [ -f "$patch_file" ] && cp "$patch_file" "$tmp_patch" || : > "$tmp_patch"
  echo "$pinned_content" > "$tmp_pinned"
  echo "$latest_content" > "$tmp_latest"
  echo "$main_content" > "$tmp_main"
  # Fix: empty content produces a 1-line file with empty string
  [ "$pinned_lines" -eq 0 ] && : > "$tmp_pinned"
  [ "$latest_lines" -eq 0 ] && : > "$tmp_latest"
  [ "$main_lines" -eq 0 ] && : > "$tmp_main"

  # Diff stats: maclocal-api vs each upstream (added = in maclocal-api but not upstream)
  read pinned_added pinned_removed <<< "$(diff_stats "$tmp_patch" "$tmp_pinned")"
  read latest_added latest_removed <<< "$(diff_stats "$tmp_patch" "$tmp_latest")"
  read main_added main_removed <<< "$(diff_stats "$tmp_patch" "$tmp_main")"

  # Upstream-vs-upstream diffs (for conflict risk: did upstream change between refs?)
  read _ul_a _ul_r <<< "$(diff_stats "$tmp_latest" "$tmp_pinned")"
  upstream_pinned_to_latest=$((_ul_a + _ul_r))
  read _um_a _um_r <<< "$(diff_stats "$tmp_main" "$tmp_latest")"
  upstream_latest_to_main=$((_um_a + _um_r))

  # Classify status (maclocal-api patch vs each upstream ref)
  pinned_status=$(classify_status "$maclocal_lines" "$pinned_lines" "$pinned_added" "$pinned_removed" "$is_new")
  latest_status=$(classify_status "$maclocal_lines" "$latest_lines" "$latest_added" "$latest_removed" "$is_new")
  main_status=$(classify_status "$maclocal_lines" "$main_lines" "$main_added" "$main_removed" "$is_new")

  # Conflict risk based on upstream divergence, not our patch differences
  risk=$(classify_risk "$is_new" "$pinned_lines" "$latest_lines" "$main_lines" \
    "$upstream_pinned_to_latest" "$upstream_latest_to_main" "$pinned_added")

  # File type
  if [ "$is_new" = "true" ]; then
    file_type="NEW"
  else
    file_type="MODIFIED"
  fi

  features=$(get_feature_desc "$patch_name")

  # Store results
  R_FILE+=("$patch_name")
  R_TARGET+=("$target_lib")
  R_TYPE+=("$file_type")
  R_MACLOCAL_LINES+=("$maclocal_lines")
  R_PINNED_LINES+=("$pinned_lines")
  R_PINNED_STATUS+=("$pinned_status")
  R_PINNED_ADDED+=("$pinned_added")
  R_PINNED_REMOVED+=("$pinned_removed")
  R_LATEST_LINES+=("$latest_lines")
  R_LATEST_STATUS+=("$latest_status")
  R_LATEST_ADDED+=("$latest_added")
  R_LATEST_REMOVED+=("$latest_removed")
  R_MAIN_LINES+=("$main_lines")
  R_MAIN_STATUS+=("$main_status")
  R_MAIN_ADDED+=("$main_added")
  R_MAIN_REMOVED+=("$main_removed")
  R_RISK+=("$risk")
  R_FEATURES+=("$features")

  # Summary counts
  SUMMARY_TOTAL=$((SUMMARY_TOTAL + 1))
  [ "$is_new" = "true" ] && SUMMARY_NEW=$((SUMMARY_NEW + 1))
  case "$risk" in
    NONE)   SUMMARY_NONE=$((SUMMARY_NONE + 1)) ;;
    LOW*)   SUMMARY_LOW=$((SUMMARY_LOW + 1)) ;;
    MEDIUM*) SUMMARY_MEDIUM=$((SUMMARY_MEDIUM + 1)) ;;
    HIGH*)  SUMMARY_HIGH=$((SUMMARY_HIGH + 1)) ;;
  esac

  log_info "$patch_name: pinned=$pinned_status latest=$latest_status main=$main_status risk=$risk"
done

# Clean temp files
rm -f "$TMP_DIR"/_*.tmp

# ============================================================
# Step 5b: Detect upstream-only files (not in maclocal-api patches)
# ============================================================
log_step "Detecting upstream-only files"

# Directories to scan in upstream mlx-swift-lm
SCAN_DIRS=(
  "Libraries/MLXLLM/Models"
  "Libraries/MLXVLM/Models"
  "Libraries/MLXLMCommon"
  "Libraries/MLXLMCommon/Tool"
)

# Build set of patched target paths for fast lookup
patched_targets=""
for tp in "${TARGET_PATHS[@]}"; do
  patched_targets="$patched_targets|$tp"
done

# Arrays for upstream-only files
U_FILE=()
U_DIR=()
U_PINNED_LINES=()
U_LATEST_LINES=()
U_MAIN_LINES=()
U_SINCE=()  # "pinned", "latest", or "main" — when it first appeared

for scan_dir in "${SCAN_DIRS[@]}"; do
  # Get file lists at each ref
  pinned_files=$(git -C "$MLX_LM_UPSTREAM" ls-tree --name-only "$PINNED_TAG" "$scan_dir/" 2>/dev/null | grep '\.swift$' || true)
  latest_files=$(git -C "$MLX_LM_UPSTREAM" ls-tree --name-only "$LM_LATEST_TAG" "$scan_dir/" 2>/dev/null | grep '\.swift$' || true)
  main_files=$(git -C "$MLX_LM_UPSTREAM" ls-tree --name-only "origin/main" "$scan_dir/" 2>/dev/null | grep '\.swift$' || true)

  # Combine all files across refs
  all_files=$(echo -e "${pinned_files}\n${latest_files}\n${main_files}" | sort -u | grep -v '^$' || true)

  while IFS= read -r filepath; do
    [ -z "$filepath" ] && continue
    # Skip if this file is in our patch set
    if echo "$patched_targets" | grep -qF "|$filepath"; then
      continue
    fi

    # Count lines at each ref
    p_content=$(get_file_at_ref "$MLX_LM_UPSTREAM" "$PINNED_TAG" "$filepath")
    l_content=$(get_file_at_ref "$MLX_LM_UPSTREAM" "$LM_LATEST_TAG" "$filepath")
    m_content=$(get_file_at_ref "$MLX_LM_UPSTREAM" "origin/main" "$filepath")
    p_lines=$(count_lines "$p_content")
    l_lines=$(count_lines "$l_content")
    m_lines=$(count_lines "$m_content")

    # Determine when it first appeared
    if [ "$p_lines" -gt 0 ]; then
      since="pinned"
    elif [ "$l_lines" -gt 0 ]; then
      since="latest"
    else
      since="main"
    fi

    fname=$(basename "$filepath")
    fdir=$(echo "$scan_dir" | sed 's|Libraries/||')

    U_FILE+=("$fname")
    U_DIR+=("$fdir")
    U_PINNED_LINES+=("$p_lines")
    U_LATEST_LINES+=("$l_lines")
    U_MAIN_LINES+=("$m_lines")
    U_SINCE+=("$since")
  done <<< "$all_files"
done

UPSTREAM_ONLY_COUNT=${#U_FILE[@]}
log_info "Found $UPSTREAM_ONLY_COUNT upstream-only files (not in maclocal-api patches)"

# ============================================================
# Step 6: Collect mlx-swift framework changes
# ============================================================
log_step "Collecting mlx-swift framework changes"

MLX_PINNED_TO_LATEST_LOG=$(git -C "$MLX_SWIFT_UPSTREAM" log --oneline "$MLX_SWIFT_PIN..$MLX_SWIFT_LATEST_TAG" 2>/dev/null || echo "(none)")
MLX_LATEST_TO_MAIN_LOG=$(git -C "$MLX_SWIFT_UPSTREAM" log --oneline "$MLX_SWIFT_LATEST_TAG..origin/main" 2>/dev/null || echo "(none)")

# ============================================================
# Step 7: Generate HTML
# ============================================================
log_step "Generating HTML report"

HTML_FILE="$OUTPUT_DIR/mlx-patch-comparison-${TIMESTAMP}.html"

badge_class() {
  case "$1" in
    NONE)      echo "none" ;;
    LOW*)      echo "low" ;;
    MEDIUM*)   echo "medium" ;;
    HIGH*)     echo "high" ;;
    NEW)       echo "new" ;;
    IDENTICAL) echo "unchanged" ;;
    CHANGED)   echo "changed" ;;
    ABSENT)    echo "unchanged" ;;
    MODIFIED)  echo "changed" ;;
    *)         echo "unchanged" ;;
  esac
}

status_cell() {
  local status="$1" lines="$2" added="$3" removed="$4"
  local cls
  cls=$(badge_class "$status")
  case "$status" in
    IDENTICAL)
      echo "<span class=\"badge unchanged\">IDENTICAL</span> ${lines}L" ;;
    CHANGED)
      echo "<span class=\"badge changed\">CHANGED</span> ${lines}L (+${added}/-${removed})" ;;
    NEW|ABSENT)
      if [ "$lines" -eq 0 ]; then
        echo "<span class=\"badge unchanged\">N/A</span>"
      else
        echo "<span class=\"badge new\">EXISTS</span> ${lines}L"
      fi ;;
    *)
      echo "$status" ;;
  esac
}

cat > "$HTML_FILE" << 'HTMLHEAD'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLX Patch Comparison: maclocal-api vs Upstream</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif; background: #0d1117; color: #e6edf3; padding: 2rem; }
  .header { text-align: center; margin-bottom: 2rem; padding: 2rem; background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%); border: 1px solid #30363d; border-radius: 12px; }
  .header h1 { font-size: 1.8rem; margin-bottom: 0.5rem; background: linear-gradient(90deg, #58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header .meta { color: #8b949e; font-size: 0.9rem; line-height: 1.8; }
  .header code { background: #21262d; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.85rem; color: #79c0ff; }

  .section { margin-bottom: 2rem; }
  .section h2 { font-size: 1.3rem; margin-bottom: 1rem; color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; }

  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
  .summary-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1.25rem; text-align: center; }
  .summary-card .value { font-size: 2rem; font-weight: 700; }
  .summary-card .label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
  .summary-card.safe .value { color: #3fb950; }
  .summary-card.risk .value { color: #f85149; }
  .summary-card.info .value { color: #58a6ff; }
  .summary-card.warn .value { color: #d29922; }

  table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
  th { background: #161b22; color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; padding: 0.75rem 0.6rem; text-align: left; border-bottom: 1px solid #30363d; }
  td { padding: 0.6rem; border-bottom: 1px solid #21262d; vertical-align: top; font-size: 0.85rem; }
  tr:hover { background: #161b22; }

  .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 12px; font-size: 0.72rem; font-weight: 600; white-space: nowrap; }
  .badge.none { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .badge.low { background: #2d2400; color: #d29922; border: 1px solid #9e6a03; }
  .badge.medium { background: #4b2e04; color: #ffa657; border: 1px solid #d18616; }
  .badge.high { background: #2d1215; color: #f85149; border: 1px solid #da3633; }
  .badge.new { background: #0c2d6b; color: #58a6ff; border: 1px solid #1f6feb; }
  .badge.unchanged { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
  .badge.changed { background: #3b1f72; color: #d2a8ff; border: 1px solid #8957e5; }

  .detail-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; margin: 0.75rem 0; font-size: 0.88rem; line-height: 1.6; }
  .detail-box code { background: #21262d; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.82rem; color: #79c0ff; }

  .feature-text { font-size: 0.82rem; color: #8b949e; max-width: 300px; }

  .framework-log { font-family: 'SF Mono', monospace; font-size: 0.8rem; line-height: 1.6; color: #8b949e; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }

  .footer { text-align: center; margin-top: 2rem; color: #484f58; font-size: 0.8rem; }
  .footer a { color: #58a6ff; }
</style>
</head>
<body>
HTMLHEAD

# Header
cat >> "$HTML_FILE" << HTMLHEADER
<div class="header">
  <h1>MLX Patch Comparison Report</h1>
  <div class="meta">
    <strong>maclocal-api</strong> commit <code>${MACLOCAL_SHA}</code> (${MACLOCAL_BRANCH}) &mdash; ${MACLOCAL_DATE}<br>
    Upstream <strong>mlx-swift-lm</strong>: pinned <code>${PINNED_TAG}</code> (${SUBMODULE_SHA:0:7}) &middot; latest tag <code>${LM_LATEST_TAG}</code> &middot; main <code>${LM_MAIN_SHA}</code><br>
    &nbsp;&nbsp;pinned&rarr;latest: ${LM_PINNED_TO_LATEST} commits &middot; latest&rarr;main: ${LM_LATEST_TO_MAIN} commits<br>
    Upstream <strong>mlx-swift</strong>: pin <code>${MLX_SWIFT_PIN}</code> &middot; latest tag <code>${MLX_SWIFT_LATEST_TAG}</code> &middot; main <code>${MLX_SWIFT_MAIN_SHA}</code><br>
    Generated: $(date '+%Y-%m-%d %H:%M:%S')
  </div>
</div>
HTMLHEADER

# Summary cards
cat >> "$HTML_FILE" << HTMLSUMMARY
<div class="summary-grid">
  <div class="summary-card info"><div class="value">${SUMMARY_TOTAL}</div><div class="label">Patch Files</div></div>
  <div class="summary-card safe"><div class="value">${SUMMARY_NONE}</div><div class="label">No Conflict</div></div>
  <div class="summary-card warn"><div class="value">$((SUMMARY_LOW + SUMMARY_MEDIUM))</div><div class="label">Low/Med Risk</div></div>
  <div class="summary-card risk"><div class="value">${SUMMARY_HIGH}</div><div class="label">High Risk</div></div>
  <div class="summary-card info"><div class="value">${SUMMARY_NEW}</div><div class="label">New Files</div></div>
</div>
HTMLSUMMARY

# Overview table
cat >> "$HTML_FILE" << TABLESTART
<div class="section">
  <h2>File-by-File Comparison</h2>
  <table>
    <thead>
      <tr>
        <th>File</th>
        <th>Target</th>
        <th>Type</th>
        <th>maclocal-api (${MACLOCAL_SHA})</th>
        <th>vs Pinned (${PINNED_TAG})</th>
        <th>vs Latest Tag (${LM_LATEST_TAG})</th>
        <th>vs Main (${LM_MAIN_SHA})</th>
        <th>Risk</th>
        <th>maclocal-api Features</th>
      </tr>
    </thead>
    <tbody>
TABLESTART

for i in "${!R_FILE[@]}"; do
  type_cls=$(badge_class "${R_TYPE[$i]}")
  risk_cls=$(badge_class "${R_RISK[$i]}")

  pinned_cell=$(status_cell "${R_PINNED_STATUS[$i]}" "${R_PINNED_LINES[$i]}" "${R_PINNED_ADDED[$i]}" "${R_PINNED_REMOVED[$i]}")
  latest_cell=$(status_cell "${R_LATEST_STATUS[$i]}" "${R_LATEST_LINES[$i]}" "${R_LATEST_ADDED[$i]}" "${R_LATEST_REMOVED[$i]}")
  main_cell=$(status_cell "${R_MAIN_STATUS[$i]}" "${R_MAIN_LINES[$i]}" "${R_MAIN_ADDED[$i]}" "${R_MAIN_REMOVED[$i]}")

  cat >> "$HTML_FILE" << TABLEROW
      <tr>
        <td><strong>${R_FILE[$i]}</strong></td>
        <td>${R_TARGET[$i]}</td>
        <td><span class="badge ${type_cls}">${R_TYPE[$i]}</span></td>
        <td>${R_MACLOCAL_LINES[$i]}L</td>
        <td>${pinned_cell}</td>
        <td>${latest_cell}</td>
        <td>${main_cell}</td>
        <td><span class="badge ${risk_cls}">${R_RISK[$i]}</span></td>
        <td><span class="feature-text">${R_FEATURES[$i]}</span></td>
      </tr>
TABLEROW
done

cat >> "$HTML_FILE" << 'TABLEEND'
    </tbody>
  </table>
</div>
TABLEEND

# mlx-swift framework section
cat >> "$HTML_FILE" << FWKHEADER
<div class="section">
  <h2>mlx-swift Framework: ${MLX_SWIFT_PIN} &rarr; ${MLX_SWIFT_LATEST_TAG} &rarr; main</h2>
  <div class="detail-box">
    <strong>${MLX_SWIFT_PIN} &rarr; ${MLX_SWIFT_LATEST_TAG}</strong> (${MLX_PINNED_TO_LATEST} commits):
    <div class="framework-log">
$(echo "$MLX_PINNED_TO_LATEST_LOG" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
    </div>
  </div>
  <div class="detail-box">
    <strong>${MLX_SWIFT_LATEST_TAG} &rarr; main</strong> (${MLX_LATEST_TO_MAIN} commits):
    <div class="framework-log">
$(echo "$MLX_LATEST_TO_MAIN_LOG" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
    </div>
  </div>
</div>
FWKHEADER

# Upstream-only files section
if [ "$UPSTREAM_ONLY_COUNT" -gt 0 ]; then
cat >> "$HTML_FILE" << UPSTART
<div class="section">
  <h2>Upstream-Only Files (Not in maclocal-api Patches)</h2>
  <div class="detail-box" style="margin-bottom: 1rem;">
    Files present in upstream <code>mlx-swift-lm</code> that are not covered by any maclocal-api patch.
    These may represent new model architectures or features available for adoption.
  </div>
  <table>
    <thead>
      <tr>
        <th>File</th>
        <th>Library</th>
        <th>Since</th>
        <th>Pinned (${PINNED_TAG})</th>
        <th>Latest Tag (${LM_LATEST_TAG})</th>
        <th>Main (${LM_MAIN_SHA})</th>
      </tr>
    </thead>
    <tbody>
UPSTART

for i in "${!U_FILE[@]}"; do
  since_cls="unchanged"
  case "${U_SINCE[$i]}" in
    latest) since_cls="new" ;;
    main)   since_cls="changed" ;;
  esac
  pl="${U_PINNED_LINES[$i]}"
  ll="${U_LATEST_LINES[$i]}"
  ml="${U_MAIN_LINES[$i]}"
  p_cell=$( [ "$pl" -gt 0 ] && echo "${pl}L" || echo "<span style=\"color:#484f58\">—</span>" )
  l_cell=$( [ "$ll" -gt 0 ] && echo "${ll}L" || echo "<span style=\"color:#484f58\">—</span>" )
  m_cell=$( [ "$ml" -gt 0 ] && echo "${ml}L" || echo "<span style=\"color:#484f58\">—</span>" )

  cat >> "$HTML_FILE" << UPROW
      <tr>
        <td>${U_FILE[$i]}</td>
        <td>${U_DIR[$i]}</td>
        <td><span class="badge ${since_cls}">${U_SINCE[$i]}</span></td>
        <td>${p_cell}</td>
        <td>${l_cell}</td>
        <td>${m_cell}</td>
      </tr>
UPROW
done

cat >> "$HTML_FILE" << 'UPEND'
    </tbody>
  </table>
</div>
UPEND
fi

# Footer
cat >> "$HTML_FILE" << HTMLFOOTER
<div class="footer">
  Generated by <code>Scripts/gen-patch-comparison.sh</code> &mdash; $(date '+%Y-%m-%d %H:%M:%S')<br>
  maclocal-api <code>${MACLOCAL_SHA}</code> &middot;
  <a href="https://github.com/ml-explore/mlx-swift-lm">ml-explore/mlx-swift-lm</a> &middot;
  <a href="https://github.com/ml-explore/mlx-swift">ml-explore/mlx-swift</a>
</div>

</body>
</html>
HTMLFOOTER

log_info "HTML report: $HTML_FILE"

# ============================================================
# Step 8: Generate CSV
# ============================================================
log_step "Generating CSV report"

CSV_FILE="$OUTPUT_DIR/mlx-patch-comparison-${TIMESTAMP}.csv"

# Header
echo "File,Target,Type,maclocal-api Lines,Pinned Lines,Pinned Status,Pinned +/-,Latest Tag Lines,Latest Tag Status,Latest Tag +/-,Main Lines,Main Status,Main +/-,Conflict Risk,maclocal-api Features" > "$CSV_FILE"

csv_escape() {
  local val="$1"
  if [[ "$val" == *","* ]] || [[ "$val" == *'"'* ]] || [[ "$val" == *$'\n'* ]]; then
    echo "\"${val//\"/\"\"}\""
  else
    echo "$val"
  fi
}

for i in "${!R_FILE[@]}"; do
  f=$(csv_escape "${R_FILE[$i]}")
  t=$(csv_escape "${R_TARGET[$i]}")
  ty=$(csv_escape "${R_TYPE[$i]}")
  ml="${R_MACLOCAL_LINES[$i]}"
  pl="${R_PINNED_LINES[$i]}"
  ps=$(csv_escape "${R_PINNED_STATUS[$i]}")
  pd="+${R_PINNED_ADDED[$i]}/-${R_PINNED_REMOVED[$i]}"
  ll="${R_LATEST_LINES[$i]}"
  ls_=$(csv_escape "${R_LATEST_STATUS[$i]}")
  ld="+${R_LATEST_ADDED[$i]}/-${R_LATEST_REMOVED[$i]}"
  mnl="${R_MAIN_LINES[$i]}"
  ms=$(csv_escape "${R_MAIN_STATUS[$i]}")
  md="+${R_MAIN_ADDED[$i]}/-${R_MAIN_REMOVED[$i]}"
  rk=$(csv_escape "${R_RISK[$i]}")
  ft=$(csv_escape "${R_FEATURES[$i]}")
  echo "$f,$t,$ty,$ml,$pl,$ps,$pd,$ll,$ls_,$ld,$mnl,$ms,$md,$rk,$ft" >> "$CSV_FILE"
done

# Append upstream-only files to CSV
for i in "${!U_FILE[@]}"; do
  f=$(csv_escape "${U_FILE[$i]}")
  t=$(csv_escape "${U_DIR[$i]}")
  pl="${U_PINNED_LINES[$i]}"
  ll="${U_LATEST_LINES[$i]}"
  mnl="${U_MAIN_LINES[$i]}"
  sn=$(csv_escape "${U_SINCE[$i]}")
  echo "$f,$t,UPSTREAM-ONLY,0,$pl,UPSTREAM,$sn,$ll,UPSTREAM,$sn,$mnl,UPSTREAM,$sn,N/A,Not in maclocal-api patches" >> "$CSV_FILE"
done

log_info "CSV report: $CSV_FILE"

# ============================================================
# Done
# ============================================================
echo ""
log_info "Done! Reports generated:"
log_info "  HTML: $HTML_FILE"
log_info "  CSV:  $CSV_FILE"
