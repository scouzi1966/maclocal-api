#!/bin/bash
# Applies Vesta MLX patch set to local vendor/mlx-swift-lm checkout.
# Usage:
#   ./Scripts/apply-mlx-patches.sh [--check] [--revert]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/patches"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MLX_LM_DIR="$PROJECT_ROOT/vendor/mlx-swift-lm"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PATCH_FILES=("Qwen3VL.swift" "Qwen3Next.swift" "GatedDelta.swift" "DeepseekV3.swift" "MiniMaxM2.swift" "NemotronH.swift" "GLM4MoeLite.swift" "GLM5MoeDsa.swift" "KimiK25.swift" "LLMModelFactory.swift" "Load.swift" "Evaluate.swift")
TARGET_PATHS=("Libraries/MLXVLM/Models/Qwen3VL.swift" "Libraries/MLXLLM/Models/Qwen3Next.swift" "Libraries/MLXLLM/Models/GatedDelta.swift" "Libraries/MLXLLM/Models/DeepseekV3.swift" "Libraries/MLXLLM/Models/MiniMaxM2.swift" "Libraries/MLXLLM/Models/NemotronH.swift" "Libraries/MLXLLM/Models/GLM4MoeLite.swift" "Libraries/MLXLLM/Models/GLM5MoeDsa.swift" "Libraries/MLXLLM/Models/KimiK25.swift" "Libraries/MLXLLM/LLMModelFactory.swift" "Libraries/MLXLMCommon/Load.swift" "Libraries/MLXLMCommon/Evaluate.swift")
NEW_FILES=("Qwen3Next.swift" "GatedDelta.swift" "MiniMaxM2.swift" "NemotronH.swift" "GLM4MoeLite.swift" "GLM5MoeDsa.swift" "KimiK25.swift")

# --- Package-level patches (sed replacements in Package.swift) ---
# Each entry: "search_pattern|replacement"
# These fix dependency version pins that can't be handled by file-copy patches.
MLX_PACKAGE_SWIFT="$MLX_LM_DIR/Package.swift"
PACKAGE_PINS=(
  '.upToNextMinor(from: "0.30.3")|exact: "0.30.3"'
)

is_new_file() {
  local filename="$1"
  for nf in "${NEW_FILES[@]}"; do
    if [ "$nf" = "$filename" ]; then
      return 0
    fi
  done
  return 1
}

apply_package_pins() {
  local pkg_file="$1"
  [ -f "$pkg_file" ] || { log_error "Package.swift not found: $pkg_file"; return 1; }

  # Backup once
  if [ ! -f "${pkg_file}.original" ]; then
    cp "$pkg_file" "${pkg_file}.original"
    log_info "Backed up original: Package.swift"
  fi

  for entry in "${PACKAGE_PINS[@]}"; do
    local search="${entry%%|*}"
    local replace="${entry##*|}"
    if grep -qF "$replace" "$pkg_file"; then
      log_info "Already pinned: $replace"
    elif grep -qF "$search" "$pkg_file"; then
      sed -i '' "s|$(echo "$search" | sed 's/[.[\/*^$]/\\&/g')|$(echo "$replace" | sed 's/[&/\]/\\&/g')|g" "$pkg_file"
      log_info "Pinned: $search → $replace"
    else
      log_warn "Pattern not found in Package.swift: $search"
    fi
  done
}

check_package_pins() {
  local pkg_file="$1"
  local all_ok=true
  [ -f "$pkg_file" ] || { log_warn "Package.swift not found: $pkg_file"; return 1; }

  for entry in "${PACKAGE_PINS[@]}"; do
    local replace="${entry##*|}"
    if grep -qF "$replace" "$pkg_file"; then
      log_info "Pinned: $replace"
    else
      log_warn "Not pinned: $replace"
      all_ok=false
    fi
  done
  $all_ok
}

revert_package_pins() {
  local pkg_file="$1"
  local backup="${pkg_file}.original"
  if [ -f "$backup" ]; then
    cp "$backup" "$pkg_file"
    rm "$backup"
    log_info "Reverted: Package.swift"
  else
    log_warn "No backup found for Package.swift"
  fi
}

is_file_patched() {
  local patch_file="$1"
  local target_file="$2"
  [ -f "$target_file" ] || return 1
  diff -q "$patch_file" "$target_file" >/dev/null 2>&1
}

apply_file() {
  local patch_file="$1"
  local target_file="$2"
  local filename
  filename="$(basename "$patch_file")"

  if is_file_patched "$patch_file" "$target_file"; then
    log_info "Already patched: $filename"
    return 0
  fi

  if is_new_file "$filename"; then
    mkdir -p "$(dirname "$target_file")"
    cp "$patch_file" "$target_file"
    log_info "Added new file: $filename"
  else
    local backup_file="${target_file}.original"
    if [ ! -f "$backup_file" ]; then
      cp "$target_file" "$backup_file"
      log_info "Backed up original: $filename"
    fi
    chmod u+w "$target_file" 2>/dev/null || true
    cp "$patch_file" "$target_file"
    log_info "Applied: $filename"
  fi
}

revert_file() {
  local target_file="$1"
  local filename
  filename="$(basename "$target_file")"
  local backup_file="${target_file}.original"

  if is_new_file "$filename"; then
    if [ -f "$target_file" ]; then
      rm "$target_file"
      log_info "Removed: $filename"
    else
      log_warn "File not found: $filename (already removed)"
    fi
  elif [ -f "$backup_file" ]; then
    chmod u+w "$target_file" 2>/dev/null || true
    cp "$backup_file" "$target_file"
    rm "$backup_file"
    log_info "Reverted: $filename"
  else
    log_warn "No backup found for: $filename (may already be original)"
  fi
}

check_patches() {
  local mlx_dir="$1"
  local all_applied=true

  for i in "${!PATCH_FILES[@]}"; do
    local patch_name="${PATCH_FILES[$i]}"
    local rel_path="${TARGET_PATHS[$i]}"
    local patch_file="$PATCHES_DIR/$patch_name"
    local target_file="$mlx_dir/$rel_path"

    if [ ! -f "$patch_file" ]; then
      log_warn "Patch file not found: $patch_name"
      all_applied=false
      continue
    fi

    if is_file_patched "$patch_file" "$target_file"; then
      log_info "Applied: $patch_name"
    else
      log_warn "Not applied: $patch_name"
      all_applied=false
    fi
  done

  $all_applied
}

main() {
  local mode="apply"

  while [[ $# -gt 0 ]]; do
    case $1 in
      --check) mode="check"; shift ;;
      --revert) mode="revert"; shift ;;
      *)
        log_error "Unknown option: $1"
        echo "Usage: $0 [--check] [--revert]"
        exit 1
        ;;
    esac
  done

  [ -d "$PATCHES_DIR" ] || { log_error "Patches directory not found: $PATCHES_DIR"; exit 1; }
  [ -d "$MLX_LM_DIR" ] || { log_error "Vendor mlx-swift-lm not found: $MLX_LM_DIR"; exit 1; }

  log_info "Found ${#PATCH_FILES[@]} file patch(es)"
  log_info "Target mlx-swift-lm: $MLX_LM_DIR"

  case "$mode" in
    check)
      local files_ok=true
      local pins_ok=true
      check_patches "$MLX_LM_DIR" || files_ok=false
      check_package_pins "$MLX_PACKAGE_SWIFT" || pins_ok=false
      if $files_ok && $pins_ok; then
        log_info "All patches are applied"
      else
        exit 1
      fi
      ;;
    revert)
      for i in "${!PATCH_FILES[@]}"; do
        revert_file "$MLX_LM_DIR/${TARGET_PATHS[$i]}"
      done
      revert_package_pins "$MLX_PACKAGE_SWIFT"
      log_info ""
      log_info "Files reverted. Clean build required."
      ;;
    apply)
      for i in "${!PATCH_FILES[@]}"; do
        local patch_name="${PATCH_FILES[$i]}"
        local patch_file="$PATCHES_DIR/$patch_name"
        local target_file="$MLX_LM_DIR/${TARGET_PATHS[$i]}"

        if [ ! -f "$patch_file" ]; then
          log_warn "Patch file not found: $patch_name"
          continue
        fi
        apply_file "$patch_file" "$target_file"
      done
      apply_package_pins "$MLX_PACKAGE_SWIFT"
      log_info ""
      log_info "Patches applied to vendor/mlx-swift-lm. Clean build required."
      ;;
  esac
}

main "$@"
