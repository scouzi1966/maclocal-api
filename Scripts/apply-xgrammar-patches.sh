#!/bin/bash
# Applies local xgrammar patch set to vendor/xgrammar.
# Usage:
#   ./Scripts/apply-xgrammar-patches.sh [--check] [--revert]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
XGRAMMAR_DIR="$PROJECT_ROOT/vendor/xgrammar"
TARGET_FILE="$XGRAMMAR_DIR/cpp/grammar_functor.cc"
BACKUP_ROOT="$PROJECT_ROOT/.patch-backups/xgrammar"
BACKUP_FILE="$BACKUP_ROOT/grammar_functor.cc.original"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

is_patched() {
  [ -f "$TARGET_FILE" ] || return 1
  grep -q 'static constexpr int16_t kSelfRecursionFlag' "$TARGET_FILE"
}

apply_patch_change() {
  [ -f "$TARGET_FILE" ] || { log_error "Target file not found: $TARGET_FILE"; exit 1; }

  if is_patched; then
    log_info "Already patched: grammar_functor.cc"
    return 0
  fi

  mkdir -p "$BACKUP_ROOT"
  if [ ! -f "$BACKUP_FILE" ]; then
    cp "$TARGET_FILE" "$BACKUP_FILE"
    log_info "Backed up original: grammar_functor.cc"
  fi

  chmod u+w "$TARGET_FILE" 2>/dev/null || true
  perl -0pi -e 's/static const int16_t kSelfRecursionFlag = -0x300;/static constexpr int16_t kSelfRecursionFlag = -0x300;/g; s/static const int16_t kSimpleCycleFlag = -0x400;/static constexpr int16_t kSimpleCycleFlag = -0x400;/g; s/static const int16_t kUnKnownFlag = -0x500;/static constexpr int16_t kUnKnownFlag = -0x500;/g' "$TARGET_FILE"
  log_info "Applied: grammar_functor.cc"
}

revert_patch_change() {
  if [ -f "$BACKUP_FILE" ]; then
    chmod u+w "$TARGET_FILE" 2>/dev/null || true
    cp "$BACKUP_FILE" "$TARGET_FILE"
    rm "$BACKUP_FILE"
    log_info "Reverted: grammar_functor.cc"
  elif git -C "$XGRAMMAR_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    chmod u+w "$TARGET_FILE" 2>/dev/null || true
    git -C "$XGRAMMAR_DIR" checkout -- "cpp/grammar_functor.cc"
    log_info "Reverted from git: grammar_functor.cc"
  else
    log_warn "No backup found for grammar_functor.cc"
  fi
}

main() {
  local mode="apply"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --check) mode="check"; shift ;;
      --revert) mode="revert"; shift ;;
      *)
        log_error "Unknown option: $1"
        echo "Usage: $0 [--check] [--revert]"
        exit 1
        ;;
    esac
  done

  [ -d "$XGRAMMAR_DIR" ] || { log_error "Vendor xgrammar not found: $XGRAMMAR_DIR"; exit 1; }

  log_info "Found 1 xgrammar patch"
  log_info "Target xgrammar: $XGRAMMAR_DIR"

  case "$mode" in
    check)
      if is_patched; then
        log_info "Applied: grammar_functor.cc"
        log_info "All xgrammar patches are applied"
      else
        log_warn "Not applied: grammar_functor.cc"
        exit 1
      fi
      ;;
    revert)
      revert_patch_change
      log_info "xgrammar patches reverted"
      ;;
    apply)
      apply_patch_change
      log_info "xgrammar patches applied"
      ;;
  esac
}

main "$@"
