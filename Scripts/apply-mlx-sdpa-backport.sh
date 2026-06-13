#!/bin/bash
# Backport mlx-swift 0.31.3's adaptive-block 2-pass SDPA into afm's pinned 0.30.3 mlx-swift checkout.
#
# WHY
# ---
# afm pins mlx-swift 0.30.3, whose 2-pass SDPA hardcodes `constexpr int blocks = 32` (the split-K
# count). 0.31.3 makes `blocks` a runtime function-constant and SCALES it with sequence length
# (32 -> up to 1024). On long-context decode this gives much more attention parallelism:
# measured decode@16k on Qwen3.6-27B-4bit / M4 Pro went ~13.0 -> ~14.4 tok/s (+10%), CORRECT
# (coherent output at ctx 31 / 4k / 16k). This is the same mechanism that makes LM Studio's newer
# MLX faster at depth.
#
# The historical "0.31.3 = garbage at long context" verdict was a METALLIB MISMATCH artifact
# (the new dispatch sets function-constant 26, but a stale prebuilt metallib's kernel had no such
# constant). With Scripts/rebuild-metallib.sh regenerating the metallib from the patched kernel,
# the adaptive kernel is correct.
#
# WHAT IT TOUCHES (all in the EPHEMERAL .build/checkouts/mlx-swift C++ tree -- wiped by
# `swift package clean` / re-resolve, so this must run after resolve and before build):
#   1. kernels/sdpa_vector.h                 <- full file from 0.31.3 (function-constant `blocks`)
#   2. scaled_dot_product_attention.cpp      <- full file from 0.31.3 (adaptive block selection)
#   3. mlx-generated/metal/sdpa_vector.h     <- full file from 0.31.3 (JIT/flattened copy, kept in sync)
#   4. backend/metal/utils.h                 <- TARGETED insertion of the 0.31.3 helper
#                                               `check_kernel_threadgroup_size` (the dispatch's only
#                                               new dependency). Insertion, not replacement, so we
#                                               don't mask other 0.30.3 utils.h content.
#
# After applying, you MUST: Scripts/rebuild-metallib.sh  (recompiles the kernel into default.metallib)
# then `swift build`.
#
# Usage: ./Scripts/apply-mlx-sdpa-backport.sh [--check] [--revert]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$SCRIPT_DIR/patches/mlx-cpp-sdpa"
MLX_CPP="$PROJECT_ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal"
MLX_GEN="$PROJECT_ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"

KERNEL_DST="$MLX_CPP/kernels/sdpa_vector.h"
DISPATCH_DST="$MLX_CPP/scaled_dot_product_attention.cpp"
GEN_DST="$MLX_GEN/sdpa_vector.h"
UTILS_DST="$MLX_CPP/utils.h"

KERNEL_SRC="$SRC_DIR/sdpa_vector.h"
DISPATCH_SRC="$SRC_DIR/scaled_dot_product_attention.cpp"
GEN_SRC="$SRC_DIR/sdpa_vector.generated.h"

HELPER_MARKER="check_kernel_threadgroup_size"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
info(){ echo "${GREEN}[INFO]${NC} $1"; }
warn(){ echo "${YELLOW}[WARN]${NC} $1"; }
err(){  echo "${RED}[ERROR]${NC} $1" >&2; }

# The helper inserted into utils.h, right before the closing `} // namespace mlx::core`.
read -r -d '' HELPER_BLOCK <<'EOF' || true

// AFM backport from mlx-swift 0.31.3: guard used by the adaptive-block SDPA dispatch.
// Throws if a kernel is launched with more threads/threadgroup than the device allows.
inline void check_kernel_threadgroup_size(
    const MTL::ComputePipelineState* kernel,
    MTL::Size group_dims,
    const std::string& name) {
  auto max_size = kernel->maxTotalThreadsPerThreadgroup();
  auto requested_size = group_dims.width * group_dims.height * group_dims.depth;

  if (max_size < requested_size) {
    std::ostringstream msg;
    msg << "Maximum threads per threadgroup is " << max_size
        << " but requested " << requested_size << " for kernel " << name << ".";
    throw std::runtime_error(msg.str());
  }
}
EOF

apply_file() { # src dst
  local src="$1" dst="$2" name; name="$(basename "$dst")"
  [ -f "$src" ] || { err "patch source missing: $src"; exit 1; }
  [ -f "$dst" ] || { err "target missing (resolve mlx-swift first): $dst"; exit 1; }
  if diff -q "$src" "$dst" >/dev/null 2>&1; then info "already applied: $name"; return; fi
  [ -f "$dst.afm-sdpa-orig" ] || cp "$dst" "$dst.afm-sdpa-orig"
  chmod u+w "$dst" 2>/dev/null || true
  cp "$src" "$dst"
  info "applied: $name"
}

apply_helper() {
  [ -f "$UTILS_DST" ] || { err "utils.h missing: $UTILS_DST"; exit 1; }
  if grep -q "$HELPER_MARKER" "$UTILS_DST"; then info "already applied: utils.h helper"; return; fi
  [ -f "$UTILS_DST.afm-sdpa-orig" ] || cp "$UTILS_DST" "$UTILS_DST.afm-sdpa-orig"
  chmod u+w "$UTILS_DST" 2>/dev/null || true
  # Insert HELPER_BLOCK before the LAST "} // namespace mlx::core"
  HELPER="$HELPER_BLOCK" perl -0pi -e '
    my $h = $ENV{HELPER};
    my $marker = "} // namespace mlx::core";
    my $idx = rindex($_, $marker);
    die "namespace close not found in utils.h\n" if $idx < 0;
    substr($_, $idx, 0) = $h . "\n";
  ' "$UTILS_DST"
  info "applied: utils.h helper (check_kernel_threadgroup_size)"
}

revert_file() { # dst
  local dst="$1" name; name="$(basename "$dst")"
  if [ -f "$dst.afm-sdpa-orig" ]; then
    chmod u+w "$dst" 2>/dev/null || true
    cp "$dst.afm-sdpa-orig" "$dst"; rm "$dst.afm-sdpa-orig"
    info "reverted: $name"
  else
    warn "no backup for $name (already original?)"
  fi
}

check_one() { # src dst name
  if diff -q "$1" "$2" >/dev/null 2>&1; then info "applied: $3"; return 0; fi
  warn "NOT applied: $3"; return 1
}

case "${1:-apply}" in
  --check)
    ok=true
    check_one "$KERNEL_SRC" "$KERNEL_DST" "sdpa_vector.h" || ok=false
    check_one "$DISPATCH_SRC" "$DISPATCH_DST" "scaled_dot_product_attention.cpp" || ok=false
    check_one "$GEN_SRC" "$GEN_DST" "mlx-generated/sdpa_vector.h" || ok=false
    if grep -q "$HELPER_MARKER" "$UTILS_DST" 2>/dev/null; then info "applied: utils.h helper"; else warn "NOT applied: utils.h helper"; ok=false; fi
    $ok || exit 1
    info "SDPA 0.31.3 backport fully applied."
    ;;
  --revert)
    revert_file "$KERNEL_DST"
    revert_file "$DISPATCH_DST"
    revert_file "$GEN_DST"
    revert_file "$UTILS_DST"
    info "Reverted SDPA backport. Run Scripts/rebuild-metallib.sh + swift build."
    ;;
  apply|"")
    [ -d "$MLX_CPP" ] || { err "mlx-swift checkout not found (run: swift package resolve)"; exit 1; }
    apply_file "$KERNEL_SRC" "$KERNEL_DST"
    apply_file "$DISPATCH_SRC" "$DISPATCH_DST"
    apply_file "$GEN_SRC" "$GEN_DST"
    apply_helper
    info ""
    info "SDPA 0.31.3 backport applied. Next: Scripts/rebuild-metallib.sh && swift build"
    ;;
  *) err "unknown option: $1"; echo "Usage: $0 [--check] [--revert]"; exit 1 ;;
esac
