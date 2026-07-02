#!/bin/bash
# Backport mlx PR #3764 (qmv_wide small-batch quantized matvec) into afm's pinned
# mlx-swift 0.30.3 checkout — AFFINE (int4/int8) scope only.
#
# WHY
# ---
# qmv (M==1) re-reads the whole quantized weight matrix per input vector; qmm needs
# M >= vector_limit. For the M in [2, vector_limit) band — exactly afm's speculative-decode
# verify step (--mtp / --eagle3, M=2-8) and the concurrent batch endpoint at B=2-8 —
# upstream added qmv_wide (merged 2026-06-26): each weight group is dequantized ONCE and
# reused across up to 5 streamed input vectors (adapted from llama.cpp kernel_mul_mv_ext).
# Upstream kernel-time speedups on M4 Pro int4: 1.2x@M=2, 1.4x@M=4/8; int8 up to 1.8x.
#
# SCOPE: affine only. Upstream also ports fp modes (nvfp4/mxfp4/mxfp8), but 0.30.3's fp
# helpers (dequantize_scale, get_pack_factor) have different signatures — a misadapted fp
# port would silently corrupt output. use_qmv_wide() is therefore gated to
# mode == "affine" && GPU gen >= 15 (upstream gates affine to gen-15+ too; M3+=gen 15+).
#
# WHAT IT TOUCHES (all in the EPHEMERAL .build/checkouts/mlx-swift tree — wiped by
# `swift package clean` / re-resolve, so this must run after resolve and before build):
#   1. kernels/quantized.h            <- full file: dequantize gains `typename W` (thread-local
#                                        decode), qmv_wide_impl, affine_qmv_wide kernel
#   2. kernels/quantized.metal        <- full file: AOT instantiation macros (hygiene; the
#                                        quantized kernels afm runs are JIT-compiled)
#   3. quantized.cpp                  <- full file: use_qmv_wide() + qmv_wide() dispatch +
#                                        QuantizedMatmul::eval_gpu routing for M >= 2
#   4. mlx-generated/quantized.cpp    <- full file: same kernel edits inside the JIT preamble
#                                        string. RUNTIME-AUTHORITATIVE: Package.swift excludes
#                                        kernels/ from the build; quantized kernels JIT-compile
#                                        from THIS file. If it diverges from kernels/quantized.h
#                                        the dispatch/kernel mismatch silently produces garbage.
#   5. mlx-generated/metal/quantized.h <- sync copy of kernels/quantized.h
#
# ORDERING (enforced below): this script does FULL-FILE replacement of files that
# apply-mlx-cpp-patches.sh (qmv_fast_wide, M==1) later modifies in place. It must run
# BEFORE apply-mlx-cpp-patches.sh, and refuses to apply if the qmv_fast_wide marker is
# already present in a target (that would either wipe the marker or pollute the backup).
# Revert order is the reverse: apply-mlx-cpp-patches.sh --revert first, then this --revert.
#
# No metallib rebuild is required for THIS patch (quantized kernels are JIT), but build.sh
# runs rebuild-metallib.sh afterwards anyway; its kernel-symbol parity check should pass.
#
# Usage: ./Scripts/apply-mlx-qmv-wide-backport.sh [--check] [--revert]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$SCRIPT_DIR/patches/mlx-cpp-qmv-wide"
CMLX="$PROJECT_ROOT/.build/checkouts/mlx-swift/Source/Cmlx"
MLX_CPP="$CMLX/mlx/mlx/backend/metal"
MLX_GEN="$CMLX/mlx-generated"

KERNEL_DST="$MLX_CPP/kernels/quantized.h"
METAL_DST="$MLX_CPP/kernels/quantized.metal"
DISPATCH_DST="$MLX_CPP/quantized.cpp"
GEN_DST="$MLX_GEN/quantized.cpp"
GEN_HDR_DST="$MLX_GEN/metal/quantized.h"

KERNEL_SRC="$SRC_DIR/quantized.h"
METAL_SRC="$SRC_DIR/quantized.metal"
DISPATCH_SRC="$SRC_DIR/quantized.cpp"
GEN_SRC="$SRC_DIR/quantized.generated.cpp"

BACKPORT_MARKER="AFM-BACKPORT qmv_wide"
CPP_PATCH_MARKER="AFM_PATCH_qmv_fast_wide"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
info(){ echo "${GREEN}[INFO]${NC} $1"; }
warn(){ echo "${YELLOW}[WARN]${NC} $1"; }
err(){  echo "${RED}[ERROR]${NC} $1" >&2; }

apply_file() { # src dst
  local src="$1" dst="$2" name; name="$(basename "$dst")"
  [ -f "$src" ] || { err "patch source missing: $src"; exit 1; }
  [ -f "$dst" ] || { err "target missing (resolve mlx-swift first): $dst"; exit 1; }
  if diff -q "$src" "$dst" >/dev/null 2>&1; then info "already applied: $name"; return; fi
  if grep -q "$BACKPORT_MARKER" "$dst"; then
    # Re-run after apply-mlx-cpp-patches.sh added its markers on top — nothing to do.
    # (If the stored patch files were UPDATED since, run --revert first to pick them up.)
    info "already applied: $name (cpp-patches on top)"; return
  fi
  if grep -q "$CPP_PATCH_MARKER" "$dst"; then
    err "$name contains the qmv_fast_wide marker but not the backport — wrong order."
    err "Run: Scripts/apply-mlx-cpp-patches.sh --revert, then this script, then re-apply it."
    exit 1
  fi
  [ -f "$dst.afm-qmvwide-orig" ] || cp "$dst" "$dst.afm-qmvwide-orig"
  chmod u+w "$dst" 2>/dev/null || true
  cp "$src" "$dst"
  info "applied: $name"
}

revert_file() { # dst
  local dst="$1" name; name="$(basename "$dst")"
  if grep -q "$CPP_PATCH_MARKER" "$dst" 2>/dev/null; then
    err "$name still carries qmv_fast_wide — run apply-mlx-cpp-patches.sh --revert first."
    exit 1
  fi
  if [ -f "$dst.afm-qmvwide-orig" ]; then
    chmod u+w "$dst" 2>/dev/null || true
    cp "$dst.afm-qmvwide-orig" "$dst"; rm "$dst.afm-qmvwide-orig"
    info "reverted: $name"
  else
    warn "no backup for $name (already original?)"
  fi
}

check_one() { # dst name  (marker check, not byte-diff: cpp-patches edits these files afterwards)
  if grep -q "$BACKPORT_MARKER" "$1" 2>/dev/null; then info "applied: $2"; return 0; fi
  warn "NOT applied: $2"; return 1
}

case "${1:-apply}" in
  --check)
    ok=true
    check_one "$KERNEL_DST" "kernels/quantized.h" || ok=false
    check_one "$METAL_DST" "kernels/quantized.metal" || ok=false
    check_one "$DISPATCH_DST" "quantized.cpp" || ok=false
    check_one "$GEN_DST" "mlx-generated/quantized.cpp" || ok=false
    check_one "$GEN_HDR_DST" "mlx-generated/metal/quantized.h" || ok=false
    $ok || exit 1
    info "qmv_wide backport fully applied."
    ;;
  --revert)
    revert_file "$KERNEL_DST"
    revert_file "$METAL_DST"
    revert_file "$DISPATCH_DST"
    revert_file "$GEN_DST"
    revert_file "$GEN_HDR_DST"
    info "Reverted qmv_wide backport. Run swift build."
    ;;
  apply|"")
    [ -d "$MLX_CPP" ] || { err "mlx-swift checkout not found (run: swift package resolve)"; exit 1; }
    apply_file "$KERNEL_SRC" "$KERNEL_DST"
    apply_file "$METAL_SRC" "$METAL_DST"
    apply_file "$DISPATCH_SRC" "$DISPATCH_DST"
    apply_file "$GEN_SRC" "$GEN_DST"
    apply_file "$KERNEL_SRC" "$GEN_HDR_DST"
    info ""
    info "qmv_wide backport applied. Next: apply-mlx-cpp-patches.sh (if used) && swift build"
    ;;
  *) err "unknown option: $1"; echo "Usage: $0 [--check] [--revert]"; exit 1 ;;
esac
