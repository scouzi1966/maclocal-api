#!/bin/bash
# Rebuild Sources/MacLocalAPI/Resources/default.metallib from the (patched) MLX
# Metal kernel sources in the mlx-swift checkout.
#
# WHY THIS EXISTS
# ---------------
# afm ships a PREBUILT default.metallib (committed to git). `swift build` does NOT
# compile any Metal — it only copies that binary into the app bundle. So editing a
# kernel source such as `sdpa_vector.h` has ZERO effect until this script regenerates
# the metallib. (Editing the dispatch C++ in scaled_dot_product_attention.cpp DOES
# take effect via the Cmlx C++ target — which is why a kernel/dispatch mismatch
# silently produces garbage: the host launches a grid the compiled kernel wasn't
# built for.)
#
# The shipped metallib is the MLX "JIT-on" minimal precompiled set. Everything else
# (softmax, quantized, AFM's qmv_fast_wide, ...) is JIT-compiled at runtime, so it is
# NOT in the metallib and must NOT be added here. The exact translation-unit set was
# recovered from the shipped binary and is pinned in METAL_TUS below.
#
# Recipe mirrors mlx/backend/metal/kernels/CMakeLists.txt:
#   xcrun -sdk macosx metal <FLAGS> -c <kernel>.metal -I<MLXROOT> -o <kernel>.air
#   xcrun -sdk macosx metal <air...> -o default.metallib       (Xcode 26: no separate `metallib` tool)
#
# PREREQUISITE: the Metal Toolchain must be installed (Xcode 26 omits it by default):
#   xcodebuild -downloadComponent MetalToolchain
#
# Usage:
#   ./Scripts/rebuild-metallib.sh            # build + verify symbol parity + install
#   ./Scripts/rebuild-metallib.sh --check    # only check toolchain availability
#   ./Scripts/rebuild-metallib.sh --no-install  # build to /tmp, verify, do NOT replace the committed metallib
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLXROOT="$ROOT_DIR/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
KDIR="$MLXROOT/mlx/backend/metal/kernels"
TARGET_METALLIB="$ROOT_DIR/Sources/MacLocalAPI/Resources/default.metallib"
OSX_MIN="26.0"            # matches `apple-macosx26.0.0` baked into the shipped metallib
BUILD_DIR="$(mktemp -d /tmp/afm-metallib.XXXXXX)"

# Exact translation-unit set found in the shipped metallib (the MLX JIT-on always-built
# set + steel_attention). Paths are relative to $KDIR. Do NOT add JIT-only kernels here.
METAL_TUS=(
  arg_reduce
  conv
  gemv
  layer_norm
  rms_norm
  rope
  scaled_dot_product_attention
  steel/attn/kernels/steel_attention
)

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
info(){ echo "${GREEN}[INFO]${NC} $1"; }
warn(){ echo "${YELLOW}[WARN]${NC} $1"; }
err(){  echo "${RED}[ERROR]${NC} $1" >&2; }
cleanup(){ rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

check_toolchain(){
  # Trivial compile probe: detects the "missing Metal Toolchain" condition on Xcode 26.
  local out
  out="$(echo 'kernel void _afm_probe(){}' | xcrun -sdk macosx metal -x metal -c - -o /dev/null 2>&1 || true)"
  if echo "$out" | grep -qi "missing Metal Toolchain"; then
    err "Metal Toolchain is NOT installed. Install it once with:"
    err "    xcodebuild -downloadComponent MetalToolchain"
    err "(then re-run this script). Status check: xcodebuild -showComponent MetalToolchain"
    return 1
  fi
  if [ -n "$out" ]; then
    warn "metal probe emitted output (continuing):"; echo "$out" >&2
  fi
  info "Metal Toolchain available."
  return 0
}

# Distinct kernel entrypoint symbols (typed/sized instantiations) — used for parity check.
kernel_symbols(){ strings "$1" | grep -oE '^[a-z_]+(_[a-z0-9]+)*_(float|float16_t|bfloat16_t)(_[0-9]+)+$' | sort -u; }

MODE="build"
case "${1:-}" in
  --check) check_toolchain; exit $? ;;
  --no-install) MODE="no-install" ;;
  "") ;;
  *) err "Unknown option: $1"; exit 1 ;;
esac

[ -d "$MLXROOT" ] || { err "mlx-swift checkout not found at $MLXROOT (run: swift package resolve)"; exit 1; }
check_toolchain || exit 1

FLAGS=(-x metal -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions -mmacosx-version-min="$OSX_MIN")

info "Compiling ${#METAL_TUS[@]} translation units -> .air ..."
AIR_FILES=()
for tu in "${METAL_TUS[@]}"; do
  src="$KDIR/$tu.metal"
  [ -f "$src" ] || { err "kernel source missing: $src"; exit 1; }
  air="$BUILD_DIR/$(basename "$tu").air"
  echo "  metal -c $tu.metal"
  xcrun -sdk macosx metal "${FLAGS[@]}" -c "$src" -I"$MLXROOT" -o "$air"
  AIR_FILES+=("$air")
done

NEW_METALLIB="$BUILD_DIR/default.metallib"
info "Linking metallib ..."
xcrun -sdk macosx metal "${AIR_FILES[@]}" -o "$NEW_METALLIB"
[ -f "$NEW_METALLIB" ] || { err "metallib link produced no output"; exit 1; }
info "Built: $(du -h "$NEW_METALLIB" | cut -f1) ($NEW_METALLIB)"

# Parity check: a kernel-internal change (e.g. BN/blocks constexpr) must NOT change the
# set of exported kernel symbols. A mismatch means the TU set is wrong or the edit added/
# removed an instantiation — surface it loudly rather than silently shipping a different lib.
if [ -f "$TARGET_METALLIB" ]; then
  OLD_SYMS="$BUILD_DIR/old.syms"; NEW_SYMS="$BUILD_DIR/new.syms"
  kernel_symbols "$TARGET_METALLIB" > "$OLD_SYMS"
  kernel_symbols "$NEW_METALLIB"   > "$NEW_SYMS"
  if diff -q "$OLD_SYMS" "$NEW_SYMS" >/dev/null; then
    info "Kernel-symbol parity OK ($(wc -l < "$NEW_SYMS" | tr -d ' ') symbols match the shipped metallib)."
  else
    warn "Kernel-symbol set DIFFERS from the shipped metallib:"
    diff "$OLD_SYMS" "$NEW_SYMS" | sed 's/^/    /' >&2 || true
    warn "If your patch intentionally adds/removes a kernel instantiation, this is expected."
    warn "Otherwise the translation-unit set in METAL_TUS is wrong — do NOT install."
  fi
else
  warn "No existing metallib to compare against (parity check skipped)."
fi

if [ "$MODE" = "no-install" ]; then
  cp "$NEW_METALLIB" /tmp/afm-default.metallib
  info "--no-install: new metallib left at /tmp/afm-default.metallib (committed one untouched)."
  exit 0
fi

BACKUP="$TARGET_METALLIB.prebuilt-backup"
[ -f "$BACKUP" ] || cp "$TARGET_METALLIB" "$BACKUP" 2>/dev/null || true
cp "$NEW_METALLIB" "$TARGET_METALLIB"
info "Installed new metallib -> $TARGET_METALLIB"
info "Backup of the previous metallib: $BACKUP"
info "Next: run 'swift build -c release' to copy it into the app bundle, then test."
