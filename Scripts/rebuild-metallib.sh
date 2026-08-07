#!/bin/bash
# Rebuild Sources/AFMKitMLX/Resources/default.metallib from the (patched) MLX
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
#   ./Scripts/rebuild-metallib.sh --no-install  # build under .build, verify, do NOT replace the committed metallib
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLXROOT="$ROOT_DIR/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
KDIR="$MLXROOT/mlx/backend/metal/kernels"
TARGET_METALLIB="$ROOT_DIR/Sources/AFMKitMLX/Resources/default.metallib"
OSX_MIN="26.0"            # matches `apple-macosx26.0.0` baked into the shipped metallib
mkdir -p "$ROOT_DIR/.build"
BUILD_DIR="$(mktemp -d "$ROOT_DIR/.build/afm-metallib.XXXXXX")"

# Exact translation-unit set found in the shipped metallib (the MLX JIT-on always-built
# set + steel_attention). Paths are relative to $KDIR. Do NOT add JIT-only kernels here.
# NOTE: `random` is REQUIRED — it provides the `rbitsc`/`rbits` RNG kernels used by any
# temperature>0 (sampled) generation. Omitting it builds a metallib that loads fine and
# works for greedy (temp=0) decode but FATAL-errors ("Unable to load kernel rbitsc") on the
# first sampled request. It has no global ctor so it isn't detected by the _GLOBAL__sub_I
# scan — it must be listed explicitly.
METAL_TUS=(
  arg_reduce
  conv
  gemv
  layer_norm
  random
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
  # Trivial compile probe. Must actually succeed — a non-zero status means the toolchain is
  # unusable (missing Metal Toolchain on Xcode 26, xcrun can't find `metal`, wrong selected
  # Xcode, etc.). Treat ANY failure as unavailable so build.sh takes its fallback path instead
  # of skipping it and then failing in the real compile.
  local out status
  out="$(echo 'kernel void _afm_probe(){}' | xcrun -sdk macosx metal -x metal -c - -o /dev/null 2>&1)"
  status=$?
  if [ "$status" -ne 0 ]; then
    if echo "$out" | grep -qi "missing Metal Toolchain"; then
      err "Metal Toolchain is NOT installed. Install it once with:"
      err "    xcodebuild -downloadComponent MetalToolchain"
      err "(then re-run this script). Status check: xcodebuild -showComponent MetalToolchain"
    else
      err "metal probe failed (exit $status) — Metal toolchain unusable:"
      echo "$out" >&2
    fi
    return 1
  fi
  if [ -n "$out" ]; then
    warn "metal probe emitted output (continuing):"; echo "$out" >&2
  fi
  info "Metal Toolchain available."
  return 0
}

# Distinct kernel entrypoint symbols (typed/sized instantiations) — used for parity check.
# Scan printable byte runs directly because Xcode 27's `strings` can refuse AIR
# produced for an older OS target even though Metal loads that AIR correctly.
kernel_symbols(){ LC_ALL=C grep -aoE '[a-z_]+(_[a-z0-9]+)*_(float|float16_t|bfloat16_t)(_[0-9]+)+' "$1" | sort -u; }

MODE="build"
ALLOW_KERNEL_CHANGE=0
for arg in "$@"; do
  case "$arg" in
    --check) check_toolchain; exit $? ;;
    --no-install) MODE="no-install" ;;
    --allow-kernel-change) ALLOW_KERNEL_CHANGE=1 ;;  # permit a changed kernel-symbol set
    "") ;;
    *) err "Unknown option: $arg"; exit 1 ;;
  esac
done

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
# The deployment target is part of the final metallib container as well as each
# AIR object. Omitting it here silently relinks macOS 26 AIR as a macOS 27
# library when Xcode 27 is selected, producing a package that builds correctly
# but fails to load on macOS 26.
xcrun -sdk macosx metal -mmacosx-version-min="$OSX_MIN" "${AIR_FILES[@]}" -o "$NEW_METALLIB"
[ -f "$NEW_METALLIB" ] || { err "metallib link produced no output"; exit 1; }
info "Built: $(du -h "$NEW_METALLIB" | cut -f1) ($NEW_METALLIB)"

# Inspect the binary directly rather than using `strings`: Xcode 27's strings
# rejects some valid Xcode 26 AIR triples before it can print their metadata.
METAL_TARGETS="$(LC_ALL=C grep -aoE 'air64(_v[0-9]+)?-apple-macosx[0-9]+(\.[0-9]+){2}' "$NEW_METALLIB" | sort -u || true)"
if [ -z "$METAL_TARGETS" ]; then
  err "Built metallib contains no readable macOS deployment target."
  exit 1
fi
while IFS= read -r target; do
  target_version="${target##*macosx}"
  target_major="${target_version%%.*}"
  if [ "$target_major" -gt "${OSX_MIN%%.*}" ]; then
    err "Built metallib targets $target_version, newer than required macOS $OSX_MIN."
    exit 1
  fi
done <<< "$METAL_TARGETS"
info "Metal deployment target OK: $(echo "$METAL_TARGETS" | tr '\n' ' ')"

# Required-kernel guard: catch whole-TU omissions that the name-pattern parity check below
# can miss (e.g. the RNG kernels have no `*_float_NxN` symbol). A metallib missing `rbits`
# loads fine and works for greedy decode but FATAL-errors on the first sampled (temp>0)
# request — exactly the failure that motivated this guard. Refuse to install if absent.
for req in rbits; do
  # Count matches without `grep -q`: -q closes the pipe on first hit, which SIGPIPEs `strings`
  # → non-zero → `set -o pipefail` would make this read as "missing" even when present.
  # `grep -c` consumes all input, so the pipeline exits cleanly.
  n=$(LC_ALL=C grep -aoc "$req" "$NEW_METALLIB" || true)
  if [ "${n:-0}" -eq 0 ]; then
    err "Built metallib is MISSING required kernel '$req' — a translation unit is absent from METAL_TUS."
    err "(Most likely 'random'.) This would FATAL on sampled generation. Refusing to install."
    exit 1
  fi
done

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
    if [ "$ALLOW_KERNEL_CHANGE" -eq 1 ]; then
      warn "--allow-kernel-change set: installing the changed kernel set anyway."
    else
      err "Refusing to replace the committed metallib with a different kernel-symbol set."
      err "If the translation-unit set in METAL_TUS is wrong, fix it. If you INTENTIONALLY"
      err "added/removed a kernel instantiation, re-run with --allow-kernel-change."
      err "(Use --no-install to build to /tmp without touching the committed metallib.)"
      exit 1
    fi
  fi
else
  warn "No existing metallib to compare against (parity check skipped)."
fi

if [ "$MODE" = "no-install" ]; then
  OUTPUT_METALLIB="$ROOT_DIR/.build/afm-default.metallib"
  cp "$NEW_METALLIB" "$OUTPUT_METALLIB"
  info "--no-install: new metallib left at $OUTPUT_METALLIB (committed one untouched)."
  exit 0
fi

BACKUP_DIR="$ROOT_DIR/test-reports/metallib"
mkdir -p "$BACKUP_DIR"
BACKUP="$BACKUP_DIR/default.metallib.prebuilt-backup"
[ -f "$BACKUP" ] || cp "$TARGET_METALLIB" "$BACKUP" 2>/dev/null || true
cp "$NEW_METALLIB" "$TARGET_METALLIB"
info "Installed new metallib -> $TARGET_METALLIB"
info "Backup of the previous metallib: $BACKUP"
info "Next: run 'swift build -c release' to copy it into the app bundle, then test."
