#!/bin/bash
# Applies performance patches to MLX C++ Metal shaders in .build/checkouts/mlx-swift/
# These are lost on clean builds (swift package clean) and must be reapplied.
# Usage: ./Scripts/apply-mlx-cpp-patches.sh [--check] [--revert]
#
# Patches:
#  1. qmv_fast_wide: packs_per_thread=4 variant for 4-bit with K >= 1024
#     Halves inner loop iterations for large K dimensions (attention, gate/up projections).

set -euo pipefail

MLX_CPP_DIR=".build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal"
MLX_GEN_DIR=".build/checkouts/mlx-swift/Source/Cmlx/mlx-generated"
QUANTIZED_H="$MLX_CPP_DIR/kernels/quantized.h"
QUANTIZED_CPP="$MLX_CPP_DIR/quantized.cpp"
QUANTIZED_METAL="$MLX_CPP_DIR/kernels/quantized.metal"
QUANTIZED_GEN="$MLX_GEN_DIR/quantized.cpp"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

MARKER="AFM_PATCH_qmv_fast_wide"

is_patched() {
  grep -q "$MARKER" "$QUANTIZED_H" 2>/dev/null && grep -q "$MARKER" "$QUANTIZED_GEN" 2>/dev/null
}

check_patches() {
  if is_patched; then
    log_info "MLX C++ patches are applied"
    return 0
  else
    log_warn "MLX C++ patches are NOT applied"
    return 1
  fi
}

revert_patches() {
  for f in "$QUANTIZED_H" "$QUANTIZED_CPP" "$QUANTIZED_METAL" "$QUANTIZED_GEN"; do
    local backup="${f}.original"
    if [ -f "$backup" ]; then
      cp "$backup" "$f"
      rm "$backup"
      log_info "Reverted: $(basename "$f") [$(dirname "$f" | sed 's|.*/||')]"
    else
      log_warn "No backup for: $(basename "$f")"
    fi
  done
}

apply_patches() {
  # Check prerequisites
  for f in "$QUANTIZED_H" "$QUANTIZED_CPP" "$QUANTIZED_METAL" "$QUANTIZED_GEN"; do
    [ -f "$f" ] || { log_error "File not found: $f (run swift package resolve first)"; exit 1; }
  done

  if is_patched; then
    log_info "MLX C++ patches already applied"
    return 0
  fi

  # Back up originals and make writable
  for f in "$QUANTIZED_H" "$QUANTIZED_CPP" "$QUANTIZED_METAL" "$QUANTIZED_GEN"; do
    local backup="${f}.original"
    if [ ! -f "$backup" ]; then
      cp "$f" "$backup"
    fi
    chmod u+w "$f" 2>/dev/null || true
  done

  # ============================================================
  # PATCH 1: quantized.h — add qmv_fast_wide_impl (packs_per_thread=4)
  # ============================================================
  log_info "Patching quantized.h — adding qmv_fast_wide_impl ..."

  # Find the line after the closing brace of qmv_fast_impl (line after "}" that follows simd_sum reduction)
  # We inject the wide variant right after qmv_fast_impl
  # Strategy: add the new function after the existing qmv_fast_impl closing brace.
  #
  # The wide variant is identical to qmv_fast_impl except packs_per_thread = 4 (not 2) for bits != 2.
  # This doubles values_per_thread (16→32) and block_size (512→1024).
  # Only safe when K >= 1024 and K % 1024 == 0.

  cat >> "$QUANTIZED_H" << WIDE_KERNEL_EOF

// ${MARKER}
// Wide variant of qmv_fast: packs_per_thread=4 for 4-bit, halves K-loop iterations for K >= 1024.
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_wide_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 2 : 4;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

// Batched and non-batched kernel wrappers for qmv_fast_wide
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void affine_qmv_fast_wide(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)x_batch_ndims;
  (void)x_shape;
  (void)x_strides;
  (void)w_batch_ndims;
  (void)w_shape;
  (void)w_strides;
  (void)s_strides;
  (void)b_strides;
  if (batched) {
    adjust_matrix_offsets<T>(
        x, w, scales, biases, y,
        out_vec_size, x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides, s_strides, b_strides, tid);
  }
  qmv_fast_wide_impl<T, group_size, bits>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

// Gather variant for MoE expert dispatch
template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv_fast_wide(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x, w, scales, biases,
      lhs_indices, rhs_indices, y,
      out_vec_size * M,
      batch_ndims, batch_shape,
      lhs_strides, rhs_strides,
      x_batch_ndims, x_shape, x_strides,
      w_batch_ndims, w_shape, w_strides,
      s_strides, b_strides, tid);
  qmv_fast_wide_impl<T, group_size, bits>(
      w, scales, biases, x, y,
      in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}
WIDE_KERNEL_EOF

  log_info "Patched quantized.h"

  # ============================================================
  # PATCH 2: quantized.metal — instantiate wide kernel variants
  # ============================================================
  log_info "Patching quantized.metal — adding wide kernel instantiation ..."

  cat >> "$QUANTIZED_METAL" << 'METAL_EOF'

// [AFM-PATCH] Wide qmv kernel instantiation (packs_per_thread=4)
#define instantiate_qmv_fast_wide(type, group_size, bits) \
  instantiate_quantized_batched(affine_qmv_fast_wide, type, group_size, bits, 1) \
  instantiate_quantized_batched(affine_qmv_fast_wide, type, group_size, bits, 0)

#define instantiate_gather_qmv_fast_wide(type, group_size, bits) \
  instantiate_quantized(affine_gather_qmv_fast_wide, type, group_size, bits)

// Only for 4-bit (the dominant quantization for our target models)
#define instantiate_qmv_wide_all(type, group_size) \
  instantiate_qmv_fast_wide(type, group_size, 4) \
  instantiate_gather_qmv_fast_wide(type, group_size, 4)

// Instantiate for float16 and bfloat16 with common group sizes
instantiate_qmv_wide_all(float16_t, 32)
instantiate_qmv_wide_all(float16_t, 64)
instantiate_qmv_wide_all(float16_t, 128)
instantiate_qmv_wide_all(bfloat16_t, 32)
instantiate_qmv_wide_all(bfloat16_t, 64)
instantiate_qmv_wide_all(bfloat16_t, 128)
METAL_EOF

  log_info "Patched quantized.metal"

  # ============================================================
  # PATCH 3: mlx-generated/quantized.cpp — inject wide kernel into embedded Metal source
  # ============================================================
  log_info "Patching mlx-generated/quantized.cpp — injecting wide kernel into JIT source ..."

  # The generated file contains the Metal shader source as a raw string literal.
  # We inject our wide kernel function before the closing )preamble" marker.
  # Use a Python one-liner for reliable multi-line insertion.
  python3 -c "
import sys
WIDE_KERNEL = '''
// ${MARKER}
// Wide variant of qmv_fast: packs_per_thread=4 for 4-bit, halves K-loop iterations for K >= 1024.
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_wide_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 2 : 4;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}


// Batched and non-batched kernel wrappers for qmv_fast_wide
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void affine_qmv_fast_wide(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant int& x_batch_ndims [[buffer(7)]],
    const constant int* x_shape [[buffer(8)]],
    const constant int64_t* x_strides [[buffer(9)]],
    const constant int& w_batch_ndims [[buffer(10)]],
    const constant int* w_shape [[buffer(11)]],
    const constant int64_t* w_strides [[buffer(12)]],
    const constant int64_t* s_strides [[buffer(13)]],
    const constant int64_t* b_strides [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)x_batch_ndims;
  (void)x_shape;
  (void)x_strides;
  (void)w_batch_ndims;
  (void)w_shape;
  (void)w_strides;
  (void)s_strides;
  (void)b_strides;
  if (batched) {
    adjust_matrix_offsets<T>(
        x, w, scales, biases, y,
        out_vec_size, x_batch_ndims, x_shape, x_strides,
        w_batch_ndims, w_shape, w_strides, s_strides, b_strides, tid);
  }
  qmv_fast_wide_impl<T, group_size, bits>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

// Gather variant for MoE expert dispatch
template <typename T, int group_size, int bits>
[[kernel]] void affine_gather_qmv_fast_wide(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int64_t* b_strides [[buffer(16)]],
    const constant int& batch_ndims [[buffer(17)]],
    const constant int* batch_shape [[buffer(18)]],
    const constant int64_t* lhs_strides [[buffer(19)]],
    const constant int64_t* rhs_strides [[buffer(20)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(
      x, w, scales, biases,
      lhs_indices, rhs_indices, y,
      out_vec_size * M,
      batch_ndims, batch_shape,
      lhs_strides, rhs_strides,
      x_batch_ndims, x_shape, x_strides,
      w_batch_ndims, w_shape, w_strides,
      s_strides, b_strides, tid);
  qmv_fast_wide_impl<T, group_size, bits>(
      w, scales, biases, x, y,
      in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}
'''
f = '$QUANTIZED_GEN'
with open(f, 'r') as fh:
    content = fh.read()
# Insert before the last separator comment before )preamble
marker = '///////////////////////////////////////////////////////////////////////////////\n)preamble\"'
if marker in content:
    content = content.replace(marker, WIDE_KERNEL + '\n' + marker)
    with open(f, 'w') as fh:
        fh.write(content)
    print('OK')
else:
    print('ERROR: could not find insertion point')
    sys.exit(1)
"

  log_info "Patched mlx-generated/quantized.cpp"

  # ============================================================
  # PATCH 4: quantized.cpp (C++ dispatch) — select wide variant when K % 1024 == 0
  # ============================================================
  log_info "Patching quantized.cpp — adding wide dispatch logic ..."

  # Patch the qmv() function: change the fast path to prefer wide when possible
  # Original:  bool fast = N % bn == 0 && K % 512 == 0;
  # After:     bool fast = N % bn == 0 && K % 512 == 0;
  #            bool wide = fast && K % 1024 == 0 && bits == 4;
  sed -i '' '/^void qmv(/,/^}/ {
    /bool fast = N % bn == 0 && K % 512 == 0;/ {
      a\
\  bool wide = fast \&\& K % 1024 == 0 \&\& bits == 4;
    }
    s/mode + (fast ? "_qmv_fast_" : "_qmv_")/mode + (wide ? "_qmv_fast_wide_" : fast ? "_qmv_fast_" : "_qmv_")/
    s/(fast ? "qmv_fast" : "qmv")/(wide ? "qmv_fast_wide" : fast ? "qmv_fast" : "qmv")/
  }' "$QUANTIZED_CPP"

  # Patch the gather_qmv() function similarly
  sed -i '' '/^void gather_qmv(/,/^}/ {
    /bool fast = N % bn == 0 && K % 512 == 0;/ {
      a\
\  bool wide = fast \&\& K % 1024 == 0 \&\& bits == 4;
    }
    s/mode + (fast ? "_gather_qmv_fast_" : "_gather_qmv_")/mode + (wide ? "_gather_qmv_fast_wide_" : fast ? "_gather_qmv_fast_" : "_gather_qmv_")/
    s/(fast ? "gather_qmv_fast" : "gather_qmv")/(wide ? "gather_qmv_fast_wide" : fast ? "gather_qmv_fast" : "gather_qmv")/
  }' "$QUANTIZED_CPP"

  log_info "Patched quantized.cpp"
  log_info ""
  log_info "MLX C++ patches applied. Clean build required."
}

mode="apply"
while [[ $# -gt 0 ]]; do
  case $1 in
    --check) mode="check"; shift ;;
    --revert) mode="revert"; shift ;;
    *) log_error "Unknown option: $1"; exit 1 ;;
  esac
done

case "$mode" in
  check) check_patches ;;
  revert) revert_patches ;;
  apply) apply_patches ;;
esac
