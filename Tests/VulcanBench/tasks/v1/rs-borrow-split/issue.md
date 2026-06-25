# Overlapping mutable borrows in `core::get_pair_mut`

The `core` crate (`splitdemo::core`) provides `get_pair_mut(v, i, j)` which
should return two **non-overlapping** mutable references into a `SplitVec`.

Currently it calls `v.get_mut(i)` and `v.get_mut(j)` sequentially. When `i == j`
(or when both indices land in the same underlying allocation region), this
produces two mutable references to the same data — undefined behavior in safe
Rust.

Expected behavior:

- `get_pair_mut` must never return overlapping `&mut` references.
- When `i == j`, the method should still return valid references (e.g. by
  returning two copies of the value, or by using a split-based approach).
- The `SplitVec` type in the `utils` crate should expose a `get_two_mut(i, j)`
  method that uses `split_at_mut` (or equivalent) to guarantee non-overlap.

The `utils` crate lives in `utils/src/lib.rs`; `get_pair_mut` lives in
`core/src/lib.rs`.
