use utils::SplitVec;

/// Returns mutable references to two elements in the vec by index.
///
/// BUG: This naively indexes the slice twice, which the borrow checker
/// should reject — but we have wrapped the vec in a SplitVec that provides
/// index-based access. The current SplitVec implementation returns
/// overlapping references when i == j or when the indices refer to adjacent
/// elements, which is unsound.
///
/// FIX: Use `slice::split_at_mut` (or `SplitVec::get_two_mut`) to obtain
/// two non-overlapping mutable references.
pub fn get_pair_mut(v: &mut SplitVec<i32>, i: usize, j: usize) -> (&mut i32, &mut i32) {
    // This is the broken implementation: it calls get_mut twice, which
    // produces overlapping mutable references when i == j or when the
    // underlying SplitVec doesn't enforce non-overlap.
    let a = v.get_mut(i).unwrap();
    let b = v.get_mut(j).unwrap();
    (a, b)
}
