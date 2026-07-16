/// A wrapper around `Vec<T>` that provides index-based mutable access.
///
/// BUG: The `get_mut` method returns a raw pointer cast to a mutable
/// reference. When called twice for the same index, this produces two
/// mutable references to the same data, which is undefined behavior.
///
/// FIX: Provide a `get_two_mut` method that uses `split_at_mut` to return
/// non-overlapping references, and change `core::get_pair_mut` to use it.
pub struct SplitVec<T> {
    data: Vec<T>,
}

impl<T> SplitVec<T> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push(&mut self, val: T) {
        self.data.push(val);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// UNSOUND: returns a mutable reference that can overlap with another
    /// reference obtained from the same method on the same index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }
}

impl<T: Default> SplitVec<T> {
    pub fn with_capacity(cap: usize) -> Self {
        Self { data: Vec::with_capacity(cap) }
    }
}
