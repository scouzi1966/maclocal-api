use core::get_pair_mut;
use utils::SplitVec;

#[test]
fn test_two_mut_different_indices() {
    let mut v: SplitVec<i32> = SplitVec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    let (a, b) = get_pair_mut(&mut v, 0, 2);
    *a += 10;
    *b += 20;
    drop(a);
    drop(b);
    assert_eq!(*v.get(0).unwrap(), 11);
    assert_eq!(*v.get(2).unwrap(), 23);
}

#[test]
fn test_two_mut_reversed_indices() {
    let mut v: SplitVec<i32> = SplitVec::new();
    v.push(10);
    v.push(20);
    v.push(30);
    // j < i: split_at_mut still works correctly.
    let (a, b) = get_pair_mut(&mut v, 2, 0);
    *a += 5;
    *b += 1;
    drop(a);
    drop(b);
    assert_eq!(*v.get(2).unwrap(), 35);
    assert_eq!(*v.get(0).unwrap(), 11);
}
