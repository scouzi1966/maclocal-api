use lib::{Hashable, Record};

#[test]
fn test_hashable_without_crypto_feature() {
    let r = Record::new(42, vec![1, 2, 3]);
    // This should compile and work without the crypto feature.
    let h = r.hash_value();
    assert_ne!(h, 0);
}

#[test]
fn test_record_new() {
    let r = Record::new(1, vec![]);
    assert_eq!(r.id, 1);
}
