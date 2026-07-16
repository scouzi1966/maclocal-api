/// A data type that should work with or without the `crypto` feature.
///
/// BUG: The `Hashable` trait impl is behind `#[cfg(feature = "crypto")]`,
/// but the trait itself is always visible. With `--no-default-features`,
/// the impl is missing, so code that uses `Hashable` on `Record` fails to
/// compile.
///
/// FIX: The `Hashable` impl should NOT be gated on the `crypto` feature.
/// Only the actual cryptographic helpers (e.g. `hash_bytes`) should be
/// behind the feature gate. The trait impl itself is a fundamental
/// operation and must always be present.

pub trait Hashable {
    fn hash_value(&self) -> u64;
}

pub struct Record {
    pub id: u32,
    pub data: Vec<u8>,
}

impl Record {
    pub fn new(id: u32, data: Vec<u8>) -> Self {
        Self { id, data }
    }
}

// BUG: This impl is behind the wrong feature gate.
#[cfg(feature = "crypto")]
impl Hashable for Record {
    fn hash_value(&self) -> u64 {
        let mut h: u64 = self.id as u64;
        for &b in &self.data {
            h = h.wrapping_mul(31).wrapping_add(b as u64);
        }
        h
    }
}

/// This helper genuinely depends on the crypto feature.
#[cfg(feature = "crypto")]
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 1;
    for &b in data {
        h = h.wrapping_mul(31).wrapping_add(b as u64);
    }
    h
}
