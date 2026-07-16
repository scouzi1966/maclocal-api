# `Record` fails to implement `Hashable` with `--no-default-features`

The `lib` crate (`splitdemo::lib`) defines a `Hashable` trait and a `Record`
struct. With default features (which include `crypto`), `Record` implements
`Hashable` and everything works.

When built with `--no-default-features`, code that calls
`Record::hash_value()` fails to compile because the `Hashable` impl for
`Record` is gated behind `#[cfg(feature = "crypto")]`.

The `crypto` feature should gate only the *cryptographic helper*
(`hash_bytes`), not the `Hashable` trait impl on `Record`. The impl itself
is a basic operation that must work regardless of feature configuration.

The `Hashable` impl and `Record` live in `lib/src/lib.rs`. The feature
definitions are in `lib/Cargo.toml`.
