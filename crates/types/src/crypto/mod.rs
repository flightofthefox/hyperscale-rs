//! Cryptographic types and helpers.
//!
//! Re-exports the Ed25519 private key from `radix_common::crypto` (the
//! transaction path) and adds [`keys`], keypair generation for that same
//! path.

pub mod keys;

pub use radix_common::crypto::Ed25519PrivateKey;
