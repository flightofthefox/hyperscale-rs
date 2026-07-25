//! Cryptographic types and helpers.
//!
//! Re-exports the Ed25519 vendor crypto types from `radix_common::crypto`
//! (the transaction path) and adds [`keys`], keypair generation for that
//! same path.

pub mod keys;

pub use radix_common::crypto::{
    Ed25519PrivateKey, Ed25519PublicKey, Ed25519Signature, verify_ed25519,
};
