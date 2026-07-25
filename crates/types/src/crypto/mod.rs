//! Cryptographic types and helpers.
//!
//! Re-exports the Ed25519 vendor crypto types from `radix_common::crypto`
//! (the transaction path) and adds workspace-level helpers split across:
//!
//! - [`keys`]: Ed25519 keypair generation for the transaction path.
//! - [`batch_verify`]: batch verification for Ed25519 transaction signatures.

pub mod batch_verify;
pub mod keys;

pub use radix_common::crypto::{
    Ed25519PrivateKey, Ed25519PublicKey, Ed25519Signature, verify_ed25519,
};
