//! Cryptographic types and helpers.
//!
//! Re-exports vendor crypto types from `radix_common::crypto` and adds
//! workspace-level helpers split across:
//!
//! - [`keys`]: Ed25519 keypair generation for the transaction path.
//! - [`batch_verify`]: batch verification for Ed25519 transaction signatures.
//! - [`bls_interop`]: byte conversions between role newtypes and the BLS
//!   scheme types, used by the signing helpers until signing moves behind
//!   the `Signer` trait.

pub mod batch_verify;
pub mod bls_interop;
pub mod keys;

pub use radix_common::crypto::{
    Bls12381G1PrivateKey, Bls12381G1PublicKey, Bls12381G2Signature, Ed25519PrivateKey,
    Ed25519PublicKey, Ed25519Signature, verify_ed25519,
};
