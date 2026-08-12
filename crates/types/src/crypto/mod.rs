//! Cryptographic types and helpers for the transaction path.
//!
//! [`ed25519`] and [`secp256k1`] are the schemes themselves; [`keys`] is
//! keypair generation over them. [`AccountSigner`] is what the envelope
//! asks of any of them.

pub mod ed25519;
pub mod keys;
pub mod secp256k1;

pub use ed25519::{Ed25519PrivateKey, Ed25519PublicKey, Ed25519Signature, verify_ed25519};
use hyperscale_vm_types::SchemeId;
pub use secp256k1::{
    Secp256k1PrivateKey, Secp256k1PublicKey, Secp256k1Signature, verify_secp256k1,
};

/// A key that can put a scheme-tagged signature on an envelope.
///
/// One method per thing the envelope records: which scheme signed, the key
/// that verifies under it, and the signature. Nothing here chooses a
/// digest — the caller passes the 32 bytes the envelope already commits
/// to, and what a scheme does with them is the scheme's own business:
/// ed25519 signs them as a message, ECDSA takes them as its prehash.
pub trait AccountSigner {
    /// The scheme this key signs under.
    fn scheme(&self) -> SchemeId;

    /// The public key, in the width and encoding its scheme registers.
    fn public_key_bytes(&self) -> Vec<u8>;

    /// Sign the 32 bytes the caller commits to.
    fn sign_digest(&self, digest: &[u8; 32]) -> Vec<u8>;
}

impl AccountSigner for Ed25519PrivateKey {
    fn scheme(&self) -> SchemeId {
        SchemeId::ED25519
    }

    fn public_key_bytes(&self) -> Vec<u8> {
        self.public_key().0.to_vec()
    }

    fn sign_digest(&self, digest: &[u8; 32]) -> Vec<u8> {
        self.sign(digest).0.to_vec()
    }
}

impl AccountSigner for Secp256k1PrivateKey {
    fn scheme(&self) -> SchemeId {
        SchemeId::SECP256K1
    }

    fn public_key_bytes(&self) -> Vec<u8> {
        self.public_key().0.to_vec()
    }

    fn sign_digest(&self, digest: &[u8; 32]) -> Vec<u8> {
        self.sign_prehash(digest).0.to_vec()
    }
}
