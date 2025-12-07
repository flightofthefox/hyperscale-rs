//! Cryptographic key pairs and signatures.
//!
//! Supports:
//! - ED25519: Fast signing for general use
//! - BLS12-381: Signature aggregation for consensus efficiency

use sbor::prelude::*;
use std::fmt;

/// Supported key types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, BasicSbor)]
pub enum KeyType {
    /// ED25519 - Fast, widely supported.
    Ed25519,
    /// BLS12-381 - Supports signature aggregation.
    Bls12381,
}

/// A cryptographic key pair for signing.
#[derive(Clone)]
pub enum KeyPair {
    /// ED25519 key pair.
    Ed25519(ed25519_dalek::SigningKey),
    /// BLS12-381 key pair.
    Bls12381(blst::min_pk::SecretKey),
}

impl KeyPair {
    /// Generate a new random Ed25519 keypair.
    pub fn generate_ed25519() -> Self {
        let mut csprng = rand::rngs::OsRng;
        let signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);
        KeyPair::Ed25519(signing_key)
    }

    /// Generate a new random BLS12-381 keypair.
    pub fn generate_bls() -> Self {
        let mut ikm = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut ikm);
        let sk = blst::min_pk::SecretKey::key_gen(&ikm, &[]).unwrap();
        KeyPair::Bls12381(sk)
    }

    /// Generate a keypair from a seed (for testing/simulation).
    pub fn from_seed(key_type: KeyType, seed: &[u8; 32]) -> Self {
        match key_type {
            KeyType::Ed25519 => {
                let signing_key = ed25519_dalek::SigningKey::from_bytes(seed);
                KeyPair::Ed25519(signing_key)
            }
            KeyType::Bls12381 => {
                let sk = blst::min_pk::SecretKey::key_gen(seed, &[]).unwrap();
                KeyPair::Bls12381(sk)
            }
        }
    }

    /// Sign a message.
    pub fn sign(&self, message: &[u8]) -> Signature {
        match self {
            KeyPair::Ed25519(signing_key) => {
                use ed25519_dalek::Signer;
                let sig = signing_key.sign(message);
                Signature::Ed25519(sig.to_bytes().to_vec())
            }
            KeyPair::Bls12381(sk) => {
                let sig = sk.sign(message, &[], &[]);
                Signature::Bls12381(sig.to_bytes().to_vec())
            }
        }
    }

    /// Get the public key.
    pub fn public_key(&self) -> PublicKey {
        match self {
            KeyPair::Ed25519(signing_key) => {
                PublicKey::Ed25519(signing_key.verifying_key().to_bytes())
            }
            KeyPair::Bls12381(sk) => PublicKey::Bls12381(sk.sk_to_pk().to_bytes().to_vec()),
        }
    }
}

/// A public key for signature verification.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, BasicSbor)]
pub enum PublicKey {
    /// ED25519 public key (32 bytes).
    Ed25519([u8; 32]),
    /// BLS12-381 public key (48 bytes compressed).
    Bls12381(Vec<u8>),
}

impl PublicKey {
    /// Verify a signature.
    pub fn verify(&self, message: &[u8], signature: &Signature) -> bool {
        match (self, signature) {
            (PublicKey::Ed25519(pk_bytes), Signature::Ed25519(sig_bytes)) => {
                use ed25519_dalek::Verifier;
                let pk = match ed25519_dalek::VerifyingKey::from_bytes(pk_bytes) {
                    Ok(pk) => pk,
                    Err(_) => return false,
                };
                if sig_bytes.len() != 64 {
                    return false;
                }
                let sig_array: [u8; 64] = match sig_bytes.as_slice().try_into() {
                    Ok(arr) => arr,
                    Err(_) => return false,
                };
                let sig = ed25519_dalek::Signature::from_bytes(&sig_array);
                pk.verify(message, &sig).is_ok()
            }
            (PublicKey::Bls12381(pk_bytes), Signature::Bls12381(sig_bytes)) => {
                let pk = match blst::min_pk::PublicKey::from_bytes(pk_bytes) {
                    Ok(pk) => pk,
                    Err(_) => return false,
                };
                let sig = match blst::min_pk::Signature::from_bytes(sig_bytes) {
                    Ok(sig) => sig,
                    Err(_) => return false,
                };
                sig.verify(true, message, &[], &[], &pk, true) == blst::BLST_ERROR::BLST_SUCCESS
            }
            _ => false, // Mismatched types
        }
    }

    /// Aggregate multiple BLS public keys.
    pub fn aggregate_bls(pubkeys: &[PublicKey]) -> Result<Self, AggregateError> {
        if pubkeys.is_empty() {
            return Err(AggregateError::Empty);
        }

        let bls_pks: Vec<_> = pubkeys
            .iter()
            .filter_map(|pk| match pk {
                PublicKey::Bls12381(bytes) => blst::min_pk::PublicKey::from_bytes(bytes).ok(),
                _ => None,
            })
            .collect();

        if bls_pks.len() != pubkeys.len() {
            return Err(AggregateError::MixedTypes);
        }

        let refs: Vec<&blst::min_pk::PublicKey> = bls_pks.iter().collect();
        let agg = blst::min_pk::AggregatePublicKey::aggregate(&refs, false)
            .map_err(|_| AggregateError::AggregationFailed)?;

        Ok(PublicKey::Bls12381(agg.to_public_key().to_bytes().to_vec()))
    }
}

impl fmt::Debug for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PublicKey::Ed25519(bytes) => {
                write!(f, "PublicKey::Ed25519({})", hex::encode(bytes))
            }
            PublicKey::Bls12381(bytes) => {
                let hex = hex::encode(bytes);
                write!(
                    f,
                    "PublicKey::Bls12381({}..{})",
                    &hex[..8],
                    &hex[hex.len() - 8..]
                )
            }
        }
    }
}

/// A cryptographic signature.
#[derive(Clone, PartialEq, Eq, BasicSbor)]
pub enum Signature {
    /// ED25519 signature (64 bytes).
    Ed25519(Vec<u8>),
    /// BLS12-381 signature (96 bytes compressed).
    Bls12381(Vec<u8>),
}

impl Signature {
    /// Create a zero/placeholder signature for testing.
    pub fn zero() -> Self {
        Signature::Ed25519(vec![0u8; 64])
    }

    /// Get signature as bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Signature::Ed25519(bytes) => bytes.to_vec(),
            Signature::Bls12381(bytes) => bytes.clone(),
        }
    }

    /// Get signature as byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Signature::Ed25519(bytes) => bytes.as_slice(),
            Signature::Bls12381(bytes) => bytes.as_slice(),
        }
    }

    /// Aggregate multiple BLS signatures.
    pub fn aggregate_bls(signatures: &[Signature]) -> Result<Self, AggregateError> {
        if signatures.is_empty() {
            return Err(AggregateError::Empty);
        }

        let bls_sigs: Vec<_> = signatures
            .iter()
            .filter_map(|s| match s {
                Signature::Bls12381(bytes) => blst::min_pk::Signature::from_bytes(bytes).ok(),
                _ => None,
            })
            .collect();

        if bls_sigs.len() != signatures.len() {
            return Err(AggregateError::MixedTypes);
        }

        let refs: Vec<&blst::min_pk::Signature> = bls_sigs.iter().collect();
        let agg = blst::min_pk::AggregateSignature::aggregate(&refs, true)
            .map_err(|_| AggregateError::AggregationFailed)?;

        Ok(Signature::Bls12381(agg.to_signature().to_bytes().to_vec()))
    }
}

impl fmt::Debug for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Signature::Ed25519(bytes) => {
                write!(f, "Signature::Ed25519({}..)", &hex::encode(bytes)[..16])
            }
            Signature::Bls12381(bytes) => {
                let hex = hex::encode(bytes);
                write!(f, "Signature::Bls12381({}..)", &hex[..16])
            }
        }
    }
}

/// Errors that can occur during aggregation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AggregateError {
    /// Empty list provided.
    #[error("Cannot aggregate empty list")]
    Empty,

    /// Mixed key/signature types.
    #[error("Cannot aggregate mixed types (ED25519 and BLS)")]
    MixedTypes,

    /// Aggregation operation failed.
    #[error("Aggregation failed")]
    AggregationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ed25519_sign_verify() {
        let keypair = KeyPair::generate_ed25519();
        let message = b"test message";

        let signature = keypair.sign(message);
        let pubkey = keypair.public_key();

        assert!(pubkey.verify(message, &signature));
    }

    #[test]
    fn test_ed25519_verify_fails_wrong_message() {
        let keypair = KeyPair::generate_ed25519();
        let message = b"test message";
        let wrong = b"wrong message";

        let signature = keypair.sign(message);
        let pubkey = keypair.public_key();

        assert!(!pubkey.verify(wrong, &signature));
    }

    #[test]
    fn test_bls_sign_verify() {
        let keypair = KeyPair::generate_bls();
        let message = b"test message";

        let signature = keypair.sign(message);
        let pubkey = keypair.public_key();

        assert!(pubkey.verify(message, &signature));
    }

    #[test]
    fn test_bls_aggregate_signatures() {
        let message = b"block hash";

        let keypair1 = KeyPair::generate_bls();
        let keypair2 = KeyPair::generate_bls();
        let keypair3 = KeyPair::generate_bls();

        let sig1 = keypair1.sign(message);
        let sig2 = keypair2.sign(message);
        let sig3 = keypair3.sign(message);

        let agg_sig = Signature::aggregate_bls(&[sig1, sig2, sig3]).unwrap();

        let pubkeys = vec![
            keypair1.public_key(),
            keypair2.public_key(),
            keypair3.public_key(),
        ];
        let agg_pubkey = PublicKey::aggregate_bls(&pubkeys).unwrap();

        assert!(agg_pubkey.verify(message, &agg_sig));
    }

    #[test]
    fn test_hyperscale_keypair_from_seed() {
        let seed = [42u8; 32];

        let kp1 = KeyPair::from_seed(KeyType::Ed25519, &seed);
        let kp2 = KeyPair::from_seed(KeyType::Ed25519, &seed);

        let msg = b"test";
        assert_eq!(kp1.sign(msg).to_bytes(), kp2.sign(msg).to_bytes());
        assert_eq!(kp1.public_key(), kp2.public_key());
    }
}
