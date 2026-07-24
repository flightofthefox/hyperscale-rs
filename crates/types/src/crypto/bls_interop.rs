//! Byte conversions between the role newtypes and the BLS scheme types.
//!
//! The role newtypes are opaque byte containers; these helpers are the
//! narrow boundary where consensus code hands them to the concrete BLS
//! operations. They live here only until the BLS ops move behind the
//! `Verifier`/`Signer` traits in their own impl crate.

use hyperscale_crypto::{AggregateSignature, ConsensusPublicKey, ConsensusSignature};
use radix_common::crypto::{Bls12381G1PublicKey, Bls12381G2Signature};

/// View a consensus public key as a BLS public key.
#[must_use]
pub const fn bls_pk(pk: &ConsensusPublicKey) -> Bls12381G1PublicKey {
    Bls12381G1PublicKey(*pk.as_bytes())
}

/// Wrap a BLS public key's bytes in the role newtype.
#[must_use]
pub const fn pk_from_bls(pk: &Bls12381G1PublicKey) -> ConsensusPublicKey {
    ConsensusPublicKey::new(pk.0)
}

/// View a consensus signature as a BLS signature.
#[must_use]
pub const fn bls_sig(sig: &ConsensusSignature) -> Bls12381G2Signature {
    Bls12381G2Signature(*sig.as_bytes())
}

/// Wrap a BLS signature's bytes in the role newtype.
#[must_use]
pub const fn sig_from_bls(sig: &Bls12381G2Signature) -> ConsensusSignature {
    ConsensusSignature::new(sig.0)
}

/// View an aggregate signature as a BLS signature.
#[must_use]
pub const fn bls_agg(agg: &AggregateSignature) -> Bls12381G2Signature {
    Bls12381G2Signature(*agg.as_bytes())
}

/// Wrap an aggregated BLS signature's bytes in the role newtype.
#[must_use]
pub const fn agg_from_bls(agg: &Bls12381G2Signature) -> AggregateSignature {
    AggregateSignature::new(agg.0)
}
