//! BLS12-381 (min-pk) implementation of the consensus crypto interface.
//!
//! The only crate that interprets the role newtypes' bytes as curve
//! points: public keys are compressed G1, signatures and VRF proofs
//! compressed G2, aggregates the G2 sum of their inputs. Rogue-key
//! safety for the unvalidated pubkey aggregation in
//! [`Verifier::verify_aggregate_same_message`] rests on validator
//! registration proving possession of every key (genesis keys are
//! operator-trusted config).

mod keys;
mod signer;
mod verifier;

pub use keys::{bls_keypair_from_seed, generate_bls_keypair};
pub use radix_common::crypto::{Bls12381G1PrivateKey, Bls12381G1PublicKey, Bls12381G2Signature};
pub use signer::{BlsSigner, InvalidKeyBytes};
pub use verifier::BlsVerifier;
