//! The scheme's single derivation core, shared by signer and verifier.
//!
//! Verification recomputes what signing produced, so both sides call
//! into these; the domain tags keep public keys, signatures, and
//! aggregates from colliding across roles.

use blake3::Hasher;
use hyperscale_crypto::{AggregateSignature, ConsensusPublicKey, ConsensusSignature};

const DOMAIN_MOCK_PK: &[u8] = b"HYPERSCALE_MOCK_PK_v1";
const DOMAIN_MOCK_SIG: &[u8] = b"HYPERSCALE_MOCK_SIG_v1";
const DOMAIN_MOCK_AGG: &[u8] = b"HYPERSCALE_MOCK_AGG_v1";

/// Fill an `N`-byte container from the hasher's extendable output.
fn fill<const N: usize>(hasher: &Hasher) -> [u8; N] {
    let mut out = [0u8; N];
    hasher.finalize_xof().fill(&mut out);
    out
}

/// `pk = BLAKE3(DOMAIN_PK ‖ seed)`, widened to the 48-byte container.
pub fn public_key(seed: &[u8; 32]) -> ConsensusPublicKey {
    let mut h = Hasher::new();
    h.update(DOMAIN_MOCK_PK);
    h.update(seed);
    ConsensusPublicKey::new(fill(&h))
}

/// `sig = BLAKE3(DOMAIN_SIG ‖ pk ‖ message)`, widened to the 96-byte
/// container. Recomputable from the public key — the scheme's
/// deliberate non-unforgeability.
pub fn signature(pk: &ConsensusPublicKey, message: &[u8]) -> ConsensusSignature {
    let mut h = Hasher::new();
    h.update(DOMAIN_MOCK_SIG);
    h.update(pk.as_bytes());
    h.update(message);
    ConsensusSignature::new(fill(&h))
}

/// Ordered fold of the input signatures. Order-sensitive by design —
/// see the crate docs.
pub fn aggregate(sigs: &[ConsensusSignature]) -> AggregateSignature {
    let mut h = Hasher::new();
    h.update(DOMAIN_MOCK_AGG);
    for sig in sigs {
        h.update(sig.as_bytes());
    }
    AggregateSignature::new(fill(&h))
}
