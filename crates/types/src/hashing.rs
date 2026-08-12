//! The protocol hash behind the VM's hashing seam.
//!
//! Blake3 over the length-framed domain and parts: pure, and framed so
//! that moving bytes across a part boundary always changes the digest.
//!
//! It sits with the wire types rather than with the effects binding
//! because both need it — an envelope's signing digest is taken here, and
//! the effect vocabulary derives every address and child key through the
//! same seam. Two definitions would be two identities for one value, and
//! the two would drift exactly once.

use blake3::Hasher as Blake3;
use hyperscale_hbor::hash::{Hash32, Hasher};

/// The protocol hash: blake3 over the length-framed domain and parts.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProtocolHasher;

impl Hasher for ProtocolHasher {
    fn hash(&self, domain: &[u8], parts: &[&[u8]]) -> Hash32 {
        let mut hasher = Blake3::new();
        hasher.update(&(domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        for part in parts {
            hasher.update(&(part.len() as u64).to_le_bytes());
            hasher.update(part);
        }
        Hash32(*hasher.finalize().as_bytes())
    }
}
