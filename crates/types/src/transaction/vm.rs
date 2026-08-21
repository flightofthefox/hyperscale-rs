//! The VM seam: the envelope re-exported, its crypto binding, and the
//! derivation trait admission runs through.
//!
//! [`TransactionEnvelope`] and its body live in `hyperscale-vm-types` —
//! the envelope is the VM's artifact, and its signed content is defined
//! there through the derived preimage. What binds here is what the leaf
//! crate deliberately does not know: the protocol hash and the signature
//! arithmetic behind [`EnvelopeExt`], the clock behind the validity
//! window, and the workspace's admission vocabulary behind [`Derivation`].

use std::sync::OnceLock;

pub use hyperscale_vm_types::{
    AccountSigner, MAX_MESSAGE_LEN, MAX_SUBINTENTS, Mode, SchemeId, SchemeVerifier, SubintentSig,
    TransactionBody, TransactionEnvelope,
};
use thiserror::Error;

use crate::crypto::{
    Ed25519PublicKey, Ed25519Signature, Secp256k1PublicKey, Secp256k1Signature, verify_ed25519,
    verify_ml_dsa_65, verify_secp256k1,
};
use crate::{
    Address, DeclaredKey, Hash, PrincipalAddr, ProtocolHasher, TimestampRange, WeightedTimestamp,
};

/// The arithmetic behind the VM's scheme registry.
///
/// The registry says how wide a scheme's material is and what verifying it
/// costs; this says what verifying it *means*. Every scheme the protocol
/// accepts answers here, and one it does not — an id no registry entry
/// claims, or material of a width its entry does not give it — answers
/// `false` alongside a signature that is simply wrong.
///
/// Every message the transaction path presents is a 32-byte hash, and no
/// scheme here digests it a second time: ECDSA takes it as its prehash,
/// and ML-DSA signs it as a message under the pure variant. A message of
/// any other width is refused by the schemes that require the digest
/// exactly, which is the same refusal as any other material this verifier
/// cannot read.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProtocolVerifier;

impl SchemeVerifier for ProtocolVerifier {
    fn verify(&self, scheme: SchemeId, key: &[u8], signature: &[u8], message: &[u8]) -> bool {
        let Some(spec) = scheme.spec() else {
            return false;
        };
        if !spec.admits(key, signature) {
            return false;
        }
        match scheme {
            SchemeId::ED25519 => {
                let (Ok(key), Ok(signature)) = (key.try_into(), signature.try_into()) else {
                    return false;
                };
                verify_ed25519(
                    message,
                    &Ed25519PublicKey(key),
                    &Ed25519Signature(signature),
                )
            }
            SchemeId::SECP256K1 => {
                let (Ok(key), Ok(signature), Ok(prehash)) = (
                    key.try_into(),
                    signature.try_into(),
                    <&[u8; 32]>::try_from(message),
                ) else {
                    return false;
                };
                verify_secp256k1(
                    prehash,
                    &Secp256k1PublicKey(key),
                    &Secp256k1Signature(signature),
                )
            }
            SchemeId::ML_DSA_65 => verify_ml_dsa_65(message, key, signature),
            _ => false,
        }
    }
}

/// The workspace's crypto and clock binding for the envelope.
///
/// The envelope defines its signed content — the preimage — and this
/// trait turns it into signatures and windows with the protocol's own
/// hash, signature, and time vocabulary.
pub trait EnvelopeExt: Sized {
    /// The domain-separated hash of the envelope's signed content —
    /// everything but the composer's own key and signature. This is
    /// also the identity fresh derivations root at: distinct signed
    /// envelopes never mint the same fresh key.
    fn signing_hash(&self) -> Hash;

    /// Sign the envelope's content with the composer's key, filling the
    /// scheme, signer, and signature fields.
    #[must_use]
    fn sign<S: AccountSigner>(self, key: &S) -> Self;

    /// Whether the composer's signature covers the envelope content
    /// under the signer's key, in the scheme the envelope names.
    fn signature_is_valid(&self) -> bool;

    /// The signed validity window as the wire's range form.
    fn validity_window(&self) -> TimestampRange;
}

impl EnvelopeExt for TransactionEnvelope {
    fn signing_hash(&self) -> Hash {
        Hash::from_hash_bytes(&self.signing_digest(&ProtocolHasher))
    }

    fn sign<S: AccountSigner>(mut self, key: &S) -> Self {
        // The scheme is signed content, so it is stamped before the
        // digest is taken; the key and signature are not, and are filled
        // after. `manifest-builder`'s own signing tier does the same over
        // its own hasher — this is the protocol hash's spelling of it,
        // for the fixtures and call sites that already hold a key.
        self.signer_scheme = key.scheme();
        let digest = self.signing_digest(&ProtocolHasher);
        self.signer = key.public_key_bytes();
        self.signature = key.sign_digest(&digest);
        self
    }

    fn signature_is_valid(&self) -> bool {
        let hash = self.signing_hash();
        ProtocolVerifier.verify(
            self.signer_scheme,
            &self.signer,
            &self.signature,
            hash.as_bytes(),
        )
    }

    fn validity_window(&self) -> TimestampRange {
        TimestampRange::new(
            WeightedTimestamp::from_millis(self.validity_start_ms),
            WeightedTimestamp::from_millis(self.validity_end_ms),
        )
    }
}

/// A transaction's derived routing identity.
///
/// Admission conflict keys and the owner prefixes that place it on
/// shards. A pure function of the envelope and genesis-static metadata —
/// derived locally at every node, never carried on the wire, so a
/// sender cannot claim a placement its content does not earn. Nullifier creation writes are in the write keys: committing a
/// subintent is an exclusive write at its canonical nullifier address.
/// Snapshot reads appear nowhere here: they are lock-free and
/// client-proven, so a snapshot-only shard is not a participant at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routing {
    /// Conflict keys for fresh reads — the shared admission class.
    pub read_keys: Vec<DeclaredKey>,
    /// Conflict keys for every mutation (writes, deltas, reserves).
    pub write_keys: Vec<DeclaredKey>,
    /// Owner prefixes behind `read_keys`, deduplicated ascending.
    pub read_prefixes: Vec<Address>,
    /// Owner prefixes behind `write_keys`, deduplicated ascending.
    pub write_prefixes: Vec<Address>,
    /// The keys whose committed values counterpart shards must carry:
    /// fresh reads plus read-modify-write priors. Deltas, blind writes,
    /// and reserves provision nothing.
    pub provision_keys: Vec<DeclaredKey>,
    /// Owner prefixes behind `provision_keys`, deduplicated ascending —
    /// the tick's provision dependency set routes on these.
    pub provision_prefixes: Vec<Address>,
    /// What each declared key is accessed under, ascending by key then
    /// mode, with the reservation amount carried where there is one.
    ///
    /// The key sets above answer "does this transaction touch that
    /// cell"; this answers "how", which is what decides whether two
    /// transactions touching one cell can be in flight together. A
    /// reservation also has to say how much, because feasibility is
    /// judged against committed balance less what is already held, and
    /// the amount is statically declared for exactly that reason.
    ///
    /// One key can appear more than once: a manifest may declare several
    /// effects on one cell, and a payer reserving twice reserves the sum.
    pub declared_modes: Vec<(DeclaredKey, Mode)>,
}

impl Routing {
    /// Every owner prefix the transaction touches, ascending, deduplicated.
    #[must_use]
    pub fn all_prefixes(&self) -> Vec<Address> {
        let mut prefixes: Vec<Address> = self
            .read_prefixes
            .iter()
            .chain(self.write_prefixes.iter())
            .copied()
            .collect();
        prefixes.sort_unstable();
        prefixes.dedup();
        prefixes
    }

    /// What `key` is declared under by this transaction, if anything.
    ///
    /// A key declared more than once yields each declaration: the caller
    /// decides whether it wants every mode or the sum of the amounts.
    pub fn modes_for<'a>(&'a self, key: &'a DeclaredKey) -> impl Iterator<Item = Mode> + 'a {
        self.declared_modes
            .iter()
            .filter(move |(declared, _)| declared == key)
            .map(|(_, mode)| *mode)
    }
}

/// Everything the bridge derives from an envelope.
///
/// The routing identity plus the declaration hash each subintent
/// signature must cover, in tree order. Derivation has already checked
/// that every bound signer address is the one the matching public key
/// derives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Derived {
    /// The routing identity.
    pub routing: Routing,
    /// The principal the envelope's own signature opens — the identity
    /// the root intent's signature badge carries.
    ///
    /// Deliberately not compared against the envelope's `fee_payer`
    /// here: whether the payer's rule admits this identity is the payer
    /// shard's verdict, taken where the payer's state is, so derivation
    /// records the identity and leaves the judgement to the one shard
    /// that can reach it.
    pub signer: PrincipalAddr,
    /// One declaration hash per bound subintent, in tree order.
    pub subintent_hashes: Vec<[u8; 32]>,
    /// The local half of the fee payer's native-resource vault cell —
    /// the substate the payer shard's reservation check reads and the
    /// fee settlement debits. The owner half is the envelope's
    /// `fee_payer`.
    pub fee_vault_local: [u8; 16],
    /// The local half of the payer's stored-authority cell, read beside
    /// the vault at the same anchored height: the reservation engages
    /// only for a signer the payer's rule admits.
    pub auth_cell_local: [u8; 16],
    /// The content addresses of every package the manifest's calls run,
    /// deduplicated. What the execution gate holds a candidate to: a
    /// shard dispatches the transaction only once it holds all of them.
    pub packages: Vec<Hash>,
    /// What including this transaction costs a block, in work units.
    ///
    /// A fixed admit-and-track charge, the declared footprint, and the
    /// signed gas limit. The fixed term is what makes a budget over this
    /// quantity bound the *number* of transactions in the drain as well
    /// as their weight: a minimal declaration prices at almost nothing
    /// and a gas limit may be zero, while every committed transaction
    /// costs a tick entry, a tick-chain entry, a receipt and mempool
    /// tracking whatever it declared.
    ///
    /// Derived locally from the manifest and published metadata like
    /// every other routing quantity — nothing about it travels on the
    /// wire, so a sender cannot understate it.
    pub work: u64,
}

/// Why a derivation did not answer.
///
/// Two different things, and a caller has to tell them apart. A refusal
/// is a verdict every node reaches alike for the same bytes. A gap is
/// this node's alone: derivation resolves a call target through the
/// records it has seen commit, and one it has not seen resolves nothing
/// here and resolves fine wherever the seal already landed. The first is
/// the envelope's fault forever; the second closes when the record
/// arrives.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum DerivationError {
    /// The envelope is inadmissible, on terms every node agrees on.
    #[error("derivation refused the envelope: {0}")]
    Refused(String),
    /// Component records this node holds none of, so it cannot say what
    /// the envelope declares. Not a verdict: the addresses are what a
    /// fetch asks its owning shard for, and derivation answers once they
    /// are seated.
    #[error("derivation wants records this node has not seen: {0:?}")]
    Unresolved(Vec<Address>),
}

impl DerivationError {
    /// The records this node would need before derivation could answer,
    /// empty for a refusal.
    #[must_use]
    pub fn unresolved(&self) -> &[Address] {
        match self {
            Self::Refused(_) => &[],
            Self::Unresolved(addresses) => addresses,
        }
    }
}

/// The seam VM admission derives through: what one node can answer.
///
/// Decode the envelope tree, admit it, and route its effect sets into
/// the workspace vocabulary. The effects bridge implements it over the
/// records and packages that node has seen commit.
///
/// Split from [`ProtocolStatics`] on the only line that matters, which
/// is whether two nodes can answer differently. Everything here reads a
/// cache the chain fills, so a node that has not seen a seal commit
/// resolves nothing a node that has resolves fine — a real difference,
/// and one a single shared installation would erase.
pub trait Derivation: Send + Sync {
    /// Derive the envelope's routing identity and subintent claims, or
    /// refuse it.
    ///
    /// # Errors
    ///
    /// [`DerivationError`] on an undecodable or inadmissible envelope,
    /// a subintent signature list that does not match the tree, or a
    /// bound signer address the matching public key does not derive.
    fn derive(&self, vm: &TransactionEnvelope) -> Result<Derived, DerivationError>;

    /// Offer one committed cell to the published-package cache.
    ///
    /// Called for every cell a block commits, on the commit path and
    /// on the sync path alike, because both derive their state from the
    /// same block content. What makes a cell a package is a property of
    /// its own bytes, so the implementation decides — this seam carries
    /// no VM vocabulary and no notion of what a package is.
    ///
    /// Feeding the cache from committed state rather than from execution
    /// is what keeps routing identical across replicas: a package is
    /// usable by transactions admitted after its block commits, and a
    /// validator whose cache lagged would refuse what its peers admit.
    fn absorb_committed_cell(&self, owner: [u8; 32], local: [u8; 16], value: &[u8]) {
        let _ = (owner, local, value);
    }
}

/// The seam the VM answers protocol questions through: what every node
/// answers alike.
///
/// A verdict over bytes a caller supplies, or a classification of bytes
/// a block committed. Neither reads anything a node accumulates, so two
/// nodes cannot disagree and nothing is gained by holding one of these
/// per node.
///
/// A trait rather than free functions because the dependency runs from
/// the effects bridge to this crate: consensus cannot call into the VM,
/// so the VM installs its answers here.
pub trait ProtocolStatics: Send + Sync {
    /// Whether `payer`'s rule admits `signer`, given the payer's
    /// stored-authority cell as read at the caller's own anchored
    /// height — `None` or empty meaning absent — and the weighted-time
    /// instant the verdict is judged at, in milliseconds.
    ///
    /// The rule's encoding is the VM's fact, so consensus hands the
    /// bytes across this seam and stays blind to them. Absent means the
    /// account is virtual and the rule is the identity its address
    /// derives; stored bytes that do not decode admit nobody, the same
    /// fail-closed verdict the execution gate gives them. The clock is
    /// what lets a matured recovery proposal govern here with nothing
    /// applying it: voters pass the judged block's own parent-QC
    /// weighted timestamp — the same instant its transactions execute
    /// under if it commits them — the proposal builder the parent QC it
    /// builds on, and mempool admission its local advisory instant.
    fn rule_admits(
        &self,
        auth_cell: Option<&[u8]>,
        payer: PrincipalAddr,
        signer: PrincipalAddr,
        clock_ms: u64,
    ) -> bool {
        let _ = clock_ms;
        match auth_cell {
            None | Some([]) => payer == signer,
            Some(_) => false,
        }
    }

    /// The content address of the package this committed cell publishes,
    /// or `None` for every other cell.
    ///
    /// What makes a cell a package is a property of its own bytes — the
    /// value re-derives the cell's local key under its owner — so the
    /// implementation decides, and this seam carries no VM vocabulary.
    /// Storage backends consult it to index a committed package's
    /// artifact bytes beside the commit that carries them.
    fn package_cell(&self, owner: [u8; 32], local: [u8; 16], value: &[u8]) -> Option<Hash> {
        let _ = (owner, local, value);
        None
    }
}

static PROTOCOL_STATICS: OnceLock<Box<dyn ProtocolStatics>> = OnceLock::new();

/// Install the process-wide protocol answers. The first installation
/// wins, so tests and node boot can both install without coordination.
///
/// There is no counterpart for [`Derivation`]: a derivation is a node's
/// own, and a process running several nodes holds one each.
pub fn install_protocol_statics(statics: Box<dyn ProtocolStatics>) {
    let _ = PROTOCOL_STATICS.set(statics);
}

/// Whether the protocol answers are installed.
#[must_use]
pub fn protocol_statics_installed() -> bool {
    PROTOCOL_STATICS.get().is_some()
}

/// The installed protocol answers.
///
/// # Panics
///
/// If none are installed — transactions cannot exist in a process that
/// never wired the VM seam.
pub fn protocol_statics() -> &'static dyn ProtocolStatics {
    PROTOCOL_STATICS
        .get()
        .expect("protocol statics not installed; node wiring installs the effects-bridge answers")
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::{AccountSigner, ProtocolVerifier, SchemeId, SchemeVerifier};
    use crate::crypto::{Ed25519PrivateKey, MlDsa65PrivateKey, Secp256k1PrivateKey};

    const DIGEST: [u8; 32] = [3u8; 32];

    fn ed() -> Ed25519PrivateKey {
        Ed25519PrivateKey::from_bytes(&[3u8; 32]).expect("32 bytes")
    }

    fn secp() -> Secp256k1PrivateKey {
        Secp256k1PrivateKey::from_bytes(&[3u8; 32]).expect("a scalar in range")
    }

    fn ml_dsa() -> MlDsa65PrivateKey {
        MlDsa65PrivateKey::from_bytes(&[3u8; 32]).expect("32 bytes")
    }

    fn signers() -> Vec<Box<dyn AccountSigner>> {
        vec![Box::new(ed()), Box::new(secp()), Box::new(ml_dsa())]
    }

    #[test]
    fn a_signature_verifies_under_the_scheme_it_was_made_in() {
        for signer in signers() {
            let key = signer.public_key_bytes();
            let signature = signer.sign_digest(&DIGEST);
            assert!(ProtocolVerifier.verify(signer.scheme(), &key, &signature, &DIGEST));
            assert!(!ProtocolVerifier.verify(signer.scheme(), &key, &signature, &[9u8; 32]));
        }
    }

    /// Material presented under a scheme that did not produce it verifies
    /// under neither: the widths disagree, and where they agree the curve
    /// does.
    #[test]
    fn a_signature_verifies_under_no_other_scheme() {
        for signer in signers() {
            let key = signer.public_key_bytes();
            let signature = signer.sign_digest(&DIGEST);
            for scheme in [SchemeId::ED25519, SchemeId::SECP256K1, SchemeId::ML_DSA_65] {
                assert_eq!(
                    ProtocolVerifier.verify(scheme, &key, &signature, &DIGEST),
                    scheme == signer.scheme(),
                );
            }
        }
    }

    /// A scheme no registry entry claims verifies nothing, whatever
    /// material is presented under it.
    #[test]
    fn an_unregistered_scheme_verifies_nothing() {
        for signer in signers() {
            let key = signer.public_key_bytes();
            let signature = signer.sign_digest(&DIGEST);
            for scheme in [SchemeId::NONE, SchemeId(4), SchemeId(u16::MAX)] {
                assert!(!ProtocolVerifier.verify(scheme, &key, &signature, &DIGEST));
            }
        }
    }

    /// Material of a width its scheme does not give it is refused before
    /// any curve arithmetic runs, so a short key is never padded out to
    /// one the curve would accept.
    #[test]
    fn material_of_the_wrong_width_refuses() {
        for signer in signers() {
            let scheme = signer.scheme();
            let key = signer.public_key_bytes();
            let signature = signer.sign_digest(&DIGEST);
            assert!(!ProtocolVerifier.verify(scheme, &key[..key.len() - 1], &signature, &DIGEST));
            assert!(!ProtocolVerifier.verify(
                scheme,
                &key,
                &signature[..signature.len() - 1],
                &DIGEST
            ));
            assert!(!ProtocolVerifier.verify(scheme, &[], &[], &DIGEST));
        }
    }

    /// ECDSA reads the message as its prehash, so a message that is not
    /// the curve's digest width is material this verifier cannot read.
    #[test]
    fn secp256k1_refuses_a_message_that_is_not_a_digest() {
        let signer = secp();
        let key = signer.public_key_bytes();
        let signature = signer.sign_digest(&DIGEST);
        assert!(ProtocolVerifier.verify(SchemeId::SECP256K1, &key, &signature, &DIGEST));
        assert!(!ProtocolVerifier.verify(SchemeId::SECP256K1, &key, &signature, &DIGEST[..31]));
    }
}
