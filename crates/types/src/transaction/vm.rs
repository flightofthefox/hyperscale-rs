//! The VM seam: the envelope re-exported, its crypto binding, and the
//! derivation trait admission runs through.
//!
//! [`TransactionEnvelope`] and its body live in `hyperscale-vm-types` —
//! the envelope is the VM's artifact, and its signed content is defined
//! there through the derived preimage. What binds here is what the leaf
//! crate deliberately does not know: the protocol hash and curve behind
//! [`EnvelopeExt`], the clock behind the validity window, and the
//! workspace's admission vocabulary behind [`VmStatics`].

use std::sync::OnceLock;

use blake3::Hasher as Blake3;
use hyperscale_hbor::HborSigned;
pub use hyperscale_vm_types::{
    MAX_MESSAGE_LEN, MAX_SUBINTENTS, Mode, SubintentSig, TransactionBody, TransactionEnvelope,
};
use thiserror::Error;

use crate::crypto::{Ed25519PrivateKey, Ed25519PublicKey, Ed25519Signature, verify_ed25519};
use crate::{Address, DeclaredKey, Hash, TimestampRange, WeightedTimestamp};

/// The workspace's crypto and clock binding for the envelope.
///
/// The envelope defines its signed content — the preimage — and this
/// trait turns it into signatures and windows with the protocol's own
/// hash, curve, and time vocabulary.
pub trait EnvelopeExt: Sized {
    /// The domain-separated hash of the envelope's signed content —
    /// everything but the composer's own key and signature. This is
    /// also the identity fresh derivations root at: distinct signed
    /// envelopes never mint the same fresh key.
    fn signing_hash(&self) -> Hash;

    /// Sign the envelope's content with the composer's key, filling the
    /// signer and signature fields.
    #[must_use]
    fn sign(self, key: &Ed25519PrivateKey) -> Self;

    /// Whether the composer's signature covers the envelope content
    /// under the signer's key.
    fn signature_is_valid(&self) -> bool;

    /// The signed validity window as the wire's range form.
    fn validity_window(&self) -> TimestampRange;
}

impl EnvelopeExt for TransactionEnvelope {
    fn signing_hash(&self) -> Hash {
        let preimage = self
            .signing_bytes()
            .expect("an envelope within its caps encodes");
        let mut hasher = Blake3::new();
        hasher.update(&preimage);
        Hash::from_hash_bytes(hasher.finalize().as_bytes())
    }

    fn sign(mut self, key: &Ed25519PrivateKey) -> Self {
        let hash = self.signing_hash();
        self.signer = key.public_key().0;
        self.signature = key.sign(hash.as_bytes()).0;
        self
    }

    fn signature_is_valid(&self) -> bool {
        let hash = self.signing_hash();
        verify_ed25519(
            hash.as_bytes(),
            &Ed25519PublicKey(self.signer),
            &Ed25519Signature(self.signature),
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
    /// One declaration hash per bound subintent, in tree order.
    pub subintent_hashes: Vec<[u8; 32]>,
    /// The local half of the fee payer's native-resource vault cell —
    /// the substate the payer shard's reservation check reads and the
    /// fee settlement debits. The owner half is the envelope's
    /// `fee_payer`.
    pub fee_vault_local: [u8; 16],
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

/// Why VM static derivation refused an envelope. Deterministic: every
/// node reaches the identical verdict for the same bytes.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("vm static derivation failed: {0}")]
pub struct VmStaticsError(pub String);

/// The seam VM admission derives through.
///
/// Decode the envelope tree, admit it, and route its effect sets into
/// the workspace vocabulary. The effects bridge implements it over the
/// genesis-static metadata; node wiring installs it at boot.
pub trait VmStatics: Send + Sync {
    /// Derive the envelope's routing identity and subintent claims, or
    /// refuse it.
    ///
    /// # Errors
    ///
    /// [`VmStaticsError`] on an undecodable or inadmissible envelope,
    /// a subintent signature list that does not match the tree, or a
    /// bound signer address the matching public key does not derive.
    fn derive(&self, vm: &TransactionEnvelope) -> Result<Derived, VmStaticsError>;

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
    fn absorb_committed_cell(&self, owner: [u8; 16], local: [u8; 16], value: &[u8]) {
        let _ = (owner, local, value);
    }
}

static VM_STATICS: OnceLock<Box<dyn VmStatics>> = OnceLock::new();

/// Install the process-wide VM statics implementation. The first
/// installation wins; later calls are ignored, so tests and node boot can
/// both install without coordination.
pub fn install_vm_statics(statics: Box<dyn VmStatics>) {
    let _ = VM_STATICS.set(statics);
}

/// Whether a VM statics implementation is installed.
#[must_use]
pub fn vm_statics_installed() -> bool {
    VM_STATICS.get().is_some()
}

/// The installed statics.
///
/// # Panics
///
/// If none is installed — transactions cannot exist in a process that
/// never wired the derivation seam.
pub fn vm_statics() -> &'static dyn VmStatics {
    VM_STATICS
        .get()
        .expect("VM statics not installed; node wiring installs the effects-bridge derivation")
        .as_ref()
}
