//! [`Block`] enum (Live/Sealed).
//!
//! [`Block`] is the raw wire form. Its verified form is
//! `Verified<Block>` — produced by composite assembly, carrying the
//! claim that the block's header is verified (and thus its parent QC is
//! verified) and every internal commitment root the block declares has
//! been checked against the inline data.

use std::sync::Arc;

use hyperscale_hbor::Hbor;
use thiserror::Error;

use crate::{
    BeaconWitnessRoot, BlockHash, BlockHeader, BlockHeight, CertificateRoot, ChainOrigin,
    Finalization, LocalReceiptRoot, MAX_FINALIZED_TX_PER_BLOCK, MAX_PROVISIONS_PER_BLOCK,
    MAX_TXS_PER_BLOCK, ProvisionHash, ProvisionTxRootsMap, Provisions, ProvisionsRoot,
    QuorumCertificate, ShardId, SharedWitnessSources, SplitChildRoots, StateRoot, Transaction,
    TransactionRoot, TxHash, ValidatorId, Verifiable, Verified, WeightedTimestamp, WitnessSources,
};

/// Shared transaction list — wrapped in `Arc` so root-verification actions
/// can hold their own owner without deep-cloning the per-tx `Arc` array.
///
/// Elements are wrapped in [`Verifiable`] so a locally-built block (whose
/// txs were admission-validated through `MempoolCoordinator`) preserves
/// the [`Verified<Transaction>`] marker through block
/// construction; wire-decoded blocks land at [`Verifiable::Unverified`]
/// because HBOR decode is a transparent passthrough into that variant.
/// Same rationale as [`SharedCertificates`].
pub type SharedTransactions = Arc<Vec<Arc<Verifiable<Transaction>>>>;

/// Build a [`SharedTransactions`] from a list of raw `Arc<Transaction>`
/// recovered from persistent storage.
///
/// Each entry is lifted via
/// [`Verified::<Transaction>::from_persisted`] — the BFT-transitive
/// trust chain (persisted ⇒ committed block ⇒ voter-validated by ≥1 honest
/// voter) justifies marking each as [`Verifiable::Verified`]. Callers
/// outside that trust source must construct
/// `Vec<Arc<Verifiable<Transaction>>>` directly.
///
#[must_use]
pub fn shared_transactions_from_raw(txs: Vec<Arc<Transaction>>) -> SharedTransactions {
    let wrapped: Vec<Arc<Verifiable<Transaction>>> = txs
        .into_iter()
        .map(|tx| {
            Arc::new(Verifiable::from(Verified::<Transaction>::from_persisted(
                (*tx).clone(),
            )))
        })
        .collect();
    Arc::new(wrapped)
}

/// Shared certificate list — same rationale as [`SharedTransactions`].
///
/// Elements are wrapped in [`Verifiable`] so an in-process upstream's
/// [`Verified<Finalization>`] marker survives across block construction
/// and downstream dispatch; wire-decoded blocks land at
/// [`Verifiable::Unverified`] because HBOR decode is a transparent
/// passthrough into that variant. Same rationale as
/// [`BlockHeader::parent_qc`](crate::BlockHeader) which carries
/// `Verifiable<QuorumCertificate>` for the same reason.
pub type SharedCertificates = Arc<Vec<Arc<Verifiable<Finalization>>>>;

/// Shared provision list — same rationale as [`SharedCertificates`].
pub type SharedProvisions = Arc<Vec<Arc<Verifiable<Provisions>>>>;

/// Gas a shard consumed across the ticks `certificates` settle.
///
/// Free-standing so the proposer can price the certificates it selected
/// before the header those certificates go under exists, while
/// [`Block::gas_consumed`] answers the same question for a built block.
/// One derivation, so the two sides cannot drift.
#[must_use]
pub fn work_over_certificates(certificates: &[Arc<Verifiable<Finalization>>]) -> u64 {
    certificates.iter().fold(0u64, |sum, tick| {
        sum.saturating_add(tick.as_unverified().attested_work())
    })
}

/// Complete block with header and transaction data.
///
/// Transactions are stored in a single flat list, sorted by hash for deterministic ordering.
///
/// Blocks have two variants reflecting their temporal lifecycle:
/// - **`Live`**: within the cross-shard execution window. Carries the
///   provisions needed to execute cross-shard ticks locally.
/// - **`Sealed`**: past the execution window (at least `MAX_FINALIZATION_DELAY` of
///   wall-clock behind the local committed tip). Ticks are finalized from
///   certs + receipts alone, so provisions are no longer needed and are
///   dropped from memory. The on-disk / storage shape is always `Sealed`.
///
/// The header's `provision_root` commits to the original provision set, so
/// `Sealed` is self-consistent — a `Live` block matches its `Sealed` form
/// modulo the provision payload.
#[derive(Debug, Clone, Hbor)]
pub enum Block {
    /// Block within its cross-shard execution window — carries provisions.
    #[hbor(discriminant = 0)]
    Live {
        /// Block header (contains all merkle roots).
        header: BlockHeader,
        /// Transactions in this block, sorted by hash. Spelled out
        /// rather than as [`SharedTransactions`] so the cap can see the
        /// collection it bounds.
        #[hbor(max = MAX_TXS_PER_BLOCK)]
        transactions: Arc<Vec<Arc<Verifiable<Transaction>>>>,
        /// Finalizations finalized in this block.
        #[hbor(max = MAX_FINALIZED_TX_PER_BLOCK)]
        certificates: Arc<Vec<Arc<Verifiable<Finalization>>>>,
        /// Provisions needed to execute cross-shard ticks locally.
        #[hbor(max = MAX_PROVISIONS_PER_BLOCK)]
        provisions: Arc<Vec<Arc<Verifiable<Provisions>>>>,
        /// Proposer-supplied beacon-witness inputs. Committed via the
        /// header's `beacon_witness_root`; carried on the body so
        /// commit-time leaf derivation is identical on every node. See
        /// [`WitnessSources`].
        witness_sources: SharedWitnessSources,
    },
    /// Block past its execution window — provision bodies dropped, but
    /// the original `ProvisionHash` list is retained so sync-serving glue
    /// can still identify which bodies the block consumed and re-attach
    /// them from the in-memory cache when promoting back to `Live`.
    #[hbor(discriminant = 1)]
    Sealed {
        /// Block header (contains all merkle roots).
        header: BlockHeader,
        /// Transactions in this block, sorted by hash. Spelled out
        /// rather than as [`SharedTransactions`] so the cap can see the
        /// collection it bounds.
        #[hbor(max = MAX_TXS_PER_BLOCK)]
        transactions: Arc<Vec<Arc<Verifiable<Transaction>>>>,
        /// Finalizations finalized in this block.
        #[hbor(max = MAX_FINALIZED_TX_PER_BLOCK)]
        certificates: Arc<Vec<Arc<Verifiable<Finalization>>>>,
        /// Content hashes of the provisions the block consumed while
        /// `Live`. Empty iff the block consumed no provisions.
        #[hbor(max = MAX_PROVISIONS_PER_BLOCK)]
        provision_hashes: Arc<Vec<ProvisionHash>>,
        /// Proposer-supplied beacon-witness inputs — retained through
        /// sealing (unlike provisions) because the beacon-witness fold
        /// consuming them can run well after the block sealed. See
        /// [`WitnessSources`].
        witness_sources: SharedWitnessSources,
    },
}

/// One side of a merge: the terminal block a child's chain ends at, as
/// much of it as the merged genesis derives from.
///
/// Both the beacon fold (reading its recorded boundary) and a keeper
/// (reading the header it commit-proved) can supply this, so the
/// derivation does not care which of the two is calling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalRef {
    /// The terminal's committed state root — the child's subtree.
    pub state_root: StateRoot,
    /// The terminal's block hash.
    pub block_hash: BlockHash,
    /// The terminal's height.
    pub height: BlockHeight,
}

// Manual PartialEq - compare transaction/certificate content, not Arc pointers.
// Provisions are excluded from equality: the header's `provision_root` already
// commits to them, and a Live and Sealed form of the same block should compare
// equal for content purposes.
impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        fn tx_lists_equal(
            a: &[Arc<Verifiable<Transaction>>],
            b: &[Arc<Verifiable<Transaction>>],
        ) -> bool {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.hash() == y.hash())
        }
        fn cert_lists_equal(
            a: &[Arc<Verifiable<Finalization>>],
            b: &[Arc<Verifiable<Finalization>>],
        ) -> bool {
            // Compare by inner FW content; the `Verifiable` marker is
            // irrelevant to whether two blocks are content-equal.
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| ***x == ***y)
        }

        self.header() == other.header()
            && tx_lists_equal(self.transactions(), other.transactions())
            && cert_lists_equal(self.certificates(), other.certificates())
    }
}

impl Eq for Block {}

impl Block {
    /// Create an empty genesis block with the given proposer and JMT state.
    /// The [`ChainOrigin`] supplies the genesis height and start-time anchor
    /// (see [`QuorumCertificate::genesis`](crate::QuorumCertificate::genesis)).
    ///
    /// Genesis is born `Live` with no provisions — the temporality machinery
    /// activates only once there are cross-shard ticks in flight.
    #[must_use]
    pub fn genesis(
        shard_id: ShardId,
        proposer: ValidatorId,
        state_root: StateRoot,
        origin: ChainOrigin,
    ) -> Self {
        Self::Live {
            header: BlockHeader::genesis(shard_id, proposer, state_root, origin),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        }
    }

    /// The deterministic genesis block of a split child — empty, wrapping
    /// [`BlockHeader::split_child_genesis`]. The beacon fold seeds the
    /// child's anchor with this block's hash; the flip installs the same
    /// block.
    #[must_use]
    pub fn split_child_genesis(
        child: ShardId,
        state_root: StateRoot,
        parent_terminal: &BlockHeader,
        parent_canonical_wt: WeightedTimestamp,
    ) -> Self {
        Self::Live {
            header: BlockHeader::split_child_genesis(
                child,
                state_root,
                parent_terminal,
                parent_canonical_wt,
            ),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        }
    }

    /// The deterministic genesis block of a merged parent — empty,
    /// wrapping [`BlockHeader::merge_parent_genesis`]. The beacon fold
    /// composes the parent's anchor from its children's terminal roots
    /// and seeds it with this block's hash; the keeper flip installs the
    /// same block.
    #[must_use]
    pub fn merge_parent_genesis(
        parent: ShardId,
        state_root: StateRoot,
        left_terminal: (BlockHash, BlockHeight),
        right_terminal: (BlockHash, BlockHeight),
        cut_wt: WeightedTimestamp,
    ) -> Self {
        Self::Live {
            header: BlockHeader::merge_parent_genesis(
                parent,
                state_root,
                left_terminal,
                right_terminal,
                cut_wt,
            ),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        }
    }

    /// Derive a split child's genesis and chain origin from its parent's
    /// terminal header alone.
    ///
    /// The one derivation every party performs: the beacon fold seeding the
    /// child's anchor, a follower flipping at the cut, and a late joiner
    /// deriving against that anchor all call this, so none of them can
    /// disagree about the block a child starts from.
    ///
    /// Both inputs are frozen chain content. The subtree root is the
    /// terminal's [`split_child_roots`](BlockHeader::split_child_roots)
    /// entry for `child`, self-verifying by the composition check below —
    /// collision resistance means a parent cannot name a pair that composes
    /// to its own committed state root without holding those subtrees.
    /// `canonical_wt` is the weighted timestamp of the canonical
    /// certificate over the terminal, which is the `parent_qc` of the
    /// terminal's committed successor and never a QC served alongside it.
    ///
    /// `None` when the terminal carries no pair, or one that does not
    /// compose to its own state root.
    #[must_use]
    pub fn split_child_genesis_from_terminal(
        child: ShardId,
        terminal: &BlockHeader,
        canonical_wt: WeightedTimestamp,
    ) -> Option<(Self, ChainOrigin)> {
        let pair = terminal.split_child_roots()?;
        if !pair.composes_to(terminal.state_root()) {
            return None;
        }
        let child_root = if child.path() & 1 == 0 {
            pair.left
        } else {
            pair.right
        };
        Some((
            Self::split_child_genesis(child, child_root, terminal, canonical_wt),
            ChainOrigin {
                genesis_height: terminal.height().next(),
                anchor_wt: canonical_wt,
            },
        ))
    }

    /// Derive a merged parent's genesis and chain origin from its two
    /// children's terminals and the cut they terminate at.
    ///
    /// The merge counterpart of
    /// [`split_child_genesis_from_terminal`](Self::split_child_genesis_from_terminal),
    /// and likewise the one derivation both the beacon fold and every
    /// keeper perform. The merged root is the internal node over the two
    /// terminal subtree roots — the inverse of the composition a split
    /// verifies, each side attested by its own chain — and the first block
    /// continues both height lines at `max(h_left, h_right) + 1`.
    ///
    /// `cut_wt` is the instant the children terminate at: the end of their
    /// scheduled terminal window, which both sides read from the same
    /// schedule. A child's terminal certificate cannot serve, unlike a
    /// split's: the fold binds whichever QC ranked highest across the
    /// committed proposal set, an outcome of the beacon's own consensus
    /// that no keeper can reproduce.
    #[must_use]
    pub fn merge_parent_genesis_from_terminals(
        parent: ShardId,
        left: TerminalRef,
        right: TerminalRef,
        cut_wt: WeightedTimestamp,
    ) -> (Self, ChainOrigin) {
        let composed = SplitChildRoots {
            left: left.state_root,
            right: right.state_root,
        }
        .composed_root();
        let genesis = Self::merge_parent_genesis(
            parent,
            composed,
            (left.block_hash, left.height),
            (right.block_hash, right.height),
            cut_wt,
        );
        let origin = ChainOrigin {
            genesis_height: genesis.height(),
            anchor_wt: cut_wt,
        };
        (genesis, origin)
    }

    /// Block header — present in both variants.
    #[must_use]
    pub const fn header(&self) -> &BlockHeader {
        match self {
            Self::Live { header, .. } | Self::Sealed { header, .. } => header,
        }
    }

    /// Transactions in the block — present in both variants. Returns a borrow
    /// of the shared handle; callers that need to hand the list to an action
    /// crossing thread boundaries `.clone()` the `Arc` (refcount bump only,
    /// no Vec or per-tx clone).
    #[must_use]
    pub const fn transactions(&self) -> &SharedTransactions {
        match self {
            Self::Live { transactions, .. } | Self::Sealed { transactions, .. } => transactions,
        }
    }

    /// Finalizations (certificates) in the block — present in both variants.
    /// See [`Self::transactions`] for the sharing rationale.
    #[must_use]
    pub const fn certificates(&self) -> &SharedCertificates {
        match self {
            Self::Live { certificates, .. } | Self::Sealed { certificates, .. } => certificates,
        }
    }

    /// Gas this shard consumed across the ticks the block settles.
    ///
    /// The increment behind the header's running gas total, and the reason
    /// that total is checkable: a block's certificates carry their own
    /// receipts and survive sealing, so the block that claims the total
    /// also carries the evidence for its own contribution. Proposer and
    /// verifier both read this, so neither can drift from the other.
    ///
    /// Attribution follows settlement rather than execution — the only
    /// division derivable from one block's content. The running total is
    /// unaffected; only which epoch a given transaction's gas falls into
    /// can shift by the settlement lag.
    #[must_use]
    pub fn attested_work(&self) -> u64 {
        work_over_certificates(self.certificates())
    }

    /// Provisions. Non-empty only for `Live`; `Sealed` blocks have
    /// dropped their provisions because the cross-shard execution window
    /// they served has passed. Use `is_live()` when the variant itself
    /// matters — this accessor flattens both cases to a slice.
    ///
    /// Elements are [`Verifiable<Provisions>`] — the in-process upstream's
    /// verification marker survives across the block boundary; consumers
    /// peek `.verified()` to skip re-verification when the marker is live.
    #[must_use]
    pub fn provisions(&self) -> &[Arc<Verifiable<Provisions>>] {
        match self {
            Self::Live { provisions, .. } => provisions,
            Self::Sealed { .. } => &[],
        }
    }

    /// Content hashes of the block's provisions, regardless of variant.
    /// Computed inline from `provisions` on `Live`; read from the carried
    /// list on `Sealed`. The two paths agree on the same block by
    /// construction: `into_sealed` derives the `Sealed` list by hashing
    /// the `Live` provisions before dropping the bodies.
    #[must_use]
    pub fn provision_hashes(&self) -> Vec<ProvisionHash> {
        match self {
            Self::Live { provisions, .. } => provisions.iter().map(|p| p.hash()).collect(),
            Self::Sealed {
                provision_hashes, ..
            } => provision_hashes.iter().copied().collect(),
        }
    }

    /// Proposer-supplied beacon-witness inputs — present in both
    /// variants. The beacon-witness leaf derivation reads these at
    /// commit, so they must be byte-identical across nodes; carrying
    /// them on the block (rather than only the in-memory manifest)
    /// makes that hold for sync-committed and reloaded blocks.
    #[must_use]
    pub const fn witness_sources(&self) -> &SharedWitnessSources {
        match self {
            Self::Live {
                witness_sources, ..
            }
            | Self::Sealed {
                witness_sources, ..
            } => witness_sources,
        }
    }

    /// True if this block is still in its `Live` variant.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        matches!(self, Self::Live { .. })
    }

    /// Convert to `Sealed` by dropping provision bodies and retaining only
    /// their hashes. Identity on an already-sealed block. This is the
    /// canonical persisted shape; sync-serving glue re-attaches provision
    /// bodies (via `into_live`) when the requester needs them.
    #[must_use]
    pub fn into_sealed(self) -> Self {
        match self {
            Self::Live {
                header,
                transactions,
                certificates,
                provisions,
                witness_sources,
            } => {
                let hashes: Vec<ProvisionHash> = provisions.iter().map(|p| p.hash()).collect();
                Self::Sealed {
                    header,
                    transactions,
                    certificates,
                    provision_hashes: Arc::new(hashes),
                    witness_sources,
                }
            }
            sealed @ Self::Sealed { .. } => sealed,
        }
    }

    /// Attach provisions, promoting `Sealed` → `Live`. Used by sync-serving
    /// to upgrade a persisted block when the requester is still inside the
    /// cross-shard execution window.
    ///
    /// # Panics
    ///
    /// Panics if invoked on a `Live` block — that would silently discard
    /// the existing provision set.
    #[must_use]
    pub fn into_live(self, provisions: SharedProvisions) -> Self {
        match self {
            Self::Sealed {
                header,
                transactions,
                certificates,
                witness_sources,
                ..
            } => Self::Live {
                header,
                transactions,
                certificates,
                provisions,
                witness_sources,
            },
            Self::Live { .. } => {
                panic!("into_live called on an already-Live block")
            }
        }
    }

    /// Compute hash of this block (hashes the header).
    #[must_use]
    pub fn hash(&self) -> BlockHash {
        self.header().hash()
    }

    /// Get block height.
    #[must_use]
    pub const fn height(&self) -> BlockHeight {
        self.header().height()
    }

    /// Get total number of transactions.
    #[must_use]
    pub fn transaction_count(&self) -> usize {
        self.transactions().len()
    }

    /// Check if this block contains a specific transaction by hash.
    #[must_use]
    pub fn contains_transaction(&self, tx_hash: &TxHash) -> bool {
        self.transactions().iter().any(|tx| tx.hash() == *tx_hash)
    }

    /// Get all transaction hashes.
    #[must_use]
    pub fn transaction_hashes(&self) -> Vec<TxHash> {
        self.transactions().iter().map(|tx| tx.hash()).collect()
    }

    /// Check if this is the genesis block.
    #[must_use]
    pub fn is_genesis(&self) -> bool {
        self.header().is_genesis()
    }

    /// Get number of finalizations in this block.
    #[must_use]
    pub fn certificate_count(&self) -> usize {
        self.certificates().len()
    }
}

/// Failure modes of [`VerifiedBlock`] composite assembly.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum VerifiedBlockAssembleError {
    /// The supplied [`VerifiedBlockHeader`] does not belong to `block`
    /// (the header's hash differs from `block.hash()`).
    #[error(
        "verified header does not match block: header.hash={header_hash:?} block.hash={block_hash:?}"
    )]
    HeaderMismatch {
        /// Hash carried by the supplied verified header.
        header_hash: BlockHash,
        /// Hash computed from the supplied block.
        block_hash: BlockHash,
    },
}

impl Verified<Block> {
    /// Composite assembly. Pairs `block` with a `Verified<BlockHeader>`
    /// after confirming the header's content matches the block, and
    /// consumes a typed witness for each per-root verification.
    ///
    /// Construction asserts:
    /// 1. The header passes [`<BlockHeader as crate::Verify>`](crate::Verify)
    ///    (which transitively asserts `parent_qc` is verified).
    /// 2. The block's contents match its declared commitment roots
    ///    (`transaction_root`, `certificate_root`, `local_receipt_root`,
    ///    `provision_root`, `provision_tx_roots`, `beacon_witness_root`).
    ///    Each per-root check is witnessed by a typed `Verified<XRoot>`
    ///    value the constructor consumes.
    ///
    /// The witnesses are taken by value: each `Verified<XRoot>` can only
    /// have been produced by its `Verify` impl (the only constructor
    /// outside `new_unchecked`), so consuming them at assemble time
    /// makes the predicate structurally unforgeable — no caller can
    /// fabricate a "verified" marker without having run the check.
    ///
    /// The witnesses are **not** rebound to the block at this layer: a
    /// `Verified<TransactionRoot>` carries a typed claim that some root
    /// equals the merkle commitment over some content, but the verifier
    /// doesn't record which block the content came from. Pairing each
    /// witness with this block's content is the caller's responsibility
    /// — in practice the verification pipeline keys per-root slots by
    /// `block_hash` so the witness fed into `assemble` is always the one
    /// produced from this block's own dispatch. Direct callers outside
    /// that pipeline (tests, future helpers) must uphold the same
    /// pairing or the resulting `Verified<Block>`'s internal-commitment
    /// claim becomes unsound.
    ///
    /// State-root verification is intentionally not a witness here. Its
    /// verified value (`Verified<StateRoot, PreparedCommit>`) carries a
    /// byproduct that the action handler side-channels via
    /// `ActionContext::commit_prepared`, so the verified value can't
    /// ride in cleanly; the JMT-replay check still gates voting and
    /// commit via the parallel pipeline path.
    ///
    /// # Errors
    ///
    /// Returns [`VerifiedBlockAssembleError::HeaderMismatch`] when the
    /// verified header does not match the supplied block.
    #[allow(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        clippy::missing_const_for_fn
    )] // typed witnesses are consumed; the moves make the assembly contract explicit
    pub fn assemble(
        block: Block,
        header: Verified<BlockHeader>,
        _tx_root: Verified<TransactionRoot>,
        _certificate_root: Verified<CertificateRoot>,
        _local_receipt_root: Verified<LocalReceiptRoot>,
        _provision_root: Verified<ProvisionsRoot>,
        _provision_tx_roots: Verified<ProvisionTxRootsMap>,
        _beacon_witness_root: Verified<BeaconWitnessRoot>,
    ) -> Result<Self, VerifiedBlockAssembleError> {
        let header_hash = header.as_ref().hash();
        let block_hash = block.hash();
        if header_hash != block_hash {
            return Err(VerifiedBlockAssembleError::HeaderMismatch {
                header_hash,
                block_hash,
            });
        }
        Ok(Self::new_unchecked(block))
    }

    /// Borrow the verified parent QC. Total by the
    /// [`Verified<Block>`] predicate, which folds in
    /// [`Verified<BlockHeader>`]'s claim that `parent_qc` sits in
    /// [`Verifiable::Verified`].
    ///
    /// # Panics
    ///
    /// Panics if the embedded header's `parent_qc` is `Unverified` —
    /// only reachable through a misuse of
    /// [`Verified::new_unchecked`].
    #[must_use]
    pub fn parent_qc_verified(&self) -> &Verified<QuorumCertificate> {
        self.as_ref()
            .header()
            .parent_qc_verifiable()
            .verified()
            .expect("Verified<Block> predicate guarantees header.parent_qc is Verified")
    }
}
