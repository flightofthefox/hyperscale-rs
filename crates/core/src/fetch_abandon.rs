//! Typed cancellation into the unified fetch protocols.
//!
//! Coordinators emit [`crate::Action::AbandonFetch`] wrapping one of these
//! variants when a previously-requested id is no longer wanted — the
//! consumer's expected-set has dropped it (verified, aged out past a
//! retention horizon, orphan-cleanup) and the in-flight fetch should stop.
//!
//! Symmetric to [`crate::FetchRequest`]: one variant per binding, same
//! keying, but with no peer pool — cancellation has no destination.
//! `io_loop`'s dispatcher matches the inner enum and feeds the ids through
//! `FetchInput::Abandoned` on the corresponding binding.

use hyperscale_types::{
    BlockHash, BlockHeight, Epoch, FinalizationHash, LeafIndex, PredecessorTerminal, ProvisionHash,
    ShardId, TxHash, ValidatorId,
};

/// Fetch-cancel family — one variant per payload type. Variants are added
/// when each binding migrates to push-cancel.
#[derive(Debug, Clone)]
pub enum FetchAbandon {
    /// Per-tx fetch keyed by [`TxHash`]. Emitted by the mempool when an
    /// expected cross-shard tx is dropped from `ExpectedTxs` without ever
    /// being admitted — block-include race (tx landed via committed block
    /// while the fetch was still in flight) or retention-horizon orphan
    /// cleanup (cross-shard DA failed entirely).
    Transactions {
        /// Tx hashes whose in-flight fetch should be cancelled.
        ids: Vec<TxHash>,
    },
    /// Cross-shard provisions fetch keyed by `(source_shard, block_height)`.
    /// Emitted when `ProvisionCoordinator`'s expected-set drops the key —
    /// verification succeeded, the entry orphaned past retention, or the
    /// source block aged past its deadline.
    RemoteProvisions {
        /// Source shard whose provisions fetch is being cancelled.
        source_shard: ShardId,
        /// Source-shard block height for the cancelled fetch.
        block_height: BlockHeight,
    },
    /// Intra-shard local-provision fetch keyed by [`ProvisionHash`]. Emitted
    /// when the provisions pipeline terminally drops a buffered batch
    /// (deadline reached, post-commit tombstone hit) so the in-flight
    /// local-DA fetch — which would otherwise stay pinned on a payload
    /// that can no longer be admitted — releases its slot.
    LocalProvisions {
        /// Provision hashes whose in-flight fetch should be cancelled.
        hashes: Vec<ProvisionHash>,
    },
    /// Per-block finalization fetch keyed by [`TickId`]. Emitted by the
    /// execution coordinator when a fetched tick fails terminal admission
    /// checks (no quorum power on a contained EC, committee keys not
    /// resolvable, signature invalid) so the FSM clears the in-flight
    /// slot it would otherwise pin on a tick that cannot be admitted.
    Finalizations {
        /// Finalization identities whose in-flight fetch should be
        /// cancelled.
        ids: Vec<FinalizationHash>,
    },
    /// Cross-shard execution-certificate fetch keyed by
    /// `(source_shard, tx_hash)`. Emitted when an EC's admission path
    /// silently drops the cert (unresolvable committee keys, invalid
    /// signature, sub-quorum signers). A dropped certificate releases every
    /// transaction it covered, since none of them got an outcome from it.
    /// Multiple aggregations can arrive for the same transactions; if a
    /// later valid one admits successfully, the abandon is a no-op on the
    /// FSM, while the failure-only case correctly releases the slot for
    /// cleanup-timer to re-fetch.
    ExecutionCerts {
        /// Transactions whose in-flight EC fetch should be cancelled.
        ids: Vec<(ShardId, TxHash)>,
    },
    /// Committed-transaction membership query keyed by
    /// `(predecessor, tx_hash)`. Emitted by the acquisition scan when a
    /// pair it previously asked about drops out of the outstanding set
    /// without an answer — the transaction expired out of the mempool,
    /// or the chain outlived its origin and the rule that wanted the
    /// answer no longer applies. Nothing else retires these ids: a
    /// terminated committee that never answers would otherwise pin them
    /// for the process's life.
    CommittedTxs {
        /// `(predecessor, transaction)` pairs whose query should stop.
        ids: Vec<(PredecessorTerminal, TxHash)>,
    },
    /// Missing-proposal fetch keyed by `(epoch, validator)`. Emitted by
    /// the beacon coordinator when a pending commit-assembly stash is
    /// evicted before its awaited fetches resolve — typically because a
    /// peer's beacon-block gossip committed the same epoch first and
    /// `adopt_block` advanced `current_epoch` past the stash, leaving
    /// the in-flight `(epoch, validator)` slots pinned on data we no
    /// longer need.
    BeaconProposal {
        /// `(epoch, validator)` pairs whose in-flight fetch should be cancelled.
        ids: Vec<(Epoch, ValidatorId)>,
    },
    /// Cross-shard witness fetch keyed by
    /// `(source_shard, block_height, committed_block_hash, lo, hi)`.
    /// Emitted by the beacon coordinator when a shard's applied watermark
    /// (`boundaries[shard].witness_leaf_count`) advances past an in-flight
    /// run — those leaves have been folded on-chain and a future
    /// contribution can't include them, so the FSM's in-flight slot should
    /// release rather than pin on a payload the tracker would only evict.
    ShardWitnesses {
        /// Anchor + range ids whose in-flight fetch should be cancelled.
        ids: Vec<(ShardId, BlockHeight, BlockHash, LeafIndex, LeafIndex)>,
    },
}
