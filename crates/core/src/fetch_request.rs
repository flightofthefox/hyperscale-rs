//! Typed dispatch into the unified fetch protocols.
//!
//! Coordinators emit [`crate::Action::Fetch`] wrapping one of these variants
//! when they need the runner to issue a network fetch. `io_loop`'s dispatcher
//! has a single `Action::Fetch(req)` arm that matches on the inner enum and
//! calls into the corresponding `instances/*.rs` module.
//!
//! Peer selection is the network layer's job: the runner derives the
//! `ShardId` from the variant (local shard for intra-shard variants;
//! `source_shard` or `tick_id.shard_id()` for cross-shard variants)
//! and hands it to `Network::request` along with `preferred`. Coordinators
//! never reach into the topology for committee membership when emitting
//! fetches.
//!
//! Per-payload variants are keyless (just `ids + preferred`); admission
//! events drive cancellation rather than scope-keyed eviction. Cross-shard
//! variants retain `(source_shard, block_height)` because that scope IS
//! the fetch key (no id-set to enumerate).

use hyperscale_types::{
    BlockHash, BlockHeight, Epoch, LeafIndex, MessageClass, ProvisionHash, ShardId, TickId, TxHash,
    ValidatorId,
};

/// Fetch family — one variant per payload type.
///
/// Common field semantics across variants:
///
/// - `shard` — committee whose members will serve the request. Local for
///   intra-shard fetches; source shard for cross-shard DA fetches.
/// - `preferred` — canonical-source hint passed straight to
///   `Network::request`. `Some(proposer)` on shard-path fetches where one
///   peer is authoritative; `None` on catch-up / DA paths that fan out.
/// - `class` — overrides the wire type's static `NetworkMessage::class()`.
///   `None` keeps the default; `Some` demotes to a less-urgent class
///   (typically `Recovery`) for catch-up / best-effort traffic.
#[derive(Debug, Clone)]
pub enum FetchRequest {
    /// Transaction bodies by `TxHash` — shard-path fetches against the
    /// local shard's committee.
    Transactions {
        /// Transaction hashes to fetch.
        ids: Vec<TxHash>,
        /// Committee shard serving the request.
        shard: ShardId,
        /// Canonical-source hint, when one exists.
        preferred: Option<ValidatorId>,
        /// Optional class override; see enum-level doc.
        class: Option<MessageClass>,
    },
    /// Intra-shard DA payload — `shard` is always local.
    LocalProvisions {
        /// Provision hashes to fetch.
        ids: Vec<ProvisionHash>,
        /// Always the local shard for this variant.
        shard: ShardId,
        /// Canonical-source hint, when one exists.
        preferred: Option<ValidatorId>,
        /// Optional class override; see enum-level doc.
        class: Option<MessageClass>,
    },
    /// Intra-shard DA payload — `shard` is always local.
    Finalizations {
        /// Tick ids whose finalizations are missing.
        ids: Vec<TickId>,
        /// Always the local shard for this variant.
        shard: ShardId,
        /// Canonical-source hint, when one exists.
        preferred: Option<ValidatorId>,
        /// Optional class override; see enum-level doc.
        class: Option<MessageClass>,
    },
    /// Cross-shard provisions fetch keyed by `(source_shard, block_height)`.
    /// Routing shard is `source_shard`; `preferred` is the source-block
    /// proposer that originated the provisions.
    RemoteProvisions {
        /// Source shard whose provisions are missing.
        source_shard: ShardId,
        /// Source-shard block height the missing provisions are anchored to.
        block_height: BlockHeight,
        /// Canonical-source hint, when one exists.
        preferred: Option<ValidatorId>,
        /// Optional class override; see enum-level doc.
        class: Option<MessageClass>,
    },
    /// Cross-shard execution-cert fetch by transaction. Routing shard is
    /// `source_shard`, which has to be carried: the requester knows only
    /// that the transaction is the source shard's business too, not which
    /// of its certificates ends up covering it. `preferred` is `None` —
    /// the designated broadcaster role is computable but health-weighted
    /// selection works equally well empirically.
    ExecutionCerts {
        /// Shard whose outcome for the transaction is missing.
        source_shard: ShardId,
        /// Transaction whose outcome is missing.
        tx_hash: TxHash,
        /// Always `None` for this variant; see variant-level doc.
        preferred: Option<ValidatorId>,
        /// Optional class override; see enum-level doc.
        class: Option<MessageClass>,
    },
    /// Cross-shard beacon-witness fetch keyed by the source shard's
    /// `(block_height, committed_block_hash, leaf range)`. Routing
    /// shard is `source_shard` (the shard whose committee anchors the
    /// witness accumulator). The whole run is one request: its range
    /// proof is scoped to the run, so a partial answer proves nothing.
    ShardWitnesses {
        /// Source shard whose witnesses we want.
        source_shard: ShardId,
        /// Height of the anchor block in the source-shard chain.
        block_height: BlockHeight,
        /// Hash of the anchor block; binds responses to the right
        /// `beacon_witness_root`.
        committed_block_hash: BlockHash,
        /// First leaf of the run in the anchor block's accumulator.
        lo: LeafIndex,
        /// End of the run, exclusive.
        hi: LeafIndex,
        /// Canonical-source hint, when one exists.
        preferred: Option<ValidatorId>,
        /// Optional class override; see enum-level doc.
        class: Option<MessageClass>,
    },
    /// Missing-proposal fetch for a beacon committee member at an
    /// in-flight epoch. The responding committee is the beacon
    /// committee at `epoch`; the runner routes via the requesting
    /// vnode's local shard (the network layer's peer selection
    /// resolves through the topology snapshot).
    BeaconProposal {
        /// Local shard the requesting vnode belongs to; threaded so
        /// the network layer has a valid committee handle for peer
        /// selection. `preferred` pins the actual destination.
        shard: ShardId,
        /// Epoch the proposal targets.
        epoch: Epoch,
        /// Validator whose proposal we're fetching.
        validator: ValidatorId,
        /// Beacon-committee peer to try first.
        preferred: Option<ValidatorId>,
        /// Optional class override; see enum-level doc.
        class: Option<MessageClass>,
    },
}
