//! The event stream the viewer renders.
//!
//! These are presentation types, not wire types: they are `serde`-encoded to
//! JavaScript values and never hashed, signed, or gossiped, so the ordered
//! collection discipline the wire types carry does not apply here.

use hyperscale_types::{BlockHeight, Round, ShardId, TxHash};
use serde::Serialize;

/// One observation, stamped with the BFT-attested time it happened at.
///
/// `wt` is the canonical weighted timestamp — the value carried in the
/// committing child's parent QC (INV-SHARD-6), never a served QC's stamp —
/// so it is monotone along a chain and identical on every replica. It is the
/// timeline's x-axis.
#[derive(Debug, Clone, Serialize)]
#[allow(missing_docs)] // `wt` is described above; `kind` is the payload
pub struct TraceEvent {
    pub wt: u64,
    pub kind: TraceKind,
}

/// A shard's trie path as a bit string, most significant bit first — `""`
/// for the root, `"1"` for its right child, `"10"` and `"11"` for that
/// child's own children.
///
/// The viewer derives the whole topology from these: one path is another's
/// parent exactly when it is a prefix of it, which is the same relation the
/// keyspace partition uses. `ShardId`'s own `Display` is a debug rendering
/// and carries no such structure.
#[derive(Debug, Clone, Serialize)]
pub struct ShardPath(pub String);

impl From<ShardId> for ShardPath {
    fn from(shard: ShardId) -> Self {
        let depth = shard.depth();
        let path = shard.path();
        let bits = (0..depth)
            .rev()
            .map(|bit| if path >> bit & 1 == 1 { '1' } else { '0' })
            .collect();
        Self(bits)
    }
}

/// What was observed. Serialized tagged, so the viewer switches on `type`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
#[allow(missing_docs)] // payload fields; their names are the documentation
pub enum TraceKind {
    #[serde(rename_all = "camelCase")]
    BlockCommitted {
        shard: ShardPath,
        height: u64,
        round: u64,
        /// A fallback block is an empty view-change recovery block: it
        /// carries no payload and reuses its parent's timestamp, which is
        /// why the timeline draws it differently from a normal block.
        fallback: bool,
        proposer: u64,
        /// Cross-shard execution waves this block opens. Single-shard
        /// transactions never appear here — the header field exists so
        /// remote shards know which certificates to expect — so this stays
        /// zero until the topology has more than one shard.
        cross_shard_waves: u32,
    },
    /// The beacon committed an epoch. One block per epoch, wall-clock paced,
    /// carrying no transactions: it decides validator set and topology, and
    /// every shard resolves its committee from the schedule this produces.
    #[serde(rename_all = "camelCase")]
    BeaconBlockCommitted { epoch: u64 },
    /// The keyspace partition changed — a split seated its children, or a
    /// merge composed its parent. Carries the whole new partition rather
    /// than a delta so a viewer that joined late still renders correctly.
    #[serde(rename_all = "camelCase")]
    TopologyChanged {
        shards: Vec<ShardPath>,
        /// Leaves that were not in the previous partition.
        appeared: Vec<ShardPath>,
        /// Leaves that were in it and are not now — a split parent, or the
        /// children a merge composed away.
        retired: Vec<ShardPath>,
    },
    #[serde(rename_all = "camelCase")]
    TxSubmitted { tx: TxLabel },
    #[serde(rename_all = "camelCase")]
    TxStatusChanged {
        tx: TxLabel,
        /// `pending`, `committed`, or a terminal `succeeded` / `aborted` /
        /// `rejected` — the outcome every participating shard agrees on
        /// (INV-EXEC-1).
        status: &'static str,
        /// Set once the transaction is ordered: the height that committed it.
        height: Option<u64>,
    },
}

/// A transaction hash, shortened to the prefix a reader can match by eye.
#[derive(Debug, Clone, Serialize)]
pub struct TxLabel(pub String);

impl From<TxHash> for TxLabel {
    fn from(hash: TxHash) -> Self {
        let rendered = format!("{hash}");
        let short: String = rendered
            .trim_start_matches("TxHash(")
            .chars()
            .take(8)
            .collect();
        Self(short)
    }
}

impl TraceEvent {
    pub(crate) fn block_committed(
        wt: u64,
        shard: ShardId,
        height: BlockHeight,
        round: Round,
        fallback: bool,
        proposer: u64,
        cross_shard_waves: u32,
    ) -> Self {
        Self {
            wt,
            kind: TraceKind::BlockCommitted {
                shard: shard.into(),
                height: height.inner(),
                round: round.inner(),
                fallback,
                proposer,
                cross_shard_waves,
            },
        }
    }

    pub(crate) const fn beacon_block(wt: u64, epoch: u64) -> Self {
        Self {
            wt,
            kind: TraceKind::BeaconBlockCommitted { epoch },
        }
    }

    pub(crate) fn topology_changed(
        wt: u64,
        shards: &[ShardId],
        appeared: Vec<ShardId>,
        retired: Vec<ShardId>,
    ) -> Self {
        let paths = |ids: Vec<ShardId>| ids.into_iter().map(ShardPath::from).collect();
        Self {
            wt,
            kind: TraceKind::TopologyChanged {
                shards: shards.iter().copied().map(ShardPath::from).collect(),
                appeared: paths(appeared),
                retired: paths(retired),
            },
        }
    }

    pub(crate) fn tx_submitted(wt: u64, tx: TxHash) -> Self {
        Self {
            wt,
            kind: TraceKind::TxSubmitted { tx: tx.into() },
        }
    }

    pub(crate) fn tx_status(
        wt: u64,
        tx: TxHash,
        status: &'static str,
        height: Option<u64>,
    ) -> Self {
        Self {
            wt,
            kind: TraceKind::TxStatusChanged {
                tx: tx.into(),
                status,
                height,
            },
        }
    }
}
