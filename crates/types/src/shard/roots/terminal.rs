//! The pair of commitments a chain leaves behind when it terminates.
//!
//! Both are computed over the same committed retention window and carried
//! by the same headers — any terminating boundary header, a split parent's
//! or a merge child's — so they are one field rather than two. A header
//! carrying one without the other would be malformed, and pairing them is
//! what makes that unrepresentable instead of checked.
//!
//! They answer different questions for different readers, which is why
//! they stay two roots rather than one. [`settled_txs`] is read by a
//! surviving *counterpart* resolving a straddling tick, covers the
//! cross-shard transactions the shard settled, and stays relevant a
//! retention horizon past the terminal. [`committed_txs`] is read by a
//! *successor* telling a replay from a first inclusion, covers every
//! transaction the shard committed, and stays relevant for a validity
//! range. One is bounded by cross-shard traffic, the other by total
//! throughput.
//!
//! [`settled_txs`]: TerminalRoots::settled_txs
//! [`committed_txs`]: TerminalRoots::committed_txs

use hyperscale_hbor::Hbor;

use crate::{CommittedTxsRoot, SettledTxsRoot};

/// The commitments a terminating boundary header carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Hbor)]
pub struct TerminalRoots {
    /// Merkle root over the cross-shard transactions this shard settled
    /// within its retention window — what lets a surviving counterpart
    /// resolve a straddling tick without walking the terminated chain.
    pub settled_txs: SettledTxsRoot,
    /// Merkle root over every transaction this shard committed within its
    /// retention window — what lets a successor tell a replay of something
    /// the predecessor committed from a first inclusion it never made.
    pub committed_txs: CommittedTxsRoot,
}
