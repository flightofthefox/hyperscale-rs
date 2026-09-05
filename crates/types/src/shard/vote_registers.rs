//! The HotStuff-2 safe-vote registers.

use std::sync::Arc;

use hyperscale_hbor::Hbor;

use crate::{Block, QuorumCertificate, Round};

/// Snapshot of a validator's monotone safe-vote registers, with the
/// certificate that authorizes the lock they record.
///
/// `locked_round` is the highest QC round the validator has voted to
/// extend; `last_voted_round` is the highest round it has voted or
/// timed out in. HotStuff-2's one-vote-per-round and lock-monotonicity
/// rules are guards over these values, so both only ever ratchet upward.
///
/// `high_qc` is what makes the lock survivable. A validator refuses to
/// vote for a block whose `parent_qc` sits below `locked_round`, so a
/// record that keeps the round without a certificate at least as high
/// describes a position the validator can never satisfy again: every
/// proposal it can build or receive extends a lower QC, and the QC that
/// would raise the lock can only form out of the votes it is refusing.
/// Carried in the same record so a durable lock always has a durable
/// justification, rather than leaving that to the order of two writes.
///
/// The certificate alone rescues a lone restarted replica, whose peers
/// still hold the block it names. A committee that restarts together has
/// no such peer, which is why [`VotePosition`] writes the block down
/// beside the record.
#[derive(Debug, Clone, PartialEq, Eq, Default, Hbor)]
pub struct SafeVoteRegisters {
    /// Highest QC round the validator has voted to extend.
    pub locked_round: Round,
    /// Highest round the validator has voted or timed out in.
    pub last_voted_round: Round,
    /// The highest QC the validator held when it wrote this record —
    /// at or above `locked_round`, since a lock rises to the round of a
    /// QC it has adopted. `None` only for a validator that has never
    /// voted, whose lock is at the origin and needs no justification.
    pub high_qc: Option<QuorumCertificate>,
}

impl SafeVoteRegisters {
    /// Field-wise maximum — the merge rule for register snapshots.
    /// Registers only ratchet, so the max of two snapshots is the most
    /// restrictive position either represents.
    ///
    /// The certificate is not a field of its own: it is the justification
    /// for a lock, so it travels with the higher `locked_round` rather
    /// than being maximized separately. Merging by QC round instead would
    /// let a snapshot's certificate outlive the lock it explains.
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        let high_qc = if other.locked_round > self.locked_round {
            other.high_qc
        } else {
            self.high_qc
        };
        Self {
            locked_round: self.locked_round.max(other.locked_round),
            last_voted_round: self.last_voted_round.max(other.last_voted_round),
            high_qc,
        }
    }
}

/// A signing position: the registers a vote or timeout ratchets, and the
/// uncommitted chain behind the certificate that justifies them.
///
/// `justification` runs from the block `registers.high_qc` certifies down
/// to the committed tip, oldest first. It is there because a certificate
/// is only usable while the block it names still exists: a proposer
/// extends the block its high QC certifies, and building on that block
/// means executing it over its own uncommitted ancestors. Only committed
/// blocks are durable otherwise, so a committee that restarts together
/// comes back holding a certificate for a block none of its members
/// retained — and every proposal it can build then sits beneath its own
/// lock. Empty when the certificate names the committed tip, which is
/// already durable.
#[derive(Debug, Clone)]
pub struct VotePosition {
    /// The registers as this signature ratcheted them.
    pub registers: SafeVoteRegisters,
    /// The uncommitted chain behind `registers.high_qc`, oldest first.
    pub justification: Vec<Arc<Block>>,
}
