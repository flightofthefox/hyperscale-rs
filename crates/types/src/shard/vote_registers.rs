//! The HotStuff-2 safe-vote registers.

use hyperscale_hbor::Hbor;

use crate::{QuorumCertificate, Round};

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
/// One replica recovers anyway, because peers that stayed up carry a
/// higher QC to it; a committee that restarts together has no such peer,
/// and wedges permanently. Carried in the same record so a durable lock
/// always has a durable justification, rather than leaving that to the
/// order of two writes.
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
