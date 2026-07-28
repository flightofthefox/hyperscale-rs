//! The per-block randomness-reveal chain and its anchor-epoch reset.
//!
//! Each block folds its proposer's reveal output into a running hash chain
//! that restarts whenever the block's anchor epoch differs from its
//! parent's. The reset makes the chain a per-epoch quantity: a block whose
//! child anchors past the epoch cut is the last of its epoch, so the chain
//! it carries is that epoch's closed value — the one the beacon folds into
//! `state.randomness`.
//!
//! A block's anchor epoch is `epoch_for(parent_qc.weighted_timestamp)`,
//! known at proposal time. It is deliberately not the epoch the block
//! *crosses into*: a crossing is judged against the QC certifying the block
//! ([`EpochWindows::is_crossing`](crate::EpochWindows::is_crossing)), which
//! does not yet exist when the block is built.
//!
//! Both derivations are pure functions of their arguments, and every
//! argument is either carried on the parent header or derived from
//! `epoch_duration_ms` — fixed at genesis and outside the governable
//! parameter set. Proposer and verifier therefore agree byte-for-byte, and
//! a verifier revisiting a header at any later time reaches the same
//! verdict.

use crate::{Epoch, Hash, RevealChain, VrfOutput};

/// Domain tag for reveal-chain links.
pub const REVEAL_CHAIN_DOMAIN_TAG: &[u8] = b"hyperscale-reveal-chain-v1";

/// The reveal chain a block commits, given its parent's chain and the two
/// anchor epochs.
///
/// Seeds a fresh chain when the epochs differ, extends the parent's when
/// they match. The one shared derivation: the proposer stamps what this
/// returns and the verifier recomputes it, so the two cannot drift.
#[must_use]
pub fn next_reveal_chain(
    parent_chain: RevealChain,
    parent_anchor_epoch: Epoch,
    own_anchor_epoch: Epoch,
    own_output: VrfOutput,
) -> RevealChain {
    let prior = if own_anchor_epoch == parent_anchor_epoch {
        parent_chain
    } else {
        RevealChain::ZERO
    };
    extend_reveal_chain(prior, own_output)
}

/// One chain link: `BLAKE3(REVEAL_CHAIN_DOMAIN_TAG ‖ prior ‖ output)`.
///
/// A fresh chain passes [`RevealChain::ZERO`] as `prior`, so the seeded and
/// extended cases share a preimage shape and a reset is not confusable with
/// a link whose predecessor happened to hash to zero.
#[must_use]
pub fn extend_reveal_chain(prior: RevealChain, output: VrfOutput) -> RevealChain {
    RevealChain::from_raw(Hash::from_parts(&[
        REVEAL_CHAIN_DOMAIN_TAG,
        prior.as_raw().as_bytes(),
        output.as_bytes(),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(seed: u8) -> VrfOutput {
        VrfOutput::new([seed; 32])
    }

    fn epoch(n: u64) -> Epoch {
        Epoch::new(n)
    }

    #[test]
    fn a_link_binds_both_its_predecessor_and_its_output() {
        let a = extend_reveal_chain(RevealChain::ZERO, output(1));
        let b = extend_reveal_chain(a, output(2));
        assert_ne!(a, b, "distinct outputs give distinct links");
        assert_ne!(
            b,
            extend_reveal_chain(RevealChain::ZERO, output(2)),
            "the same output off a different predecessor gives a different link"
        );
    }

    #[test]
    fn same_anchor_epoch_extends_the_parent_chain() {
        let parent = extend_reveal_chain(RevealChain::ZERO, output(1));
        assert_eq!(
            next_reveal_chain(parent, epoch(4), epoch(4), output(2)),
            extend_reveal_chain(parent, output(2)),
        );
    }

    #[test]
    fn a_new_anchor_epoch_seeds_a_fresh_chain() {
        let parent = extend_reveal_chain(RevealChain::ZERO, output(1));
        let reset = next_reveal_chain(parent, epoch(4), epoch(5), output(2));
        assert_eq!(reset, extend_reveal_chain(RevealChain::ZERO, output(2)));
        assert_ne!(reset, extend_reveal_chain(parent, output(2)));
    }

    /// The epoch a chain closes is independent of how far the anchor
    /// jumped: a shard that produced nothing for several epochs reseeds
    /// once, not once per skipped epoch.
    #[test]
    fn a_multi_epoch_gap_seeds_once() {
        let parent = extend_reveal_chain(RevealChain::ZERO, output(1));
        assert_eq!(
            next_reveal_chain(parent, epoch(4), epoch(9), output(2)),
            next_reveal_chain(parent, epoch(4), epoch(5), output(2)),
        );
    }

    /// Two epochs' chains over identical output runs stay distinct only
    /// through their outputs — the reset carries no epoch number, so a
    /// replayed run reproduces the value. The beacon's exactly-once gate
    /// keys on the crossed epoch, not on chain uniqueness.
    #[test]
    fn a_reset_chain_is_a_function_of_its_run_alone() {
        let a = next_reveal_chain(RevealChain::ZERO, epoch(1), epoch(2), output(7));
        let b = next_reveal_chain(RevealChain::ZERO, epoch(5), epoch(6), output(7));
        assert_eq!(a, b);
    }
}
