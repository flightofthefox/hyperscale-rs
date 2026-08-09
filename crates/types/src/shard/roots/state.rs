//! State-root verification typestate.
//!
//! [`StateRoot`] is verified by replaying a block's finalizations
//! against the JMT rooted at the parent's state root and comparing the
//! resulting root against the header's claim. The JMT replay itself
//! happens inside the storage backend's `prepare_block_commit`; the
//! verifier here is a thin equality check.
//!
//! The replay's other byproduct — the [`PreparedCommit`] closure — is
//! orthogonal `IoLoop` pipeline data, not part of the verification
//! predicate. The action handler routes it through `commit_prepared`
//! separately from the verified handle. Predicate at
//! [`impl Verify<StateRootContext>`](Verify::verify) below.
//!
//! [`StateRoot`]: crate::StateRoot
//! [`PreparedCommit`]: crate::PreparedCommit

use hyperscale_hbor::Hbor;
use hyperscale_jmt::{Blake3Hasher, Hasher};
use thiserror::Error;

use crate::{CommittedTxsRoot, Hash, SettledTxsRoot, StateRoot, Verified, Verify};

/// The two child hashes of the JMT root node behind a header's
/// `state_root` — `r_p0` / `r_p1` for a shard whose split executes at the
/// next epoch boundary.
///
/// Carried on every header of the split-pending shard's final epoch, so
/// whichever block terminates the chain delivers the children of exactly
/// the root the beacon anchors. `StateRoot::ZERO` marks an absent side
/// (the JMT hashes absent children as the empty hash).
///
/// Verified beside the state root: `hash_internal(left, right)` must
/// equal the recomputed root, which pins the pair by collision
/// resistance. A ≤1-key tree has a leaf root, and leaf/internal hashing
/// is domain-separated, so no pair verifies against it — the check fails
/// closed on the degenerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hbor)]
pub struct SplitChildRoots {
    /// Subtree root at the left child's prefix (`path‖0`).
    pub left: StateRoot,
    /// Subtree root at the right child's prefix (`path‖1`).
    pub right: StateRoot,
}

impl SplitChildRoots {
    /// The internal-node hash the pair composes to —
    /// `hash_internal(left, right)`.
    #[must_use]
    pub fn composed_root(&self) -> StateRoot {
        StateRoot::from_raw(Hash::from_hash_bytes(&Blake3Hasher::hash_internal(&[
            *self.left.as_bytes(),
            *self.right.as_bytes(),
        ])))
    }

    /// Whether `hash_internal(left, right)` reproduces `root` — the pair
    /// is exactly the two children of the internal node behind `root`.
    #[must_use]
    pub fn composes_to(&self, root: StateRoot) -> bool {
        self.composed_root() == root
    }
}

/// Inputs the [`StateRoot`] verifier checks against.
///
/// [`StateRoot`]: crate::StateRoot
pub struct StateRootContext<'a> {
    /// Root produced by replaying the block's finalizations against
    /// the JMT.
    pub computed_root: &'a StateRoot,
    /// The header's `split_child_roots` claim.
    pub claimed_split_child_roots: Option<SplitChildRoots>,
    /// Whether the block's window requires the claim — true exactly when
    /// the next epoch's trie replaces the shard with its two children
    /// (the split-pending shard's final epoch).
    pub split_child_roots_required: bool,
    /// The header's `settled_txs_root` claim.
    pub claimed_settled_txs_root: Option<SettledTxsRoot>,
    /// Root recomputed by walking the committed retention window, present
    /// exactly when [`Self::terminal_roots_required`] is set.
    pub computed_settled_txs_root: Option<SettledTxsRoot>,
    /// The header's `committed_txs_root` claim.
    pub claimed_committed_txs_root: Option<CommittedTxsRoot>,
    /// Root recomputed by walking the committed retention window, present
    /// exactly when [`Self::terminal_roots_required`] is set.
    pub computed_committed_txs_root: Option<CommittedTxsRoot>,
    /// Whether the block's window requires both terminal-boundary claims —
    /// set on a terminating shard's boundary header. One predicate for
    /// both roots: they are carried by the same headers, so a header
    /// carrying one and not the other is malformed either way.
    pub terminal_roots_required: bool,
}

/// Failure modes of [`StateRoot`] verification.
///
/// [`StateRoot`]: crate::StateRoot
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum StateRootVerifyError {
    /// JMT replay computed a different root than the header claimed.
    /// Distinguishes a Byzantine proposer from an honest one; the
    /// receipt-root pre-flight check (run before this verifier on the
    /// shared dispatch path) already eliminates the
    /// receipts-don't-match case.
    #[error("computed state root {computed:?} ≠ claimed {expected:?}")]
    Mismatch {
        /// Header's claimed state root.
        expected: StateRoot,
        /// Root produced by replaying receipts against the JMT.
        computed: StateRoot,
    },

    /// The block's window is the split-pending shard's final epoch, but
    /// the header carries no `split_child_roots`.
    #[error("split child roots required in the final epoch but absent")]
    MissingSplitChildRoots,

    /// The header carries `split_child_roots` outside a split-pending
    /// shard's final epoch.
    #[error("split child roots carried outside a split-pending final epoch")]
    UnexpectedSplitChildRoots,

    /// The claimed pair does not compose to the computed root —
    /// `hash_internal(left, right) ≠ computed_root`. Also the fail-closed
    /// path for a ≤1-key tree, whose root is a leaf no pair composes to.
    #[error("split child roots {left:?}/{right:?} do not compose to {computed:?}")]
    SplitChildRootsMismatch {
        /// Claimed left child subtree root.
        left: StateRoot,
        /// Claimed right child subtree root.
        right: StateRoot,
        /// Root produced by replaying receipts against the JMT.
        computed: StateRoot,
    },

    /// The block terminates the shard at a boundary but the header carries
    /// no `settled_txs_root`.
    #[error("settled transaction root required at a terminating boundary but absent")]
    MissingSettledTxsRoot,

    /// The header carries `settled_txs_root` outside a terminating
    /// boundary header.
    #[error("settled transaction root carried outside a terminating boundary")]
    UnexpectedSettledTxsRoot,

    /// The claimed settled-transaction root differs from the root recomputed
    /// over the committed retention window.
    #[error("settled transaction root {claimed:?} ≠ recomputed {computed:?}")]
    SettledTxsRootMismatch {
        /// Header's claimed settled-transaction root.
        claimed: SettledTxsRoot,
        /// Root recomputed by walking the committed retention window.
        computed: Option<SettledTxsRoot>,
    },

    /// The block terminates the shard at a boundary but the header carries
    /// no `committed_txs_root`.
    #[error("committed transaction root required at a terminating boundary but absent")]
    MissingCommittedTxsRoot,

    /// The header carries `committed_txs_root` outside a terminating
    /// boundary header.
    #[error("committed transaction root carried outside a terminating boundary")]
    UnexpectedCommittedTxsRoot,

    /// The claimed committed-transaction root differs from the root
    /// recomputed over the committed retention window.
    #[error("committed transaction root {claimed:?} ≠ recomputed {computed:?}")]
    CommittedTxsRootMismatch {
        /// Header's claimed committed-transaction root.
        claimed: CommittedTxsRoot,
        /// Root recomputed by walking the committed retention window.
        computed: Option<CommittedTxsRoot>,
    },
}

impl Verified<StateRoot> {
    /// Pipeline-attestation gate for slot prefill. The trust source is
    /// the verification pipeline's per-root tracking: an earlier verifier
    /// run already accepted `root` (success path of
    /// [`<StateRoot as Verify>::verify`](Verify::verify)).
    #[must_use]
    pub const fn from_pipeline_attestation(root: StateRoot) -> Self {
        Self::new_unchecked(root)
    }
}

/// Construction asserts: the supplied `computed_root` (produced by
/// replaying the block's finalizations against the JMT rooted at the
/// parent's state root) equals the wrapped [`StateRoot`], and the
/// header's `split_child_roots` claim is present exactly when the window
/// requires it and composes to the computed root.
impl Verify<&StateRootContext<'_>> for StateRoot {
    type Error = StateRootVerifyError;

    fn verify(&self, ctx: &StateRootContext<'_>) -> Result<Verified<Self>, Self::Error> {
        if *ctx.computed_root != *self {
            return Err(StateRootVerifyError::Mismatch {
                expected: *self,
                computed: *ctx.computed_root,
            });
        }
        match (
            ctx.split_child_roots_required,
            ctx.claimed_split_child_roots,
        ) {
            (true, None) => return Err(StateRootVerifyError::MissingSplitChildRoots),
            (false, Some(_)) => return Err(StateRootVerifyError::UnexpectedSplitChildRoots),
            (true, Some(claimed)) if !claimed.composes_to(*ctx.computed_root) => {
                return Err(StateRootVerifyError::SplitChildRootsMismatch {
                    left: claimed.left,
                    right: claimed.right,
                    computed: *ctx.computed_root,
                });
            }
            _ => {}
        }
        match (ctx.terminal_roots_required, ctx.claimed_settled_txs_root) {
            (true, None) => return Err(StateRootVerifyError::MissingSettledTxsRoot),
            (false, Some(_)) => return Err(StateRootVerifyError::UnexpectedSettledTxsRoot),
            (true, Some(claimed)) if Some(claimed) != ctx.computed_settled_txs_root => {
                return Err(StateRootVerifyError::SettledTxsRootMismatch {
                    claimed,
                    computed: ctx.computed_settled_txs_root,
                });
            }
            _ => {}
        }
        match (ctx.terminal_roots_required, ctx.claimed_committed_txs_root) {
            (true, None) => return Err(StateRootVerifyError::MissingCommittedTxsRoot),
            (false, Some(_)) => return Err(StateRootVerifyError::UnexpectedCommittedTxsRoot),
            (true, Some(claimed)) if Some(claimed) != ctx.computed_committed_txs_root => {
                return Err(StateRootVerifyError::CommittedTxsRootMismatch {
                    claimed,
                    computed: ctx.computed_committed_txs_root,
                });
            }
            _ => {}
        }
        Ok(Verified::new_unchecked(*self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Hash;

    fn composing_pair() -> (SplitChildRoots, StateRoot) {
        let left = StateRoot::from_raw(Hash::from_bytes(b"left subtree"));
        let right = StateRoot::from_raw(Hash::from_bytes(b"right subtree"));
        let root = StateRoot::from_raw(Hash::from_hash_bytes(&Blake3Hasher::hash_internal(&[
            *left.as_bytes(),
            *right.as_bytes(),
        ])));
        (SplitChildRoots { left, right }, root)
    }

    #[test]
    fn final_epoch_pair_composing_to_the_root_verifies() {
        let (pair, root) = composing_pair();
        assert!(
            root.verify(&StateRootContext {
                computed_root: &root,
                claimed_split_child_roots: Some(pair),
                split_child_roots_required: true,
                claimed_settled_txs_root: None,
                computed_settled_txs_root: None,
                claimed_committed_txs_root: None,
                computed_committed_txs_root: None,
                terminal_roots_required: false,
            })
            .is_ok()
        );
    }

    #[test]
    fn missing_pair_in_the_final_epoch_is_rejected() {
        let (_, root) = composing_pair();
        assert_eq!(
            root.verify(&StateRootContext {
                computed_root: &root,
                claimed_split_child_roots: None,
                split_child_roots_required: true,
                claimed_settled_txs_root: None,
                computed_settled_txs_root: None,
                claimed_committed_txs_root: None,
                computed_committed_txs_root: None,
                terminal_roots_required: false,
            })
            .unwrap_err(),
            StateRootVerifyError::MissingSplitChildRoots,
        );
    }

    #[test]
    fn pair_outside_the_final_epoch_is_rejected() {
        let (pair, root) = composing_pair();
        assert_eq!(
            root.verify(&StateRootContext {
                computed_root: &root,
                claimed_split_child_roots: Some(pair),
                split_child_roots_required: false,
                claimed_settled_txs_root: None,
                computed_settled_txs_root: None,
                claimed_committed_txs_root: None,
                computed_committed_txs_root: None,
                terminal_roots_required: false,
            })
            .unwrap_err(),
            StateRootVerifyError::UnexpectedSplitChildRoots,
        );
    }

    #[test]
    fn non_composing_pair_is_rejected() {
        let (pair, root) = composing_pair();
        let forged = SplitChildRoots {
            left: StateRoot::from_raw(Hash::from_bytes(b"forged")),
            right: pair.right,
        };
        assert_eq!(
            root.verify(&StateRootContext {
                computed_root: &root,
                claimed_split_child_roots: Some(forged),
                split_child_roots_required: true,
                claimed_settled_txs_root: None,
                computed_settled_txs_root: None,
                claimed_committed_txs_root: None,
                computed_committed_txs_root: None,
                terminal_roots_required: false,
            })
            .unwrap_err(),
            StateRootVerifyError::SplitChildRootsMismatch {
                left: forged.left,
                right: forged.right,
                computed: root,
            },
        );
    }

    #[test]
    fn root_mismatch_is_reported_before_the_pair_check() {
        let (pair, root) = composing_pair();
        let other = StateRoot::from_raw(Hash::from_bytes(b"other"));
        assert!(matches!(
            other
                .verify(&StateRootContext {
                    computed_root: &root,
                    claimed_split_child_roots: Some(pair),
                    split_child_roots_required: true,
                    claimed_settled_txs_root: None,
                    computed_settled_txs_root: None,
                    claimed_committed_txs_root: None,
                    computed_committed_txs_root: None,
                    terminal_roots_required: false,
                })
                .unwrap_err(),
            StateRootVerifyError::Mismatch { .. },
        ));
    }

    /// A satisfied committed-transaction claim, so a settled-side test
    /// fails only on the check it is about.
    fn satisfied_committed(required: bool) -> Option<CommittedTxsRoot> {
        required.then(|| CommittedTxsRoot::from_raw(Hash::from_bytes(b"committed")))
    }

    /// A satisfied settled-transaction claim, the mirror of
    /// [`satisfied_committed`].
    fn satisfied_settled(required: bool) -> Option<SettledTxsRoot> {
        required.then(|| SettledTxsRoot::from_raw(Hash::from_bytes(b"settled")))
    }

    /// A context isolating the settled-transaction checks: the state root
    /// matches, no split-child-roots claim is in play, and the committed
    /// root is supplied consistently.
    fn settled_ctx(
        root: &StateRoot,
        claimed: Option<SettledTxsRoot>,
        computed: Option<SettledTxsRoot>,
        required: bool,
    ) -> StateRootContext<'_> {
        let committed = satisfied_committed(required);
        StateRootContext {
            computed_root: root,
            claimed_split_child_roots: None,
            split_child_roots_required: false,
            claimed_settled_txs_root: claimed,
            computed_settled_txs_root: computed,
            claimed_committed_txs_root: committed,
            computed_committed_txs_root: committed,
            terminal_roots_required: required,
        }
    }

    /// The mirror of [`settled_ctx`], isolating the committed-transaction
    /// checks with the settled root supplied consistently.
    fn committed_ctx(
        root: &StateRoot,
        claimed: Option<CommittedTxsRoot>,
        computed: Option<CommittedTxsRoot>,
        required: bool,
    ) -> StateRootContext<'_> {
        let settled = satisfied_settled(required);
        StateRootContext {
            computed_root: root,
            claimed_split_child_roots: None,
            split_child_roots_required: false,
            claimed_settled_txs_root: settled,
            computed_settled_txs_root: settled,
            claimed_committed_txs_root: claimed,
            computed_committed_txs_root: computed,
            terminal_roots_required: required,
        }
    }

    #[test]
    fn settled_txs_root_matching_the_recompute_verifies() {
        let root = StateRoot::from_raw(Hash::from_bytes(b"state"));
        let settled = SettledTxsRoot::from_raw(Hash::from_bytes(b"settled"));
        assert!(
            root.verify(&settled_ctx(&root, Some(settled), Some(settled), true))
                .is_ok()
        );
    }

    #[test]
    fn missing_settled_txs_root_at_a_boundary_is_rejected() {
        let root = StateRoot::from_raw(Hash::from_bytes(b"state"));
        let recomputed = SettledTxsRoot::from_raw(Hash::from_bytes(b"settled"));
        assert_eq!(
            root.verify(&settled_ctx(&root, None, Some(recomputed), true))
                .unwrap_err(),
            StateRootVerifyError::MissingSettledTxsRoot,
        );
    }

    #[test]
    fn settled_txs_root_outside_a_boundary_is_rejected() {
        let root = StateRoot::from_raw(Hash::from_bytes(b"state"));
        let settled = SettledTxsRoot::from_raw(Hash::from_bytes(b"settled"));
        assert_eq!(
            root.verify(&settled_ctx(&root, Some(settled), None, false))
                .unwrap_err(),
            StateRootVerifyError::UnexpectedSettledTxsRoot,
        );
    }

    #[test]
    fn settled_txs_root_diverging_from_the_recompute_is_rejected() {
        let root = StateRoot::from_raw(Hash::from_bytes(b"state"));
        let claimed = SettledTxsRoot::from_raw(Hash::from_bytes(b"claimed"));
        let computed = SettledTxsRoot::from_raw(Hash::from_bytes(b"computed"));
        assert_eq!(
            root.verify(&settled_ctx(&root, Some(claimed), Some(computed), true))
                .unwrap_err(),
            StateRootVerifyError::SettledTxsRootMismatch {
                claimed,
                computed: Some(computed),
            },
        );
    }

    #[test]
    fn committed_txs_root_matching_the_recompute_verifies() {
        let root = StateRoot::from_raw(Hash::from_bytes(b"state"));
        let committed = CommittedTxsRoot::from_raw(Hash::from_bytes(b"committed window"));
        assert!(
            root.verify(&committed_ctx(
                &root,
                Some(committed),
                Some(committed),
                true
            ))
            .is_ok()
        );
    }

    #[test]
    fn missing_committed_txs_root_at_a_boundary_is_rejected() {
        let root = StateRoot::from_raw(Hash::from_bytes(b"state"));
        let recomputed = CommittedTxsRoot::from_raw(Hash::from_bytes(b"committed"));
        assert_eq!(
            root.verify(&committed_ctx(&root, None, Some(recomputed), true))
                .unwrap_err(),
            StateRootVerifyError::MissingCommittedTxsRoot,
        );
    }

    #[test]
    fn committed_txs_root_outside_a_boundary_is_rejected() {
        let root = StateRoot::from_raw(Hash::from_bytes(b"state"));
        let committed = CommittedTxsRoot::from_raw(Hash::from_bytes(b"committed"));
        assert_eq!(
            root.verify(&committed_ctx(&root, Some(committed), None, false))
                .unwrap_err(),
            StateRootVerifyError::UnexpectedCommittedTxsRoot,
        );
    }

    #[test]
    fn committed_txs_root_diverging_from_the_recompute_is_rejected() {
        let root = StateRoot::from_raw(Hash::from_bytes(b"state"));
        let claimed = CommittedTxsRoot::from_raw(Hash::from_bytes(b"claimed"));
        let computed = CommittedTxsRoot::from_raw(Hash::from_bytes(b"computed"));
        assert_eq!(
            root.verify(&committed_ctx(&root, Some(claimed), Some(computed), true))
                .unwrap_err(),
            StateRootVerifyError::CommittedTxsRootMismatch {
                claimed,
                computed: Some(computed),
            },
        );
    }

    /// Both roots ride the one predicate, so a header carrying only the
    /// settled half at a terminating boundary is refused just as a header
    /// carrying neither is.
    #[test]
    fn one_predicate_governs_both_terminal_roots() {
        let root = StateRoot::from_raw(Hash::from_bytes(b"state"));
        let settled = SettledTxsRoot::from_raw(Hash::from_bytes(b"settled"));
        let committed = CommittedTxsRoot::from_raw(Hash::from_bytes(b"committed"));

        let settled_only = StateRootContext {
            computed_root: &root,
            claimed_split_child_roots: None,
            split_child_roots_required: false,
            claimed_settled_txs_root: Some(settled),
            computed_settled_txs_root: Some(settled),
            claimed_committed_txs_root: None,
            computed_committed_txs_root: Some(committed),
            terminal_roots_required: true,
        };
        assert_eq!(
            root.verify(&settled_only).unwrap_err(),
            StateRootVerifyError::MissingCommittedTxsRoot,
        );

        let committed_only = StateRootContext {
            computed_root: &root,
            claimed_split_child_roots: None,
            split_child_roots_required: false,
            claimed_settled_txs_root: None,
            computed_settled_txs_root: Some(settled),
            claimed_committed_txs_root: Some(committed),
            computed_committed_txs_root: Some(committed),
            terminal_roots_required: true,
        };
        assert_eq!(
            root.verify(&committed_only).unwrap_err(),
            StateRootVerifyError::MissingSettledTxsRoot,
        );
    }
}
