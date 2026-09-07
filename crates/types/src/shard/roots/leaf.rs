//! A header root that is the merkle root over one leaf per item of a
//! block section, and the one way every such root is computed and
//! verified.

use std::fmt;

use thiserror::Error;

use crate::{Hash, Verified, Verify, compute_merkle_root};

/// A header root over a block section: one leaf per item, in block
/// order, under a padded merkle tree; the zero root for an empty
/// section.
///
/// Each implementor says what its items are and how one leafs; what
/// the root over them is, and what verifying a claimed root against
/// the section means, is said here once.
pub trait LeafRoot: Copy + PartialEq + fmt::Debug {
    /// The item one leaf covers.
    type Leaf;

    /// The root of an empty section.
    const ZERO: Self;

    /// The root over the merkle root `raw` of a non-empty section.
    fn from_raw(raw: Hash) -> Self;

    /// One item's leaf.
    fn leaf(item: &Self::Leaf) -> Hash;

    /// The root over `items`, in block order.
    fn over(items: &[Self::Leaf]) -> Self {
        if items.is_empty() {
            return Self::ZERO;
        }
        let leaves: Vec<Hash> = items.iter().map(Self::leaf).collect();
        Self::from_raw(compute_merkle_root(&leaves))
    }
}

/// The one way a leaf root's verification fails: the section recomputes
/// to a root other than the one the header claims.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
#[error("computed root {computed:?} ≠ claimed {expected:?}")]
pub struct RootMismatch<R> {
    /// The header's claimed root.
    pub expected: R,
    /// The root the section recomputes to.
    pub computed: R,
}

/// Construction asserts: `self` is [`LeafRoot::over`] the section.
impl<'a, R: LeafRoot> Verify<&'a [R::Leaf]> for R {
    type Error = RootMismatch<R>;

    fn verify(&self, items: &'a [R::Leaf]) -> Result<Verified<Self>, Self::Error> {
        let computed = R::over(items);
        if computed != *self {
            return Err(RootMismatch {
                expected: *self,
                computed,
            });
        }
        Ok(Verified::new_unchecked(*self))
    }
}

impl<R: LeafRoot> Verified<R> {
    /// The root over `items`. Verified by construction.
    #[must_use]
    pub fn compute(items: &[R::Leaf]) -> Self {
        Self::new_unchecked(R::over(items))
    }

    /// Re-wrap a root the verification pipeline's per-root tracking has
    /// already confirmed.
    #[must_use]
    pub const fn from_pipeline_attestation(root: R) -> Self {
        Self::new_unchecked(root)
    }
}
