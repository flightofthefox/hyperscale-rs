//! [`ProvisionsRoot`]: the root over a block's provision batches, one
//! leaf per batch hash.

use crate::{Hash, LeafRoot, ProvisionsRoot};

impl LeafRoot for ProvisionsRoot {
    type Leaf = Hash;

    const ZERO: Self = Self::ZERO;

    fn from_raw(raw: Hash) -> Self {
        Self::from_raw(raw)
    }

    fn leaf(batch_hash: &Self::Leaf) -> Hash {
        *batch_hash
    }
}
