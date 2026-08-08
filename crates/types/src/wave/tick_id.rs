//! [`TickId`] — the identity of one shard's execution batch.

use std::fmt::{self, Display};

use hyperscale_hbor::Hbor;

use crate::{BlockHeight, ShardId};

/// Identifier of a shard's execution batch: the shard, and the height of
/// the block whose commit chained it onto the last.
///
/// Globally unique, and self-contained — no composite
/// `(block_hash, …)` key is needed anywhere. It names no destination:
/// which shards a batch's transactions reach is a question about those
/// transactions, and answering it here would partition one batch's
/// outcomes into several certificates by destination set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Hbor)]
pub struct TickId {
    shard_id: ShardId,
    block_height: BlockHeight,
}

impl TickId {
    /// The batch committed by `shard_id` at `block_height`.
    #[must_use]
    pub const fn new(shard_id: ShardId, block_height: BlockHeight) -> Self {
        Self {
            shard_id,
            block_height,
        }
    }

    /// The shard that committed the block containing this batch's transactions.
    #[must_use]
    pub const fn shard_id(&self) -> ShardId {
        self.shard_id
    }

    /// Block height at which the batch's transactions were committed.
    #[must_use]
    pub const fn block_height(&self) -> BlockHeight {
        self.block_height
    }
}

impl Display for TickId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tick(shard={}, h={})",
            self.shard_id.inner(),
            self.block_height.inner()
        )
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_hbor::{from_slice as hbor_from_slice, to_vec as hbor_to_vec};

    use super::*;

    #[test]
    fn hbor_round_trip() {
        let tick = TickId::new(ShardId::leaf(3, 0), BlockHeight::new(42));
        let bytes = hbor_to_vec(&tick).unwrap();
        assert_eq!(hbor_from_slice::<TickId>(&bytes).unwrap(), tick);
    }

    /// Ordering is `(shard, height)` lexicographically — the order a
    /// proposer's certificate list is built in, and the order settlement
    /// follows.
    #[test]
    fn orders_by_shard_then_height() {
        let a = TickId::new(ShardId::leaf(3, 0), BlockHeight::new(1));
        let b = TickId::new(ShardId::leaf(3, 0), BlockHeight::new(2));
        let c = TickId::new(ShardId::leaf(3, 1), BlockHeight::new(1));
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn display_names_the_shard_and_height() {
        let shard = ShardId::leaf(3, 1);
        let tick = TickId::new(shard, BlockHeight::new(7));
        assert_eq!(
            tick.to_string(),
            format!("Tick(shard={}, h=7)", shard.inner())
        );
    }
}
