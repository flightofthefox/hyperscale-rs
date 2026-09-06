//! Which transactions a shard writes a committed-transaction cell for.
//!
//! The cell is what a leg reads to prove its core never committed the
//! transaction, and only a core of more than one shard is asked that
//! way: its shards settle on each other's certificates with no clock,
//! so a claim absent past the deadline may be pending there rather than
//! never coming. A core of one shard answers with its consumer's claim
//! cell, whose commit the deadline fences, and writes no cell. Every
//! shard of a multi-shard core writes it, so which one a leg names as
//! lowest does not matter.
//!
//! Decided here, off the engine's freeze, and handed to the store as the
//! block's creations: placement is the one fact the derivation reads
//! beyond the block, the store holds no topology, and a re-derivation
//! from the legs' declared roles would settle them differently from the
//! freeze a leg's probe reads.

use hyperscale_storage::committed_tx_cells;
use hyperscale_types::{Address, ShardId, ShardTrie, SubstateKey, Transaction};
use hyperscale_vm_types::LegShape;

use crate::legs::Classified;

/// Whether `local` writes the committed cell for a transaction of these
/// `legs` and `owners` under `trie`: the shape divides, its core spans
/// more than one shard, and `local` is in that core.
#[must_use]
pub fn writes_committed_cell(
    legs: &[LegShape],
    owners: &[Address],
    trie: &ShardTrie,
    local: ShardId,
) -> bool {
    let classified = Classified::freeze(legs, owners, trie);
    classified.decomposed().holds()
        && classified.core().len() > 1
        && classified.core().contains(&local)
}

/// The committed cells a block of `shard` writes under `trie`.
///
/// One per transaction it carries that [`writes_committed_cell`] names,
/// derived the same way by the proposer, every voter, a syncing replica
/// and a split child following the block.
#[must_use]
pub fn committed_cells<'a>(
    shard: ShardId,
    trie: &ShardTrie,
    transactions: impl IntoIterator<Item = &'a Transaction>,
) -> Vec<(SubstateKey, Vec<u8>)> {
    committed_tx_cells(
        shard,
        transactions
            .into_iter()
            .filter(|tx| writes_committed_cell(tx.legs(), tx.owners(), trie, shard)),
    )
}

#[cfg(test)]
mod tests {
    use hyperscale_types::AddressClass;
    use hyperscale_vm_effects::Hash32;
    use hyperscale_vm_types::{LegRole, SubintentHash, ValueEdge};

    use super::*;

    /// A component on the leaf at `path` of a four-leaf trie.
    fn on(path: u8, seed: u8) -> Address {
        let mut body = [seed; 31];
        body[0] = (path << 6) | (seed & 0x3F);
        Address::new(body, AddressClass::Component)
    }

    fn leg(target: Address, role: LegRole, edges: &[(u32, u32)]) -> LegShape {
        LegShape {
            target,
            role,
            edges: edges
                .iter()
                .map(|&(source, output)| ValueEdge {
                    source,
                    output,
                    non_fungible: false,
                })
                .collect(),
            presents: Vec::new(),
            declares: vec![target],
            intent: SubintentHash(Hash32([7; 32])),
            local: 0,
            expiry_ms: 1_000,
        }
    }

    /// A transfer's core is the sign-in alone and a swap's the venue
    /// alone: neither writes a cell on any shard. A route's core spans
    /// both venues' shards, and each of them writes it — the trader's
    /// shard, a leg off that core, does not.
    #[test]
    fn only_a_core_spanning_shards_writes_the_cell() {
        let trie = ShardTrie::uniform(2);
        let leaves: Vec<ShardId> = (0..4).map(|path| ShardId::leaf(2, path)).collect();
        let alice = on(0, 0x11);
        let bob = on(1, 0x22);
        let venue = on(2, 0x33);
        let other = on(3, 0x44);

        let transfer = [
            leg(alice, LegRole::Attesting, &[]),
            leg(alice, LegRole::Inbound, &[]),
            leg(bob, LegRole::Outbound, &[(1, 0)]),
        ];
        let swap = [
            leg(alice, LegRole::Attesting, &[]),
            leg(alice, LegRole::Inbound, &[]),
            leg(venue, LegRole::Core, &[(1, 0)]),
            leg(alice, LegRole::Outbound, &[(2, 0)]),
        ];
        let route = [
            leg(alice, LegRole::Attesting, &[]),
            leg(alice, LegRole::Inbound, &[]),
            leg(venue, LegRole::Core, &[(1, 0)]),
            leg(other, LegRole::Core, &[(2, 0)]),
            leg(alice, LegRole::Outbound, &[(3, 0)]),
        ];
        for &shard in &leaves {
            assert!(!writes_committed_cell(&transfer, &[], &trie, shard));
            assert!(!writes_committed_cell(&swap, &[], &trie, shard));
        }
        let writes: Vec<ShardId> = leaves
            .iter()
            .copied()
            .filter(|&shard| writes_committed_cell(&route, &[], &trie, shard))
            .collect();
        assert_eq!(writes, vec![ShardId::leaf(2, 2), ShardId::leaf(2, 3)]);
    }

    /// A split child following its parent's block derives the parent's
    /// creations under the parent's own window, the trie its committer
    /// froze the block's shapes under, and its half recomposes the
    /// parent's root. The cut places the parent's owners on its
    /// children, so under the child's own window no core contains the
    /// parent, the same block would create nothing, and the halves
    /// would not recompose.
    #[test]
    fn a_followed_block_is_classified_under_its_parents_window() {
        use std::sync::Arc;

        use hyperscale_storage::BoundaryStore;
        use hyperscale_storage::test_helpers::{block_settling, make_state_writes};
        use hyperscale_storage_memory::SimShardStorage;
        use hyperscale_types::test_utils::{StubVmStatics, test_transaction};
        use hyperscale_types::{
            Block, BlockHeight, ConsensusReceipt, GlobalReceiptHash, SplitChildRoots,
            StoredReceipt, TxHash, Verifiable, shard_prefix_path,
        };

        let parent = ShardId::leaf(2, 2);
        let (left, right) = parent.children();
        let window = ShardTrie::uniform(2);
        let cut = ShardTrie::from_leaves([
            ShardId::leaf(2, 0),
            ShardId::leaf(2, 1),
            left,
            right,
            ShardId::leaf(2, 3),
        ]);
        let alice = on(0, 0x11);
        let venue = on(2, 0x33);
        let other = on(3, 0x44);
        let route = vec![
            leg(alice, LegRole::Attesting, &[]),
            leg(alice, LegRole::Inbound, &[]),
            leg(venue, LegRole::Core, &[(1, 0)]),
            leg(other, LegRole::Core, &[(2, 0)]),
            leg(alice, LegRole::Outbound, &[(3, 0)]),
        ];
        let tx = test_transaction(1).with_legs(&StubVmStatics, route);
        let committed = committed_cells(parent, &window, [&tx]);
        assert_eq!(committed.len(), 1, "the parent is one of the route's core");
        let under_cut = committed_cells(parent, &cut, [&tx]);
        assert!(
            under_cut.is_empty(),
            "under the child's window no core contains the parent"
        );

        // The cell lands on the parent's left half; a settled write on
        // the right half keeps both subtrees populated, so the halves
        // recompose as internal nodes rather than a lone leaf.
        let right_half = StoredReceipt::synced(
            TxHash::ZERO,
            Arc::new(ConsensusReceipt::Succeeded {
                receipt_hash: GlobalReceiptHash::ZERO,
                writes: make_state_writes(0xA0, 1, vec![1; 4]),
                beacon_witness_events: Vec::new(),
                events: Vec::new(),
            }),
        );
        let Block::Live {
            header,
            certificates,
            provisions,
            abandonment_records,
            state_proofs,
            witness_sources,
            ..
        } = block_settling(BlockHeight::new(1), vec![right_half])
        else {
            unreachable!("the fixture builds a live block");
        };
        let block = Block::Live {
            header,
            transactions: Arc::new(vec![Arc::new(Verifiable::from(tx))]),
            certificates,
            provisions,
            abandonment_records,
            state_proofs,
            witness_sources,
        };

        let parent_root = SimShardStorage::new(shard_prefix_path(parent))
            .follow_block_writes(&block, &committed)
            .expect("the parent commits its block");
        let followed = |creations: &[(SubstateKey, Vec<u8>)]| SplitChildRoots {
            left: SimShardStorage::new(shard_prefix_path(left))
                .follow_block_writes(&block, creations)
                .expect("a child follows"),
            right: SimShardStorage::new(shard_prefix_path(right))
                .follow_block_writes(&block, creations)
                .expect("a child follows"),
        };
        assert!(
            followed(&committed).composes_to(parent_root),
            "children following under the parent's window recompose its root"
        );
        assert!(
            !followed(&under_cut).composes_to(parent_root),
            "under their own window they would not"
        );
    }
}
