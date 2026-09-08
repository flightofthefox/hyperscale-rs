//! Which transactions a shard writes a committed-transaction cell for.
//!
//! Every one it commits. The cell is what a leg reads to prove a
//! counterpart never committed the transaction, and the derivation reads
//! the block and the committing shard alone — no placement, no
//! classification, nothing a replica might hold differently. A prober
//! naming the cell and the shard whose commit wrote it name it from the
//! same two facts.
//!
//! Which absences *answer* is a separate rule and stays where it is:
//! `Probed::read` asks a core of more than one shard through this cell
//! and a core of one through its consumer's claim, because a core that
//! committed and refused writes no claim, and "never committed" and
//! "committed and refused" are different facts.

use hyperscale_storage::committed_tx_cells;
use hyperscale_types::{ShardId, SubstateKey, Transaction};

/// The committed cells a block of `shard` writes.
///
/// One per transaction it carries, derived the same way by the
/// proposer, every voter, a syncing replica and a split child following
/// the block — which for a child is the whole point: the creations turn
/// on nothing that the cut changes, so following the parent's block
/// under any window recomposes the parent's root.
#[must_use]
pub fn committed_cells<'a>(
    shard: ShardId,
    transactions: impl IntoIterator<Item = &'a Transaction>,
) -> Vec<(SubstateKey, Vec<u8>)> {
    committed_tx_cells(shard, transactions)
}

#[cfg(test)]
mod tests {
    use hyperscale_types::AddressClass;
    use hyperscale_vm_effects::Hash32;
    use hyperscale_vm_types::{Address, LegRole, LegShape, SubintentHash, ValueEdge};

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

    /// Every transaction writes the cell, whatever shape it has and
    /// whatever the trie says about where its nodes sit.
    ///
    /// The shape used to decide it — only a core spanning more than one
    /// shard wrote one — which made the block's creations a function of
    /// placement, and so of which window a reader classified under. A
    /// split child following its parent's block classified under the
    /// cut placed no core on the parent and derived no cell, and the
    /// halves did not recompose the parent's root. Nothing here reads a
    /// trie, so there is no window to get wrong.
    #[test]
    fn every_transaction_writes_the_cell_under_any_window() {
        use hyperscale_types::test_utils::{StubVmStatics, test_transaction};

        let alice = on(0, 0x11);
        let bob = on(1, 0x22);
        let venue = on(2, 0x33);
        let other = on(3, 0x44);

        let transfer = vec![
            leg(alice, LegRole::Attesting, &[]),
            leg(alice, LegRole::Inbound, &[]),
            leg(bob, LegRole::Outbound, &[(1, 0)]),
        ];
        let route = vec![
            leg(alice, LegRole::Attesting, &[]),
            leg(alice, LegRole::Inbound, &[]),
            leg(venue, LegRole::Core, &[(1, 0)]),
            leg(other, LegRole::Core, &[(2, 0)]),
            leg(alice, LegRole::Outbound, &[(3, 0)]),
        ];

        for (label, legs) in [("a transfer", transfer), ("a route", route)] {
            let tx = test_transaction(1).with_legs(&StubVmStatics, legs);
            for shard in (0..4).map(|path| ShardId::leaf(2, path)) {
                assert_eq!(
                    committed_cells(shard, [&tx]).len(),
                    1,
                    "{label} writes one cell on {shard:?}",
                );
            }
        }
    }

    /// A split child following its parent's block derives the parent's
    /// creations and its half recomposes the parent's root — under the
    /// child's own window, which is the case that used to fail.
    #[test]
    fn a_followed_block_recomposes_under_the_childs_own_window() {
        use std::sync::Arc;

        use hyperscale_storage::BoundaryStore;
        use hyperscale_storage::test_helpers::{block_settling, make_state_writes};
        use hyperscale_storage_memory::SimShardStorage;
        use hyperscale_types::test_utils::test_transaction;
        use hyperscale_types::{
            Block, BlockHeight, ConsensusReceipt, GlobalReceiptHash, SplitChildRoots,
            StoredReceipt, TxHash, Verifiable, shard_prefix_path,
        };

        let parent = ShardId::leaf(2, 2);
        let (left, right) = parent.children();
        let tx = test_transaction(1);
        let committed = committed_cells(parent, [&tx]);
        assert_eq!(committed.len(), 1);

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
        let children = SplitChildRoots {
            left: SimShardStorage::new(shard_prefix_path(left))
                .follow_block_writes(&block, &committed)
                .expect("a child follows"),
            right: SimShardStorage::new(shard_prefix_path(right))
                .follow_block_writes(&block, &committed)
                .expect("a child follows"),
        };
        assert!(children.composes_to(parent_root));
    }
}
