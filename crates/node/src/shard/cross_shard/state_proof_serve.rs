//! Inbound state-proof handling.
//!
//! Answers a peer asking what this shard's state held at a committed
//! height for a set of keys — present or absent, one multiproof over
//! all of them. The requester holds the commit-proven header for the
//! height and checks the proof against its state root, so this server
//! is trusted for nothing: a proof against any other tree fails to
//! reconstruct that root and is rotated off.

use hyperscale_metrics::record_fetch_response_sent;
use hyperscale_storage::tree::proofs::generate_proof;
use hyperscale_storage::{PendingChain, ShardStorage};
use hyperscale_types::network::request::GetStateProofRequest;
use hyperscale_types::network::response::GetStateProofResponse;

/// Serve an inbound state-proof query from the committed chain.
///
/// Returns `not_found` when the JMT version the height names is not
/// held — never committed here, or pruned past the tree's history — so
/// the requester rotates rather than reading an empty tree as an empty
/// answer. Nothing is asked of the keys: a key nothing ever wrote proves
/// absent like any other.
#[must_use]
pub fn serve_state_proof_request<S: ShardStorage>(
    pending_chain: &std::sync::Arc<PendingChain<S>>,
    req: &GetStateProofRequest,
) -> GetStateProofResponse {
    let view = pending_chain.view_at_committed_tip();
    generate_proof(view.as_ref(), &req.keys, req.height).map_or_else(
        || {
            record_fetch_response_sent("state_proof", 0);
            GetStateProofResponse::not_found()
        },
        |proof| {
            record_fetch_response_sent("state_proof", req.keys.len());
            GetStateProofResponse::found(proof)
        },
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hyperscale_storage::test_helpers::make_test_certified;
    use hyperscale_storage::{
        PendingChain, ShardChainWriter, SubstateStore, committed_tx_cell_key,
    };
    use hyperscale_storage_memory::SimShardStorage;
    use hyperscale_types::test_utils::test_transaction;
    use hyperscale_types::{
        AggregateSignature, BeaconWitnessCommit, BeaconWitnessLeafCount, Block, BlockHash,
        BlockHeader, BlockHeaderParts, BlockHeight, Inclusion, ProposerTimestamp,
        QuorumCertificate, Round, ShardId, SignerBitfield, StateRoot, Transaction, Verifiable,
        WeightedTimestamp, WitnessSources,
    };

    use super::*;

    const SHARD: ShardId = ShardId::ROOT;

    /// A one-block chain committing `test_transaction(1)`, and its state
    /// root at that height.
    fn chain() -> (Arc<PendingChain<SimShardStorage>>, StateRoot) {
        let storage = SimShardStorage::default();
        let parent_qc = QuorumCertificate::new(
            BlockHash::ZERO,
            SHARD,
            BlockHeight::new(0),
            BlockHash::ZERO,
            Round::INITIAL,
            SignerBitfield::new(4),
            AggregateSignature::new([0u8; 96]),
            WeightedTimestamp::from_millis(1_000),
        );
        let header = BlockHeader::new(BlockHeaderParts {
            shard_id: SHARD,
            height: BlockHeight::new(1),
            parent_block_hash: BlockHash::ZERO,
            parent_qc: parent_qc.into(),
            timestamp: ProposerTimestamp::from_millis(1_000),
            provision_tx_roots: std::collections::BTreeMap::new(),
            ..Default::default()
        });
        let txs: Vec<Arc<Verifiable<Transaction>>> =
            vec![Arc::new(Verifiable::from(test_transaction(1)))];
        let block = Block::Live {
            header,
            transactions: Arc::new(txs),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            abandonment_records: Arc::new(Vec::new()),
            state_proofs: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        };
        storage.commit_block(
            &make_test_certified(block),
            &[],
            &BeaconWitnessCommit::empty(BeaconWitnessLeafCount::ZERO),
        );
        let root = storage.state_root();
        (Arc::new(PendingChain::new(Arc::new(storage))), root)
    }

    /// The committed cell of a transaction the chain committed proves
    /// present under the height's root, one it never saw proves absent,
    /// and a height not held answers `not_found`.
    #[test]
    fn proves_the_committed_cell_present_or_absent_under_the_root() {
        let (chain, root) = chain();
        let cell = |tx: &Transaction| {
            committed_tx_cell_key(
                SHARD,
                tx.hash(),
                tx.validity_range().end_timestamp_exclusive,
            )
        };
        let committed = cell(&test_transaction(1));
        let never = cell(&test_transaction(99));
        let keys = vec![never, committed];

        let response = serve_state_proof_request(
            &chain,
            &GetStateProofRequest::new(BlockHeight::new(1), keys.clone()),
        );
        let proof = response.proof.expect("the height is held");
        assert_eq!(
            proof.inclusions(root, SHARD, &keys).unwrap(),
            vec![(never, Inclusion::Absent), (committed, Inclusion::Present)]
        );

        let unheld = serve_state_proof_request(
            &chain,
            &GetStateProofRequest::new(BlockHeight::new(7), keys),
        );
        assert!(
            unheld.proof.is_none(),
            "a height not held is not an empty tree"
        );
    }
}
