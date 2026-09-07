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
/// Returns `not_found` when the height is one this shard no longer
/// answers for — below the retention floor — or when the JMT version it
/// names is not held at all, so the requester rotates rather than
/// reading an empty tree as an empty answer. The floor is asked rather
/// than inferred from whether the collector has run: the same bound the
/// provisions server serves under, so both refuse the same heights.
/// Nothing is asked of the keys: a key nothing ever wrote proves absent
/// like any other.
#[must_use]
pub fn serve_state_proof_request<S: ShardStorage>(
    pending_chain: &std::sync::Arc<PendingChain<S>>,
    req: &GetStateProofRequest,
) -> GetStateProofResponse {
    let view = pending_chain.view_at_committed_tip();
    if !view.serves_at(req.height) {
        record_fetch_response_sent("state_proof", 0);
        return GetStateProofResponse::not_found();
    }
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

    use hyperscale_storage::test_helpers::{commit_settled_at, make_test_certified};
    use hyperscale_storage::{
        PendingChain, SubstateStore, committed_tx_cell_key, committed_tx_cells,
    };
    use hyperscale_storage_memory::SimShardStorage;
    use hyperscale_types::test_utils::test_transaction;
    use hyperscale_types::{
        AggregateSignature, BeaconWitnessCommit, BeaconWitnessLeafCount, Block, BlockHash,
        BlockHeader, BlockHeaderParts, BlockHeight, Inclusion, ProposerTimestamp,
        QuorumCertificate, RETENTION_HORIZON, Round, ShardId, SignerBitfield, StateRoot,
        Transaction, Verifiable, WeightedTimestamp, WitnessSources,
    };

    use super::*;

    const SHARD: ShardId = ShardId::ROOT;

    /// A chain committing `test_transaction(1)` in its first block, one
    /// block per entry of `stamps`, and the state root at that first
    /// height — the one a proof taken there reconstructs.
    fn chain_of(stamps: &[u64]) -> (Arc<PendingChain<SimShardStorage>>, StateRoot) {
        let storage = SimShardStorage::default();
        let mut first_root = StateRoot::ZERO;
        for (index, ts_ms) in stamps.iter().enumerate() {
            let height = u64::try_from(index).expect("small fixture") + 1;
            let parent_qc = QuorumCertificate::new(
                BlockHash::ZERO,
                SHARD,
                BlockHeight::new(height - 1),
                BlockHash::ZERO,
                Round::INITIAL,
                SignerBitfield::new(4),
                AggregateSignature::new([0u8; 96]),
                WeightedTimestamp::from_millis(*ts_ms),
            );
            let header = BlockHeader::new(BlockHeaderParts {
                shard_id: SHARD,
                height: BlockHeight::new(height),
                parent_block_hash: BlockHash::ZERO,
                parent_qc: parent_qc.into(),
                timestamp: ProposerTimestamp::from_millis(*ts_ms),
                provision_tx_roots: std::collections::BTreeMap::new(),
                ..Default::default()
            });
            let txs: Vec<Arc<Verifiable<Transaction>>> = if height == 1 {
                vec![Arc::new(Verifiable::from(test_transaction(1)))]
            } else {
                Vec::new()
            };
            let block = Block::Live {
                header,
                transactions: Arc::new(txs),
                certificates: Arc::new(Vec::new()),
                provisions: Arc::new(Vec::new()),
                abandonment_records: Arc::new(Vec::new()),
                state_proofs: Arc::new(Vec::new()),
                witness_sources: Arc::new(WitnessSources::empty()),
            };
            let creations = committed_tx_cells(
                SHARD,
                block.transactions().iter().map(|tx| tx.as_unverified()),
            );
            commit_settled_at(
                &storage,
                &make_test_certified(block),
                &creations,
                &[],
                &BeaconWitnessCommit::empty(BeaconWitnessLeafCount::ZERO),
            );
            if height == 1 {
                first_root = storage.state_root();
            }
        }
        (Arc::new(PendingChain::new(Arc::new(storage))), first_root)
    }

    /// A one-block chain committing `test_transaction(1)`, and its state
    /// root at that height.
    fn chain() -> (Arc<PendingChain<SimShardStorage>>, StateRoot) {
        chain_of(&[1_000])
    }

    /// A height the shard no longer answers for is refused, and the
    /// refusal is the retention floor's rather than the collector's:
    /// nothing here depends on whether a sweep has run, so this server
    /// and the provisions server serve the same span.
    #[test]
    fn a_height_below_the_retention_floor_is_not_served() {
        let horizon = u64::try_from(RETENTION_HORIZON.as_millis()).expect("fits");
        let (chain, _) = chain_of(&[1_000, 1_001 + horizon]);
        let key = committed_tx_cell_key(
            SHARD,
            test_transaction(1).hash(),
            test_transaction(1).validity_range().end_timestamp_exclusive,
        );

        let view = chain.view_at_committed_tip();
        assert!(
            !view.serves_at(BlockHeight::new(1)),
            "block one has aged out"
        );
        assert!(view.serves_at(BlockHeight::new(2)));

        let refused = serve_state_proof_request(
            &chain,
            &GetStateProofRequest::new(BlockHeight::new(1), vec![key]),
        );
        assert!(
            refused.proof.is_none(),
            "a height past the horizon is answered not-found, not proved",
        );
        let served = serve_state_proof_request(
            &chain,
            &GetStateProofRequest::new(BlockHeight::new(2), vec![key]),
        );
        assert!(served.proof.is_some(), "and the tip is still answered");
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
        let attested = proof.inclusions(root, SHARD, &keys).unwrap();
        assert_eq!(attested[0], (never, Inclusion::Absent));
        assert_eq!(attested[1].0, committed);
        assert!(attested[1].1.is_present());

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
