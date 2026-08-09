//! Inbound committed-transaction membership handling.
//!
//! Answers a reshape successor asking whether this shard, before it
//! terminated, committed a transaction whose validity window opened
//! before the successor's origin. The successor refuses all such
//! transactions by default; an `Absent` answer here is what lets it admit
//! one, so absence is proven against the terminal's `committed_txs_root`
//! and inclusion is not — see [`CommittedTxVerdict`].
//!
//! The set is reconstructed off the committed chain over the same window
//! the terminal's proposer rooted, so the proofs verify against the
//! attested root without this server being trusted for any of it.

use hyperscale_metrics::record_fetch_response_sent;
use hyperscale_storage::{BlockForSync, PendingChain, ShardStorage};
use hyperscale_types::network::request::GetCommittedTxsRequest;
use hyperscale_types::network::response::{CommittedTxVerdict, GetCommittedTxsResponse};
use hyperscale_types::{TxHash, prove_committed_tx_absent};

/// Serve an inbound committed-transaction query from the local chain.
///
/// Returns `not_found` when the terminal block isn't held or the stored
/// block's hash doesn't match the requested terminal — the requester
/// rotates peers.
#[must_use]
pub fn serve_committed_txs_request<S: ShardStorage>(
    pending_chain: &PendingChain<S>,
    req: &GetCommittedTxsRequest,
) -> GetCommittedTxsResponse {
    let Some(BlockForSync { block, .. }) = pending_chain.block_for_sync(req.terminal_height) else {
        record_fetch_response_sent("committed_txs", 0);
        return GetCommittedTxsResponse::not_found();
    };
    if block.hash() != req.terminal_block_hash {
        record_fetch_response_sent("committed_txs", 0);
        return GetCommittedTxsResponse::not_found();
    }
    let Some(parent_height) = block.height().prev() else {
        // Genesis carries no transactions and never terminates a chain.
        record_fetch_response_sent("committed_txs", 0);
        return GetCommittedTxsResponse::not_found();
    };

    let own: Vec<TxHash> = block.transactions().iter().map(|tx| tx.hash()).collect();
    // Sorted and deduplicated by the walk's `BTreeSet`, which is what the
    // absence proofs' leaf indices are relative to.
    let members: Vec<TxHash> = pending_chain
        .committed_txs_in_window(
            block.header().parent_block_hash(),
            parent_height,
            block.header().parent_qc().weighted_timestamp(),
            own,
        )
        .into_iter()
        .collect();

    let verdicts = req
        .tx_hashes
        .iter()
        .map(|tx_hash| {
            prove_committed_tx_absent(&members, tx_hash)
                .map_or(CommittedTxVerdict::Committed, CommittedTxVerdict::Absent)
        })
        .collect();
    record_fetch_response_sent("committed_txs", 1);
    GetCommittedTxsResponse::found(verdicts)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hyperscale_storage::ShardChainWriter;
    use hyperscale_storage::test_helpers::make_test_certified;
    use hyperscale_storage_memory::SimShardStorage;
    use hyperscale_types::test_utils::test_transaction;
    use hyperscale_types::{
        AggregateSignature, BeaconWitnessCommit, BeaconWitnessLeafCount, Block, BlockHash,
        BlockHeader, BlockHeaderParts, BlockHeight, Hash, ProposerTimestamp, QuorumCertificate,
        RETENTION_HORIZON, Round, ShardId, SignerBitfield, Transaction, Verifiable,
        WeightedTimestamp, WitnessSources, committed_txs_root_from_hashes,
    };

    use super::*;

    const SHARD: ShardId = ShardId::ROOT;

    fn tx_hash(seed: u8) -> TxHash {
        test_transaction(seed).hash()
    }

    /// Commit a block at `height` carrying `test_transaction(seed)` for
    /// each seed, with `pred_wt` as its parent-QC weighted timestamp —
    /// the clock the window floor reads.
    fn commit_block(
        storage: &SimShardStorage,
        height: u64,
        parent: BlockHash,
        pred_wt: u64,
        seeds: &[u8],
    ) -> BlockHash {
        let parent_qc = QuorumCertificate::new(
            parent,
            SHARD,
            BlockHeight::new(height.saturating_sub(1)),
            BlockHash::ZERO,
            Round::INITIAL,
            SignerBitfield::new(4),
            AggregateSignature::new([0u8; 96]),
            WeightedTimestamp::from_millis(pred_wt),
        );
        let header = BlockHeader::new(BlockHeaderParts {
            shard_id: SHARD,
            height: BlockHeight::new(height),
            parent_block_hash: parent,
            parent_qc: parent_qc.into(),
            timestamp: ProposerTimestamp::from_millis(1_000 * height),
            provision_tx_roots: std::collections::BTreeMap::new(),
            ..Default::default()
        });
        let txs: Vec<Arc<Verifiable<Transaction>>> = seeds
            .iter()
            .map(|&seed| Arc::new(Verifiable::from(test_transaction(seed))))
            .collect();
        let block = Block::Live {
            header,
            transactions: Arc::new(txs),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        };
        let hash = block.hash();
        storage.commit_block(
            &make_test_certified(block),
            &BeaconWitnessCommit::empty(BeaconWitnessLeafCount::ZERO),
        );
        hash
    }

    /// A three-block chain committing seeds 1..=6, and its terminal hash.
    fn chain() -> (PendingChain<SimShardStorage>, BlockHash) {
        let storage = SimShardStorage::default();
        let mut parent = BlockHash::ZERO;
        for (h, seeds) in [(1u64, &[1u8, 2][..]), (2, &[3, 4]), (3, &[5, 6])] {
            parent = commit_block(&storage, h, parent, 1_000 * h, seeds);
        }
        (PendingChain::new(Arc::new(storage)), parent)
    }

    /// A transaction the chain committed answers `Committed`; one it never
    /// saw answers `Absent` with a proof that verifies against the root
    /// the terminal's proposer would have attested.
    #[test]
    fn answers_membership_against_the_attested_root() {
        let (pending_chain, terminal) = chain();
        let absent = tx_hash(99);
        let req = GetCommittedTxsRequest::new(
            BlockHeight::new(3),
            terminal,
            vec![tx_hash(1), tx_hash(5), absent],
        );
        let verdicts = serve_committed_txs_request(&pending_chain, &req)
            .verdicts
            .expect("terminal block is held");
        assert_eq!(verdicts.len(), 3);

        assert_eq!(
            verdicts[0],
            CommittedTxVerdict::Committed,
            "seed 1 committed"
        );
        assert_eq!(
            verdicts[1],
            CommittedTxVerdict::Committed,
            "seed 5 committed"
        );

        // The root the terminal's proposer attests over this window.
        let root = committed_txs_root_from_hashes(
            [1u8, 2, 3, 4, 5, 6]
                .iter()
                .map(|&s| tx_hash(s))
                .collect::<Vec<_>>()
                .iter(),
        );
        let CommittedTxVerdict::Absent(proof) = &verdicts[2] else {
            panic!("an uncommitted transaction must answer Absent");
        };
        assert!(proof.proves_absent(&absent, root));
    }

    /// Every transaction the chain committed is refused an absence proof
    /// — the property the successor's safety rests on, checked across the
    /// whole window rather than one sample.
    #[test]
    fn no_committed_transaction_is_shown_absent() {
        let (pending_chain, terminal) = chain();
        let committed: Vec<TxHash> = (1..=6u8).map(tx_hash).collect();
        let req = GetCommittedTxsRequest::new(BlockHeight::new(3), terminal, committed.clone());
        let verdicts = serve_committed_txs_request(&pending_chain, &req)
            .verdicts
            .expect("terminal block is held");
        for (verdict, tx) in verdicts.iter().zip(&committed) {
            assert_eq!(
                verdict,
                &CommittedTxVerdict::Committed,
                "{tx:?} was committed and must not be shown absent"
            );
        }
    }

    /// The window floors at `RETENTION_HORIZON` behind the terminal's
    /// anchor: a transaction committed below the floor is outside the set
    /// the terminal roots, so it answers `Absent` against that root — and
    /// its proof verifies, because the root excludes it too.
    #[test]
    fn a_transaction_below_the_floor_is_outside_the_window() {
        let rh_ms = RETENTION_HORIZON.as_secs() * 1000;
        let storage = SimShardStorage::default();
        let mut parent = commit_block(&storage, 1, BlockHash::ZERO, 1_000, &[10]);
        parent = commit_block(&storage, 2, parent, rh_ms + 10_000, &[11]);
        let terminal = commit_block(&storage, 3, parent, rh_ms + 11_000, &[12]);
        let pending_chain = PendingChain::new(Arc::new(storage));

        let below_floor = tx_hash(10);
        let req = GetCommittedTxsRequest::new(BlockHeight::new(3), terminal, vec![below_floor]);
        let verdicts = serve_committed_txs_request(&pending_chain, &req)
            .verdicts
            .expect("terminal block is held");

        let root = committed_txs_root_from_hashes([tx_hash(11), tx_hash(12)].iter());
        let CommittedTxVerdict::Absent(proof) = &verdicts[0] else {
            panic!("a below-floor transaction is outside the rooted window");
        };
        assert!(proof.proves_absent(&below_floor, root));
    }

    /// An empty query is answered, not refused — the caller asked nothing
    /// and gets nothing back, rather than reading a `not_found` and
    /// rotating peers over it.
    #[test]
    fn an_empty_query_answers_empty() {
        let (pending_chain, terminal) = chain();
        let req = GetCommittedTxsRequest::new(BlockHeight::new(3), terminal, Vec::new());
        assert_eq!(
            serve_committed_txs_request(&pending_chain, &req).verdicts,
            Some(Vec::new())
        );
    }

    /// A hash mismatch against the stored block serves `not_found`.
    #[test]
    fn wrong_terminal_hash_serves_not_found() {
        let (pending_chain, _) = chain();
        let req = GetCommittedTxsRequest::new(
            BlockHeight::new(3),
            BlockHash::from_raw(Hash::from_bytes(b"other-chain")),
            vec![tx_hash(1)],
        );
        assert!(
            serve_committed_txs_request(&pending_chain, &req)
                .verdicts
                .is_none()
        );
    }

    /// An unheld height serves `not_found`.
    #[test]
    fn unheld_height_serves_not_found() {
        let pending_chain = PendingChain::new(Arc::new(SimShardStorage::default()));
        let req =
            GetCommittedTxsRequest::new(BlockHeight::new(7), BlockHash::ZERO, vec![tx_hash(1)]);
        assert!(
            serve_committed_txs_request(&pending_chain, &req)
                .verdicts
                .is_none()
        );
    }
}
