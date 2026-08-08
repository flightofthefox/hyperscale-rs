//! Inbound settled-transaction window request handling.
//!
//! Serves a terminated shard's complete settled-transaction window list to a
//! surviving counterpart resolving cross-shard transactions across a split
//! boundary. The request names the terminal block `B`; the server
//! reconstructs `S_P` off its committed chain over the window reaching
//! back to the terminating reshape's admission — the same set `B`'s
//! `settled_txs_root` commits — so the requester accepts the list
//! against the beacon-attested root. No
//! per-block QC: completeness is the merkle root, not block-by-block
//! verification.

use hyperscale_metrics::record_fetch_response_sent;
use hyperscale_storage::{BlockForSync, PendingChain, ShardStorage};
use hyperscale_types::network::request::GetSettledTxsRequest;
use hyperscale_types::network::response::GetSettledTxsResponse;
use hyperscale_types::{MAX_FINALIZED_TX_PER_BLOCK, WeightedTimestamp, local_settled_tx_hashes};

/// Serve an inbound settled-transaction window request from the local chain.
///
/// The served set is the **cross-shard** transactions the terminated shard settled
/// in the window — the only ones a counterpart's fence can query (see
/// [`local_settled_tx_hashes`]) — so it stays proportional to cross-shard
/// traffic, not total throughput.
///
/// `window_floor` is the shard's settled-window floor read off the serving
/// node's topology projection — the same value the terminal's proposer
/// floored the attested root at, so the recomputed list matches it. A
/// projection that no longer carries the floor serves a narrower window;
/// the requester's root check catches the mismatch and rotates peers.
///
/// Returns `not_found` when the terminal block isn't held or the stored
/// block's hash doesn't match the requested terminal — the requester
/// rotates peers. Returns `not_found` too when the window set exceeds the
/// wire cap (logged loudly; within-cap for any realistic cross-shard load).
#[must_use]
pub fn serve_settled_txs_request<S: ShardStorage>(
    pending_chain: &PendingChain<S>,
    window_floor: Option<WeightedTimestamp>,
    req: &GetSettledTxsRequest,
) -> GetSettledTxsResponse {
    let Some(BlockForSync { block, .. }) = pending_chain.block_for_sync(req.terminal_height) else {
        record_fetch_response_sent("settled_txs", 0);
        return GetSettledTxsResponse::not_found();
    };
    if block.hash() != req.terminal_block_hash {
        record_fetch_response_sent("settled_txs", 0);
        return GetSettledTxsResponse::not_found();
    }

    let shard = block.header().shard_id();
    let own = local_settled_tx_hashes(block.certificates().iter(), shard);
    let Some(parent_height) = block.height().prev() else {
        // Genesis carries no certificates and never terminates a split.
        record_fetch_response_sent("settled_txs", 0);
        return GetSettledTxsResponse::not_found();
    };
    let set = pending_chain.settled_txs_in_window(
        shard,
        block.header().parent_block_hash(),
        parent_height,
        block.header().parent_qc().weighted_timestamp(),
        window_floor,
        own,
    );

    // A window exceeding the wire cap serves `not_found` rather than
    // shipping a response the receiver would reject at decode. The set is
    // the cross-shard settled transactions only — one entry each, across a
    // window spanning the retention horizon — so the headroom is that many
    // cross-shard transactions per horizon, not per block. An overflow
    // means cross-shard throughput outran the single-shot transfer and the
    // design must escalate to paged or JMT-absence-proof delivery (c2).
    // Log it loudly rather than letting the requester read the overflow
    // `not_found` as a plain "block not held" and rotate peers forever.
    let window = set.len();
    if window > MAX_FINALIZED_TX_PER_BLOCK {
        tracing::warn!(
            shard = ?shard,
            terminal_height = req.terminal_height.inner(),
            window,
            cap = MAX_FINALIZED_TX_PER_BLOCK,
            "settled-transaction window exceeds the wire cap; serving not_found — \
             cross-shard load outran the one-shot transfer (escalate to c2)"
        );
        record_fetch_response_sent("settled_txs", 0);
        return GetSettledTxsResponse::not_found();
    }
    record_fetch_response_sent("settled_txs", 1);
    GetSettledTxsResponse::found(set.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use hyperscale_storage::ShardChainWriter;
    use hyperscale_storage::test_helpers::make_test_certified;
    use hyperscale_storage_memory::SimShardStorage;
    use hyperscale_types::{
        AggregateSignature, BeaconWitnessCommit, BeaconWitnessLeafCount, BeaconWitnessRoot, Block,
        BlockHash, BlockHeader, BlockHeight, CertificateRoot, ExecutionCertificate,
        ExecutionOutcome, FinalizedWave, GlobalReceiptHash, GlobalReceiptRoot, Hash,
        LocalReceiptRoot, ProposerTimestamp, ProvisionsRoot, QuorumCertificate, RETENTION_HORIZON,
        RevealChain, Round, ShardId, ShardLoad, SignerBitfield, StateRoot, TickId, TransactionRoot,
        TxHash, TxOutcome, ValidatorId, Verifiable, Verified, WeightedTimestamp, WitnessSources,
        WorkInFlight, settled_txs_root_from_hashes,
    };

    use super::*;

    const SHARD: ShardId = ShardId::ROOT;

    /// The transaction the wave at `height` settles — distinct per wave,
    /// so a window over several waves has one entry each.
    fn settled_tx(height: u64) -> TxHash {
        TxHash::from(Hash::from_bytes(&height.to_le_bytes()))
    }

    fn finalized_wave(height: u64) -> Arc<Verifiable<FinalizedWave>> {
        // Cross-shard wave (non-empty `remote_shards`): the settled set
        // commits only cross-shard waves, so single-shard fixtures would be
        // filtered out before the merkle root.
        let wave = TickId::new(SHARD, BlockHeight::new(height));
        let ec = ExecutionCertificate::new(
            wave,
            WeightedTimestamp::from_millis(1),
            GlobalReceiptRoot::ZERO,
            vec![TxOutcome::new(
                settled_tx(height),
                ExecutionOutcome::Succeeded {
                    receipt_hash: GlobalReceiptHash::ZERO,
                },
            )],
            AggregateSignature::new([0u8; 96]),
            SignerBitfield::new(4),
        );
        // A counterpart's certificate for the same transaction: what makes
        // it reach beyond this shard, and so what puts it in the settled set.
        let remote = ExecutionCertificate::new(
            TickId::new(ShardId::from_heap_index(2), BlockHeight::new(height)),
            WeightedTimestamp::from_millis(1),
            GlobalReceiptRoot::ZERO,
            vec![TxOutcome::new(
                settled_tx(height),
                ExecutionOutcome::Succeeded {
                    receipt_hash: GlobalReceiptHash::ZERO,
                },
            )],
            AggregateSignature::new([0u8; 96]),
            SignerBitfield::new(4),
        );
        Arc::new(Verifiable::from(FinalizedWave::new(
            wave,
            vec![Arc::new(ec), Arc::new(remote)],
            vec![],
        )))
    }

    fn commit_block(
        storage: &SimShardStorage,
        height: u64,
        parent: BlockHash,
        pred_wt: u64,
        certs: &[Arc<Verifiable<FinalizedWave>>],
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
        let header = BlockHeader::new(
            SHARD,
            BlockHeight::new(height),
            parent,
            parent_qc,
            ValidatorId::new(0),
            ProposerTimestamp::from_millis(1_000 * height),
            Round::INITIAL,
            false,
            StateRoot::ZERO,
            TransactionRoot::ZERO,
            *Verified::<CertificateRoot>::compute(certs).as_ref(),
            LocalReceiptRoot::ZERO,
            ProvisionsRoot::ZERO,
            Vec::new(),
            std::collections::BTreeMap::new(),
            WorkInFlight::ZERO,
            BeaconWitnessRoot::ZERO,
            BeaconWitnessLeafCount::ZERO,
            BeaconWitnessLeafCount::ZERO,
            RevealChain::ZERO,
            None,
            None,
            ShardLoad::ZERO,
        );
        let block = Block::Live {
            header,
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(certs.to_vec()),
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

    /// The served window list recomputes to the terminal block's
    /// `settled_txs_root` — every block's settled transaction over the window.
    #[test]
    fn serves_the_full_settled_window() {
        let storage = SimShardStorage::default();
        let mut parent = BlockHash::ZERO;
        for h in 1..=3 {
            parent = commit_block(&storage, h, parent, 1_000 * h, &[finalized_wave(h)]);
        }
        let terminal = parent;
        let pending_chain = PendingChain::new(Arc::new(storage));

        let req = GetSettledTxsRequest::new(BlockHeight::new(3), terminal);
        let response = serve_settled_txs_request(&pending_chain, None, &req);
        let served = response.txs.expect("terminal block is held");

        let expected: BTreeSet<TxHash> = (1..=3).map(settled_tx).collect();
        assert_eq!(served.iter().copied().collect::<BTreeSet<_>>(), expected);
        // The fence accepts iff the recomputed root equals the attested one.
        assert_eq!(
            settled_txs_root_from_hashes(served.iter()),
            settled_txs_root_from_hashes(expected.iter()),
        );
    }

    /// A schedule-supplied floor reaches settlements older than the
    /// anchor-relative horizon: a transaction settled early in the terminating
    /// shard's scheduled window — below `terminal − RETENTION_HORIZON` —
    /// is served only when the floor covers it.
    #[test]
    fn window_floor_serves_early_settlements() {
        let rh_ms = RETENTION_HORIZON.as_secs() * 1000;
        let storage = SimShardStorage::default();
        let mut parent = commit_block(&storage, 1, BlockHash::ZERO, 1_000, &[finalized_wave(1)]);
        parent = commit_block(&storage, 2, parent, rh_ms + 10_000, &[finalized_wave(2)]);
        let terminal = commit_block(&storage, 3, parent, rh_ms + 11_000, &[finalized_wave(3)]);
        let pending_chain = PendingChain::new(Arc::new(storage));
        let req = GetSettledTxsRequest::new(BlockHeight::new(3), terminal);

        // Anchor-only floor: the early settlement falls outside the window.
        let narrow = serve_settled_txs_request(&pending_chain, None, &req)
            .txs
            .expect("terminal block is held");
        assert_eq!(narrow.len(), 2);

        // The floor reaches back past the early settlement.
        let wide = serve_settled_txs_request(
            &pending_chain,
            Some(WeightedTimestamp::from_millis(500)),
            &req,
        )
        .txs
        .expect("terminal block is held");
        assert_eq!(wide.len(), 3);
    }

    /// A hash mismatch against the stored block serves `not_found`.
    #[test]
    fn wrong_terminal_hash_serves_not_found() {
        let storage = SimShardStorage::default();
        let _ = commit_block(&storage, 1, BlockHash::ZERO, 1_000, &[finalized_wave(1)]);
        let pending_chain = PendingChain::new(Arc::new(storage));
        let req = GetSettledTxsRequest::new(
            BlockHeight::new(1),
            BlockHash::from_raw(Hash::from_bytes(b"other-chain")),
        );
        assert!(
            serve_settled_txs_request(&pending_chain, None, &req)
                .txs
                .is_none()
        );
    }

    /// An unheld height serves `not_found`.
    #[test]
    fn unheld_height_serves_not_found() {
        let storage = Arc::new(SimShardStorage::default());
        let pending_chain = PendingChain::new(storage);
        let req = GetSettledTxsRequest::new(BlockHeight::new(7), BlockHash::ZERO);
        assert!(
            serve_settled_txs_request(&pending_chain, None, &req)
                .txs
                .is_none()
        );
    }
}
