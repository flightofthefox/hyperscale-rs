//! Per-shard settled-set acquisition.
//!
//! When a remote shard `P` terminates — a split's parent, or either child
//! of a merge — a surviving counterpart must learn `S_P`, the transactions `P`
//! settled at or before its terminal block, so the boundary fence can
//! resolve cross-shard `Finalization`s naming `P`. The acquisition scan
//! keys on a shard having left the trie, so it covers both reshapes
//! identically. It owns one acquisition per
//! past-terminal shard: a single verified fetch of `P`'s complete settled
//! window list, checked against the beacon-attested `settled_txs_root`
//! the node read from its own fold.
//!
//! Sans-io like the [`Sync`](crate::sync) FSMs: methods fold an input and
//! return [`SettledTxsAcquisitionOutput`]s the I/O glue turns into
//! network requests and a `SettledTxsReconstructed` event. A correct
//! terminal committee satisfies the root check on the first fetch; a
//! `not_found` or a list that doesn't recompute to the attested root
//! rotates the peer and retries on the next tick. Each driver self-expires
//! once the node's chain advances past `P`'s terminal-evidence expiry,
//! beyond which the fence rejects any outcome naming `P` regardless.

use std::collections::{BTreeSet, HashMap};

use hyperscale_types::network::request::GetSettledTxsRequest;
use hyperscale_types::network::response::GetSettledTxsResponse;
use hyperscale_types::{
    ShardId, TerminalEvidence, TxHash, ValidatorId, WeightedTimestamp, settled_txs_root_from_hashes,
};

/// One in-flight acquisition of a terminated shard's settled set.
struct AcquisitionDriver {
    /// `P`'s terminal, the root its list must recompute to, and when the
    /// answer stops being readable.
    evidence: TerminalEvidence,
    /// `P`'s terminal committee, asked in rotation. Empty falls back to
    /// shard-routed peer selection.
    peers: Vec<ValidatorId>,
    /// Rotates through `peers` on each `not_found` / mismatch / failure.
    cursor: usize,
    /// Whether a fetch is outstanding — withholds duplicate fetches.
    in_flight: bool,
}

impl AcquisitionDriver {
    const fn request(&self) -> GetSettledTxsRequest {
        GetSettledTxsRequest::new(self.evidence.height, self.evidence.block_hash)
    }

    fn peer(&self) -> Option<ValidatorId> {
        if self.peers.is_empty() {
            None
        } else {
            Some(self.peers[self.cursor % self.peers.len()])
        }
    }
}

/// What the I/O glue should do after folding an input into [`SettledTxsAcquisition`].
pub enum SettledTxsAcquisitionOutput {
    /// Issue the window fetch against `shard`'s terminal committee, biased
    /// to `peer`.
    Fetch {
        /// The terminated shard being acquired.
        shard: ShardId,
        /// Preferred terminal-committee member, or `None` to route by
        /// shard alone.
        peer: Option<ValidatorId>,
        /// The window list request.
        request: GetSettledTxsRequest,
    },
    /// The fetched list verified against the attested root — `S_P` is
    /// complete.
    Complete {
        /// The terminated shard whose settled set this is.
        shard: ShardId,
        /// Tick-ids `shard` settled at or before its terminal block.
        txs: BTreeSet<TxHash>,
        /// `shard`'s terminal weighted timestamp.
        terminal_wt: WeightedTimestamp,
    },
}

/// Drives one settled-set acquisition per past-terminal shard. One per
/// [`ShardIo`](crate::shard::ShardIo); shared across the shard's vnodes, so a
/// duplicate start for an already-targeted terminal is deduplicated.
#[derive(Default)]
pub struct SettledTxsAcquisition {
    drivers: HashMap<ShardId, AcquisitionDriver>,
}

impl SettledTxsAcquisition {
    /// An empty acquisition set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            drivers: HashMap::new(),
        }
    }

    /// True while any acquisition is unfinished — keeps the shard's
    /// `FetchTick` alive so parked acquisitions retry and expired ones
    /// drop.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        !self.drivers.is_empty()
    }

    /// Begin (or retry) acquiring `shard`'s settled set. A start for a
    /// terminal already in flight is a no-op; one for a parked driver
    /// re-issues the fetch against the next peer; one naming a different
    /// terminal block replaces the running driver.
    pub fn start(
        &mut self,
        shard: ShardId,
        evidence: TerminalEvidence,
        peers: Vec<ValidatorId>,
    ) -> Vec<SettledTxsAcquisitionOutput> {
        if let Some(driver) = self.drivers.get_mut(&shard)
            && driver.evidence.block_hash == evidence.block_hash
        {
            if driver.in_flight {
                return vec![];
            }
            driver.in_flight = true;
            return vec![SettledTxsAcquisitionOutput::Fetch {
                shard,
                peer: driver.peer(),
                request: driver.request(),
            }];
        }
        let driver = AcquisitionDriver {
            evidence,
            peers,
            cursor: 0,
            in_flight: true,
        };
        let out = SettledTxsAcquisitionOutput::Fetch {
            shard,
            peer: driver.peer(),
            request: driver.request(),
        };
        self.drivers.insert(shard, driver);
        vec![out]
    }

    /// Fold a window response into `shard`'s acquisition. A list that
    /// recomputes to the attested root completes; `not_found` or a
    /// mismatch rotates the peer and parks for the next tick.
    pub fn on_response(
        &mut self,
        shard: ShardId,
        response: &GetSettledTxsResponse,
    ) -> Vec<SettledTxsAcquisitionOutput> {
        let Some(driver) = self.drivers.get_mut(&shard) else {
            return vec![];
        };
        driver.in_flight = false;
        let Some(txs) = &response.txs else {
            driver.cursor = driver.cursor.wrapping_add(1);
            return vec![];
        };
        if settled_txs_root_from_hashes(txs.iter()) != driver.evidence.attested_root {
            driver.cursor = driver.cursor.wrapping_add(1);
            return vec![];
        }
        let set: BTreeSet<TxHash> = txs.iter().copied().collect();
        let driver = self
            .drivers
            .remove(&shard)
            .expect("just matched as present");
        vec![SettledTxsAcquisitionOutput::Complete {
            shard,
            txs: set,
            terminal_wt: driver.evidence.terminal_wt,
        }]
    }

    /// A transport-level failure of the outstanding fetch. Re-arms the
    /// driver and rotates the peer; the next tick re-issues.
    pub fn on_failure(&mut self, shard: ShardId) {
        if let Some(driver) = self.drivers.get_mut(&shard) {
            driver.in_flight = false;
            driver.cursor = driver.cursor.wrapping_add(1);
        }
    }

    /// Drop acquisitions whose evidence window has passed (the fence
    /// rejects naming the shard regardless), then re-issue every parked
    /// acquisition's fetch. `now_wt` is the node's current chain weighted
    /// timestamp, or `None` before the first commit.
    pub fn on_tick(
        &mut self,
        now_wt: Option<WeightedTimestamp>,
    ) -> Vec<SettledTxsAcquisitionOutput> {
        if let Some(now) = now_wt {
            self.drivers
                .retain(|_, d| d.evidence.expires.is_none_or(|expiry| now <= expiry));
        }
        let mut outputs = Vec::new();
        for (&shard, driver) in &mut self.drivers {
            if !driver.in_flight {
                driver.in_flight = true;
                outputs.push(SettledTxsAcquisitionOutput::Fetch {
                    shard,
                    peer: driver.peer(),
                    request: driver.request(),
                });
            }
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hyperscale_storage::test_helpers::make_test_certified;
    use hyperscale_storage::{PendingChain, ShardChainWriter};
    use hyperscale_storage_memory::SimShardStorage;
    use hyperscale_types::{
        AggregateSignature, BeaconWitnessCommit, BeaconWitnessLeafCount, Block, BlockHash,
        BlockHeader, BlockHeaderParts, BlockHeight, CertificateRoot, ExecutionCertificate,
        ExecutionOutcome, Finalization, GlobalReceiptHash, GlobalReceiptRoot, Hash,
        ProposerTimestamp, QuorumCertificate, Round, SettledTxsRoot, ShardId, SignerBitfield,
        TickHalf, TickId, TxHash, TxOutcome, ValidatorId, Verifiable, Verified, WeightedTimestamp,
        WitnessSources, settled_txs_root_from_hashes,
    };

    use super::*;
    use crate::shard::cross_shard::settled_txs_serve::serve_settled_txs_request;

    const SHARD: ShardId = ShardId::ROOT;

    /// The transaction the tick at `height` settles — distinct per tick,
    /// so a window over several ticks has one entry each.
    fn settled_tx(height: u64) -> TxHash {
        TxHash::from(Hash::from_bytes(&height.to_le_bytes()))
    }

    fn finalization(height: u64) -> Arc<Verifiable<Finalization>> {
        let tick = local_tick(height);
        let ec = ExecutionCertificate::new(
            tick,
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
        Arc::new(Verifiable::from(Finalization::new(
            tick,
            TickHalf::Determined,
            vec![Arc::new(ec), Arc::new(remote)],
            vec![],
        )))
    }

    /// Commit `count` blocks (1..=count), each carrying its own settled
    /// tick, and return the storage, the terminal hash, and the attested
    /// settled root over the whole window.
    fn served_chain(count: u64) -> (Arc<SimShardStorage>, BlockHash, SettledTxsRoot) {
        let storage = Arc::new(SimShardStorage::default());
        let mut parent = BlockHash::ZERO;
        let mut terminal = BlockHash::ZERO;
        for h in 1..=count {
            let certs = [finalization(h)];
            let parent_qc = QuorumCertificate::new(
                parent,
                SHARD,
                BlockHeight::new(h.saturating_sub(1)),
                BlockHash::ZERO,
                Round::INITIAL,
                SignerBitfield::new(4),
                AggregateSignature::new([0u8; 96]),
                WeightedTimestamp::from_millis(1_000 * h),
            );
            let header = BlockHeader::new(BlockHeaderParts {
                shard_id: SHARD,
                height: BlockHeight::new(h),
                parent_block_hash: parent,
                parent_qc: parent_qc.into(),
                timestamp: ProposerTimestamp::from_millis(1_000 * h),
                certificate_root: *Verified::<CertificateRoot>::compute(&certs).as_ref(),
                provision_tx_roots: std::collections::BTreeMap::new(),
                ..Default::default()
            });
            let block = Block::Live {
                header,
                transactions: Arc::new(Vec::new()),
                certificates: Arc::new(certs.to_vec()),
                provisions: Arc::new(Vec::new()),
                abandonment_records: Arc::new(Vec::new()),
                state_proofs: Arc::new(Vec::new()),
                witness_sources: Arc::new(WitnessSources::empty()),
            };
            parent = block.hash();
            terminal = block.hash();
            storage.commit_block(
                &make_test_certified(block),
                &[],
                &[],
                &BeaconWitnessCommit::empty(BeaconWitnessLeafCount::ZERO),
            );
        }
        let root =
            settled_txs_root_from_hashes((1..=count).map(settled_tx).collect::<Vec<_>>().iter());
        (storage, terminal, root)
    }

    /// A cross-shard tick (non-empty `remote_shards`): the settled set
    /// commits only cross-shard txs, so a single-shard fixture would be
    /// filtered out before the merkle root.
    fn local_tick(height: u64) -> TickId {
        TickId::new(SHARD, BlockHeight::new(height))
    }

    /// One verified fetch against a served chain completes with the whole
    /// window list, and the driver drops.
    #[test]
    fn acquires_against_a_served_chain() {
        let (storage, terminal, root) = served_chain(3);
        let pending_chain = PendingChain::new(storage);

        let mut host = SettledTxsAcquisition::new();
        let mut outputs = host.start(
            SHARD,
            TerminalEvidence {
                height: BlockHeight::new(3),
                block_hash: terminal,
                terminal_wt: WeightedTimestamp::from_millis(9_000),
                attested_root: root,
                expires: None,
            },
            vec![ValidatorId::new(7)],
        );

        let mut completed = None;
        while let Some(output) = outputs.pop() {
            match output {
                SettledTxsAcquisitionOutput::Fetch { shard, request, .. } => {
                    assert_eq!(shard, SHARD);
                    let response = serve_settled_txs_request(&pending_chain, None, &request);
                    outputs.extend(host.on_response(SHARD, &response));
                }
                SettledTxsAcquisitionOutput::Complete {
                    shard,
                    txs,
                    terminal_wt,
                } => {
                    assert_eq!(shard, SHARD);
                    assert_eq!(terminal_wt, WeightedTimestamp::from_millis(9_000));
                    completed = Some(txs);
                }
            }
        }

        assert_eq!(
            completed.expect("acquisition completes"),
            BTreeSet::from([settled_tx(1), settled_tx(2), settled_tx(3)]),
        );
        assert!(!host.has_pending(), "the driver drops on completion");
    }

    /// A list that doesn't recompute to the attested root is rejected: the
    /// peer rotates and the acquisition parks rather than recording a
    /// forged set.
    #[test]
    fn root_mismatch_parks_and_rotates() {
        let (storage, terminal, _) = served_chain(3);
        let pending_chain = PendingChain::new(storage);

        let mut host = SettledTxsAcquisition::new();
        // Attest a root the served chain cannot satisfy.
        let wrong_root = settled_txs_root_from_hashes([&settled_tx(99)]);
        let _ = host.start(
            SHARD,
            TerminalEvidence {
                height: BlockHeight::new(3),
                block_hash: terminal,
                terminal_wt: WeightedTimestamp::from_millis(9_000),
                attested_root: wrong_root,
                expires: None,
            },
            vec![ValidatorId::new(0), ValidatorId::new(1)],
        );
        let request = GetSettledTxsRequest::new(BlockHeight::new(3), terminal);
        let response = serve_settled_txs_request(&pending_chain, None, &request);
        let parked = host.on_response(SHARD, &response);
        assert!(parked.is_empty(), "a mismatch parks rather than completes");
        assert!(host.has_pending());

        // The tick re-arms the fetch against the rotated peer.
        let ticked = host.on_tick(Some(WeightedTimestamp::from_millis(9_100)));
        assert!(matches!(
            ticked.as_slice(),
            [SettledTxsAcquisitionOutput::Fetch { .. }]
        ));
    }

    /// A driver whose stamped evidence window has passed drops on tick;
    /// one whose window is still open (no handoff stamp yet) is held.
    #[test]
    fn expires_past_the_evidence_window() {
        let mut host = SettledTxsAcquisition::new();
        let _ = host.start(
            SHARD,
            TerminalEvidence {
                height: BlockHeight::new(2),
                block_hash: BlockHash::ZERO,
                terminal_wt: WeightedTimestamp::from_millis(1_000),
                attested_root: settled_txs_root_from_hashes(std::iter::empty()),
                expires: Some(WeightedTimestamp::from_millis(6_000)),
            },
            vec![],
        );
        assert!(host.has_pending());

        let _ = host.on_tick(Some(WeightedTimestamp::from_millis(6_000)));
        assert!(host.has_pending(), "at the boundary the driver still runs");

        let outputs = host.on_tick(Some(WeightedTimestamp::from_millis(6_001)));
        assert!(outputs.is_empty());
        assert!(!host.has_pending(), "the expired driver drops");
    }

    /// A driver with no stamp yet never self-expires — the window is open
    /// until the beacon stamps the handoff complete.
    #[test]
    fn an_open_window_never_expires() {
        let mut host = SettledTxsAcquisition::new();
        let _ = host.start(
            SHARD,
            TerminalEvidence {
                height: BlockHeight::new(2),
                block_hash: BlockHash::ZERO,
                terminal_wt: WeightedTimestamp::from_millis(1_000),
                attested_root: settled_txs_root_from_hashes(std::iter::empty()),
                expires: None,
            },
            vec![],
        );
        let _ = host.on_tick(Some(WeightedTimestamp::from_millis(1_000_000)));
        assert!(host.has_pending(), "an open window holds the driver");
    }

    /// A duplicate start for the same terminal while a fetch is in flight
    /// is a no-op; a start for a different terminal replaces the driver.
    #[test]
    fn dedupes_by_terminal_block() {
        let mut host = SettledTxsAcquisition::new();
        let root = settled_txs_root_from_hashes(std::iter::empty());
        let _ = host.start(
            SHARD,
            TerminalEvidence {
                height: BlockHeight::new(2),
                block_hash: BlockHash::from_raw(Hash::from_bytes(b"terminal-a")),
                terminal_wt: WeightedTimestamp::from_millis(1),
                attested_root: root,
                expires: None,
            },
            vec![],
        );
        let dup = host.start(
            SHARD,
            TerminalEvidence {
                height: BlockHeight::new(2),
                block_hash: BlockHash::from_raw(Hash::from_bytes(b"terminal-a")),
                terminal_wt: WeightedTimestamp::from_millis(1),
                attested_root: root,
                expires: None,
            },
            vec![],
        );
        assert!(dup.is_empty(), "same terminal in flight does not re-fetch");

        let replaced = host.start(
            SHARD,
            TerminalEvidence {
                height: BlockHeight::new(3),
                block_hash: BlockHash::from_raw(Hash::from_bytes(b"terminal-b")),
                terminal_wt: WeightedTimestamp::from_millis(1),
                attested_root: root,
                expires: None,
            },
            vec![],
        );
        assert!(
            matches!(
                replaced.as_slice(),
                [SettledTxsAcquisitionOutput::Fetch { request, .. }]
                    if request.terminal_height == BlockHeight::new(3)
            ),
            "a revised terminal restarts the acquisition from the new block",
        );
    }
}
