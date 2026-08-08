//! Settled-txs acquisition I/O glue.
//!
//! Bridges
//! [`SettledTxsAcquisition`](super::settled_set::SettledTxsAcquisition)'s
//! scheduling to the network and to the state machine. It owns the
//! one-shot fetch-and-verify; this layer turns its
//! [`SettledTxsAcquisitionOutput`]s into `GetSettledTxsRequest`
//! fetches and the verified `Complete` into a `SettledTxsReconstructed`
//! event for the fence.

use hyperscale_core::ProtocolEvent;
use hyperscale_dispatch::Dispatch;
use hyperscale_network::{Network, ResponseVerdict};
use hyperscale_storage::ShardStorage;
use hyperscale_types::network::response::GetSettledTxsResponse;
use hyperscale_types::{
    BlockHash, BlockHeight, SettledTxsRoot, ShardId, TxHash, ValidatorId, WeightedTimestamp,
};

use super::settled_set::SettledTxsAcquisitionOutput;
use crate::shard::{ShardLoop, ShardScopedInput, push_shard_input};

impl<S, N, D> ShardLoop<S, N, D>
where
    S: ShardStorage,
    N: Network,
    D: Dispatch,
{
    // ─── Action dispatch ────────────────────────────────────────────────

    /// Handle `Action::StartSettledTxsAcquisition`: begin (or retry) a
    /// terminated shard's settled-set acquisition and dispatch the
    /// window fetch.
    pub(crate) fn process_start_settled_txs_acquisition(
        &mut self,
        shard: ShardId,
        terminal_height: BlockHeight,
        terminal_block_hash: BlockHash,
        terminal_wt: WeightedTimestamp,
        attested_root: SettledTxsRoot,
        peers: Vec<ValidatorId>,
    ) {
        let outputs = self.io.cross_shard.settled_set_sync.start(
            shard,
            terminal_height,
            terminal_block_hash,
            terminal_wt,
            attested_root,
            peers,
        );
        self.process_settled_txs_acquisition_outputs(outputs);
    }

    // ─── step() handlers ────────────────────────────────────────────────

    /// Network callback: a settled-set window list arrived for
    /// `source_shard` (`None` when the peer didn't hold the terminal).
    pub(crate) fn handle_settled_txs_response_received(
        &mut self,
        source_shard: ShardId,
        txs: Option<Vec<TxHash>>,
    ) {
        let response = GetSettledTxsResponse { txs };
        let outputs = self
            .io
            .cross_shard
            .settled_set_sync
            .on_response(source_shard, &response);
        self.process_settled_txs_acquisition_outputs(outputs);
    }

    /// Network callback: a settled-set fetch failed at the transport
    /// level. The host re-arms and the next `FetchTick` retries.
    pub(crate) fn handle_settled_txs_fetch_failed(&mut self, source_shard: ShardId) {
        self.io
            .cross_shard
            .settled_set_sync
            .on_failure(source_shard);
    }

    /// Drop expired acquisitions and re-issue every parked one on the
    /// periodic tick. The node's current chain weighted timestamp bounds
    /// the self-expiry.
    pub(crate) fn settled_set_tick(&mut self) {
        let now_wt = self
            .io
            .pending_chain
            .latest_qc()
            .map(|qc| qc.weighted_timestamp());
        let outputs = self.io.cross_shard.settled_set_sync.on_tick(now_wt);
        self.process_settled_txs_acquisition_outputs(outputs);
    }

    // ─── Output processing ──────────────────────────────────────────────

    /// Route host outputs: `Fetch` → network request, `Complete` →
    /// `SettledTxsReconstructed` event for the fence.
    fn process_settled_txs_acquisition_outputs(
        &mut self,
        outputs: Vec<SettledTxsAcquisitionOutput>,
    ) {
        let local_shard = self.shard;
        for output in outputs {
            match output {
                SettledTxsAcquisitionOutput::Fetch {
                    shard,
                    peer,
                    request,
                } => {
                    let es = self.event_sender().clone();
                    self.process.network.request(
                        shard,
                        peer,
                        request,
                        None,
                        Box::new(move |result: Result<GetSettledTxsResponse, _>| {
                            match result {
                                Ok(response) => push_shard_input(
                                    &es,
                                    local_shard,
                                    ShardScopedInput::SettledTxsResponseReceived {
                                        source_shard: shard,
                                        txs: response.txs,
                                    },
                                ),
                                Err(_) => push_shard_input(
                                    &es,
                                    local_shard,
                                    ShardScopedInput::SettledTxsFetchFailed {
                                        source_shard: shard,
                                    },
                                ),
                            }
                            ResponseVerdict::Accept
                        }),
                    );
                }
                SettledTxsAcquisitionOutput::Complete {
                    shard,
                    txs,
                    terminal_wt,
                } => {
                    self.dispatch_event(ProtocolEvent::SettledTxsReconstructed {
                        shard,
                        txs,
                        terminal_wt,
                    });
                }
            }
        }
    }
}
