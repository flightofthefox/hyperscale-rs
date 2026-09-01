//! Block-proposal helpers used by the shard consensus-driven dispatch arms.
//!
//! Both the post-dispatch proposal-retry hook and the QC-formed path build
//! proposals from the same triple — ready txs from mempool, finalizations
//! from execution, queued provisions — so the gather logic lives once here.

use std::sync::Arc;

use hyperscale_core::Action;
use hyperscale_types::{
    AbandonmentRecord, Finalization, MAX_TXS_PER_BLOCK, Provisions, TopologySchedule,
    TopologySnapshot, Transaction, Verifiable, Verified,
};

use super::ShardParticipation;

/// Inputs gathered for building a block proposal.
pub(in crate::state) struct ProposalInputs {
    pub ready_txs: Vec<Arc<Verified<Transaction>>>,
    pub finalizations: Vec<Arc<Verifiable<Finalization>>>,
    pub provisions: Vec<Arc<Verifiable<Provisions>>>,
    pub abandonment_records: Vec<AbandonmentRecord>,
}

impl ShardParticipation {
    /// Gather all inputs needed for a block proposal.
    ///
    /// Used by both `on_proposal_timer` and `on_qc_formed` to avoid duplicating
    /// the ready-transaction + abort intents + certificates gathering logic.
    pub(in crate::state) fn gather_proposal_inputs(
        &self,
        sched: &TopologySchedule,
    ) -> ProposalInputs {
        // The wire cap, not the packing bound — a block cannot encode
        // more than this however light its transactions are. What decides
        // how many are actually offered is the work budget inside
        // `ready_transactions`. The overhead compensates for QC-chain
        // duplicates shard consensus filters during proposal building.
        let parent = self.shard_coordinator.proposal_parent_block_hash();
        // The transactions the ancestor chain above the committed tip
        // already carries. Proposal building drops them as duplicates, so
        // the budget has to be raised by what will be dropped or the
        // block comes out short.
        let (ancestor_txs, _) = self.shard_coordinator.collect_qc_chain_hashes(parent);
        let max_txs = MAX_TXS_PER_BLOCK + ancestor_txs.len();
        // The budget reads the chain, not a local claim set: the parent
        // header carries what this shard still owes in work.
        let in_flight = self.shard_coordinator.proposal_parent_in_flight();
        let ready_txs =
            self.mempool_coordinator
                .ready_transactions(max_txs, in_flight.inner(), self.now);
        let finalizations = self.execution_coordinator.get_finalizations();
        // What departed counterparts left of this chain's business, while
        // the settled sets that say so can still be read.
        let abandonment_records = self.execution_coordinator.pending_abandonment_records();
        let queued = self.provisions_coordinator.queued_provisions(self.now);

        // The engagement gate: a non-payer shard proposes a cross-shard
        // transaction only beside its payer bundle — this proposal's
        // own provisions — or after an earlier block absorbed it. The
        // bundle is the transaction commit proof (verified against a
        // commit-proven payer header), so locks engage only on committed
        // payer evidence; a mis-paired inclusion is backstopped by the
        // dispatch gate's required-set check.
        let topology = sched.head();
        let ready_txs = ready_txs
            .into_iter()
            .filter(|tx| self.engagement_held(tx, topology, &queued))
            .collect();

        // Provisions coordinator stores `Verified` internally; lift each
        // batch into the `Verifiable` transport shape so the marker
        // survives across the proposal-build action.
        let provisions = queued
            .into_iter()
            .map(|v| Arc::new((*v).clone().into()))
            .collect();

        ProposalInputs {
            ready_txs,
            finalizations,
            provisions,
            abandonment_records,
        }
    }

    /// Whether the engagement evidence for `tx` is in hand: not a
    /// transaction, single-shard, our shard is the payer's, the payer's
    /// bundle rides in `queued`, or an earlier block already absorbed it.
    fn engagement_held(
        &self,
        tx: &Arc<Verified<Transaction>>,
        topology: &TopologySnapshot,
        queued: &[Arc<Verified<Provisions>>],
    ) -> bool {
        if topology.is_single_shard_transaction(tx.as_ref()) {
            return true;
        }
        let payer_shard = topology.shard_trie().shard_for_prefix(tx.body().fee_payer);
        if payer_shard == self.local_shard {
            return true;
        }
        let tx_hash = tx.hash();
        self.execution_coordinator
            .has_provisions_from(tx_hash, payer_shard)
            || queued.iter().any(|bundle| {
                bundle.source_shard() == payer_shard
                    && bundle
                        .transactions()
                        .iter()
                        .any(|entry| entry.tx_hash == tx_hash)
            })
    }

    /// Shared proposal logic for the post-dispatch retry hook and the
    /// QC-formed path.
    pub(in crate::state) fn try_event_driven_proposal(
        &mut self,
        sched: &TopologySchedule,
    ) -> Vec<Action> {
        let inputs = self.gather_proposal_inputs(sched);

        self.shard_coordinator.try_propose(
            sched,
            &inputs.ready_txs,
            inputs.finalizations,
            inputs.provisions,
            inputs.abandonment_records,
        )
    }
}
