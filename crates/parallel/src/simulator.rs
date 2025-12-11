//! Parallel simulator using rayon for CPU parallelism.
//!
//! Processes nodes in a step-based loop, using rayon to parallelize
//! event processing across CPU cores. This gives us:
//! - Deterministic simulated time (we control when time advances)
//! - CPU parallelism (rayon processes nodes on multiple cores)
//! - No scheduling issues (no tokio task coordination needed)

use crate::config::ParallelConfig;
use crate::metrics::SimulationReport;
use crate::router::Destination;
use hyperscale_bft::{BftConfig, RecoveredState};
use hyperscale_core::{Action, Event, OutboundMessage, StateMachine, TimerId};
use hyperscale_node::NodeStateMachine;
use hyperscale_simulation::SimStorage;
use hyperscale_types::{
    Block, BlockHeader, BlockHeight, Hash, KeyPair, PartitionNumber, PublicKey, QuorumCertificate,
    RoutableTransaction, ShardGroupId, Signature, StaticTopology, Topology, TransactionDecision,
    TransactionStatus, ValidatorId, ValidatorInfo, ValidatorSet,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info};

/// A simulated node in the parallel simulator.
pub struct SimNode {
    index: u32,
    state: NodeStateMachine,
    storage: SimStorage,
    /// Pending inbound messages from other nodes.
    inbound_queue: VecDeque<Arc<OutboundMessage>>,
    /// Pending internal events (timer fires, crypto results, etc.)
    internal_queue: VecDeque<Event>,
    /// Pending transaction submissions.
    tx_queue: VecDeque<Event>,
    /// Outbound messages to be routed after processing.
    outbound_messages: Vec<(Destination, Arc<OutboundMessage>)>,
    /// Transaction status updates to report.
    status_updates: Vec<(Hash, TransactionStatus)>,
    /// Pending timers: timer_id -> fire_time
    pending_timers: HashMap<TimerId, Duration>,
    /// Simulated time for this node.
    simulated_time: Duration,
}

impl SimNode {
    /// Create a new simulated node.
    pub fn new(index: u32, state: NodeStateMachine, storage: SimStorage) -> Self {
        Self {
            index,
            state,
            storage,
            inbound_queue: VecDeque::new(),
            internal_queue: VecDeque::new(),
            tx_queue: VecDeque::new(),
            outbound_messages: Vec::new(),
            status_updates: Vec::new(),
            pending_timers: HashMap::new(),
            simulated_time: Duration::ZERO,
        }
    }

    /// Submit a transaction to this node.
    pub fn submit_transaction(&mut self, event: Event) {
        self.tx_queue.push_back(event);
    }

    /// Deliver an inbound message to this node.
    pub fn deliver_message(&mut self, message: Arc<OutboundMessage>) {
        self.inbound_queue.push_back(message);
    }

    /// Process all pending events until the queues are empty.
    /// Returns the number of events processed.
    pub fn drain_events(&mut self) -> usize {
        let mut total_processed = 0;

        loop {
            let mut processed = 0;

            // Process internal events first (highest priority)
            while let Some(event) = self.internal_queue.pop_front() {
                self.handle_event(event);
                processed += 1;
            }

            // Process inbound messages
            while let Some(msg) = self.inbound_queue.pop_front() {
                let event = msg.to_received_event();
                self.handle_event(event);
                processed += 1;
            }

            // Process transaction submissions
            while let Some(event) = self.tx_queue.pop_front() {
                self.handle_event(event);
                processed += 1;
            }

            total_processed += processed;

            // If nothing was processed, we're done
            if processed == 0 {
                break;
            }
        }

        total_processed
    }

    /// Take outbound messages (clears the buffer).
    pub fn take_outbound_messages(&mut self) -> Vec<(Destination, Arc<OutboundMessage>)> {
        std::mem::take(&mut self.outbound_messages)
    }

    /// Take status updates (clears the buffer).
    pub fn take_status_updates(&mut self) -> Vec<(Hash, TransactionStatus)> {
        std::mem::take(&mut self.status_updates)
    }

    /// Advance simulated time and fire any due timers.
    pub fn advance_time(&mut self, new_time: Duration) {
        self.simulated_time = new_time;
        self.state.set_time(new_time);

        // Check for timers that should fire
        let due_timers: Vec<TimerId> = self
            .pending_timers
            .iter()
            .filter(|(_, fire_time)| **fire_time <= new_time)
            .map(|(id, _)| *id)
            .collect();

        for timer_id in due_timers {
            self.pending_timers.remove(&timer_id);
            let event = match timer_id {
                TimerId::Proposal => Event::ProposalTimer,
                TimerId::ViewChange => Event::ViewChangeTimer,
                TimerId::Cleanup => Event::CleanupTimer,
            };
            self.internal_queue.push_back(event);
        }
    }

    /// Handle a single event.
    fn handle_event(&mut self, event: Event) {
        // Process through state machine (time is set by advance_time)
        let actions = self.state.handle(event);

        // Execute actions
        for action in actions {
            self.execute_action(action);
        }
    }

    /// Execute an action from the state machine.
    fn execute_action(&mut self, action: Action) {
        match action {
            Action::BroadcastToShard { shard, message } => {
                self.outbound_messages
                    .push((Destination::Shard(shard), Arc::new(message)));
            }

            Action::BroadcastGlobal { message } => {
                self.outbound_messages
                    .push((Destination::Global, Arc::new(message)));
            }

            Action::SetTimer { id, duration } => {
                // Schedule timer to fire at current_time + duration
                let fire_time = self.simulated_time + duration;
                self.pending_timers.insert(id, fire_time);
            }

            Action::CancelTimer { id } => {
                self.pending_timers.remove(&id);
            }

            Action::EnqueueInternal { event } => {
                self.internal_queue.push_back(event);
            }

            Action::EmitTransactionStatus { tx_hash, status } => {
                if status.is_final() {
                    self.status_updates.push((tx_hash, status));
                }
            }

            // Signature verification - synchronous
            Action::VerifyVoteSignature {
                vote,
                public_key,
                signing_message,
            } => {
                let valid = public_key.verify(&signing_message, &vote.signature);
                self.internal_queue
                    .push_back(Event::VoteSignatureVerified { vote, valid });
            }

            Action::VerifyProvisionSignature {
                provision,
                public_key,
            } => {
                let msg = provision.signing_message();
                let valid = public_key.verify(&msg, &provision.signature);
                self.internal_queue
                    .push_back(Event::ProvisionSignatureVerified { provision, valid });
            }

            Action::VerifyStateVoteSignature { vote, public_key } => {
                let msg = vote.signing_message();
                let valid = public_key.verify(&msg, &vote.signature);
                self.internal_queue
                    .push_back(Event::StateVoteSignatureVerified { vote, valid });
            }

            Action::VerifyStateCertificateSignature {
                certificate,
                public_keys,
            } => {
                let msg = certificate.signing_message();
                let signer_keys: Vec<_> = public_keys
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| certificate.signers.is_set(*i))
                    .map(|(_, pk)| pk.clone())
                    .collect();
                let valid = Self::verify_aggregated_signature(
                    &signer_keys,
                    &msg,
                    &certificate.aggregated_signature,
                );
                self.internal_queue
                    .push_back(Event::StateCertificateSignatureVerified { certificate, valid });
            }

            Action::VerifyQcSignature {
                qc,
                public_keys,
                block_hash,
                signing_message,
            } => {
                let signer_keys: Vec<_> = public_keys
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| qc.signers.is_set(*i))
                    .map(|(_, pk)| pk.clone())
                    .collect();
                let valid = if signer_keys.is_empty() {
                    false
                } else {
                    Self::verify_aggregated_signature(
                        &signer_keys,
                        &signing_message,
                        &qc.aggregated_signature,
                    )
                };
                self.internal_queue
                    .push_back(Event::QcSignatureVerified { block_hash, valid });
            }

            Action::VerifyViewChangeVoteSignature {
                vote,
                public_key,
                signing_message,
            } => {
                let valid = public_key.verify(&signing_message, &vote.signature);
                self.internal_queue
                    .push_back(Event::ViewChangeVoteSignatureVerified { vote, valid });
            }

            Action::VerifyViewChangeHighestQc {
                vote,
                public_keys,
                signing_message,
            } => {
                let signer_keys: Vec<_> = public_keys
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| vote.highest_qc.signers.is_set(*i))
                    .map(|(_, pk)| pk.clone())
                    .collect();
                let valid = if signer_keys.is_empty() {
                    vote.highest_qc.is_genesis()
                } else {
                    Self::verify_aggregated_signature(
                        &signer_keys,
                        &signing_message,
                        &vote.highest_qc.aggregated_signature,
                    )
                };
                self.internal_queue
                    .push_back(Event::ViewChangeHighestQcVerified { vote, valid });
            }

            Action::VerifyViewChangeCertificateSignature {
                certificate,
                public_keys,
                signing_message,
            } => {
                let signer_keys: Vec<_> = public_keys
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| certificate.signers.is_set(*i))
                    .map(|(_, pk)| pk.clone())
                    .collect();
                let valid = Self::verify_aggregated_signature(
                    &signer_keys,
                    &signing_message,
                    &certificate.aggregated_signature,
                );
                self.internal_queue
                    .push_back(Event::ViewChangeCertificateSignatureVerified {
                        certificate,
                        valid,
                    });
            }

            // Transaction execution - simplified
            Action::ExecuteTransactions {
                block_hash,
                transactions,
                ..
            } => {
                let results = transactions
                    .iter()
                    .map(|tx| hyperscale_types::ExecutionResult {
                        transaction_hash: tx.hash(),
                        success: true,
                        state_root: Hash::ZERO,
                        writes: vec![],
                        error: None,
                    })
                    .collect();
                self.internal_queue.push_back(Event::TransactionsExecuted {
                    block_hash,
                    results,
                });
            }

            Action::ExecuteCrossShardTransaction { tx_hash, .. } => {
                let result = hyperscale_types::ExecutionResult {
                    transaction_hash: tx_hash,
                    success: true,
                    state_root: Hash::ZERO,
                    writes: vec![],
                    error: None,
                };
                self.internal_queue
                    .push_back(Event::CrossShardTransactionExecuted { tx_hash, result });
            }

            Action::ComputeMerkleRoot { tx_hash, .. } => {
                self.internal_queue.push_back(Event::MerkleRootComputed {
                    tx_hash,
                    root: Hash::ZERO,
                });
            }

            // Storage actions
            Action::PersistBlock { block, qc } => {
                let height = block.header.height;
                self.storage.put_block(height, block, qc);
            }

            Action::PersistTransactionCertificate { certificate } => {
                let local_shard = self.state.shard();
                if let Some((_, proof)) = certificate
                    .shard_proofs
                    .iter()
                    .find(|(shard, _)| **shard == local_shard)
                {
                    self.storage
                        .commit_certificate_with_writes(&certificate, &proof.state_writes);
                } else {
                    self.storage
                        .put_certificate(certificate.transaction_hash, certificate);
                }
            }

            Action::PersistOwnVote {
                height,
                round,
                block_hash,
            } => {
                self.storage.put_own_vote(height.0, round, block_hash);
            }

            Action::FetchStateEntries { tx_hash, nodes } => {
                use hyperscale_engine::SubstateStore;
                let entries: Vec<_> = nodes
                    .iter()
                    .flat_map(|node_id| {
                        self.storage
                            .list_substates_for_node(node_id)
                            .map(
                                |(partition, sort_key, value)| hyperscale_types::StateEntry {
                                    node_id: *node_id,
                                    partition: PartitionNumber(partition),
                                    sort_key: sort_key.0,
                                    value: Some(value),
                                },
                            )
                            .collect::<Vec<_>>()
                    })
                    .collect();
                self.internal_queue
                    .push_back(Event::StateEntriesFetched { tx_hash, entries });
            }

            Action::FetchBlock { height } => {
                let block = self.storage.get_block(height).map(|(b, _)| b);
                self.internal_queue
                    .push_back(Event::BlockFetched { height, block });
            }

            Action::FetchChainMetadata => {
                let height = self.storage.committed_height();
                let (hash, qc) = self
                    .storage
                    .get_block(height)
                    .map(|(b, q)| (Some(b.hash()), Some(q)))
                    .unwrap_or((None, None));
                self.internal_queue
                    .push_back(Event::ChainMetadataFetched { height, hash, qc });
            }

            Action::EmitCommittedBlock { block } => {
                debug!(
                    node = self.index,
                    height = block.header.height.0,
                    "Block committed"
                );
            }
        }
    }

    fn verify_aggregated_signature(
        signer_keys: &[PublicKey],
        message: &[u8],
        signature: &Signature,
    ) -> bool {
        if signer_keys.is_empty() {
            *signature == Signature::zero()
        } else {
            match PublicKey::aggregate_bls(signer_keys) {
                Ok(aggregated_pk) => aggregated_pk.verify(message, signature),
                Err(_) => false,
            }
        }
    }
}

/// Parallel simulator using rayon for multi-core CPU parallelism.
///
/// Processes nodes synchronously in a step-based loop, using rayon to
/// parallelize event processing across CPU cores. Each step:
/// 1. Processes all pending events on all nodes (in parallel)
/// 2. Collects outbound messages
/// 3. Routes messages to recipients
/// 4. Collects transaction status updates
/// 5. Advances simulated time and fires due timers
pub struct ParallelSimulator {
    config: ParallelConfig,
    nodes: Vec<SimNode>,
    /// Shard membership: shard -> list of node indices
    shard_members: HashMap<ShardGroupId, Vec<u32>>,
    /// Simulated time
    simulated_time: Duration,
    /// Metrics
    submitted: AtomicU64,
    completed: AtomicU64,
    rejected: AtomicU64,
    /// In-flight transaction hashes
    in_flight: HashSet<Hash>,
    /// Latencies in microseconds
    latencies: Vec<u64>,
    /// Submission times for latency calculation (simulated time)
    submission_times: HashMap<Hash, Duration>,
}

impl ParallelSimulator {
    /// Create a new parallel simulator.
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            shard_members: HashMap::new(),
            simulated_time: Duration::ZERO,
            submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            in_flight: HashSet::new(),
            latencies: Vec::new(),
            submission_times: HashMap::new(),
        }
    }

    /// Initialize the simulator (create nodes, run genesis).
    pub fn initialize(&mut self) {
        let num_shards = self.config.num_shards;
        let validators_per_shard = self.config.validators_per_shard as u32;
        let total_nodes = num_shards * self.config.validators_per_shard;

        info!(
            num_shards,
            validators_per_shard, total_nodes, "Initializing parallel simulator"
        );

        // Generate keys deterministically
        let seed = self.config.seed;
        let keys: Vec<KeyPair> = (0..total_nodes)
            .map(|i| {
                let mut seed_bytes = [0u8; 32];
                let key_seed = seed.wrapping_add(i as u64).wrapping_mul(0x517cc1b727220a95);
                seed_bytes[..8].copy_from_slice(&key_seed.to_le_bytes());
                seed_bytes[8..16].copy_from_slice(&(i as u64).to_le_bytes());
                KeyPair::from_seed(hyperscale_types::KeyType::Bls12381, &seed_bytes)
            })
            .collect();
        let public_keys: Vec<PublicKey> = keys.iter().map(|k| k.public_key()).collect();

        // Build global validator set
        let global_validators: Vec<ValidatorInfo> = (0..total_nodes)
            .map(|i| ValidatorInfo {
                validator_id: ValidatorId(i as u64),
                public_key: public_keys[i].clone(),
                voting_power: 1,
            })
            .collect();
        let global_validator_set = ValidatorSet::new(global_validators);

        // Build shard committees
        let mut shard_committees: HashMap<ShardGroupId, Vec<ValidatorId>> = HashMap::new();
        for shard_id in 0..num_shards {
            let shard = ShardGroupId(shard_id as u64);
            let shard_start = shard_id * self.config.validators_per_shard;
            let shard_end = shard_start + self.config.validators_per_shard;
            let committee: Vec<ValidatorId> = (shard_start..shard_end)
                .map(|i| ValidatorId(i as u64))
                .collect();
            shard_committees.insert(shard, committee);
            self.shard_members
                .insert(shard, (shard_start..shard_end).map(|i| i as u32).collect());
        }

        // Create nodes
        for shard_id in 0..num_shards {
            let shard = ShardGroupId(shard_id as u64);
            let shard_start = shard_id * self.config.validators_per_shard;

            for v in 0..self.config.validators_per_shard {
                let node_index = shard_start + v;
                let validator_id = ValidatorId(node_index as u64);

                let topology: Arc<dyn Topology> = Arc::new(StaticTopology::with_shard_committees(
                    validator_id,
                    shard,
                    num_shards as u64,
                    &global_validator_set,
                    shard_committees.clone(),
                ));

                let state = NodeStateMachine::new(
                    node_index as u32,
                    topology,
                    keys[node_index].clone(),
                    BftConfig::default(),
                    RecoveredState::default(),
                );

                let storage = SimStorage::new();
                let node = SimNode::new(node_index as u32, state, storage);
                self.nodes.push(node);
            }
        }

        // Initialize genesis for each shard
        for shard_id in 0..num_shards {
            let genesis_header = BlockHeader {
                height: BlockHeight(0),
                parent_hash: Hash::from_bytes(&[0u8; 32]),
                parent_qc: QuorumCertificate::genesis(),
                proposer: ValidatorId((shard_id * self.config.validators_per_shard) as u64),
                timestamp: 0,
                round: 0,
                is_fallback: false,
            };
            let genesis_block = Block {
                header: genesis_header,
                transactions: vec![],
                committed_certificates: vec![],
                deferred: vec![],
                aborted: vec![],
            };

            let shard_start = shard_id * self.config.validators_per_shard;
            let shard_end = shard_start + self.config.validators_per_shard;

            for node_index in shard_start..shard_end {
                let node = &mut self.nodes[node_index];
                let actions = node.state.initialize_genesis(genesis_block.clone());
                for action in actions {
                    node.execute_action(action);
                }
            }
        }

        info!("Genesis initialized for all shards");
    }

    /// Submit a transaction to the appropriate node.
    pub fn submit_transaction(&mut self, tx: RoutableTransaction) {
        let tx_hash = tx.hash();
        let target_shard = tx
            .declared_writes
            .first()
            .map(|w| hyperscale_types::shard_for_node(w, self.config.num_shards as u64))
            .unwrap_or(ShardGroupId(0));

        // Submit to first validator in target shard
        let node_index = (target_shard.0 as usize * self.config.validators_per_shard) as u32;
        let event = Event::SubmitTransaction { tx };
        self.nodes[node_index as usize].submit_transaction(event);

        self.submitted.fetch_add(1, Ordering::Relaxed);
        self.in_flight.insert(tx_hash);
        self.submission_times.insert(tx_hash, self.simulated_time);
    }

    /// Run one step of the simulation.
    /// Returns the number of events processed across all nodes.
    pub fn step(&mut self) -> usize {
        // Step 1: Process all nodes in parallel
        let events_processed: usize = self
            .nodes
            .par_iter_mut()
            .map(|node| node.drain_events())
            .sum();

        // Step 2: Collect outbound messages from all nodes
        let mut all_messages: Vec<(u32, Destination, Arc<OutboundMessage>)> = Vec::new();
        for (i, node) in self.nodes.iter_mut().enumerate() {
            for (dest, msg) in node.take_outbound_messages() {
                all_messages.push((i as u32, dest, msg));
            }
        }

        // Step 3: Route messages to recipients
        for (from, dest, msg) in all_messages {
            let recipients = self.get_recipients(from, &dest);
            for recipient in recipients {
                self.nodes[recipient as usize].deliver_message(msg.clone());
            }
        }

        // Step 4: Collect status updates
        for node in &mut self.nodes {
            for (tx_hash, status) in node.take_status_updates() {
                if self.in_flight.remove(&tx_hash) {
                    // Calculate latency
                    if let Some(submit_time) = self.submission_times.remove(&tx_hash) {
                        let latency = self.simulated_time.saturating_sub(submit_time);
                        self.latencies.push(latency.as_micros() as u64);
                    }

                    match status {
                        TransactionStatus::Completed(TransactionDecision::Accept) => {
                            self.completed.fetch_add(1, Ordering::Relaxed);
                        }
                        TransactionStatus::Completed(TransactionDecision::Reject)
                        | TransactionStatus::Aborted { .. } => {
                            self.rejected.fetch_add(1, Ordering::Relaxed);
                        }
                        TransactionStatus::Retried { new_tx } => {
                            // Track new hash
                            self.in_flight.insert(new_tx);
                            if let Some(submit_time) = self.submission_times.get(&tx_hash) {
                                self.submission_times.insert(new_tx, *submit_time);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Step 5: Advance simulated time and fire due timers on all nodes
        self.simulated_time += Duration::from_millis(1);
        for node in &mut self.nodes {
            node.advance_time(self.simulated_time);
        }

        events_processed
    }

    /// Get recipient node indices for a destination.
    fn get_recipients(&self, _from: u32, dest: &Destination) -> Vec<u32> {
        match dest {
            Destination::Shard(shard) => self.shard_members.get(shard).cloned().unwrap_or_default(),
            Destination::Global => (0..self.nodes.len() as u32).collect(),
            Destination::Validator(validator) => vec![validator.0 as u32],
        }
    }

    /// Get current metrics.
    pub fn metrics(&self) -> (u64, u64, u64, usize) {
        (
            self.submitted.load(Ordering::Relaxed),
            self.completed.load(Ordering::Relaxed),
            self.rejected.load(Ordering::Relaxed),
            self.in_flight.len(),
        )
    }

    /// Get simulated time.
    pub fn simulated_time(&self) -> Duration {
        self.simulated_time
    }

    /// Finalize and produce report.
    ///
    /// TPS is calculated from wall clock duration for meaningful performance metrics.
    /// Latency is in simulated time (milliseconds converted to microseconds).
    pub fn finalize(mut self, wall_clock_duration: Duration) -> SimulationReport {
        let (submitted, completed, rejected, in_flight) = self.metrics();

        // Compute latency percentiles (in simulated microseconds)
        self.latencies.sort_unstable();
        let p50 = percentile(&self.latencies, 0.50);
        let p90 = percentile(&self.latencies, 0.90);
        let p99 = percentile(&self.latencies, 0.99);
        let max = self.latencies.last().copied().unwrap_or(0);
        let avg = if self.latencies.is_empty() {
            0
        } else {
            self.latencies.iter().sum::<u64>() / self.latencies.len() as u64
        };

        // TPS based on wall clock time (meaningful performance metric)
        let tps = if wall_clock_duration.as_secs_f64() > 0.0 {
            completed as f64 / wall_clock_duration.as_secs_f64()
        } else {
            0.0
        };

        SimulationReport {
            duration: wall_clock_duration,
            submitted,
            completed,
            rejected,
            retries: 0,
            in_flight: in_flight as u64,
            messages_dropped_loss: 0,
            messages_dropped_partition: 0,
            tps,
            latency_p50_us: p50,
            latency_p90_us: p90,
            latency_p99_us: p99,
            latency_max_us: max,
            latency_avg_us: avg,
        }
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_simulator_basic() {
        let config = ParallelConfig::new(2, 3);
        let mut sim = ParallelSimulator::new(config);
        sim.initialize();

        // Run a few steps
        for _ in 0..10 {
            sim.step();
        }

        let (submitted, _completed, _rejected, in_flight) = sim.metrics();
        assert_eq!(submitted, 0);
        assert_eq!(in_flight, 0);
    }
}
