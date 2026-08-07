//! Pure execution functions invoked from the node's delegated-action dispatcher.
//!
//! These functions implement the asynchronous side of the execution
//! state machine: signature verification, execution-vote aggregation into
//! [`ExecutionCertificate`]s, transaction execution against a
//! [`SubstateView`], and cross-shard provisioning requests. They are
//! kept free of node/runner concerns so the dispatcher only handles
//! event plumbing — sharing the handlers between production and
//! simulation keeps execution behavior identical across both backends.

use std::sync::Arc;

use hyperscale_core::{
    Action, ActionContext, ProtocolEvent, TickExecutionGroup, WaveExecutionResult,
};
use hyperscale_engine::{ExecutedTx, TickTxInput, WaveBatchContext};
use hyperscale_metrics::record_execution_latency;
use hyperscale_network::Network;
use hyperscale_storage::{ProvisionalTx, ShardStorage, TickOutput};
use hyperscale_types::network::notification::{
    ExecutionCertificatesNotification, ExecutionVotesNotification,
};
use hyperscale_types::{
    BlockHeight, ExecutionCertificate, ExecutionCertificateContext,
    ExecutionCertificatesSenderMessage, ExecutionVote, ExecutionVotesSenderMessage,
    FinalizedWaveContext, Stopwatch, StoredReceipt, TxHash, TxOutcome, Verifiable, Verified,
    signed_bytes,
};

// ============================================================================
// Wave-based execution voting handlers
// ============================================================================

/// Split a batch's executed records into the three parallel streams the
/// wave consumes: outcomes for the vote, execution receipts, and the fee
/// receipts held in reserve against an abort.
fn split_execution_outputs(executed: Vec<ExecutedTx>) -> ExecutionOutputs {
    let mut outcomes = Vec::with_capacity(executed.len());
    let mut results = Vec::with_capacity(executed.len());
    let mut fee_receipts = Vec::new();
    let mut work = Vec::with_capacity(executed.len());
    for mut tx in executed {
        outcomes.push(tx.outcome());
        work.push((tx.tx_hash, tx.attested_work));
        if let Some(fee) = tx.fee_receipt.take() {
            fee_receipts.push(StoredReceipt::synced(tx.tx_hash, Arc::new(fee)));
        }
        results.push(StoredReceipt::from(tx));
    }
    ExecutionOutputs {
        outcomes,
        results,
        fee_receipts,
        attested_work: work,
    }
}

/// The four per-batch products execution hands the wave: the outcomes it
/// votes, the receipts it stores, the charges an attempt that applied
/// nothing still settles, and what this shard attests it did.
struct ExecutionOutputs {
    outcomes: Vec<TxOutcome>,
    results: Vec<StoredReceipt>,
    fee_receipts: Vec<StoredReceipt>,
    attested_work: Vec<(TxHash, u64)>,
}

/// Fold one wave group's executed records into the tick output.
///
/// The single-shard wave's writes are determined at commit: execution
/// receipts and their unconditional fee charges fold into the readable
/// map, in canonical (tx-hash) order — the order the batch fold computed
/// their absolutes in. A cross-shard wave's records sit beside the fold
/// as per-tx provisional entries until the wave resolves: the execution
/// writes on one side, the reserve fee charge on the other, whichever
/// the wave's verdict picks.
fn accumulate_tick_output(
    output: &mut TickOutput,
    group: &TickExecutionGroup,
    executed: &[ExecutedTx],
) {
    let mut ordered: Vec<&ExecutedTx> = executed.iter().collect();
    ordered.sort_by_key(|tx| tx.tx_hash);
    if group.wave_id.is_zero() {
        output.determined_wave = Some(group.wave_id.clone());
        for tx in ordered {
            if let Some(writes) = tx.consensus.writes() {
                for (key, change) in &writes.cells {
                    output.determined.cells.insert(*key, change.clone());
                }
            }
            if let Some(fee) = tx.fee_receipt.as_ref().and_then(|fee| fee.writes()) {
                for (key, change) in &fee.cells {
                    output.determined.cells.insert(*key, change.clone());
                }
            }
        }
    } else {
        let entries: Vec<ProvisionalTx> = ordered
            .into_iter()
            .map(|tx| ProvisionalTx {
                tx_hash: tx.tx_hash,
                writes: tx.consensus.writes().cloned(),
                reserve: tx
                    .fee_receipt
                    .as_ref()
                    .and_then(|fee| fee.writes())
                    .cloned(),
            })
            .collect();
        output.provisional.insert(group.wave_id.clone(), entries);
    }
}

/// Outcomes flow through `ctx.notify`. Variants owned by other coordinator
/// crates hit `unreachable!()` — node's dispatcher routes by variant prefix.
///
/// # Panics
///
/// Panics if the dispatcher routes a variant owned by another crate, or if
/// the executor breaks its "one result per input transaction" contract.
#[allow(clippy::too_many_lines)] // single dispatch over execution-owned Action variants
pub fn handle_action<S, N>(action: Action, ctx: &ActionContext<'_, S, N>)
where
    S: ShardStorage,
    N: Network,
{
    match action {
        Action::AggregateExecutionCertificate {
            wave_id,
            global_receipt_root,
            votes,
            committee,
        } => {
            let certificate = Verified::<ExecutionCertificate>::aggregate(
                ctx.verifier,
                &wave_id,
                global_receipt_root,
                &votes,
                &committee,
            );
            ctx.notify_protocol(ProtocolEvent::ExecutionCertificateAggregated {
                wave_id,
                certificate: Arc::new(certificate),
            });
        }
        Action::VerifyAndAggregateExecutionVotes {
            wave_id,
            block_hash,
            votes,
        } => {
            let verified_votes = Verified::<ExecutionVote>::verify_batch(
                ctx.verifier,
                ctx.topology_snapshot.network(),
                votes,
            );
            ctx.notify_protocol(ProtocolEvent::ExecutionVotesVerifiedAndAggregated {
                wave_id,
                block_hash,
                verified_votes,
            });
        }
        Action::VerifyExecutionCertificateSignature {
            certificate,
            public_keys,
            ..
        } => {
            let ctx_ec = ExecutionCertificateContext {
                verifier: ctx.verifier,
                network: ctx.topology_snapshot.network(),
                public_keys: &public_keys,
            };
            let result = certificate
                .upgrade(&ctx_ec)
                .map(Arc::new)
                .map_err(|(raw, err)| (Arc::new(raw), err));
            ctx.notify_protocol(ProtocolEvent::ExecutionCertificateSignatureVerified { result });
        }
        Action::VerifyFinalizedWave {
            wave,
            ec_public_keys,
        } => {
            let fw_ctx = FinalizedWaveContext {
                verifier: ctx.verifier,
                network: ctx.topology_snapshot.network(),
                ec_public_keys: &ec_public_keys,
            };
            let result = Arc::unwrap_or_clone(wave)
                .upgrade(&fw_ctx)
                .map(Arc::new)
                .map_err(|(raw, err)| (Arc::new(raw), err));
            ctx.notify_protocol(ProtocolEvent::FinalizedWaveVerified { result });
        }
        Action::ExecuteTransactions {
            tick,
            block_hash,
            tick_ts,
            tick_reveal,
            groups,
        } => {
            let start = Stopwatch::start();
            let shard_trie = ctx.topology_snapshot.shard_trie();
            // The previous tick's output over the persisted base is this
            // tick's baseline. Ticks dispatch serially, so every tick at
            // or below `tick - 1` has been appended by now.
            let view = ctx
                .tick_chain
                .view_at(BlockHeight::new(tick.inner().saturating_sub(1)));
            let view_snap = view.snapshot();
            let wave_ctx = WaveBatchContext {
                par: ctx.par,
                local_shard: ctx.shard,
                shard_trie,
                block_hash,
                wave_start_ts: tick_ts,
                wave_start_reveal: tick_reveal,
            };
            let inputs: Vec<TickTxInput<'_>> = groups
                .iter()
                .flat_map(|group| {
                    let wave_abortable = !group.wave_id.is_zero();
                    group.requests.iter().map(move |r| TickTxInput {
                        transaction: &r.transaction,
                        provisions: &r.provisions,
                        clock: r.clock,
                        randomness: r.randomness,
                        wave_abortable,
                    })
                })
                .collect();
            let executed = ctx
                .executor
                .execute_tick_batch(&wave_ctx, &view_snap, &inputs);
            record_execution_latency(start.elapsed().as_secs_f64());

            // Fan the flat result vector (input order) back to its groups,
            // building the tick output alongside the per-wave results.
            let mut remaining = executed;
            let mut output = TickOutput::default();
            let mut waves: Vec<WaveExecutionResult> = Vec::with_capacity(groups.len());
            for group in &groups {
                let rest = remaining.split_off(group.requests.len());
                let executed_group = std::mem::replace(&mut remaining, rest);
                accumulate_tick_output(&mut output, group, &executed_group);
                let ExecutionOutputs {
                    outcomes: tx_outcomes,
                    results,
                    fee_receipts,
                    attested_work,
                } = split_execution_outputs(executed_group);
                waves.push(WaveExecutionResult {
                    wave_id: group.wave_id.clone(),
                    results,
                    tx_outcomes,
                    fee_receipts,
                    attested_work,
                });
            }

            // Append before notifying: the coordinator dispatches the next
            // tick on this event, and its baseline must include this one.
            ctx.tick_chain.append(tick, output);
            ctx.notify_protocol(ProtocolEvent::ExecutionBatchCompleted { tick, waves });
        }

        // ── Sign + broadcast actions ──────────────────────────────────────
        Action::SignAndSendExecutionVote {
            block_hash,
            block_height,
            vote_anchor_ts,
            wave_id,
            global_receipt_root: _,
            tx_outcomes,
            leader,
        } => {
            let local_shard = ctx.shard;
            let validator_id = ctx.me;
            let network = ctx.topology_snapshot.network();

            let Ok(verified) = Verified::<ExecutionVote>::sign_local(
                network,
                block_hash,
                block_height,
                vote_anchor_ts,
                wave_id,
                local_shard,
                tx_outcomes,
                validator_id,
                ctx.signer.as_ref(),
            ) else {
                tracing::error!(?block_hash, "cannot sign execution vote; abstaining");
                return;
            };

            // Send vote to the wave leader (unicast). When the leader is a
            // colocated vnode the local-dispatch fast path preserves the
            // `Verifiable::Verified` marker, letting the handler skip
            // re-verification of our own signature.
            if leader != validator_id {
                let batch_msg = signed_bytes(
                    &ExecutionVotesSenderMessage::new(local_shard, std::iter::once(&*verified)),
                    network,
                );
                let Ok(batch_sig) = ctx.signer.sign(&batch_msg) else {
                    tracing::error!(
                        ?block_hash,
                        "cannot sign execution vote batch; skipping send"
                    );
                    return;
                };
                let batch = ExecutionVotesNotification::new(
                    vec![Verifiable::from(verified.clone())],
                    validator_id,
                    batch_sig,
                );
                ctx.network.notify(&[leader], &batch);
            }

            // Feed own vote to state machine only if we are the leader.
            if leader == validator_id {
                ctx.notify_protocol(ProtocolEvent::VerifiedExecutionVoteReceived {
                    vote: verified,
                });
            }
        }

        Action::BroadcastExecutionCertificate {
            shard: _,
            certificate,
            recipients,
        } => {
            let cert = Arc::unwrap_or_clone(certificate).into_inner();
            let msg = signed_bytes(
                &ExecutionCertificatesSenderMessage::new(
                    cert.shard_id(),
                    std::slice::from_ref(&cert),
                ),
                ctx.topology_snapshot.network(),
            );
            let Ok(sig) = ctx.signer.sign(&msg) else {
                tracing::error!("cannot sign execution certificate batch; skipping broadcast");
                return;
            };
            let batch = ExecutionCertificatesNotification::new(vec![cert], ctx.me, sig);
            ctx.network.notify(&recipients, &batch);
        }

        _ => unreachable!("hyperscale_execution::handle_action called with non-execution action"),
    }
}
