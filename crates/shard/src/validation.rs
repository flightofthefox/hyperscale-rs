//! Pure pre-vote validation helpers.
//!
//! These functions check a received block header or block contents against
//! the rules every honest validator applies before voting:
//!
//! - Header structure: proposer selection, parent-QC quorum, timestamp bounds.
//! - Block contents: transaction ordering, `ticks` recomputation, and
//!   cross-ancestor transaction uniqueness.
//!
//! Everything here is stateless — callers supply `committed_height`,
//! `qc_chain_tx_hashes`, etc. explicitly. The async verification pipeline
//! lives in [`crate::verification`]; this module is just the pure rules.
//!
//! Errors are returned as human-readable strings so the caller can log a
//! single diagnostic line at the rejection site.
use std::collections::HashSet;
use std::sync::Arc;

use hyperscale_engine::writes_committed_cell;
use hyperscale_types::{
    Block, BlockHeader, BlockHeight, FinalizationHash, LocalTimestamp, MAX_PROVISIONS_PER_BLOCK,
    MAX_ROUND_GAP, MAX_SWEEPABLE_CREATED_PER_BLOCK, MAX_TIMESTAMP_DELAY, MAX_TIMESTAMP_RUSH,
    MAX_UNSETTLED_PER_BLOCK, ProvisionHash, QuorumCertificate, ShardId, ShardLoad,
    TopologySnapshot, Transaction, TxHash, Verifiable, VoteCount, abandonment_root_from_records,
    state_proofs_root_from_bundles, sweep_admits_block,
};

use crate::commit_dedup::CommitDedupIndex;

/// True if `qc.signers()` represents at least 2f+1 of the local committee's
/// voting power. The synced-block apply path and consensus pre-vote path
/// both call this — without it, a single Byzantine signer suffices to pass
/// the signature-only `VerifyQcSignature` check that follows.
#[must_use]
pub fn qc_has_local_quorum_power(
    topology_snapshot: &TopologySnapshot,
    local_shard: ShardId,
    qc: &QuorumCertificate,
) -> bool {
    let committee = topology_snapshot.consensus_committee_for_shard(local_shard);
    let qc_power: VoteCount = qc
        .signers()
        .set_indices()
        .filter_map(|i| committee.get(i))
        .map(|&vid| {
            topology_snapshot
                .vote_of(vid)
                .expect("committee member has voting power (TopologySnapshot invariant)")
        })
        .sum();
    VoteCount::has_quorum(qc_power, topology_snapshot.committee_votes(local_shard))
}

/// True if `qc`'s `weighted_timestamp` is implausibly far ahead of `now`.
///
/// The weighted timestamp rides outside the QC's signed message
/// (`BlockVoteMessage` covers only shard/height/round/hashes), so a Byzantine
/// proposer or forwarder can rewrite it on an otherwise-genuine QC and still
/// pass `VerifyQcSignature`. A far-future value poisons the BFT clock that
/// anchors transaction-validity windows — honest transactions fall outside the
/// window (blocks go empty) and the aggregation floor propagates the skew
/// irreversibly. An honestly-aggregated weighted timestamp is a mean of voters'
/// clocks from an earlier round, so it leads ours by at most the honest skew
/// envelope; anything beyond is rejected. Checked wherever an untrusted QC
/// enters chain state: header validation, synced-block admission,
/// timeout-quorum `high_qc` adoption, and local QC aggregation (per-vote
/// timestamps are equally unsigned, so the aggregated mean is no more
/// trustworthy than a received QC's field).
#[must_use]
pub fn qc_weighted_timestamp_too_far_ahead(qc: &QuorumCertificate, now: LocalTimestamp) -> bool {
    let weighted_ms = qc.weighted_timestamp().as_millis();
    let max_ahead_ms =
        u64::try_from((MAX_TIMESTAMP_DELAY + MAX_TIMESTAMP_RUSH).as_millis()).unwrap_or(u64::MAX);
    weighted_ms > now.as_millis().saturating_add(max_ahead_ms)
}

/// Validate block header structure, proposer, and parent QC quorum. Returns
/// `Err(..)` with a human-readable reason on any check failure.
///
/// The header's two committee-keyed checks resolve against different
/// committees at an epoch boundary: the proposer of block `h` belongs to
/// `committee(h)` (`proposer_committee`), while the parent QC over `h-1` was
/// signed by `committee(h-1)` (`parent_committee`). Both are anchored on
/// `h-1`'s header, so both are `None` when it hasn't arrived and the caller
/// can't resolve them; `parent_committee` is additionally `None` when the
/// parent QC is genesis (no quorum to check). A skipped proposer check is
/// re-run against the exact committee before this node votes.
pub fn validate_header(
    proposer_committee: Option<&TopologySnapshot>,
    parent_committee: Option<&TopologySnapshot>,
    local_shard: ShardId,
    header: &BlockHeader,
    committed_height: BlockHeight,
    now: LocalTimestamp,
) -> Result<(), String> {
    let height = header.height();
    let round = header.round();

    if height <= committed_height {
        return Err(format!(
            "height {} is at or below committed height {}",
            height.inner(),
            committed_height.inner()
        ));
    }

    if let Some(committee) = proposer_committee {
        validate_proposer(committee, local_shard, header)?;
    }

    // The round span between the parent QC and this block is the number of
    // skipped rounds every validator materializes as `MissedProposal`
    // beacon-witness leaves. Bound it so a Byzantine proposer (the
    // deterministic proposer for arbitrarily large rounds) can't name itself
    // at a runaway round and force an unbounded per-block allocation.
    let parent_round = header.parent_qc().round();
    if round < parent_round {
        return Err(format!(
            "round {} is below parent QC round {}",
            round.inner(),
            parent_round.inner()
        ));
    }
    if round.inner() - parent_round.inner() > MAX_ROUND_GAP {
        return Err(format!(
            "round gap {} exceeds maximum {MAX_ROUND_GAP} (round {}, parent QC round {})",
            round.inner() - parent_round.inner(),
            round.inner(),
            parent_round.inner()
        ));
    }

    if !header.parent_qc().is_genesis() {
        // The parent QC's signing committee is `committee(h-1)`. When the
        // caller can't resolve it (we don't hold `h-1`'s header yet), skip the
        // quorum **pre-check** — it's a cheap DoS filter, and the parent QC is
        // fully signature-verified against the exact committee before we ever vote,
        // once `h-1` arrives. The structural checks below need no committee.
        if let Some(parent_committee) = parent_committee
            && !qc_has_local_quorum_power(parent_committee, local_shard, header.parent_qc())
        {
            return Err("parent QC does not have quorum".to_string());
        }

        if header.parent_qc().height().next() != height {
            return Err(format!(
                "parent QC height {} doesn't match block height {} - 1",
                header.parent_qc().height().inner(),
                height.inner()
            ));
        }

        if header.parent_block_hash() != header.parent_qc().block_hash() {
            return Err(format!(
                "parent_block_hash {:?} doesn't match parent_qc.block_hash() {:?}",
                header.parent_block_hash(),
                header.parent_qc().block_hash()
            ));
        }
    } else if height != committed_height.next() {
        return Err(format!(
            "genesis QC only valid for first block after committed height, got height {}",
            height.inner()
        ));
    }

    // The parent QC's `weighted_timestamp` anchors this block's
    // transaction-validity window but rides outside the QC's signed message, so
    // a Byzantine proposer or forwarder can forge it; a far-future value forces
    // empty blocks and propagates irreversibly through the aggregation floor.
    // See [`qc_weighted_timestamp_too_far_ahead`].
    if qc_weighted_timestamp_too_far_ahead(header.parent_qc(), now) {
        return Err(format!(
            "parent QC weighted timestamp {} is too far ahead (now: {})",
            header.parent_qc().weighted_timestamp().as_millis(),
            now.as_millis()
        ));
    }

    validate_timestamp(header, now)?;

    Ok(())
}

/// Validate that `header` names the proposer its committee elects for the
/// header's round.
pub fn validate_proposer(
    proposer_committee: &TopologySnapshot,
    local_shard: ShardId,
    header: &BlockHeader,
) -> Result<(), String> {
    let expected_proposer = proposer_committee.proposer_for(local_shard, header.round());
    if header.proposer() != expected_proposer {
        return Err(format!(
            "wrong proposer: expected {:?}, got {:?}",
            expected_proposer,
            header.proposer()
        ));
    }
    Ok(())
}

/// Validate that the proposer's timestamp is within acceptable bounds.
///
/// The timestamp must not be more than [`MAX_TIMESTAMP_DELAY`] behind our
/// clock nor more than [`MAX_TIMESTAMP_RUSH`] ahead.
///
/// Skipped for genesis blocks (fixed zero timestamp) and fallback blocks,
/// which inherit the parent's weighted timestamp and so can sit below the
/// delay threshold during extended view changes. The carve-out is sound
/// because `header.timestamp()` is a non-authenticated liveness hint with
/// no consensus consumer — the BFT clock is the QC's `weighted_timestamp`,
/// aggregated from voters' own clocks, not this field.
pub fn validate_timestamp(header: &BlockHeader, now: LocalTimestamp) -> Result<(), String> {
    if header.is_genesis() {
        return Ok(());
    }
    if header.is_fallback() {
        return Ok(());
    }

    let max_delay_ms = u64::try_from(MAX_TIMESTAMP_DELAY.as_millis()).unwrap_or(u64::MAX);
    let max_rush_ms = u64::try_from(MAX_TIMESTAMP_RUSH.as_millis()).unwrap_or(u64::MAX);

    let now_ms = now.as_millis();
    let header_ts_ms = header.timestamp().as_millis();

    if header_ts_ms < now_ms.saturating_sub(max_delay_ms) {
        return Err(format!(
            "proposer timestamp {header_ts_ms} is too old (now: {now_ms}, max delay: {max_delay_ms}ms)"
        ));
    }

    if header_ts_ms > now_ms.saturating_add(max_rush_ms) {
        return Err(format!(
            "proposer timestamp {header_ts_ms} is too far ahead (now: {now_ms}, max rush: {max_rush_ms}ms)"
        ));
    }

    Ok(())
}

/// Validate transaction ordering in a proposed block: transactions must be
/// sorted by hash (ascending, strict). Intra-block duplicate detection falls
/// out of the same check.
pub fn validate_transaction_ordering(block: &Block) -> Result<(), String> {
    verify_hash_sorted(block.transactions(), "transactions")
}

/// Validate that no transaction in the block has already been committed or
/// appears in an ancestor block above committed height (the QC chain).
/// Intra-block duplicates are excluded by the strict hash-ordering check.
///
/// Caller must precompute `qc_chain_tx_hashes` via the driver's QC-chain
/// walk; this function keeps validation pure and does not reach into pending
/// block storage itself.
pub fn validate_no_duplicate_transactions(
    block: &Block,
    qc_chain_tx_hashes: &HashSet<TxHash>,
    dedup_index: &CommitDedupIndex,
) -> Result<(), String> {
    if block.transactions().is_empty() {
        return Ok(());
    }

    for tx in block.transactions().iter() {
        let tx_hash = tx.hash();
        if qc_chain_tx_hashes.contains(&tx_hash) {
            return Err(format!(
                "transaction {tx_hash} already in QC chain ancestor"
            ));
        }
        if dedup_index.contains_tx(&tx_hash) {
            return Err(format!(
                "transaction {tx_hash} already committed within its validity window"
            ));
        }
    }
    Ok(())
}

/// Validate that no transaction the block's finalizations resolve has
/// already been resolved — by an earlier finalization in this same block,
/// by an ancestor above committed height, or by a committed block within
/// the retention window.
///
/// **This is what makes settlement and abandonment exclusive.** A shard
/// abandons a transaction its counterpart left in flight, and settles one
/// whose coverage completed; both are verdicts, and a transaction that
/// took two of them would be settled on one reading of the chain and
/// aborted on another. One rule covers both directions because it asks
/// about the transaction rather than about which kind of verdict each
/// certificate carried.
///
/// The tick check beside it is the same rule at a coarser key, for the
/// case the fine key cannot reach: a manifest names the ticks an ancestor
/// carries but not the transactions under them, so an ancestor whose
/// finalizations this node has not yet fetched contributes nothing to
/// `qc_chain_resolved_txs` and is caught here instead.
///
/// An abandoning record is held to the same rule. It licenses an abort and
/// carries the terms of one, so a record naming a transaction the chain
/// has already resolved is a request for a second verdict on it — and one
/// every replica would honour, since a replica reconstructs the entry from
/// the record precisely when it cannot check the name against an account
/// of its own. A settling record is held only to its own block: it names
/// a transaction the chain resolved before its consumer could accept.
///
/// Both proposer and validator hit `record_block_committed` synchronously
/// during their respective commit handlers, so their `dedup_index` reflects
/// the same just-committed ticks at the same logical moment. Validation
/// against this shared state is therefore safe under the on-qc-formed race.
pub fn validate_no_duplicate_resolutions(
    block: &Block,
    qc_chain_resolved_txs: &HashSet<TxHash>,
    qc_chain_finalizations: &HashSet<FinalizationHash>,
    dedup_index: &CommitDedupIndex,
) -> Result<(), String> {
    let mut resolved_here: HashSet<TxHash> = HashSet::new();
    let mut carried_here: HashSet<FinalizationHash> = HashSet::new();
    for fw in block.certificates().iter() {
        // The certificate's own identity, held to once per block and once
        // per chain. A tick whose members reach no verdict names nothing
        // the per-transaction rule below can hold it to, so without this
        // the same certificate rides every block.
        let receipt_hash = fw.receipt_hash();
        if !carried_here.insert(receipt_hash) {
            return Err(format!(
                "finalization {receipt_hash:?} appears twice in the same block"
            ));
        }
        if qc_chain_finalizations.contains(&receipt_hash) {
            return Err(format!(
                "finalization {receipt_hash:?} is already carried by a QC chain ancestor"
            ));
        }
        if dedup_index.contains_finalization(&receipt_hash) {
            return Err(format!(
                "finalization {receipt_hash:?} was already committed within its retention window"
            ));
        }
        // Every name is held to once per block; only a deciding one is
        // held against what the chain already resolved, since a leg's
        // finalization resolves nothing and the reclaim's may follow it.
        for outcome in fw.local_ec().tx_outcomes() {
            let tx_hash = outcome.tx_hash();
            if !resolved_here.insert(tx_hash) {
                return Err(format!(
                    "transaction {tx_hash} resolved twice within the same block"
                ));
            }
            if outcome.decides() {
                reject_if_resolved(tx_hash, qc_chain_resolved_txs, dedup_index)?;
            }
        }
    }
    for verdict in block.abandonment_records() {
        for tx_hash in verdict.tx_hashes() {
            if resolved_here.contains(&tx_hash) {
                return Err(format!(
                    "abandonment record names {tx_hash}, which the same block resolves"
                ));
            }
            // A settling record names a transaction the chain resolved
            // by design — the issuer's own verdict committed before its
            // consumer could accept — so only an abandoning one is a
            // request for a second verdict.
            if verdict.evidence().abandons() {
                reject_if_resolved(tx_hash, qc_chain_resolved_txs, dedup_index)?;
            }
        }
    }
    Ok(())
}

/// Refuse `tx_hash` if the chain has already reached a verdict on it — by
/// an ancestor above committed height, or by a committed block within the
/// retention window.
fn reject_if_resolved(
    tx_hash: TxHash,
    qc_chain_resolved_txs: &HashSet<TxHash>,
    dedup_index: &CommitDedupIndex,
) -> Result<(), String> {
    if qc_chain_resolved_txs.contains(&tx_hash) {
        return Err(format!(
            "transaction {tx_hash} already resolved by a QC chain ancestor"
        ));
    }
    if dedup_index.contains_resolved_tx(&tx_hash) {
        return Err(format!(
            "transaction {tx_hash} already resolved within its retention window"
        ));
    }
    Ok(())
}

/// Validate that no provisions batch in the block has already been committed
/// or appears in an ancestor block above committed height. Mirrors
/// [`validate_no_duplicate_transactions`] but for `ProvisionHash`.
///
/// Without this check, the on-qc-formed race could cause a proposer to
/// re-include a just-committed batch — the duplicate is technically
/// idempotent (admission no-ops via `pipeline.has_verified`), but the
/// re-inclusion wastes block bytes and re-runs verification. Validators
/// reject it outright.
pub fn validate_no_duplicate_provisions(
    block: &Block,
    qc_chain_provision_hashes: &HashSet<ProvisionHash>,
    dedup_index: &CommitDedupIndex,
) -> Result<(), String> {
    if block.provisions().is_empty() {
        return Ok(());
    }

    for batch in block.provisions() {
        let provision_hash = batch.hash();
        if qc_chain_provision_hashes.contains(&provision_hash) {
            return Err(format!(
                "provisions batch {provision_hash:?} already in QC chain ancestor"
            ));
        }
        if dedup_index.contains_provision(&provision_hash) {
            return Err(format!(
                "provisions batch {provision_hash:?} already committed within its retention window"
            ));
        }
    }
    Ok(())
}

/// Validate that no provisions batch in the block is fenced by a recovery
/// pending in the block's governing snapshot: content from a recovering
/// shard above its attested frontier is rejected network-wide (INV-SEC-8),
/// and folding the check into block validity keeps every replica's verdict
/// a pure function of the block's own anchor — a block anchored before the
/// recovery folded resolves a snapshot without the record and stays valid.
pub fn validate_provisions_not_fenced(
    topology_snapshot: &TopologySnapshot,
    block: &Block,
) -> Result<(), String> {
    for batch in block.provisions() {
        let source_shard = batch.source_shard();
        if topology_snapshot.recovery_fences(source_shard, batch.block_height()) {
            return Err(format!(
                "provisions batch {:?} from recovering shard {source_shard:?} above the \
                 attested frontier",
                batch.hash(),
            ));
        }
    }
    Ok(())
}

/// A non-payer shard engages a cross-shard transaction only against
/// its payer bundle — the transaction commit proof. The block is valid
/// only if each such transaction's bundle rides in this same block or a
/// committed bundle named it within the retention window. Deterministic
/// over block content plus committed chain content; the proposer's own
/// selection gate makes honest proposals satisfy it, so this closes the
/// Byzantine-proposer path to engaging counterpart locks before the
/// payer shard commits.
pub fn validate_engagement(
    topology_snapshot: &TopologySnapshot,
    local_shard: ShardId,
    block: &Block,
    dedup_index: &CommitDedupIndex,
) -> Result<(), String> {
    for tx in block.transactions().iter() {
        if topology_snapshot.is_single_shard_transaction(tx) {
            continue;
        }
        let payer_shard = topology_snapshot
            .shard_trie()
            .shard_for_prefix(tx.body().fee_payer);
        if payer_shard == local_shard {
            continue;
        }
        let tx_hash = tx.hash();
        let in_block = block.provisions().iter().any(|batch| {
            batch.source_shard() == payer_shard
                && batch
                    .transactions()
                    .iter()
                    .any(|entry| entry.tx_hash == tx_hash)
        });
        if !in_block && !dedup_index.contains_provision_tx(payer_shard, tx_hash) {
            return Err(format!(
                "cross-shard VM transaction {tx_hash} lacks its payer bundle \
                 from {payer_shard:?}"
            ));
        }
    }
    Ok(())
}

/// Every package a transaction names must be one this window may run:
/// registered, past its maturity window, or born with the chain.
///
/// The window is what every node fetches a newly registered artifact's
/// bytes in, so what this establishes is that by the time a transaction
/// can run, the code it runs is code the whole committee holds. Without
/// it, whether a transaction executes or refuses for want of code is a
/// question about whose fetch finished first — and replicas that answer
/// it differently attest different ticks.
///
/// Stated as the permission rather than the refusal, which is what makes
/// it checkable here. Refusing only the registered-and-immature would
/// leave the window between a publish committing on its own shard and
/// the beacon registering it, and closing that by argument — about which
/// nodes hold which metadata, and so which of them could have built the
/// transaction at all — is an invariant no reader can check locally.
///
/// Deterministic over block content plus the window-frozen registry
/// every member of the committee shares; the proposer's own selection
/// gate makes honest proposals satisfy it.
pub fn validate_packages_usable(
    topology_snapshot: &TopologySnapshot,
    block: &Block,
) -> Result<(), String> {
    for tx in block.transactions().iter() {
        if let Some(package) = topology_snapshot.unusable_package_of(tx) {
            return Err(format!(
                "transaction {} names package {package}, which this window cannot run",
                tx.hash()
            ));
        }
    }
    Ok(())
}

/// The cells a block's transactions create for a sweep to retire must
/// fit the per-block creation cap.
///
/// The sweep's own cap bounds how fast a shard can retire these; this
/// bounds how fast it can be made to owe them. Only the pair bounds the
/// resident population — a creation rate above the removal cap is a
/// backlog that grows for as long as the load lasts, and a sweep that
/// bounds ordinary operation and not the peak is not a bound.
///
/// Counted off the derivations, which is where the answer is: what makes
/// a write sweepable is the family it belongs to, and nothing about a
/// routed key says which family it is — and counted for this shard,
/// since a cell lands only on the shard owning its prefix and that is
/// the shard whose sweep retires it — plus the committed-transaction
/// cell the chain itself writes for every transaction the block
/// carries. Deterministic over block content and the window's trie,
/// since every replica derives the same transactions the same way.
pub fn validate_sweepable_creation(
    topology_snapshot: &TopologySnapshot,
    local_shard: ShardId,
    block: &Block,
) -> Result<(), String> {
    let trie = topology_snapshot.shard_trie();
    let created = block.transactions().iter().fold(0usize, |total, tx| {
        total.saturating_add(
            tx.sweepable_writes_on(trie, local_shard)
                + usize::from(writes_committed_cell(
                    tx.legs(),
                    tx.owners(),
                    trie,
                    local_shard,
                )),
        )
    });
    if !sweep_admits_block(created) {
        return Err(format!(
            "block creates {created} sweepable cells, past the per-block cap of \
             {MAX_SWEEPABLE_CREATED_PER_BLOCK}"
        ));
    }
    Ok(())
}

/// The header's running work total must be its parent's advanced by the
/// work the block's own certificates report.
///
/// Pure over the block plus one scalar off the parent header, which is what
/// keeps a shard's attested work honest without any storage read: a
/// proposer inflating its shard's emission weight has to inflate receipts
/// its committee already checked under `local_receipt_root`.
fn validate_block_work(block: &Block, parent_load: Option<ShardLoad>) -> Result<(), String> {
    // An unresolvable parent load is this node's own gap, not the block's,
    // so it abstains rather than rejecting: recovery reads the scalar off
    // the committed tip's stored header and a fresh start seeds `ZERO`, so
    // the only way here is a store with no block at its committed height.
    let Some(parent_load) = parent_load else {
        tracing::warn!(
            height = block.height().inner(),
            "Skipping the work-total check — parent load unresolvable"
        );
        return Ok(());
    };
    let claimed = block.header().load().cumulative_work;
    let expected = parent_load
        .advance(block.attested_work(), None)
        .cumulative_work;
    if claimed != expected {
        return Err(format!(
            "header claims cumulative work {claimed} but the parent's {} \
             plus this block's {} is {expected}",
            parent_load.cumulative_work,
            block.attested_work(),
        ));
    }
    Ok(())
}

/// Run all pre-vote block-contents checks: transaction ordering, `ticks`
/// recomputation, and cross-ancestor uniqueness for txs, certs, and
/// provisions. Returns a single diagnostic on the first failure so the
/// caller can log once.
#[allow(clippy::too_many_arguments)] // single dispatch over the pre-vote content checks
pub fn validate_block_for_vote(
    topology_snapshot: &TopologySnapshot,
    local_shard: ShardId,
    block: &Block,
    qc_chain_tx_hashes: &HashSet<TxHash>,
    qc_chain_resolved_txs: &HashSet<TxHash>,
    qc_chain_finalizations: &HashSet<FinalizationHash>,
    qc_chain_provision_hashes: &HashSet<ProvisionHash>,
    dedup_index: &CommitDedupIndex,
    coasting: bool,
    parent_load: Option<ShardLoad>,
) -> Result<(), String> {
    if coasting {
        validate_coast_block_empty(block)?;
    }
    validate_block_work(block, parent_load)?;
    validate_transactions_verified(block)?;
    validate_transaction_ordering(block)?;
    validate_no_duplicate_transactions(block, qc_chain_tx_hashes, dedup_index)?;
    validate_no_duplicate_resolutions(
        block,
        qc_chain_resolved_txs,
        qc_chain_finalizations,
        dedup_index,
    )?;
    validate_no_duplicate_provisions(block, qc_chain_provision_hashes, dedup_index)?;
    validate_provisions_not_fenced(topology_snapshot, block)?;
    validate_packages_usable(topology_snapshot, block)?;
    validate_sweepable_creation(topology_snapshot, local_shard, block)?;
    validate_engagement(topology_snapshot, local_shard, block, dedup_index)?;
    validate_abandonment_records_well_formed(block)?;
    validate_state_proofs_well_formed(block)?;
    Ok(())
}

/// Validate the block's state-proof bundles against the header that
/// commits them and against the one form a section may take.
///
/// Structural only — whether each proof reconstructs its anchor's root
/// is the delegated check, and whether the anchor is the commit-proven
/// header's is the vote fence's. What this establishes is that every
/// replica reads the same section: the root binds the bundles to the
/// header, and the canonical order — within each bundle and across
/// them — means one set of answers has one encoding.
pub fn validate_state_proofs_well_formed(block: &Block) -> Result<(), String> {
    let bundles = block.state_proofs();
    let computed = state_proofs_root_from_bundles(bundles);
    let claimed = block.header().state_proofs_root();
    if computed != claimed {
        return Err(format!(
            "state proofs root {claimed:?} does not commit the block's bundles {computed:?}"
        ));
    }
    if bundles.len() > MAX_PROVISIONS_PER_BLOCK {
        return Err(format!(
            "block carries {} state proofs, over the cap of {MAX_PROVISIONS_PER_BLOCK}",
            bundles.len()
        ));
    }
    for (at, claim) in bundles.iter().enumerate() {
        if !claim.is_well_formed() {
            return Err(format!(
                "counterpart claim {at} is empty, over its cap, out of order, or names a \
                 verdict that licenses nothing"
            ));
        }
        if at > 0 && bundles[at - 1] >= *claim {
            return Err(format!(
                "counterpart claim {at} repeats or precedes the one before it"
            ));
        }
    }
    Ok(())
}

/// Validate the block's abandonment records against the header that
/// commits them, against the one form a record may take, and against the
/// budget they share.
///
/// Structural only — whether the records tell the truth is a question for
/// the departed shards' settled sets, which this cannot see. What it
/// establishes is that every replica reads the same claim: the root binds
/// the records to the header, and the canonical order — within each
/// record and across them — means one claim has one encoding, so two
/// proposers naming the same transactions cannot produce blocks that
/// differ.
///
/// The budget checked is the sum across every record, because
/// [`MAX_UNSETTLED_PER_BLOCK`] doubles as each record's own decode cap and
/// that cap alone would let a block spend it once per record.
pub fn validate_abandonment_records_well_formed(block: &Block) -> Result<(), String> {
    let verdicts = block.abandonment_records();
    let computed = abandonment_root_from_records(verdicts);
    let claimed = block.header().abandonment_root();
    if computed != claimed {
        return Err(format!(
            "abandonment root {claimed:?} does not commit the block's records {computed:?}"
        ));
    }

    let mut named = 0usize;
    let mut previous: Option<(ShardId, u8)> = None;
    for verdict in verdicts {
        if !verdict.is_well_formed() {
            return Err(format!(
                "abandonment record for {:?} is empty, over its cap, or out of order",
                verdict.shard(),
            ));
        }
        // Ascending by shard and then by arm, which gives uniqueness and
        // one encoding per claim set together — two records for one
        // shard under one arm would leave which answer counts to the
        // reader, and a reordering would be a second form of the same
        // block. One shard may carry several arms: what it claimed and
        // what it left unclaimed are different transactions.
        let position = (verdict.shard(), verdict.evidence().discriminant());
        if previous.is_some_and(|previous| previous >= position) {
            return Err(format!(
                "abandonment record for {:?} repeats or precedes the one before it",
                verdict.shard(),
            ));
        }
        previous = Some(position);
        named = named.saturating_add(verdict.unsettled().len());
    }
    if named > MAX_UNSETTLED_PER_BLOCK {
        return Err(format!(
            "abandonment records name {named} transactions, over the drain's own bound of \
             {MAX_UNSETTLED_PER_BLOCK}",
        ));
    }
    Ok(())
}

/// A coast block — one whose parent QC's weighted timestamp lands past
/// the shard's terminal window — exists only to certify the crossing. It
/// must carry no content of any kind, so state stays frozen at the
/// crossing's root: no transactions, no certificates, no provisions, and
/// no boundary records, which a chain whose own capacity to resolve
/// anything ended at its cut has nothing left to write down.
fn validate_coast_block_empty(block: &Block) -> Result<(), String> {
    if !block.transactions().is_empty() {
        return Err(format!(
            "coast block past the terminal window carries {} transactions",
            block.transactions().len()
        ));
    }
    if !block.certificates().is_empty() {
        return Err(format!(
            "coast block past the terminal window carries {} certificates",
            block.certificates().len()
        ));
    }
    if !block.provisions().is_empty() {
        return Err(format!(
            "coast block past the terminal window carries {} provisions",
            block.provisions().len()
        ));
    }
    if !block.abandonment_records().is_empty() {
        return Err(format!(
            "coast block past the terminal window carries {} abandonment records",
            block.abandonment_records().len()
        ));
    }
    if !block.state_proofs().is_empty() {
        return Err(format!(
            "coast block past the terminal window carries {} state proofs",
            block.state_proofs().len()
        ));
    }
    Ok(())
}

/// Refuse to vote on a block whose `transactions` entries are not all
/// `Verifiable::Verified`. Honest voters source every tx from local
/// admission-validated state (mempool / fetch cache); an `Unverified` entry
/// means assembly couldn't obtain or validate the body, and voting would
/// break the BFT-transitive trust chain that downstream `from_persisted`
/// gates rely on.
fn validate_transactions_verified(block: &Block) -> Result<(), String> {
    for tx in block.transactions().iter() {
        if tx.verified().is_none() {
            return Err(format!(
                "transaction {} is not admission-validated",
                tx.hash()
            ));
        }
    }
    Ok(())
}

/// Verify that a list of transactions is sorted by hash in strict ascending
/// order. `section` is used in the error message for diagnostics.
fn verify_hash_sorted(txs: &[Arc<Verifiable<Transaction>>], section: &str) -> Result<(), String> {
    for window in txs.windows(2) {
        if window[0].hash() >= window[1].hash() {
            return Err(format!(
                "{} section not in hash order: {} >= {}",
                section,
                window[0].hash(),
                window[1].hash()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use hyperscale_crypto_bls::BlsSigner;
    use hyperscale_types::test_utils::{
        TestCommittee, make_finalization, make_undecided_finalization, stub_abort_charge,
        test_principal,
    };
    use hyperscale_types::{
        AbandonmentRecord, AbandonmentRoot, Address, AddressClass, AggregateSignature, BlockHash,
        BlockHeader, BlockHeaderParts, ChainOrigin, CounterpartClaim, Deadline, Finalization, Hash,
        LocalKey, MAX_SUBINTENTS, MerkleInclusionProof, NetworkDefinition, PrincipalAddr,
        ProposerTimestamp, ProvisionEntry, Provisions, QuorumCertificate, Round, ShardId,
        ShardLoad, Signer, SignerBitfield, StateAnchor, StateProofBundle, StateProofsRoot,
        StateRoot, SubstateKey, TimestampRange, Transaction, TransactionDecision, UnsettledTx,
        ValidatorId, ValidatorInfo, ValidatorSet, VerdictClaim, Verifiable, Verified,
        WeightedTimestamp, WitnessSources, test_utils,
    };

    use super::*;

    fn topology_snapshot() -> TopologySnapshot {
        let committee = TestCommittee::new(4, 42);
        let validators: Vec<ValidatorInfo> = (0..committee.size())
            .map(|i| ValidatorInfo {
                validator_id: committee.validator_id(i),
                public_key: *committee.public_key(i),
            })
            .collect();
        TopologySnapshot::new(
            NetworkDefinition::simulator(),
            1,
            ValidatorSet::new(validators),
        )
    }

    fn local_shard() -> ShardId {
        ShardId::ROOT
    }

    fn header_at_height(height: BlockHeight, timestamp_ms: u64) -> BlockHeader {
        BlockHeader::new(BlockHeaderParts {
            height,
            parent_block_hash: BlockHash::from_raw(Hash::from_bytes(b"parent")),
            parent_qc: QuorumCertificate::genesis(ShardId::ROOT, ChainOrigin::ROOT).into(),
            proposer: ValidatorId::new(height.inner() % 4),
            timestamp: ProposerTimestamp::from_millis(timestamp_ms),
            round: Round::new(0),
            provision_tx_roots: std::collections::BTreeMap::new(),
            ..Default::default()
        })
    }

    fn header_with_overrides(
        base: &BlockHeader,
        round: Option<Round>,
        is_fallback: Option<bool>,
        parent_block_hash: Option<BlockHash>,
        proposer: Option<ValidatorId>,
    ) -> BlockHeader {
        BlockHeader::new(BlockHeaderParts {
            shard_id: base.shard_id(),
            height: base.height(),
            parent_block_hash: parent_block_hash.unwrap_or_else(|| base.parent_block_hash()),
            parent_qc: base.parent_qc().clone().into(),
            proposer: proposer.unwrap_or_else(|| base.proposer()),
            timestamp: base.timestamp(),
            round: round.unwrap_or_else(|| base.round()),
            is_fallback: is_fallback.unwrap_or_else(|| base.is_fallback()),
            state_root: base.state_root(),
            transaction_root: base.transaction_root(),
            certificate_root: base.certificate_root(),
            local_receipt_root: base.local_receipt_root(),
            provision_root: base.provision_root(),
            provision_tx_roots: base.provision_tx_roots().clone(),
            work_in_flight: base.work_in_flight(),
            ..Default::default()
        })
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_timestamp
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_timestamp_skips_genesis() {
        let now = LocalTimestamp::from_millis(100_000);
        let header = header_with_overrides(
            &header_at_height(BlockHeight::new(0), 0),
            None,
            None,
            Some(BlockHash::from_raw(Hash::from_bytes(b"genesis_parent"))),
            Some(ValidatorId::new(0)),
        );
        assert!(validate_timestamp(&header, now).is_ok());
    }

    #[test]
    fn validate_timestamp_accepts_within_bounds() {
        let now = LocalTimestamp::from_millis(100_000);
        for ts_ms in [99_000, 100_000, 101_000] {
            let header = header_at_height(BlockHeight::new(1), ts_ms);
            assert!(
                validate_timestamp(&header, now).is_ok(),
                "ts_ms={ts_ms} should be within bounds"
            );
        }
    }

    #[test]
    fn validate_timestamp_rejects_too_old() {
        let now = LocalTimestamp::from_millis(100_000);
        let header = header_at_height(BlockHeight::new(1), 50_000);
        let err = validate_timestamp(&header, now).unwrap_err();
        assert!(err.contains("too old"));
    }

    #[test]
    fn validate_timestamp_rejects_too_far_ahead() {
        let now = LocalTimestamp::from_millis(100_000);
        let header = header_at_height(BlockHeight::new(1), 110_000);
        let err = validate_timestamp(&header, now).unwrap_err();
        assert!(err.contains("too far ahead"));
    }

    #[test]
    fn validate_timestamp_at_boundary() {
        let now = LocalTimestamp::from_millis(100_000);

        // Exactly max delay (now - 30s) — OK.
        assert!(validate_timestamp(&header_at_height(BlockHeight::new(1), 70_000), now).is_ok());
        // Just past max delay — fail.
        assert!(validate_timestamp(&header_at_height(BlockHeight::new(1), 69_999), now).is_err());
        // Exactly max rush (now + 2s) — OK.
        assert!(validate_timestamp(&header_at_height(BlockHeight::new(1), 102_000), now).is_ok());
        // Just past max rush — fail.
        assert!(validate_timestamp(&header_at_height(BlockHeight::new(1), 102_001), now).is_err());
    }

    #[test]
    fn validate_timestamp_skips_fallback_blocks() {
        let now = LocalTimestamp::from_millis(100_000);

        // 50s old would normally fail (MAX_TIMESTAMP_DELAY = 30s), but fallback
        // blocks inherit the parent's weighted timestamp across view changes.
        let base = header_at_height(BlockHeight::new(1), 50_000);
        let header_fallback =
            header_with_overrides(&base, Some(Round::new(5)), Some(true), None, None);
        assert!(validate_timestamp(&header_fallback, now).is_ok());

        let header_normal =
            header_with_overrides(&base, Some(Round::new(5)), Some(false), None, None);
        assert!(validate_timestamp(&header_normal, now).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_header round-gap bound
    // ═══════════════════════════════════════════════════════════════════════

    fn header_at_round(height: BlockHeight, round: Round, topo: &TopologySnapshot) -> BlockHeader {
        let base = header_at_height(height, 100_000);
        let proposer = topo.proposer_for(local_shard(), round);
        header_with_overrides(&base, Some(round), None, None, Some(proposer))
    }

    #[test]
    fn validate_header_rejects_runaway_round_gap() {
        let topo = topology_snapshot();
        let now = LocalTimestamp::from_millis(100_000);
        let height = BlockHeight::new(1);
        let header = header_at_round(height, Round::new(MAX_ROUND_GAP + 1), &topo);

        let err = validate_header(
            Some(&topo),
            Some(&topo),
            local_shard(),
            &header,
            BlockHeight::new(0),
            now,
        )
        .unwrap_err();
        assert!(err.contains("round gap"), "got: {err}");
    }

    #[test]
    fn validate_header_accepts_round_gap_at_cap() {
        let topo = topology_snapshot();
        let now = LocalTimestamp::from_millis(100_000);
        let height = BlockHeight::new(1);
        let header = header_at_round(height, Round::new(MAX_ROUND_GAP), &topo);

        assert!(
            validate_header(
                Some(&topo),
                Some(&topo),
                local_shard(),
                &header,
                BlockHeight::new(0),
                now
            )
            .is_ok()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_header parent-QC weighted-timestamp bound
    // ═══════════════════════════════════════════════════════════════════════

    /// A non-genesis parent QC for height 1 with quorum signers (3 of the
    /// 4-member committee) and a chosen `weighted_timestamp`.
    fn quorum_parent_qc(weighted_ms: u64) -> QuorumCertificate {
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        signers.set(1);
        signers.set(2);
        QuorumCertificate::new(
            BlockHash::from_raw(Hash::from_bytes(b"parent_block")),
            ShardId::ROOT,
            BlockHeight::new(1),
            BlockHash::from_raw(Hash::from_bytes(b"grandparent")),
            Round::new(0),
            signers,
            AggregateSignature::ZERO,
            WeightedTimestamp::from_millis(weighted_ms),
        )
    }

    /// A height-2, round-1 header that extends `parent_qc`, with the correct
    /// proposer and a valid proposer timestamp, so the parent-QC timestamp
    /// bound is the only check under test.
    fn header_extending(parent_qc: QuorumCertificate, now: LocalTimestamp) -> BlockHeader {
        let round = Round::new(1);
        let proposer = topology_snapshot().proposer_for(local_shard(), round);
        BlockHeader::new(BlockHeaderParts {
            height: BlockHeight::new(2),
            parent_block_hash: parent_qc.block_hash(),
            parent_qc: parent_qc.into(),
            proposer,
            timestamp: ProposerTimestamp::from_millis(now.as_millis()),
            round,
            provision_tx_roots: std::collections::BTreeMap::new(),
            ..Default::default()
        })
    }

    #[test]
    fn validate_header_rejects_far_future_parent_qc_timestamp() {
        let topo = topology_snapshot();
        let now = LocalTimestamp::from_millis(1_000_000);

        // Parent QC an hour ahead of our clock — far beyond the honest skew
        // envelope. The unsigned `weighted_timestamp` lets a Byzantine peer
        // forge this on an otherwise-genuine QC.
        let header = header_extending(quorum_parent_qc(now.as_millis() + 3_600_000), now);

        let err = validate_header(
            Some(&topo),
            Some(&topo),
            local_shard(),
            &header,
            BlockHeight::new(0),
            now,
        )
        .unwrap_err();
        assert!(
            err.contains("parent QC weighted timestamp"),
            "expected far-future parent QC rejection, got: {err}"
        );
    }

    #[test]
    fn validate_header_accepts_recent_parent_qc_timestamp() {
        let topo = topology_snapshot();
        let now = LocalTimestamp::from_millis(1_000_000);

        // Honest case: the parent QC was aggregated a few seconds ago, so its
        // weighted timestamp sits just behind our clock.
        let header = header_extending(quorum_parent_qc(now.as_millis() - 5_000), now);

        assert!(
            validate_header(
                Some(&topo),
                Some(&topo),
                local_shard(),
                &header,
                BlockHeight::new(0),
                now
            )
            .is_ok(),
            "honest recent parent QC timestamp must pass"
        );
    }

    #[test]
    fn qc_weighted_timestamp_bound_is_the_honest_skew_envelope() {
        let now = LocalTimestamp::from_millis(1_000_000);
        let envelope_ms =
            u64::try_from((MAX_TIMESTAMP_DELAY + MAX_TIMESTAMP_RUSH).as_millis()).unwrap();

        // Behind our clock (the honest case) and exactly at the envelope: kept.
        assert!(!qc_weighted_timestamp_too_far_ahead(
            &quorum_parent_qc(now.as_millis() - 100_000),
            now
        ));
        assert!(!qc_weighted_timestamp_too_far_ahead(
            &quorum_parent_qc(now.as_millis() + envelope_ms),
            now
        ));

        // One millisecond past the envelope: rejected.
        assert!(qc_weighted_timestamp_too_far_ahead(
            &quorum_parent_qc(now.as_millis() + envelope_ms + 1),
            now
        ));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_header two-committee resolution (epoch boundary)
    // ═══════════════════════════════════════════════════════════════════════

    /// A uniform-power committee over `ids`, one shard.
    fn committee_with_ids(ids: &[u64]) -> TopologySnapshot {
        let validators: Vec<ValidatorInfo> = ids
            .iter()
            .map(|&id| {
                let mut seed = [0u8; 32];
                seed[..8].copy_from_slice(&id.to_le_bytes());
                ValidatorInfo {
                    validator_id: ValidatorId::new(id),
                    public_key: BlsSigner::from_seed(&seed).public_key(),
                }
            })
            .collect();
        TopologySnapshot::new(
            NetworkDefinition::simulator(),
            1,
            ValidatorSet::new(validators),
        )
    }

    /// A non-genesis parent QC over height 1 with a single signer — below
    /// quorum in any committee of more than one member.
    fn single_signer_parent_qc(weighted_ms: u64) -> QuorumCertificate {
        let mut signers = SignerBitfield::new(4);
        signers.set(0);
        QuorumCertificate::new(
            BlockHash::from_raw(Hash::from_bytes(b"parent_block")),
            ShardId::ROOT,
            BlockHeight::new(1),
            BlockHash::from_raw(Hash::from_bytes(b"grandparent")),
            Round::new(0),
            signers,
            AggregateSignature::ZERO,
            WeightedTimestamp::from_millis(weighted_ms),
        )
    }

    /// A height-2, round-1 header extending `parent_qc`, proposed by `proposer`,
    /// with a valid proposer timestamp — so proposer and parent-QC quorum are
    /// the only committee-keyed checks under test.
    fn header_with_proposer(
        parent_qc: QuorumCertificate,
        proposer: ValidatorId,
        now: LocalTimestamp,
    ) -> BlockHeader {
        BlockHeader::new(BlockHeaderParts {
            height: BlockHeight::new(2),
            parent_block_hash: parent_qc.block_hash(),
            parent_qc: parent_qc.into(),
            proposer,
            timestamp: ProposerTimestamp::from_millis(now.as_millis()),
            round: Round::new(1),
            provision_tx_roots: std::collections::BTreeMap::new(),
            ..Default::default()
        })
    }

    #[test]
    fn validate_header_keys_proposer_and_parent_on_distinct_committees() {
        // At an epoch boundary the proposer of block `h` belongs to
        // `committee(h)` while `h`'s parent QC was signed by `committee(h-1)`.
        // `validate_header` draws the proposer from the first committee and
        // checks the parent-QC quorum against the second; passing the committees
        // in the wrong roles rejects the header.
        let now = LocalTimestamp::from_millis(1_000_000);
        let parent_committee = committee_with_ids(&[0, 1, 2, 3]); // committee(h-1)
        let proposer_committee = committee_with_ids(&[10, 11, 12, 13]); // committee(h)

        let round = Round::new(1);
        let proposer = proposer_committee.proposer_for(local_shard(), round);
        let header = header_with_proposer(quorum_parent_qc(now.as_millis() - 5_000), proposer, now);

        assert!(
            validate_header(
                Some(&proposer_committee),
                Some(&parent_committee),
                local_shard(),
                &header,
                BlockHeight::new(0),
                now,
            )
            .is_ok(),
            "header must validate under committee(h) proposer + committee(h-1) quorum",
        );

        let err = validate_header(
            Some(&parent_committee),
            Some(&proposer_committee),
            local_shard(),
            &header,
            BlockHeight::new(0),
            now,
        )
        .unwrap_err();
        assert!(
            err.contains("wrong proposer"),
            "drawing the proposer from the parent committee must reject: {err}"
        );
    }

    #[test]
    fn validate_header_skips_parent_quorum_when_committee_unresolved() {
        // When `h-1`'s header hasn't arrived its committee can't be resolved, so
        // the caller passes `None` and the cheap quorum pre-check is skipped —
        // the parent QC is still fully signature-verified against the exact committee
        // before this node votes. A resolved committee runs the pre-check and
        // rejects a sub-quorum parent QC.
        let topo = topology_snapshot();
        let now = LocalTimestamp::from_millis(1_000_000);
        let proposer = topo.proposer_for(local_shard(), Round::new(1));
        let header = header_with_proposer(
            single_signer_parent_qc(now.as_millis() - 5_000),
            proposer,
            now,
        );

        let err = validate_header(
            Some(&topo),
            Some(&topo),
            local_shard(),
            &header,
            BlockHeight::new(0),
            now,
        )
        .unwrap_err();
        assert!(
            err.contains("parent QC does not have quorum"),
            "a resolved parent committee must enforce the quorum pre-check: {err}"
        );

        assert!(
            validate_header(
                Some(&topo),
                None,
                local_shard(),
                &header,
                BlockHeight::new(0),
                now
            )
            .is_ok(),
            "an unresolved parent committee must skip the quorum pre-check",
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_transaction_ordering
    // ═══════════════════════════════════════════════════════════════════════

    fn block_with_transactions(
        height: BlockHeight,
        transactions: Vec<Arc<Verifiable<Transaction>>>,
    ) -> Block {
        Block::Live {
            header: header_at_height(height, 100_000),
            transactions: Arc::new(transactions),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
            abandonment_records: Arc::new(Vec::new()),
            state_proofs: Arc::new(Vec::new()),
        }
    }

    /// A block creating sweepable cells fits the cap right up to what a
    /// transaction can add, and one that would pass it is refused.
    ///
    /// The count is summed off the derivations rather than off anything
    /// the header claims, so a proposer cannot understate what its block
    /// will make the chain carry.
    ///
    /// An envelope binds at most `MAX_SUBINTENTS`, so the cap is reached
    /// by a block of fully composed transactions rather than by one
    /// transaction — which is the shape it is sized for.
    #[test]
    fn a_block_may_create_sweepable_cells_up_to_the_cap() {
        // Each fully composed transaction creates its subintents'
        // nullifiers, all on this one shard; its core is this shard
        // alone, so the chain writes no committed cell for it.
        let full = MAX_SWEEPABLE_CREATED_PER_BLOCK / MAX_SUBINTENTS;
        let mut txs: Vec<Arc<Verifiable<Transaction>>> = (0..full)
            .map(|i| {
                Arc::new(Verifiable::from(test_utils::stub_transaction_binding(
                    u32::try_from(i).expect("fewer than u32 transactions"),
                    MAX_SUBINTENTS,
                    test_utils::test_validity_range(),
                )))
            })
            .collect();
        let at_cap = block_with_transactions(BlockHeight::new(3), txs.clone());
        let topo = topology_snapshot();
        assert!(validate_sweepable_creation(&topo, local_shard(), &at_cap).is_ok());

        txs.push(Arc::new(Verifiable::from(
            test_utils::stub_transaction_binding(
                u32::MAX,
                MAX_SUBINTENTS,
                test_utils::test_validity_range(),
            ),
        )));
        let over = block_with_transactions(BlockHeight::new(3), txs);
        let err = validate_sweepable_creation(&topo, local_shard(), &over)
            .expect_err("past the cap is refused");
        assert!(err.contains("sweepable cells"), "{err}");

        // A block that binds nothing creates nothing, whatever else it
        // carries — the common case must not pay for this rule.
        let plain = block_with_transactions(BlockHeight::new(3), vec![tx(1)]);
        assert!(validate_sweepable_creation(&topo, local_shard(), &plain).is_ok());
    }

    /// A block carrying records, rooted the way the header claims.
    fn block_with_verdicts(verdicts: Vec<AbandonmentRecord>, root: AbandonmentRoot) -> Block {
        let base = header_at_height(BlockHeight::new(6), 100_000);
        Block::Live {
            header: BlockHeader::new(BlockHeaderParts {
                height: base.height(),
                parent_block_hash: base.parent_block_hash(),
                parent_qc: base.parent_qc().clone().into(),
                proposer: base.proposer(),
                timestamp: base.timestamp(),
                round: base.round(),
                provision_tx_roots: std::collections::BTreeMap::new(),
                abandonment_root: root,
                ..Default::default()
            }),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            abandonment_records: Arc::new(verdicts),
            state_proofs: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
        }
    }

    /// A block carrying `bundles` under a header claiming `root`.
    fn block_with_state_proofs(bundles: Vec<CounterpartClaim>, root: StateProofsRoot) -> Block {
        let base = header_at_height(BlockHeight::new(6), 100_000);
        Block::Live {
            header: BlockHeader::new(BlockHeaderParts {
                height: base.height(),
                parent_block_hash: base.parent_block_hash(),
                parent_qc: base.parent_qc().clone().into(),
                proposer: base.proposer(),
                timestamp: base.timestamp(),
                round: base.round(),
                provision_tx_roots: std::collections::BTreeMap::new(),
                state_proofs_root: root,
                ..Default::default()
            }),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(Vec::new()),
            abandonment_records: Arc::new(Vec::new()),
            state_proofs: Arc::new(bundles),
            witness_sources: Arc::new(WitnessSources::empty()),
        }
    }

    /// A cells claim against `ROOT` at `height`, answering for `keys`.
    fn bundle_at(height: u64, keys: &[u8]) -> CounterpartClaim {
        CounterpartClaim::Cells(StateProofBundle::new(
            StateAnchor {
                shard: ShardId::ROOT,
                height: BlockHeight::new(height),
                state_root: StateRoot::from_raw(Hash::from_bytes(b"root")),
            },
            WeightedTimestamp::from_millis(height * 1_000),
            keys.iter().map(|seed| SubstateKey {
                owner: Address::new([*seed; 31], AddressClass::Component),
                local: LocalKey([*seed; 16]),
            }),
            MerkleInclusionProof::dummy(),
        ))
    }

    /// The section is bound to the header's root and held to one form:
    /// ascending without repeats, every bundle naming something. A
    /// second form of the same answers, or a root that does not commit
    /// them, is refused before any proof is walked.
    #[test]
    fn a_state_proof_section_is_held_to_its_root_and_form() {
        let bundles = vec![bundle_at(3, &[1]), bundle_at(4, &[2, 3])];
        let root = state_proofs_root_from_bundles(&bundles);
        assert!(
            validate_state_proofs_well_formed(&block_with_state_proofs(bundles.clone(), root))
                .is_ok()
        );

        let err = validate_state_proofs_well_formed(&block_with_state_proofs(
            bundles.clone(),
            StateProofsRoot::ZERO,
        ))
        .expect_err("a root that does not commit the bundles is refused");
        assert!(err.contains("does not commit"), "{err}");

        let reversed: Vec<CounterpartClaim> = bundles.iter().rev().cloned().collect();
        let err = validate_state_proofs_well_formed(&block_with_state_proofs(
            reversed.clone(),
            state_proofs_root_from_bundles(&reversed),
        ))
        .expect_err("out of order is a second form of the same section");
        assert!(err.contains("repeats or precedes"), "{err}");

        let repeated = vec![bundle_at(3, &[1]), bundle_at(3, &[1])];
        let err = validate_state_proofs_well_formed(&block_with_state_proofs(
            repeated.clone(),
            state_proofs_root_from_bundles(&repeated),
        ))
        .expect_err("a repeated bundle is refused");
        assert!(err.contains("repeats or precedes"), "{err}");

        let CounterpartClaim::Cells(cells) = bundle_at(3, &[1]) else {
            unreachable!("a cells claim")
        };
        let empty = vec![CounterpartClaim::Cells(StateProofBundle {
            keys: Vec::new(),
            ..cells
        })];
        let err = validate_state_proofs_well_formed(&block_with_state_proofs(
            empty.clone(),
            state_proofs_root_from_bundles(&empty),
        ))
        .expect_err("a bundle naming no key is refused");
        assert!(err.contains("empty"), "{err}");

        // A verdict that licenses nothing is as malformed as a bundle
        // naming no key: both cost a leaf for an answer nothing reads.
        let accepted = vec![CounterpartClaim::Verdict(VerdictClaim {
            shard: ShardId::ROOT,
            tx_hash: TxHash::from(Hash::from_bytes(b"tx")),
            anchor_ts: WeightedTimestamp::from_millis(1),
            decision: TransactionDecision::Accept,
            digest: Hash::from_bytes(b"digest"),
        })];
        let err = validate_state_proofs_well_formed(&block_with_state_proofs(
            accepted.clone(),
            state_proofs_root_from_bundles(&accepted),
        ))
        .expect_err("an acceptance licenses no record");
        assert!(err.contains("licenses nothing"), "{err}");
    }

    fn named(tx_hash: TxHash) -> UnsettledTx {
        UnsettledTx {
            tx_hash,
            deadline: Deadline::of(WeightedTimestamp::from_millis(900)),
            declared_work: 11,
            charge: stub_abort_charge(11),
        }
    }

    fn verdict(shard: ShardId, seeds: &[u8]) -> AbandonmentRecord {
        AbandonmentRecord::departed(
            shard,
            WeightedTimestamp::from_millis(1_000),
            seeds
                .iter()
                .map(|&seed| named(TxHash::from(Hash::from_bytes(&[seed; 32])))),
        )
    }

    /// The header commits the records, so a block whose root does not
    /// cover what it carries is refused before anyone asks whether the
    /// records are true.
    #[test]
    fn a_block_whose_root_does_not_commit_its_records_is_refused() {
        let records = vec![verdict(ShardId::ROOT, &[1, 2])];
        let honest = abandonment_root_from_records(&records);
        assert!(
            validate_abandonment_records_well_formed(&block_with_verdicts(records.clone(), honest))
                .is_ok()
        );

        let err = validate_abandonment_records_well_formed(&block_with_verdicts(
            records,
            AbandonmentRoot::ZERO,
        ))
        .unwrap_err();
        assert!(err.contains("does not commit"), "{err}");
    }

    /// One claim has one encoding. A record out of its canonical order is
    /// a second form of the same claim and is refused, so two proposers
    /// naming the same transactions cannot build differing blocks.
    #[test]
    fn a_record_out_of_its_canonical_form_is_refused() {
        let malformed =
            AbandonmentRecord::departed(ShardId::ROOT, WeightedTimestamp::from_millis(1_000), []);
        let root = abandonment_root_from_records(std::slice::from_ref(&malformed));
        let err =
            validate_abandonment_records_well_formed(&block_with_verdicts(vec![malformed], root))
                .unwrap_err();
        assert!(
            err.contains("empty, over its cap, or out of order"),
            "{err}"
        );
    }

    /// Ascending by shard, which is what gives uniqueness and one encoding
    /// per claim set at once: two records for one shard would leave which
    /// answer counts to the reader, and a reordering would be a second
    /// form of the same block.
    #[test]
    fn records_out_of_shard_order_or_repeating_a_shard_are_refused() {
        let (left, right) = ShardId::ROOT.children();
        assert!(left < right, "the fixture relies on the child ordering");

        let ordered = vec![verdict(left, &[1]), verdict(right, &[2])];
        let root = abandonment_root_from_records(&ordered);
        assert!(
            validate_abandonment_records_well_formed(&block_with_verdicts(ordered, root)).is_ok()
        );

        for records in [
            vec![verdict(right, &[2]), verdict(left, &[1])],
            vec![verdict(left, &[1]), verdict(left, &[2])],
        ] {
            let root = abandonment_root_from_records(&records);
            let err = validate_abandonment_records_well_formed(&block_with_verdicts(records, root))
                .unwrap_err();
            assert!(err.contains("repeats or precedes"), "{err}");
        }

        // One shard under two arms is two answers about two sets of
        // transactions, in arm order.
        let two_arms = vec![
            verdict(left, &[1]),
            AbandonmentRecord::accepted(
                left,
                WeightedTimestamp::from_millis(9),
                [named(TxHash::from(Hash::from_bytes(&[2; 32])))],
            ),
        ];
        let root = abandonment_root_from_records(&two_arms);
        assert!(
            validate_abandonment_records_well_formed(&block_with_verdicts(two_arms, root)).is_ok()
        );
        let arms_reversed = vec![
            AbandonmentRecord::accepted(
                left,
                WeightedTimestamp::from_millis(9),
                [named(TxHash::from(Hash::from_bytes(&[2; 32])))],
            ),
            verdict(left, &[1]),
        ];
        let root = abandonment_root_from_records(&arms_reversed);
        assert!(
            validate_abandonment_records_well_formed(&block_with_verdicts(arms_reversed, root))
                .is_err()
        );
    }

    /// The drain is one budget across every departure a block answers
    /// for. Each record's own cap is the same figure, because one
    /// departure may hold the whole of it — so without the sum a block
    /// could spend the budget once per record.
    #[test]
    fn records_naming_more_than_the_drain_can_hold_are_refused() {
        // Two records, each half the budget plus one, so neither trips its
        // own cap and together they clear the block's.
        let half = MAX_UNSETTLED_PER_BLOCK / 2 + 1;
        let (left, right) = ShardId::ROOT.children();
        let span = |shard: ShardId, from: usize| {
            AbandonmentRecord::departed(
                shard,
                WeightedTimestamp::from_millis(1_000),
                (from..from + half)
                    .map(|i| named(TxHash::from(Hash::from_bytes(&i.to_le_bytes())))),
            )
        };
        let records = vec![span(left, 0), span(right, half)];
        for record in &records {
            assert!(record.is_well_formed(), "each record is within its own cap");
        }

        let root = abandonment_root_from_records(&records);
        let err = validate_abandonment_records_well_formed(&block_with_verdicts(records, root))
            .unwrap_err();
        assert!(err.contains("over the drain's own bound"), "{err}");
    }

    /// The running work total is a validity condition, not a hint: a header
    /// claiming more than its parent's total plus its own certificates'
    /// work is rejected, and the honest claim passes. A block with no
    /// certificates consumes nothing, so it must repeat its parent's total
    /// rather than reset.
    #[test]
    fn a_header_cannot_overstate_its_shard_s_work() {
        let parent = ShardLoad::ZERO.advance(500, None);
        // The fixture carries no certificates, so the honest claim is the
        // parent's total unchanged.
        let honest = block_with_transactions(BlockHeight::new(1), Vec::new());
        assert_eq!(honest.attested_work(), 0);
        assert_eq!(honest.header().load().cumulative_work, 0);

        // Claiming zero against a parent that has consumed 500 understates,
        // and is refused just as an overstatement is.
        let err = validate_block_work(&honest, Some(parent)).unwrap_err();
        assert!(err.contains("cumulative work"), "{err}");

        // The matching claim passes.
        assert!(validate_block_work(&honest, Some(ShardLoad::ZERO)).is_ok());

        // An unresolvable parent load abstains rather than rejecting.
        assert!(validate_block_work(&honest, None).is_ok());
    }

    fn tx(seed: u8) -> Arc<Verifiable<Transaction>> {
        Arc::new(Verifiable::from(test_utils::test_transaction(seed)))
    }

    fn sorted_txs(seeds: &[u8]) -> Vec<Arc<Verifiable<Transaction>>> {
        let mut txs: Vec<_> = seeds.iter().map(|&s| tx(s)).collect();
        txs.sort_by_key(|t| t.hash());
        txs
    }

    #[test]
    fn validate_transaction_ordering_accepts_empty_block() {
        let block = block_with_transactions(BlockHeight::new(5), vec![]);
        assert!(validate_transaction_ordering(&block).is_ok());
    }

    #[test]
    fn validate_transaction_ordering_accepts_single_tx() {
        let block = block_with_transactions(BlockHeight::new(5), vec![tx(1)]);
        assert!(validate_transaction_ordering(&block).is_ok());
    }

    #[test]
    fn validate_transaction_ordering_accepts_sorted() {
        let block = block_with_transactions(BlockHeight::new(5), sorted_txs(&[10, 20, 30]));
        assert!(validate_transaction_ordering(&block).is_ok());
    }

    #[test]
    fn validate_transaction_ordering_rejects_reversed() {
        let mut txs = sorted_txs(&[10, 20, 30]);
        txs.reverse();
        let block = block_with_transactions(BlockHeight::new(5), txs);
        let err = validate_transaction_ordering(&block).unwrap_err();
        assert!(err.contains("not in hash order"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_no_duplicate_transactions
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn validate_no_duplicate_transactions_accepts_empty_block() {
        let block = block_with_transactions(BlockHeight::new(5), vec![]);
        let qc_chain = HashSet::new();
        let dedup_index = CommitDedupIndex::new();
        assert!(validate_no_duplicate_transactions(&block, &qc_chain, &dedup_index).is_ok());
    }

    #[test]
    fn validate_no_duplicate_transactions_accepts_unique() {
        let block = block_with_transactions(BlockHeight::new(5), sorted_txs(&[10, 20]));
        let qc_chain = HashSet::new();
        let dedup_index = CommitDedupIndex::new();
        assert!(validate_no_duplicate_transactions(&block, &qc_chain, &dedup_index).is_ok());
    }

    #[test]
    fn validate_no_duplicate_transactions_rejects_qc_chain_dup() {
        let txs = sorted_txs(&[10, 20]);
        let dup_hash = txs[0].hash();
        let block = block_with_transactions(BlockHeight::new(6), txs);
        let qc_chain: HashSet<_> = std::iter::once(dup_hash).collect();
        let dedup_index = CommitDedupIndex::new();
        let err = validate_no_duplicate_transactions(&block, &qc_chain, &dedup_index).unwrap_err();
        assert!(err.contains("already in QC chain ancestor"));
    }

    #[test]
    fn validate_no_duplicate_transactions_rejects_retention_dup() {
        let txs = sorted_txs(&[10, 20]);
        let dup_tx = Arc::clone(&txs[0]);
        let block = block_with_transactions(BlockHeight::new(6), txs);
        let qc_chain = HashSet::new();
        let mut dedup_index = CommitDedupIndex::new();
        dedup_index.register_committed_txs(&[dup_tx]);
        let err = validate_no_duplicate_transactions(&block, &qc_chain, &dedup_index).unwrap_err();
        assert!(err.contains("already committed"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_no_duplicate_certificates
    // ═══════════════════════════════════════════════════════════════════════

    fn block_with_certificates(height: BlockHeight, certificates: Vec<Arc<Finalization>>) -> Block {
        let wrapped: Vec<Arc<Verifiable<Finalization>>> = certificates
            .into_iter()
            .map(|fw| Arc::new((*fw).clone().into()))
            .collect();
        Block::Live {
            header: header_at_height(height, 100_000),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(wrapped),
            provisions: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
            abandonment_records: Arc::new(Vec::new()),
            state_proofs: Arc::new(Vec::new()),
        }
    }

    fn finalization_at(height: u64) -> Arc<Finalization> {
        Arc::new(make_finalization(
            BlockHeight::new(height),
            TxHash::from(Hash::from_bytes(
                &[u8::try_from(height).unwrap_or(u8::MAX); 32],
            )),
            TransactionDecision::Accept,
        ))
    }

    /// A finalization over `tx_hash`, on a tick distinct from any other
    /// built at the same height — the shape a second verdict takes.
    fn finalization_over(height: u64, tx_hash: TxHash) -> Arc<Finalization> {
        Arc::new(make_finalization(
            BlockHeight::new(height),
            tx_hash,
            TransactionDecision::Aborted,
        ))
    }

    fn no_resolutions(block: &Block, dedup_index: &CommitDedupIndex) -> Result<(), String> {
        validate_no_duplicate_resolutions(block, &HashSet::new(), &HashSet::new(), dedup_index)
    }

    #[test]
    fn validate_no_duplicate_resolutions_accepts_empty_block() {
        let block = block_with_certificates(BlockHeight::new(5), vec![]);
        assert!(no_resolutions(&block, &CommitDedupIndex::new()).is_ok());
    }

    #[test]
    fn validate_no_duplicate_resolutions_accepts_unique() {
        let block = block_with_certificates(BlockHeight::new(5), vec![finalization_at(1)]);
        assert!(no_resolutions(&block, &CommitDedupIndex::new()).is_ok());
    }

    /// The same certificate again is refused on its identity, which is
    /// the rule that answers whether or not its members reach a verdict.
    /// The transaction rule below covers the other shape — a *different*
    /// certificate reaching a second verdict for one name.
    #[test]
    fn validate_no_duplicate_resolutions_rejects_retention_dup() {
        let fw = finalization_at(1);
        let block = block_with_certificates(BlockHeight::new(6), vec![Arc::clone(&fw)]);
        let mut dedup_index = CommitDedupIndex::new();
        dedup_index.register_committed_certs(&[Arc::new((*fw).clone().into())]);
        let err = no_resolutions(&block, &dedup_index).unwrap_err();
        assert!(
            err.contains("was already committed within its retention window"),
            "{err}"
        );
    }

    /// A certificate whose members reach no verdict is held to the chain
    /// exactly like one that does. Nothing about its names can refuse it
    /// — it resolves none — so identity is the only thing that can.
    #[test]
    fn a_committed_certificate_deciding_nothing_cannot_ride_a_second_block() {
        let fw = Arc::new(make_undecided_finalization(
            BlockHeight::new(1),
            TxHash::from(Hash::from_bytes(b"retired")),
            TransactionDecision::Accept,
        ));
        assert_eq!(fw.deciding_tx_hashes().count(), 0);
        let block = block_with_certificates(BlockHeight::new(6), vec![Arc::clone(&fw)]);
        let mut dedup_index = CommitDedupIndex::new();
        dedup_index.register_committed_certs(&[Arc::new((*fw).clone().into())]);
        let err = no_resolutions(&block, &dedup_index).unwrap_err();
        assert!(
            err.contains("was already committed within its retention window"),
            "{err}"
        );
    }

    /// A second verdict for one transaction is refused however it is
    /// dressed. The tick differs, so nothing about the certificate's
    /// identity gives it away — settlement and abandonment are exclusive
    /// because the rule asks about the transaction.
    #[test]
    fn a_transaction_a_committed_block_resolved_cannot_be_resolved_again() {
        let settled = finalization_at(1);
        let tx_hash = settled
            .tx_hashes()
            .next()
            .expect("a tick names its members");
        let mut dedup_index = CommitDedupIndex::new();
        dedup_index.register_committed_certs(&[Arc::new((*settled).clone().into())]);

        let abandoned = finalization_over(9, tx_hash);
        assert_ne!(abandoned.tick_id(), settled.tick_id());
        let block = block_with_certificates(BlockHeight::new(6), vec![abandoned]);

        let err = no_resolutions(&block, &dedup_index).unwrap_err();
        assert!(
            err.contains("already resolved within its retention window"),
            "{err}"
        );
    }

    /// A boundary record is a request for a verdict, so it is held to the
    /// same rule. It reaches replicas that hold no account of the
    /// transaction and rebuild one from it, which is exactly the replica
    /// that cannot tell the name is stale — so the block carrying it is
    /// where the staleness has to be caught.
    #[test]
    fn a_record_naming_a_resolved_transaction_is_refused() {
        let settled = finalization_at(1);
        let tx_hash = settled
            .tx_hashes()
            .next()
            .expect("a tick names its members");
        let mut dedup_index = CommitDedupIndex::new();
        dedup_index.register_committed_certs(&[Arc::new((*settled).clone().into())]);

        let block = block_with_verdicts(
            vec![AbandonmentRecord::departed(
                ShardId::ROOT,
                WeightedTimestamp::from_millis(1_000),
                [named(tx_hash)],
            )],
            AbandonmentRoot::ZERO,
        );
        let err = no_resolutions(&block, &dedup_index).unwrap_err();
        assert!(
            err.contains("already resolved within its retention window"),
            "{err}"
        );
    }

    /// And a record naming what the same block resolves, which no window
    /// or index has seen yet.
    #[test]
    fn a_record_naming_what_its_own_block_resolves_is_refused() {
        let settled = finalization_at(1);
        let tx_hash = settled
            .tx_hashes()
            .next()
            .expect("a tick names its members");
        let block = Block::Live {
            header: header_at_height(BlockHeight::new(6), 100_000),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(vec![Arc::new((*settled).clone().into())]),
            provisions: Arc::new(Vec::new()),
            state_proofs: Arc::new(Vec::new()),
            witness_sources: Arc::new(WitnessSources::empty()),
            abandonment_records: Arc::new(vec![AbandonmentRecord::departed(
                ShardId::ROOT,
                WeightedTimestamp::from_millis(1_000),
                [named(tx_hash)],
            )]),
        };
        let err = no_resolutions(&block, &CommitDedupIndex::new()).unwrap_err();
        assert!(err.contains("which the same block resolves"), "{err}");
    }

    /// The same, one block earlier: an ancestor above committed height
    /// has resolved it and nothing has committed yet.
    #[test]
    fn a_transaction_an_ancestor_resolved_cannot_be_resolved_again() {
        let settled = finalization_at(1);
        let tx_hash = settled
            .tx_hashes()
            .next()
            .expect("a tick names its members");
        let ancestor_resolved: HashSet<TxHash> = std::iter::once(tx_hash).collect();

        let block =
            block_with_certificates(BlockHeight::new(6), vec![finalization_over(9, tx_hash)]);
        let err = validate_no_duplicate_resolutions(
            &block,
            &ancestor_resolved,
            &HashSet::new(),
            &CommitDedupIndex::new(),
        )
        .unwrap_err();
        assert!(
            err.contains("already resolved by a QC chain ancestor"),
            "{err}"
        );
    }

    /// And within one block, where neither the ancestor walk nor the
    /// retention window can see it.
    #[test]
    fn a_transaction_cannot_be_resolved_twice_within_one_block() {
        let settled = finalization_at(1);
        let tx_hash = settled
            .tx_hashes()
            .next()
            .expect("a tick names its members");
        let block = block_with_certificates(
            BlockHeight::new(6),
            vec![settled, finalization_over(9, tx_hash)],
        );
        let err = no_resolutions(&block, &CommitDedupIndex::new()).unwrap_err();
        assert!(
            err.contains("resolved twice within the same block"),
            "{err}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_no_duplicate_provisions
    // ═══════════════════════════════════════════════════════════════════════

    fn block_with_provisions(height: BlockHeight, provisions: Vec<Arc<Provisions>>) -> Block {
        let wrapped: Vec<Arc<Verifiable<Provisions>>> = provisions
            .into_iter()
            .map(|p| Arc::new((*p).clone().into()))
            .collect();
        Block::Live {
            header: header_at_height(height, 100_000),
            transactions: Arc::new(Vec::new()),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(wrapped),
            witness_sources: Arc::new(WitnessSources::empty()),
            abandonment_records: Arc::new(Vec::new()),
            state_proofs: Arc::new(Vec::new()),
        }
    }

    fn provisions_with_seed(seed: u8) -> Arc<Provisions> {
        let tx_hash = TxHash::from(Hash::from_bytes(&[seed; 32]));
        Arc::new(Provisions::new(
            ShardId::leaf(1, 0),
            ShardId::leaf(1, 1),
            BlockHeight::new(u64::from(seed)),
            WeightedTimestamp::ZERO,
            MerkleInclusionProof::dummy(),
            vec![ProvisionEntry::new(tx_hash, vec![])],
        ))
    }

    #[test]
    fn validate_no_duplicate_provisions_accepts_empty_block() {
        let block = block_with_provisions(BlockHeight::new(5), vec![]);
        let qc_chain = HashSet::new();
        let dedup_index = CommitDedupIndex::new();
        assert!(validate_no_duplicate_provisions(&block, &qc_chain, &dedup_index).is_ok());
    }

    #[test]
    fn validate_no_duplicate_provisions_accepts_unique() {
        let block = block_with_provisions(BlockHeight::new(5), vec![provisions_with_seed(1)]);
        let qc_chain = HashSet::new();
        let dedup_index = CommitDedupIndex::new();
        assert!(validate_no_duplicate_provisions(&block, &qc_chain, &dedup_index).is_ok());
    }

    #[test]
    fn validate_no_duplicate_provisions_rejects_qc_chain_dup() {
        let p = provisions_with_seed(1);
        let dup_hash = p.hash();
        let block = block_with_provisions(BlockHeight::new(6), vec![p]);
        let qc_chain: HashSet<_> = std::iter::once(dup_hash).collect();
        let dedup_index = CommitDedupIndex::new();
        let err = validate_no_duplicate_provisions(&block, &qc_chain, &dedup_index).unwrap_err();
        assert!(err.contains("already in QC chain ancestor"));
    }

    #[test]
    fn validate_no_duplicate_provisions_rejects_retention_dup() {
        let p = provisions_with_seed(1);
        let block = block_with_provisions(BlockHeight::new(6), vec![Arc::clone(&p)]);
        let qc_chain = HashSet::new();
        let mut dedup_index = CommitDedupIndex::new();
        dedup_index
            .register_committed_provisions(&[p.hash()], WeightedTimestamp::from_millis(1_000));
        let err = validate_no_duplicate_provisions(&block, &qc_chain, &dedup_index).unwrap_err();
        assert!(err.contains("already committed"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_provisions_not_fenced
    // ═══════════════════════════════════════════════════════════════════════

    /// A snapshot whose `shard` is under a pending recovery fenced at
    /// `frontier` — the governing snapshot of a block validated after the
    /// recovery folded.
    fn snapshot_recovering(shard: ShardId, frontier: BlockHeight) -> TopologySnapshot {
        use hyperscale_types::{Epoch, NetworkDefinition, RecoveryCause, ShardRecovery};
        TopologySnapshot::new(
            NetworkDefinition::simulator(),
            1,
            ValidatorSet::new(Vec::new()),
        )
        .with_pending_recoveries(
            std::iter::once((
                shard,
                ShardRecovery {
                    cause: RecoveryCause::Halt,
                    rotated_at: Epoch::new(2),
                    retained: Vec::new(),
                    attested_frontier: frontier,
                },
            ))
            .collect(),
        )
    }

    #[test]
    fn validate_provisions_not_fenced_rejects_above_frontier() {
        // provisions_with_seed(9) sources from ShardId::leaf(1, 0) at
        // height 9 — above a frontier of 5 the batch is fenced content.
        let block = block_with_provisions(BlockHeight::new(6), vec![provisions_with_seed(9)]);
        let snapshot = snapshot_recovering(ShardId::leaf(1, 0), BlockHeight::new(5));
        let err = validate_provisions_not_fenced(&snapshot, &block).unwrap_err();
        assert!(err.contains("recovering shard"));
    }

    #[test]
    fn validate_provisions_not_fenced_accepts_frontier_and_below() {
        // At the frontier the batch is legitimate pre-failure history.
        let block = block_with_provisions(BlockHeight::new(6), vec![provisions_with_seed(5)]);
        let snapshot = snapshot_recovering(ShardId::leaf(1, 0), BlockHeight::new(5));
        assert!(validate_provisions_not_fenced(&snapshot, &block).is_ok());
    }

    #[test]
    fn validate_provisions_not_fenced_ignores_other_shards_and_clean_snapshots() {
        let block = block_with_provisions(BlockHeight::new(6), vec![provisions_with_seed(9)]);
        // Recovery pending for a different shard.
        let other = snapshot_recovering(ShardId::leaf(1, 1), BlockHeight::new(5));
        assert!(validate_provisions_not_fenced(&other, &block).is_ok());
        // No recovery pending at all — a pre-fold governing snapshot.
        let clean = snapshot_recovering(ShardId::leaf(1, 1), BlockHeight::new(5));
        let mut recoveries = clean.pending_recoveries().clone();
        recoveries.clear();
        let clean = clean.with_pending_recoveries(recoveries);
        assert!(validate_provisions_not_fenced(&clean, &block).is_ok());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_transactions_verified / validate_block_for_vote verified-arm
    // ═══════════════════════════════════════════════════════════════════════

    fn verified_tx(seed: u8) -> Arc<Verifiable<Transaction>> {
        Arc::new(Verifiable::from(test_utils::verified_test_transaction(
            seed,
        )))
    }

    fn sorted_verified_txs(seeds: &[u8]) -> Vec<Arc<Verifiable<Transaction>>> {
        let mut txs: Vec<_> = seeds.iter().map(|&s| verified_tx(s)).collect();
        txs.sort_by_key(|t| t.hash());
        txs
    }

    #[test]
    fn validate_transactions_verified_accepts_empty_block() {
        let block = block_with_transactions(BlockHeight::new(1), vec![]);
        assert!(validate_transactions_verified(&block).is_ok());
    }

    #[test]
    fn validate_transactions_verified_accepts_all_verified() {
        let block =
            block_with_transactions(BlockHeight::new(1), sorted_verified_txs(&[10, 20, 30]));
        assert!(validate_transactions_verified(&block).is_ok());
    }

    #[test]
    fn validate_transactions_verified_rejects_any_unverified() {
        // Mix one Unverified entry into an otherwise-Verified block.
        let mut txs = sorted_verified_txs(&[10, 20]);
        let unverified = tx(30);
        txs.push(unverified);
        txs.sort_by_key(|t| t.hash());
        let block = block_with_transactions(BlockHeight::new(1), txs);
        let err = validate_transactions_verified(&block).unwrap_err();
        assert!(err.contains("not admission-validated"));
    }

    #[test]
    fn validate_block_for_vote_rejects_unverified_before_other_checks() {
        // Out-of-order + Unverified: the verified-check fires first and
        // short-circuits before ordering is examined.
        let topo = topology_snapshot();
        let mut txs = sorted_verified_txs(&[10, 20]);
        txs.reverse(); // intentionally mis-sort to prove short-circuit
        txs.push(tx(30)); // Unverified entry
        let block = block_with_transactions(BlockHeight::new(1), txs);
        let err = validate_block_for_vote(
            &topo,
            local_shard(),
            &block,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &CommitDedupIndex::new(),
            false,
            Some(ShardLoad::ZERO),
        )
        .unwrap_err();
        assert!(err.contains("not admission-validated"));
    }

    #[test]
    fn coast_blocks_must_be_empty() {
        // Past the terminal window a block exists only to certify the
        // crossing: any content fails the pre-vote check.
        let topo = topology_snapshot();
        let with_tx = block_with_transactions(BlockHeight::new(1), sorted_verified_txs(&[10]));
        let err = validate_block_for_vote(
            &topo,
            local_shard(),
            &with_tx,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &CommitDedupIndex::new(),
            true,
            Some(ShardLoad::ZERO),
        )
        .unwrap_err();
        assert!(err.contains("coast block"), "{err}");

        // A boundary record is content too. A chain whose capacity to
        // resolve anything ended at its cut has nothing left to write
        // down, so the rule covers every body list rather than three of
        // four.
        let records = vec![verdict(ShardId::ROOT, &[1])];
        let with_record =
            block_with_verdicts(records.clone(), abandonment_root_from_records(&records));
        let err = validate_block_for_vote(
            &topo,
            local_shard(),
            &with_record,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &CommitDedupIndex::new(),
            true,
            Some(ShardLoad::ZERO),
        )
        .unwrap_err();
        assert!(err.contains("abandonment records"), "{err}");

        let empty = block_with_transactions(BlockHeight::new(1), Vec::new());
        assert!(
            validate_block_for_vote(
                &topo,
                local_shard(),
                &empty,
                &HashSet::new(),
                &HashSet::new(),
                &HashSet::new(),
                &HashSet::new(),
                &CommitDedupIndex::new(),
                true,
                Some(ShardLoad::ZERO),
            )
            .is_ok()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // validate_engagement
    // ═══════════════════════════════════════════════════════════════════════

    /// A signed stub transaction whose derived owners are exactly
    /// `owners`, paying from `payer`, wrapped verified for block content.
    fn stub_tx(payer: PrincipalAddr, owners: &[Address]) -> Arc<Verifiable<Transaction>> {
        test_utils::install_stub_protocol_statics();
        let validity = TimestampRange::new(
            WeightedTimestamp::ZERO,
            WeightedTimestamp::from_millis(100_000),
        );
        Arc::new(Verifiable::from(Verified::new_unchecked_for_test(
            test_utils::stub_transaction(payer, owners, 1_000, validity),
        )))
    }

    fn block_with_tx(
        tx: &Arc<Verifiable<Transaction>>,
        provisions: Vec<Arc<Verifiable<Provisions>>>,
    ) -> Block {
        Block::Live {
            header: header_at_height(BlockHeight::new(1), 100_000),
            transactions: Arc::new(vec![Arc::clone(tx)]),
            certificates: Arc::new(Vec::new()),
            provisions: Arc::new(provisions),
            witness_sources: Arc::new(WitnessSources::empty()),
            abandonment_records: Arc::new(Vec::new()),
            state_proofs: Arc::new(Vec::new()),
        }
    }

    #[test]
    fn engagement_demands_the_payer_bundle() {
        // Two-shard trie: a clear top bit routes to leaf(1, 0), a set one
        // to leaf(1, 1). The payer lives on the remote shard.
        let topo = TestCommittee::new(4, 42).topology_snapshot(2);
        let local = ShardId::leaf(1, 0);
        let local_owner = test_principal(0x01);
        let payer_owner = test_principal(0x81);
        let tx = stub_tx(payer_owner, &[local_owner.address(), payer_owner.address()]);
        let tx_hash = tx.hash();

        // No bundle anywhere: the counterpart must not engage.
        let bare = block_with_tx(&tx, Vec::new());
        let err = validate_engagement(&topo, local, &bare, &CommitDedupIndex::new()).unwrap_err();
        assert!(err.contains("payer bundle"), "{err}");

        // The payer's empty-entry bundle in the same block: engaged.
        let bundle: Arc<Verifiable<Provisions>> = Arc::new(
            Verified::<Provisions>::new_unchecked_for_test(Provisions::new(
                ShardId::leaf(1, 1),
                local,
                BlockHeight::new(1),
                WeightedTimestamp::ZERO,
                MerkleInclusionProof::dummy(),
                vec![ProvisionEntry::new(tx_hash, vec![])],
            ))
            .into(),
        );
        let paired = block_with_tx(&tx, vec![Arc::clone(&bundle)]);
        assert!(validate_engagement(&topo, local, &paired, &CommitDedupIndex::new()).is_ok());

        // A bundle committed within the retention window: engaged.
        let mut dedup = CommitDedupIndex::new();
        dedup.register_committed_provision_txs(
            std::slice::from_ref(&bundle),
            WeightedTimestamp::from_millis(1_000),
        );
        assert!(validate_engagement(&topo, local, &bare, &dedup).is_ok());

        // A bundle from the wrong source shard is not engagement evidence.
        let wrong_source: Arc<Verifiable<Provisions>> = Arc::new(
            Verified::<Provisions>::new_unchecked_for_test(Provisions::new(
                local,
                ShardId::leaf(1, 1),
                BlockHeight::new(1),
                WeightedTimestamp::ZERO,
                MerkleInclusionProof::dummy(),
                vec![ProvisionEntry::new(tx_hash, vec![])],
            ))
            .into(),
        );
        let mispaired = block_with_tx(&tx, vec![wrong_source]);
        let err =
            validate_engagement(&topo, local, &mispaired, &CommitDedupIndex::new()).unwrap_err();
        assert!(err.contains("payer bundle"), "{err}");
    }

    #[test]
    fn engagement_exempts_the_payer_shard_and_single_shard_legs() {
        let topo = TestCommittee::new(4, 42).topology_snapshot(2);
        let local_owner = test_principal(0x01);
        let payer_owner = test_principal(0x81);

        // At the payer's own shard the reservation, not the bundle, is
        // the gate — no bundle demanded.
        let cross = stub_tx(payer_owner, &[local_owner.address(), payer_owner.address()]);
        let at_payer = block_with_tx(&cross, Vec::new());
        assert!(
            validate_engagement(
                &topo,
                ShardId::leaf(1, 1),
                &at_payer,
                &CommitDedupIndex::new()
            )
            .is_ok()
        );

        // A single-shard leg engages nothing remotely.
        let single = stub_tx(local_owner, &[local_owner.address()]);
        let local_only = block_with_tx(&single, Vec::new());
        assert!(
            validate_engagement(
                &topo,
                ShardId::leaf(1, 0),
                &local_only,
                &CommitDedupIndex::new()
            )
            .is_ok()
        );
    }
}
