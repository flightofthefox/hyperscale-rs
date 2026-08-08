//! The VM engine's wave-batch executor.
//!
//! `execute_wave_batch` runs one wave's VM sub-batch end to end: derive
//! each transaction's manifest and effect set through the bridge
//! (exactly the derivation admission ran), pre-read the declared cells
//! from the wave snapshot into an owned committed base, hand the batch
//! to `vm_kernel::execute_batch`, then fold the schedule-invariant
//! receipts into per-transaction absolute `database_updates` in
//! canonical order against the batch baseline — the same fold the
//! kernel's apply phase performs, checked against its end state before
//! anything is returned.
//!
//! Batch receipts are batch-dependent (reservation feasibility is judged
//! with the whole batch's holds in place), so VM outputs are never
//! memoized in the per-transaction `ProcessExecutionCache` — the same
//! transaction in a different block may abort differently.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use blake3::hash as blake3_hash;
use hyperscale_effects_bridge::vm_statics::{PackageCache, package_key};
use hyperscale_effects_bridge::{
    BridgeStatics, PoolRegistry, ProtocolHasher, admit_package, check_target_authority,
    decode_tree, envelope_identity, witness_from_event,
};
use hyperscale_metrics::record_transaction_executed;
use hyperscale_storage::SubstateDatabase;
use hyperscale_types::{
    BeaconWitnessEvent, BeaconWitnessRoot, ConsensusReceipt, Event, EventExt, EventRoot,
    ExecutionMetadata, FeeSummary, GlobalReceipt, Hash, ProvisionalHolds, RevealChain, Stake,
    StakePoolSeat, StateWrites, SubstateEntry, Transaction, TxHash, Verified, compute_merkle_root,
    install_vm_statics,
};
use hyperscale_vm_effects::{
    Address, Declaration, EffectTarget, InstanceRegistry, NodeCall, PackageHash,
    PrefixShardResolver, RoleId, SubstateKey, admit_tree, package_hash, route_tree,
};
use hyperscale_vm_kernel::{
    Base, BatchTx, EnvInputs, ExecutionMode, Locality, ManifestWalk, Outcome, Receipt, amount_cell,
    execute_batch,
};

use crate::backend::EngineBackend;
use crate::genesis::{World, genesis_world_with_pools};
use crate::sharding::writes_root;
use crate::{
    CachedOutput, CrossShardTxInput, ExecutedTx, TickTxInput, WaveBatchContext, project_to_shard,
};

/// Whether a derivation holds a gated node to its target's authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetAuthority {
    /// The rule as the chain applies it.
    Required,
    /// Every gated node is treated as authorised. A preview grant only,
    /// for answering what an envelope would do before the accounts it
    /// touches have signed for it.
    Assumed,
}

/// One derived transaction, as the batch consumes it.
///
/// The walk itself is the kernel's: routing lowers each manifest node to
/// the invocation its package's ABI binding describes, and
/// [`ManifestWalk`] performs them over the engine backend. Nothing on
/// this side names a method.
pub struct PreparedTx {
    /// The lowered invocations the kernel walks, in manifest node order.
    /// Envelope trees lower into one flat list, so nothing downstream
    /// sees intent structure.
    pub calls: Vec<NodeCall>,
    /// The routed declaration, both views: the folded set scheduling
    /// reads, and the clause order the capability table is built in —
    /// which is what a lowered call's handle positions index.
    pub declaration: Declaration,
    /// The subintent nullifier keys the batch entry enforces.
    pub nullifiers: Vec<SubstateKey>,
    /// The envelope's signed execution ceiling, in fuel — one budget for
    /// the whole transaction, however many nodes its manifest walks.
    pub gas_limit: u64,
}

/// The protocol crypto hash behind the kernel's hashing host function
/// and fresh-ID derivation.
pub fn protocol_hash(data: &[u8]) -> [u8; 32] {
    *blake3_hash(data).as_bytes()
}

/// Domain tag for the per-transaction randomness draw.
const DOMAIN_TX_RANDOMNESS: &[u8] = b"hyperscale/engine/tx-randomness";

/// The transaction's randomness draw: the payer block's reveal chain —
/// its proposer's VRF reveal, attested by the committee that committed
/// the transaction — domain-separated by the transaction hash.
///
/// Anchoring on the payer block is what makes the draw a property of the
/// transaction rather than of whichever block a participant executes it
/// in, so every participant of a cross-shard transaction derives one
/// receipt. Mixing the hash keeps two transactions in one payer block
/// from sharing a draw.
pub fn tx_randomness(anchor: RevealChain, tx: TxHash) -> [u8; 32] {
    *Hash::from_parts(&[
        DOMAIN_TX_RANDOMNESS,
        anchor.as_raw().as_bytes(),
        tx.as_bytes(),
    ])
    .as_bytes()
}

/// The batch's committed baseline: the declared cells pre-read from the
/// wave's JMT-backed snapshot at materialize time.
///
/// Cells only — ordered collections and locks are absent from the
/// current stdlib surface, and reservations never persist across
/// batches, so `holds` is empty by construction. Every kernel read flows
/// through a capability for a declared effect, so pre-reading exactly
/// the declared point targets is complete.
#[derive(Debug, Default)]
pub struct VmBase {
    pub cells: BTreeMap<SubstateKey, Vec<u8>>,
    /// What legs of unresolved waves hold against these cells. Empty for
    /// a baseline with nothing in flight over it — a preview, or a shard
    /// with no cross-shard leg outstanding.
    pub holds: ProvisionalHolds,
}

impl Base for VmBase {
    fn cell(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.cells.get(&key).cloned()
    }

    fn entries_in_range(
        &self,
        _owner: Address,
        _collection: RoleId,
        _lo: u128,
        _hi: u128,
        _limit: usize,
    ) -> Vec<(u128, Vec<u8>)> {
        Vec::new()
    }

    fn is_locked(&self, _key: SubstateKey) -> bool {
        false
    }

    fn holds(&self, key: SubstateKey) -> BTreeMap<TxHash, u128> {
        self.holds.get(&key).cloned().unwrap_or_default()
    }

    fn into_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync> {
        self
    }
}

/// The VM engine: the genesis-static world, the compiled stdlib guests,
/// and the batch scheduling mode.
pub struct Executor {
    pub(crate) world: World,
    pub(crate) backend: EngineBackend,
    pub(crate) mode: ExecutionMode,
}

impl Executor {
    /// Build the engine for the genesis-funded `accounts` and install the
    /// process-wide VM statics (first installation wins, so co-hosted
    /// nodes sharing one genesis coexist).
    ///
    /// # Panics
    ///
    /// Panics if the committed stdlib artifact fails validation or
    /// compilation — a build defect surfaced at boot, not in a wave.
    #[must_use]
    pub fn new(accounts: &[([u8; 16], u128)], mode: ExecutionMode) -> Self {
        Self::with_pools(accounts, &[], mode)
    }

    /// [`Self::new`] seating `pools` as the stake pools the beacon folds
    /// for: `(instance address, the identifier it is folded under)`.
    ///
    /// # Panics
    ///
    /// As [`Self::new`].
    #[must_use]
    pub fn with_pools(
        accounts: &[([u8; 16], u128)],
        pools: &[StakePoolSeat],
        mode: ExecutionMode,
    ) -> Self {
        let world = genesis_world_with_pools(accounts, pools);
        install_vm_statics(Box::new(BridgeStatics {
            cache: world.cache.clone(),
            instances: world.instances.clone(),
        }));
        Self {
            world,
            backend: EngineBackend::new(),
            mode,
        }
    }

    /// The published-package cache this engine routes against.
    ///
    /// Shared with the installed statics rather than copied, so a package
    /// a committed block publishes is visible to admission and to
    /// execution at the same instant.
    #[must_use]
    pub const fn packages(&self) -> &PackageCache {
        &self.world.cache
    }

    /// Derive one transaction's invocations, effect set, and nullifiers
    /// — the same `decode → admit → route` admission ran; refusal here
    /// means the transaction bypassed admission and fails
    /// deterministically.
    pub(crate) fn prepare(&self, tx: &Transaction) -> Result<PreparedTx, String> {
        self.prepare_with_authority(tx, TargetAuthority::Required)
    }

    /// [`Self::prepare`] with the target-authority rule made optional.
    ///
    /// Only a preview waives it, and only when its caller asked to be
    /// shown what an envelope would do before its counterparties have
    /// signed. Nothing on the commit path reaches this with
    /// [`TargetAuthority::Assumed`].
    pub(crate) fn prepare_with_authority(
        &self,
        tx: &Transaction,
        authority: TargetAuthority,
    ) -> Result<PreparedTx, String> {
        let vm = tx.body();
        let packages = self.world.cache.load();
        let tree = decode_tree(
            vm.call_tree()
                .ok_or_else(|| "publish body in a call sub-batch".to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if authority == TargetAuthority::Required {
            check_target_authority(&tree, vm.fee_payer, &packages, &self.world.instances)
                .map_err(|error| error.0)?;
        }
        let admitted = admit_tree(
            &tree,
            envelope_identity(vm),
            &packages,
            &self.world.instances,
            &ProtocolHasher,
        )
        .map_err(|error| format!("admission: {error}"))?;
        let routing = route_tree(
            &admitted,
            &packages,
            &self.world.instances,
            &ProtocolHasher,
            &PrefixShardResolver { bits: 0 },
        )
        .map_err(|error| format!("routing: {error}"))?;
        // Both views of the declaration, straight from the fold: the
        // folded set that scheduling and judging read, and the clause
        // order capability materialization walks. Unioning `per_shard`
        // here would reach the same set but discard the order, which is
        // what a guest's positional handle parameters are indexed by.
        let declaration = routing
            .declaration()
            .map_err(|error| format!("declaration: {error:?}"))?;
        Ok(PreparedTx {
            calls: routing.calls,
            declaration,
            nullifiers: admitted
                .subintents
                .iter()
                .map(|record| record.nullifier)
                .collect(),
            gas_limit: vm.gas_limit,
        })
    }
}

/// Fuel and the abort reason (if any) as node-local metadata.
/// How an abort reads in a diagnostic: the kernel's own verdict, or the
/// deterministic text a trap carried.
///
/// One derivation, because a preview quotes the same verdict a wave
/// records and two copies would drift apart silently. Never
/// consensus-critical — it lands in the node-local metadata, which a
/// syncing replica does not carry.
///
/// # Panics
///
/// Panics on a completed outcome, which is not an abort.
pub fn abort_reason(outcome: &Outcome) -> String {
    match outcome {
        Outcome::UserError { reason } | Outcome::ProtocolError { reason } => reason.clone(),
        Outcome::Infeasible { key, amount } => format!("infeasible: {amount} uncovered on {key:?}"),
        Outcome::ConstraintUnmet {
            node,
            param,
            amount,
        } => format!("constraint unmet: node {node} parameter {param} carried {amount}"),
        Outcome::NullifierSpent { key } => format!("subintent already spent at {key:?}"),
        Outcome::Completed { .. } => unreachable!("aborts only"),
    }
}

fn vm_metadata(fuel: u64, error: Option<String>) -> ExecutionMetadata {
    ExecutionMetadata::new(
        FeeSummary {
            total_execution_cost: Some(u128::from(fuel) * Stake::ATTOS_PER_WHOLE),
            total_royalty_cost: None,
            total_storage_cost: None,
            total_tipping_cost: None,
        },
        Vec::new(),
        error,
    )
}

/// Assemble one kernel receipt into the projected [`ExecutedTx`]: fold
/// its delta, root its writes, and run the shard projection. Aborts
/// carry their reason and fuel in the node-local metadata.
/// Apply the payer's fee burn on top of a transaction's kernel-mirroring
/// fold. The burn is part of the receipt's writes — and so of its
/// attested `writes_root` and the sync-replayable work items — while the
/// pre-fee `running` map stays the kernel differential's source: the
/// applied value of a fee-bearing cell is always
/// `saturating_sub(pre-fee value, cumulative fees)`.
fn apply_fee_burn(
    writes: &mut BTreeMap<SubstateKey, Option<Vec<u8>>>,
    running: &BTreeMap<SubstateKey, Option<Vec<u8>>>,
    base: &VmBase,
    fees_applied: &mut BTreeMap<SubstateKey, u128>,
    fee: Option<PayerFee>,
    fuel: u64,
) {
    // The transaction's own burn first: the attested actual — fuel, until
    // real pricing lands — capped at the signed ceiling.
    let mut own_vault = None;
    if let Some(payer) = fee {
        let burn = u128::from(fuel).min(payer.max_fee);
        if burn > 0 {
            *fees_applied.entry(payer.vault).or_insert(0) += burn;
            own_vault = Some(payer.vault);
        }
    }
    // Re-derive the fee-bearing cells this transaction itself carries —
    // its own payer's vault, plus any earlier-debited vault its update
    // set covers — from the pre-fee fold: a write of a debited cell must
    // carry the cumulative burn, or its absolute update would revert
    // earlier debits at commit. Cells only *other* transactions debited
    // stay out: a receipt is one transaction's effect record, and
    // stamping a sibling's vault into it would put a write in it the
    // transaction never made.
    let touched: Vec<SubstateKey> = writes
        .keys()
        .filter(|key| fees_applied.contains_key(*key))
        .copied()
        .chain(own_vault.filter(|vault| !writes.contains_key(vault)))
        .collect();
    for vault in touched {
        let fees = fees_applied[&vault];
        let prefee = writes
            .get(&vault)
            .cloned()
            .or_else(|| running.get(&vault).cloned())
            .unwrap_or_else(|| base.cells.get(&vault).cloned());
        let Some(bytes) = prefee else {
            continue;
        };
        let Ok(cell): Result<[u8; 16], _> = bytes.as_slice().try_into() else {
            continue;
        };
        let debited = u128::from_le_bytes(cell).saturating_sub(fees);
        // The burn folds outside the kernel store, so it applies the
        // store's own rule itself: a zero balance is an absent cell, and
        // the leaf goes with the bond it carried.
        writes.insert(vault, amount_cell(debited).map(|cell| cell.to_vec()));
    }
}

/// What this shard, as a transaction's fee payer, charges it.
#[derive(Clone, Copy)]
pub struct PayerFee {
    pub vault: SubstateKey,
    /// The signed ceiling a success burns up to, and what the sender's
    /// own defect costs.
    pub max_fee: u128,
    /// The class floor: what an attempt owes when nothing it did was its
    /// sender's fault.
    pub floor: u128,
    /// Whether a wave can abort this transaction after it executed —
    /// true for a cross-shard leg, which is the one shape whose effects
    /// are discarded after the engine completed them.
    pub wave_abortable: bool,
}

/// What an attempt that applied no effects owes, by why it applied none.
///
/// Charging nothing is not an option: a transaction that consumed its
/// limit and then trapped would cost its sender less than the same work
/// succeeding, which is the inversion that makes failure the cheaper way
/// to buy execution.
///
/// The consumed work itself cannot price this. Fuel at a trap is not
/// agreed between the runtimes — one flushes its in-register counter
/// while the other charges every executed operator — so a charge derived
/// from it would differ across replicas on the same transaction. Both
/// amounts below are functions of signed content alone.
pub const fn charge_for(outcome: &Outcome, payer: PayerFee) -> Option<u128> {
    match outcome {
        // Completed here means the engine applied the effects. Only a
        // wave can still discard them, and only for a cross-shard leg —
        // that receipt is built in reserve and settles the floor if the
        // abort comes.
        Outcome::Completed { .. } => {
            if payer.wave_abortable {
                Some(payer.floor)
            } else {
                None
            }
        }
        // The sender's own defect, and the only class worth grinding: it
        // pays the ceiling it declared. Not the work consumed — that is
        // unknowable — but the sender chose the bound, and anything less
        // leaves failure discounted against success.
        Outcome::UserError { .. } => Some(payer.max_fee),
        // A lost deterministic race. The sender did nothing wrong and
        // could not have avoided it, so it pays only the floor covering
        // the declaration work its attempt really did consume.
        //
        // A signed edge bound the produced amount missed is the same
        // class: the sender declared what it would accept and the world
        // moved between signing and execution. So is a spent subintent —
        // a conflict tiebreak or a stale declaration, neither of which a
        // composer can see at signing time.
        Outcome::Infeasible { .. }
        | Outcome::ConstraintUnmet { .. }
        | Outcome::NullifierSpent { .. } => Some(payer.floor),
        // The kernel's own defect. `materialize_abort` refuses to price
        // it to the sender, and the burn agrees.
        Outcome::ProtocolError { .. } => None,
    }
}

/// The fold's mutable state across a batch: the pre-fee kernel-mirror
/// map (the differential's source) and the cumulative fee burns layered
/// on top of it.
struct FoldState {
    running: BTreeMap<SubstateKey, Option<Vec<u8>>>,
    fees_applied: BTreeMap<SubstateKey, u128>,
}

/// Build the receipt an abort of this transaction settles: the payer's
/// vault debited by the class floor, and nothing else.
///
/// The value is read as of every canonically earlier transaction's
/// applied effect and fee, but without this transaction's own — an abort
/// discards those, so the burn must not be layered on top of them.
fn build_fee_receipt(
    ctx: &WaveBatchContext<'_>,
    base: &VmBase,
    fold: &FoldState,
    tx_hash: TxHash,
    vault: SubstateKey,
    floor: u128,
) -> Option<ConsensusReceipt> {
    let prefee = fold
        .running
        .get(&vault)
        .cloned()
        .unwrap_or_else(|| base.cells.get(&vault).cloned())?;
    let cell: [u8; 16] = prefee.as_slice().try_into().ok()?;
    let applied = u128::from_le_bytes(cell)
        .saturating_sub(fold.fees_applied.get(&vault).copied().unwrap_or(0));
    let debited = applied.saturating_sub(floor);

    let writes = StateWrites {
        cells: BTreeMap::from([(vault, Some(debited.to_le_bytes().to_vec()))]),
    };
    let receipt_hash = GlobalReceipt::new(
        true,
        EventRoot::ZERO,
        BeaconWitnessRoot::ZERO,
        writes_root(&writes),
    )
    .receipt_hash();
    // No gas: this receipt settles a floor, it does not report execution.
    // The transaction whose abort it settles consumed real work, but that
    // work is unattested — a failed outcome carries no gas either — so an
    // abort contributes nothing to its shard's emission weight. Pricing
    // aborted work is the floor's job, not the weight's.
    let cached = CachedOutput::succeeded(
        writes,
        receipt_hash,
        vm_metadata(0, None),
        0,
        Vec::new(),
        Vec::new(),
    );
    Some(project_to_shard(&cached, tx_hash, ctx.local_shard, ctx.shard_trie).consensus)
}

/// What judging and storing one artifact costs, whatever the verdict:
/// the shard reached it from these bytes before it knew the answer.
///
/// One unit per byte is a placeholder until measured baselines set the
/// real rate, like every other number in the fee model.
pub const fn publish_work(artifact: &[u8]) -> u64 {
    artifact.len() as u64
}

/// Settle one publish: the artifact lands in its content-addressed cell
/// under the publisher, and the fee burns from the publisher's vault.
///
/// A publish never enters the kernel — there is no manifest to run — so
/// it settles outside the batch fold rather than inside it. That is
/// sound because a publish declares exactly two keys, its own package
/// cell and its own fee vault, and both are exclusive: no sibling in the
/// block can be touching either, so there are no earlier burns to layer
/// on and nothing for the kernel differential to check.
fn assemble_published_tx(
    ctx: &WaveBatchContext<'_>,
    base: &VmBase,
    vm_tx: TxHash,
    publisher: [u8; 16],
    artifact: &[u8],
    fee: Option<PayerFee>,
    locality: &Locality,
) -> ExecutedTx {
    let tx_hash = vm_tx;
    let work = publish_work(artifact);

    // Admission reached the whole verdict from these same bytes, so a
    // refusal here means the transaction bypassed admission — the same
    // condition `prepare` treats as a deterministic failure.
    let refusal = admit_package(artifact).err().map(|error| error.0);

    let mut fold = FoldState {
        running: BTreeMap::new(),
        fees_applied: BTreeMap::new(),
    };
    let cached = if let Some(reason) = &refusal {
        CachedOutput::failed(vm_metadata(work, Some(reason.clone())))
    } else {
        {
            let mut writes = StateWrites::default();
            if locality.is_local(Address(publisher)) {
                let package = package_hash(&ProtocolHasher, artifact);
                // Content-addressed, so republishing the same artifact
                // writes the same bytes to the same cell: idempotent by
                // construction rather than by a first-write-wins branch.
                writes
                    .cells
                    .insert(package_key(publisher, package), Some(artifact.to_vec()));
            }
            apply_fee_burn(
                &mut writes.cells,
                &fold.running,
                base,
                &mut fold.fees_applied,
                fee,
                work,
            );
            let receipt_hash = GlobalReceipt::new(
                true,
                EventRoot::ZERO,
                BeaconWitnessRoot::ZERO,
                writes_root(&writes),
            )
            .receipt_hash();
            CachedOutput::succeeded(
                writes,
                receipt_hash,
                vm_metadata(work, None),
                work,
                Vec::new(),
                Vec::new(),
            )
        }
    };
    // A refused artifact is the sender's own defect — they chose what to
    // publish — so it pays the ceiling, exactly as a trap does. Charging
    // less would leave a rejected publish cheaper than an accepted one.
    let fee_receipt = match (&refusal, fee) {
        (Some(_), Some(payer)) => {
            build_fee_receipt(ctx, base, &fold, tx_hash, payer.vault, payer.max_fee)
        }
        _ => None,
    };

    let mut executed = project_to_shard(&cached, tx_hash, ctx.local_shard, ctx.shard_trie);
    executed.fee_receipt = fee_receipt;
    executed.attested_work = work;
    executed
}

/// What the kernel reported for one transaction: the effect record every
/// participant derives identically, and this shard's own attested share.
#[derive(Clone, Copy)]
struct KernelOutput<'a> {
    receipt: &'a Receipt,
    work: u64,
}

/// What every transaction in a batch assembles against: the pre-read
/// baseline its receipts fold over, the share of the world this shard
/// applies, and what the witness lift needs to decide whether an emitted
/// event is a beacon fact — the pools the network recognises, what code
/// each instance runs, and the code a pool must be running.
#[derive(Clone, Copy)]
struct BatchInputs<'a> {
    base: &'a VmBase,
    locality: &'a Locality,
    pools: &'a PoolRegistry,
    instances: &'a InstanceRegistry,
    staking_package: PackageHash,
}

fn assemble_executed_tx(
    ctx: &WaveBatchContext<'_>,
    inputs: BatchInputs<'_>,
    fold: &mut FoldState,
    vm_tx: TxHash,
    kernel: KernelOutput<'_>,
    fee: Option<PayerFee>,
) -> ExecutedTx {
    let BatchInputs { base, locality, .. } = inputs;
    let KernelOutput {
        receipt,
        work: attested_work,
    } = kernel;
    let tx_hash = vm_tx;
    // Built before this transaction's own burn folds in: a charge settles
    // over the state its siblings left, not over its own.
    let fee_receipt = fee.and_then(|payer| {
        let amount = charge_for(&receipt.outcome, payer)?;
        let built = build_fee_receipt(ctx, base, fold, tx_hash, payer.vault, amount)?;
        // A charge that settles unconditionally joins the cumulative
        // burn: a sibling folded later that writes this vault must carry
        // the debit, or its absolute update would revert the charge at
        // commit. The floor a completed cross-shard leg holds in reserve
        // stays out — whether it settles is the wave's verdict, unknown
        // at fold time.
        if !matches!(receipt.outcome, Outcome::Completed { .. }) {
            *fold.fees_applied.entry(payer.vault).or_insert(0) += amount;
        }
        Some(built)
    });
    let cached = if matches!(receipt.outcome, Outcome::Completed { .. }) {
        // The kernel's own flatten: this receipt's owned part folded to
        // absolute cells, reading through the batch's running state to
        // the pre-read base.
        let mut writes = receipt.delta.flatten(locality, &mut |key| {
            fold.running
                .get(&key)
                .map_or_else(|| base.cells.get(&key).cloned(), Clone::clone)
        });
        // The pre-fee fold is the kernel differential's source: update it
        // before the burn layers on.
        for (key, change) in &writes.cells {
            fold.running.insert(*key, change.clone());
        }
        apply_fee_burn(
            &mut writes.cells,
            &fold.running,
            base,
            &mut fold.fees_applied,
            fee,
            receipt.fuel,
        );
        // Every participant derives the same events from the same
        // manifest, so the root covers the whole union while each shard's
        // receipt keeps only what its own instances emitted.
        // The kernel's record is the wire record — one shared type, so
        // there is nothing to convert.
        let events: Vec<Event> = receipt.events.clone();
        // The beacon facts among them. Read here rather than at
        // projection because this is where the world that decides is in
        // reach, and read from the whole union rather than one shard's
        // share so every participant derives the same set — which shard
        // keeps a fact is settled once, at projection, by the same rule
        // that settles which shard keeps the event.
        let witnesses: Vec<([u8; 16], BeaconWitnessEvent)> = events
            .iter()
            .filter_map(|event| {
                witness_from_event(
                    event,
                    inputs.pools,
                    inputs.instances,
                    inputs.staking_package,
                )
                .map(|witness| (event.emitter.0, witness))
            })
            .collect();
        let event_hashes: Vec<Hash> = events.iter().map(EventExt::hash).collect();
        let receipt_hash = GlobalReceipt::new(
            true,
            EventRoot::from_raw(compute_merkle_root(&event_hashes)),
            BeaconWitnessRoot::ZERO,
            writes_root(&writes),
        )
        .receipt_hash();
        CachedOutput::succeeded(
            writes,
            receipt_hash,
            vm_metadata(receipt.fuel, None),
            receipt.fuel,
            events,
            witnesses,
        )
    } else {
        CachedOutput::failed(vm_metadata(
            receipt.fuel,
            Some(abort_reason(&receipt.outcome)),
        ))
    };
    let mut executed = project_to_shard(&cached, tx_hash, ctx.local_shard, ctx.shard_trie);
    executed.fee_receipt = fee_receipt;
    executed.attested_work = attested_work;
    executed
}

impl Executor {
    /// The batch pipeline every dispatch arm shares: derive, pre-read the
    /// local baseline, layer provisioned remote cells, execute under the
    /// shard's locality, fold local keys, and project. `abortable` names
    /// the members a wave verdict can still discard — the cross-shard
    /// legs; a batch without any executes under total locality.
    #[allow(clippy::too_many_lines)] // one pipeline, stages in order
    fn run_batch(
        &self,
        ctx: &WaveBatchContext<'_>,
        snapshot: &(dyn SubstateDatabase + Sync),
        transactions: &[Arc<Verified<Transaction>>],
        provisions_by_tx: &BTreeMap<TxHash, Vec<Arc<Vec<SubstateEntry>>>>,
        env_by_tx: &BTreeMap<TxHash, EnvInputs>,
        abortable: &BTreeSet<TxHash>,
    ) -> Vec<ExecutedTx> {
        if transactions.is_empty() {
            return Vec::new();
        }
        // A cross-shard leg declares remote cells, so its writes must be
        // filtered to the local subtree; a batch of genuinely single-shard
        // members owns every key it declares and total locality is the
        // same filter without the trie walk.
        let locality = if abortable.is_empty() {
            Locality::All
        } else {
            let trie = ctx.shard_trie.clone();
            let local_shard = ctx.local_shard;
            Locality::Owned(Arc::new(move |owner: Address| {
                trie.shard_for_prefix(owner) == local_shard
            }))
        };
        // Publishes carry no manifest, so they never reach the kernel;
        // they settle in their own pass below.
        let publishes: BTreeMap<TxHash, ([u8; 16], Vec<u8>)> = transactions
            .iter()
            .filter_map(|tx| {
                let vm = tx.body();
                let artifact = vm.artifact()?;
                Some((tx.hash(), (vm.fee_payer.0, artifact.to_vec())))
            })
            .collect();

        // Derive every transaction; refusals become deterministic
        // failures without touching the batch.
        let mut prepared: BTreeMap<TxHash, PreparedTx> = BTreeMap::new();
        let mut refused: BTreeMap<TxHash, String> = BTreeMap::new();
        for tx in transactions {
            let vm_tx = tx.hash();
            if publishes.contains_key(&vm_tx) {
                continue;
            }
            match self.prepare(tx) {
                Ok(entry) => {
                    record_transaction_executed();
                    prepared.insert(vm_tx, entry);
                }
                Err(reason) => {
                    tracing::warn!(tx_hash = ?tx.hash(), reason, "VM transaction refused at execution");
                    refused.insert(tx.hash(), reason);
                }
            }
        }

        // The committed baseline: provisioned remote cells first — a
        // key's owner prefix routes it to exactly one source, so nothing
        // arbitrates — then the locally owned declared cells from the
        // wave snapshot.
        let mut cells: BTreeMap<SubstateKey, Vec<u8>> = BTreeMap::new();
        for lists in provisions_by_tx.values() {
            for entries in lists {
                for entry in entries.iter() {
                    if let Some(value) = entry.value.as_ref() {
                        cells.insert(entry.key, value.clone());
                    }
                }
            }
        }
        for entry in prepared.values() {
            for effect in entry.declaration.set.iter() {
                if let EffectTarget::Point(key) = effect.target
                    && locality.is_local(key.owner)
                    && let Some(value) = snapshot.substate(key)
                {
                    cells.insert(key, value);
                }
            }
        }
        // A manifest needn't touch its payer's own vault — a publish
        // declares no effects at all, and a call can spend entirely from
        // other parties' cells. The burn re-derives its debit from this
        // baseline, so every collectible payer's vault joins it: an
        // absent baseline value is indistinguishable from an absent
        // cell, and the burn would silently apply to nothing.
        //
        // Collectible means the vault routes to the executing shard by
        // the trie, not by wave locality: a single-shard wave's
        // `Locality::All` claims every owner, and reading another
        // shard's cell out of this shard's store is nondeterministic —
        // members disagree on what they hold outside their own subtree,
        // and a split baseline splits the wave's receipt roots.
        for tx in transactions {
            let key = tx.fee_vault();
            if ctx.shard_trie.shard_for_prefix(key.owner) == ctx.local_shard
                && let Some(value) = snapshot.substate(key)
            {
                cells.insert(key, value);
            }
        }
        // Trie-routed like the pre-read above, and for the same reason:
        // a declaration spans every participating shard, so the holds it
        // implies do too, and a shard that reported one against a cell it
        // holds none of would judge a reservation as exceeding a balance
        // it cannot see. A wave's own locality cannot decide this — the
        // single-shard arm's `Locality::All` claims every owner.
        let holds = ctx
            .holds
            .iter()
            .filter(|(key, _)| ctx.shard_trie.shard_for_prefix(key.owner) == ctx.local_shard)
            .map(|(key, held)| (*key, held.clone()))
            .collect();
        let base = Arc::new(VmBase { cells, holds });

        let batch: Vec<BatchTx> = prepared
            .iter()
            .map(|(vm_tx, entry)| {
                // Total: both dispatch arms build the map from the same
                // transactions the derivation ran over, and `prepared` is
                // a subset of those.
                let env = env_by_tx
                    .get(vm_tx)
                    .copied()
                    .expect("every prepared transaction has an environment");
                BatchTx::new(
                    *vm_tx,
                    entry.declaration.clone(),
                    env.clock_ms,
                    env.randomness,
                )
                .with_calls(entry.calls.clone())
                .with_nullifiers(entry.nullifiers.clone())
                .with_gas_limit(entry.gas_limit)
            })
            .collect();
        let walk = ManifestWalk {
            backend: &self.backend,
        };
        let outcome = execute_batch(
            Arc::clone(&base) as Arc<dyn Base>,
            &batch,
            &walk,
            protocol_hash,
            self.mode,
            &locality,
        )
        .unwrap_or_else(|error| panic!("BFT CRITICAL: VM batch execution failed: {error}"));

        // Fold receipts into per-transaction absolute updates in
        // canonical order, then check the folded end state against the
        // kernel's own applied store — the fold must be the same fold.
        // The fee payers this shard settles: a completed transaction
        // burns its attested actual from its payer's vault, on the
        // payer's shard only.
        // Trie-routed, like the pre-read: a wave's own locality cannot
        // decide fee ownership, because the single-shard arm's
        // `Locality::All` would claim payers whose vaults live on
        // shards this wave never engaged.
        let fee_by_tx: BTreeMap<TxHash, PayerFee> = transactions
            .iter()
            .filter_map(|tx| {
                let vm = tx.body();
                let vault = tx.fee_vault();
                if ctx.shard_trie.shard_for_prefix(vault.owner) != ctx.local_shard {
                    return None;
                }
                Some((
                    tx.hash(),
                    PayerFee {
                        vault,
                        max_fee: vm.max_fee,
                        floor: vm.abort_floor(),
                        wave_abortable: abortable.contains(&tx.hash()),
                    },
                ))
            })
            .collect();

        let mut fold = FoldState {
            running: BTreeMap::new(),
            fees_applied: BTreeMap::new(),
        };
        let mut folded: BTreeMap<TxHash, ExecutedTx> = BTreeMap::new();
        for (vm_tx, receipt) in &outcome.receipts {
            // The kernel priced this shard's share under the same locality
            // the receipts were applied through, so nothing here re-derives
            // it — a workspace-side filter would be a second opinion on a
            // quantity that must agree.
            let kernel = KernelOutput {
                receipt,
                work: outcome.work.get(vm_tx).map_or(0, |w| w.units),
            };
            let executed = assemble_executed_tx(
                ctx,
                BatchInputs {
                    base: &base,
                    locality: &locality,
                    pools: &self.world.pools,
                    instances: &self.world.instances,
                    staking_package: self.world.staking_package,
                },
                &mut fold,
                *vm_tx,
                kernel,
                fee_by_tx.get(vm_tx).copied(),
            );
            folded.insert(*vm_tx, executed);
        }

        for (vm_tx, (publisher, artifact)) in &publishes {
            folded.insert(
                *vm_tx,
                assemble_published_tx(
                    ctx,
                    &base,
                    *vm_tx,
                    *publisher,
                    artifact,
                    fee_by_tx.get(vm_tx).copied(),
                    &locality,
                ),
            );
        }

        // The differential: every folded key's end value must equal the
        // kernel's applied store. A mismatch is a fold defect — receipts
        // silently diverging from kernel semantics — and must never ship.
        for (key, change) in &fold.running {
            let applied = Base::cell(&outcome.store, *key);
            assert_eq!(
                change.as_ref(),
                applied.as_ref(),
                "BFT CRITICAL: VM fold diverged from the kernel apply at {key:?}"
            );
        }

        // Reassemble in input order.
        transactions
            .iter()
            .map(|tx| {
                let vm_tx = tx.hash();
                folded.get(&vm_tx).cloned().unwrap_or_else(|| {
                    let reason = refused
                        .get(&tx.hash())
                        .cloned()
                        .unwrap_or_else(|| "missing batch receipt".to_string());
                    let cached = CachedOutput::failed(vm_metadata(0, Some(reason)));
                    project_to_shard(&cached, tx.hash(), ctx.local_shard, ctx.shard_trie)
                })
            })
            .collect()
    }
}

impl Executor {
    /// Execute `transactions` against `snapshot` and project each result
    /// to the context's local shard.
    ///
    /// The unit is the batch: the whole of it goes to the
    /// deterministic-parallel executor at once, which returns one
    /// [`ExecutedTx`] per input transaction, in input order.
    #[must_use]
    pub fn execute_wave_batch(
        &self,
        ctx: &WaveBatchContext<'_>,
        snapshot: &(dyn SubstateDatabase + Sync),
        transactions: &[Arc<Verified<Transaction>>],
    ) -> Vec<ExecutedTx> {
        // A single-shard batch commits in one block, so every member's
        // environment is anchored on the wave-start block.
        let env_by_tx: BTreeMap<TxHash, EnvInputs> = transactions
            .iter()
            .map(|tx| {
                (
                    tx.hash(),
                    EnvInputs {
                        clock_ms: ctx.wave_start_ts.as_millis(),
                        randomness: tx_randomness(ctx.wave_start_reveal, tx.hash()),
                    },
                )
            })
            .collect();
        self.run_batch(
            ctx,
            snapshot,
            transactions,
            &BTreeMap::new(),
            &env_by_tx,
            &BTreeSet::new(),
        )
    }

    /// Execute a cross-shard sub-batch: `snapshot` carries local state,
    /// each request its remote provisions. One [`ExecutedTx`] per input,
    /// in input order, projected to the context's local shard.
    #[must_use]
    pub fn execute_cross_shard_batch(
        &self,
        ctx: &WaveBatchContext<'_>,
        snapshot: &(dyn SubstateDatabase + Sync),
        requests: &[CrossShardTxInput<'_>],
    ) -> Vec<ExecutedTx> {
        let transactions: Vec<Arc<Verified<Transaction>>> =
            requests.iter().map(|r| Arc::clone(r.transaction)).collect();
        let provisions_by_tx: BTreeMap<TxHash, Vec<Arc<Vec<SubstateEntry>>>> = requests
            .iter()
            .map(|r| (r.transaction.hash(), r.provisions.to_vec()))
            .collect();
        // Each request carries the environment its payer block fixed:
        // remote-payer legs the anchors off the payer's bundle, everything
        // else the wave-start block's own.
        let env_by_tx: BTreeMap<TxHash, EnvInputs> = requests
            .iter()
            .map(|r| {
                (
                    r.transaction.hash(),
                    EnvInputs {
                        clock_ms: r.clock.as_millis(),
                        randomness: tx_randomness(r.randomness, r.transaction.hash()),
                    },
                )
            })
            .collect();
        let abortable: BTreeSet<TxHash> = transactions.iter().map(|tx| tx.hash()).collect();
        self.run_batch(
            ctx,
            snapshot,
            &transactions,
            &provisions_by_tx,
            &env_by_tx,
            &abortable,
        )
    }

    /// Execute one tick's whole batch — single-shard members beside
    /// cross-shard legs — against `snapshot`. One [`ExecutedTx`] per
    /// input, in input order, projected to the context's local shard.
    ///
    /// Batch composition is consensus input: conflict grouping and the
    /// fold run over the whole tick, so every replica must compose the
    /// same members in the same tick.
    #[must_use]
    pub fn execute_tick_batch(
        &self,
        ctx: &WaveBatchContext<'_>,
        snapshot: &(dyn SubstateDatabase + Sync),
        inputs: &[TickTxInput<'_>],
    ) -> Vec<ExecutedTx> {
        let transactions: Vec<Arc<Verified<Transaction>>> =
            inputs.iter().map(|i| Arc::clone(i.transaction)).collect();
        let provisions_by_tx: BTreeMap<TxHash, Vec<Arc<Vec<SubstateEntry>>>> = inputs
            .iter()
            .filter(|i| !i.provisions.is_empty())
            .map(|i| (i.transaction.hash(), i.provisions.to_vec()))
            .collect();
        let env_by_tx: BTreeMap<TxHash, EnvInputs> = inputs
            .iter()
            .map(|i| {
                (
                    i.transaction.hash(),
                    EnvInputs {
                        clock_ms: i.clock.as_millis(),
                        randomness: tx_randomness(i.randomness, i.transaction.hash()),
                    },
                )
            })
            .collect();
        let abortable: BTreeSet<TxHash> = inputs
            .iter()
            .filter(|i| i.wave_abortable)
            .map(|i| i.transaction.hash())
            .collect();
        self.run_batch(
            ctx,
            snapshot,
            &transactions,
            &provisions_by_tx,
            &env_by_tx,
            &abortable,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reveal(seed: &[u8]) -> RevealChain {
        RevealChain::from_raw(Hash::from_bytes(seed))
    }

    fn tx(seed: &[u8]) -> TxHash {
        TxHash::from(Hash::from_bytes(seed))
    }

    /// The draw is a pure function of the payer block's reveal chain and
    /// the transaction hash: participants agreeing on the anchor derive
    /// one draw, and two transactions under one anchor derive two.
    #[test]
    fn a_draw_is_fixed_by_its_anchor_and_transaction() {
        let anchor = reveal(b"payer block");
        assert_eq!(
            tx_randomness(anchor, tx(b"a")),
            tx_randomness(anchor, tx(b"a"))
        );
        assert_ne!(
            tx_randomness(anchor, tx(b"a")),
            tx_randomness(anchor, tx(b"b"))
        );
        assert_ne!(
            tx_randomness(anchor, tx(b"a")),
            tx_randomness(reveal(b"another block"), tx(b"a"))
        );
    }
}
