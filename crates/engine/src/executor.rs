//! The VM engine's tick-batch executor.
//!
//! `execute_tick_batch` runs one tick's VM sub-batch end to end: derive
//! each transaction's manifest and effect set through the bridge
//! (exactly the derivation admission ran), pre-read the declared cells
//! from the tick snapshot into an owned committed base, hand the batch
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
use hyperscale_effects_bridge::vm_statics::{
    PackageCache, config_key, package_key, principal_for, record_address,
};
use hyperscale_effects_bridge::{
    BridgeStatics, PoolRegistry, ProtocolHasher, admit_package, decode_tree, envelope_identity,
    witness_from_event,
};
use hyperscale_metrics::record_transaction_executed;
use hyperscale_storage::entry_from_leaf;
use hyperscale_types::{
    BeaconWitnessEvent, BeaconWitnessRoot, ConsensusReceipt, Derivation, Event, EventExt,
    EventRoot, ExecutionMetadata, FeeSummary, GlobalReceipt, Hash, Movement, PrincipalAddr,
    ProvisionalHolds, RevealChain, Stake, StakePoolSeat, StateWrites, SubstateEntry, Transaction,
    TxHash, Verified, compute_merkle_root, install_protocol_statics,
};
use hyperscale_vm_effects::{
    Declaration, InstanceRegistry, NodeCall, PackageHash, PrefixShardResolver, admit_tree,
    package_hash, route_tree,
};
use hyperscale_vm_kernel::{
    Baseline, BatchTx, EnvInputs, ExecutionMode, Locality, ManifestWalk, Receipt, Substates,
    execute_batch,
};
use hyperscale_vm_types::{
    Address, CallTarget, CollectionId, EffectSet, EffectTarget, EntryKey, Outcome, SubstateKey,
    UnmetCondition,
};

use crate::backend::EngineBackend;
use crate::genesis::{GenesisPackages, World, genesis_world_with_pools};
use crate::sharding::writes_root;
use crate::{CachedOutput, ExecutedTx, TickBatchContext, TickTxInput, project_to_shard};

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

/// The component address a record's own contents derive, or `None` for
/// bytes that are not a record.
///
/// What verifies a fetched record: the address is the hash of the
/// record, so bytes claiming to be one either derive the address asked
/// for or derive some other and are dropped.
#[must_use]
pub fn instance_of_record(record: &[u8]) -> Option<Address> {
    record_address(record)
}

/// The content address of a package artifact, as the workspace hash the
/// beacon registry and fetch protocol carry.
#[must_use]
pub fn artifact_package(artifact: &[u8]) -> Hash {
    Hash::from(package_hash(&ProtocolHasher, artifact).0)
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

/// The batch's committed baseline: the declared content pre-read from
/// the tick's JMT-backed snapshot at materialize time — point cells and
/// the entries of declared collection intervals.
///
/// Every kernel read flows through a capability for a declared effect,
/// so pre-reading exactly the declared targets is complete. An
/// instance's configuration leaf is one of them: committed state the
/// fence on every method reads, immutable by its one-way write door
/// rather than by a lock, so nothing here is locked.
#[derive(Debug, Default)]
pub struct TickBaseline {
    pub cells: BTreeMap<SubstateKey, Vec<u8>>,
    /// The committed entries of every declared collection interval,
    /// keyed by entry identity — [`materialize_declared`] fills it, and
    /// the kernel's range capabilities read through it.
    pub entries: BTreeMap<EntryKey, Vec<u8>>,
    /// What legs of unresolved ticks hold against these cells. Empty for
    /// a baseline with nothing in flight over it — a preview, or a shard
    /// with no cross-shard leg outstanding.
    pub holds: ProvisionalHolds,
}

impl Substates for TickBaseline {
    fn cell(&self, key: SubstateKey) -> Option<Vec<u8>> {
        self.cells.get(&key).cloned()
    }

    fn entries_in_range(
        &self,
        owner: Address,
        collection: CollectionId,
        lo: u128,
        hi: u128,
        limit: usize,
    ) -> Vec<(u128, Vec<u8>)> {
        if lo > hi {
            return Vec::new();
        }
        let lo_key = EntryKey {
            owner,
            collection,
            order: lo,
        };
        let hi_key = EntryKey {
            owner,
            collection,
            order: hi,
        };
        self.entries
            .range(lo_key..=hi_key)
            .take(limit)
            .map(|(key, value)| (key.order, value.clone()))
            .collect()
    }
}

impl Baseline for TickBaseline {
    fn is_locked(&self, _key: SubstateKey) -> bool {
        false
    }

    fn holds(&self, key: SubstateKey) -> BTreeMap<TxHash, u128> {
        self.holds.get(&key).cloned().unwrap_or_default()
    }
}

/// Materialize one declaration's baseline from the tick snapshot: for
/// every declared effect this shard owns, the committed content it
/// reads — the point cell, or the entries of the collection interval.
///
/// The one pre-read: `run_batch` and `preview` both fill their baselines
/// here, so ordered-collection support exists in exactly one place and
/// the preview cannot drift from execution. The match is exhaustive over
/// the target vocabulary — a new target form fails here at compile time
/// rather than executing against a silently empty baseline.
pub fn materialize_declared(
    snapshot: &(dyn Substates + Sync),
    declared: &EffectSet,
    locality: &Locality,
    base: &mut TickBaseline,
) {
    for effect in declared.iter() {
        // An entry is its width-one interval — the same normalization
        // `admission_key` applies, so the two vocabulary walks stay
        // parallel.
        let (owner, collection, lo, hi, cap) = match effect.target {
            EffectTarget::Point(key) => {
                if locality.is_local(key.owner)
                    && let Some(value) = snapshot.cell(key)
                {
                    base.cells.insert(key, value);
                }
                continue;
            }
            EffectTarget::Entry {
                owner,
                collection,
                order,
            } => (owner, collection, order, order, 1),
            EffectTarget::Range {
                owner,
                collection,
                lo,
                hi,
                cap,
            } => (owner, collection, lo, hi, cap),
        };
        if locality.is_local(owner) {
            for (order, value) in snapshot.entries_in_range(owner, collection, lo, hi, cap as usize)
            {
                base.entries.insert(
                    EntryKey {
                        owner,
                        collection,
                        order,
                    },
                    value,
                );
            }
        }
    }
}

/// The VM engine: the genesis-static world, the compiled stdlib guests,
/// and the batch scheduling mode.
pub struct Executor {
    pub(crate) world: World,
    pub(crate) backend: EngineBackend,
    pub(crate) mode: ExecutionMode,
    derivation: Arc<BridgeStatics>,
}

impl Executor {
    /// Build the engine, its derivation, and the process-wide protocol
    /// answers (first installation wins, so co-hosted nodes sharing one
    /// genesis coexist).
    ///
    /// # Panics
    ///
    /// Panics if the committed stdlib artifact fails validation or
    /// compilation — a build defect surfaced at boot, not in a tick.
    #[must_use]
    pub fn new(mode: ExecutionMode) -> Self {
        Self::with_genesis(&[], &GenesisPackages::protocol(), mode)
    }

    /// [`Self::new`] over this network's genesis: `pools` seated as the
    /// stake pools the beacon folds for, and `packages` as the code the
    /// chain is born running.
    ///
    /// # Panics
    ///
    /// As [`Self::new`].
    #[must_use]
    pub fn with_genesis(
        pools: &[StakePoolSeat],
        packages: &GenesisPackages,
        mode: ExecutionMode,
    ) -> Self {
        let world = genesis_world_with_pools(pools, packages);
        let backend = EngineBackend::new(packages);
        // The two halves split on whether a node can answer differently.
        // The derivation reads this engine's caches and is held here, so
        // a process standing several nodes up gives each its own; the
        // protocol answers read nothing a node accumulates, so they are
        // installed once for the process.
        let derivation = Arc::new(BridgeStatics {
            cache: world.cache.clone(),
            instances: world.instances.clone(),
            artifact_sink: Some(Arc::new(backend.absorber())),
        });
        install_protocol_statics(Box::new(BridgeStatics {
            cache: world.cache.clone(),
            instances: world.instances.clone(),
            artifact_sink: None,
        }));
        Self {
            world,
            backend,
            mode,
            derivation,
        }
    }

    /// This engine's derivation: what its node can resolve an envelope
    /// against.
    #[must_use]
    pub fn derivation(&self) -> Arc<dyn Derivation> {
        Arc::clone(&self.derivation) as Arc<dyn Derivation>
    }

    /// A second engine beside this one: its own copy of the world, its
    /// own derivation over that copy, and its own compiled code.
    ///
    /// A harness standing several nodes up in one process uses this so
    /// each of them holds only what it has itself seen — the code a
    /// publish put on the chain and the record an instantiation sealed
    /// alike — the way separate processes would. A node on a shard where
    /// neither committed genuinely cannot answer for them until its own
    /// fetch lands, which is what puts the acquisition paths under test
    /// instead of around them.
    ///
    /// The backend starts at the protocol's own code rather than at this
    /// engine's, so a genesis fixture package is acquired too.
    #[must_use]
    pub fn peer(&self, mode: ExecutionMode) -> Self {
        let world = self.world.fork();
        let backend = EngineBackend::new(&GenesisPackages::protocol());
        let derivation = Arc::new(BridgeStatics {
            cache: world.cache.clone(),
            instances: world.instances.clone(),
            artifact_sink: Some(Arc::new(backend.absorber())),
        });
        Self {
            world,
            backend,
            mode,
            derivation,
        }
    }

    /// The published-package cache this engine routes against.
    ///
    /// Shared with this engine's derivation rather than copied, so a
    /// package a committed block publishes is visible to admission and to
    /// execution at the same instant.
    #[must_use]
    pub const fn packages(&self) -> &PackageCache {
        &self.world.cache
    }

    /// Seed one committed artifact from a store's package index: metadata
    /// into the cache, code into the backend.
    ///
    /// The boot half of what commit-time absorption does live — a
    /// restarted node re-learns its packages from the index its own
    /// commits wrote. The index holds only committed, self-identified
    /// artifacts, so a refusal here means a corrupt store and is skipped
    /// loudly rather than trusted.
    pub fn install_artifact(&self, artifact: &[u8]) {
        match admit_package(artifact) {
            Ok(metadata) => {
                self.world
                    .cache
                    .publish(package_hash(&ProtocolHasher, artifact), metadata);
                self.backend.absorb_artifact(artifact);
            }
            Err(error) => {
                tracing::warn!(
                    reason = %error,
                    "indexed package artifact refused admission"
                );
            }
        }
    }

    /// Whether `package`'s code resolves without waiting — built, or
    /// refused by a build every replica refuses alike.
    #[must_use]
    pub fn package_code_settled(&self, package: PackageHash) -> bool {
        self.backend.code_settled(package)
    }

    /// Whether the artifact behind `package` — named by the workspace
    /// hash the beacon registry carries — is judged or being built.
    #[must_use]
    pub fn package_known(&self, package: Hash) -> bool {
        self.backend.code_known(PackageHash(package.as_hash32()))
    }

    /// The cell a component's record is sealed into — where a node
    /// serving a record fetch reads, and what the address alone names.
    ///
    /// Unlike a package, whose cell sits under whoever published it, a
    /// component's leaf sits under the component itself. So a request
    /// naming an address is a request naming a key, and serving one
    /// needs no index over what a store has committed.
    #[must_use]
    pub fn instance_record_key(instance: Address) -> SubstateKey {
        config_key(instance)
    }

    /// Seat one fetched record under the component it derives.
    ///
    /// The re-derivation is the whole verification, and it happens here
    /// as well as at the response boundary: a component's address is the
    /// hash of its record, so bytes deriving any other address seat
    /// nothing. Idempotent — a record is immutable once sealed, so a
    /// second copy is the same copy.
    pub fn install_instance(&self, instance: Address, record: &[u8]) {
        let _ = self.world.instances.absorb_cell(
            instance,
            Self::instance_record_key(instance).local.0,
            record,
        );
    }

    /// Whether this node answers for the component at `instance`.
    #[must_use]
    pub fn instance_known(&self, instance: Address) -> bool {
        CallTarget::try_from(instance)
            .is_ok_and(|target| self.world.instances.load().get(target).is_some())
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
        // The composer identity is what the signature's own key opens,
        // never the payer field: the payer names who is debited, and
        // whether the payer's rule admits this signer was the payer
        // shard's verdict before the transaction committed.
        let signer = principal_for(vm.signer_scheme, &vm.signer)
            .ok_or_else(|| "the envelope's signer key derives no principal".to_string())?;
        // The records the chain answers with. Admission composes the
        // envelope's own over these itself, and holds each to standing
        // for the seal of the component it derives.
        let admitted = admit_tree(
            &tree,
            signer,
            envelope_identity(vm),
            &packages,
            &self.world.instances.load(),
            &ProtocolHasher,
        )
        .map_err(|error| format!("admission: {error}"))?;
        let routing = route_tree(&admitted, &PrefixShardResolver { bits: 0 });
        // Both views of the declaration, straight from the fold: the
        // folded set that scheduling and judging read, and the clause
        // order capability materialization walks. Unioning `per_shard`
        // here would reach the same set but discard the order, which is
        // what a guest's positional handle parameters are indexed by.
        let declaration = routing.declaration().clone();
        Ok(PreparedTx {
            calls: match authority {
                TargetAuthority::Required => routing.calls,
                // A preview shown before its counterparties have signed:
                // every guarded call is answered as if whoever it names
                // had presented themselves. The lie is told here and
                // nowhere else, so nothing on the commit path can reach
                // it.
                TargetAuthority::Assumed => routing
                    .calls
                    .into_iter()
                    .map(|call| NodeCall {
                        requires: Vec::new(),
                        ..call
                    })
                    .collect(),
            },
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
/// How an abort reads in a diagnostic: its class, rendered.
///
/// The only place an abort becomes prose, and it is the right one — the
/// metadata is node-local, so nothing downstream of here is hashed or
/// compared between committees.
///
/// One derivation, because a preview quotes the same verdict a tick
/// records and two copies would drift apart silently. Never
/// consensus-critical — it lands in the node-local metadata, which a
/// syncing replica does not carry.
///
/// # Panics
///
/// Panics on a completed outcome, which is not an abort.
pub fn abort_reason(outcome: &Outcome) -> String {
    match outcome {
        Outcome::UserError { reason } | Outcome::ProtocolError { reason } => format!("{reason:?}"),
        Outcome::Infeasible { key, amount } => format!("infeasible: {amount} uncovered on {key:?}"),
        Outcome::ConstraintUnmet {
            node,
            param,
            amount,
        } => format!("constraint unmet: node {node} parameter {param} carried {amount}"),
        Outcome::NullifierSpent { key } => format!("subintent already spent at {key:?}"),
        Outcome::Declined { node, code } => format!("node {node} declined with code {code}"),
        Outcome::ConditionUnmet { condition } => match condition {
            UnmetCondition::Holds { target, required } => {
                format!("leaf of {target:?} is not {required:?} as the condition required")
            }
            UnmetCondition::Satisfies { node } => {
                format!("node {node} presents nothing that satisfies a required rule")
            }
        },
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
/// Debit the payer's vault by what this transaction burned.
///
/// A movement, like every other debit, which is what makes the burn
/// compose with whatever else reaches the vault instead of having to be
/// re-derived into each sibling's absolute. There is nothing cumulative
/// to track and nothing to re-stamp: two transactions burning against
/// one vault each record their own debit and settlement adds them.
fn apply_fee_burn(writes: &mut StateWrites, fee: Option<PayerFee>, fuel: u64) {
    let Some(payer) = fee else {
        return;
    };
    // The attested actual — fuel, until real pricing lands — capped at
    // the signed ceiling.
    let burn = u128::from(fuel).min(payer.max_fee);
    if burn == 0 {
        return;
    }
    let entry = writes.movements.entry(payer.vault).or_default();
    *entry = entry.then(Movement {
        credit: 0,
        debit: burn,
    });
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
    /// Whether a tick can abort this transaction after it executed —
    /// true for a cross-shard leg, which is the one shape whose effects
    /// are discarded after the engine completed them.
    pub abortable: bool,
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
        // tick can still discard them, and only for a cross-shard leg —
        // that receipt is built in reserve and settles the floor if the
        // abort comes.
        Outcome::Completed { .. } => {
            if payer.abortable {
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
        // composer can see at signing time. And so is an unauthorized
        // call: a stored rule can change between signing and execution,
        // so presented authority a target no longer admits is a stale
        // declaration, not a defect. A write whose presence requirement
        // the committed leaf no longer meets joins them on the same
        // reading: a leaf somebody else created or removed is the same
        // event as a rule they changed.
        //
        // A declared refusal joins them and is the one that need not
        // stay: the export returned, so unlike every other abort here
        // its consumed work is agreed between the runtimes and already
        // attested on the receipt, which is what a refundable execution
        // charge would settle against. Until one exists it pays the
        // floor, and the discount a package could engineer by burning
        // its budget before declining is bounded by that transaction's
        // own gas.
        Outcome::Infeasible { .. }
        | Outcome::ConstraintUnmet { .. }
        | Outcome::NullifierSpent { .. }
        | Outcome::ConditionUnmet { .. }
        | Outcome::Declined { .. } => Some(payer.floor),
        // The kernel's own defect. `materialize_abort` refuses to price
        // it to the sender, and the burn agrees.
        Outcome::ProtocolError { .. } => None,
    }
}

/// The fold's mutable state across a batch: the kernel-mirror map a
/// later transaction reads what an earlier one left through, and the
/// differential's source.
///
/// Fees do not appear here. A burn is a debit, a debit is a movement,
/// and a movement needs no baseline — so nothing has to track what
/// siblings have already burned in order to keep a later absolute from
/// reverting it.
struct FoldState {
    running: BTreeMap<SubstateKey, Option<Vec<u8>>>,
    /// The entry mirror beside the cells: exclusive writes with no
    /// movement form, folded only so the differential can hold over
    /// them too.
    running_entries: BTreeMap<EntryKey, Option<Vec<u8>>>,
}

/// Build the receipt an abort of this transaction settles: the payer's
/// vault debited by the class floor, and nothing else.
///
/// A debit rather than a value, so it neither reads nor depends on what
/// the vault held when the transaction ran. An abort discards this
/// transaction's own effects, and the floor lands on whatever its
/// siblings left — which is what a movement means and what an absolute
/// computed here could not have expressed without re-deriving their
/// burns.
fn build_fee_receipt(
    ctx: &TickBatchContext<'_>,
    tx_hash: TxHash,
    vault: SubstateKey,
    floor: u128,
) -> ConsensusReceipt {
    let writes = StateWrites {
        cells: BTreeMap::new(),
        movements: BTreeMap::from([(
            vault,
            Movement {
                credit: 0,
                debit: floor,
            },
        )]),
        entries: BTreeMap::new(),
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
    project_to_shard(&cached, tx_hash, ctx.local_shard, ctx.shard_trie).consensus
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
    ctx: &TickBatchContext<'_>,
    vm_tx: TxHash,
    publisher: PrincipalAddr,
    artifact: &[u8],
    fee: Option<PayerFee>,
    locality: &Locality,
) -> ExecutedTx {
    let tx_hash = vm_tx;
    let work = publish_work(artifact);

    // Admission reached the whole verdict from these same bytes, so a
    // refusal here means the transaction bypassed admission — the same
    // condition `prepare` treats as a deterministic failure.
    let refusal = admit_package(artifact).err().map(|error| error.to_string());

    let cached = refusal.as_ref().map_or_else(
        || {
            let mut writes = StateWrites::default();
            if locality.is_local(publisher) {
                let package = package_hash(&ProtocolHasher, artifact);
                // Content-addressed, so republishing the same artifact
                // writes the same bytes to the same cell: idempotent by
                // construction rather than by a first-write-wins branch.
                writes
                    .cells
                    .insert(package_key(publisher, package), Some(artifact.to_vec()));
            }
            apply_fee_burn(&mut writes, fee, work);
            let receipt_hash = GlobalReceipt::new(
                true,
                EventRoot::ZERO,
                BeaconWitnessRoot::ZERO,
                writes_root(&writes),
            )
            .receipt_hash();
            // The publish becomes a beacon fact: paired with the
            // publisher as emitter, so the shard owning the publisher's
            // prefix — the one that keeps the cell — is the one whose
            // witness stream carries it, exactly once.
            let package = package_hash(&ProtocolHasher, artifact);
            let witnesses = vec![(
                publisher.address(),
                BeaconWitnessEvent::PackagePublished {
                    package: Hash::from(package.0),
                    publisher: publisher.address(),
                },
            )];
            CachedOutput::succeeded(
                writes,
                receipt_hash,
                vm_metadata(work, None),
                work,
                Vec::new(),
                witnesses,
            )
        },
        |reason| CachedOutput::failed(vm_metadata(work, Some(reason.clone()))),
    );
    // A refused artifact is the sender's own defect — they chose what to
    // publish — so it pays the ceiling, exactly as a trap does. Charging
    // less would leave a rejected publish cheaper than an accepted one.
    let fee_receipt = match (&refusal, fee) {
        (Some(_), Some(payer)) => Some(build_fee_receipt(ctx, tx_hash, payer.vault, payer.max_fee)),
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
    base: &'a TickBaseline,
    locality: &'a Locality,
    pools: &'a PoolRegistry,
    instances: &'a InstanceRegistry,
    staking_package: PackageHash,
}

fn assemble_executed_tx(
    ctx: &TickBatchContext<'_>,
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
    let fee_receipt = fee.and_then(|payer| {
        let amount = charge_for(&receipt.outcome, payer)?;
        Some(build_fee_receipt(ctx, tx_hash, payer.vault, amount))
    });
    let cached = if matches!(receipt.outcome, Outcome::Completed { .. }) {
        // What the receipt carries: exclusive writes as absolutes,
        // everything commutative as the movement it was. Unresolved,
        // because the state this lands on is not the state it ran
        // against — that is settlement's question.
        let mut writes = receipt.delta.project(locality);
        // The batch's own fold is the one reader whose baseline really is
        // this one: a later transaction in this tick must see what an
        // earlier one left. It mirrors the kernel's store, so it takes
        // the resolved form and takes it before the burn, which the
        // kernel does not know about.
        let resolved = writes.resolve(&mut |key| {
            fold.running
                .get(&key)
                .map_or_else(|| base.cells.get(&key).cloned(), Clone::clone)
        });
        for (key, change) in resolved.cells() {
            fold.running.insert(*key, change.clone());
        }
        for (key, change) in resolved.entries() {
            fold.running_entries.insert(*key, change.clone());
        }
        apply_fee_burn(&mut writes, fee, receipt.fuel);
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
        let witnesses: Vec<(Address, BeaconWitnessEvent)> = events
            .iter()
            .filter_map(|event| {
                witness_from_event(
                    event,
                    inputs.pools,
                    inputs.instances,
                    inputs.staking_package,
                )
                .map(|witness| (event.emitter, witness))
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
    /// the members a tick verdict can still discard — the cross-shard
    /// legs; a batch without any executes under total locality.
    #[allow(clippy::too_many_lines)] // one pipeline, stages in order
    fn run_batch(
        &self,
        ctx: &TickBatchContext<'_>,
        snapshot: &(dyn Substates + Sync),
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
        let publishes: BTreeMap<TxHash, (PrincipalAddr, Vec<u8>)> = transactions
            .iter()
            .filter_map(|tx| {
                let vm = tx.body();
                let artifact = vm.artifact()?;
                Some((tx.hash(), (vm.fee_payer, artifact.to_vec())))
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
        // arbitrates — then the locally owned declared content from the
        // tick snapshot.
        let mut base = TickBaseline::default();
        for lists in provisions_by_tx.values() {
            for entries in lists {
                for entry in entries.iter() {
                    if let Some(value) = entry.value.as_ref() {
                        // A provisioned leaf is a cell or an
                        // ordered-collection entry, and the leaf says
                        // which: an entry re-derives its own key, and
                        // lands in the interval map the kernel's range
                        // capabilities read.
                        if let Some((entry_key, entry_value)) = entry_from_leaf(entry.key, value) {
                            base.entries.insert(entry_key, entry_value);
                        } else {
                            base.cells.insert(entry.key, value.clone());
                        }
                    }
                }
            }
        }
        for entry in prepared.values() {
            materialize_declared(snapshot, &entry.declaration.set, &locality, &mut base);
        }
        // A manifest needn't touch its payer's own vault — a publish
        // declares no effects at all, and a call can spend entirely from
        // other parties' cells. The burn re-derives its debit from this
        // baseline, so every collectible payer's vault joins it: an
        // absent baseline value is indistinguishable from an absent
        // cell, and the burn would silently apply to nothing.
        //
        // Collectible means the vault routes to the executing shard by
        // the trie, not by tick locality: a local-only tick's
        // `Locality::All` claims every owner, and reading another
        // shard's cell out of this shard's store is nondeterministic —
        // members disagree on what they hold outside their own subtree,
        // and a split baseline splits the tick's receipt roots.
        for tx in transactions {
            let key = tx.fee_vault();
            if ctx.shard_trie.shard_for_prefix(key.owner) == ctx.local_shard
                && let Some(value) = snapshot.cell(key)
            {
                base.cells.insert(key, value);
            }
        }
        // Trie-routed like the pre-read above, and for the same reason:
        // a declaration spans every participating shard, so the holds it
        // implies do too, and a shard that reported one against a cell it
        // holds none of would judge a reservation as exceeding a balance
        // it cannot see. A tick's own locality cannot decide this — the
        // single-shard arm's `Locality::All` claims every owner.
        base.holds = ctx
            .holds
            .iter()
            .filter(|(key, _)| ctx.shard_trie.shard_for_prefix(key.owner) == ctx.local_shard)
            .map(|(key, held)| (*key, held.clone()))
            .collect();
        let base = Arc::new(base);

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
                BatchTx::new(*vm_tx, entry.declaration.clone(), env)
                    .with_calls(entry.calls.clone())
                    .with_nullifiers(entry.nullifiers.clone())
                    .with_gas_limit(entry.gas_limit)
            })
            .collect();
        let walk = ManifestWalk {
            backend: &self.backend,
        };
        let outcome = execute_batch(
            Arc::clone(&base) as Arc<dyn Baseline>,
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
        // Trie-routed, like the pre-read: a tick's own locality cannot
        // decide fee ownership, because the single-shard arm's
        // `Locality::All` would claim payers whose vaults live on
        // shards this tick never engaged.
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
                        abortable: abortable.contains(&tx.hash()),
                    },
                ))
            })
            .collect();

        let mut fold = FoldState {
            running: BTreeMap::new(),
            running_entries: BTreeMap::new(),
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
                    instances: &self.world.instances.load(),
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
            let applied = Substates::cell(&outcome.store, *key);
            assert_eq!(
                change.as_ref(),
                applied.as_ref(),
                "BFT CRITICAL: VM fold diverged from the kernel apply at {key:?}"
            );
        }
        for (key, change) in &fold.running_entries {
            let applied = Substates::entries_in_range(
                &outcome.store,
                key.owner,
                key.collection,
                key.order,
                key.order,
                1,
            )
            .into_iter()
            .next()
            .map(|(_, value)| value);
            assert_eq!(
                change.as_ref(),
                applied.as_ref(),
                "BFT CRITICAL: VM fold diverged from the kernel apply at entry {key:?}"
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
    /// to the context's local shard, all under the context's own
    /// environment.
    ///
    /// The unit is the batch: the whole of it goes to the
    /// deterministic-parallel executor at once, which returns one
    /// [`ExecutedTx`] per input transaction, in input order. The
    /// per-member environments a tick resolves are
    /// [`execute_tick_batch`](Self::execute_tick_batch)'s business.
    #[must_use]
    pub fn execute_batch(
        &self,
        ctx: &TickBatchContext<'_>,
        snapshot: &(dyn Substates + Sync),
        transactions: &[Arc<Verified<Transaction>>],
    ) -> Vec<ExecutedTx> {
        // Every member reads the context's own clock and randomness:
        // one block committed them all.
        let env_by_tx: BTreeMap<TxHash, EnvInputs> = transactions
            .iter()
            .map(|tx| {
                (
                    tx.hash(),
                    EnvInputs {
                        clock_ms: ctx.tick_ts.as_millis(),
                        randomness: tx_randomness(ctx.tick_reveal, tx.hash()),
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
        ctx: &TickBatchContext<'_>,
        snapshot: &(dyn Substates + Sync),
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
            .filter(|i| i.abortable)
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
    use hyperscale_types::{AddressClass, LocalKey, Presence};
    use hyperscale_vm_types::AbortReason;

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

    /// A declared refusal is priced as the lost race it is, not as the
    /// defect it is not.
    ///
    /// The whole point of the class is that an author who declares a
    /// failure ends up better off than one who traps into the same
    /// outcome; pricing both at the ceiling would leave the fee schedule
    /// arguing against the ergonomics.
    #[test]
    fn a_decline_pays_the_floor_and_a_trap_pays_the_ceiling() {
        let payer = PayerFee {
            vault: SubstateKey {
                owner: Address::new([1; 31], AddressClass::Component),
                local: LocalKey([0; 16]),
            },
            floor: 7,
            max_fee: 1_000,
            abortable: false,
        };
        assert_eq!(
            charge_for(&Outcome::Declined { node: 0, code: 3 }, payer),
            Some(payer.floor)
        );
        assert_eq!(
            charge_for(
                &Outcome::UserError {
                    reason: AbortReason::Unreachable
                },
                payer
            ),
            Some(payer.max_fee)
        );
    }

    /// A presence requirement the committed leaf no longer meets is the
    /// same lost race a changed rule is, and pays the same floor.
    ///
    /// The protocol cannot tell a sender who declared wrongly from one
    /// somebody raced to the leaf — the two leave identical state — so
    /// pricing the ambiguity at the ceiling would charge every honest
    /// loser to reach the careless caller, and would make any leaf a
    /// third party can create a lever for spending somebody else's
    /// declared maximum.
    #[test]
    fn an_unmet_presence_pays_the_floor_a_changed_rule_pays() {
        let payer = PayerFee {
            vault: SubstateKey {
                owner: Address::new([1; 31], AddressClass::Component),
                local: LocalKey([0; 16]),
            },
            floor: 7,
            max_fee: 1_000,
            abortable: false,
        };
        let leaf = EffectTarget::Point(SubstateKey {
            owner: Address::new([2; 31], AddressClass::Component),
            local: LocalKey([9; 16]),
        });
        // A declared condition is the same precondition-on-committed-state
        // class, in both of the shapes it is judged in.
        assert_eq!(
            charge_for(
                &Outcome::ConditionUnmet {
                    condition: UnmetCondition::Holds {
                        target: leaf,
                        required: Presence::Present,
                    },
                },
                payer
            ),
            Some(payer.floor),
        );
        assert_eq!(
            charge_for(
                &Outcome::ConditionUnmet {
                    condition: UnmetCondition::Satisfies { node: 0 },
                },
                payer
            ),
            Some(payer.floor),
        );
    }
}
