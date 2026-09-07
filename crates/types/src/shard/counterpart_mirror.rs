//! What this node has heard counterparts say about the transactions its
//! legs issued for, as one mirror.
//!
//! Three facts live here, each about one `(transaction, counterpart)`
//! pair: a core shard's refusal, read off its certificate; a
//! counterpart's proved absence of a cell; and a consumer's acceptance,
//! read off its certificate. Each licenses an abandonment record, and each is asked
//! about twice — once by the execution coordinator, composing the record
//! to offer, and once by the vote fence, checking a record a block
//! carries against what this validator itself holds.
//!
//! Two mirrors of one fact would let a record pass the fence that its
//! own composer would never have offered, and the difference between
//! them would be nobody's to notice. So the fact has one home and both
//! consumers read it, exactly as
//! [`ProvenAnchors`](crate::ProvenAnchors) holds the anchors those same
//! two ask about.
//!
//! # Node-local, and shared
//!
//! What a validator has heard is its own view — it is why the fence
//! defers rather than refusing — so nothing here is consensus content
//! and there is no determinism to preserve. One instance per host,
//! shared by handle.
//!
//! # Retention
//!
//! An entry speaks for a transaction this shard still owes an outcome
//! for, and means nothing once it does not. The execution coordinator
//! owns that ledger and is the only writer here, so it is the one that
//! says what to drop, through [`CounterpartMirror::retain`]. There is no
//! clock in this file: a second retention rule stated against one would
//! be a second answer to when a fact stops being true.

use std::collections::{BTreeSet, HashMap};
use std::sync::RwLock;

use crate::{Absence, Acceptance, Probed, Refusal, SettledTxSet, ShardId, TxHash};

/// The facts, under one lock: they are written together at a commit and
/// read together at a vote, so splitting them would buy contention
/// nobody is waiting on.
#[derive(Debug, Default)]
struct Mirrored {
    refusals: HashMap<(TxHash, ShardId), Refusal>,
    /// Keyed by the question too: a core's committed cell proved absent
    /// is not a delivery's claim proved absent, and they license
    /// different records held to different floors.
    absences: HashMap<(TxHash, ShardId, Probed), Absence>,
    acceptances: HashMap<(TxHash, ShardId), Acceptance>,
    /// Complete settled-transaction sets of shards that have terminated,
    /// each verified against its beacon-attested terminal root. Absence
    /// from a set is proof, not ignorance.
    settled: HashMap<ShardId, SettledTxSet>,
    /// What this shard's own ledger says each departed shard was party
    /// to, taken when its set arrived: a departure record may name only
    /// these, since one naming a stranger would abandon business the
    /// departed shard never had here.
    parties: HashMap<ShardId, BTreeSet<TxHash>>,
}

/// Every counterpart's word this node holds, by transaction and shard.
#[derive(Debug, Default)]
pub struct CounterpartMirror {
    inner: RwLock<Mirrored>,
}

impl CounterpartMirror {
    /// An empty mirror.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a core shard's refusal, first word winning.
    ///
    /// `true` when this is the first refusal held for the pair — a
    /// second certificate restates a decision already mirrored, and the
    /// anchor the record is checked against must not move under it.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned, which means a consumer panicked holding
    /// it — the node is already unsound at that point.
    pub fn record_refusal(&self, tx_hash: TxHash, shard: ShardId, refusal: Refusal) -> bool {
        let mut mirrored = self.write();
        let vacant = !mirrored.refusals.contains_key(&(tx_hash, shard));
        if vacant {
            mirrored.refusals.insert((tx_hash, shard), refusal);
        }
        vacant
    }

    /// Record a counterpart's proved absence of a cell, first proof
    /// winning.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn record_absence(
        &self,
        tx_hash: TxHash,
        shard: ShardId,
        probed: Probed,
        absence: Absence,
    ) {
        self.write()
            .absences
            .entry((tx_hash, shard, probed))
            .or_insert(absence);
    }

    /// Record a consumer's acceptance, first word winning.
    ///
    /// `true` when this is the first acceptance held for the pair.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn record_acceptance(
        &self,
        tx_hash: TxHash,
        shard: ShardId,
        acceptance: Acceptance,
    ) -> bool {
        let mut mirrored = self.write();
        let vacant = !mirrored.acceptances.contains_key(&(tx_hash, shard));
        if vacant {
            mirrored.acceptances.insert((tx_hash, shard), acceptance);
        }
        vacant
    }

    /// The refusal held for one pair.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    #[must_use]
    pub fn refusal(&self, tx_hash: TxHash, shard: ShardId) -> Option<Refusal> {
        self.read().refusals.get(&(tx_hash, shard)).copied()
    }

    /// The absence held for one pair.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    #[must_use]
    pub fn absence(&self, tx_hash: TxHash, shard: ShardId, probed: Probed) -> Option<Absence> {
        self.read().absences.get(&(tx_hash, shard, probed)).copied()
    }

    /// The acceptance held for one pair.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    #[must_use]
    pub fn acceptance(&self, tx_hash: TxHash, shard: ShardId) -> Option<Acceptance> {
        self.read().acceptances.get(&(tx_hash, shard)).copied()
    }

    /// Every refusal held, with the pair it speaks for.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    #[must_use]
    pub fn refusals(&self) -> Vec<(TxHash, ShardId, Refusal)> {
        self.read()
            .refusals
            .iter()
            .map(|(&(tx_hash, shard), &refusal)| (tx_hash, shard, refusal))
            .collect()
    }

    /// Every absence held of the kind `probed` asks, with the pair it
    /// speaks for.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    #[must_use]
    pub fn absences(&self, probed: Probed) -> Vec<(TxHash, ShardId, Absence)> {
        self.read()
            .absences
            .iter()
            .filter(|((_, _, kind), _)| *kind == probed)
            .map(|(&(tx_hash, shard, _), &absence)| (tx_hash, shard, absence))
            .collect()
    }

    /// Every acceptance held, with the pair it speaks for.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    #[must_use]
    pub fn acceptances(&self) -> Vec<(TxHash, ShardId, Acceptance)> {
        self.read()
            .acceptances
            .iter()
            .map(|(&(tx_hash, shard), &acceptance)| (tx_hash, shard, acceptance))
            .collect()
    }

    /// Record a terminated shard's settled set, with what this shard's
    /// ledger says it was party to.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn record_settled(&self, shard: ShardId, settled: SettledTxSet, parties: BTreeSet<TxHash>) {
        let mut mirrored = self.write();
        mirrored.settled.insert(shard, settled);
        mirrored.parties.insert(shard, parties);
    }

    /// Read the settled sets in place, without copying them.
    ///
    /// The sets are whole transaction sets of a departed chain, so every
    /// consumer reads them behind the guard rather than taking one.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn with_settled<R>(&self, read: impl FnOnce(&HashMap<ShardId, SettledTxSet>) -> R) -> R {
        read(&self.read().settled)
    }

    /// Read in place what this shard's ledger said `shard` was party to
    /// when its set arrived. `None` where no set is held, which is the
    /// caller's cue to defer.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn with_parties<R>(
        &self,
        shard: ShardId,
        read: impl FnOnce(Option<&BTreeSet<TxHash>>) -> R,
    ) -> R {
        read(self.read().parties.get(&shard))
    }

    /// Drop the settled sets of shards `readable` no longer attests.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn retain_departures(&self, readable: &dyn Fn(ShardId) -> bool) {
        let mut mirrored = self.write();
        mirrored.settled.retain(|&shard, _| readable(shard));
        mirrored.parties.retain(|&shard, _| readable(shard));
    }

    /// Drop everything said about a transaction `held` does not name.
    ///
    /// The one retention rule. Called by the execution coordinator with
    /// its own ledger's answer, since that ledger is what an entry here
    /// speaks for.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn retain(&self, held: &dyn Fn(TxHash) -> bool) {
        let mut mirrored = self.write();
        mirrored.refusals.retain(|&(tx_hash, _), _| held(tx_hash));
        mirrored.absences.retain(|&(tx_hash, ..), _| held(tx_hash));
        mirrored
            .acceptances
            .retain(|&(tx_hash, _), _| held(tx_hash));
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Mirrored> {
        self.inner.read().expect("counterpart mirror lock poisoned")
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Mirrored> {
        self.inner
            .write()
            .expect("counterpart mirror lock poisoned")
    }
}
