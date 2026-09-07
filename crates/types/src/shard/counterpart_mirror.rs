//! What this node has heard counterparts say about the transactions its
//! legs issued for, as one mirror.
//!
//! One fact lives here, about one `(transaction, counterpart, question)`
//! triple: what the counterpart said and when — a verdict read off its
//! certificate, or a cell's absence read off a proof the chain
//! committed. Each licenses an abandonment record, and each is asked
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

use crate::{Heard, Question, SettledTxSet, ShardId, TxHash};

/// The facts, under one lock: they are written together at a commit and
/// read together at a vote, so splitting them would buy contention
/// nobody is waiting on.
#[derive(Debug, Default)]
struct Mirrored {
    /// What each counterpart said to each question about each
    /// transaction, first word winning: a shard says one thing per
    /// question, and the moment a record is checked against must not
    /// move under it.
    heard: HashMap<(TxHash, ShardId, Question), Heard>,
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

    /// Record what `shard` said about `tx_hash`, first word winning.
    ///
    /// `true` when this is the first word held for the question — a
    /// second certificate or proof restates an answer already mirrored.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned, which means a consumer panicked holding
    /// it — the node is already unsound at that point.
    pub fn record(&self, tx_hash: TxHash, shard: ShardId, heard: Heard) -> bool {
        let mut mirrored = self.write();
        let key = (tx_hash, shard, heard.question);
        let vacant = !mirrored.heard.contains_key(&key);
        if vacant {
            mirrored.heard.insert(key, heard);
        }
        vacant
    }

    /// What `shard` said to `question` about `tx_hash`, if anything.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    #[must_use]
    pub fn heard(&self, tx_hash: TxHash, shard: ShardId, question: Question) -> Option<Heard> {
        self.read().heard.get(&(tx_hash, shard, question)).copied()
    }

    /// Everything held, with the pair each speaks for.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    #[must_use]
    pub fn all(&self) -> Vec<(TxHash, ShardId, Heard)> {
        self.read()
            .heard
            .iter()
            .map(|(&(tx_hash, shard, _), &heard)| (tx_hash, shard, heard))
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
        self.write().heard.retain(|&(tx_hash, ..), _| held(tx_hash));
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
