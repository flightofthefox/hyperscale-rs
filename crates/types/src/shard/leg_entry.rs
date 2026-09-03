//! A leg entry as it survives a restart.
//!
//! A shard's execution ledger is a fold over its committed chain, so
//! what it holds is bounded by how far back a restart replays. A leg
//! entry outlives that: it stands until the record cell it would take
//! back is retired, and a record is retired on a counterpart's evidence
//! rather than on a clock, so a counterpart halted for a day leaves both
//! standing for a day. The entry is written down for exactly that
//! reason.
//!
//! A row carries the account and the trie its classification was frozen
//! against, and nothing else. A committed transaction's body is written
//! to the store and never pruned, so the legs, the owners and the
//! crossings the reclaim and the retirement walk are read back off it,
//! and the classification is the freeze of those against the trie here —
//! the same answer the committing block reached, since the freeze is a
//! function of exactly those three.

use std::collections::BTreeSet;

use hyperscale_hbor::Hbor;

use crate::{AbortCharge, Address, ShardId, ShardTrie, TxHash, Unsettleable, WeightedTimestamp};

/// What this shard's part in a transaction is, which decides what its
/// entry waits on and what ends it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hbor)]
pub enum LegEntryKind {
    /// This shard's verdict is the transaction's, or a share of it.
    #[hbor(discriminant = 0)]
    Whole,
    /// This shard only delivers for the transaction.
    #[hbor(discriminant = 1)]
    Delivery,
    /// This shard ran only a leg: it froze divided with this shard
    /// outside the core set.
    #[hbor(discriminant = 2)]
    Leg,
    /// This shard's own verdict resolved the transaction and the entry
    /// stays for the reclaim of what its deliveries never claim.
    #[hbor(discriminant = 3)]
    Remainder,
}

/// What a tick of this shard's composed over a leg entry's records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hbor)]
pub enum LegEntryTaken {
    /// The reclaim, on a record that says the counterpart never will.
    #[hbor(discriminant = 0)]
    Reclaim,
    /// The retirement, on a record that says every consumer claimed.
    #[hbor(discriminant = 1)]
    Retire,
}

/// One leg entry, in the form a restart reads it back in.
///
/// Every figure here is one the ledger derived from committed content,
/// so a row and the fold that would have produced it agree; the store is
/// what carries an entry past the window that fold reaches.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct LegEntry {
    /// The transaction the entry answers for.
    pub tx_hash: TxHash,
    /// The moment past which the transaction can no longer finalize
    /// anywhere.
    pub deadline: WeightedTimestamp,
    /// The reservation its committing block took against the drain.
    pub declared_work: u64,
    /// What an abort of it burns, and out of whose vault.
    pub charge: AbortCharge,
    /// The frontier its committing block anchored at.
    pub committed_ts: WeightedTimestamp,
    /// The owner prefixes it reaches outside this shard.
    pub remote_prefixes: BTreeSet<Address>,
    /// Whether a tick of this shard's took it as a member.
    pub certified: bool,
    /// Whether a committed finalization of this shard's settled its
    /// price.
    pub charged: bool,
    /// What this shard's part in it is.
    pub kind: LegEntryKind,
    /// Which member a tick of this shard's has taken its records for.
    pub taken: Option<LegEntryTaken>,
    /// The shard a committed record says left it unsettled.
    pub unsettled_by: Option<ShardId>,
    /// What that record established.
    pub evidence: Option<Unsettleable>,
    /// The consumer shards a committed record says claimed what it
    /// issued.
    pub claimed_by: BTreeSet<ShardId>,
    /// The trie its classification was frozen against — the one
    /// placement fact the freeze reads, and the reason a row can carry
    /// the classification without carrying its shape.
    pub trie: ShardTrie,
}
