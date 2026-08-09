//! Storage traits and shared types.
//!
//! This crate defines the storage abstraction used by runners to persist substate state,
//! along with shared types and utilities that both in-memory and `RocksDB` storage
//! implementations need.
//!
//! # Design
//!
//! Storage is an implementation detail of runners, not the state machine.
//! The state machine emits `Action::ExecuteTransactions` and receives
//! `ProtocolEvent::ExecutionBatchCompleted` - it never touches storage directly.
//!
//! Runners own storage and pass it to the executor:
//! - `SimulationRunner` uses in-memory storage (`SimShardStorage`)
//! - `ProductionRunner` uses `RocksDB` (`RocksDbShardStorage`)
//!
//! # Jellyfish Merkle Tree (JMT)
//!
//! The `tree` module provides the binary Blake3 JMT state tree adapter.
//! Storage backends implement `jmt::TreeReader` to provide tree access —
//! both `RocksDB` and `SimShardStorage` hook into the same trait.

#![warn(missing_docs)]

pub mod beacon;
pub mod lock_recover;
pub mod shard;
pub mod tree;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_helpers;

pub use beacon::chain_reader::BeaconChainReader;
pub use beacon::chain_writer::BeaconChainWriter;
pub use beacon::ratify_registers::RatifyRegisterStore;
pub use beacon::storage::BeaconStorage;
use hyperscale_jmt::TreeReader;
use hyperscale_types::{SettledWrites, SubstateKey};
pub use shard::boundary::{
    AdoptSource, BOUNDARY_RETAIN, BoundaryStore, ImportCursor, ImportProgress, WitnessSeed,
};
pub use shard::chain_reader::{BlockForSync, ShardChainReader};
pub use shard::chain_writer::{ParentAnchor, ShardChainWriter};
pub use shard::genesis::GenesisCommit;
pub use shard::pending_chain::{BaseReadCache, ChainEntry, PendingChain, SubstateView};
pub use shard::recovered_state::RecoveredState;
pub use shard::store::{SubstateStore, VersionedStore};
pub use shard::tick_certs::{covers_strictly_more, widest_tick_copies};
pub use shard::tick_chain::{
    ProvisionalTx, TickChain, TickOutput, TickResolution, TickView, TickViewSnapshot,
};
pub use shard::unresolved::{ReplayWindow, replay_window, unresolved_replay_floor};
pub use shard::vote_registers::SafeVoteRegisterStore;
pub use shard::writes::{
    filter_writes_to_prefix, fold_state_writes, merge_state_writes, merge_writes_from_receipts,
};
pub use tree::{CollectedWrites, JmtSnapshot};

/// Umbrella bound for storage backends threaded as a generic `S` through
/// node-side machinery (the `IoLoop` and its delegated action handler).
///
/// Use this only at sites that *thread* storage generically — i.e. the
/// `IoLoop<S>` impls and entry points that must satisfy every capability
/// `IoLoop` ultimately needs. For narrower scopes (block commit, shard consensus
/// proposal building, provision handlers), bound on the specific traits
/// directly so the signature reflects what the function actually touches.
pub trait ShardStorage:
    ShardChainWriter
    + SubstateStore
    + VersionedStore
    + ShardChainReader
    + TreeReader
    + BoundaryStore
    + SafeVoteRegisterStore
    + Send
    + Sync
    + 'static
{
}

impl<S> ShardStorage for S where
    S: ShardChainWriter
        + SubstateStore
        + VersionedStore
        + ShardChainReader
        + TreeReader
        + BoundaryStore
        + SafeVoteRegisterStore
        + Send
        + Sync
        + 'static
{
}

/// Read access to a substate store.
///
/// Object-safe: the execution seam borrows one as `dyn SubstateDatabase`
/// so a single batch entry point serves every backend's snapshot type.
pub trait SubstateDatabase {
    /// The value at `key`, or `None` if absent.
    fn substate(&self, key: SubstateKey) -> Option<Vec<u8>>;
}

/// Write access to a substate store. Test and genesis paths commit
/// through it directly; the live path goes through `ShardChainWriter`.
pub trait CommittableSubstateDatabase {
    /// Apply `writes` to the store. Values only — a store holds values,
    /// so whatever moved has already been resolved against what it moved
    /// from.
    fn commit(&mut self, writes: &SettledWrites);
}
