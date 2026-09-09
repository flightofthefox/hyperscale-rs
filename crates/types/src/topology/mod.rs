//! Topology and validator set.
//!
//! - [`genesis`]: [`genesis::GenesisValidators`], the validators a runner
//!   projects its initial snapshot from at genesis.
//! - [`schedule`]: per-epoch [`TopologySchedule`] resolving committees by
//!   weighted timestamp.
//! - [`settled_set`]: the split-boundary settled-set predicate shared by
//!   the vote fence and the finalize gate.
//! - [`shard_prefix`]: the JMT root path a shard's state tree is rooted at.
//! - [`snapshot`]: read-only [`TopologySnapshot`] view used by subsystems.
//! - [`trie`]: the active shard partition as a binary [`trie::ShardTrie`].
//! - [`validator`]: [`ValidatorInfo`] / [`ValidatorSet`].

pub mod genesis;
pub mod network;
pub mod schedule;
pub mod settled_set;
pub mod shard_prefix;
pub mod snapshot;
pub mod trie;
pub mod validator;
