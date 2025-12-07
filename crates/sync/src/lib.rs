//! Block synchronization state machine.
//!
//! This crate implements the sync protocol for catching up to the chain when
//! a node falls behind (due to partitions, crashes, or late joining).
//!
//! # Protocol Overview
//!
//! 1. **Discovery**: Node detects it's behind when receiving block headers
//!    or QCs that reference heights beyond its committed height.
//!
//! 2. **Parallel Fetching**: Request blocks from multiple peers in parallel,
//!    starting from committed_height + 1 up to the target height.
//!
//! 3. **Verification**: Each fetched block is verified against its QC before
//!    being applied. Bad blocks trigger retry from different peers.
//!
//! 4. **Application**: Verified blocks are applied in order, updating the
//!    committed height and notifying the BFT state machine.
//!
//! # Architecture
//!
//! ```text
//! BftState ──────────────────────────────────────────────────────┐
//!    │                                                           │
//!    │ on_block_header() detects we're behind                   │
//!    ▼                                                           │
//! SyncNeeded { target_height, target_hash, from }               │
//!    │                                                           │
//!    ▼                                                           │
//! SyncState                                                      │
//!    │                                                           │
//!    ├─► RequestSyncBlock (to multiple peers in parallel)        │
//!    │                                                           │
//!    │◄── SyncBlockReceived (blocks arrive)                     │
//!    │                                                           │
//!    ├─► Verify block against QC                                │
//!    │                                                           │
//!    ├─► SyncBlockReadyToApply (verified, consecutive)           │
//!    │                                                           │
//!    └──────────────────────────────────────────────────────────►│
//!                                                   BftState applies
//! ```

mod config;
mod state;

pub use config::SyncConfig;
pub use state::SyncState;
