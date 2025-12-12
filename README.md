# hyperscale-rs

> **Warning**: Work in progress. Do not use.

Rust implementation of Hyperscale consensus protocol.

**What's different:**
- Pure consensus layer — no I/O, no locks, no async
- Created with deterministic simulation testing in mind
- Faster two-chain commit consensus based on HotStuff-2
- Optimistic pipelining — proposers propose immediately after QC formation
- Improved cross-shard livelock prevention
- Real Radix Engine integration

## Crates

| Crate | Purpose |
|-------|---------|
| `hyperscale-types` | Core types: hashes, blocks, votes, QCs, keys, transactions, topology |
| `hyperscale-core` | The `StateMachine` and `SubStateMachine` traits that everything implements |
| `hyperscale-messages` | Network message serialization (SBOR encoding) |
| `hyperscale-bft` | BFT consensus: block proposal, voting, QC formation, view changes |
| `hyperscale-execution` | Transaction execution with cross-shard 2PC coordination |
| `hyperscale-mempool` | Transaction pool management |
| `hyperscale-livelock` | Cross-shard deadlock detection and prevention |
| `hyperscale-sync` | Block synchronization for nodes catching up |
| `hyperscale-node` | Composes all sub-state machines into the main `NodeStateMachine` |
| `hyperscale-engine` | Radix Engine integration for smart contract execution |
| `hyperscale-simulation` | Deterministic simulator with configurable network conditions |
| `hyperscale-simulator` | CLI tool for running simulations with metrics |
| `hyperscale-production` | Production runner: libp2p networking, RocksDB storage, thread pools |
| `hyperscale-spammer` | Transaction spammer CLI and library for load testing |

## Building

```bash
cargo build --release
```

## Running the Simulator

```bash
cargo run --release --bin hyperscale-sim
```

## Running Tests

```bash
cargo test
```
