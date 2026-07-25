//! The session's event stream is well ordered and reproducible.
//!
//! Runs natively: the derivation is target independent, so the properties the
//! browser depends on are checked without a wasm toolchain.

use hyperscale_demo::{Session, ShardPath, TraceKind};
use hyperscale_simulation::SimConfig;
use hyperscale_types::ShardId;

fn config() -> SimConfig {
    SimConfig {
        shard_size: 4,
        ..Default::default()
    }
}

/// Drive a session for `steps` half-second steps, returning every event.
fn run(seed: u64, steps: usize) -> Vec<hyperscale_demo::TraceEvent> {
    let mut session = Session::new(&config(), seed);
    (0..steps).flat_map(|_| session.step(500)).collect()
}

#[test]
fn a_session_commits_blocks_and_reports_them_in_weighted_time_order() {
    let events = run(42, 40);

    assert!(
        events.len() > 10,
        "20s of simulated time should commit well past ten blocks, got {}",
        events.len(),
    );

    let stamps: Vec<u64> = events.iter().map(|event| event.wt).collect();
    let mut sorted = stamps.clone();
    sorted.sort_unstable();
    assert_eq!(stamps, sorted, "events must arrive in weighted-time order");

    // Heights are contiguous from the first reported block: the walk reports
    // every committed height exactly once, never skipping or repeating.
    let heights: Vec<u64> = events
        .iter()
        .map(|event| match event.kind {
            TraceKind::BlockCommitted { height, .. } => height,
        })
        .collect();
    let first = heights[0];
    let expected: Vec<u64> = (first..first + heights.len() as u64).collect();
    assert_eq!(heights, expected, "reported heights must be contiguous");
}

#[test]
fn one_seed_replays_to_the_same_event_stream() {
    let first = run(42, 20);
    let second = run(42, 20);

    let render = |events: &[hyperscale_demo::TraceEvent]| format!("{events:?}");
    assert_eq!(
        render(&first),
        render(&second),
        "a seeded session must replay identically",
    );
}

#[test]
fn shard_paths_spell_the_trie_so_a_parent_prefixes_its_children() {
    let root = ShardPath::from(ShardId::ROOT).0;
    let left = ShardPath::from(ShardId::leaf(1, 0)).0;
    let right = ShardPath::from(ShardId::leaf(1, 1)).0;
    let right_left = ShardPath::from(ShardId::leaf(2, 0b10)).0;
    let right_right = ShardPath::from(ShardId::leaf(2, 0b11)).0;

    assert_eq!(root, "");
    assert_eq!(left, "0");
    assert_eq!(right, "1");
    assert_eq!(right_left, "10");
    assert_eq!(right_right, "11");

    // The relation the viewer draws edges from.
    assert!(right_left.starts_with(&right));
    assert!(right_right.starts_with(&right));
    assert!(!right_left.starts_with(&left));
}

#[test]
fn a_different_seed_produces_a_different_run() {
    // Same length, different content: the chains diverge in timing even
    // though both make progress.
    let a = run(42, 20);
    let b = run(7, 20);
    assert!(!a.is_empty() && !b.is_empty());
    assert_ne!(format!("{a:?}"), format!("{b:?}"));
}
