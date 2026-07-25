//! The session's event stream is well ordered and reproducible.
//!
//! Runs natively: the derivation is target independent, so the properties the
//! browser depends on are checked without a wasm toolchain.

use std::collections::BTreeMap;

use hyperscale_demo::{Session, SessionConfig, ShardPath, TraceEvent, TraceKind};
use hyperscale_types::ShardId;

fn config() -> SessionConfig {
    SessionConfig::default()
}

/// Drive a session for `steps` half-second steps, returning every event.
fn run(seed: u64, steps: usize) -> Vec<TraceEvent> {
    let mut session = Session::new(config(), seed);
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
        .filter_map(|event| match event.kind {
            TraceKind::BlockCommitted { height, .. } => Some(height),
            _ => None,
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

    let render = |events: &[TraceEvent]| format!("{events:?}");
    assert_eq!(
        render(&first),
        render(&second),
        "a seeded session must replay identically",
    );
}

#[test]
fn a_submitted_transfer_settles_and_reports_every_transition() {
    let mut session = Session::new(config(), 42);
    let mut events = Vec::new();

    // Submit a few transfers early, then let them run to a terminal outcome.
    for _ in 0..3 {
        session.submit_transfer();
        events.extend(session.step(500));
    }
    for _ in 0..60 {
        events.extend(session.step(500));
    }

    let submitted = events
        .iter()
        .filter(|e| matches!(e.kind, TraceKind::TxSubmitted { .. }))
        .count();
    assert_eq!(submitted, 3, "every submission is reported");

    let statuses: Vec<&str> = events
        .iter()
        .filter_map(|e| match &e.kind {
            TraceKind::TxStatusChanged { status, .. } => Some(*status),
            _ => None,
        })
        .collect();

    assert!(
        statuses.contains(&"committed"),
        "a transfer must be ordered into a block, saw {statuses:?}",
    );
    assert!(
        statuses.contains(&"succeeded"),
        "a funded transfer must reach a terminal success, saw {statuses:?}",
    );

    // Every transaction reaches a terminal outcome — nothing is left in
    // flight once the wave deadline has long passed (INV-EXEC-5).
    let terminal = statuses
        .iter()
        .filter(|s| matches!(**s, "succeeded" | "aborted" | "rejected"))
        .count();
    assert_eq!(terminal, 3, "every submission terminates, saw {statuses:?}");

    // A single-shard topology opens no cross-shard waves: that header field
    // exists to tell remote shards which certificates to expect.
    assert!(
        events.iter().all(|e| !matches!(
            e.kind,
            TraceKind::BlockCommitted {
                cross_shard_waves: 1..,
                ..
            }
        )),
        "one shard means no cross-shard waves",
    );
}

#[test]
fn the_root_shard_splits_while_the_session_is_being_watched() {
    let mut session = Session::new(
        SessionConfig {
            max_shards: 2,
            ..SessionConfig::default()
        },
        42,
    );

    // The session opens on a single ROOT shard — the split has not happened
    // yet, which is the whole point: a viewer sees it occur.
    assert_eq!(
        session
            .live_shards()
            .into_iter()
            .map(|s| ShardPath::from(s).0)
            .collect::<Vec<_>>(),
        vec![String::new()],
        "a session opens at genesis, on one shard",
    );

    // The split lands about six epochs in — trigger, admission, cohort
    // draw, snap-sync, readiness gate, flip — so give it headroom.
    let mut events = Vec::new();
    for _ in 0..600 {
        events.extend(session.step(500));
    }

    let splits: Vec<(Vec<String>, Vec<String>)> = events
        .iter()
        .filter_map(|e| match &e.kind {
            TraceKind::TopologyChanged {
                appeared, retired, ..
            } => Some((
                appeared.iter().map(|s| s.0.clone()).collect(),
                retired.iter().map(|s| s.0.clone()).collect(),
            )),
            _ => None,
        })
        .collect();

    assert_eq!(
        splits.len(),
        1,
        "exactly one partition change, saw {splits:?}"
    );
    let (appeared, retired) = &splits[0];
    assert_eq!(appeared, &["0".to_string(), "1".to_string()]);
    assert_eq!(
        retired,
        &[String::new()],
        "the root retires into its children"
    );

    // Blocks must be reported on the root before the split and on both
    // children after it — the timeline needs all three lanes.
    let mut per_shard: BTreeMap<String, u32> = BTreeMap::new();
    for event in &events {
        if let TraceKind::BlockCommitted { shard, .. } = &event.kind {
            *per_shard.entry(shard.0.clone()).or_default() += 1;
        }
    }
    assert_eq!(
        per_shard.len(),
        3,
        "root then both children, saw {per_shard:?}"
    );
    assert!(
        per_shard.values().all(|n| *n > 3),
        "every lane runs a real chain, saw {per_shard:?}",
    );
}

#[test]
fn transfers_reach_a_terminal_outcome_on_either_side_of_a_split() {
    // Status is per host and a host only tracks the shards it serves, so a
    // session that polls one host reports nothing for transactions routed to
    // the other child — they look stuck in flight forever when they in fact
    // settled. Every submission here must reach a terminal outcome.
    let mut session = Session::new(
        SessionConfig {
            max_shards: 2,
            ..SessionConfig::default()
        },
        42,
    );

    let mut events = Vec::new();
    for _ in 0..440 {
        events.extend(session.step(500));
    }
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, TraceKind::TopologyChanged { .. })),
        "the split must have happened before transactions are submitted",
    );

    let mut submitted = 0;
    for i in 0..460 {
        if i % 25 == 0 {
            session.submit_transfer();
            submitted += 1;
        }
        events.extend(session.step(500));
    }

    let mut outcome: BTreeMap<String, String> = BTreeMap::new();
    for event in &events {
        match &event.kind {
            TraceKind::TxSubmitted { tx } => {
                outcome.insert(tx.0.clone(), "never reported".to_string());
            }
            TraceKind::TxStatusChanged { tx, status, .. } => {
                outcome.insert(tx.0.clone(), (*status).to_string());
            }
            _ => {}
        }
    }

    assert_eq!(outcome.len(), submitted, "every submission is reported");
    let unresolved: Vec<_> = outcome
        .iter()
        .filter(|(_, s)| !matches!(s.as_str(), "succeeded" | "aborted" | "rejected"))
        .collect();
    assert!(
        unresolved.is_empty(),
        "every transfer settles on whichever child owns it; unresolved: {unresolved:?}",
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
