//! Every transfer a session submits reaches an outcome, reshape or no.
//!
//! The rest of the demo's coverage submits either side of a split and
//! waits for it to land first. This submits straight through one, which
//! is the only way to cover the transfers that arrive while the parent
//! has stopped including and its children are not yet live.

use std::collections::{BTreeMap, BTreeSet};

use hyperscale_demo::{Session, SessionConfig, TraceEvent, TraceKind, TxLabel};

/// Labels a transfer can finish at. Anything else — `pending`,
/// `committed`, or no status at all — is a transfer still owed an answer.
fn is_terminal(label: &str) -> bool {
    matches!(label, "succeeded" | "aborted" | "rejected")
}

/// A transfer submitted just before a split's terminal is admitted to a
/// mempool that is about to die, and the chain has no later block to
/// include it in. Nothing it holds can carry it: its successors build
/// their own pools, and the terminal sweep only decides transactions the
/// chain committed. It reaches an outcome only because the dissolving
/// chain offers back what it never included.
#[test]
fn transfers_submitted_across_a_split_all_reach_an_outcome() {
    let mut session = Session::new(
        SessionConfig {
            max_shards: 2,
            ..SessionConfig::default()
        },
        42,
    );

    let mut events: Vec<TraceEvent> = Vec::new();
    let mut submitted: Vec<String> = Vec::new();

    // Steadily from the start and well past the split: where the cut
    // falls relative to any one submission is not something a session can
    // be told, so the train covers the window instead of aiming at it.
    for i in 0..1_200 {
        if i % 10 == 0 {
            submitted.push(TxLabel::from(session.submit_transfer()).0);
        }
        events.extend(session.step(500));
    }
    // Quiet time, so the last submissions have room to settle.
    for _ in 0..400 {
        events.extend(session.step(500));
    }

    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, TraceKind::TopologyChanged { .. })),
        "the split must land inside the run, or nothing here is being tested",
    );

    // Last status reported per transaction.
    let mut last: BTreeMap<String, String> = BTreeMap::new();
    for event in &events {
        if let TraceKind::TxStatusChanged { tx, status, .. } = &event.kind {
            last.insert(tx.0.clone(), (*status).to_string());
        }
    }

    let stuck: BTreeSet<(&str, &str)> = submitted
        .iter()
        .map(|tx| {
            (
                tx.as_str(),
                last.get(tx).map_or("<no status at all>", String::as_str),
            )
        })
        .filter(|(_, label)| !is_terminal(label))
        .collect();

    assert!(
        stuck.is_empty(),
        "{} of {} transfers never reached an outcome: {stuck:?}",
        stuck.len(),
        submitted.len(),
    );
}
