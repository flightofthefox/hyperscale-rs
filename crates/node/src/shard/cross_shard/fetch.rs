//! Cross-shard fetch bindings.
//!
//! The [`FetchBinding`] impls for the cross-shard data-availability payloads —
//! provisions, execution certificates, finalizations, and local provisions.
//! Each `fetch_mut` resolves the binding's `Fetch` instance out of this shard's
//! [`CrossShardState`](super::CrossShardState). The generic engine, the
//! `FetchBinding` trait, and the shared `partition_solicited` helper live in
//! [`crate::fetch`].

use std::collections::BTreeMap;
use std::sync::Arc;

use crossbeam::channel::Sender;
use hyperscale_core::ProtocolEvent;
use hyperscale_network::{Network, ResponseVerdict};
use hyperscale_storage::ShardStorage;
use hyperscale_types::network::request::{
    GetCommittedTxsRequest, GetExecutionCertsRequest, GetFinalizationsRequest,
    GetLocalProvisionsRequest, GetProvisionsRequest, GetStateProofRequest,
};
use hyperscale_types::network::response::CommittedTxVerdict;
use hyperscale_types::{
    BlockHeight, ExecutionCertificate, Finalization, FinalizationHash, MessageClass,
    PredecessorTerminal, ProvisionHash, ShardId, StateAnchor, SubstateKey, TxHash, ValidatorId,
    Verifiable,
};

use crate::fetch::{Fetch, FetchBinding, partition_solicited};
use crate::shard::{HostEvent, ShardIo, ShardScopedInput, push_protocol_event, push_shard_input};

// ─── Type aliases ──────────────────────────────────────────────────────
/// Local-provision fetch keyed by [`ProvisionHash`].
pub type LocalProvisionFetch = Fetch<ProvisionHash>;
/// Finalization fetch keyed by [`TickId`].
pub type FinalizationFetch = Fetch<FinalizationHash>;
/// Cross-shard execution-cert fetch keyed by [`TickId`].
pub type ExecCertFetch = Fetch<(ShardId, TxHash)>;
/// Cross-shard provision fetch keyed by
/// `(source_shard, target_shard, block_height)`. `source_shard` selects
/// the responding committee; `target_shard` rides in the body for
/// response filtering on the responder.
pub type ProvisionFetch = Fetch<(ShardId, ShardId, BlockHeight)>;
/// Committed-transaction membership fetch keyed by
/// `(predecessor, tx_hash)`. The predecessor's shard selects the
/// responding committee, its terminal rides in the body as the window to
/// reconstruct, and its `committed_txs_root` is the key each absence
/// proof is checked against.
pub type CommittedTxFetch = Fetch<(PredecessorTerminal, TxHash)>;
pub type StateProofFetch = Fetch<(StateAnchor, SubstateKey)>;

// ─── Bindings ──────────────────────────────────────────────────────────

/// Marker type for the per-block local-provision fetch.
pub struct LocalProvisionBinding;

impl FetchBinding for LocalProvisionBinding {
    type Id = ProvisionHash;

    const NAME: &'static str = "local_provision";

    fn fetch_mut<S: ShardStorage>(shard: &mut ShardIo<S>) -> &mut Fetch<ProvisionHash> {
        &mut shard.cross_shard.local_provision
    }

    fn dispatch_chunk<N: Network>(
        ids: Vec<ProvisionHash>,
        local_shard: ShardId,
        shard: ShardId,
        preferred: Option<ValidatorId>,
        class: Option<MessageClass>,
        network: &N,
        sender: &Sender<HostEvent>,
    ) {
        let es = sender.clone();
        let hs = ids.clone();
        network.request(
            shard,
            preferred,
            GetLocalProvisionsRequest::new(ids),
            class,
            Box::new(move |result| {
                if let Ok(resp) = result {
                    let split =
                        partition_solicited(resp.entries, &hs, |entry| [entry.provisions.hash()]);
                    // Push the bundled source header BEFORE the provisions
                    // so the verification pipeline has a chance to admit it
                    // first. The header is QC-self-authenticating; sender is
                    // the fetched-header sentinel (no peer attestation).
                    for entry in split.kept {
                        if let Some(certified_header) = entry.source_header {
                            push_protocol_event(
                                &es,
                                local_shard,
                                ProtocolEvent::UnverifiedRemoteHeaderReceived {
                                    certified_header,
                                    sender: ValidatorId::new(u64::MAX),
                                },
                            );
                        }
                        push_protocol_event(
                            &es,
                            local_shard,
                            ProtocolEvent::UnverifiedProvisionsReceived {
                                provisions: entry.provisions,
                            },
                        );
                    }
                    let had_misses = !split.missing.is_empty();
                    if had_misses {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::LocalProvisionsFetchFailed {
                                hashes: split.missing,
                            },
                        );
                    }
                    // Reject the response if the peer shipped unsolicited
                    // provisions OR if any requested hash was missing.
                    if split.unsolicited > 0 || had_misses {
                        ResponseVerdict::Reject
                    } else {
                        ResponseVerdict::Accept
                    }
                } else {
                    push_shard_input(
                        &es,
                        local_shard,
                        ShardScopedInput::LocalProvisionsFetchFailed { hashes: hs },
                    );
                    ResponseVerdict::Accept
                }
            }),
        );
    }
}

/// Marker type for the per-block finalization fetch.
pub struct FinalizationBinding;

impl FetchBinding for FinalizationBinding {
    type Id = FinalizationHash;

    const NAME: &'static str = "finalization";

    fn fetch_mut<S: ShardStorage>(shard: &mut ShardIo<S>) -> &mut Fetch<FinalizationHash> {
        &mut shard.cross_shard.finalization
    }

    fn dispatch_chunk<N: Network>(
        ids: Vec<FinalizationHash>,
        local_shard: ShardId,
        shard: ShardId,
        preferred: Option<ValidatorId>,
        class: Option<MessageClass>,
        network: &N,
        sender: &Sender<HostEvent>,
    ) {
        let es = sender.clone();
        let requested_ids = ids.clone();
        network.request(
            shard,
            preferred,
            GetFinalizationsRequest::new(ids),
            class,
            Box::new(move |result| {
                if let Ok(resp) = result {
                    let split = partition_solicited(resp.finalizations, &requested_ids, |w| {
                        [w.receipt_hash()]
                    });
                    if !split.kept.is_empty() {
                        // Refcount is 1 right after decode, so each unwrap moves.
                        let finalizations: Vec<Arc<Verifiable<Finalization>>> = split
                            .kept
                            .into_iter()
                            .map(|arc| Arc::new(Arc::unwrap_or_clone(arc).into()))
                            .collect();
                        push_protocol_event(
                            &es,
                            local_shard,
                            ProtocolEvent::FinalizationsReceived { finalizations },
                        );
                    }
                    let had_misses = !split.missing.is_empty();
                    if had_misses {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::FinalizationsFetchFailed { ids: split.missing },
                        );
                    }
                    // Reject responses with unsolicited ticks (peer scoring;
                    // also avoids wasted signature verification on items we never
                    // asked for) or with any missing requested id.
                    if split.unsolicited > 0 || had_misses {
                        ResponseVerdict::Reject
                    } else {
                        ResponseVerdict::Accept
                    }
                } else {
                    push_shard_input(
                        &es,
                        local_shard,
                        ShardScopedInput::FinalizationsFetchFailed { ids: requested_ids },
                    );
                    ResponseVerdict::Accept
                }
            }),
        );
    }
}

/// Marker type for the cross-shard execution-cert fetch.
pub struct ExecCertBinding;

impl FetchBinding for ExecCertBinding {
    /// `(source_shard, tx_hash)` — the shard whose outcome is missing and
    /// the transaction it is missing for. The certificate's own identity
    /// is not a key here: the requester knows which shards its tick waits
    /// on, not which certificate each will put the transaction in.
    type Id = (ShardId, TxHash);

    const NAME: &'static str = "exec_cert";

    fn fetch_mut<S: ShardStorage>(shard: &mut ShardIo<S>) -> &mut Fetch<(ShardId, TxHash)> {
        &mut shard.cross_shard.exec_cert
    }

    fn dispatch_chunk<N: Network>(
        ids: Vec<(ShardId, TxHash)>,
        local_shard: ShardId,
        shard: ShardId,
        preferred: Option<ValidatorId>,
        class: Option<MessageClass>,
        network: &N,
        sender: &Sender<HostEvent>,
    ) {
        let es = sender.clone();
        let failed_ids = ids.clone();
        network.request(
            shard,
            preferred,
            GetExecutionCertsRequest {
                tx_hashes: ids.into_iter().map(|(_, tx_hash)| tx_hash).collect(),
            },
            class,
            Box::new(move |result| {
                if let Ok(response) = result {
                    let certs = response.certificates.unwrap_or_default();
                    // One certificate answers for every transaction of its
                    // batch, so it clears every requested key it covers.
                    let split = partition_solicited(certs, &failed_ids, |c| {
                        let cert_shard = c.shard_id();
                        c.tx_outcomes()
                            .iter()
                            .map(|outcome| (cert_shard, outcome.tx_hash()))
                            .collect::<Vec<_>>()
                    });
                    let had_misses = !split.missing.is_empty();
                    if !split.kept.is_empty() {
                        // Refcount is 1 right after decode, so each unwrap moves.
                        let certificates: Vec<Verifiable<ExecutionCertificate>> = split
                            .kept
                            .into_iter()
                            .map(|arc| Arc::unwrap_or_clone(arc).into())
                            .collect();
                        push_protocol_event(
                            &es,
                            local_shard,
                            ProtocolEvent::ExecutionCertificatesReceived { certificates },
                        );
                    }
                    if had_misses {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::ExecCertFetchFailed {
                                hashes: split.missing,
                            },
                        );
                    }
                    // Reject the response if the peer shipped unsolicited
                    // ECs (peer scoring; also avoids wasted signature verification
                    // on items we never asked for) or any missing id.
                    if split.unsolicited > 0 || had_misses {
                        ResponseVerdict::Reject
                    } else {
                        ResponseVerdict::Accept
                    }
                } else {
                    push_shard_input(
                        &es,
                        local_shard,
                        ShardScopedInput::ExecCertFetchFailed { hashes: failed_ids },
                    );
                    ResponseVerdict::Accept
                }
            }),
        );
    }
}

/// Pair a committed-transaction response with the transactions it
/// answers for, or `None` when the response is unusable.
///
/// Verdicts are positional, so a length that doesn't match the request
/// is malformed rather than partial — nothing in it can be paired up.
/// Absence is the answer that relaxes the successor's standing refusal,
/// so it is the one that has to lift to `terminal.committed_txs_root`;
/// `Committed` is what the successor already assumes and carries no
/// proof.
///
/// One bad entry condemns the whole response rather than being skipped.
/// A peer that got any of it wrong has said nothing this node can lift
/// to the attested root, and picking through it would let a peer choose
/// which questions get answered.
fn verified_answers(
    verdicts: &[CommittedTxVerdict],
    terminal: PredecessorTerminal,
    tx_hashes: &[TxHash],
) -> Option<Vec<(TxHash, bool)>> {
    if verdicts.len() != tx_hashes.len() {
        return None;
    }
    verdicts
        .iter()
        .zip(tx_hashes)
        .map(|(verdict, hash)| match verdict {
            CommittedTxVerdict::Committed => Some((*hash, false)),
            CommittedTxVerdict::Absent(proof) => proof
                .proves_absent(hash, terminal.committed_txs_root)
                .then_some((*hash, true)),
        })
        .collect()
}

/// Marker type for the committed-transaction membership fetch.
pub struct CommittedTxBinding;

impl FetchBinding for CommittedTxBinding {
    /// `(predecessor, tx_hash)` — the chain that ran before this one and
    /// a transaction whose membership in its committed set decides
    /// whether this chain may admit it. The predecessor rides whole
    /// because the request names its terminal and the answer is checked
    /// against that terminal's root.
    type Id = (PredecessorTerminal, TxHash);

    const NAME: &'static str = "committed_tx";

    fn fetch_mut<S: ShardStorage>(shard: &mut ShardIo<S>) -> &mut Fetch<Self::Id> {
        &mut shard.cross_shard.committed_tx
    }

    fn dispatch_chunk<N: Network>(
        ids: Vec<Self::Id>,
        local_shard: ShardId,
        shard: ShardId,
        preferred: Option<ValidatorId>,
        class: Option<MessageClass>,
        network: &N,
        sender: &Sender<HostEvent>,
    ) {
        // A chunk is grouped by `(shard, preferred, class)`, which does
        // not separate two terminals of the same shard, and one request
        // resolves against exactly one terminal. Split by terminal here
        // rather than assume the chunk is uniform.
        let mut by_terminal: BTreeMap<PredecessorTerminal, Vec<TxHash>> = BTreeMap::new();
        for (predecessor, tx_hash) in ids {
            by_terminal.entry(predecessor).or_default().push(tx_hash);
        }
        for (predecessor, tx_hashes) in by_terminal {
            debug_assert_eq!(
                shard, predecessor.shard,
                "CommittedTxBinding routes to the predecessor; the scan sets it from the id",
            );
            let requested: Vec<Self::Id> =
                tx_hashes.iter().map(|hash| (predecessor, *hash)).collect();
            let request =
                GetCommittedTxsRequest::new(predecessor.height, predecessor.block_hash, tx_hashes);
            let es = sender.clone();
            network.request(
                shard,
                preferred,
                request,
                class,
                Box::new(move |result| {
                    let asked: Vec<TxHash> = requested.iter().map(|(_, hash)| *hash).collect();
                    let Ok(response) = result else {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::CommittedTxsFetchFailed { ids: requested },
                        );
                        return ResponseVerdict::Accept;
                    };
                    // This peer doesn't hold the named terminal — rotate.
                    let Some(verdicts) = response.verdicts else {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::CommittedTxsFetchFailed { ids: requested },
                        );
                        return ResponseVerdict::Reject;
                    };
                    let Some(answers) = verified_answers(&verdicts, predecessor, &asked) else {
                        tracing::warn!(
                            predecessor = ?predecessor.shard,
                            asked = asked.len(),
                            answered = verdicts.len(),
                            "Dropping committed-transaction response: unusable verdicts"
                        );
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::CommittedTxsFetchFailed { ids: requested },
                        );
                        return ResponseVerdict::Reject;
                    };
                    // Release the slots before delivering the answers, so
                    // the freed capacity is available if handling the
                    // delivery re-drives this fetch.
                    push_shard_input(
                        &es,
                        local_shard,
                        ShardScopedInput::CommittedTxsFetchFulfilled { ids: requested },
                    );
                    push_protocol_event(
                        &es,
                        local_shard,
                        ProtocolEvent::PrecutResolutionsReceived {
                            predecessor: predecessor.shard,
                            answers,
                        },
                    );
                    ResponseVerdict::Accept
                }),
            );
        }
    }
}

/// Marker type for the state-proof fetch against a commit-proven remote
/// header.
pub struct StateProofBinding;

impl FetchBinding for StateProofBinding {
    /// `(anchor, key)` — the commit-proven state the proof reconstructs
    /// and one key whose presence under it is asked. The anchor rides
    /// whole because its root is what the answer is checked against
    /// before it reaches the coordinator.
    type Id = (StateAnchor, SubstateKey);

    const NAME: &'static str = "state_proof";

    fn fetch_mut<S: ShardStorage>(shard: &mut ShardIo<S>) -> &mut Fetch<Self::Id> {
        &mut shard.cross_shard.state_proof
    }

    fn dispatch_chunk<N: Network>(
        ids: Vec<Self::Id>,
        local_shard: ShardId,
        shard: ShardId,
        preferred: Option<ValidatorId>,
        class: Option<MessageClass>,
        network: &N,
        sender: &Sender<HostEvent>,
    ) {
        // A chunk is grouped by `(shard, preferred, class)`, which does
        // not separate two anchors of the same shard, and one proof
        // reconstructs exactly one root. Split by anchor here.
        let mut by_anchor: BTreeMap<StateAnchor, Vec<SubstateKey>> = BTreeMap::new();
        for (anchor, key) in ids {
            by_anchor.entry(anchor).or_default().push(key);
        }
        for (anchor, keys) in by_anchor {
            debug_assert_eq!(
                shard, anchor.shard,
                "StateProofBinding routes to the anchor's shard; the runner sets it from the id",
            );
            let requested: Vec<Self::Id> = keys.iter().map(|key| (anchor, *key)).collect();
            let request = GetStateProofRequest::new(anchor.height, keys.clone());
            let es = sender.clone();
            network.request(
                shard,
                preferred,
                request,
                class,
                Box::new(move |result| {
                    let Ok(response) = result else {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::StateProofFetchFailed { ids: requested },
                        );
                        return ResponseVerdict::Accept;
                    };
                    // This peer doesn't hold the height — rotate.
                    let Some(proof) = response.proof else {
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::StateProofFetchFailed { ids: requested },
                        );
                        return ResponseVerdict::Reject;
                    };
                    // Checked here so an unusable proof rotates the peer
                    // rather than reaching a block. What it says is read
                    // off the block that carries it, by every replica.
                    if let Err(error) = proof.inclusions(anchor.state_root, anchor.shard, &keys) {
                        tracing::warn!(
                            shard = ?anchor.shard,
                            height = anchor.height.inner(),
                            %error,
                            "Dropping state-proof response: unusable proof"
                        );
                        push_shard_input(
                            &es,
                            local_shard,
                            ShardScopedInput::StateProofFetchFailed { ids: requested },
                        );
                        return ResponseVerdict::Reject;
                    }
                    // Release the slots before delivering the proof, so
                    // the freed capacity is available if handling the
                    // delivery re-drives this fetch.
                    push_shard_input(
                        &es,
                        local_shard,
                        ShardScopedInput::StateProofFetchFulfilled { ids: requested },
                    );
                    push_protocol_event(
                        &es,
                        local_shard,
                        ProtocolEvent::StateProofVerified {
                            anchor,
                            keys,
                            proof,
                        },
                    );
                    ResponseVerdict::Accept
                }),
            );
        }
    }
}

/// Marker type for the cross-shard provision fetch.
pub struct ProvisionBinding;

impl FetchBinding for ProvisionBinding {
    type Id = (ShardId, ShardId, BlockHeight);

    const NAME: &'static str = "provision";

    /// Cross-shard provisions are addressed by a single `(shard, height)` —
    /// each request targets exactly one scope.
    const PER_ID: bool = true;

    fn fetch_mut<S: ShardStorage>(shard: &mut ShardIo<S>) -> &mut Fetch<Self::Id> {
        &mut shard.cross_shard.provision
    }

    fn dispatch_chunk<N: Network>(
        ids: Vec<(ShardId, ShardId, BlockHeight)>,
        local_shard: ShardId,
        shard: ShardId,
        preferred: Option<ValidatorId>,
        class: Option<MessageClass>,
        network: &N,
        sender: &Sender<HostEvent>,
    ) {
        // PER_ID means the dispatcher hands us exactly one id at a time.
        debug_assert_eq!(ids.len(), 1);
        let (source_shard, target_shard, block_height) = ids[0];
        debug_assert_eq!(
            shard, source_shard,
            "ProvisionBinding routes to the source shard; the runner sets it from the variant"
        );
        // `target_shard` (the requester's shard) is the body field: the
        // source filters provisions by which shard is asking. Routing
        // shard `shard = source_shard` picks the responding committee.
        let request = GetProvisionsRequest {
            block_height,
            target_shard,
        };
        let es = sender.clone();
        network.request(
            shard,
            preferred,
            request,
            class,
            Box::new(move |result| {
                let push_fetch_failed = || {
                    push_shard_input(
                        &es,
                        local_shard,
                        ShardScopedInput::ProvisionsFetchFailed {
                            source_shard,
                            block_height,
                        },
                    );
                };
                let Ok(response) = result else {
                    push_fetch_failed();
                    return ResponseVerdict::Accept;
                };
                let Some(provisions) = response.provisions else {
                    push_fetch_failed();
                    return ResponseVerdict::Reject;
                };
                if provisions.source_shard() != source_shard
                    || provisions.target_shard() != target_shard
                    || provisions.block_height() != block_height
                {
                    tracing::warn!(
                        expected_source = source_shard.inner(),
                        got_source = provisions.source_shard().inner(),
                        expected_target = target_shard.inner(),
                        got_target = provisions.target_shard().inner(),
                        expected_height = block_height.inner(),
                        got_height = provisions.block_height().inner(),
                        "Dropping provision fetch response: scope mismatch"
                    );
                    push_fetch_failed();
                    return ResponseVerdict::Reject;
                }
                if provisions.transactions().is_empty() {
                    // Empty-but-scope-matched response is still a miss for
                    // the requester: the FSM has nothing to admit, so
                    // without an explicit `Failed` the id stays in_flight
                    // forever.
                    push_fetch_failed();
                    return ResponseVerdict::Reject;
                }
                push_protocol_event(
                    &es,
                    local_shard,
                    ProtocolEvent::UnverifiedProvisionsReceived { provisions },
                );
                ResponseVerdict::Accept
            }),
        );
    }
}

#[cfg(test)]
mod committed_tx_tests {
    use hyperscale_types::{
        BlockHash, BlockHeight, CommittedTxsRoot, Hash, committed_txs_root_from_hashes,
        prove_committed_tx_absent,
    };

    use super::*;

    fn tx(seed: u8) -> TxHash {
        TxHash::from(Hash::from_bytes(&[seed]))
    }

    /// A committed set of seeds 0..8 and the terminal that roots it.
    fn terminal_over(seeds: std::ops::Range<u8>) -> (PredecessorTerminal, Vec<TxHash>) {
        let mut members: Vec<TxHash> = seeds.map(tx).collect();
        members.sort_unstable();
        let terminal = PredecessorTerminal {
            shard: ShardId::leaf(1, 0),
            height: BlockHeight::new(9),
            block_hash: BlockHash::ZERO,
            committed_txs_root: committed_txs_root_from_hashes(members.iter()),
        };
        (terminal, members)
    }

    fn absence(members: &[TxHash], probe: TxHash) -> CommittedTxVerdict {
        CommittedTxVerdict::Absent(
            prove_committed_tx_absent(members, &probe).expect("probe is not a member"),
        )
    }

    /// The two verdicts map to the two answers, in the order asked.
    #[test]
    fn verifies_each_answer_against_the_terminal_root() {
        let (terminal, members) = terminal_over(0..8);
        let probe = tx(200);
        let answers = verified_answers(
            &[CommittedTxVerdict::Committed, absence(&members, probe)],
            terminal,
            &[members[0], probe],
        )
        .expect("both verdicts are usable");
        assert_eq!(answers, vec![(members[0], false), (probe, true)]);
    }

    /// An absence proof lifted from a different set doesn't verify
    /// against this terminal's root, and the whole response goes with it
    /// — including the `Committed` answer beside it, which on its own
    /// would have been fine.
    #[test]
    fn a_proof_against_another_root_condemns_the_response() {
        let (terminal, _) = terminal_over(0..8);
        let (_, other_members) = terminal_over(100..108);
        let probe = tx(200);
        assert!(
            verified_answers(
                &[
                    CommittedTxVerdict::Committed,
                    absence(&other_members, probe)
                ],
                terminal,
                &[tx(0), probe],
            )
            .is_none()
        );
    }

    /// A transaction the predecessor really committed cannot be shown
    /// absent: no proof over the rooted set brackets a member.
    #[test]
    fn a_member_has_no_absence_proof() {
        let (_, members) = terminal_over(0..8);
        assert!(prove_committed_tx_absent(&members, &members[3]).is_none());
    }

    /// Short and long answers are both malformed rather than partial —
    /// positional pairing has nothing to anchor on.
    #[test]
    fn a_length_mismatch_is_unusable() {
        let (terminal, _) = terminal_over(0..8);
        assert!(
            verified_answers(&[CommittedTxVerdict::Committed], terminal, &[tx(0), tx(1)]).is_none()
        );
        assert!(
            verified_answers(
                &[CommittedTxVerdict::Committed, CommittedTxVerdict::Committed],
                terminal,
                &[tx(0)],
            )
            .is_none()
        );
    }

    /// An empty set roots to `ZERO` and every absence over it is free,
    /// so a predecessor that committed nothing in its window answers
    /// every query without a tree to walk.
    #[test]
    fn an_empty_committed_set_proves_every_absence() {
        let terminal = PredecessorTerminal {
            shard: ShardId::leaf(1, 0),
            height: BlockHeight::new(9),
            block_hash: BlockHash::ZERO,
            committed_txs_root: CommittedTxsRoot::ZERO,
        };
        let probe = tx(7);
        let answers = verified_answers(&[absence(&[], probe)], terminal, &[probe])
            .expect("absence over an empty set verifies");
        assert_eq!(answers, vec![(probe, true)]);
    }
}
