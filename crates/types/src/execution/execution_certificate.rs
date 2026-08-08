//! [`ExecutionCertificate`] — aggregated 2f+1 signature over a wave's
//! per-tx outcomes.
//!
//! [`ExecutionCertificate`] is the raw wire form. Its verified form is
//! `Verified<ExecutionCertificate>`; predicate at
//! [`impl Verify<&ExecutionCertificateContext<'_>>`](Verify::verify) below.

use std::collections::HashMap;
use std::fmt::{self, Debug, Formatter};

use hyperscale_crypto::{ConsensusSignature, Verifier};
use hyperscale_hbor::error::{DecodeError as HborDecodeError, EncodeError as HborEncodeError};
use hyperscale_hbor::{
    Decoder as HborDecoder, Encoder as HborEncoder, HborDecode, HborEncode, HborWidth,
    bounded as hbor_bounded, to_vec as hbor_to_vec,
};
use thiserror::Error;

use crate::{
    AggregateSignature, BlockHeight, ConsensusPublicKey, ExecutionVote, ExecutionVoteMessage,
    GlobalReceiptRoot, Hash, MAX_TXS_PER_BLOCK, NetworkDefinition, RETENTION_HORIZON, ShardId,
    SignerBitfield, TickId, TxOutcome, ValidatorId, Verified, Verify, WeightedTimestamp,
    compute_global_receipt_root, signed_bytes,
};

/// Aggregated certificate for an execution wave.
///
/// Contains the signature aggregated signature from 2f+1 validators plus per-tx
/// outcomes so remote shards can extract individual transaction results.
pub struct ExecutionCertificate {
    tick_id: TickId,
    vote_anchor_ts: WeightedTimestamp,
    global_receipt_root: GlobalReceiptRoot,
    tx_outcomes: Vec<TxOutcome>,
    aggregated_signature: AggregateSignature,
    signers: SignerBitfield,
    /// Cached HBOR-encoded bytes. Populated at construction or after
    /// deserialization to avoid re-serialization on storage writes.
    cached_bytes: Option<Vec<u8>>,
}

impl Debug for ExecutionCertificate {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionCertificate")
            .field("tick_id", &self.tick_id)
            .field("vote_anchor_ts", &self.vote_anchor_ts)
            .field("global_receipt_root", &self.global_receipt_root)
            .field("tx_outcomes", &self.tx_outcomes)
            .field("aggregated_signature", &self.aggregated_signature)
            .field("signers", &self.signers)
            .finish_non_exhaustive()
    }
}

impl Clone for ExecutionCertificate {
    fn clone(&self) -> Self {
        Self {
            tick_id: self.tick_id,
            vote_anchor_ts: self.vote_anchor_ts,
            global_receipt_root: self.global_receipt_root,
            tx_outcomes: self.tx_outcomes.clone(),
            aggregated_signature: self.aggregated_signature,
            signers: self.signers.clone(),
            cached_bytes: self.cached_bytes.clone(),
        }
    }
}

impl PartialEq for ExecutionCertificate {
    fn eq(&self, other: &Self) -> bool {
        self.tick_id == other.tick_id
            && self.vote_anchor_ts == other.vote_anchor_ts
            && self.global_receipt_root == other.global_receipt_root
            && self.tx_outcomes == other.tx_outcomes
            && self.aggregated_signature == other.aggregated_signature
            && self.signers == other.signers
    }
}

impl Eq for ExecutionCertificate {}

// Manual codec: cached_bytes is derived, not serialized.
// Manual codec — the decode side recomputes the receipt root over the
// carried outcomes. The signature aggregate only commits to
// (global_receipt_root, tx_count), not to tx_outcomes content; without
// this check a Byzantine aggregator could ship a signature-valid EC whose
// outcomes don't hash to the signed root, slipping bogus per-tx results
// past every downstream consumer (gossip ingress, fetch ingress,
// Finalization admission).

impl HborWidth for ExecutionCertificate {
    const MIN_ENCODED_LEN: usize = 1;
}

impl HborEncode for ExecutionCertificate {
    fn encode(&self, encoder: &mut HborEncoder<'_>) -> Result<(), HborEncodeError> {
        encoder.nested(&self.tick_id)?;
        encoder.nested(&self.vote_anchor_ts)?;
        encoder.nested(&self.global_receipt_root)?;
        hbor_bounded::check_encoded_len("tx_outcomes", self.tx_outcomes.len(), MAX_TXS_PER_BLOCK)?;
        encoder.nested(&self.tx_outcomes)?;
        encoder.nested(&self.aggregated_signature)?;
        encoder.nested(&self.signers)
    }
}

impl HborDecode for ExecutionCertificate {
    fn decode(decoder: &mut HborDecoder<'_>) -> Result<Self, HborDecodeError> {
        let tick_id: TickId = decoder.nested()?;
        let vote_anchor_ts: WeightedTimestamp = decoder.nested()?;
        let global_receipt_root: GlobalReceiptRoot = decoder.nested()?;
        let tx_outcomes: Vec<TxOutcome> = decoder
            .descend(|decoder| hbor_bounded::decode_bounded_vec(decoder, MAX_TXS_PER_BLOCK))?;
        let aggregated_signature: AggregateSignature = decoder.nested()?;
        let signers: SignerBitfield = decoder.nested()?;
        if compute_global_receipt_root(&tx_outcomes) != global_receipt_root {
            return Err(HborDecodeError::FailedValidation(
                "tx outcomes do not hash to the signed receipt root",
            ));
        }
        let mut ec = Self {
            tick_id,
            vote_anchor_ts,
            global_receipt_root,
            tx_outcomes,
            aggregated_signature,
            signers,
            cached_bytes: None,
        };
        ec.populate_cached_bytes();
        Ok(ec)
    }
}

impl ExecutionCertificate {
    /// Create a new execution certificate.
    #[must_use]
    pub fn new(
        tick_id: TickId,
        vote_anchor_ts: WeightedTimestamp,
        global_receipt_root: GlobalReceiptRoot,
        tx_outcomes: Vec<TxOutcome>,
        aggregated_signature: AggregateSignature,
        signers: SignerBitfield,
    ) -> Self {
        let mut ec = Self {
            tick_id,
            vote_anchor_ts,
            global_receipt_root,
            tx_outcomes,
            aggregated_signature,
            signers,
            cached_bytes: None,
        };
        ec.populate_cached_bytes();
        ec
    }

    /// Self-contained wave identifier (shard + height + remote dependencies).
    #[must_use]
    pub const fn tick_id(&self) -> &TickId {
        &self.tick_id
    }

    /// Consensus height at which quorum was reached.
    ///
    /// Must match the `vote_anchor_ts` in the aggregated votes. Needed to
    /// reconstruct the signing message for signature verification.
    #[must_use]
    pub const fn vote_anchor_ts(&self) -> WeightedTimestamp {
        self.vote_anchor_ts
    }

    /// Merkle root over per-tx outcome leaves.
    #[must_use]
    pub const fn global_receipt_root(&self) -> GlobalReceiptRoot {
        self.global_receipt_root
    }

    /// Per-transaction outcomes (in wave order = block order).
    #[must_use]
    pub const fn tx_outcomes(&self) -> &Vec<TxOutcome> {
        &self.tx_outcomes
    }

    /// signature aggregated signature from 2f+1 validators.
    #[must_use]
    pub const fn aggregated_signature(&self) -> AggregateSignature {
        self.aggregated_signature
    }

    /// Which validators signed (bitfield indexed by committee position).
    #[must_use]
    pub const fn signers(&self) -> &SignerBitfield {
        &self.signers
    }

    /// The shard that produced this certificate.
    #[must_use]
    pub const fn shard_id(&self) -> ShardId {
        self.tick_id.shard_id()
    }

    /// Deadline past which this certificate is provably useless on every shard.
    ///
    /// Anchored on `vote_anchor_ts` — the wave's BFT-authenticated commit
    /// timestamp. Past `vote_anchor_ts + RETENTION_HORIZON` every tx in the
    /// wave has expired its `validity_range` and terminated (success or
    /// abort), so no shard can still reference this EC.
    #[must_use]
    pub fn deadline(&self) -> WeightedTimestamp {
        self.vote_anchor_ts.plus(RETENTION_HORIZON)
    }

    /// Block height (the block containing the wave's transactions).
    #[must_use]
    pub const fn block_height(&self) -> BlockHeight {
        self.tick_id.block_height()
    }

    /// Pre-serialized wire bytes, if available.
    #[must_use]
    pub fn cached_wire_bytes(&self) -> Option<&[u8]> {
        self.cached_bytes.as_deref()
    }

    /// Content hash over the full wire encoding (including
    /// `aggregated_signature` and `signers`). Distinguishes byte-identical
    /// retransmits — useful as an in-flight dedup key — while still treating
    /// different aggregations of the same logical EC as distinct, so a peer
    /// supplying a valid aggregation after a bad one isn't dropped.
    ///
    /// # Panics
    ///
    /// Panics if HBOR encoding fails — closed type, infallible in practice.
    #[must_use]
    pub fn wire_hash(&self) -> Hash {
        self.cached_bytes.as_deref().map_or_else(
            || {
                let bytes = hbor_to_vec(self).expect("EC HBOR encoding must succeed");
                Hash::from_parts(&[&bytes])
            },
            |bytes| Hash::from_parts(&[bytes]),
        )
    }

    fn populate_cached_bytes(&mut self) {
        self.cached_bytes = Some(hbor_to_vec(self).expect("EC HBOR encoding must succeed"));
    }

    /// Build the canonical signing message used by every constituent vote
    /// (and the aggregated certificate).
    ///
    /// Same message as [`ExecutionVote::signing_message`]; reconstructed
    /// from the EC's own fields so verifiers don't need a vote sample.
    #[must_use]
    pub fn signing_message(&self, network: &NetworkDefinition) -> Vec<u8> {
        signed_bytes(
            &ExecutionVoteMessage {
                vote_anchor_ts: self.vote_anchor_ts,
                tick_id: self.tick_id,
                shard_group: self.shard_id(),
                global_receipt_root: self.global_receipt_root,
                tx_count: u32::try_from(self.tx_outcomes.len()).unwrap_or(u32::MAX),
            },
            network,
        )
    }
}

/// Inputs the [`ExecutionCertificate`] verifier reads against. Borrows
/// everything; nothing is consumed.
#[derive(Debug, Clone, Copy)]
pub struct ExecutionCertificateContext<'a> {
    /// Network identifier — feeds the domain-separated signing message.
    pub network: &'a NetworkDefinition,
    /// Committee public keys in committee order. The certificate's
    /// `signers` bitfield indexes into this slice.
    pub public_keys: &'a [ConsensusPublicKey],
    /// Scheme verifier the aggregate check runs through.
    pub verifier: &'a dyn Verifier,
}

/// Failure modes of [`ExecutionCertificate`] verification.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ExecutionCertificateVerifyError {
    /// `signers` bitfield is empty but `aggregated_signature` is not the
    /// zero signature. An empty signer set must be paired with the zero
    /// signature; any other pairing is ill-formed.
    #[error("empty signer set paired with non-zero aggregated signature")]
    EmptySignersWithNonZeroSignature,
    /// The aggregated signature did not validate against the
    /// aggregated public key derived from `signers` over the canonical
    /// signing message. Also covers public-key aggregation failures.
    #[error("aggregated signature invalid")]
    BadAggregatedSignature,
}

/// Construction asserts: the aggregated signature validates against
/// the public key formed by aggregating `public_keys[i]` for every `i`
/// set in `signers`, over the canonical [`ExecutionVoteMessage`] derived
/// from the certificate's `(vote_anchor_ts, tick_id, shard_id,
/// global_receipt_root, tx_count)`. Empty signer sets must carry the
/// zero signature.
///
/// Construction goes through one of three gates:
///
/// - [`<ExecutionCertificate as Verify>::verify`](Verify::verify) —
///   runs the predicate against a committee public-key vector.
/// - [`Verified::<ExecutionCertificate>::aggregate`] — builds the
///   certificate from a quorum of verified votes; the predicate holds
///   by construction (each verified vote's signature aggregates into
///   the certificate's signature, the signers bitfield mirrors the
///   committee indices of those voters).
/// - [`Verified::<ExecutionCertificate>::from_persisted`] — re-wraps
///   a certificate that satisfied the predicate at write time.
impl Verify<&ExecutionCertificateContext<'_>> for ExecutionCertificate {
    type Error = ExecutionCertificateVerifyError;

    fn verify(&self, ctx: &ExecutionCertificateContext<'_>) -> Result<Verified<Self>, Self::Error> {
        let signer_keys: Vec<ConsensusPublicKey> = ctx
            .public_keys
            .iter()
            .enumerate()
            .filter(|(i, _)| self.signers.is_set(*i))
            .map(|(_, pk)| *pk)
            .collect();

        if signer_keys.is_empty() {
            if self.aggregated_signature == AggregateSignature::ZERO {
                return Ok(Verified::new_unchecked(self.clone()));
            }
            return Err(ExecutionCertificateVerifyError::EmptySignersWithNonZeroSignature);
        }

        let message = self.signing_message(ctx.network);
        if !ctx.verifier.verify_aggregate_same_message(
            &message,
            &self.aggregated_signature,
            &signer_keys,
        ) {
            return Err(ExecutionCertificateVerifyError::BadAggregatedSignature);
        }
        Ok(Verified::new_unchecked(self.clone()))
    }
}

impl Verified<ExecutionCertificate> {
    /// Build a [`Verified<ExecutionCertificate>`] from a quorum of
    /// verified votes.
    ///
    /// The caller is responsible for the quorum-power check; this gate
    /// only asserts the predicate (signature aggregation + signer-bit
    /// mapping). Every input vote is assumed to share the same signing
    /// message — the `VoteTracker` bucketing key `(global_receipt_root,
    /// vote_anchor_ts)` plus the per-wave `tick_id` and `shard_id`
    /// uniquely determine that message, so a single bucket's contents
    /// satisfy this contract by construction.
    ///
    /// Validators not in `committee` contribute their signature to the
    /// aggregate but no bit to `signers`; the resulting EC would fail
    /// verify. The caller filters non-committee voters upstream.
    ///
    /// # Panics
    ///
    /// Panics if `votes` is empty, or if signature aggregation of the
    /// individually-verified signatures fails — both indicate an
    /// upstream invariant violation (predicate bypass, sub-quorum
    /// input, or scheme library bug).
    #[must_use]
    pub fn aggregate(
        verifier: &dyn Verifier,
        tick_id: &TickId,
        global_receipt_root: GlobalReceiptRoot,
        votes: &[Verified<ExecutionVote>],
        committee: &[ValidatorId],
    ) -> Self {
        let tx_outcomes = votes
            .iter()
            .find(|v| compute_global_receipt_root(v.tx_outcomes()) == global_receipt_root)
            .map(|v| v.tx_outcomes().to_vec())
            .expect("verified votes guarantee at least one with matching outcomes");

        let committee_index: HashMap<ValidatorId, usize> = committee
            .iter()
            .enumerate()
            .map(|(idx, &vid)| (vid, idx))
            .collect();

        let mut seen_validators: std::collections::HashSet<ValidatorId> =
            std::collections::HashSet::new();
        let mut unique_votes: Vec<&Verified<ExecutionVote>> = votes
            .iter()
            .filter(|vote| seen_validators.insert(vote.validator()))
            .collect();
        // Fold in bitfield (committee-position) order — the verifier
        // recomputes the aggregate in set-bit order, and order-sensitive
        // schemes require the fold to match.
        unique_votes.sort_by_key(|vote| {
            committee_index
                .get(&vote.validator())
                .copied()
                .unwrap_or(usize::MAX)
        });

        let signatures: Vec<ConsensusSignature> =
            unique_votes.iter().map(|vote| vote.signature()).collect();
        let aggregated_signature = if signatures.is_empty() {
            AggregateSignature::ZERO
        } else {
            verifier
                .aggregate(&signatures)
                .expect("aggregation of upstream-verified signatures cannot fail")
        };
        let mut signers = SignerBitfield::new(committee.len());
        for vote in &unique_votes {
            if let Some(&idx) = committee_index.get(&vote.validator()) {
                signers.set(idx);
            }
        }

        let vote_anchor_ts = votes
            .first()
            .map_or(WeightedTimestamp::ZERO, |v| v.vote_anchor_ts());

        // SAFETY: every input vote satisfies the `ExecutionVote`
        // predicate against its own pubkey for the shared signing
        // message determined by `(vote_anchor_ts, tick_id,
        // shard_id, global_receipt_root, tx_count)`. Aggregating
        // those signatures and mirroring the committee indices in
        // `signers` produces an EC whose predicate is structurally
        // equivalent: the aggregated pubkey at verify time recombines
        // the same per-validator pubkeys, and signature aggregate-verify
        // succeeds.
        Self::new_unchecked(ExecutionCertificate::new(
            *tick_id,
            vote_anchor_ts,
            global_receipt_root,
            tx_outcomes,
            aggregated_signature,
            signers,
        ))
    }

    /// Re-wrap a certificate that satisfied the predicate at write
    /// time. ECs ride into storage embedded inside `Verified<Finalization>`
    /// values inside the `Verified<CertifiedBlock>` argument to
    /// `commit_block`, so unverified ECs can't reach the write path.
    /// Storage rehydration paths use this gate to avoid re-running
    /// aggregation on every load.
    #[must_use]
    pub const fn from_persisted(cert: ExecutionCertificate) -> Self {
        // SAFETY: the certificate's predicate held at write time;
        // storage is the trust source. Mirrors
        // `Verified::<QuorumCertificate>::from_persisted` on the
        // shard side.
        Self::new_unchecked(cert)
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_crypto::Signer;
    use hyperscale_crypto_bls::{BlsSigner, BlsVerifier};
    use hyperscale_hbor::from_slice as hbor_from_slice;

    use super::*;
    use crate::{BlockHash, BlockHeight, ExecutionOutcome, GlobalReceiptHash, TxHash};

    fn outcome(seed: u8) -> TxOutcome {
        TxOutcome::new(
            TxHash::from(Hash::from_bytes(&[seed; 4])),
            ExecutionOutcome::Succeeded {
                receipt_hash: GlobalReceiptHash::from_raw(Hash::from_bytes(&[seed + 100; 4])),
            },
        )
    }

    fn tick_id() -> TickId {
        TickId::new(ShardId::leaf(1, 0), BlockHeight::new(7))
    }

    /// Build a signed vote with the given signing key. Used for fixture
    /// construction; the resulting `Verified<ExecutionVote>` would also
    /// satisfy `<ExecutionVote as Verify>::verify` against `sk.public_key()`.
    fn signed_vote(
        net: &NetworkDefinition,
        sk: &BlsSigner,
        validator: u64,
        outcomes: Vec<TxOutcome>,
    ) -> Verified<ExecutionVote> {
        Verified::<ExecutionVote>::sign_local(
            net,
            BlockHash::from_raw(Hash::from_bytes(b"block")),
            BlockHeight::new(7),
            WeightedTimestamp::from_millis(11),
            tick_id(),
            ShardId::leaf(1, 0),
            outcomes,
            ValidatorId::new(validator),
            sk,
        )
        .expect("sign")
    }

    /// Aggregate produces an EC whose predicate verifies against the
    /// matching committee public keys — the canonical sign-then-verify
    /// round trip across the typed gates.
    #[test]
    fn aggregate_roundtrips_through_verify() {
        let net = NetworkDefinition::simulator();
        let committee: Vec<ValidatorId> = (0..4).map(ValidatorId::new).collect();
        let sks: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let pks: Vec<ConsensusPublicKey> = sks.iter().map(BlsSigner::public_key).collect();
        let outcomes = vec![outcome(1), outcome(2)];
        let root = compute_global_receipt_root(&outcomes);

        let votes: Vec<Verified<ExecutionVote>> = (0..4)
            .map(|i| signed_vote(&net, &sks[usize::try_from(i).unwrap()], i, outcomes.clone()))
            .collect();

        let cert = Verified::<ExecutionCertificate>::aggregate(
            &BlsVerifier,
            &tick_id(),
            root,
            &votes,
            &committee,
        );

        let ctx = ExecutionCertificateContext {
            verifier: &BlsVerifier,
            network: &net,
            public_keys: &pks,
        };
        let raw = cert.into_inner();
        raw.verify(&ctx)
            .expect("aggregate output must satisfy its own predicate");
    }

    /// A certificate whose `aggregated_signature` was tampered with
    /// fails the signature check; the verifier returns `BadAggregatedSignature`.
    #[test]
    fn verify_rejects_bad_aggregated_signature() {
        let net = NetworkDefinition::simulator();
        let committee: Vec<ValidatorId> = (0..4).map(ValidatorId::new).collect();
        let sks: Vec<BlsSigner> = (0..4).map(|_| BlsSigner::generate()).collect();
        let pks: Vec<ConsensusPublicKey> = sks.iter().map(BlsSigner::public_key).collect();
        let outcomes = vec![outcome(1)];
        let root = compute_global_receipt_root(&outcomes);

        let votes: Vec<Verified<ExecutionVote>> = (0..4)
            .map(|i| signed_vote(&net, &sks[usize::try_from(i).unwrap()], i, outcomes.clone()))
            .collect();
        let cert = Verified::<ExecutionCertificate>::aggregate(
            &BlsVerifier,
            &tick_id(),
            root,
            &votes,
            &committee,
        )
        .into_inner();

        let tampered = ExecutionCertificate::new(
            *cert.tick_id(),
            cert.vote_anchor_ts(),
            cert.global_receipt_root(),
            cert.tx_outcomes().clone(),
            AggregateSignature::new([0xFF; 96]),
            cert.signers().clone(),
        );

        let ctx = ExecutionCertificateContext {
            verifier: &BlsVerifier,
            network: &net,
            public_keys: &pks,
        };
        assert_eq!(
            tampered.verify(&ctx),
            Err(ExecutionCertificateVerifyError::BadAggregatedSignature)
        );
    }

    /// A certificate with an empty signer set but a non-zero aggregated
    /// signature is ill-formed and rejected before the signature check runs.
    #[test]
    fn verify_rejects_empty_signers_with_nonzero_signature() {
        let net = NetworkDefinition::simulator();
        let pks: Vec<ConsensusPublicKey> =
            (0..4).map(|_| BlsSigner::generate().public_key()).collect();

        let outcomes = vec![outcome(1)];
        let root = compute_global_receipt_root(&outcomes);
        let cert = ExecutionCertificate::new(
            tick_id(),
            WeightedTimestamp::from_millis(11),
            root,
            outcomes,
            AggregateSignature::new([0xAA; 96]),
            SignerBitfield::new(4),
        );

        let ctx = ExecutionCertificateContext {
            verifier: &BlsVerifier,
            network: &net,
            public_keys: &pks,
        };
        assert_eq!(
            cert.verify(&ctx),
            Err(ExecutionCertificateVerifyError::EmptySignersWithNonZeroSignature)
        );
    }

    /// A certificate with an empty signer set and the zero signature is
    /// well-formed (this is the "no validators voted" shape) and
    /// verifies.
    #[test]
    fn verify_accepts_empty_signers_with_zero_signature() {
        let net = NetworkDefinition::simulator();
        let pks: Vec<ConsensusPublicKey> =
            (0..4).map(|_| BlsSigner::generate().public_key()).collect();

        let outcomes = vec![outcome(1)];
        let root = compute_global_receipt_root(&outcomes);
        let cert = ExecutionCertificate::new(
            tick_id(),
            WeightedTimestamp::from_millis(11),
            root,
            outcomes,
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );

        let ctx = ExecutionCertificateContext {
            verifier: &BlsVerifier,
            network: &net,
            public_keys: &pks,
        };
        cert.verify(&ctx)
            .expect("empty signers + zero signature must verify");
    }

    /// Aggregation maps each voter to their committee index in the
    /// signer bitfield; non-voters' bits stay clear. Catches bitfield
    /// off-by-one regressions in `Verified::<EC>::aggregate`.
    #[test]
    fn aggregate_produces_signer_bitfield_in_committee_order() {
        let net = NetworkDefinition::simulator();
        let committee: Vec<ValidatorId> = (0..4).map(ValidatorId::new).collect();
        let sk1 = BlsSigner::generate();
        let sk3 = BlsSigner::generate();
        let outcomes = vec![outcome(1)];
        let root = compute_global_receipt_root(&outcomes);

        let votes = vec![
            signed_vote(&net, &sk1, 1, outcomes.clone()),
            signed_vote(&net, &sk3, 3, outcomes.clone()),
        ];

        let cert = Verified::<ExecutionCertificate>::aggregate(
            &BlsVerifier,
            &tick_id(),
            root,
            &votes,
            &committee,
        )
        .into_inner();
        assert!(cert.signers().is_set(1));
        assert!(cert.signers().is_set(3));
        assert!(!cert.signers().is_set(0));
        assert!(!cert.signers().is_set(2));
        assert_eq!(cert.tx_outcomes(), &outcomes);
    }

    /// Duplicate votes from the same validator collapse to a single
    /// signer bit and a single signature contribution.
    #[test]
    fn aggregate_dedups_votes_from_same_validator() {
        let net = NetworkDefinition::simulator();
        let committee = vec![ValidatorId::new(0), ValidatorId::new(1)];
        let sk0 = BlsSigner::generate();
        let outcomes = vec![outcome(1)];
        let root = compute_global_receipt_root(&outcomes);

        let votes = vec![
            signed_vote(&net, &sk0, 0, outcomes.clone()),
            signed_vote(&net, &sk0, 0, outcomes),
        ];

        let cert = Verified::<ExecutionCertificate>::aggregate(
            &BlsVerifier,
            &tick_id(),
            root,
            &votes,
            &committee,
        )
        .into_inner();
        assert!(cert.signers().is_set(0));
        assert!(!cert.signers().is_set(1));
        assert_eq!(cert.signers().count_ones(), 1);
    }

    /// An EC whose outcomes' `work` was mutated after the root was
    /// fixed fails the decode-side receipt-root recompute — the leaf
    /// covers work, so an aggregator cannot ship a signature-valid EC
    /// with forged work.
    #[test]
    fn decode_rejects_tampered_work() {
        let outcomes = vec![TxOutcome::attesting(
            TxHash::from(Hash::from_bytes(b"worked-tx")),
            ExecutionOutcome::Aborted,
            7,
        )];
        let root = compute_global_receipt_root(&outcomes);

        let forged: Vec<TxOutcome> = outcomes
            .iter()
            .map(|o| TxOutcome::attesting(o.tx_hash(), o.outcome().clone(), 999))
            .collect();
        let cert = ExecutionCertificate::new(
            tick_id(),
            WeightedTimestamp::from_millis(11),
            root,
            forged,
            AggregateSignature::ZERO,
            SignerBitfield::new(4),
        );

        let bytes = hbor_to_vec(&cert).expect("encode");
        let err = hbor_from_slice::<ExecutionCertificate>(&bytes)
            .expect_err("tampered work must fail the receipt-root recompute");
        assert!(matches!(err, HborDecodeError::FailedValidation(_)));
    }

    /// An EC verified against a public-key slice that doesn't match the
    /// signing committee fails the signature check.
    #[test]
    fn verify_rejects_wrong_public_keys() {
        let net = NetworkDefinition::simulator();
        let committee = vec![ValidatorId::new(0), ValidatorId::new(1)];
        let sk0 = BlsSigner::generate();
        let sk1 = BlsSigner::generate();
        let outcomes = vec![outcome(1)];
        let root = compute_global_receipt_root(&outcomes);

        let votes = vec![
            signed_vote(&net, &sk0, 0, outcomes.clone()),
            signed_vote(&net, &sk1, 1, outcomes),
        ];
        let cert = Verified::<ExecutionCertificate>::aggregate(
            &BlsVerifier,
            &tick_id(),
            root,
            &votes,
            &committee,
        )
        .into_inner();

        let wrong_pks: Vec<ConsensusPublicKey> =
            (0..2).map(|_| BlsSigner::generate().public_key()).collect();
        let ctx = ExecutionCertificateContext {
            verifier: &BlsVerifier,
            network: &net,
            public_keys: &wrong_pks,
        };
        assert_eq!(
            cert.verify(&ctx),
            Err(ExecutionCertificateVerifyError::BadAggregatedSignature)
        );
    }
}
