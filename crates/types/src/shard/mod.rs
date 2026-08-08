//! Shard consensus types.
//!
//! - [`block`]: [`Block`] (the Live/Sealed enum).
//! - [`certified`]: [`CertifiedBlock`] pairing of a block with its certifying QC.
//! - [`certified_header`]: [`CertifiedBlockHeader`] cross-shard trust attestation.
//! - [`evidence`]: [`ShardVoteEquivocation`] self-proving double-vote evidence.
//! - [`fork_fence`]: [`ForkFence`](fork_fence::ForkFence) — the gossip-timed
//!   quiesce every cross-shard consumer engages against a proven fork.
//! - [`header`]: [`BlockHeader`] (shard-voted metadata).
//! - [`limits`]: protocol-level caps on per-block payload sizes.
//! - [`load`]: [`ShardLoad`](load::ShardLoad) — the attested-work and
//!   stored-byte totals a header attests, which the beacon reweights
//!   emission by.
//! - [`manifest`]: hash-level [`BlockManifest`] and denormalized [`BlockMetadata`].
//! - [`quorum_certificate`]: [`QuorumCertificate`] aggregating shard consensus votes.
//! - [`roots`]: per-block merkle root helpers used by [`BlockHeader`] consumers.
//! - [`storage_commit`]: type-erased [`PreparedCommit`](storage_commit::PreparedCommit)
//!   closure, [`SyncHint`](storage_commit::SyncHint), and
//!   [`BeaconWitnessCommit`](storage_commit::BeaconWitnessCommit) payload
//!   threaded through shard block commits.
//! - [`timeout`]: [`Timeout`] view-change share that drives the pacemaker.
//! - [`vote`]: [`BlockVote`] shard consensus vote.
//! - [`vote_registers`]: snapshot type for the two monotone safe-vote
//!   registers ([`SafeVoteRegisters`](vote_registers::SafeVoteRegisters)).
//! - [`witness_sources`]: [`WitnessSources`] — the proposer-supplied
//!   beacon-witness inputs a block carries.

#[allow(clippy::module_inception)]
mod block;
pub mod certified;
pub mod certified_header;
pub mod chain_origin;
pub mod commit_proof;
pub mod evidence;
pub mod fork_fence;
pub mod header;
pub mod inventory;
pub mod limits;
pub mod load;
pub mod manifest;
pub mod quorum_certificate;
pub mod reshape;
pub mod roots;
pub mod storage_commit;
pub mod timeout;
pub mod vote;
pub mod vote_registers;
pub mod witness_sources;

pub use block::{
    Block, SharedCertificates, SharedProvisions, SharedTransactions, TerminalRef,
    VerifiedBlockAssembleError, shared_transactions_from_raw, work_over_certificates,
};
pub use witness_sources::{SharedWitnessSources, WitnessSources};

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use hyperscale_hbor::{
        DecodeError, Hbor, from_slice as hbor_from_slice, to_vec as hbor_to_vec,
    };

    use super::*;
    use crate::test_utils::{install_stub_vm_statics, stub_transaction, test_validity_range};
    use crate::{
        AggregateSignature, BeaconWitnessLeafCount, BeaconWitnessRoot, BlockHash, BlockHeader,
        BlockHeight, CertificateRoot, ChainOrigin, ExecutionCertificate, ExecutionOutcome,
        FinalizedWave, GlobalReceiptHash, GlobalReceiptRoot, Hash, LocalReceiptRoot,
        ProposerTimestamp, ProvisionsRoot, QuorumCertificate, RevealChain, Round, ShardId,
        ShardLoad, SignerBitfield, StateRoot, TransactionRoot, TxHash, TxOutcome, ValidatorId,
        Verifiable, Verified, WaveCertificate, WaveId, WeightedTimestamp, WorkInFlight,
    };

    #[test]
    fn test_block_header_hash_deterministic() {
        let header = BlockHeader::new(
            ShardId::leaf(1, 0),
            BlockHeight::new(1),
            BlockHash::from_raw(Hash::from_bytes(b"parent")),
            QuorumCertificate::genesis(ShardId::leaf(1, 0), ChainOrigin::ROOT),
            ValidatorId::new(0),
            ProposerTimestamp::from_millis(1_234_567_890),
            Round::INITIAL,
            false,
            StateRoot::ZERO,
            TransactionRoot::ZERO,
            CertificateRoot::ZERO,
            LocalReceiptRoot::ZERO,
            ProvisionsRoot::ZERO,
            Vec::new(),
            std::collections::BTreeMap::new(),
            WorkInFlight::ZERO,
            BeaconWitnessRoot::ZERO,
            BeaconWitnessLeafCount::ZERO,
            BeaconWitnessLeafCount::ZERO,
            RevealChain::ZERO,
            None,
            None,
            ShardLoad::ZERO,
        );

        let hash1 = header.hash();
        let hash2 = header.hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_genesis_block() {
        let genesis = Block::genesis(
            ShardId::leaf(1, 0),
            ValidatorId::new(0),
            StateRoot::ZERO,
            ChainOrigin::ROOT,
        );

        assert!(genesis.is_genesis());
        assert_eq!(genesis.height(), BlockHeight::new(0));
        assert_eq!(genesis.transaction_count(), 0);
        assert_eq!(genesis.header().transaction_root(), TransactionRoot::ZERO);
        assert_eq!(
            genesis.header().parent_qc(),
            &QuorumCertificate::genesis(ShardId::leaf(1, 0), ChainOrigin::ROOT)
        );
    }

    #[test]
    fn test_compute_transaction_root_empty() {
        let root = Verified::<TransactionRoot>::compute(&[]).into_inner();
        assert_eq!(root, TransactionRoot::ZERO);
    }

    #[test]
    fn test_compute_transaction_root_deterministic() {
        install_stub_vm_statics();
        let tx = Arc::new(Verifiable::from(stub_transaction(
            [1; 16],
            &[[2; 16]],
            1_000,
            test_validity_range(),
        )));

        let root1 = Verified::<TransactionRoot>::compute(std::slice::from_ref(&tx)).into_inner();
        let root2 = Verified::<TransactionRoot>::compute(std::slice::from_ref(&tx)).into_inner();
        assert_eq!(root1, root2);
        assert_ne!(root1, TransactionRoot::ZERO);
    }

    #[test]
    fn test_compute_certificate_root_empty() {
        let root = Verified::<CertificateRoot>::compute(&[]).into_inner();
        assert_eq!(root, CertificateRoot::ZERO);
    }

    #[test]
    fn test_compute_certificate_root_deterministic() {
        let make_fw = |seed: u8| -> Arc<Verifiable<FinalizedWave>> {
            let ec = Arc::new(ExecutionCertificate::new(
                WaveId::new(
                    ShardId::leaf(1, 0),
                    BlockHeight::new(10),
                    BTreeSet::from([ShardId::leaf(1, 1)]),
                ),
                WeightedTimestamp::from_millis(11),
                GlobalReceiptRoot::from_raw(Hash::from_bytes(&[seed + 100; 4])),
                vec![TxOutcome::new(
                    TxHash::from(Hash::from_bytes(&[seed; 4])),
                    ExecutionOutcome::Succeeded {
                        receipt_hash: GlobalReceiptHash::from_raw(Hash::from_bytes(
                            &[seed + 50; 4],
                        )),
                    },
                )],
                AggregateSignature::new([0u8; 96]),
                SignerBitfield::new(4),
            ));
            Arc::new(
                FinalizedWave::new(
                    Arc::new(WaveCertificate::new(
                        WaveId::new(
                            ShardId::leaf(1, 0),
                            BlockHeight::new(10),
                            BTreeSet::from([ShardId::leaf(1, 1)]),
                        ),
                        vec![ec],
                    )),
                    vec![],
                )
                .into(),
            )
        };

        let certs = vec![make_fw(1), make_fw(2)];
        let root1 = Verified::<CertificateRoot>::compute(&certs).into_inner();
        let root2 = Verified::<CertificateRoot>::compute(&certs).into_inner();
        assert_eq!(root1, root2);
        assert_ne!(root1, CertificateRoot::ZERO);
    }

    #[test]
    fn test_compute_certificate_root_single_cert() {
        let ec = Arc::new(ExecutionCertificate::new(
            WaveId::new(ShardId::leaf(1, 0), BlockHeight::new(10), BTreeSet::new()),
            WeightedTimestamp::from_millis(11),
            GlobalReceiptRoot::from_raw(Hash::from_bytes(b"receipt")),
            vec![TxOutcome::new(
                TxHash::from(Hash::from_bytes(b"tx1")),
                ExecutionOutcome::Succeeded {
                    receipt_hash: GlobalReceiptHash::from_raw(Hash::from_bytes(b"rh")),
                },
            )],
            AggregateSignature::new([0u8; 96]),
            SignerBitfield::new(4),
        ));
        let cert = Arc::new(WaveCertificate::new(
            WaveId::new(ShardId::leaf(1, 0), BlockHeight::new(10), BTreeSet::new()),
            vec![ec],
        ));
        let expected_receipt_hash = cert.receipt_hash();
        let fw: Arc<Verifiable<FinalizedWave>> = Arc::new(FinalizedWave::new(cert, vec![]).into());

        let root = Verified::<CertificateRoot>::compute(std::slice::from_ref(&fw)).into_inner();
        // Single cert: certificate_root should equal the cert's receipt_hash
        assert_eq!(root.into_raw(), expected_receipt_hash.into_raw());
    }

    #[test]
    fn test_genesis_certificate_root_is_zero() {
        let genesis = Block::genesis(
            ShardId::leaf(1, 0),
            ValidatorId::new(0),
            StateRoot::ZERO,
            ChainOrigin::ROOT,
        );
        assert_eq!(genesis.header().certificate_root(), CertificateRoot::ZERO);
    }

    #[test]
    fn certified_block_decode_rejects_qc_block_hash_mismatch() {
        use crate::{CertifiedBlock, ShardLoad};

        // Forge a non-genesis block paired with a genesis QC. Without the
        // pairing check at decode this slips past the synced-block apply
        // path's `qc.is_genesis()` quorum-power bypass.
        let mut bad_block = Block::genesis(
            ShardId::leaf(1, 0),
            ValidatorId::new(0),
            StateRoot::ZERO,
            ChainOrigin::ROOT,
        )
        .into_sealed()
        .into_live(Arc::new(Vec::new()));
        if let Block::Live { ref mut header, .. } = bad_block {
            *header = BlockHeader::new(
                header.shard_id(),
                BlockHeight::new(7),
                header.parent_block_hash(),
                header.parent_qc().clone(),
                header.proposer(),
                header.timestamp(),
                header.round(),
                header.is_fallback(),
                header.state_root(),
                header.transaction_root(),
                header.certificate_root(),
                header.local_receipt_root(),
                header.provision_root(),
                header.cross_shard_txs().clone(),
                header.provision_tx_roots().clone(),
                header.work_in_flight(),
                header.beacon_witness_root(),
                header.beacon_witness_leaf_count(),
                header.beacon_witness_base(),
                RevealChain::ZERO,
                None,
                None,
                ShardLoad::ZERO,
            );
        }
        let genesis_qc = QuorumCertificate::genesis(ShardId::leaf(1, 0), ChainOrigin::ROOT);
        let bytes = hbor_to_vec(&CertifiedBlockWire {
            block: bad_block,
            qc: genesis_qc,
        })
        .unwrap();
        let err = hbor_from_slice::<CertifiedBlock>(&bytes).unwrap_err();
        assert!(matches!(err, DecodeError::FailedValidation(_)));
    }

    /// Wire-shape twin of `CertifiedBlock` that skips the pairing invariant
    /// during encode, so tests can construct adversarial byte streams.
    #[derive(Hbor)]
    struct CertifiedBlockWire {
        block: Block,
        qc: QuorumCertificate,
    }
}
