//! Golden-bytes pins for the wire structs whose signature/key fields the
//! crypto seam touches. The expected hex was captured from the SBOR
//! encodings before the role newtypes were introduced; any drift means a
//! wire-format break, which the seam must not cause.

use hex::encode as hex_encode;
use hyperscale_types::{
    AggregateSignature, BlockHash, BlockHeight, BlockVote, ConsensusPublicKey, ConsensusSignature,
    Hash, PcCompactVote, PcQc1, PcValueElement, PcVector, PositionalBundle, ProposerTimestamp,
    QuorumCertificate, Round, ShardId, SignerBitfield, ValidatorId, ValidatorInfo,
    WeightedTimestamp,
};
use sbor::BasicEncode;
use sbor::prelude::basic_encode;

fn assert_golden<T: BasicEncode>(value: &T, expected_hex: &str, label: &str) {
    let actual = hex_encode(basic_encode(value).unwrap());
    assert_eq!(
        actual, expected_hex,
        "{label}: SBOR encoding drifted from the pre-seam golden bytes"
    );
}

fn golden_signers() -> SignerBitfield {
    let mut signers = SignerBitfield::new(4);
    signers.set(0);
    signers.set(2);
    signers
}

#[test]
fn quorum_certificate_golden_bytes() {
    let qc = QuorumCertificate::new(
        BlockHash::from_raw(Hash::from_bytes(b"golden-qc-block")),
        ShardId::leaf(2, 0b10),
        BlockHeight::new(42),
        BlockHash::from_raw(Hash::from_bytes(b"golden-qc-parent")),
        Round::new(3),
        golden_signers(),
        AggregateSignature::new([0x22; 96]),
        WeightedTimestamp::from_millis(1_700_000_000_123),
    );
    assert_golden(&qc, EXPECTED_QC, "QuorumCertificate");
}

#[test]
fn block_vote_golden_bytes() {
    let vote = BlockVote::from_parts(
        BlockHash::from_raw(Hash::from_bytes(b"golden-vote-block")),
        ShardId::leaf(1, 0b1),
        BlockHeight::new(7),
        Round::new(1),
        ValidatorId::new(5),
        ConsensusSignature::new([0x33; 96]),
        ProposerTimestamp::from_millis(1_700_000_000_456),
    );
    assert_golden(&vote, EXPECTED_BLOCK_VOTE, "BlockVote");
}

#[test]
fn validator_info_golden_bytes() {
    let info = ValidatorInfo {
        validator_id: ValidatorId::new(9),
        public_key: ConsensusPublicKey::new([0x44; 48]),
    };
    assert_golden(&info, EXPECTED_VALIDATOR_INFO, "ValidatorInfo");
}

#[test]
fn pc_qc1_golden_bytes() {
    let x = PcVector::new([
        PcValueElement::from_digest([0x55; 32], b"golden"),
        PcValueElement::from_digest([0x66; 32], b"golden"),
    ]);
    let x_signers = PositionalBundle::new(
        golden_signers(),
        vec![
            PcCompactVote::new(2, None),
            PcCompactVote::new(1, Some(PcValueElement::from_digest([0x77; 32], b"golden"))),
        ],
    );
    let qc1 = PcQc1::new(x, x_signers, AggregateSignature::new([0x88; 96]));
    assert_golden(&qc1, EXPECTED_PC_QC1, "PcQc1");
}

const EXPECTED_QC: &str = "5b2108200720e7c8d8fecb84404480148f65172ce95b9984e1ec56f84494e6ad90391dc36cd7210209020000000a02000000000000000a2a000000000000002007202356d22ed37a6538fc945a8e30ea532612029962653bf993560b365ae3fe594c0a03000000000000002102200701050a04000000000000002007602222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222220a7b68e5cf8b010000";
const EXPECTED_BLOCK_VOTE: &str = "5b2107200720b3e13454f2b0dab7471e3e5db927fd492e114b13595a462f5a4c066bd516783b210209010000000a01000000000000000a07000000000000000a01000000000000000a05000000000000002007603333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333330ac869e5cf8b010000";
const EXPECTED_VALIDATOR_INFO: &str = "5b21020a0900000000000000200730444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444";
const EXPECTED_PC_QC1: &str = "5b2103202002072055555555555555555555555555555555555555555555555555555555555555550720666666666666666666666666666666666666666666666666666666666666666621022102200701050a04000000000000002021020209020000002200000209010000002201012007207777777777777777777777777777777777777777777777777777777777777777200760888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888";
