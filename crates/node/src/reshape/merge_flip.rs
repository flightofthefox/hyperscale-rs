//! The merged-parent genesis flip's deterministic core.
//!
//! A keeper reforms the parent from its two children's terminated
//! chains: each child's terminal block `B` — the crossing that ends its
//! chain — and its certifying quorum certificate. Every input to the
//! genesis is in those two blocks and the schedule both halves already
//! read, so a keeper derives it at the cut rather than an epoch later
//! when the beacon composes the same value.

use hyperscale_types::{
    Block, BlockHeader, ChainOrigin, QuorumCertificate, ShardId, SplitChildRoots, WeightedTimestamp,
};

/// Derive a merged parent's genesis block and chain origin from its two
/// children's certified terminal blocks.
///
/// `left`/`right` are the terminal blocks of `parent`'s `path‖0` and
/// `path‖1` children in canonical order — the order
/// [`BlockHeader::merge_parent_genesis`] composes — each with a QC
/// certifying it. The merged root is the internal node over the two
/// terminal subtree roots, the inverse of the composition a split
/// verifies; each side is attested by its own chain, so the pair cannot
/// name a subtree neither terminal committed. The first block continues
/// both height lines at `max(h_p0, h_p1) + 1`.
///
/// `cut_wt` is the instant the children terminate at — the end of their
/// scheduled terminal window, which the beacon fold composes from the
/// same schedule. The terminal QCs confirm the terminals are certified
/// but their own weighted timestamps are never the clock: the fold binds
/// whichever QC `canonical_boundary_qcs` ranked highest across the
/// committed proposal set, so no keeper could reproduce that choice.
///
/// # Errors
///
/// Fails when a quorum certificate does not certify its terminal block.
pub fn merge_genesis_from_terminals(
    parent: ShardId,
    left: (&BlockHeader, &QuorumCertificate),
    right: (&BlockHeader, &QuorumCertificate),
    cut_wt: WeightedTimestamp,
) -> Result<(Block, ChainOrigin), String> {
    let (left_terminal, left_qc) = left;
    let (right_terminal, right_qc) = right;
    if left_qc.block_hash() != left_terminal.hash() {
        return Err("the left quorum certificate does not certify the left terminal".to_string());
    }
    if right_qc.block_hash() != right_terminal.hash() {
        return Err("the right quorum certificate does not certify the right terminal".to_string());
    }
    let composed = SplitChildRoots {
        left: left_terminal.state_root(),
        right: right_terminal.state_root(),
    }
    .composed_root();
    let genesis = Block::merge_parent_genesis(
        parent,
        composed,
        (left_terminal.hash(), left_terminal.height()),
        (right_terminal.hash(), right_terminal.height()),
        cut_wt,
    );
    let origin = ChainOrigin {
        genesis_height: genesis.height(),
        anchor_wt: cut_wt,
    };
    Ok((genesis, origin))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use hyperscale_types::{
        AggregateSignature, BeaconWitnessLeafCount, BeaconWitnessRoot, BlockHash, BlockHeight,
        CertificateRoot, ChainOrigin, Hash, InFlightCount, LocalReceiptRoot, ProposerTimestamp,
        ProvisionsRoot, QuorumCertificate, Round, ShardId, SignerBitfield, SplitChildRoots,
        StateRoot, TransactionRoot, ValidatorId, WeightedTimestamp,
    };

    use super::*;

    fn terminal_header(shard: ShardId, height: u64, state_root: StateRoot) -> BlockHeader {
        BlockHeader::new(
            shard,
            BlockHeight::new(height),
            BlockHash::from_raw(Hash::from_bytes(b"parent")),
            QuorumCertificate::genesis(shard, ChainOrigin::ROOT),
            ValidatorId::new(2),
            ProposerTimestamp::ZERO,
            Round::new(7),
            false,
            state_root,
            TransactionRoot::ZERO,
            CertificateRoot::ZERO,
            LocalReceiptRoot::ZERO,
            ProvisionsRoot::ZERO,
            Vec::new(),
            BTreeMap::new(),
            InFlightCount::ZERO,
            BeaconWitnessRoot::ZERO,
            BeaconWitnessLeafCount::ZERO,
            BeaconWitnessLeafCount::ZERO,
            None,
            None,
        )
    }

    fn certifying_qc(terminal: &BlockHeader, wt: u64) -> QuorumCertificate {
        QuorumCertificate::new(
            terminal.hash(),
            terminal.shard_id(),
            terminal.height(),
            terminal.parent_block_hash(),
            Round::new(9),
            SignerBitfield::new(4),
            AggregateSignature::ZERO,
            WeightedTimestamp::from_millis(wt),
        )
    }

    /// The derivation reproduces exactly the genesis the beacon fold
    /// composes: same inputs, same hash, height `max + 1`, clock at the
    /// cut. The served QCs carry higher-round re-certification timestamps
    /// the derivation must ignore.
    #[test]
    fn derivation_reproduces_the_fold_composition() {
        let parent = ShardId::leaf(1, 0);
        let (left, right) = parent.children();
        let left_root = StateRoot::from_raw(Hash::from_bytes(b"left subtree"));
        let right_root = StateRoot::from_raw(Hash::from_bytes(b"right subtree"));
        let composed = SplitChildRoots {
            left: left_root,
            right: right_root,
        }
        .composed_root();

        // Children terminate at heights 8 and 9, at a cut of 2000ms.
        let left_terminal = terminal_header(left, 8, left_root);
        let right_terminal = terminal_header(right, 9, right_root);
        let left_qc = certifying_qc(&left_terminal, 2_400);
        let right_qc = certifying_qc(&right_terminal, 2_600);
        let cut_wt = WeightedTimestamp::from_millis(2_000);

        // The fold's composition over the same inputs.
        let expected = Block::merge_parent_genesis(
            parent,
            composed,
            (left_terminal.hash(), left_terminal.height()),
            (right_terminal.hash(), right_terminal.height()),
            cut_wt,
        );

        let (genesis, origin) = merge_genesis_from_terminals(
            parent,
            (&left_terminal, &left_qc),
            (&right_terminal, &right_qc),
            cut_wt,
        )
        .expect("derives");
        assert_eq!(genesis.hash(), expected.hash());
        assert_eq!(genesis.header().state_root(), composed);
        assert_eq!(origin.genesis_height, BlockHeight::new(10));
        assert_eq!(origin.anchor_wt, cut_wt);
    }

    /// The merged root is composed from the terminals themselves, so a
    /// terminal naming a different subtree yields a different genesis
    /// rather than silently adopting the beacon's.
    #[test]
    fn a_different_terminal_root_changes_the_genesis() {
        let parent = ShardId::leaf(1, 0);
        let (left, right) = parent.children();
        let cut_wt = WeightedTimestamp::from_millis(2_000);
        let right_terminal = terminal_header(right, 9, StateRoot::from_raw(Hash::from_bytes(b"r")));
        let right_qc = certifying_qc(&right_terminal, 2_600);

        let derive = |root: &[u8]| {
            let left_terminal =
                terminal_header(left, 8, StateRoot::from_raw(Hash::from_bytes(root)));
            let left_qc = certifying_qc(&left_terminal, 2_400);
            merge_genesis_from_terminals(
                parent,
                (&left_terminal, &left_qc),
                (&right_terminal, &right_qc),
                cut_wt,
            )
            .expect("derives")
            .0
            .hash()
        };

        assert_ne!(derive(b"honest"), derive(b"forged"));
    }

    /// A quorum certificate that doesn't certify its terminal block is
    /// rejected before any composition.
    #[test]
    fn uncertified_terminal_is_rejected() {
        let parent = ShardId::leaf(1, 0);
        let (left, right) = parent.children();
        let left_terminal = terminal_header(left, 8, StateRoot::ZERO);
        let right_terminal = terminal_header(right, 9, StateRoot::ZERO);
        let good = certifying_qc(&left_terminal, 2_400);
        // A QC certifying the wrong block.
        let bad = certifying_qc(&right_terminal, 2_600);
        assert!(
            merge_genesis_from_terminals(
                parent,
                (&left_terminal, &bad),
                (&right_terminal, &good),
                WeightedTimestamp::ZERO,
            )
            .is_err()
        );
    }
}
