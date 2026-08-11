//! Reshape store adoption — the simulation mirror of the `RocksDB`
//! backend's checkpoint hard-link and subtree adoption.
//!
//! [`SimShardStorage::clone_for_split_child`] is the checkpoint: a deep
//! copy of the parent's substate and tree state re-rooted at the child's
//! prefix, with fresh consensus state (the parent's blocks stay with the
//! parent). [`SimShardStorage::adopt_split_child`] then re-points the
//! clone at the parent root's child-side slot;
//! [`SimShardStorage::adopt_followed_child`] re-points an observer's
//! followed store at the child root its own metadata names.
//! [`SimShardStorage::adopt_merge_parent`] is the inverse: a keeper's
//! `parent`-rooted store already holds both children's subtrees and the
//! stitched root, so adoption only records the merged genesis as the
//! committed tip.

use std::sync::{Arc, RwLock};

use hyperscale_jmt::{NibblePath, Node, NodeKey, TreeReader};
use hyperscale_storage::AdoptSource;
use hyperscale_storage::lock_recover::{read_or_recover, write_or_recover};
use hyperscale_storage::tree::Jmt;
use hyperscale_types::{Block, CertifiedBlock, ChainOrigin, Hash, StateRoot, Verified};

use super::core::{SimImportStaging, SimShardStorage};
use super::state::ConsensusState;

impl SimShardStorage {
    /// The simulation's checkpoint hard-link: a deep copy of this
    /// store's substate and tree state, re-rooted at `child_prefix`,
    /// with fresh consensus state. The sibling half rides along as dead
    /// weight outside the child's prefix, exactly as in the hard-linked
    /// production checkpoint.
    #[must_use]
    pub fn clone_for_split_child(&self, child_prefix: NibblePath) -> Self {
        let mut shared = read_or_recover(&self.state).clone();
        shared.tree_store.set_root_path(child_prefix);
        Self {
            state: Arc::new(RwLock::new(shared)),
            consensus: Arc::new(RwLock::new(ConsensusState::new())),
            jmt_history_length: self.jmt_history_length,
            boundary_pins: Arc::new(RwLock::new(std::collections::BTreeSet::new())),
            import_staging: Arc::new(RwLock::new(SimImportStaging::default())),
        }
    }

    /// Install a reshape successor's derived `genesis` as this store's
    /// chain origin and committed tip. The simulation mirror of the
    /// production backend's `adopt_genesis`, sharing its structure so the
    /// two harnesses cannot diverge on what an adoption admits.
    ///
    /// `source` names only how the tree reaches the genesis version; the
    /// adopted root is then checked against the root the `genesis` names.
    ///
    /// # Errors
    ///
    /// Fails when the genesis block does not sit at the origin's height,
    /// when the store's vintage does not match what `source` requires,
    /// when the tree cannot yield the named subtree, or when the adopted
    /// root does not match the genesis's.
    pub fn adopt_genesis(
        &self,
        origin: ChainOrigin,
        genesis: &Block,
        source: AdoptSource,
    ) -> Result<StateRoot, String> {
        if genesis.height() != origin.genesis_height {
            return Err(format!(
                "genesis block at height {} does not sit at the origin's {}",
                genesis.height(),
                origin.genesis_height,
            ));
        }
        let recorded_origin = read_or_recover(&self.consensus).chain_origin;
        let mut shared = write_or_recover(&self.state);
        // A re-run over an already-adopted store returns the recorded
        // adoption: the tip sits at the genesis height under this origin,
        // and the parent slot the first run consumed is gone.
        let at_genesis_version = shared.current_block_height == origin.genesis_height;
        if at_genesis_version && recorded_origin == origin {
            return Ok(shared.current_root_hash);
        }

        let adopted = match source {
            AdoptSource::InPlace => {
                if shared.current_block_height != origin.genesis_height {
                    return Err(format!(
                        "merged store at version {} does not sit at the genesis height {}",
                        shared.current_block_height, origin.genesis_height,
                    ));
                }
                shared.current_root_hash
            }
            AdoptSource::ParentSubtree | AdoptSource::FollowedTip => {
                let child_path = shared.tree_store.root_path();
                if child_path.is_empty() {
                    return Err("split adoption requires a non-root child prefix".to_string());
                }
                let (node, root) = if matches!(source, AdoptSource::ParentSubtree) {
                    // The metadata is the parent's; the child root hangs off
                    // the parent root's child-side slot.
                    let version = shared.current_block_height.inner();
                    let mut parent_path = child_path.clone();
                    parent_path.truncate(child_path.len() - 1);
                    let side = usize::from(child_path.bits_at(child_path.len() - 1, 1));
                    let parent_root = shared
                        .tree_store
                        .get_node(&NodeKey::new(version, parent_path))
                        .ok_or("clone carries no parent root node")?;
                    let Node::Internal(parent_root) = parent_root.as_ref() else {
                        return Err(
                            "parent root collapsed to a leaf; a ≤1-key parent cannot split"
                                .to_string(),
                        );
                    };
                    match &parent_root.children[side] {
                        None => (None, StateRoot::ZERO),
                        Some(slot) => {
                            let node = shared
                                .tree_store
                                .get_node(&NodeKey::new(slot.version, child_path))
                                .ok_or("clone carries no child subtree root node")?;
                            (
                                Some(node),
                                StateRoot::from_raw(Hash::from_hash_bytes(&slot.hash)),
                            )
                        }
                    }
                } else {
                    // The store's own tip is the adopted subtree; an empty
                    // half never advanced it past the zero root.
                    let root = shared.current_root_hash;
                    if root == StateRoot::ZERO {
                        (None, root)
                    } else {
                        let version = shared.current_block_height.inner();
                        (
                            Some(
                                shared
                                    .tree_store
                                    .get_node(&NodeKey::new(version, child_path))
                                    .ok_or(
                                        "followed store holds no root node at its tip version",
                                    )?,
                            ),
                            root,
                        )
                    }
                };
                if root != genesis.header().state_root() {
                    return Err(format!(
                        "adopted root {root:?} does not match the genesis state root {:?}",
                        genesis.header().state_root(),
                    ));
                }
                install_adoption(&mut shared, origin, node, root)?;
                root
            }
        };
        if adopted != genesis.header().state_root() {
            return Err(format!(
                "adopted root {adopted:?} does not match the genesis state root {:?}",
                genesis.header().state_root(),
            ));
        }
        drop(shared);
        self.install_genesis_tip(origin, genesis);
        Ok(adopted)
    }

    /// Record the child's deterministic genesis as the committed tip —
    /// the consensus half of an adoption: the genesis block with its
    /// deterministic certified pairing, the committed height and hash,
    /// no latest QC (the child chain holds none at its genesis), and
    /// the chain origin for recovery.
    fn install_genesis_tip(&self, origin: ChainOrigin, genesis: &Block) {
        let pair = Verified::<CertifiedBlock>::genesis_certified(genesis.clone());
        let mut consensus = write_or_recover(&self.consensus);
        consensus
            .blocks
            .insert(genesis.height(), pair.as_ref().clone());
        consensus.committed_height = genesis.height();
        consensus.committed_hash = Some(genesis.hash());
        consensus.committed_qc = None;
        consensus.chain_origin = origin;
    }
}

/// Shared adoption tail: copy the child root node (when the side is
/// non-empty) to the genesis version, seed the substate byte total, and
/// move the tip to the genesis.
fn install_adoption(
    shared: &mut super::state::SharedState,
    origin: ChainOrigin,
    child_node: Option<Arc<Node>>,
    child_root: StateRoot,
) -> Result<(), String> {
    let genesis_version = origin.genesis_height.inner();
    let genesis_root_key = NodeKey::new(genesis_version, shared.tree_store.root_path());
    let bytes = match child_node {
        None => 0,
        Some(node) => {
            shared.tree_store.insert(genesis_root_key.clone(), node);
            Jmt::sum_subtree_value_lens(&shared.tree_store, &genesis_root_key)
                .map_err(|e| format!("split adoption byte sum: {e:?}"))?
        }
    };
    shared.substate_bytes.insert(genesis_version, bytes);
    shared.current_block_height = origin.genesis_height;
    shared.current_root_hash = child_root;
    Ok(())
}

#[cfg(test)]
mod tests {
    use hyperscale_jmt::{Blake3Hasher, Hasher, KEY_BYTES};
    use hyperscale_storage::test_helpers::import_boundary_state;
    use hyperscale_storage::{AdoptSource, WitnessSeed};
    use hyperscale_types::{
        AddressClass, Block, BlockHash, BlockHeight, ChainOrigin, Hash, ShardId, StateRoot,
        SubstateKey, SubstateLeaf, ValidatorId, WeightedTimestamp,
    };

    use super::*;

    /// An import leaf whose top byte places it under one trie half.
    fn leaf(top: u8) -> SubstateLeaf {
        let mut key = [0u8; KEY_BYTES];
        key[0] = top;
        key[31] = AddressClass::Component.tag();
        SubstateLeaf {
            key: SubstateKey::from_bytes(key).expect("a stored leaf key names an address"),
            value: vec![top],
        }
    }

    /// A merged parent store: one leaf on each half so the root is the
    /// internal node `r_p`, imported at the genesis height the terminals
    /// continue (`max(9, 8) + 1 = 10`).
    fn merged_store() -> (SimShardStorage, StateRoot) {
        let store = SimShardStorage::default();
        let root = import_boundary_state(
            &store,
            BlockHeight::new(10),
            &[leaf(0x00), leaf(0x80)],
            WitnessSeed::default(),
        )
        .unwrap();
        (store, root)
    }

    fn merge_genesis(state_root: StateRoot) -> (Block, ChainOrigin) {
        let cut = WeightedTimestamp::from_millis(10_000);
        let genesis = Block::merge_parent_genesis(
            ShardId::ROOT,
            state_root,
            (
                BlockHash::from_raw(Hash::from_bytes(b"left terminal")),
                BlockHeight::new(9),
            ),
            (
                BlockHash::from_raw(Hash::from_bytes(b"right terminal")),
                BlockHeight::new(8),
            ),
            cut,
        );
        let origin = ChainOrigin {
            genesis_height: genesis.height(),
            anchor_wt: cut,
        };
        (genesis, origin)
    }

    /// Adoption records the merged genesis as the committed tip over the
    /// already-built tree: the recovered state names the genesis, its
    /// root, origin, and the imported substate byte total.
    #[test]
    fn merge_adoption_records_the_merged_genesis_tip() {
        let (store, root) = merged_store();
        assert_ne!(root, StateRoot::ZERO, "two halves compose an internal root");
        let (genesis, origin) = merge_genesis(root);
        assert_eq!(genesis.height(), BlockHeight::new(10));

        let adopted = store
            .adopt_genesis(origin, &genesis, AdoptSource::InPlace)
            .unwrap();
        assert_eq!(adopted, root);

        let recovered = store.load_recovered_state();
        assert_eq!(recovered.committed_height, BlockHeight::new(10));
        assert_eq!(recovered.committed_hash, Some(genesis.hash()));
        assert_eq!(recovered.jmt_root, Some(root));
        assert_eq!(recovered.chain_origin, origin);
        assert_eq!(recovered.substate_bytes, 2);
    }

    /// A genesis claiming a different root than the store holds fails
    /// closed — the keeper's tree and the beacon's composition disagree.
    #[test]
    fn merge_adoption_rejects_a_root_mismatch() {
        let (store, root) = merged_store();
        let (_, origin) = merge_genesis(root);
        let (wrong, _) = merge_genesis(StateRoot::from_raw(Hash::from_bytes(b"forged root")));
        assert!(
            store
                .adopt_genesis(origin, &wrong, AdoptSource::InPlace)
                .is_err()
        );
    }

    /// A parent store at the trie root holding two left-side leaves and
    /// one right-side leaf, committed at height 9, with its terminal root.
    fn split_parent() -> (SimShardStorage, StateRoot) {
        let store = SimShardStorage::default();
        let root = import_boundary_state(
            &store,
            BlockHeight::new(9),
            &[leaf(0x00), leaf(0x01), leaf(0x80)],
            WitnessSeed::default(),
        )
        .unwrap();
        (store, root)
    }

    fn child_path(side: u8) -> NibblePath {
        let mut path = NibblePath::empty();
        path.push_bits(side, 1);
        path
    }

    fn origin_at_10() -> ChainOrigin {
        ChainOrigin {
            genesis_height: BlockHeight::new(10),
            anchor_wt: WeightedTimestamp::from_millis(42_000),
        }
    }

    fn child_of(side: u8) -> ShardId {
        let (left, right) = ShardId::ROOT.children();
        if side == 0 { left } else { right }
    }

    /// A deterministic split-child genesis at height 10 adopting
    /// `state_root`, derived over a synthetic parent terminal at height 9.
    fn split_genesis(child: ShardId, state_root: StateRoot) -> Block {
        let terminal = Block::genesis(
            ShardId::ROOT,
            ValidatorId::new(0),
            StateRoot::ZERO,
            ChainOrigin {
                genesis_height: BlockHeight::new(9),
                anchor_wt: WeightedTimestamp::ZERO,
            },
        );
        Block::split_child_genesis(
            child,
            state_root,
            terminal.header(),
            WeightedTimestamp::from_millis(42_000),
        )
    }

    /// The child-side slot of `parent`'s root node — the subtree a clone of
    /// it adopts, and so the root that clone's genesis must name.
    fn child_subtree_root(parent: &SimShardStorage, side: u8) -> StateRoot {
        let path = child_path(side);
        let mut parent_path = path.clone();
        parent_path.truncate(path.len() - 1);
        let slot_index = usize::from(path.bits_at(path.len() - 1, 1));
        let node = {
            let shared = read_or_recover(&parent.state);
            let version = shared.current_block_height.inner();
            shared
                .tree_store
                .get_node(&NodeKey::new(version, parent_path))
                .expect("the parent holds a root node")
        };
        let Node::Internal(root) = node.as_ref() else {
            panic!("a ≤1-key parent cannot split");
        };
        let slot = root.children[slot_index]
            .as_ref()
            .expect("both halves are populated");
        StateRoot::from_raw(Hash::from_hash_bytes(&slot.hash))
    }

    /// Both halves adopt; their roots compose to the parent's terminal
    /// root and their counts partition the leaves. Adoption is idempotent:
    /// a re-run returns the recorded root rather than failing on the
    /// parent slot the first run consumed.
    #[test]
    fn split_adoption_partitions_and_is_idempotent() {
        let (parent, parent_root) = split_parent();
        let mut roots = Vec::new();
        for side in [0u8, 1u8] {
            let child = parent.clone_for_split_child(child_path(side));
            // The genesis must name the subtree the clone actually holds —
            // that equality is what gates the seat.
            let expected = child_subtree_root(&parent, side);
            let genesis = split_genesis(child_of(side), expected);
            let root = child
                .adopt_genesis(origin_at_10(), &genesis, AdoptSource::ParentSubtree)
                .unwrap();
            assert_eq!(root, expected);
            assert_ne!(root, StateRoot::ZERO);
            roots.push(root);

            let recovered = child.load_recovered_state();
            assert_eq!(recovered.committed_height, BlockHeight::new(10));
            assert_eq!(recovered.committed_hash, Some(genesis.hash()));
            assert_eq!(recovered.chain_origin, origin_at_10());
            assert_eq!(recovered.substate_bytes, if side == 0 { 2 } else { 1 });

            assert_eq!(
                child
                    .adopt_genesis(origin_at_10(), &genesis, AdoptSource::ParentSubtree)
                    .unwrap(),
                root,
                "re-run returns the recorded adoption",
            );
        }

        assert_eq!(
            Blake3Hasher::hash_internal(&[*roots[0].as_bytes(), *roots[1].as_bytes()]),
            *parent_root.as_bytes(),
            "adopted roots compose to the parent's terminal root",
        );
    }
}
