//! Genesis seeding: this network's allocations, as substate writes.
//!
//! [`GenesisConfig`] is the canonical input — the accounts a deployment
//! funds and the stake pools its beacon folds facts for — so two nodes
//! with the same config install byte-identical state. The world those
//! pools are seated in is [`hyperscale_effects_bridge::genesis`]'s; what
//! is here is the state genesis writes, which is the half that needs the
//! kernel's own amount encoding.

pub use hyperscale_effects_bridge::genesis::{
    OWNER_BADGE_RECORD, World, account_artifact, genesis_publisher, genesis_world,
    genesis_world_with_pools, owner_badge_id, pool_address, pool_meta, pool_owner_badge,
    stake_unit, staking_artifact,
};
use hyperscale_effects_bridge::{ProtocolHasher, validator_key};
pub use hyperscale_effects_bridge::{XRD, entropy_key, vault_key};
use hyperscale_types::{EntryKey, Hash, PrincipalAddr, SettledWrites, StakePoolSeat};
use hyperscale_vm_effects::{
    Address, holdings_collection, instance_data_key, package_hash, resource_record_key,
};
use hyperscale_vm_kernel::encode_amount;
use hyperscale_vm_stdlib::genesis_writes as stdlib_genesis_writes;

use crate::executor::artifact_package;

/// Configuration for genesis bootstrapping.
#[derive(Debug, Clone, Default)]
pub struct GenesisConfig {
    /// Funded accounts: owner prefix and initial balance. Seeded as
    /// identity-keyed vault cells. Funding is all this does — an account
    /// is callable whether or not it appears here.
    pub accounts: Vec<(PrincipalAddr, u128)>,

    /// Stake pools the beacon folds facts for: the pool instance's owner
    /// prefix and the identifier it is folded under. Seated as stake pool
    /// package instances in the process's VM statics, which is what makes
    /// their emitted events beacon facts — running the package never
    /// does, because anyone may run the package.
    pub pools: Vec<StakePoolSeat>,
}

/// The genesis substate writes.
///
/// The protocol's stdlib flash composed with this network's allocations:
/// a seated pool's validator records, its owner-badge custody, and one
/// [`XRD`] vault cell per funded account, identity-keyed under the
/// owner's prefix.
///
/// # Panics
///
/// Panics if the badge record exceeds its wire depth, which the one
/// fixed record cannot.
#[must_use]
pub fn genesis_writes(
    accounts: &[(PrincipalAddr, u128)],
    pools: &[StakePoolSeat],
) -> SettledWrites {
    // Each stdlib package as a committed cell, under the same content
    // address a publish would place it at. Genesis is then the cache's
    // cold start in the literal sense — the same projection of committed
    // state every later block extends, rather than a second source the
    // cache would have to be told about separately.
    let mut writes = stdlib_genesis_writes(&ProtocolHasher);
    let staking_package = package_hash(&ProtocolHasher, staking_artifact());
    // A seated pool's record of the validators it already operates.
    // Beacon genesis creates those memberships directly in beacon state,
    // so without this the contract would hold no record of validators it
    // demonstrably operates — and its own methods would refuse to speak
    // about them.
    for seat in pools {
        let pool = pool_address(staking_package, seat);
        for (validator, pubkey) in &seat.founding {
            writes.cells.insert(
                validator_key(pool, validator.inner()),
                Some(pubkey.as_bytes().to_vec()),
            );
        }
        // Custody of the pool: the badge's record under the pool, the
        // instance's data cell, and the holdings entry in the seat's
        // operator account. The badge and its id both derive from the
        // pool, so these three writes are the whole seating — nothing
        // stores a mapping, and selling the pool is an ordinary
        // holdings transfer from here on.
        let badge = pool_owner_badge(pool);
        let id = owner_badge_id(pool);
        writes.cells.insert(
            resource_record_key(&ProtocolHasher, pool, badge),
            Some(
                OWNER_BADGE_RECORD
                    .to_cell()
                    .expect("a record encodes within its wire depth"),
            ),
        );
        writes.cells.insert(
            instance_data_key(&ProtocolHasher, pool, badge, id),
            Some(vec![1]),
        );
        writes.entries.insert(
            EntryKey {
                owner: seat.operator.address(),
                collection: holdings_collection(&ProtocolHasher, seat.operator, badge),
                order: u128::from(id),
            },
            Some(vec![1]),
        );
    }
    for (address, balance) in accounts {
        writes.cells.insert(
            vault_key(*address, *XRD),
            Some(encode_amount(*balance).to_vec()),
        );
    }
    SettledWrites::from_parts(writes.cells, writes.entries)
}

/// The packages the chain is born running, as the beacon registry holds
/// them: `(content address, the prefix their bytes sit under)`.
///
/// Genesis seeds these so the block-validity rule can ask one question of
/// every package a transaction names. They are usable from the genesis
/// epoch and no node ever fetches them, because every node compiles them
/// at boot — the publisher is what the registry keys fetching on, and it
/// is never consulted for these.
#[must_use]
pub fn genesis_package_facts() -> Vec<(Hash, Address)> {
    let publisher = Address::from(genesis_publisher(&ProtocolHasher));
    [account_artifact(), staking_artifact()]
        .into_iter()
        .map(|artifact| (artifact_package(artifact), publisher))
        .collect()
}

#[cfg(test)]
mod tests {
    use hyperscale_types::test_utils::test_principal;

    use super::*;

    #[test]
    fn genesis_writes_are_identity_keyed_vault_cells() {
        let alice = test_principal(0x11);
        let bob = test_principal(0x22);
        let writes = genesis_writes(&[(alice, 500), (bob, 700)], &[]);
        // Two funded accounts' vault cells, plus the stdlib packages
        // under the publisher no key derives.
        assert_eq!(writes.cells().len(), 4);
        assert!(
            writes
                .cells()
                .keys()
                .any(|key| key.owner == genesis_publisher(&ProtocolHasher))
        );

        for (owner, balance) in [(alice, 500u128), (bob, 700)] {
            let key = vault_key(owner, *XRD);
            assert_eq!(key.owner, owner);
            assert_eq!(
                writes.cells().get(&key),
                Some(&Some(encode_amount(balance).to_vec()))
            );
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn both_runtimes_accept_the_section_carrying_artifact() {
        // The blessed engine's acceptance is proved by every guest test
        // in this crate, which now runs the artifact. The reference
        // interpreter ships only on wasm32, so its acceptance has no
        // native witness unless one is written.
        use hyperscale_vm_ref::RefComponent;

        RefComponent::decode(account_artifact())
            .expect("the reference interpreter decodes the stdlib artifact");
    }
}
