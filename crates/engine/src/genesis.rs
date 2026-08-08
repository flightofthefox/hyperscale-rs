//! Genesis seeding: the stdlib world and the funded-account cells.
//!
//! [`GenesisConfig`] is the canonical input — the accounts a deployment
//! funds and the stake pools its beacon folds facts for — so two nodes
//! with the same config install byte-identical state. The metadata cache
//! holds the stdlib account package, the instance registry binds each
//! funded address to it, and the funded balances land as identity-keyed
//! vault cells in one genesis batch.

use hyperscale_effects_bridge::vm_statics::PackageCache;
use hyperscale_effects_bridge::{PoolRegistry, ProtocolHasher, admit_package, validator_key};
pub use hyperscale_effects_bridge::{XRD, entropy_key, vault_key};
use hyperscale_types::{SettledWrites, StakePoolSeat};
use hyperscale_vm_effects::{
    Address, InstanceMeta, InstanceRegistry, MetadataCache, PackageHash, Value, package_hash,
};
use hyperscale_vm_kernel::encode_amount;
use hyperscale_vm_stdlib::genesis_writes as stdlib_genesis_writes;
pub use hyperscale_vm_stdlib::{GENESIS_PUBLISHER, account_artifact, staking_artifact};

/// Configuration for genesis bootstrapping.
#[derive(Debug, Clone, Default)]
pub struct GenesisConfig {
    /// Funded accounts: owner prefix and initial balance. Seeded as
    /// identity-keyed vault cells and registered as account-package
    /// instances in the process's VM statics.
    pub accounts: Vec<([u8; 16], u128)>,

    /// Stake pools the beacon folds facts for: the pool instance's owner
    /// prefix and the identifier it is folded under. Seated as stake pool
    /// package instances in the process's VM statics, which is what makes
    /// their emitted events beacon facts — running the package never
    /// does, because anyone may run the package.
    pub pools: Vec<StakePoolSeat>,
}

/// The genesis-static world: published stdlib metadata and the funded
/// accounts' instance registrations.
#[derive(Debug, Clone)]
pub struct World {
    /// Published package metadata, growing as blocks commit.
    pub cache: PackageCache,
    /// Instance registrations.
    pub instances: InstanceRegistry,
    /// The stdlib account package's content address.
    pub account_package: PackageHash,
    /// The stdlib stake pool package's content address — the code a
    /// recognised pool must be running for its events to be read as
    /// beacon facts.
    pub staking_package: PackageHash,
    /// The stake pools the beacon folds for. Empty on a network with no
    /// staking surface, which is every network until genesis seats one.
    pub pools: PoolRegistry,
}

/// Build the world for `accounts` (owner prefix, balance): the account
/// package published under its artifact hash, one instance per funded
/// address.
///
/// The published signatures are the ones the artifact declares, admitted
/// through the same check a publish transaction runs — genesis is the
/// cold start of the cache, not a second source of truth for it.
///
/// # Panics
///
/// Panics if the stdlib artifact would not be admissible as a published
/// package — a build defect, not a runtime condition.
#[must_use]
pub fn genesis_world(accounts: &[([u8; 16], u128)]) -> World {
    genesis_world_with_pools(accounts, &[])
}

/// [`genesis_world`] seating `pools` as the stake pools the beacon folds
/// for: `(instance address, the identifier it is folded under)`.
///
/// A pool is an instance of the stdlib stake pool package configured with
/// the resource it stakes and the resource it issues. Seating it here is
/// what makes its events beacon facts — the package alone never does,
/// because anyone may run the package.
///
/// # Panics
///
/// Panics if a stdlib artifact would not be admissible as a published
/// package — a build defect, not a runtime condition.
#[must_use]
pub fn genesis_world_with_pools(accounts: &[([u8; 16], u128)], pools: &[StakePoolSeat]) -> World {
    let artifact = account_artifact();
    let account_package = package_hash(&ProtocolHasher, artifact);
    let metadata =
        admit_package(artifact).expect("the stdlib account artifact publishes as a package");
    let mut seed = MetadataCache::new();
    seed.publish(account_package, metadata);

    let staking_package = package_hash(&ProtocolHasher, staking_artifact());
    seed.publish(
        staking_package,
        admit_package(staking_artifact())
            .expect("the stdlib stake pool artifact publishes as a package"),
    );

    let cache = PackageCache::new(seed);
    let mut instances = InstanceRegistry::new();
    for (address, _) in accounts {
        instances.register(
            Address(*address),
            InstanceMeta {
                package: account_package,
                config: vec![],
            },
        );
    }
    let mut registry = PoolRegistry::new();
    for seat in pools {
        instances.register(
            Address(seat.address),
            InstanceMeta {
                package: staking_package,
                // The resource a delegation is denominated in, the one the
                // pool issues against it, and the principal its operator
                // surface admits. The pool's own identity is its address,
                // so nothing here names it.
                config: vec![
                    Value::Address(XRD),
                    Value::Address(stake_unit(seat.address)),
                    Value::Address(Address(seat.operator)),
                ],
            },
        );
        registry.register(seat.address, seat.id);
    }
    World {
        cache,
        instances,
        account_package,
        staking_package,
        pools: registry,
    }
}

/// The resource a pool at `address` issues against delegations.
///
/// Derived from the pool rather than configured, so two pools can never be
/// seated on one stake-unit resource and a holder's units always name the
/// pool that owes them.
#[must_use]
pub fn stake_unit(pool: [u8; 16]) -> Address {
    Address(vault_key(pool, XRD).local.0)
}

/// The genesis substate writes.
///
/// The protocol's stdlib flash composed with this network's allocations:
/// a seated pool's validator records and one [`XRD`] vault cell per
/// funded account, identity-keyed under the owner's prefix.
#[must_use]
pub fn genesis_writes(accounts: &[([u8; 16], u128)], pools: &[StakePoolSeat]) -> SettledWrites {
    // The stdlib package as a committed cell, under the same content
    // address a publish would place it at. Genesis is then the cache's
    // cold start in the literal sense — the same projection of committed
    // state every later block extends, rather than a second source the
    // cache would have to be told about separately.
    let mut writes = stdlib_genesis_writes(&ProtocolHasher);
    // A seated pool's record of the validators it already operates.
    // Beacon genesis creates those memberships directly in beacon state,
    // so without this the contract would hold no record of validators it
    // demonstrably operates — and its own methods would refuse to speak
    // about them.
    for seat in pools {
        for (validator, pubkey) in &seat.founding {
            writes.cells.insert(
                validator_key(seat.address, validator.inner()),
                Some(pubkey.as_bytes().to_vec()),
            );
        }
    }
    for (address, balance) in accounts {
        writes.cells.insert(
            vault_key(*address, XRD),
            Some(encode_amount(*balance).to_vec()),
        );
    }
    SettledWrites::from_absolutes(writes.cells)
}

#[cfg(test)]
mod tests {
    use hyperscale_vm_stdlib::{ACCOUNT_COMPONENT, account_metadata};

    use super::*;
    use crate::account_address;

    #[test]
    fn genesis_writes_are_identity_keyed_vault_cells() {
        let alice = [0x11u8; 16];
        let bob = [0x22u8; 16];
        let writes = genesis_writes(&[(alice, 500), (bob, 700)], &[]);
        // Two funded accounts' vault cells, plus the stdlib package under
        // the publisher no key derives.
        assert_eq!(writes.cells().len(), 3);
        assert!(
            writes
                .cells()
                .keys()
                .any(|key| key.owner == GENESIS_PUBLISHER)
        );

        for (owner, balance) in [(alice, 500u128), (bob, 700)] {
            let key = vault_key(owner, XRD);
            assert_eq!(key.owner.0, owner);
            assert_eq!(
                writes.cells().get(&key),
                Some(&Some(encode_amount(balance).to_vec()))
            );
        }
    }

    #[test]
    fn the_stdlib_artifact_describes_itself() {
        let artifact = account_artifact();

        // The code is the committed blob and the section is what was
        // added, so the address covers both.
        assert!(artifact.starts_with(ACCOUNT_COMPONENT));
        assert!(artifact.len() > ACCOUNT_COMPONENT.len());
        assert_ne!(
            package_hash(&ProtocolHasher, artifact),
            package_hash(&ProtocolHasher, ACCOUNT_COMPONENT)
        );

        // What genesis publishes is admitted out of the artifact by the
        // publish check, and it is the signature set the stdlib authors:
        // the real guest's exports back every method it declares.
        let declared = admit_package(artifact).expect("publishes as a package");
        assert_eq!(declared, account_metadata());
        let world = genesis_world(&[]);
        assert_eq!(
            world.cache.load().get(world.account_package),
            Some(&declared)
        );
        assert_eq!(
            world.account_package,
            package_hash(&ProtocolHasher, artifact)
        );
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

    #[test]
    fn the_world_binds_every_funded_account_to_the_stdlib_package() {
        let world = genesis_world(&[([0x11; 16], 1), ([0x22; 16], 2)]);
        assert!(world.cache.load().get(world.account_package).is_some());
        for address in [[0x11; 16], [0x22; 16]] {
            assert_eq!(
                world.instances.get(Address(address)).map(|m| m.package),
                Some(world.account_package)
            );
        }
    }

    #[test]
    fn account_addresses_derive_deterministically_from_keys() {
        let a = account_address(&[7u8; 32]);
        assert_eq!(a, account_address(&[7u8; 32]));
        assert_ne!(a, account_address(&[8u8; 32]));
    }
}
