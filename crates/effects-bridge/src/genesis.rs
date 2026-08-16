//! The genesis-static world: what a cold start already knows.
//!
//! Published stdlib metadata, the blueprint serving every principal, and
//! the pool instances a deployment seats. It is a projection of genesis
//! configuration and immutable artifacts and nothing else, which is why it
//! sits beside the rest of this crate's bindings rather than inside the
//! engine: a load generator, a wallet and a scenario all need to resolve a
//! target, and none of them executes anything.

use hyperscale_types::{ComponentAddr, ResourceAddr, StakePoolSeat};
use hyperscale_vm_effects::stdlib::OWNER_BADGE;
use hyperscale_vm_effects::{
    Address, Fungibility, Hasher, InstanceMeta, InstanceRegistry, MetadataCache, PackageHash,
    ResourceRecord, Value, package_hash, resource_address,
};
pub use hyperscale_vm_stdlib::{account_artifact, genesis_publisher, staking_artifact};

use crate::vm_statics::PackageCache;
use crate::{PoolRegistry, ProtocolHasher, XRD, admit_protocol_package};

/// The genesis-static world: published stdlib metadata, the blueprint
/// serving every principal, and any seated pool instances.
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

/// Build the world: the stdlib packages published under their artifact
/// hashes, and the account package bound as the one serving every
/// principal.
///
/// Funded accounts are not an input. A principal address commits its own
/// auth material, so an account is callable without anything registered
/// for it — genesis seeds balances, not instances.
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
pub fn genesis_world() -> World {
    genesis_world_with_pools(&[])
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
pub fn genesis_world_with_pools(pools: &[StakePoolSeat]) -> World {
    let artifact = account_artifact();
    let account_package = package_hash(&ProtocolHasher, artifact);
    let metadata = admit_protocol_package(artifact)
        .expect("the stdlib account artifact publishes as a package");
    let mut seed = MetadataCache::new();
    seed.publish(account_package, metadata);

    let staking_package = package_hash(&ProtocolHasher, staking_artifact());
    seed.publish(
        staking_package,
        admit_protocol_package(staking_artifact())
            .expect("the stdlib stake pool artifact publishes as a package"),
    );

    let cache = PackageCache::new(seed);
    let mut instances = InstanceRegistry::new();
    // Funded accounts need nothing registered: a principal address
    // commits its own auth material, and the blueprint serving every
    // principal is protocol-defined.
    instances.serve_principals(account_package);
    let mut registry = PoolRegistry::new();
    for seat in pools {
        let address = instances.create(&ProtocolHasher, pool_meta(staking_package, seat));
        registry.register(address, seat.id);
    }
    World {
        cache,
        instances,
        account_package,
        staking_package,
        pools: registry,
    }
}

/// A genesis-seated pool's creation-fixed record.
///
/// Its configuration is the resource a delegation is denominated in and
/// nothing else. The resource the pool *issues* and the owner badge its
/// operator surface admits are both derived from the pool rather than
/// configured, and the pool's own identity is its address — so none of
/// them is named here.
///
/// The salt stands in for a creating transaction's fresh id, which
/// genesis has none of: the pool's own beacon identifier separates two
/// pools that would otherwise be seated identically.
#[must_use]
pub fn pool_meta(staking_package: PackageHash, seat: &StakePoolSeat) -> InstanceMeta {
    InstanceMeta {
        package: staking_package,
        config: vec![Value::Address(XRD.address())],
        salt: ProtocolHasher.hash(DOMAIN_GENESIS_SALT, &[&seat.id.inner().to_le_bytes()]),
    }
}

/// The address genesis seats `seat` at.
#[must_use]
pub fn pool_address(staking_package: PackageHash, seat: &StakePoolSeat) -> ComponentAddr {
    pool_meta(staking_package, seat).address(&ProtocolHasher)
}

/// The domain separating a genesis instance's salt.
///
/// A hashing input, so its bytes are part of every seated pool's address
/// and fixed by that rather than by where the constant is declared.
const DOMAIN_GENESIS_SALT: &[u8] = b"hyperscale/engine/genesis-instance";

/// The resource a pool issues against delegations.
///
/// A resource address under the pool's own provenance, which is what the
/// staking signature's own derivation evaluates to — so two pools can
/// never be seated on one stake-unit resource, a holder's units always
/// name the pool that owes them, and the address says both facts on
/// sight.
#[must_use]
pub fn stake_unit(pool: impl Into<Address>) -> ResourceAddr {
    resource_address(&ProtocolHasher, pool, &[])
}

/// The pool's owner badge: the identity its operator surface admits.
///
/// Derived like the stake unit and separated from it by the stdlib's
/// badge material — the same derivation the operator gate evaluates, so
/// holding this resource is operating the pool and selling the pool is
/// transferring it.
#[must_use]
pub fn pool_owner_badge(pool: impl Into<Address>) -> ResourceAddr {
    resource_address(
        &ProtocolHasher,
        pool,
        &[Value::Bytes(OWNER_BADGE.to_vec()).canonical_bytes()],
    )
}

/// The badge instance genesis seats: derived from the pool like the
/// badge itself, so it is recomputable from the seat and stored nowhere.
#[must_use]
pub fn owner_badge_id(pool: impl Into<Address>) -> u64 {
    let digest = ProtocolHasher.hash(DOMAIN_OWNER_BADGE_ID, &[&pool.into().to_bytes()]);
    let [b0, b1, b2, b3, b4, b5, b6, b7, ..] = digest.0;
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
}

/// The domain separating a badge instance's id from every other
/// derivation of a pool's address.
const DOMAIN_OWNER_BADGE_ID: &[u8] = b"hyperscale/engine/owner-badge-id";

/// The owner badge's resource record: one non-fungible kind.
pub const OWNER_BADGE_RECORD: ResourceRecord = ResourceRecord {
    kind: Fungibility::NonFungible,
};

#[cfg(test)]
mod tests {
    use hyperscale_types::test_utils::test_principal;
    use hyperscale_vm_stdlib::{ACCOUNT_COMPONENT, account_metadata};

    use super::*;
    use crate::account_address;

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
        let declared = admit_protocol_package(artifact).expect("publishes as a package");
        assert_eq!(declared, account_metadata());
        let world = genesis_world();
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
    fn the_world_binds_every_principal_to_the_stdlib_package() {
        // Funding is not what makes an account callable: the world is
        // built without naming any address, and an address nothing has
        // ever funded resolves the same as one genesis seeds.
        let world = genesis_world();
        assert!(world.cache.load().get(world.account_package).is_some());
        for address in [test_principal(0x11), test_principal(0x22)] {
            assert_eq!(
                world.instances.get(address).map(|m| m.package),
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
