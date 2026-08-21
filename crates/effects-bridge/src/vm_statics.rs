//! The envelope's static derivation: the tree wire codec and the
//! [`VmStatics`] implementation admission verifies through.
//!
//! The envelope tree travels as canonical HBOR of the mirror types
//! below — the vocabulary crate deliberately has no wire encoding, so
//! this module owns it. Derivation is `decode → admit → route` over the
//! process's genesis-static metadata, rooted at the envelope's signing
//! hash, projected into the workspace's admission vocabulary:
//! substate-granular keys for point effects, interval-granular keys for
//! collection effects (an entry is its width-one interval), reads and
//! snapshots in the shared class, every other mode exclusive. Subintent
//! nullifier creation writes ride the routed sets, so admission
//! conflicts on them like any other exclusive key.

use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};

use arc_swap::ArcSwap;
use hyperscale_hbor::{from_slice as hbor_from_slice, to_vec as hbor_to_vec};
use hyperscale_types::{
    DeclaredKey, DeclaredRange, Derived, EnvelopeExt, Hash, MAX_STATE_ENTRIES_PER_TX, Routing,
    TransactionEnvelope, VmStatics, VmStaticsError, declared_work,
};
use hyperscale_vm_effects::vocabulary::{AUTH, CONFIG, VAULT};
use hyperscale_vm_effects::{
    AuthCell, EnvelopeTree, InstanceRegistry, ManifestHash, MetadataCache, PRIMARY, PackageHash,
    PackageMetadata, PrefixShardResolver, Presented, Routing as RoutedTransaction, Value,
    admit_tree, child_key, footprint, package_hash, package_key as canonical_package_key,
    principal_address, route_tree, xrd,
};
use hyperscale_vm_fixtures::lottery;
use hyperscale_vm_stdlib::staking;
use hyperscale_vm_types::{
    Address, EffectSet, EffectTarget, Mode, PrincipalAddr, ResourceAddr, SchemeId, SubstateKey,
};

use crate::ProtocolHasher;
use crate::artifact::admit_package;

/// The protocol fee and transfer resource: the genesis publisher's
/// primary issue.
///
/// A resource like any other, minted by an address no signer reaches —
/// so supply moves only where the protocol writes state directly, and
/// the address sits where a hash puts it, on no shard by preference.
pub static XRD: LazyLock<ResourceAddr> = LazyLock::new(|| xrd(&ProtocolHasher));

/// The vault cell for `resource` under `owner` — the same child key the
/// stdlib account metadata's effect clauses compute.
#[must_use]
pub fn vault_key(owner: impl Into<Address>, resource: impl Into<Address>) -> SubstateKey {
    child_key(
        &ProtocolHasher,
        owner,
        VAULT,
        &[Value::Address(resource.into()).canonical_bytes()],
    )
}

/// The stored-authority cell under `owner` — what `securify` writes,
/// `authorize` reads, and the payer shard's binding verdict consults.
#[must_use]
pub fn auth_key(owner: impl Into<Address>) -> SubstateKey {
    child_key(&ProtocolHasher, owner, AUTH, &[])
}

/// A stake pool's record of one validator it operates: the cell the
/// pool's operator methods declare, keyed by the validator.
///
/// Non-empty means the pool took this validator on and holds the key it
/// registered. The methods that speak about an existing validator read
/// it, so genesis has to write it for members the beacon created before
/// the contract existed.
#[must_use]
pub fn validator_key(pool: impl Into<Address>, validator: u64) -> SubstateKey {
    child_key(
        &ProtocolHasher,
        pool,
        staking::VALIDATORS,
        &[Value::U64(validator).canonical_bytes()],
    )
}

/// A lottery's settled-round cell: where its `draw` records the
/// transaction's randomness beside the entrant it selected. Mirrors the
/// effect signature the method declares.
#[must_use]
pub fn draw_key(lottery: impl Into<Address>) -> SubstateKey {
    child_key(&ProtocolHasher, lottery, lottery::OUTCOME, &[])
}

/// An instance's configuration leaf: the seal its instantiation writes.
///
/// Holds the whole creation-fixed record, and its presence is what makes
/// the component actual — every method the package declares reads it
/// under a presence condition, and the one write the slot admits is
/// refused where the leaf is already there.
#[must_use]
pub fn config_key(owner: impl Into<Address>) -> SubstateKey {
    child_key(&ProtocolHasher, owner, CONFIG, &[])
}

/// Where `publisher`'s copy of the package addressed by `package` lives:
/// the vocabulary's own derivation, bound to the protocol hasher.
#[must_use]
pub fn package_key(publisher: impl Into<Address>, package: PackageHash) -> SubstateKey {
    canonical_package_key(&ProtocolHasher, publisher, package)
}

/// The principal address `public_key` opens under `scheme`, or `None` if
/// the scheme registers nothing or gives its keys another width.
///
/// The address commits to the key and the scheme, so genesis funding,
/// transaction builders and admission all derive the same address from
/// the same key — and admission verifies a signer against its target by
/// recomputing this, with nothing to look up. Material no registered
/// scheme claims opens no account at all, so a key that arrives under the
/// wrong tag derives nothing rather than deriving somewhere unreachable.
#[must_use]
pub fn principal_for(scheme: SchemeId, public_key: &[u8]) -> Option<PrincipalAddr> {
    scheme
        .spec()
        .filter(|spec| spec.admits_key(public_key))
        .map(|_| principal_address(&ProtocolHasher, scheme, public_key))
}

/// The principal address an ed25519 public key opens — the ed25519 case
/// of [`principal_for`], for the callers that hold a typed key.
#[must_use]
pub fn account_address(public_key: &[u8; 32]) -> PrincipalAddr {
    principal_address(&ProtocolHasher, SchemeId::ED25519, public_key)
}

/// Encode an envelope tree to its canonical bytes.
///
/// The vocabulary owns its codec; this is the seam's name for it.
///
/// # Panics
///
/// On a tree past the vocabulary's own caps — one no admission path can
/// have accepted.
#[must_use]
pub fn encode_tree(tree: &EnvelopeTree) -> Vec<u8> {
    hbor_to_vec(tree).expect("a tree within its caps encodes")
}

/// Decode wire bytes into an envelope tree.
///
/// # Errors
///
/// [`VmStaticsError`] on malformed or non-canonical bytes.
pub fn decode_tree(bytes: &[u8]) -> Result<EnvelopeTree, VmStaticsError> {
    hbor_from_slice(bytes).map_err(|error| VmStaticsError(format!("tree decode: {error}")))
}

/// How a routed declaration lands in the workspace's admission
/// vocabulary: the three key classes, and the mode behind each key.
struct DeclaredAccess {
    read_keys: BTreeSet<DeclaredKey>,
    write_keys: BTreeSet<DeclaredKey>,
    provision_keys: BTreeSet<DeclaredKey>,
    declared_modes: Vec<(DeclaredKey, Mode)>,
}

/// Sort a routed transaction's effects into the classes admission,
/// provisioning, and scheduling each read.
///
/// Fresh reads share, mutations exclude, and a locked read takes no
/// admission key and makes no participant — its target cannot change, so
/// nothing can contend on it. The provision set is what a counterpart
/// shard cannot execute without: fresh reads and read-modify-write
/// priors, never a delta or a reservation, neither of which depends on
/// the value it changes.
///
/// The modes ride alongside rather than being recoverable from the sets,
/// because the sets have collapsed delta, reserve and write into one
/// exclusive class by the time they are built — and which of the three a
/// key holds is exactly what decides whether two transactions may be in
/// flight on it together.
fn classify_declared_access(routing: &RoutedTransaction) -> DeclaredAccess {
    let mut access = DeclaredAccess {
        read_keys: BTreeSet::new(),
        write_keys: BTreeSet::new(),
        provision_keys: BTreeSet::new(),
        declared_modes: Vec::new(),
    };
    for effect in routing.per_shard.values().flat_map(EffectSet::iter) {
        let key = admission_key(&effect.target);
        match effect.mode {
            Mode::Read => {
                access.read_keys.insert(key);
                access.provision_keys.insert(key);
            }
            Mode::Write => {
                access.write_keys.insert(key);
                access.provision_keys.insert(key);
            }
            Mode::Delta | Mode::Reserve { .. } => {
                access.write_keys.insert(key);
            }
            Mode::Locked => continue,
        }
        access.declared_modes.push((key, effect.mode));
    }
    access.declared_modes.sort_unstable();
    access
}

/// Refuse a declaration whose provisions could outgrow one bundle.
///
/// The wire codec refuses a provision bundle past
/// [`MAX_STATE_ENTRIES_PER_TX`], so a declaration that could ask for
/// more is refused here — at admission, before an honest server does
/// enumeration work a bundle it cannot encode would throw away. A cell
/// serves one entry; a range serves at most its declared cap.
fn check_provision_weight(provision_keys: &BTreeSet<DeclaredKey>) -> Result<(), VmStaticsError> {
    let weight: usize = provision_keys
        .iter()
        .map(|key| match key {
            DeclaredKey::Cell(_) => 1,
            DeclaredKey::Range(range) => usize::try_from(range.cap).unwrap_or(usize::MAX),
        })
        .fold(0, usize::saturating_add);
    if weight > MAX_STATE_ENTRIES_PER_TX {
        return Err(VmStaticsError(format!(
            "declared provisions could serve {weight} entries, past the \
             {MAX_STATE_ENTRIES_PER_TX} one bundle may carry"
        )));
    }
    Ok(())
}

/// The admission key for one effect target: substate-granular for
/// points, interval-granular for collection targets — an entry is its
/// width-one interval.
const fn admission_key(target: &EffectTarget) -> DeclaredKey {
    match target {
        EffectTarget::Point(key) => DeclaredKey::Cell(*key),
        EffectTarget::Entry {
            owner,
            collection,
            order,
        } => DeclaredKey::Range(DeclaredRange {
            owner: *owner,
            collection: *collection,
            lo: *order,
            hi: *order,
            cap: 1,
        }),
        EffectTarget::Range {
            owner,
            collection,
            lo,
            hi,
            cap,
        } => DeclaredKey::Range(DeclaredRange {
            owner: *owner,
            collection: *collection,
            lo: *lo,
            hi: *hi,
            cap: *cap,
        }),
    }
}

/// The envelope's identity: its signing hash through the workspace's
/// protocol hash, as the vocabulary's hash type.
#[must_use]
pub fn envelope_identity(vm: &TransactionEnvelope) -> ManifestHash {
    ManifestHash(vm.signing_hash().as_hash32())
}

/// A consumer of committed package artifacts — the engine's compile
/// pipeline registers one so a package's code is being compiled from the
/// moment its cell commits, not from its first call.
pub type ArtifactSink = Arc<dyn Fn(&[u8]) + Send + Sync>;

/// The bridge's [`VmStatics`]: `decode → admit → route` over the
/// process's genesis-static metadata.
pub struct BridgeStatics {
    /// Published package metadata, growing as blocks commit.
    pub cache: PackageCache,
    /// Instance registrations, genesis-static.
    pub instances: InstanceRegistry,
    /// Where a committed package's artifact bytes are handed on, beside
    /// the metadata absorption.
    pub artifact_sink: Option<ArtifactSink>,
}

/// The package a committed cell publishes, or `None` for every other
/// cell.
///
/// A package cell is self-identifying: its key is the content address of
/// the very bytes it stores, so recomputing the address from the value
/// and rebuilding the key answers the question without any side channel,
/// any tag, and any trust in what wrote it. A cell of any other kind
/// cannot match except by finding a hash collision.
///
/// The preamble is tested first because this runs over every cell of
/// every commit, and hashing a whole value to learn that it was never
/// code is the one cost that scales with what the chain writes rather
/// than with what it publishes. Admission demands the deterministic wasm
/// profile of anything it lets through, so no artifact that could
/// publish is turned away here.
#[must_use]
pub fn committed_package(owner: Address, local: [u8; 16], value: &[u8]) -> Option<PackageHash> {
    if !value.starts_with(WASM_PREAMBLE) {
        return None;
    }
    let package = package_hash(&ProtocolHasher, value);
    (package_key(owner, package).local.0 == local).then_some(package)
}

/// The four bytes every wasm artifact opens with.
const WASM_PREAMBLE: &[u8] = b"\0asm";

/// The published-package cache: content-addressed, shared process-wide,
/// and grown from committed state.
///
/// One cache per process rather than one per shard, because a package is
/// immutable and named by the hash of its own bytes — two shards holding
/// it hold the same thing, and a node running several vnodes has no
/// reason to hold it twice.
///
/// Reads are lock-free because every admission derivation takes one:
/// swapping a whole new map in on the rare publish costs a clone that
/// nothing waits on, where a lock would put every derivation behind the
/// commit path.
#[derive(Clone, Debug)]
pub struct PackageCache(Arc<ArcSwap<MetadataCache>>);

impl PackageCache {
    /// A cache seeded with the packages a cold start already knows.
    #[must_use]
    pub fn new(seed: MetadataCache) -> Self {
        Self(Arc::new(ArcSwap::from_pointee(seed)))
    }

    /// The current published set.
    #[must_use]
    pub fn load(&self) -> Arc<MetadataCache> {
        self.0.load_full()
    }

    /// Publish `metadata` under `package` unless it is already there.
    ///
    /// First-write-wins by content address, which is what makes
    /// republishing idempotent: equal hash means equal artifact, so the
    /// entry can never need replacing.
    ///
    /// # Panics
    ///
    /// Panics if the metadata fails the cache's publish check. Every
    /// caller feeds this from a committed artifact that already cleared
    /// admission, so a refusal here is a node defect, never an input.
    pub fn publish(&self, package: PackageHash, metadata: PackageMetadata) {
        if self.load().get(package).is_some() {
            return;
        }
        let mut next = (*self.load()).clone();
        next.publish(package, metadata)
            .expect("everything published here cleared the artifact gate");
        self.0.store(Arc::new(next));
    }

    /// Publish the package a committed cell holds, if it holds one, and
    /// say whether it did.
    ///
    /// A package cell is self-identifying — see [`committed_package`] —
    /// and its bytes still have to clear admission before anything is
    /// published under them. The two together are the whole of what
    /// makes a cell a package, which is why the answer is returned
    /// rather than inferred again by the caller: the code that runs an
    /// artifact and the metadata that routes calls into it must never
    /// disagree about whether it was admitted.
    pub fn absorb_cell(&self, owner: impl Into<Address>, local: [u8; 16], value: &[u8]) -> bool {
        let Some(package) = committed_package(owner.into(), local, value) else {
            return false;
        };
        let Ok(metadata) = admit_package(value) else {
            return false;
        };
        self.publish(package, metadata);
        true
    }
}

impl BridgeStatics {
    /// A publish's routing: an exclusive write on the package cell, and
    /// one on the publisher's fee vault.
    ///
    /// The vault is declared even though no signature asks for it. A
    /// completed transaction burns its fee there, and declaring it is
    /// what makes two publishes by one payer conflict — without it they
    /// share a block and settle two burns against one cell, which is the
    /// exposure a call transaction avoids only because its own withdraw
    /// happens to name the same vault.
    fn derive_publish(
        vm: &TransactionEnvelope,
        signer: PrincipalAddr,
        artifact: &[u8],
    ) -> Result<Derived, VmStaticsError> {
        if !vm.subintent_sigs.is_empty() {
            return Err(VmStaticsError("a publish carries no subintents".into()));
        }
        // The artifact has to describe itself before it is addressed:
        // what the address covers is code and signatures together, so an
        // artifact that declares nothing is not a package.
        admit_package(artifact)?;

        let publisher = vm.fee_payer;
        let package = package_hash(&ProtocolHasher, artifact);
        let cell = package_key(publisher, package);
        let vault = vault_key(publisher, *XRD);
        let mut write_keys = vec![DeclaredKey::Cell(cell), DeclaredKey::Cell(vault)];
        write_keys.sort_unstable();
        write_keys.dedup();

        // A publish never reaches the kernel, so it declares no effects
        // for `footprint` to price. Its footprint stands in as the two
        // exclusive cells it claims plus the artifact it writes whole
        // into state — the largest transaction the protocol admits, and
        // one the declared side would otherwise price as the smallest.
        let footprint = (write_keys.len() as u64).saturating_add(artifact.len() as u64);
        let work = declared_work(footprint, vm.gas_limit, vm.signature_work());

        Ok(Derived {
            work,
            signer,
            routing: Routing {
                read_prefixes: Vec::new(),
                write_prefixes: vec![publisher.address()],
                provision_prefixes: Vec::new(),
                read_keys: Vec::new(),
                declared_modes: write_keys.iter().map(|key| (*key, Mode::Write)).collect(),
                write_keys,
                provision_keys: Vec::new(),
            },
            subintent_hashes: Vec::new(),
            fee_vault_local: vault.local.0,
            auth_cell_local: auth_key(publisher).local.0,
            // A publish runs no package: it writes one and calls nothing.
            packages: Vec::new(),
        })
    }
}

impl VmStatics for BridgeStatics {
    fn absorb_committed_cell(&self, owner: [u8; 32], local: [u8; 16], value: &[u8]) {
        let Ok(owner) = Address::from_bytes(owner) else {
            return;
        };
        if self.cache.absorb_cell(owner, local, value)
            && let Some(sink) = &self.artifact_sink
        {
            sink(value);
        }
    }

    fn package_cell(&self, owner: [u8; 32], local: [u8; 16], value: &[u8]) -> Option<Hash> {
        let owner = Address::from_bytes(owner).ok()?;
        committed_package(owner, local, value).map(|package| Hash::from(package.0))
    }

    fn rule_admits(
        &self,
        auth_cell: Option<&[u8]>,
        payer: PrincipalAddr,
        signer: PrincipalAddr,
        clock_ms: u64,
    ) -> bool {
        // The verdict is the kernel gate's own, judged over the envelope
        // signer alone and the primary — paying is governed by whatever
        // governs `authorize`.
        AuthCell::admits(
            auth_cell.unwrap_or_default(),
            payer.address(),
            PRIMARY,
            &[Presented::Identity(signer.into())],
            clock_ms,
        )
    }

    fn derive(&self, vm: &TransactionEnvelope) -> Result<Derived, VmStaticsError> {
        // The identity the envelope's own signature opens: what the root
        // intent presents as evidence, and the identity the payer's rule
        // must admit. Whether it does is the payer shard's verdict —
        // taken where the payer's state is, as a condition of the fee
        // reservation engaging — so derivation records the identity and
        // never compares it against the payer field.
        let Some(signer) = principal_for(vm.signer_scheme, &vm.signer) else {
            return Err(VmStaticsError(
                "the envelope's signer key derives no principal".into(),
            ));
        };
        if let Some(artifact) = vm.artifact() {
            return Self::derive_publish(vm, signer, artifact);
        }
        let tree = decode_tree(vm.call_tree().unwrap_or_default())?;
        if vm.subintent_sigs.len() != tree.subintents.len() {
            return Err(VmStaticsError(format!(
                "envelope binds {} subintents but carries {} signatures",
                tree.subintents.len(),
                vm.subintent_sigs.len()
            )));
        }
        // Bind every declared signer address to its public key; the
        // signatures themselves verify at the transaction gate, over the
        // declaration hashes returned here.
        for (index, (sig, subintent)) in vm.subintent_sigs.iter().zip(&tree.subintents).enumerate()
        {
            if principal_for(sig.scheme, &sig.public_key) != Some(subintent.signer) {
                return Err(VmStaticsError(format!(
                    "subintent {index} signer address does not match its public key"
                )));
            }
        }
        let packages = self.cache.load();
        // The per-envelope registry: what genesis serves plus the tree's
        // own presented instance records, each registered at exactly the
        // address it derives — the whole of instantiation, composed
        // identically on every node from the signed tree alone.
        let instances = self
            .instances
            .with_instances(&tree.instances, &ProtocolHasher);
        let admitted = admit_tree(
            &tree,
            signer,
            envelope_identity(vm),
            &packages,
            &instances,
            &ProtocolHasher,
        )
        .map_err(|error| VmStaticsError(format!("admission: {error}")))?;
        let routing = route_tree(&admitted, &PrefixShardResolver { bits: 0 });

        let DeclaredAccess {
            read_keys,
            write_keys,
            provision_keys,
            declared_modes,
        } = classify_declared_access(&routing);
        check_provision_weight(&provision_keys)?;
        let prefixes = |keys: &BTreeSet<DeclaredKey>| -> Vec<Address> {
            keys.iter()
                .map(DeclaredKey::owner)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        };
        // Every package the lowered calls run, deduplicated — what the
        // execution gate holds the transaction to on each shard.
        let packages: Vec<Hash> = routing
            .calls
            .iter()
            .map(|call| Hash::from(call.package.0))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        // What this transaction costs a block, on the engine's own
        // schedule: the fixed charge for carrying it, what it declared it
        // would touch, and the ceiling it signed for its own execution.
        // The declaration spans every shard it routes to, because the
        // reservation is taken once against the whole of it.
        let work = declared_work(
            routing
                .per_shard
                .values()
                .fold(0u64, |total, set| total.saturating_add(footprint(set))),
            vm.gas_limit,
            vm.signature_work(),
        );
        Ok(Derived {
            work,
            signer,
            routing: Routing {
                read_prefixes: prefixes(&read_keys),
                write_prefixes: prefixes(&write_keys),
                provision_prefixes: prefixes(&provision_keys),
                read_keys: read_keys.into_iter().collect(),
                write_keys: write_keys.into_iter().collect(),
                provision_keys: provision_keys.into_iter().collect(),
                declared_modes,
            },
            subintent_hashes: admitted
                .subintents
                .iter()
                .map(|record| record.subintent.0.0)
                .collect(),
            fee_vault_local: vault_key(vm.fee_payer, *XRD).local.0,
            auth_cell_local: auth_key(vm.fee_payer).local.0,
            packages,
        })
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::{
        CallTarget, Ed25519PrivateKey, NetworkId, Secp256k1PrivateKey, TX_UNITS, TransactionBody,
    };
    use hyperscale_vm_effects::vocabulary::VAULT;
    use hyperscale_vm_effects::{
        AuthBase, Constraint, EdgeRef, EvidenceRef, GraphArg, GraphNode, Hasher, IntentDecl,
        ManifestGraph, PackageHash, Presented, Proposal, RoleTable, StoredRule, Subintent,
        SubintentHash, YieldBinding, YieldParam, child_key, nullifier_key,
    };
    use hyperscale_vm_manifest_builder::signing::sign_subintent;
    use hyperscale_vm_stdlib::account;
    use hyperscale_vm_types::{AddressClass, CollectionId, ResourceAddr};

    use super::*;

    const RES_X: ResourceAddr = ResourceAddr::new([0xE1; 31]);
    const RES_Y: ResourceAddr = ResourceAddr::new([0xE2; 31]);

    fn key(seed: u8) -> Ed25519PrivateKey {
        Ed25519PrivateKey::from_bytes(&[seed; 32]).unwrap()
    }

    fn composer_addr() -> PrincipalAddr {
        account_address(&key(7).public_key().0)
    }

    fn bob_addr() -> PrincipalAddr {
        account_address(&key(9).public_key().0)
    }

    fn statics() -> BridgeStatics {
        let package = PackageHash(ProtocolHasher.hash(b"package", &[b"account"]));
        let mut cache = MetadataCache::new();
        cache
            .publish(package, account::metadata())
            .expect("the account package publishes");
        let mut instances = InstanceRegistry::new();
        // The composer and its counterparty are principals: their
        // addresses derive from their keys, so nothing is registered for
        // them.
        instances.serve_principals(package);
        BridgeStatics {
            cache: PackageCache::new(cache),
            instances,
            artifact_sink: None,
        }
    }

    /// The sign-in every fixture graph leads with: node 0 of its intent,
    /// which is what the withdraw fixtures point their proofs at.
    fn sign_in(target: impl Into<CallTarget>) -> GraphNode {
        GraphNode {
            target: target.into(),
            method: "authorize".into(),
            args: vec![],
            evidence: [EvidenceRef::IntentSignature].into(),
        }
    }

    fn withdraw(target: impl Into<CallTarget>, resource: ResourceAddr, amount: u128) -> GraphNode {
        GraphNode {
            target: target.into(),
            method: "withdraw".into(),
            args: vec![
                GraphArg::Literal(Value::Address(resource.address())),
                GraphArg::Literal(Value::U128(amount)),
            ],
            evidence: [EvidenceRef::Node(0)].into(),
        }
    }

    fn deposit_edge(
        target: impl Into<CallTarget>,
        producer: u32,
        resource: ResourceAddr,
    ) -> GraphNode {
        GraphNode {
            target: target.into(),
            method: "deposit".into(),
            args: vec![GraphArg::Edge {
                edge: EdgeRef {
                    producer,
                    output: 0,
                },
                constraints: vec![Constraint::ResourceIs(resource)],
            }],
            evidence: BTreeSet::new(),
        }
    }

    fn deposit_param(target: impl Into<CallTarget>, param: u32) -> GraphNode {
        GraphNode {
            target: target.into(),
            method: "deposit".into(),
            args: vec![GraphArg::Param(param)],
            evidence: BTreeSet::new(),
        }
    }

    fn single_intent_tree(nodes: Vec<GraphNode>) -> EnvelopeTree {
        EnvelopeTree {
            root: IntentDecl {
                graph: ManifestGraph { nodes },
                params: Vec::new(),
            },
            root_bindings: Vec::new(),
            subintents: Vec::new(),
            instances: Vec::new(),
            resources: Vec::new(),
        }
    }

    /// The two-signer composition: the composer pays X for the
    /// subintent's Y.
    fn composed_tree() -> EnvelopeTree {
        EnvelopeTree {
            root: IntentDecl {
                graph: ManifestGraph {
                    nodes: vec![
                        sign_in(composer_addr()),
                        withdraw(composer_addr(), RES_X, 100),
                        deposit_param(composer_addr(), 0),
                    ],
                },
                params: vec![YieldParam {
                    resource: RES_Y,
                    constraints: vec![Constraint::MinAmount(10)],
                }],
            },
            root_bindings: vec![YieldBinding {
                intent: 1,
                edge: EdgeRef {
                    producer: 1,
                    output: 0,
                },
            }],
            subintents: vec![Subintent {
                decl: IntentDecl {
                    graph: ManifestGraph {
                        nodes: vec![
                            sign_in(bob_addr()),
                            withdraw(bob_addr(), RES_Y, 10),
                            deposit_param(bob_addr(), 0),
                        ],
                    },
                    params: vec![YieldParam {
                        resource: RES_X,
                        constraints: vec![Constraint::MinAmount(100)],
                    }],
                },
                signer: bob_addr(),
                bindings: vec![YieldBinding {
                    intent: 0,
                    edge: EdgeRef {
                        producer: 1,
                        output: 0,
                    },
                }],
            }],
            instances: Vec::new(),
            resources: Vec::new(),
        }
    }

    fn envelope(tree: &EnvelopeTree, subintent_keys: &[&Ed25519PrivateKey]) -> TransactionEnvelope {
        let subintent_sigs = tree
            .subintents
            .iter()
            .zip(subintent_keys)
            .map(|(subintent, signer)| {
                let hash = subintent.decl.hash(&ProtocolHasher);
                sign_subintent(*signer, &hash.0.0)
            })
            .collect();
        TransactionEnvelope {
            body: TransactionBody::Call(encode_tree(tree)),
            subintent_sigs,
            fee_payer: composer_addr(),
            max_fee: 1_000,
            gas_limit: 1_000_000,
            validity_start_ms: 0,
            validity_end_ms: 1_000_000,
            message: Vec::new(),
            network: NetworkId(242),
            signer_scheme: SchemeId::NONE,
            signer: Vec::new(),
            signature: Vec::new(),
        }
        .sign(&key(7))
    }

    /// Work is what a block pays to carry a transaction: a fixed charge
    /// nobody escapes, plus what it declared, plus what it signed for.
    ///
    /// The fixed term is the part that matters. Without it a minimal
    /// zero-gas transaction would price at almost nothing, and a budget
    /// over work would bound weight while the transaction count — which
    /// is what tick entries, tick-chain entries and receipts scale with
    /// — ran free.
    #[test]
    fn work_prices_the_fixed_cost_of_carrying_a_transaction() {
        let tree = single_intent_tree(vec![
            sign_in(composer_addr()),
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 1, RES_X),
        ]);
        let derived = statics().derive(&envelope(&tree, &[])).expect("derives");

        // The envelope's own ceiling is in there, and so is a charge no
        // declaration can shrink.
        assert!(
            derived.work > TX_UNITS + 1_000_000,
            "work must carry the fixed charge and the signed limit: {}",
            derived.work
        );

        // A second recipient declares more, so it costs more — nothing
        // else about the two envelopes differs.
        let wider = single_intent_tree(vec![
            sign_in(composer_addr()),
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 1, RES_X),
            withdraw(composer_addr(), RES_Y, 10),
            deposit_edge(bob_addr(), 3, RES_Y),
        ]);
        let wider = statics().derive(&envelope(&wider, &[])).expect("derives");
        assert!(
            wider.work > derived.work,
            "a wider declaration must not be cheaper: {} vs {}",
            wider.work,
            derived.work
        );
    }

    /// What a transaction's signatures cost to check is priced with the
    /// rest of what it declares, so a wider scheme is a fee fact rather
    /// than free verification.
    #[test]
    fn work_prices_the_signatures_the_envelope_carries() {
        let tree = single_intent_tree(vec![
            sign_in(composer_addr()),
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 1, RES_X),
        ]);
        let ed = statics().derive(&envelope(&tree, &[])).expect("derives");

        let secp = Secp256k1PrivateKey::from_bytes(&[7u8; 32]).expect("a scalar in range");
        let payer = principal_for(SchemeId::SECP256K1, &secp.public_key().0)
            .expect("a registered scheme opens an account");
        let secp_tree = single_intent_tree(vec![
            sign_in(payer),
            withdraw(payer, RES_X, 100),
            deposit_edge(bob_addr(), 1, RES_X),
        ]);
        let mut wider = envelope(&secp_tree, &[]);
        wider.fee_payer = payer;
        let wider = statics().derive(&wider.sign(&secp)).expect("derives");

        assert!(
            wider.work > ed.work,
            "a wider signature scheme must not verify for free: {} vs {}",
            wider.work,
            ed.work
        );
    }

    #[test]
    fn the_tree_codec_round_trips() {
        let tree = composed_tree();
        let decoded = decode_tree(&encode_tree(&tree)).unwrap();
        assert_eq!(decoded, tree);
        assert!(decode_tree(&[0xFF, 0x00]).is_err());
    }

    #[test]
    fn a_transfer_derives_substate_keys_and_owner_prefixes() {
        let tree = single_intent_tree(vec![
            sign_in(composer_addr()),
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 1, RES_X),
        ]);
        let derived = statics().derive(&envelope(&tree, &[])).expect("derives");

        // Reserve at the sender's vault and deltas at the recipient's:
        // all exclusive-class, substate-granular, under the two owners.
        // The sign-in reads the sender's rule cell, which is the one
        // shared key and the one provision — its absence is what the
        // read carries to every participant.
        let sender_vault = child_key(
            &ProtocolHasher,
            composer_addr(),
            VAULT,
            &[Value::Address(RES_X.address()).canonical_bytes()],
        );
        assert!(derived.routing.write_keys.contains(&DeclaredKey::substate(
            composer_addr().address(),
            sender_vault.local.0
        )));
        let rule_cell =
            DeclaredKey::substate(composer_addr().address(), auth_key(composer_addr()).local.0);
        assert_eq!(derived.routing.read_keys, vec![rule_cell]);
        assert_eq!(derived.routing.provision_keys, vec![rule_cell]);
        assert_eq!(
            derived.routing.provision_prefixes,
            vec![composer_addr().address()]
        );
        assert!(derived.subintent_hashes.is_empty());
        let mut owners = vec![composer_addr(), bob_addr()];
        owners.sort_unstable();
        assert_eq!(derived.routing.write_prefixes, owners);
    }

    /// The payer shard's binding verdict across the securify boundary:
    /// absent means the virtual rule, stored bytes mean the governing
    /// primary and nothing else, and bytes that decode as no cell admit
    /// nobody.
    #[test]
    fn the_stored_rule_governs_the_payer_binding() {
        let statics = statics();

        // Virtual: the payer's own identity and no other, whatever the
        // clock says.
        assert!(statics.rule_admits(None, composer_addr(), composer_addr(), 0));
        assert!(statics.rule_admits(Some(&[]), composer_addr(), composer_addr(), u64::MAX));
        assert!(!statics.rule_admits(None, composer_addr(), bob_addr(), 0));

        // Securified to Bob: the old identity is dead, the rule's lives.
        let bob_rules = || {
            RoleTable::uniform(&StoredRule::Require(Presented::Identity(bob_addr().into())))
                .expect("a rule within the caps")
        };
        let cell = AuthCell::new(AuthBase::new(1_000, bob_rules()))
            .to_bytes()
            .unwrap();
        assert!(statics.rule_admits(Some(&cell), composer_addr(), bob_addr(), 0));
        assert!(!statics.rule_admits(Some(&cell), composer_addr(), composer_addr(), 0));

        // A pending proposal moves the payer binding at its instant and
        // not before: the retired primary stops paying the moment the
        // recovery matures, with nothing applying it.
        let composer_rules = RoleTable::uniform(&StoredRule::Require(Presented::Identity(
            composer_addr().into(),
        )))
        .expect("a rule within the caps");
        let recovering = AuthCell {
            base: AuthBase::new(1_000, bob_rules()),
            proposal: Some(Proposal {
                effective_at_ms: 5_000,
                base: AuthBase::new(1_000, composer_rules),
            }),
        }
        .to_bytes()
        .unwrap();
        assert!(statics.rule_admits(Some(&recovering), composer_addr(), bob_addr(), 4_999));
        assert!(!statics.rule_admits(Some(&recovering), composer_addr(), composer_addr(), 4_999));
        assert!(statics.rule_admits(Some(&recovering), composer_addr(), composer_addr(), 5_000));
        assert!(!statics.rule_admits(Some(&recovering), composer_addr(), bob_addr(), 5_000));

        // A frozen account binds no fees: the acting entry was removed,
        // and an absent entry denies whoever asks — the recovery intent
        // pays from somewhere else.
        let mut frozen_roles = bob_rules();
        frozen_roles.remove(PRIMARY);
        let frozen = AuthCell::new(AuthBase::new(1_000, frozen_roles))
            .to_bytes()
            .unwrap();
        assert!(!statics.rule_admits(Some(&frozen), composer_addr(), bob_addr(), 0));
        assert!(!statics.rule_admits(Some(&frozen), composer_addr(), composer_addr(), 0));

        // Bytes no cell decodes from admit nobody — fail closed, like
        // the execution gate. Bare rule bytes are among them: the write
        // path stores frames.
        assert!(!statics.rule_admits(Some(&[0xFF, 0xFF]), composer_addr(), composer_addr(), 0));
        let bare = StoredRule::Require(Presented::Identity(bob_addr().into()))
            .to_bytes()
            .unwrap();
        assert!(!statics.rule_admits(Some(&bare), composer_addr(), bob_addr(), 0));
    }

    #[test]
    fn a_composed_envelope_derives_the_nullifier_write() {
        let tree = composed_tree();
        let bob = key(9);
        let derived = statics()
            .derive(&envelope(&tree, &[&bob]))
            .expect("derives");

        let hash = tree.subintents[0].decl.hash(&ProtocolHasher);
        assert_eq!(derived.subintent_hashes, vec![hash.0.0]);
        let nullifier = nullifier_key(&ProtocolHasher, bob_addr(), hash);
        assert!(derived.routing.write_keys.contains(&DeclaredKey::substate(
            bob_addr().address(),
            nullifier.local.0
        )));
    }

    #[test]
    fn a_fee_payer_the_composer_does_not_own_derives_unbound() {
        // The whole fee path debits whatever this field names, so
        // whether the payer's rule admits the signer is the payer
        // shard's block-validity verdict, taken where the payer's state
        // is. Derivation refuses nothing here: it records the identity
        // the envelope's key opens, and records it from the key rather
        // than from the payer field — so a stranger naming someone
        // else's account gets their own badge, never the account's.
        let tree = single_intent_tree(vec![
            sign_in(composer_addr()),
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 1, RES_X),
        ]);
        let mut stolen = envelope(&tree, &[]);
        stolen.fee_payer = bob_addr();
        let stolen = stolen.sign(&key(7));

        assert!(stolen.signature_is_valid(), "the composer signed it");
        let derived = statics().derive(&stolen).expect("derives");
        assert_eq!(
            derived.signer,
            composer_addr(),
            "the recorded identity is the key's, not the payer field's"
        );
        assert_ne!(
            derived.signer, stolen.fee_payer,
            "which is exactly the mismatch the payer shard's verdict reads"
        );
    }

    /// The signer's identity is derived under the scheme the envelope
    /// names, so a second scheme's key opens its own account and the
    /// same seed under two schemes is two accounts.
    #[test]
    fn the_signer_identity_derives_under_the_envelopes_scheme() {
        let secp = Secp256k1PrivateKey::from_bytes(&[7u8; 32]).expect("a scalar in range");
        let payer = principal_for(SchemeId::SECP256K1, &secp.public_key().0)
            .expect("a registered scheme opens an account");
        assert_ne!(
            payer,
            composer_addr(),
            "one seed, two schemes, two accounts"
        );

        let tree = single_intent_tree(vec![
            sign_in(payer),
            withdraw(payer, RES_X, 100),
            deposit_edge(bob_addr(), 1, RES_X),
        ]);
        let mut signed = envelope(&tree, &[]);
        signed.fee_payer = payer;
        let signed = signed.sign(&secp);

        assert!(signed.signature_is_valid());
        let derived = statics().derive(&signed).expect("derives");
        assert_eq!(derived.signer, payer);
    }

    /// A withdrawal from an account the envelope carries no signature for
    /// is well-formed and derives: the composer's own badge is presented,
    /// and what admission asks of a guarded call is that it present
    /// something. Whether Bob's account admits that badge is Bob's
    /// account's answer, and the engine's theft test is where it is
    /// asserted.
    #[test]
    fn a_withdrawal_from_an_unsigned_account_derives_and_defers() {
        let tree = single_intent_tree(vec![
            sign_in(composer_addr()),
            withdraw(bob_addr(), RES_X, 100),
            deposit_edge(composer_addr(), 1, RES_X),
        ]);
        assert!(statics().derive(&envelope(&tree, &[])).is_ok());

        // Reversed, it is the ordinary transfer: the composer withdraws
        // from their own account and Bob is credited without being asked.
        // One signature, because only the spending side is gated.
        let transfer = single_intent_tree(vec![
            sign_in(composer_addr()),
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 1, RES_X),
        ]);
        assert!(statics().derive(&envelope(&transfer, &[])).is_ok());
    }

    /// A method writing a leaf under its target's prefix and moving no
    /// funds is exactly the shape that is easy to leave open — so
    /// `securify`, which consumes nothing, is held to the same evidence
    /// rule a withdrawal is.
    #[test]
    fn a_leaf_write_presenting_nothing_is_refused() {
        let node = |evidence: BTreeSet<EvidenceRef>| GraphNode {
            target: composer_addr().into(),
            method: "securify".into(),
            args: vec![
                GraphArg::Literal(Value::Bytes(
                    RoleTable::uniform(&StoredRule::Require(Presented::Identity(
                        bob_addr().into(),
                    )))
                    .expect("a rule within the caps")
                    .to_bytes()
                    .unwrap(),
                )),
                GraphArg::Literal(Value::U64(86_400_000)),
            ],
            evidence,
        };
        // A guarded method reached with no evidence at all is a defect in
        // the signed form, so derivation refuses it and nobody pays.
        // Whether the evidence a call *does* present satisfies its
        // target is the target's own question, answered at execution.
        let refused = statics()
            .derive(&envelope(
                &single_intent_tree(vec![node(BTreeSet::new())]),
                &[],
            ))
            .expect_err("refuses");
        assert!(refused.0.contains("evidence"), "{}", refused.0);
        // A signature proof is not this method's to read either: it signs
        // in, and the write takes what the sign-in minted.
        let refused = statics()
            .derive(&envelope(
                &single_intent_tree(vec![node([EvidenceRef::IntentSignature].into())]),
                &[],
            ))
            .expect_err("refuses");
        assert!(refused.0.contains("signature proof"), "{}", refused.0);
        assert!(
            statics()
                .derive(&envelope(
                    &single_intent_tree(vec![
                        sign_in(composer_addr()),
                        node([EvidenceRef::Node(0)].into()),
                    ]),
                    &[]
                ))
                .is_ok()
        );
    }

    /// A proof is scoped to the intent whose signature produced it, so a
    /// node draws the identity of its own intent's signer and no other —
    /// which is the mechanism the subintent primitive was built for.
    #[test]
    fn a_proof_carries_its_own_intents_signer() {
        let bob = key(9);
        let derived = statics()
            .derive(&envelope(&composed_tree(), &[&bob]))
            .expect("both sides withdraw from themselves");
        // Nothing about the identities survives into the routing view,
        // so what this pins is that the tree derives at all: each intent
        // presents its own signer, and the withdrawals name those same
        // accounts.
        assert!(!derived.routing.write_keys.is_empty());

        // The same envelope with Bob's withdrawal moved into the
        // composer's intent still derives — its proof now carries the
        // composer, which Bob's account does not admit, and the verdict
        // on that is the account's to give at execution.
        let mut stolen = composed_tree();
        stolen.root.graph.nodes[1] = withdraw(bob_addr(), RES_X, 100);
        assert!(statics().derive(&envelope(&stolen, &[&bob])).is_ok());
    }

    #[test]
    fn a_mismatched_subintent_signer_is_refused() {
        // The tree binds BOB's address, but the carried key is another's.
        let tree = composed_tree();
        let impostor = key(11);
        let refused = statics().derive(&envelope(&tree, &[&impostor]));
        assert!(refused.is_err());

        // A missing signature list is a distinct refusal.
        let mut unsigned = envelope(&tree, &[&key(9)]);
        unsigned.subintent_sigs.clear();
        assert!(statics().derive(&unsigned).is_err());
    }

    #[test]
    fn an_inadmissible_tree_is_refused() {
        // The produced bucket is never consumed: linearity refuses it.
        let tree = single_intent_tree(vec![withdraw(composer_addr(), RES_X, 100)]);
        assert!(statics().derive(&envelope(&tree, &[])).is_err());
    }

    #[test]
    fn a_nullifier_hash_needs_a_subintent_hash_type() {
        // Pin the record type wiring: the routed hash is the declaration
        // hash, reconstructible from the decoded tree alone.
        let tree = composed_tree();
        let decoded = decode_tree(&encode_tree(&tree)).unwrap();
        assert_eq!(
            decoded.subintents[0].decl.hash(&ProtocolHasher),
            tree.subintents[0].decl.hash(&ProtocolHasher)
        );
        let _typed: SubintentHash = tree.subintents[0].decl.hash(&ProtocolHasher);
    }

    /// The provision-weight cap: cells count one, ranges count their
    /// declared cap, and a set the wire codec could not carry as one
    /// bundle is refused at admission.
    #[test]
    fn a_declaration_past_one_bundles_weight_is_refused() {
        let owner = Address::new([9; 31], AddressClass::Component);
        let range = |cap: u32, salt: u128| {
            DeclaredKey::Range(DeclaredRange {
                owner,
                collection: CollectionId([3; 16]),
                lo: salt,
                hi: salt,
                cap,
            })
        };
        let cap_u32 = u32::try_from(MAX_STATE_ENTRIES_PER_TX).unwrap();

        let at_cap: BTreeSet<DeclaredKey> = [range(cap_u32, 0)].into();
        assert!(check_provision_weight(&at_cap).is_ok());

        let over: BTreeSet<DeclaredKey> = [range(cap_u32, 0), range(1, 1)].into();
        assert!(check_provision_weight(&over).is_err());

        let cells_count: BTreeSet<DeclaredKey> = [
            DeclaredKey::substate(owner, [1; 16]),
            range(cap_u32 - 1, 0),
            range(1, 1),
        ]
        .into();
        assert!(check_provision_weight(&cells_count).is_err());
    }
}
