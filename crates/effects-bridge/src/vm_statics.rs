//! The envelope's static derivation: the tree wire codec and the
//! [`VmStatics`] implementation admission verifies through.
//!
//! The envelope tree travels as canonical HBOR of the mirror types
//! below — the vocabulary crate deliberately has no wire encoding, so
//! this module owns it. Derivation is `decode → admit → route` over the
//! process's genesis-static metadata, rooted at the envelope's signing
//! hash, projected into the workspace's admission vocabulary:
//! substate-granular keys for point effects, owner-granular keys for
//! collection effects (entries and ranges conflict at their owner —
//! conservative, never unsound), reads and snapshots in the shared
//! class, every other mode exclusive. Subintent nullifier creation
//! writes ride the routed sets, so admission conflicts on them like any
//! other exclusive key.

use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};

use arc_swap::ArcSwap;
use hyperscale_hbor::{from_slice as hbor_from_slice, to_vec as hbor_to_vec};
use hyperscale_types::{
    DeclaredKey, Derived, EnvelopeExt, Routing, TransactionEnvelope, VmStatics, VmStaticsError,
    declared_work,
};
use hyperscale_vm_effects::stdlib::{ENTROPY, VALIDATORS, VAULT, XRD as XRD_ROLE};
use hyperscale_vm_effects::{
    Accessibility, Address, EffectSet, EffectTarget, EnvelopeTree, InstanceRegistry, ManifestHash,
    MetadataCache, Mode, PackageHash, PackageMetadata, PrefixShardResolver, RoleId,
    Routing as RoutedTransaction, SchemeId, SubstateKey, Value, admit_tree, child_key, footprint,
    native_address, package_hash, principal_address, route_tree,
};

use crate::ProtocolHasher;
use crate::artifact::admit_package;

/// The native fee and transfer resource of the VM namespace.
///
/// Derived from its protocol role rather than picked, so it sits where a
/// hash puts it and no shard holds it by preference.
pub static XRD: LazyLock<Address> = LazyLock::new(|| native_address(&ProtocolHasher, XRD_ROLE));

/// The vault cell for `resource` under `owner` — the same child key the
/// stdlib account metadata's effect clauses compute.
#[must_use]
pub fn vault_key(owner: Address, resource: Address) -> SubstateKey {
    child_key(
        &ProtocolHasher,
        owner,
        VAULT,
        &[Value::Address(resource).canonical_bytes()],
    )
}

/// A stake pool's record of one validator it operates: the cell the
/// pool's operator methods declare, keyed by the validator.
///
/// Non-empty means the pool took this validator on and holds the key it
/// registered. The methods that speak about an existing validator read
/// it, so genesis has to write it for members the beacon created before
/// the contract existed.
#[must_use]
pub fn validator_key(pool: Address, validator: u64) -> SubstateKey {
    child_key(
        &ProtocolHasher,
        pool,
        VALIDATORS,
        &[Value::U64(validator).canonical_bytes()],
    )
}

/// An account's entropy leaf: where the stdlib's `stamp-entropy` records
/// the transaction's randomness draw. Mirrors the effect signature the
/// method declares.
#[must_use]
pub fn entropy_key(owner: Address) -> SubstateKey {
    child_key(&ProtocolHasher, owner, ENTROPY, &[])
}

/// The role a published package's artifact sits under, in the reserved
/// band the vocabulary's nullifier role occupies the top of.
///
/// A package cell lives under its publisher's own prefix, and no
/// package's metadata can declare an effect on this role — the account
/// signatures name vault, claims, config and entropy — so the cell is
/// reachable by the publish path and by nothing else.
pub const PACKAGE_ROLE: RoleId = RoleId(0xFFFE);

/// Where `publisher`'s copy of the package addressed by `package` lives.
///
/// Keyed by content address under the publisher, so republishing the
/// same artifact is the same cell — which is what makes publishing
/// idempotent rather than a conflict.
#[must_use]
pub fn package_key(publisher: Address, package: PackageHash) -> SubstateKey {
    child_key(
        &ProtocolHasher,
        publisher,
        PACKAGE_ROLE,
        &[package.0.0.to_vec()],
    )
}

/// The principal address an ed25519 public key opens.
///
/// The address commits to the key and the scheme, so genesis funding,
/// transaction builders and admission all derive the same address from
/// the same key — and admission verifies a signer against its target by
/// recomputing this, with nothing to look up.
#[must_use]
pub fn account_address(public_key: &[u8; 32]) -> Address {
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

/// The admission key for one effect target: substate-granular for points,
/// owner-granular for collection targets.
const fn admission_key(target: &EffectTarget) -> DeclaredKey {
    match target {
        EffectTarget::Point(key) => DeclaredKey::Cell(*key),
        EffectTarget::Entry { owner, .. } | EffectTarget::Range { owner, .. } => {
            DeclaredKey::Prefix(*owner)
        }
    }
}

/// The envelope's identity: its signing hash through the workspace's
/// protocol hash, as the vocabulary's hash type.
#[must_use]
pub fn envelope_identity(vm: &TransactionEnvelope) -> ManifestHash {
    ManifestHash(vm.signing_hash().as_hash32())
}

/// The bridge's [`VmStatics`]: `decode → admit → route` over the
/// process's genesis-static metadata.
pub struct BridgeStatics {
    /// Published package metadata, growing as blocks commit.
    pub cache: PackageCache,
    /// Instance registrations, genesis-static.
    pub instances: InstanceRegistry,
}

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
    pub fn publish(&self, package: PackageHash, metadata: PackageMetadata) {
        if self.load().get(package).is_some() {
            return;
        }
        let mut next = (*self.load()).clone();
        next.publish(package, metadata);
        self.0.store(Arc::new(next));
    }

    /// Publish the package a committed cell holds, if it holds one.
    ///
    /// A package cell is self-identifying: its key is the content
    /// address of the very bytes it stores, so recomputing the address
    /// from the value and rebuilding the key answers whether this cell
    /// is a package without any side channel, any tag, and any trust in
    /// what wrote it. A cell of any other kind cannot match except by
    /// finding a hash collision.
    pub fn absorb_cell(&self, owner: Address, local: [u8; 16], value: &[u8]) {
        let package = package_hash(&ProtocolHasher, value);
        if package_key(owner, package).local.0 != local {
            return;
        }
        if let Ok(metadata) = admit_package(value) {
            self.publish(package, metadata);
        }
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
        let work = declared_work(footprint, vm.gas_limit);

        Ok(Derived {
            work,
            routing: Routing {
                read_prefixes: Vec::new(),
                write_prefixes: vec![publisher],
                provision_prefixes: Vec::new(),
                read_keys: Vec::new(),
                declared_modes: write_keys.iter().map(|key| (*key, Mode::Write)).collect(),
                write_keys,
                provision_keys: Vec::new(),
            },
            subintent_hashes: Vec::new(),
            fee_vault_local: vault.local.0,
        })
    }
}

/// Refuse a node whose target's method admits a principal the intent
/// carrying the node is not.
///
/// One authority per intent: the composer's account covers the root
/// intent's nodes, and a subintent's declared signer covers its own.
/// Nothing crosses that line — a second party signs their own
/// declaration and never the composer's, which is what the subintent
/// primitive exists to express.
///
/// What this judges is the structural half — that each intent's declared
/// authority covers the nodes it carries. The cryptographic half, that a
/// declared authority is backed by a signature in the envelope, is the
/// fee-payer and subintent-signer bindings in [`BridgeStatics::derive`].
/// Together they are one rule, and it is a pure function of signed
/// content: no state read, no rule evaluation, and a refusal that costs
/// the sender nothing because the transaction never enters a block.
///
/// A method may name its principal two ways, and both resolve here
/// against content that cannot change after the target exists. The
/// target's own authority is satisfiable only by the key its address
/// derives from, so a gated method on a target no key derives — a
/// component instance, say — is uncallable rather than open. A method
/// naming a configuration field reaches a principal the instance was
/// created with, which is how an object nobody owns admits somebody at
/// all; a field that holds no address names nobody, and the method is
/// uncallable for the same reason. Both fall the safe way.
///
/// # Errors
///
/// [`VmStaticsError`] naming the first node whose principal the envelope
/// does not carry.
pub fn check_target_authority(
    tree: &EnvelopeTree,
    composer: Address,
    packages: &MetadataCache,
    instances: &InstanceRegistry,
) -> Result<(), VmStaticsError> {
    let root = std::iter::once((composer, &tree.root, None));
    let bound = tree
        .subintents
        .iter()
        .enumerate()
        .map(|(index, subintent)| (subintent.signer, &subintent.decl, Some(index)));
    for (authority, decl, subintent) in root.chain(bound) {
        for (position, node) in decl.graph.nodes.iter().enumerate() {
            // A target that resolves to nothing is admission's refusal to
            // make, and admission makes it.
            let Some(meta) = instances.get(node.target) else {
                continue;
            };
            let Some(signature) = packages
                .get(meta.package)
                .and_then(|package| package.methods.get(&node.method))
            else {
                continue;
            };
            let admits = match signature.accessibility {
                Accessibility::Public => continue,
                Accessibility::RequiresTargetAuth => Some(node.target),
                Accessibility::RequiresConfiguredAuth(field) => {
                    match meta.config.get(field as usize) {
                        Some(Value::Address(principal)) => Some(*principal),
                        _ => None,
                    }
                }
            };
            if admits == Some(authority) {
                continue;
            }
            let intent = subintent.map_or_else(
                || "the root intent".to_owned(),
                |index| format!("subintent {index}"),
            );
            return Err(VmStaticsError(format!(
                "{intent} node {position} calls `{}`, which admits an authority the envelope does \
                 not carry",
                node.method
            )));
        }
    }
    Ok(())
}

impl VmStatics for BridgeStatics {
    fn absorb_committed_cell(&self, owner: [u8; 32], local: [u8; 16], value: &[u8]) {
        let Ok(owner) = Address::from_bytes(owner) else {
            return;
        };
        self.cache.absorb_cell(owner, local, value);
    }

    fn derive(&self, vm: &TransactionEnvelope) -> Result<Derived, VmStaticsError> {
        // The payer is the composer, and this is what makes that true.
        // Every fee rule debits the account this field names — the
        // reservation a payer shard enforces as block validity, the
        // burn a completed transaction writes, the floor an abort
        // settles — and the composer's signature is the only authority
        // in the envelope. An unbound payer field is therefore a debit
        // on an account that authorised nothing, spendable by anyone
        // who knows its address.
        if account_address(&vm.signer) != vm.fee_payer {
            return Err(VmStaticsError(
                "fee payer is not the composer's own account".into(),
            ));
        }
        if let Some(artifact) = vm.artifact() {
            return Self::derive_publish(vm, artifact);
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
            if account_address(&sig.public_key) != subintent.signer {
                return Err(VmStaticsError(format!(
                    "subintent {index} signer address does not match its public key"
                )));
            }
        }
        let packages = self.cache.load();
        // What the fee-payer binding above does for the one field that
        // debits an account, generalised to every node that touches one.
        check_target_authority(&tree, vm.fee_payer, &packages, &self.instances)?;
        let admitted = admit_tree(
            &tree,
            envelope_identity(vm),
            &packages,
            &self.instances,
            &ProtocolHasher,
        )
        .map_err(|error| VmStaticsError(format!("admission: {error}")))?;
        let routing = route_tree(
            &admitted,
            &packages,
            &self.instances,
            &ProtocolHasher,
            &PrefixShardResolver { bits: 0 },
        )
        .map_err(|error| VmStaticsError(format!("routing: {error}")))?;

        let DeclaredAccess {
            read_keys,
            write_keys,
            provision_keys,
            declared_modes,
        } = classify_declared_access(&routing);
        let prefixes = |keys: &BTreeSet<DeclaredKey>| -> Vec<Address> {
            keys.iter()
                .map(DeclaredKey::owner)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        };
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
        );
        Ok(Derived {
            work,
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
        })
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::{Ed25519PrivateKey, NetworkId, SubintentSig, TX_UNITS, TransactionBody};
    use hyperscale_vm_effects::stdlib::{VAULT, account_metadata};
    use hyperscale_vm_effects::{
        AddressClass, Constraint, EdgeRef, GraphArg, GraphNode, Hasher, InstanceMeta, IntentDecl,
        ManifestGraph, PackageHash, Subintent, SubintentHash, YieldBinding, YieldParam, child_key,
        nullifier_key,
    };

    use super::*;

    const RES_X: Address = Address::new([0xE1; 31], AddressClass::Component);
    const RES_Y: Address = Address::new([0xE2; 31], AddressClass::Component);

    fn key(seed: u8) -> Ed25519PrivateKey {
        Ed25519PrivateKey::from_bytes(&[seed; 32]).unwrap()
    }

    fn composer_addr() -> Address {
        account_address(&key(7).public_key().0)
    }

    fn bob_addr() -> Address {
        account_address(&key(9).public_key().0)
    }

    fn statics() -> BridgeStatics {
        let package = PackageHash(ProtocolHasher.hash(b"package", &[b"account"]));
        let mut cache = MetadataCache::new();
        cache.publish(package, account_metadata());
        let mut instances = InstanceRegistry::new();
        for address in [composer_addr(), bob_addr()] {
            instances.register(
                address,
                InstanceMeta {
                    package,
                    config: vec![],
                },
            );
        }
        BridgeStatics {
            cache: PackageCache::new(cache),
            instances,
        }
    }

    fn withdraw(target: Address, resource: Address, amount: u128) -> GraphNode {
        GraphNode {
            target,
            method: "withdraw".into(),
            args: vec![
                GraphArg::Literal(Value::Address(resource)),
                GraphArg::Literal(Value::U128(amount)),
            ],
        }
    }

    fn deposit_edge(target: Address, producer: u32, resource: Address) -> GraphNode {
        GraphNode {
            target,
            method: "deposit".into(),
            args: vec![GraphArg::Edge {
                edge: EdgeRef {
                    producer,
                    output: 0,
                },
                constraints: vec![Constraint::ResourceIs(resource)],
            }],
        }
    }

    fn deposit_param(target: Address, param: u32) -> GraphNode {
        GraphNode {
            target,
            method: "deposit".into(),
            args: vec![GraphArg::Param(param)],
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
        }
    }

    /// The two-signer composition: the composer pays X for the
    /// subintent's Y.
    fn composed_tree() -> EnvelopeTree {
        EnvelopeTree {
            root: IntentDecl {
                graph: ManifestGraph {
                    nodes: vec![
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
                    producer: 0,
                    output: 0,
                },
            }],
            subintents: vec![Subintent {
                decl: IntentDecl {
                    graph: ManifestGraph {
                        nodes: vec![
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
                        producer: 0,
                        output: 0,
                    },
                }],
            }],
        }
    }

    fn envelope(tree: &EnvelopeTree, subintent_keys: &[&Ed25519PrivateKey]) -> TransactionEnvelope {
        let subintent_sigs = tree
            .subintents
            .iter()
            .zip(subintent_keys)
            .map(|(subintent, signer)| {
                let hash = subintent.decl.hash(&ProtocolHasher);
                SubintentSig {
                    public_key: signer.public_key().0,
                    signature: signer.sign(hash.0.0).0,
                }
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
            signer: [0; 32],
            signature: [0; 64],
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
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 0, RES_X),
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
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 0, RES_X),
            withdraw(composer_addr(), RES_Y, 10),
            deposit_edge(bob_addr(), 2, RES_Y),
        ]);
        let wider = statics().derive(&envelope(&wider, &[])).expect("derives");
        assert!(
            wider.work > derived.work,
            "a wider declaration must not be cheaper: {} vs {}",
            wider.work,
            derived.work
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
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 0, RES_X),
        ]);
        let derived = statics().derive(&envelope(&tree, &[])).expect("derives");

        // Reserve at the sender's vault and deltas at the recipient's:
        // all exclusive-class, substate-granular, under the two owners.
        let sender_vault = child_key(
            &ProtocolHasher,
            composer_addr(),
            VAULT,
            &[Value::Address(RES_X).canonical_bytes()],
        );
        assert!(derived.routing.write_keys.contains(&DeclaredKey::substate(
            composer_addr(),
            sender_vault.local.0
        )));
        assert!(derived.routing.read_keys.is_empty());
        // A commutative-only transfer provisions nothing at all.
        assert!(derived.routing.provision_keys.is_empty());
        assert!(derived.routing.provision_prefixes.is_empty());
        assert!(derived.subintent_hashes.is_empty());
        let mut owners = vec![composer_addr(), bob_addr()];
        owners.sort_unstable();
        assert_eq!(derived.routing.write_prefixes, owners);
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
        assert!(
            derived
                .routing
                .write_keys
                .contains(&DeclaredKey::substate(bob_addr(), nullifier.local.0))
        );
    }

    #[test]
    fn a_fee_payer_the_composer_does_not_own_is_refused() {
        // The whole fee path debits whatever this field names, and the
        // composer's signature is the only authority in the envelope —
        // so naming someone else's account has to be refused before any
        // of it runs, or that account is spendable by a stranger.
        let tree = single_intent_tree(vec![
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 0, RES_X),
        ]);
        let mut stolen = envelope(&tree, &[]);
        stolen.fee_payer = bob_addr();
        let stolen = stolen.sign(&key(7));

        assert!(stolen.signature_is_valid(), "the composer signed it");
        let refused = statics().derive(&stolen).expect_err("refuses");
        assert!(refused.0.contains("fee payer"), "{}", refused.0);

        // The composer paying from their own account is the admitted
        // case, so the check bites on ownership and not on fees at all.
        assert!(statics().derive(&envelope(&tree, &[])).is_ok());
    }

    /// The theft the gate closes: a manifest withdrawing from an account
    /// the envelope carries no signature for.
    #[test]
    fn a_withdrawal_from_an_unsigned_account_is_refused() {
        let tree = single_intent_tree(vec![
            withdraw(bob_addr(), RES_X, 100),
            deposit_edge(composer_addr(), 0, RES_X),
        ]);
        let refused = statics()
            .derive(&envelope(&tree, &[]))
            .expect_err("refuses");
        assert!(
            refused.0.contains("the root intent node 0"),
            "{}",
            refused.0
        );
        assert!(refused.0.contains("withdraw"), "{}", refused.0);

        // Reversed, it is the ordinary transfer: the composer withdraws
        // from their own account and Bob is credited without being asked.
        // One signature, because only the spending side is gated.
        let transfer = single_intent_tree(vec![
            withdraw(composer_addr(), RES_X, 100),
            deposit_edge(bob_addr(), 0, RES_X),
        ]);
        assert!(statics().derive(&envelope(&transfer, &[])).is_ok());
    }

    /// The stamp writes a leaf under its target's prefix and moves no
    /// funds, which is exactly why it is easy to leave open.
    #[test]
    fn a_stamp_on_an_unsigned_account_is_refused() {
        let stamp = |target: Address| {
            single_intent_tree(vec![GraphNode {
                target,
                method: "stamp-entropy".into(),
                args: vec![],
            }])
        };
        let refused = statics()
            .derive(&envelope(&stamp(bob_addr()), &[]))
            .expect_err("refuses");
        assert!(refused.0.contains("stamp-entropy"), "{}", refused.0);
        assert!(
            statics()
                .derive(&envelope(&stamp(composer_addr()), &[]))
                .is_ok()
        );
    }

    /// A second party's funds are reachable exactly when that party
    /// signed the node that touches them — which is the mechanism the
    /// subintent primitive was built for.
    #[test]
    fn a_subintents_signature_covers_its_own_nodes_and_no_others() {
        let bob = key(9);
        // Both sides withdraw from themselves under their own signature.
        assert!(
            statics()
                .derive(&envelope(&composed_tree(), &[&bob]))
                .is_ok()
        );

        // The same envelope with Bob's withdrawal moved into the
        // composer's intent: Bob signed a declaration, not this node, so
        // his signature does not reach it.
        let mut stolen = composed_tree();
        stolen.root.graph.nodes[0] = withdraw(bob_addr(), RES_X, 100);
        let refused = statics()
            .derive(&envelope(&stolen, &[&bob]))
            .expect_err("refuses");
        assert!(refused.0.contains("the root intent"), "{}", refused.0);

        // And the mirror: a subintent reaching into the composer's
        // account. The composer signed the envelope, not this subintent's
        // declaration, so the composer's key does not reach it either.
        let mut reversed = composed_tree();
        reversed.subintents[0].decl.graph.nodes[0] = withdraw(composer_addr(), RES_Y, 10);
        let refused = statics()
            .derive(&envelope(&reversed, &[&bob]))
            .expect_err("refuses");
        assert!(refused.0.contains("subintent 0"), "{}", refused.0);
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
}
