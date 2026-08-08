//! [`Transaction`] — the carried form of a signed envelope.
//!
//! [`Transaction`] is the raw wire form. Its verified form is
//! `Verified<Transaction>`; predicate at
//! [`impl Verify<NetworkId>`](Verify::verify) below.
//!
//! The admission conflict keys, the owner prefixes that place the
//! transaction on shards, and the validity window are all derived
//! locally from the signed envelope rather than carried beside it — a
//! sender cannot claim a placement its content does not earn. The
//! derivations are cached on the value.

use std::fmt::{self, Debug, Formatter};
use std::sync::OnceLock;

use blake3::Hasher;
use hyperscale_hbor::{Hbor, from_slice as hbor_from_slice, to_vec as hbor_to_vec};
use thiserror::Error;

use crate::crypto::{Ed25519PublicKey, Ed25519Signature, verify_ed25519};
use crate::transaction::vm::vm_statics;
use crate::{
    DeclaredKey, Derived, EnvelopeExt, Hash, LocalKey, MAX_TX_BYTES_LEN, NetworkId, Routing,
    ShardTrie, SubstateKey, TimestampRange, TransactionEnvelope, TxHash, Verified, Verify,
    VmStaticsError,
};

/// A signed transaction as the network carries it.
///
/// `serialized_bytes` is the canonical wire form — the HBOR
/// encoding of the envelope. The hash covers those exact bytes, so a peer
/// cannot ship one encoding and have us key it by another. Every other
/// field is a lazily-populated cache, skipped on the wire and rebuilt at
/// each end.
#[derive(Hbor)]
pub struct Transaction {
    /// HBOR-encoded [`TransactionEnvelope`] bytes — the canonical wire form.
    #[hbor(max = MAX_TX_BYTES_LEN)]
    serialized_bytes: Vec<u8>,

    /// Decoded envelope, populated by `body()` on first access from
    /// `serialized_bytes`. Constructors pre-populate. Not on the wire.
    #[hbor(skip)]
    body: OnceLock<TransactionEnvelope>,

    /// Derived routing identity and subintent claims, populated at
    /// verification (or lazily for committed transactions). Not on the
    /// wire — derivation is local by construction.
    #[hbor(skip)]
    derived: OnceLock<Derived>,

    /// Content hash, populated on first call to `hash()` via
    /// `blake3(&serialized_bytes)`. `::new` pre-populates. Not on the
    /// wire — recomputed at each end so a peer can't ship `(hash=X,
    /// tx_bytes=Y)` and have us key the bogus body by X.
    #[hbor(skip)]
    hash: OnceLock<Hash>,

    /// Pre-encoded wire bytes of the full `Transaction`,
    /// populated lazily by `cached_wire_bytes()`. Lets the commit thread
    /// hand bytes to `cf_put_raw` without re-encoding.
    #[hbor(skip)]
    cached_bytes: OnceLock<Vec<u8>>,
}

// Manual PartialEq/Eq - compare by hash for efficiency
impl PartialEq for Transaction {
    fn eq(&self, other: &Self) -> bool {
        self.hash() == other.hash()
    }
}

impl Eq for Transaction {}

// Manual Clone - OnceLock doesn't implement Clone. Every populated cache
// is copied so the clone doesn't pay first-access cost twice; in
// particular the derivation rides across clones, so the work a fresh tx
// incurs at admission is amortized over every later raw clone (tick-state
// extract, mempool block-commit lift, proposal build).
impl Clone for Transaction {
    fn clone(&self) -> Self {
        let body = OnceLock::new();
        if let Some(t) = self.body.get() {
            let _ = body.set(t.clone());
        }
        let derived = OnceLock::new();
        if let Some(r) = self.derived.get() {
            let _ = derived.set(r.clone());
        }
        let hash = OnceLock::new();
        if let Some(h) = self.hash.get() {
            let _ = hash.set(*h);
        }
        let cached_bytes = OnceLock::new();
        if let Some(b) = self.cached_bytes.get() {
            let _ = cached_bytes.set(b.clone());
        }
        Self {
            serialized_bytes: self.serialized_bytes.clone(),
            body,
            derived,
            hash,
            cached_bytes,
        }
    }
}

// Manual Debug — skip the cached_bytes field.
impl Debug for Transaction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Transaction")
            .field("hash", &self.hash())
            .finish_non_exhaustive()
    }
}

impl Transaction {
    /// The admission conflict keys for this transaction's reads — the
    /// routed effect sets' shared-mode keys. Nothing key-granular is
    /// carried on the wire.
    #[must_use]
    pub fn admission_read_keys(&self) -> Vec<DeclaredKey> {
        self.routing().read_keys.clone()
    }

    /// The admission conflict keys for this transaction's writes; see
    /// [`Self::admission_read_keys`].
    #[must_use]
    pub fn admission_write_keys(&self) -> Vec<DeclaredKey> {
        self.routing().write_keys.clone()
    }

    /// Every admission conflict key, reads then writes.
    #[must_use]
    pub fn admission_keys(&self) -> Vec<DeclaredKey> {
        let mut keys = self.admission_read_keys();
        keys.extend(self.admission_write_keys());
        keys
    }

    /// What including this transaction costs a block, in work units:
    /// the fixed admit-and-track charge, the declared footprint, and the
    /// signed gas limit.
    ///
    /// The packing bound sums this over the drain. Derived locally, so
    /// it is not something a sender can understate.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::routing`].
    #[must_use]
    pub fn work(&self) -> u64 {
        self.derived().work
    }

    /// Half-open `WeightedTimestamp` range during which this tx may be
    /// included in a block. Anchored on the parent QC's `weighted_timestamp`
    /// at every check site. Signer-chosen, chain-enforced.
    #[must_use]
    pub fn validity_range(&self) -> TimestampRange {
        self.body().validity_window()
    }

    /// Create a transaction from a signed envelope.
    ///
    /// # Panics
    ///
    /// Panics if the `TransactionEnvelope` cannot be HBOR-encoded; it is a
    /// closed wire type, so encoding is infallible in practice.
    #[must_use]
    pub fn new(vm: TransactionEnvelope) -> Self {
        let payload = hbor_to_vec(&vm).expect("an envelope within its caps encodes");
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let hash = Hash::from_hash_bytes(hasher.finalize().as_bytes());

        let body_lock = OnceLock::new();
        let _ = body_lock.set(vm);
        let hash_lock = OnceLock::new();
        let _ = hash_lock.set(hash);

        Self {
            serialized_bytes: payload,
            body: body_lock,
            derived: OnceLock::new(),
            hash: hash_lock,
            cached_bytes: OnceLock::new(),
        }
    }

    /// Get the transaction hash (content-addressed).
    ///
    /// Computes `blake3(serialized_bytes)` on first call and caches the
    /// result. `::new` pre-populates the cache.
    pub fn hash(&self) -> TxHash {
        TxHash::from(*self.hash.get_or_init(|| {
            let mut hasher = Hasher::new();
            hasher.update(&self.serialized_bytes);
            Hash::from_hash_bytes(hasher.finalize().as_bytes())
        }))
    }

    /// Decode the envelope, or refuse malformed bytes. The fallible path
    /// for wire input; [`Self::body`] is the post-verification accessor.
    ///
    /// # Errors
    ///
    /// [`TransactionVerifyError::UndecodableBody`] when the bytes
    /// do not decode as an envelope.
    pub fn try_body(&self) -> Result<&TransactionEnvelope, TransactionVerifyError> {
        if let Some(body) = self.body.get() {
            return Ok(body);
        }
        let decoded = hbor_from_slice::<TransactionEnvelope>(&self.serialized_bytes)
            .map_err(|_| TransactionVerifyError::UndecodableBody)?;
        Ok(self.body.get_or_init(|| decoded))
    }

    /// The decoded envelope. Constructors pre-populate it; wire-decoded
    /// transactions populate it at verification.
    ///
    /// # Panics
    ///
    /// Panics if `serialized_bytes` does not decode. Wire-decoded
    /// transactions are verified (which decodes fallibly) before this is
    /// invoked.
    pub fn body(&self) -> &TransactionEnvelope {
        self.try_body()
            .expect("Transaction.serialized_bytes failed body decode")
    }

    /// The derived routing identity.
    ///
    /// Derives through the installed [`crate::VmStatics`] on first access
    /// and caches per transaction.
    ///
    /// # Panics
    ///
    /// Panics if derivation refuses the envelope — unreachable for
    /// verified or committed transactions, whose envelopes already
    /// derived cleanly at admission — or if no statics are installed.
    #[must_use]
    pub fn routing(&self) -> &Routing {
        &self.derived().routing
    }

    /// The fee payer's native-resource vault cell — what the payer
    /// shard's reservation check reads and the fee settlement debits.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::routing`].
    #[must_use]
    pub fn fee_vault(&self) -> SubstateKey {
        SubstateKey {
            owner: self.body().fee_payer,
            local: LocalKey(self.derived().fee_vault_local),
        }
    }

    /// The cached derivation, or a panic naming the refusal.
    fn derived(&self) -> &Derived {
        self.try_derived()
            .unwrap_or_else(|error| panic!("derivation failed on an admitted transaction: {error}"))
    }

    /// Derive (or fetch the cached) envelope derivation, fallibly — the
    /// verification path.
    ///
    /// # Errors
    ///
    /// [`VmStaticsError`] from the installed derivation.
    pub fn try_derived(&self) -> Result<&Derived, VmStaticsError> {
        if let Some(derived) = self.derived.get() {
            return Ok(derived);
        }
        let derived = vm_statics().derive(self.body())?;
        Ok(self.derived.get_or_init(|| derived))
    }

    /// Get the cached serialized envelope bytes.
    ///
    /// Use this for computing transaction merkle roots (avoids
    /// re-serialization) and for network encoding.
    pub fn serialized_bytes(&self) -> &[u8] {
        &self.serialized_bytes
    }

    /// Pre-serialized wire bytes of the full `Transaction`.
    /// Computed on first call and cached.
    ///
    /// # Panics
    ///
    /// Panics if HBOR encoding fails — that's a programmer error since
    /// every field is `Hbor` and the type itself is closed.
    pub fn cached_wire_bytes(&self) -> &[u8] {
        self.cached_bytes
            .get_or_init(|| hbor_to_vec(self).expect("Transaction HBOR encode is infallible"))
    }

    /// Check if this transaction is cross-shard under a uniform `num_shards`-way
    /// partition. For the live partition use
    /// [`TopologySnapshot::is_cross_shard_transaction`], which routes against the
    /// active [`ShardTrie`]; this by-count form is for genesis and offline tooling.
    ///
    /// [`TopologySnapshot::is_cross_shard_transaction`]: crate::TopologySnapshot::is_cross_shard_transaction
    pub fn is_cross_shard(&self, num_shards: u64) -> bool {
        let trie = ShardTrie::uniform_from_count(num_shards);
        let mut shards = self
            .routing()
            .write_prefixes
            .iter()
            .map(|prefix| trie.shard_for_prefix(*prefix));
        let Some(first) = shards.next() else {
            return false;
        };
        shards.any(|shard| shard != first)
    }
}

/// Failure modes of [`Transaction`] verification.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum TransactionVerifyError {
    /// The body bytes do not decode as an envelope.
    #[error("transaction body bytes are undecodable")]
    UndecodableBody,
    /// The envelope names a different network than this session's.
    #[error("transaction is signed for network {signed}, not this network {session}")]
    WrongNetwork {
        /// The network the envelope names.
        signed: u8,
        /// The network this node verifies for.
        session: u8,
    },
    /// The envelope's composer signature does not cover its content.
    #[error("transaction signature is invalid")]
    InvalidSignature,
    /// A subintent signature does not cover its declaration hash.
    #[error("subintent {0} signature is invalid")]
    InvalidSubintentSignature(u32),
    /// Static derivation refused the envelope.
    #[error(transparent)]
    Derivation(#[from] VmStaticsError),
}

/// Construction asserts: the body decodes, the envelope names this
/// session's network, the composer's ed25519 signature covers the
/// envelope content, the tree admits and routes under the installed
/// [`crate::VmStatics`] (which caches the derived identity on the
/// transaction and binds every subintent signer address to its public
/// key), and every subintent signature covers its declaration hash.
///
/// The network check runs before the signature: the named network is
/// signed content, so a transaction composed for another network fails
/// here whatever its signature says, and a re-targeted one fails the
/// signature.
///
/// Construction goes through one of two gates:
///
/// - [`<Transaction as Verify>::verify`](Verify::verify) — runs
///   the predicate.
/// - [`Verified::<Transaction>::new_unchecked`] — re-wraps a
///   transaction whose predicate already held via an out-of-band trust
///   source (storage-recovery, where the value was validated before
///   persistence; equivalent-attestation paths). Every call site
///   carries a `// SAFETY:` comment naming the trust source.
impl Verify<NetworkId> for Transaction {
    type Error = TransactionVerifyError;

    fn verify(&self, network: NetworkId) -> Result<Verified<Self>, Self::Error> {
        let vm = self.try_body()?;
        if vm.network != network {
            return Err(TransactionVerifyError::WrongNetwork {
                signed: vm.network.0,
                session: network.0,
            });
        }
        if !vm.signature_is_valid() {
            return Err(TransactionVerifyError::InvalidSignature);
        }
        // Derivation checks the tree, the signature arity, and the
        // signer-address binding; the signatures themselves verify here,
        // over the derived declaration hashes.
        let derived = self.try_derived()?;
        for (index, (sig, subintent)) in vm
            .subintent_sigs
            .iter()
            .zip(&derived.subintent_hashes)
            .enumerate()
        {
            let valid = verify_ed25519(
                subintent,
                &Ed25519PublicKey(sig.public_key),
                &Ed25519Signature(sig.signature),
            );
            if !valid {
                return Err(TransactionVerifyError::InvalidSubintentSignature(
                    u32::try_from(index).unwrap_or(u32::MAX),
                ));
            }
        }
        Ok(Verified::new_unchecked(self.clone()))
    }
}

impl Verified<Transaction> {
    /// Re-wrap a `Transaction` whose trust derives from inclusion
    /// in a committed block.
    ///
    /// Trust chain (BFT-transitive):
    /// 1. The tx is contained in a `CertifiedBlock`.
    /// 2. `CertifiedBlock` carries a QC attesting ≥2f+1 voting power.
    /// 3. Voters refuse to vote on blocks whose `Block.transactions`
    ///    entries are not all `Verifiable::Verified` — see
    ///    `validate_block_for_vote` in `crates/shard`.
    /// 4. Therefore every tx in a committed block was admission-validated
    ///    by at least one honest voter through the standard `Verify` gate.
    ///
    /// Callers: mempool reload from a committed block, storage
    /// rehydration into block containers, and any other path that
    /// surfaces a raw `Transaction` whose container is itself
    /// the trust anchor.
    #[must_use]
    pub const fn from_persisted(tx: Transaction) -> Self {
        Self::new_unchecked(tx)
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_hbor::{
        DecodeError, from_slice as hbor_from_slice, to_vec as hbor_to_vec, varint,
    };
    use hyperscale_vm_types::{Address, Mode};

    use super::*;
    use crate::test_utils::test_validity_range;
    use crate::{
        Ed25519PrivateKey, SubintentSig, TransactionBody, VmStatics, declared_work,
        install_vm_statics,
    };

    struct StubStatics;

    /// The declaration hash the stub claims for `b"with-subintent"`
    /// trees; the fixture's subintent signature covers it.
    const STUB_SUBINTENT_HASH: [u8; 32] = [0x5A; 32];

    impl VmStatics for StubStatics {
        fn derive(&self, vm: &TransactionEnvelope) -> Result<Derived, VmStaticsError> {
            if vm.call_tree().unwrap_or_default() == b"inadmissible" {
                return Err(VmStaticsError("stub refusal".into()));
            }
            let subintent_hashes = if vm.call_tree().unwrap_or_default() == b"with-subintent" {
                vec![STUB_SUBINTENT_HASH]
            } else {
                Vec::new()
            };
            Ok(Derived {
                fee_vault_local: [0xEE; 16],
                routing: Routing {
                    read_keys: vec![DeclaredKey::prefix([0x11; 16])],
                    write_keys: vec![DeclaredKey::substate([0x22; 16], [0x01; 16])],
                    read_prefixes: vec![Address([0x11; 16])],
                    write_prefixes: vec![Address([0x22; 16])],
                    provision_keys: vec![DeclaredKey::prefix([0x11; 16])],
                    provision_prefixes: vec![Address([0x11; 16])],
                    declared_modes: vec![
                        (DeclaredKey::prefix([0x11; 16]), Mode::Read),
                        (DeclaredKey::substate([0x22; 16], [0x01; 16]), Mode::Write),
                    ],
                },
                subintent_hashes,
                work: declared_work(0, 0),
            })
        }
    }

    /// The network every fixture in this module signs for.
    const TEST_NETWORK: NetworkId = NetworkId(242);

    fn test_envelope(tree: &[u8]) -> TransactionEnvelope {
        let key = Ed25519PrivateKey::from_bytes(&[7u8; 32]).unwrap();
        let range = test_validity_range();
        TransactionEnvelope {
            body: TransactionBody::Call(tree.to_vec()),
            subintent_sigs: Vec::new(),
            fee_payer: Address([0xAA; 16]),
            max_fee: 1_000,
            gas_limit: 1_000_000,
            validity_start_ms: range.start_timestamp_inclusive.as_millis(),
            validity_end_ms: range.end_timestamp_exclusive.as_millis(),
            message: Vec::new(),
            network: TEST_NETWORK,
            signer: [0; 32],
            signature: [0; 64],
        }
        .sign(&key)
    }

    fn fixture(tree: &[u8]) -> Transaction {
        install_vm_statics(Box::new(StubStatics));
        Transaction::new(test_envelope(tree))
    }

    #[test]
    fn roundtrip_preserves_hash_and_body() {
        let tx = fixture(b"graph bytes");
        let bytes = hbor_to_vec(&tx).unwrap();
        let decoded: Transaction = hbor_from_slice(&bytes).unwrap();
        assert_eq!(decoded.hash(), tx.hash());
        assert_eq!(decoded.try_body().unwrap(), tx.body());
    }

    #[test]
    fn admission_keys_derive_through_the_installed_statics() {
        let tx = fixture(b"graph bytes");
        assert_eq!(
            tx.admission_read_keys(),
            vec![DeclaredKey::prefix([0x11; 16])]
        );
        assert_eq!(
            tx.admission_write_keys(),
            vec![DeclaredKey::substate([0x22; 16], [0x01; 16])]
        );
        assert!(!tx.is_cross_shard(1));
    }

    #[test]
    fn the_validity_window_is_read_off_the_signed_envelope() {
        let tx = fixture(b"graph bytes");
        assert_eq!(tx.validity_range(), test_validity_range());
    }

    #[test]
    fn verification_checks_signature_and_derivation() {
        let good = fixture(b"graph bytes");
        assert!(good.verify(TEST_NETWORK).is_ok());

        // A tampered signature refuses.
        let mut vm = good.body().clone();
        vm.signature[0] ^= 1;
        let bad_signature = Transaction::new(vm);
        assert_eq!(
            bad_signature.verify(TEST_NETWORK).unwrap_err(),
            TransactionVerifyError::InvalidSignature
        );

        // A refused tree surfaces the derivation error.
        let inadmissible = fixture(b"inadmissible");
        assert!(matches!(
            inadmissible.verify(TEST_NETWORK).unwrap_err(),
            TransactionVerifyError::Derivation(_)
        ));

        // Garbage in the body field is an undecodable body, not a panic.
        let mut bytes = fixture(b"graph bytes").serialized_bytes().to_vec();
        bytes.truncate(3);
        let garbage = Transaction {
            serialized_bytes: bytes,
            body: OnceLock::new(),
            derived: OnceLock::new(),
            hash: OnceLock::new(),
            cached_bytes: OnceLock::new(),
        };
        assert_eq!(
            garbage.verify(TEST_NETWORK).unwrap_err(),
            TransactionVerifyError::UndecodableBody
        );
    }

    /// A transaction signed for one network fails verification under
    /// another's session — before its signature is even consulted, and
    /// re-targeting it breaks the signature since the network is signed
    /// content.
    #[test]
    fn verification_refuses_a_foreign_networks_transaction() {
        let tx = fixture(b"graph bytes");
        assert!(tx.verify(TEST_NETWORK).is_ok());
        assert_eq!(
            tx.verify(NetworkId(7)).unwrap_err(),
            TransactionVerifyError::WrongNetwork {
                signed: TEST_NETWORK.0,
                session: 7,
            }
        );

        let mut retargeted = tx.body().clone();
        retargeted.network = NetworkId(7);
        assert_eq!(
            Transaction::new(retargeted)
                .verify(NetworkId(7))
                .unwrap_err(),
            TransactionVerifyError::InvalidSignature,
        );
    }

    #[test]
    fn verification_checks_subintent_signatures() {
        install_vm_statics(Box::new(StubStatics));

        // A subintent signature must cover the derived declaration hash.
        let subintent_key = Ed25519PrivateKey::from_bytes(&[9u8; 32]).unwrap();
        let composer_key = Ed25519PrivateKey::from_bytes(&[7u8; 32]).unwrap();
        let mut envelope = test_envelope(b"with-subintent");
        envelope.subintent_sigs = vec![SubintentSig {
            public_key: subintent_key.public_key().0,
            signature: subintent_key.sign(STUB_SUBINTENT_HASH).0,
        }];
        let composed = envelope.sign(&composer_key);
        assert!(
            Transaction::new(composed.clone())
                .verify(TEST_NETWORK)
                .is_ok()
        );

        let mut forged = composed;
        forged.subintent_sigs[0].signature[0] ^= 1;
        let forged = forged.sign(&composer_key);
        assert_eq!(
            Transaction::new(forged).verify(TEST_NETWORK).unwrap_err(),
            TransactionVerifyError::InvalidSubintentSignature(0)
        );
    }

    #[test]
    fn the_envelope_hash_covers_the_signed_window() {
        // Identical content in different windows is two transactions —
        // the signed window is the natural discriminator dedup keys on;
        // byte-identical envelopes stay one transaction.
        let key = Ed25519PrivateKey::from_bytes(&[7u8; 32]).unwrap();
        let base = test_envelope(b"graph bytes");
        let mut shifted = base.clone();
        shifted.validity_start_ms += 1;
        let shifted = shifted.sign(&key);
        assert_ne!(
            Transaction::new(base.clone()).hash(),
            Transaction::new(shifted).hash()
        );
        assert_eq!(
            Transaction::new(base.clone()).hash(),
            Transaction::new(base).hash()
        );
    }

    #[test]
    fn decoded_hash_is_blake3_of_tx_bytes_not_wire_value() {
        // The hash isn't on the wire; decode pulls only `serialized_bytes`
        // and the lazy `hash()` call computes blake3 over those bytes.
        let tx = fixture(b"graph bytes");
        let bytes = hbor_to_vec(&tx).unwrap();
        let decoded: Transaction = hbor_from_slice(&bytes).unwrap();
        let mut hasher = Hasher::new();
        hasher.update(decoded.serialized_bytes());
        let expected = TxHash::from(Hash::from_hash_bytes(hasher.finalize().as_bytes()));
        assert_eq!(decoded.hash(), expected);
    }

    #[test]
    fn decode_rejects_oversized_tx_bytes() {
        // Hand-roll a payload whose `serialized_bytes` length prefix
        // exceeds MAX_TX_BYTES_LEN. The bound check must fire before
        // allocating the full Vec.
        let mut buf = Vec::new();
        varint::write(&mut buf, MAX_TX_BYTES_LEN + 1).unwrap();
        buf.extend(std::iter::repeat_n(0u8, MAX_TX_BYTES_LEN + 1));
        let err = hbor_from_slice::<Transaction>(&buf).unwrap_err();
        assert!(matches!(
            err,
            DecodeError::BoundExceeded { max, actual }
                if max == MAX_TX_BYTES_LEN && actual == MAX_TX_BYTES_LEN + 1
        ));
    }

    #[test]
    fn decode_rejects_the_declared_set_shape() {
        // Hand-roll a wire layout carrying envelope bytes plus declared
        // read/write sets and a mirrored validity range, to confirm a
        // peer can't ship fields the derivation owns.
        let tx = fixture(b"graph bytes");
        let mut buf = hbor_to_vec(&tx).unwrap();
        buf.extend_from_slice(&hbor_to_vec(&Vec::<[u8; 16]>::new()).unwrap());
        buf.extend_from_slice(&hbor_to_vec(&Vec::<[u8; 16]>::new()).unwrap());
        buf.extend_from_slice(&hbor_to_vec(&0u64).unwrap());
        let err = hbor_from_slice::<Transaction>(&buf).unwrap_err();
        assert!(matches!(err, DecodeError::TrailingBytes { .. }));
    }
}
