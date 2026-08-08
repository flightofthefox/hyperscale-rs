//! Column family definitions, constants, and handle resolution.
//!
//! This is the single source of truth for what column families exist,
//! what they store, and how their keys/values are encoded.

use hyperscale_types::{
    BlockMetadata, ChainOrigin, ConsensusReceipt, ExecutionCertificate, ExecutionMetadata,
    Finalization, FinalizationHash, Hash, Round, SafeVoteRegisters, ShardWitnessPayload,
    SubstateKey, TickId, Transaction, ValidatorId,
};
use rocksdb::{ColumnFamily, DB};

use super::jmt_stored::{StaleTreePart, StoredNodeKey, VersionedStoredNode};
use super::substate_key::SubstateKeyCodec;
use super::versioned_key::VersionedSubstateKeyCodec;
use crate::typed_cf::{
    BeU64Codec, ChainOriginCodec, DbCodec, DbEncode, HashCodec, HborCodec, JmtKeyCodec, RawCodec,
    TypedCf,
};

// ─── CF name constants ───────────────────────────────────────────────────────

/// Column family name for the default CF (chain metadata, JMT metadata).
pub const DEFAULT_CF: &str = "default";

/// Column family name for substate data. Stores the current value per
/// unversioned substate key. History for recent writes lives in
/// `STATE_HISTORY_CF` (same key + write-version suffix, value is the
/// pre-write prior state). Current-state reads are a direct point
/// lookup; historical reads at version V seek the smallest
/// state-history entry for the key with `write_version > V` and return
/// its prior value.
pub const STATE_CF: &str = "state";

/// Column family name for the per-write state-history log used by
/// historical reads.
/// Key: `(substate_key, write_version)`; value: the prior value at that
/// key immediately before the write at `write_version`. A `None` value
/// means "key was absent before the write."
pub const STATE_HISTORY_CF: &str = "state_history";

/// Column family name for block metadata (header + manifest) keyed by height.
pub const BLOCKS_CF: &str = "blocks";

/// Column family name for transactions keyed by hash.
pub const TRANSACTIONS_CF: &str = "transactions";

/// Column family name for finalizations keyed by hash.
pub const CERTIFICATES_CF: &str = "certificates";

/// Column family name for JMT tree nodes.
pub const JMT_NODES_CF: &str = "jmt_nodes";

/// Column family for stale JMT nodes pending garbage collection.
/// Key: `version_BE_8B` (the version at which nodes became stale).
/// Value: HBOR-encoded `Vec<StaleTreePart>`.
/// GC deletes entries older than `current_version - jmt_history_length`.
pub const STALE_JMT_NODES_CF: &str = "stale_jmt_nodes";

/// Column family indexing `state_history` entries by their write version so
/// GC can delete retention-expired history without scanning the whole
/// `state_history` CF.
///
/// Key: `version_BE_8B` — the `write_version` at which these history entries
/// were created (one entry per block commit).
/// Value: HBOR-encoded `Vec<Vec<u8>>` — the list of raw `state_history` keys
/// (i.e. `storage_key_bytes ++ BE8(version)`) written at that version.
///
/// Written alongside every `state_history` entry. GC iterates this CF in
/// version order (cheap — version-keyed), breaks at `version >= cutoff`, and
/// issues one `delete_cf` per listed history key plus one for the stale-set
/// entry itself. Mirrors the `stale_jmt_nodes` pattern.
pub const STALE_STATE_HISTORY_CF: &str = "stale_state_history";

/// Column family for the consensus portion of stored receipts, keyed by
/// tx hash. Companion to [`EXECUTION_METADATA_CF`] (same key, separate CF
/// so metadata can be pruned on its own cycle).
pub const CONSENSUS_RECEIPTS_CF: &str = "consensus_receipts";

/// Column family for the local-only [`ExecutionMetadata`] (fees, logs,
/// error), keyed by tx hash. Absent when the tx was synced from a peer.
pub const EXECUTION_METADATA_CF: &str = "execution_metadata";

/// Column family for execution certificates keyed by [`TickId`].
pub const EXECUTION_CERTS_CF: &str = "execution_certs";

/// Column family mapping a transaction to the certificate carrying its
/// outcome, keyed by tx hash with a [`TickId`] value.
///
/// A counterpart shard asks for outcomes by transaction — it learned of
/// the transaction from our committed header and has no way to know which
/// certificate we ended up putting it in. This index is how the fetch
/// responder answers that from storage once the in-memory cache has
/// evicted. Written in the same batch as [`EXECUTION_CERTS_CF`], one entry
/// per attested outcome.
pub const TX_CERT_INDEX_CF: &str = "tx_cert_index";

/// Column family for beacon-witness leaves on this shard.
///
/// Key: `leaf_index` as a big-endian `u64` — lex order matches
/// monotonic leaf order so the fetch responder can range-scan to
/// reconstruct an accumulator at any committed block. Storage is
/// scoped per-shard, so the shard id is implicit in the key.
/// Value: HBOR-encoded [`ShardWitnessPayload`]. Append-only; pruning
/// follows the retention horizon configured at the runtime layer.
pub const BEACON_WITNESSES_CF: &str = "beacon_witnesses";

/// Column family for the committed substate byte total per version.
///
/// Key: `version_BE_8B`; value: HBOR-encoded `u64` — the sum of every JMT
/// leaf's value byte length after the commit at that version. Written in
/// the same batch as the commit (crash-consistent with the tree), one
/// entry per committed version. Consensus-critical: shard-witness
/// derivation reads the byte total behind a block's parent state, so it
/// must be identical on every replica. GC prunes entries below the same
/// `jmt_history_length` cutoff as historical tree data.
pub const SUBSTATE_BYTES_CF: &str = "substate_bytes";

/// Column family for durable safe-vote registers, keyed by validator.
///
/// Key: `validator_id_BE_8B`. Value: packed 32-byte record
/// `[chain_origin_16B][locked_round_BE_8B][last_voted_round_BE_8B]`.
/// The chain-origin tag binds the record to the chain incarnation that
/// wrote it — a child store seeded from a parent shard's checkpoint
/// carries the parent's records, and reads ignore any record whose tag
/// differs from the store's current origin. Written with a synchronous
/// (fsynced) write before the corresponding vote or timeout signature
/// leaves the process.
pub const SAFE_VOTE_REGISTERS_CF: &str = "safe_vote_registers";

/// Column family staging verified snap-sync chunks before finalize.
///
/// Key: the substate key's 32 bytes — its JMT leaf key, so the bytewise
/// comparator makes a full scan leaf-sorted, which the chunked finalize
/// build relies on. Value: the raw substate value. Written one atomic
/// batch per verified chunk together with the import progress record in
/// the default CF; the store proper is untouched until
/// `finalize_boundary_import` builds the state from the staged leaves
/// and clears this CF.
pub const IMPORT_STAGING_CF: &str = "import_staging";

// Default-CF metadata keys are defined as MetadataEntry types in typed_cf.rs.
// See CommittedHeightEntry, CommittedHashEntry, CommittedQcEntry, JmtMetadataEntry.

/// CFs with high write throughput — get larger write buffers and tiered compression.
/// State, state-history log, and JMT nodes are updated on every block commit.
pub const HOT_WRITE_COLUMN_FAMILIES: &[&str] = &[STATE_CF, STATE_HISTORY_CF, JMT_NODES_CF];

/// All column families used by the storage layer.
pub const ALL_COLUMN_FAMILIES: &[&str] = &[
    DEFAULT_CF,
    BLOCKS_CF,
    TRANSACTIONS_CF,
    STATE_CF,
    STATE_HISTORY_CF,
    STALE_STATE_HISTORY_CF,
    CERTIFICATES_CF,
    JMT_NODES_CF,
    STALE_JMT_NODES_CF,
    CONSENSUS_RECEIPTS_CF,
    EXECUTION_METADATA_CF,
    EXECUTION_CERTS_CF,
    TX_CERT_INDEX_CF,
    BEACON_WITNESSES_CF,
    SUBSTATE_BYTES_CF,
    SAFE_VOTE_REGISTERS_CF,
    IMPORT_STAGING_CF,
];

// ─── CfHandles ───────────────────────────────────────────────────────────────

/// Column family handles resolved from a `DB` reference.
///
/// Provides typed field access to all column families without repeating
/// `.cf_handle(NAME).expect(...)`. Cheap to construct (`HashMap` lookups only).
/// Column family handles — fields are private, access only through
/// [`TypedCf::handle()`](crate::typed_cf::TypedCf::handle).
pub struct CfHandles<'a> {
    state: &'a ColumnFamily,
    state_history: &'a ColumnFamily,
    stale_state_history: &'a ColumnFamily,
    blocks: &'a ColumnFamily,
    transactions: &'a ColumnFamily,
    certificates: &'a ColumnFamily,
    jmt_nodes: &'a ColumnFamily,
    stale_jmt_nodes: &'a ColumnFamily,
    consensus_receipts: &'a ColumnFamily,
    execution_metadata: &'a ColumnFamily,
    execution_certs: &'a ColumnFamily,
    tx_cert_index: &'a ColumnFamily,
    beacon_witnesses: &'a ColumnFamily,
    substate_bytes: &'a ColumnFamily,
    safe_vote_registers: &'a ColumnFamily,
    import_staging: &'a ColumnFamily,
}

impl<'a> CfHandles<'a> {
    /// Resolve all column family handles from the database.
    ///
    /// # Panics
    /// Panics if any expected column family is missing.
    pub fn resolve(db: &'a DB) -> Self {
        let resolve = |name: &str| -> &'a ColumnFamily {
            db.cf_handle(name)
                .unwrap_or_else(|| panic!("column family '{name}' must exist"))
        };
        Self {
            state: resolve(STATE_CF),
            state_history: resolve(STATE_HISTORY_CF),
            stale_state_history: resolve(STALE_STATE_HISTORY_CF),
            blocks: resolve(BLOCKS_CF),
            transactions: resolve(TRANSACTIONS_CF),
            certificates: resolve(CERTIFICATES_CF),
            jmt_nodes: resolve(JMT_NODES_CF),
            stale_jmt_nodes: resolve(STALE_JMT_NODES_CF),
            consensus_receipts: resolve(CONSENSUS_RECEIPTS_CF),
            execution_metadata: resolve(EXECUTION_METADATA_CF),
            execution_certs: resolve(EXECUTION_CERTS_CF),
            tx_cert_index: resolve(TX_CERT_INDEX_CF),
            beacon_witnesses: resolve(BEACON_WITNESSES_CF),
            substate_bytes: resolve(SUBSTATE_BYTES_CF),
            safe_vote_registers: resolve(SAFE_VOTE_REGISTERS_CF),
            import_staging: resolve(IMPORT_STAGING_CF),
        }
    }
}

// ─── Typed CF definitions ────────────────────────────────────────────────────

// Block / Transaction storage

pub struct BlocksCf;
impl TypedCf for BlocksCf {
    const NAME: &'static str = BLOCKS_CF;
    type Key = u64; // block height
    type Value = BlockMetadata;
    type KeyCodec = BeU64Codec;
    type ValueCodec = HborCodec<BlockMetadata>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.blocks
    }
}

pub struct TransactionsCf;
impl TypedCf for TransactionsCf {
    const NAME: &'static str = TRANSACTIONS_CF;
    type Key = Hash;
    type Value = Transaction;
    type KeyCodec = HashCodec;
    type ValueCodec = HborCodec<Transaction>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.transactions
    }
}

pub struct CertificatesCf;
impl TypedCf for CertificatesCf {
    const NAME: &'static str = CERTIFICATES_CF;
    type Key = FinalizationHash;
    type Value = Finalization;
    type KeyCodec = HborCodec<FinalizationHash>;
    type ValueCodec = HborCodec<Finalization>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.certificates
    }
}

// JMT

pub struct JmtNodesCf;
impl TypedCf for JmtNodesCf {
    const NAME: &'static str = JMT_NODES_CF;
    type Key = StoredNodeKey;
    type Value = VersionedStoredNode;
    type KeyCodec = JmtKeyCodec;
    type ValueCodec = HborCodec<VersionedStoredNode>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.jmt_nodes
    }
}

pub struct StaleJmtNodesCf;
impl TypedCf for StaleJmtNodesCf {
    const NAME: &'static str = STALE_JMT_NODES_CF;
    type Key = u64; // version at which nodes became stale
    type Value = Vec<StaleTreePart>;
    type KeyCodec = BeU64Codec;
    type ValueCodec = HborCodec<Vec<StaleTreePart>>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.stale_jmt_nodes
    }
}

/// Committed substate byte total per version; see [`SUBSTATE_BYTES_CF`].
pub struct SubstateBytesCf;
impl TypedCf for SubstateBytesCf {
    const NAME: &'static str = SUBSTATE_BYTES_CF;
    type Key = u64; // version
    type Value = u64; // byte total after this version's commit
    type KeyCodec = BeU64Codec;
    type ValueCodec = HborCodec<u64>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.substate_bytes
    }
}

/// Staged snap-sync leaves awaiting finalize; see [`IMPORT_STAGING_CF`].
pub struct ImportStagingCf;
impl TypedCf for ImportStagingCf {
    const NAME: &'static str = IMPORT_STAGING_CF;
    type Key = SubstateKey; // the leaf key by identity
    type Value = Vec<u8>; // raw substate value
    type KeyCodec = SubstateKeyCodec;
    type ValueCodec = RawCodec;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.import_staging
    }
}

/// Version-indexed list of `state_history` keys written at each version.
/// Enables incremental GC of `state_history` — GC walks this CF in version
/// order, deletes the listed history keys for each version ≤ cutoff, and
/// drops the stale-set entry itself. No full `state_history` scan.
pub struct StaleStateHistoryCf;
impl TypedCf for StaleStateHistoryCf {
    const NAME: &'static str = STALE_STATE_HISTORY_CF;
    type Key = u64; // write_version
    type Value = Vec<Vec<u8>>; // raw `state_history` keys (storage_key ++ BE8(version))
    type KeyCodec = BeU64Codec;
    type ValueCodec = HborCodec<Vec<Vec<u8>>>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.stale_state_history
    }
}

/// State — current-value-per-key source of truth.
///
/// Key: the substate key's 32 bytes. Value: opaque substate bytes. An
/// absent row means "no value for this key" — deletions do
/// `batch.delete_cf(state_cf, K)`, not a tombstone sentinel.
///
/// Current reads are direct point lookups. Historical reads at version V
/// go through the companion `StateHistoryCf`: seek the smallest history
/// entry for K with `write_version > V` and return its stored prior value.
///
/// No prefix extractor: the dominant op is `get_cf(K)` (point reads plus
/// the commit path's `capture_history` `multi_get`), gated by whole-key
/// bloom (rocksdb default). A prefix extractor would add a second bloom
/// per SST, doubling filter-cache footprint and evicting data blocks
/// without improving point-read latency. `list_at_prefix` still works
/// without a prefix extractor — it just can't short-circuit SSTs via
/// prefix bloom.
pub struct StateCf;
impl TypedCf for StateCf {
    const NAME: &'static str = STATE_CF;
    type Key = SubstateKey;
    type Value = Vec<u8>;
    type KeyCodec = SubstateKeyCodec;
    type ValueCodec = RawCodec;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.state
    }
}

/// State-history log — per-write prior-value entries for historical reads.
///
/// Key: `(substate_key, write_version)` encoded as
/// `substate_key_bytes ++ write_version_BE_8B`. Value:
/// `Option<Vec<u8>>` — the value the key held immediately before the
/// write at `write_version`. `None` means "key was absent before the
/// write."
///
/// Every write to `StateCf` at version V captures a history entry at
/// `(K, V)` (except during genesis / bootstrap, which skips history
/// writes). GC deletes entries older than the retention window; `StateCf`
/// is always authoritative for the current tip.
///
/// Read-only: historical reads reconstruct the value-at-V by seeking the
/// smallest entry for K with `v' > V`. Nothing ever mutates `StateCf`
/// from this log.
pub struct StateHistoryCf;
impl TypedCf for StateHistoryCf {
    const NAME: &'static str = STATE_HISTORY_CF;
    type Key = (SubstateKey, u64); // (substate key, write_version)
    type Value = Option<Vec<u8>>;
    type KeyCodec = VersionedSubstateKeyCodec;
    type ValueCodec = HborCodec<Option<Vec<u8>>>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.state_history
    }
}

// Receipts

pub struct ConsensusReceiptsCf;
impl TypedCf for ConsensusReceiptsCf {
    const NAME: &'static str = CONSENSUS_RECEIPTS_CF;
    type Key = Hash;
    type Value = ConsensusReceipt;
    type KeyCodec = HashCodec;
    type ValueCodec = HborCodec<ConsensusReceipt>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.consensus_receipts
    }
}

pub struct ExecutionMetadataCf;
impl TypedCf for ExecutionMetadataCf {
    const NAME: &'static str = EXECUTION_METADATA_CF;
    type Key = Hash;
    type Value = ExecutionMetadata;
    type KeyCodec = HashCodec;
    type ValueCodec = HborCodec<ExecutionMetadata>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.execution_metadata
    }
}

// Execution Certificates

pub struct ExecutionCertsCf;
impl TypedCf for ExecutionCertsCf {
    const NAME: &'static str = EXECUTION_CERTS_CF;
    type Key = TickId;
    type Value = ExecutionCertificate;
    type KeyCodec = HborCodec<TickId>;
    type ValueCodec = HborCodec<ExecutionCertificate>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.execution_certs
    }
}

pub struct TxCertIndexCf;
impl TypedCf for TxCertIndexCf {
    const NAME: &'static str = TX_CERT_INDEX_CF;
    type Key = Hash;
    type Value = TickId;
    type KeyCodec = HashCodec;
    type ValueCodec = HborCodec<TickId>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.tx_cert_index
    }
}

// Beacon witnesses.

/// Key codec for the [`BeaconWitnessesCf`] CF: a `u64` leaf index
/// encoded big-endian. BE preserves lexicographic order so a full scan
/// returns leaves in monotonic index order. The shard is implicit —
/// storage is scoped per-shard.
#[derive(Default)]
pub struct BeaconWitnessKeyCodec;

impl DbEncode<u64> for BeaconWitnessKeyCodec {
    fn encode_to(&self, value: &u64, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&value.to_be_bytes());
    }
}

impl DbCodec<u64> for BeaconWitnessKeyCodec {
    fn decode(&self, bytes: &[u8]) -> u64 {
        assert_eq!(bytes.len(), 8, "beacon-witness key must be 8 bytes");
        u64::from_be_bytes(bytes.try_into().expect("length checked above"))
    }
}

pub struct BeaconWitnessesCf;
impl TypedCf for BeaconWitnessesCf {
    const NAME: &'static str = BEACON_WITNESSES_CF;
    type Key = u64;
    type Value = ShardWitnessPayload;
    type KeyCodec = BeaconWitnessKeyCodec;
    type ValueCodec = HborCodec<ShardWitnessPayload>;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.beacon_witnesses
    }
}

// Safe-vote registers.

/// Key codec for [`SafeVoteRegistersCf`]: validator id as a big-endian
/// `u64`, so recovery's full scan decodes in stable validator order.
#[derive(Default)]
pub struct ValidatorIdCodec;

impl DbEncode<ValidatorId> for ValidatorIdCodec {
    fn encode_to(&self, value: &ValidatorId, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&value.inner().to_be_bytes());
    }
}

impl DbCodec<ValidatorId> for ValidatorIdCodec {
    fn decode(&self, bytes: &[u8]) -> ValidatorId {
        let arr: [u8; 8] = bytes.try_into().expect("validator key must be 8 bytes");
        ValidatorId::new(u64::from_be_bytes(arr))
    }
}

/// Value codec for [`SafeVoteRegistersCf`]: packed 32-byte record
/// `[chain_origin_16B][locked_round_BE_8B][last_voted_round_BE_8B]`.
#[derive(Default)]
pub struct SafeVoteRegisterRecordCodec;

impl DbEncode<(ChainOrigin, SafeVoteRegisters)> for SafeVoteRegisterRecordCodec {
    fn encode_to(&self, value: &(ChainOrigin, SafeVoteRegisters), buf: &mut Vec<u8>) {
        ChainOriginCodec.encode_to(&value.0, buf);
        buf.extend_from_slice(&value.1.locked_round.inner().to_be_bytes());
        buf.extend_from_slice(&value.1.last_voted_round.inner().to_be_bytes());
    }
}

impl DbCodec<(ChainOrigin, SafeVoteRegisters)> for SafeVoteRegisterRecordCodec {
    fn decode(&self, bytes: &[u8]) -> (ChainOrigin, SafeVoteRegisters) {
        assert_eq!(
            bytes.len(),
            32,
            "safe-vote register record must be 32 bytes"
        );
        let origin = ChainOriginCodec.decode(&bytes[..16]);
        let registers = SafeVoteRegisters {
            locked_round: Round::new(u64::from_be_bytes(
                bytes[16..24].try_into().expect("length checked above"),
            )),
            last_voted_round: Round::new(u64::from_be_bytes(
                bytes[24..32].try_into().expect("length checked above"),
            )),
        };
        (origin, registers)
    }
}

/// Durable safe-vote registers per validator; see [`SAFE_VOTE_REGISTERS_CF`].
pub struct SafeVoteRegistersCf;
impl TypedCf for SafeVoteRegistersCf {
    const NAME: &'static str = SAFE_VOTE_REGISTERS_CF;
    type Key = ValidatorId;
    type Value = (ChainOrigin, SafeVoteRegisters);
    type KeyCodec = ValidatorIdCodec;
    type ValueCodec = SafeVoteRegisterRecordCodec;
    type Handles<'a> = CfHandles<'a>;
    fn handle<'a>(cf: &Self::Handles<'a>) -> &'a ColumnFamily {
        cf.safe_vote_registers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beacon_witness_key_codec_round_trip() {
        let codec = BeaconWitnessKeyCodec;
        for leaf in [0u64, 42, u64::MAX] {
            let mut buf = Vec::new();
            codec.encode_to(&leaf, &mut buf);
            assert_eq!(buf.len(), 8);
            assert_eq!(codec.decode(&buf), leaf);
        }
    }

    /// BE encoding so sorting encoded keys lexicographically matches
    /// ascending leaf-index order — the responder's prefix scan relies
    /// on this for monotonic iteration.
    #[test]
    fn beacon_witness_key_codec_preserves_monotonic_order() {
        let codec = BeaconWitnessKeyCodec;
        let mut encoded: Vec<Vec<u8>> = [10u64, 0, 5, 1, 256]
            .iter()
            .map(|leaf| {
                let mut buf = Vec::new();
                codec.encode_to(leaf, &mut buf);
                buf
            })
            .collect();
        encoded.sort();
        let decoded: Vec<u64> = encoded.iter().map(|b| codec.decode(b)).collect();
        assert_eq!(decoded, vec![0, 1, 5, 10, 256]);
    }
}
