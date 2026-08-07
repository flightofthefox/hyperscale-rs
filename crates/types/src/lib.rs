//! Core types for Hyperscale consensus.
//!
//! This crate provides the foundational types used throughout the consensus
//! implementation:
//!
//! - **Primitives**: hashes, merkle roots, signer bitfields, randomness
//! - **Identifiers**: `ValidatorId`, `ShardId`, `BlockHeight`, etc.
//! - **Consensus types**: Block, `BlockHeader`, `QuorumCertificate`, etc.
//! - **Wave types**: `WaveId`, `ExecutionVote`, `ExecutionCertificate`, `WaveCertificate`, etc.
//! - **Network traits**: Message markers for serialization
//!
//! # Design Philosophy
//!
//! The foundation layer: every wire struct and the [`Verify`] predicate that
//! admits it live here. Signature checks route through the injected
//! [`Verifier`], so no scheme is named anywhere in the crate — its only
//! workspace dependencies are that crypto interface and the JMT whose
//! inclusion proofs the provisioning types verify.

mod crypto;
pub mod network;
mod primitives;
mod provisioning;
mod signing;
pub mod state_key;
mod time;
mod verifiable;

// Consensus types
mod beacon;
mod receipt;
mod shard;
mod topology;
mod transaction;
mod wave;

pub use beacon::{
    BEACON_SIGNER_COUNT, BeaconBlock, BeaconCert, BeaconChainConfig, BeaconGenesisConfig,
    BeaconProposal, BeaconProposalEquivocationMismatch, BeaconProposalVerifyContext,
    BeaconProposalVerifyError, BeaconState, BeaconWitnessEvent, CandidateBeaconBlock,
    CandidateBeaconBlockVerifyError, CandidateVerifyContext, CertifiedBeaconBlock,
    CertifiedBeaconBlockPairingError, CertifiedBeaconBlockVerifyContext,
    CertifiedBeaconBlockVerifyError, CohortSeat, CommitteeTransition, CompletedRecovery,
    EMISSION_PARTICIPATION_WEIGHT, EMISSION_STORAGE_WEIGHT, EMISSION_WORK_WEIGHT,
    EMISSIONS_PER_EPOCH, EPOCHS_PER_YEAR, GenesisPool, GenesisValidator, HALT_THRESHOLD_EPOCHS,
    IMPOUND_EPOCHS_DEFAULT, JAIL_COOLDOWN_EPOCHS, JailReason, KeeperSeat, KeptSeat,
    MAX_BEACON_COMMITTEE, MAX_BEACON_WITNESS_EVENTS_PER_TX, MAX_EQUIVOCATIONS_PER_PROPOSER,
    MAX_PREFIX_SIGS, MAX_RANGE_PROOF_NODES, MAX_READY_SIGNALS_PER_BLOCK, MAX_SHARDS,
    MAX_VOTE_VECTOR_LEN, MAX_WITNESS_PROOF_DEPTH, MAX_WITNESSES_PER_FETCH, MAX_WITNESSES_PER_SHARD,
    MIN_BEACON_COMMITTEE_SIZE, MIN_STAKE_FLOOR, MISSED_PROPOSAL_JAIL_THRESHOLD, NetworkParams,
    ObserverSeat, PC_VALUE_ELEMENT_BYTES, POOL_BUFFER_TARGET, PRODUCTION_BEACON_COMMITTEE_SIZE,
    ParamBoundsError, ParamProposal, ParamVote, PcCompactVote, PcDivergingProof, PcQc1,
    PcQc1VerifyError, PcQc2, PcQc2VerifyError, PcQc3, PcQc3VerifyError, PcSignerLengths,
    PcValueElement, PcVector, PcVote1, PcVote1VerifyError, PcVote2, PcVote2VerifyError, PcVote3,
    PcVote3VerifyError, PcVoteEquivocation, PcVoteEquivocationContext,
    PcVoteEquivocationVerifyError, PcVoteRound, PcVoteVerifyContext, PcXpProof, PendingReshape,
    PendingRotation, PendingWithdrawal, PoolConviction, RESHAPE_HANDOFF_TTL_EPOCHS,
    RESHAPE_READY_TTL_EPOCHS, RESHAPE_TRIGGER_TTL_EPOCHS, RatifyCert, RatifyCertVerifyError,
    RatifyPhase, RatifyVerifyContext, RatifyVote, RatifyVoteRecord, RatifyVoteVerifyError,
    ReadySignal, RecoveryCause, SHARD_CAPACITY, SHARD_WITNESS_LEAF_DOMAIN_TAG,
    SHUFFLE_SYNC_HEADROOM, SPC_INPUT_DWELL, SPC_VIEW_TIMEOUT, ScheduledSplit, ShardBoundary,
    ShardCommittee, ShardEpochContribution, ShardRecovery, ShardWitnessPayload, SkipReport,
    SlotEffects, SpcCert, SpcCertVerifyError, SpcEmptyViewMsg, SpcEmptyViewMsgVerifyError,
    SpcHighTriple, SpcHighTripleVerifyError, SpcNewCommitMsg, SpcNewCommitMsgVerifyError,
    SpcProposalObject, SpcProposalObjectVerifyError, SpcVerifyContext, StakePool,
    TOKENS_PER_YEAR_TARGET, TransitionCause, UNBONDING_WINDOW_EPOCHS, ValidatorRecord,
    ValidatorStatus, build_indirect_cert, build_qc1, build_qc2, build_qc3, build_ratify_cert,
    byzantine_threshold, genesis_config_hash, hash_high_value, mce, mcp, qc1_certify,
    ratify_quorum, ready_signal_window, sign_empty_view_msg, sign_ratify_vote, sign_vote1,
    sign_vote2, sign_vote3, skip_target, verify_block_cert, verify_block_equivocations,
    verify_cert, verify_certified, verify_empty_view_msg, verify_proposal_object, verify_qc1,
    verify_qc2, verify_qc3, verify_ratify_cert, verify_ratify_vote, verify_vote_equivocation,
    verify_vote1, verify_vote2, verify_vote3,
};
pub use crypto::Ed25519PrivateKey;
pub use crypto::keys::{ed25519_keypair_from_seed, generate_ed25519_keypair};
pub use hyperscale_crypto::{
    AggregateError, AggregateSignature, CONSENSUS_PUBLIC_KEY_BYTES, CONSENSUS_SIGNATURE_BYTES,
    ConsensusPublicKey, ConsensusSignature, SignError, Signer, VRF_PROOF_BYTES, Verifier,
    VrfOutput, VrfProof,
};
pub use hyperscale_hbor::HborSigned;
pub use hyperscale_vm_types::{Address, LocalKey, MAX_CELL_VALUE_LEN, StateWrites, SubstateKey};
pub use network::{
    GossipMessage, MessageClass, NetworkMessage, Request, Signed, SignedContext, SignedVerifyError,
    TopicScope,
};
pub use primitives::bloom::{BloomFilter, BloomKey, DEFAULT_FPR, MAX_BITS};
pub use primitives::hash::{Hash, TypedHash};
pub use primitives::hash_kinds::{
    BeaconBlockHash, BeaconWitnessRoot, BlockHash, CertificateRoot, EventRoot, GenesisConfigHash,
    GlobalReceiptHash, GlobalReceiptRoot, LocalReceiptRoot, ProvisionHash, ProvisionTxRoot,
    ProvisionsRoot, RevealChain, SettledWavesRoot, StateRoot, TransactionRoot, TxHash,
    WaveReceiptHash, WritesRoot,
};
pub use primitives::identifiers::{
    Attempt, BeaconWitnessLeafCount, BlockHeight, Epoch, HeaderFetchCount, InFlightCount,
    LeafIndex, RatifyRound, Round, ShardId, SpcView, Stake, StakePoolId, StakePoolSeat,
    ValidatorId, VoteCount,
};
pub use primitives::merkle::{
    compute_merkle_root, compute_merkle_root_with_proof, compute_range_proof,
    verify_merkle_inclusion, verify_range_inclusion,
};
pub use primitives::positional_bundle::PositionalBundle;
pub use primitives::randomness::{RANDOMNESS_BYTES, Randomness};
pub use primitives::signer_bitfield::SignerBitfield;
pub use provisioning::entry::ProvisionEntry;
pub use provisioning::limits::{MAX_MERKLE_PROOF_LEN, MAX_STATE_ENTRIES_PER_TX};
pub use provisioning::proof::MerkleInclusionProof;
pub use provisioning::provisions::{Provisions, ProvisionsContext, ProvisionsVerifyError};
pub use provisioning::substate::{SubstateEntry, SubstateLeaf};
pub use receipt::consensus::{ConsensusReceipt, FAILED_RECEIPT_HASH, absorb_committed_cells};
pub use receipt::event::{
    Event, EventExt, MAX_EVENT_PAYLOAD_BYTES, MAX_EVENT_TYPES, MAX_EVENTS_PER_TX,
};
pub use receipt::global::GlobalReceipt;
pub use receipt::metadata::{ExecutionMetadata, FeeSummary, LogLevel};
pub use receipt::stored::StoredReceipt;
pub use shard::certified::{CertifiedBlock, CertifiedBlockHashMismatch, LinkageError};
pub use shard::certified_header::{CertifiedBlockHeader, CertifiedHeaderVerifyError};
pub use shard::chain_origin::ChainOrigin;
pub use shard::commit_proof::{
    CommitProof, CommitProofVerifyError, MAX_COMMIT_PROOF_ANCESTRY, ResolvedCommittee,
};
pub use shard::evidence::{
    ShardForkProof, ShardForkProofVerifyError, ShardVoteEquivocation, ShardVoteEquivocationContext,
    ShardVoteEquivocationVerifyError, verify_shard_vote_equivocation,
};
pub use shard::fork_fence::ForkFence;
pub use shard::header::{BlockHeader, BlockHeaderParentQcMismatch, BlockHeaderVerifyError};
pub use shard::inventory::{ElidedCertifiedBlock, Inventory, RehydrateError, RehydrationMiss};
pub use shard::limits::{
    MAX_BLOCK_WORK, MAX_FINALIZED_TX_PER_BLOCK, MAX_PROVISIONS_PER_BLOCK, MAX_ROUND_GAP,
    MAX_TX_IN_FLIGHT, MAX_TXS_PER_BLOCK, TX_ADMISSION_WORK,
};
pub use shard::load::ShardLoad;
pub use shard::manifest::{BlockManifest, BlockMetadata};
pub use shard::quorum_certificate::{QcContext, QcVerifyError, QuorumCertificate};
pub use shard::reshape::{ReshapeThresholds, ReshapeTrigger};
pub use shard::roots::{
    BeaconWitnessRootContext, BeaconWitnessRootVerifyError, CertRootVerifyError,
    CertificateRootContext, LocalReceiptRootContext, LocalReceiptRootVerifyError,
    ProvisionRootVerifyError, ProvisionTxRootsContext, ProvisionTxRootsMap,
    ProvisionTxRootsVerifyError, ProvisionsRootContext, REVEAL_CHAIN_DOMAIN_TAG, SplitChildRoots,
    StateRootContext, StateRootVerifyError, TransactionRootContext, TxRootVerifyError,
    certificate_root_from_receipt_hashes, commit_witness_window, derive_leaves,
    derive_reshape_trigger, extend_reveal_chain, local_settled_wave_ids,
    missed_proposals_since_prev_commit, next_reveal_chain, ready_leaf_payload,
    settled_waves_root_from_ids,
};
pub use shard::storage_commit::{BeaconWitnessCommit, PreparedCommit, SyncHint};
pub use shard::timeout::{Timeout, TimeoutContext, TimeoutVerifyError};
pub use shard::vote::{BlockVote, BlockVoteContext, BlockVoteVerifyError};
pub use shard::vote_registers::SafeVoteRegisters;
pub use shard::{
    Block, SharedCertificates, SharedProvisions, SharedTransactions, SharedWitnessSources,
    TerminalRef, VerifiedBlockAssembleError, WitnessSources, shared_transactions_from_raw,
    work_over_certificates,
};
pub use signing::{
    BeaconRevealMessage, BlockProposalMessage, BlockVoteMessage, CertifiedBlockHeaderSenderMessage,
    ExecutionCertificatesSenderMessage, ExecutionVoteMessage, ExecutionVotesSenderMessage,
    NetworkId, PcRound, PcScope, PcVoteMessage, ProvisionsSenderMessage, RatifyVoteMessage,
    ShardRevealMessage, SpcEmptyViewMessage, SpcRelayKind, SpcRelayMessage,
    VALIDATOR_BIND_NONCE_LEN, ValidatorAddressMessage, ValidatorBindMessage,
    ValidatorPossessionProofMessage, beacon_reveal_sign, beacon_reveal_verify, shard_reveal_sign,
    shard_reveal_verify, signed_bytes, validator_possession_proof_sign,
    validator_possession_proof_verify, vrf_output_from_proof,
};
pub use time::epoch_windows::EpochWindows;
pub use time::limits::{MAX_TIMESTAMP_DELAY, MAX_TIMESTAMP_RUSH};
pub use time::range::{MAX_VALIDITY_RANGE, TimestampRange};
pub use time::stopwatch::Stopwatch;
pub use time::timeouts::{
    EPOCH_DURATION, MAX_PROGRESS_WAIT, QUIESCE_MARGIN, RATIFY_ROUND_TIMEOUT,
    REMOTE_HEADER_RETENTION, RETENTION_HORIZON, SKIP_TIMEOUT, VIEW_CHANGE_TIMEOUT,
    VIEW_CHANGE_TIMEOUT_INCREMENT, VIEW_CHANGE_TIMEOUT_MAX, WAVE_TIMEOUT,
};
pub use time::timestamp::{LocalTimestamp, ProposerTimestamp, WeightedTimestamp};
pub use topology::awaiting::AwaitingTopologyBuffer;
pub use topology::genesis::GenesisValidators;
pub use topology::network::{NetworkDefinition, UnknownNetwork};
pub use topology::schedule::{
    QuiesceCut, RoutingCommittees, ScheduleLookup, SplitAtBoundary, TopologySchedule,
};
pub use topology::settled_set::{SettledSetVerdict, SettledWaveSet, settled_set_verdict};
pub use topology::shard_prefix::shard_prefix_path;
pub use topology::snapshot::{ReshapeSeat, ShardAnchor, TopologySnapshot};
pub use topology::trie::ShardTrie;
pub use topology::validator::{ValidatorInfo, ValidatorSet};
pub use transaction::declared_key::DeclaredKey;
pub use transaction::limits::MAX_TX_BYTES_LEN;
pub use transaction::status::{
    TransactionDecision, TransactionError, TransactionStatus, TransactionStatusParseError,
};
pub use transaction::vm::{
    Derived, EnvelopeExt, MAX_MESSAGE_LEN, MAX_SUBINTENTS, Routing, SubintentSig, TransactionBody,
    TransactionEnvelope, VmStatics, VmStaticsError, install_vm_statics, vm_statics_installed,
};
pub use transaction::wire::{Transaction, TransactionVerifyError};
pub use verifiable::{Verifiable, Verified, Verify};
pub use wave::certificate::{
    MAX_EXECUTION_CERTIFICATES_PER_WAVE, WaveCertificate, wave_receipt_hash,
};
pub use wave::computation::{compute_waves, wave_leader, wave_leader_at};
pub use wave::execution_certificate::{
    ExecutionCertificate, ExecutionCertificateContext, ExecutionCertificateVerifyError,
};
pub use wave::finalized::{
    FinalizedWave, FinalizedWaveContext, FinalizedWaveVerifyError, ReceiptValidationError,
};
pub use wave::id::{MAX_REMOTE_SHARDS_PER_WAVE, WaveId};
pub use wave::outcome::{ExecutionOutcome, TxOutcome};
pub use wave::receipt_tree::{
    compute_global_receipt_root, compute_global_receipt_root_with_proof, tx_outcome_leaf,
};
pub use wave::vote::{ExecutionVote, ExecutionVoteContext, ExecutionVoteVerifyError};

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
