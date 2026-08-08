//! [`WaveCertificate`] — proof of execution finalization carrying every
//! participating shard's [`ExecutionCertificate`].

use std::sync::Arc;

use blake3::Hasher;
use hyperscale_hbor::{Hbor, to_vec as hbor_to_vec};

use crate::{ExecutionCertificate, Hash, TickId, Verifiable, Verified, WaveReceiptHash};

/// Cap on execution certificates accepted in a single `WaveCertificate` at
/// decode time.
///
/// A wave's EC set is one local EC plus at most one EC per participating
/// remote shard (and may include a few extras if a remote shard committed
/// the wave's transactions across multiple blocks). 1024 is well above any
/// realistic shard count and bounds the per-element pre-allocation that
/// would otherwise let a peer claim billions of inner ECs and OOM the
/// validator at decode time.
pub const MAX_EXECUTION_CERTIFICATES_PER_WAVE: usize = 1024;

/// Hash a wave certificate's execution-certificate identities into its
/// [`WaveReceiptHash`] — the leaf a block's `certificate_root` commits.
///
/// Each EC contributes its `(shard_id, tick_id)` pair; the shard is the
/// wave's own (`TickId::shard_id`), so a verifier holding only the
/// certificate's EC wave-ids reproduces the hash without the EC bodies.
/// Order matters and is the certificate's stored order (sorted by
/// `(shard_id, tick_id)` at construction); callers reproducing the hash
/// feed the same order.
///
/// # Panics
///
/// Panics if HBOR encoding of a `ShardId` or `TickId` fails — closed
/// wire types, infallible in practice.
#[must_use]
pub fn wave_receipt_hash<'a>(ec_tick_ids: impl IntoIterator<Item = &'a TickId>) -> WaveReceiptHash {
    let mut hasher = Hasher::new();
    for tick_id in ec_tick_ids {
        hasher.update(&hbor_to_vec(&tick_id.shard_id()).unwrap());
        hasher.update(&hbor_to_vec(tick_id).unwrap());
    }
    WaveReceiptHash::from_raw(Hash::from_hash_bytes(hasher.finalize().as_bytes()))
}

/// Wave certificate — proof of execution finalization for a wave.
///
/// Contains the execution certificates from all participating shards.
/// Per-tx decisions (Accept/Reject/Aborted) are derived from the ECs.
/// Every wave resolves through the EC path — there is no all-abort fallback.
///
/// # Invariant (well-formed WC)
///
/// A well-formed `WaveCertificate` contains **exactly one local EC** — the
/// EC where `ec.tick_id() == wc.tick_id`. The local EC is the authoritative
/// source for the wave's tx set and canonical (block) ordering. Remote ECs
/// attest against their own wave decompositions and may cover only subsets;
/// the local shard, by construction, produces a single EC per wave.
///
/// Enforced at construction by `WaveCertificateTracker::create_wave_certificate`
/// and at the wire boundary by `WaveCertificate`'s decode impl.
/// Downstream helpers like [`FinalizedWave::local_ec`](crate::FinalizedWave::local_ec)
/// `expect` this invariant.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
#[hbor(validate = check_wave_certificate)]
pub struct WaveCertificate {
    tick_id: TickId,
    #[hbor(max = MAX_EXECUTION_CERTIFICATES_PER_WAVE)]
    execution_certificates: Vec<Arc<Verifiable<ExecutionCertificate>>>,
}

/// The exactly-one-local-EC invariant, enforced at the wire boundary. Zero
/// local ECs would crash `FinalizedWave::local_ec()`; multiple would let
/// downstream code silently disagree on which EC is authoritative for tx
/// ordering.
fn check_wave_certificate(wc: &WaveCertificate) -> Result<(), &'static str> {
    let local = wc
        .execution_certificates
        .iter()
        .filter(|ec| ec.tick_id() == &wc.tick_id)
        .count();
    if local == 1 {
        Ok(())
    } else {
        Err("a wave certificate carries exactly one local execution certificate")
    }
}

impl WaveCertificate {
    /// Build a `WaveCertificate` from its parts. Each EC lands
    /// [`Verifiable::Unverified`] — this is the constructor for wire
    /// reconstruction and tests, where the ECs carry no verification
    /// marker. Locally aggregated waves that already hold
    /// [`Verified<ExecutionCertificate>`]s use [`Self::from_verified_ecs`]
    /// to carry their markers through.
    ///
    /// Does not validate the exactly-one-local-EC invariant; that is
    /// enforced at the wire boundary by the `Decode` impl and at the
    /// build boundary by `WaveCertificateTracker::create_wave_certificate`.
    ///
    /// # Panics
    ///
    /// Panics if `execution_certificates.len() > MAX_EXECUTION_CERTIFICATES_PER_WAVE`.
    #[must_use]
    pub fn new(tick_id: TickId, execution_certificates: Vec<Arc<ExecutionCertificate>>) -> Self {
        Self {
            tick_id,
            execution_certificates: execution_certificates
                .into_iter()
                .map(|ec| Arc::new(Verifiable::from(Arc::unwrap_or_clone(ec))))
                .collect(),
        }
    }

    /// Build a `WaveCertificate` from execution certificates that have
    /// already cleared their per-EC signature predicate, carrying the
    /// [`Verifiable::Verified`] marker on each.
    ///
    /// Used by `WaveCertificateTracker::create_wave_certificate`, whose
    /// ECs were produced through the
    /// [`Verified::<ExecutionCertificate>::aggregate`] gate. Keeping the
    /// marker leaves the wave's ECs internally consistent with the
    /// [`Verified<FinalizedWave>`](crate::FinalizedWave) they're sealed
    /// into, so a later [`FinalizedWave::verify`](crate::FinalizedWave)
    /// short-circuits them instead of re-checking.
    ///
    /// Does not validate the exactly-one-local-EC invariant; see
    /// [`Self::new`].
    ///
    /// # Panics
    ///
    /// Panics if `execution_certificates.len() > MAX_EXECUTION_CERTIFICATES_PER_WAVE`.
    #[must_use]
    pub fn from_verified_ecs(
        tick_id: TickId,
        execution_certificates: Vec<Verified<ExecutionCertificate>>,
    ) -> Self {
        Self {
            tick_id,
            execution_certificates: execution_certificates
                .into_iter()
                .map(|ec| Arc::new(Verifiable::from(ec)))
                .collect(),
        }
    }

    /// Self-contained wave identifier (shard + height + remote dependencies).
    /// Globally unique. `hash(tick_id)` = identity key for manifest/storage.
    #[must_use]
    pub const fn tick_id(&self) -> &TickId {
        &self.tick_id
    }

    /// Execution certificates from all participating shards.
    /// Always includes the local EC (see invariant above).
    /// May contain multiple ECs from the same remote shard — this happens when
    /// a remote shard committed this wave's transactions across multiple blocks,
    /// producing separate ECs.
    /// Sorted by (`shard_id`, `tick_id`) for deterministic `receipt_hash`.
    ///
    /// Each EC rides as `Verifiable<ExecutionCertificate>`: wire-decoded
    /// certificates land [`Verifiable::Unverified`]; locally assembled
    /// ones carry the [`Verifiable::Verified`] marker from
    /// [`Self::from_verified_ecs`].
    #[must_use]
    pub fn execution_certificates(&self) -> &[Arc<Verifiable<ExecutionCertificate>>] {
        &self.execution_certificates
    }

    /// Compute the receipt hash for this wave certificate.
    ///
    /// Hashes sorted (`shard_id`, `tick_id`) pairs. The vec is
    /// pre-sorted at construction time for deterministic ordering. At most
    /// one valid EC exists per `tick_id` (signature verification upstream
    /// enforces this), so committing to `tick_id` is content-equivalent.
    #[must_use]
    pub fn receipt_hash(&self) -> WaveReceiptHash {
        wave_receipt_hash(self.execution_certificates.iter().map(|ec| ec.tick_id()))
    }

    /// The wave-ids of every execution certificate this certificate
    /// carries, in stored (`receipt_hash`) order. The minimal reveal a
    /// remote verifier needs to reproduce [`Self::receipt_hash`] — the
    /// shard of each is derivable as `TickId::shard_id`.
    #[must_use]
    pub fn ec_tick_ids(&self) -> Vec<TickId> {
        self.execution_certificates
            .iter()
            .map(|ec| *ec.tick_id())
            .collect()
    }
}
