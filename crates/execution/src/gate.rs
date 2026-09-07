//! The one gate a fetched certificate passes on its way to the crypto
//! pool, whether it arrives alone or inside a finalization.
//!
//! Every contained certificate is held to the same four checks: the
//! halt-recovery freeze, the committee its anchor resolves, that
//! committee's quorum power, and its keys. What differs between the two
//! arrivals is what the slot is keyed by and what a refusal releases,
//! which is what [`Attested`] says.

use std::sync::Arc;

use hyperscale_core::FetchIds;
use hyperscale_types::{
    ConsensusPublicKey, ExecutionCertificate, Finalization, Hash, ScheduleLookup, TickId,
    TopologySchedule, Verifiable,
};

use crate::lookups::{
    committee_public_keys_for_shard, ec_has_shard_quorum_power, fetch_keys_covered,
};

/// A fetched artifact carrying certificates to verify: one, or a
/// finalization's several.
pub trait Attested {
    /// The content hash an in-flight verification is keyed by, so a
    /// byte-identical retransmit does not dispatch twice. Different
    /// aggregations of one logical certificate hash differently and
    /// each dispatch — a first with a bad signature may be followed by
    /// a valid one.
    fn slot(&self) -> Hash;

    /// The tick the artifact is about, for the log.
    fn tick_id(&self) -> &TickId;

    /// Every certificate the artifact carries, in the order the keys are
    /// returned in.
    fn certificates(&self) -> impl Iterator<Item = &ExecutionCertificate>;

    /// What refusing the artifact releases: it answers for nothing it
    /// claimed, so each claim goes back to being fetchable.
    fn abandon(&self) -> FetchIds;
}

impl Attested for Verifiable<ExecutionCertificate> {
    fn slot(&self) -> Hash {
        self.wire_hash()
    }

    fn tick_id(&self) -> &TickId {
        ExecutionCertificate::tick_id(self)
    }

    fn certificates(&self) -> impl Iterator<Item = &ExecutionCertificate> {
        std::iter::once(self.as_unverified())
    }

    fn abandon(&self) -> FetchIds {
        FetchIds::ExecutionCerts(fetch_keys_covered(self))
    }
}

impl Attested for Arc<Verifiable<Finalization>> {
    /// A tick can settle in more than one part, so identity is the
    /// finalization's own content.
    fn slot(&self) -> Hash {
        self.receipt_hash().into_raw()
    }

    fn tick_id(&self) -> &TickId {
        Finalization::tick_id(self)
    }

    fn certificates(&self) -> impl Iterator<Item = &ExecutionCertificate> {
        self.execution_certificates()
            .iter()
            .map(|ec| ec.as_unverified())
    }

    fn abandon(&self) -> FetchIds {
        FetchIds::Finalizations(vec![self.receipt_hash()])
    }
}

/// What the gate answers for one certificate.
pub enum Gated {
    /// Verify it against these keys.
    Keys(Vec<ConsensusPublicKey>),
    /// This node's beacon has not reached the certificate's committee
    /// epoch, so the signing committee cannot be resolved yet. Pure
    /// catch-up: under lookahead the committee is already globally
    /// fixed, so the certificate is parked for replay rather than
    /// refused.
    BeaconBehind,
    /// Refused for good, with why.
    Refused(&'static str),
}

/// Hold one certificate to the checks every arrival passes before its
/// signature is worth verifying.
///
/// The halt-recovery freeze first: a certificate from a recovering
/// shard above the beacon-attested frontier is one the retained
/// beyond-f committee could only have produced after the halt. It
/// resolves the old committee at its stale anchor and its signatures
/// verify, so without this fence a forged finalization would export
/// cross-shard. Then the committee seated at its own anchor on its own
/// shard: below the schedule floor it is past its retention horizon,
/// provably terminal everywhere and never resolvable again. Then that
/// committee's quorum power — a single Byzantine signer produces a
/// cryptographically valid certificate — and its keys.
#[must_use]
pub fn gate_certificate(schedule: &TopologySchedule, ec: &ExecutionCertificate) -> Gated {
    let shard = ec.shard_id();
    if schedule.recovery_fences(shard, ec.block_height()) {
        return Gated::Refused("from a recovering shard past the freeze frontier");
    }
    let committee = match schedule.lookup(ec.vote_anchor_ts()) {
        ScheduleLookup::Committee(committee) => committee,
        ScheduleLookup::NotYetCommitted => return Gated::BeaconBehind,
        ScheduleLookup::Evicted => {
            return Gated::Refused("committee epoch below the schedule floor");
        }
    };
    if !ec_has_shard_quorum_power(committee, ec) {
        return Gated::Refused("lacks quorum power on its shard");
    }
    committee_public_keys_for_shard(committee, shard).map_or(
        Gated::Refused("committee keys unresolvable — snapshot incomplete"),
        Gated::Keys,
    )
}
