//! The publish gate, as the chain reaches it.
//!
//! The verdict itself is [`hyperscale_vm_gate`]'s: every clause is a pure
//! function of the artifact's bytes, so it is the same verdict wherever it
//! is reached, and it lives beside the vocabulary it judges rather than
//! beside one of its callers. What this module adds is the chain's error
//! type — admission carries [`VmStaticsError`] through a long path that
//! has nothing to do with packages — and nothing else.
//!
//! Keeping the seam this thin is what makes `cargo hyperscale` honest: a
//! package that builds locally has passed the call admission runs, not a
//! reimplementation of it that agrees until it does not.

use hyperscale_types::VmStaticsError;
pub use hyperscale_vm_effects::METADATA_SECTION;
use hyperscale_vm_effects::PackageMetadata;
use hyperscale_vm_gate::{
    GateError, admit_package as admit, admit_protocol_package as admit_protocol,
    attach_metadata as attach, extract_metadata as extract,
};

fn chain(error: GateError) -> VmStaticsError {
    VmStaticsError(error.0)
}

/// Attach `metadata` to a component artifact as its metadata section.
///
/// The result is the publishable artifact: same code, one section longer,
/// and a different content address.
///
/// # Errors
///
/// [`VmStaticsError`] if the artifact's section framing is malformed, if
/// it already declares a metadata section, or if the metadata is past a
/// bound the codec enforces — the chain's byte budget included, judged
/// here so nothing assembles an artifact admission would refuse on size.
pub fn attach_metadata(
    artifact: &[u8],
    metadata: &PackageMetadata,
) -> Result<Vec<u8>, VmStaticsError> {
    attach(artifact, metadata).map_err(chain)
}

/// The effect metadata a component artifact declares, if it declares any.
///
/// # Errors
///
/// [`VmStaticsError`] if the artifact's section framing is malformed, if
/// it declares the metadata section more than once, or if the section's
/// payload is oversized or not canonical metadata.
pub fn extract_metadata(artifact: &[u8]) -> Result<Option<PackageMetadata>, VmStaticsError> {
    extract(artifact).map_err(chain)
}

/// The metadata a publish admits from an artifact, or why it does not.
///
/// See [`hyperscale_vm_gate::admit_package`] for what is checked and why
/// the whole verdict is reachable from the bytes alone.
///
/// # Errors
///
/// [`VmStaticsError`] on an artifact outside the profile, an absent or
/// non-canonical metadata section, a declared method the component does
/// not export, an ABI binding the export's type cannot honour, or a
/// claim to totality, which only [`admit_protocol_package`] grants.
pub fn admit_package(artifact: &[u8]) -> Result<PackageMetadata, VmStaticsError> {
    admit(artifact).map_err(chain)
}

/// Admit an artifact the protocol supplies rather than a publisher.
///
/// Identical to [`admit_package`] but for the totality mark, which a
/// publisher cannot claim and which this one reads against the code
/// rather than takes on faith. Genesis seeds the stdlib through here;
/// nothing reachable from a transaction does, so the distinction is a
/// fact about the caller rather than about the bytes.
///
/// # Errors
///
/// As [`admit_package`], except that a claim to totality is checked
/// against the artifact instead of refused, and fails admission when the
/// code does not support it.
pub fn admit_protocol_package(artifact: &[u8]) -> Result<PackageMetadata, VmStaticsError> {
    admit_protocol(artifact).map_err(chain)
}

#[cfg(test)]
mod tests {
    use hyperscale_vm_stdlib::account_artifact;

    use super::{admit_package, admit_protocol_package};

    /// The seam, not the verdict: the gate's own clauses are tested where
    /// they live, and what this pins is that the chain reaches them and
    /// carries their sentence through its own error type.
    #[test]
    fn the_chain_reaches_the_gates_verdict() {
        let artifact = account_artifact();
        assert!(admit_protocol_package(artifact).is_ok());

        let refused = admit_package(artifact).expect_err("a publish cannot claim totality");
        assert!(refused.0.contains("claims totality"), "{}", refused.0);
    }
}
