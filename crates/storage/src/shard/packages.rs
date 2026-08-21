//! The committed-package artifact index a backend keeps beside its state.

use hyperscale_types::{Hash, SubstateKey, protocol_statics, protocol_statics_installed};

/// The content address a committed cell publishes, or `None` for a cell
/// that publishes nothing.
///
/// The judgement is the VM's: a package cell is self-identifying, its
/// key being the content address of the very bytes it holds, so no tag
/// and no trust in the writer enters into it. Every backend derives the
/// index through here — the commit batch and the import that rebuilds a
/// store from leaves alike — because an index built one way at commit
/// and another way at import is an index a turned-over committee cannot
/// serve from.
///
/// Without the protocol answers installed (bare storage tests) nothing
/// is a package, matching the cache-absorption seam.
#[must_use]
pub fn package_of_cell(key: SubstateKey, value: &[u8]) -> Option<Hash> {
    if !protocol_statics_installed() {
        return None;
    }
    protocol_statics().package_cell(key.owner.to_bytes(), key.local.0, value)
}

/// Read access to the package artifacts this store's committed state
/// publishes, by content address.
///
/// Derived state — the package cells are the authority — kept so a
/// restarting node re-learns published code without scanning cells it
/// cannot name. The default is the empty index, for stores that never
/// commit a publish (test doubles, ephemeral views).
pub trait PackageArtifactStore {
    /// Every indexed artifact's bytes.
    fn package_artifacts(&self) -> Vec<Vec<u8>> {
        Vec::new()
    }

    /// One indexed artifact's bytes, by content address.
    ///
    /// A point lookup, because the serve path runs on the blocking pool
    /// where a scan would pin a thread per request.
    fn package_artifact(&self, package: Hash) -> Option<Vec<u8>> {
        let _ = package;
        None
    }
}
