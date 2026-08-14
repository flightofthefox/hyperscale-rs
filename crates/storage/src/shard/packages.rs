//! The committed-package artifact index a backend keeps beside its state.

use hyperscale_types::Hash;

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
