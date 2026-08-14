//! The committed-package artifact index a backend keeps beside its state.

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
}
