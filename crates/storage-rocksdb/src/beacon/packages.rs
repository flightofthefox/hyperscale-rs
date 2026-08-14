//! Fetched-package persistence — `FetchedPackageStore` for
//! [`RocksDbBeaconStorage`].

use hyperscale_storage::FetchedPackageStore;
use hyperscale_types::Hash;
use rocksdb::{WriteBatch, WriteOptions};

use super::column_families::FetchedPackagesCf;
use super::core::RocksDbBeaconStorage;
use crate::typed_cf::{TypedCf, batch_put, iter_all};

impl FetchedPackageStore for RocksDbBeaconStorage {
    fn store_fetched_package(&self, package: Hash, artifact: &[u8]) {
        let cf = self.cf();
        let mut batch = WriteBatch::default();
        batch_put::<FetchedPackagesCf>(
            &mut batch,
            FetchedPackagesCf::handle(&cf),
            &package,
            &artifact.to_vec(),
        );
        // Content-addressed and reconstructible by refetch, so an
        // unfsynced write risks nothing a restart cannot recover.
        let opts = WriteOptions::default();
        self.db
            .write_opt(batch, &opts)
            .expect("fetched-package write failed");
    }

    fn fetched_packages(&self) -> Vec<Vec<u8>> {
        let cf = self.cf();
        iter_all::<FetchedPackagesCf>(&self.db, FetchedPackagesCf::handle(&cf))
            .map(|(_, artifact)| artifact)
            .collect()
    }
}
