//! Shared helpers for the libp2p transport test binaries.
//!
//! The `network` and `validator_bind` suites each compile their own copy of
//! this module and use a different subset, so a helper unused in one binary
//! isn't dead code.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use hyperscale_crypto_bls::{BlsSigner, generate_bls_keypair};
use hyperscale_network::ValidatorKeyMap;
use hyperscale_types::{Signer, ValidatorId};

/// Budget for a transport connection / validator-bind handshake to complete
/// over localhost QUIC.
pub const CONNECTION_TIMEOUT: Duration = Duration::from_secs(5);

/// Create a dummy bind signing key + validator key map for tests that create
/// adapters directly. Returns the consensus signer (consumed by the
/// validator-bind service to produce per-session signatures) plus the keymap
/// that will verify signatures from this validator.
pub fn test_bind_args(validator_id: ValidatorId) -> (Arc<dyn Signer>, Arc<ValidatorKeyMap>) {
    let bls_key = BlsSigner::new(generate_bls_keypair());
    let pubkey = bls_key.public_key();
    let mut keys = ValidatorKeyMap::new();
    keys.insert(validator_id, pubkey);
    (Arc::new(bls_key), Arc::new(keys))
}
