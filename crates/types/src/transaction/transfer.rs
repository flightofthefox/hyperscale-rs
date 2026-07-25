//! Building the XRD transfer every client sends.
//!
//! The manifest shape — lock a fee on the payer, withdraw, deposit the whole
//! worktop or abort — is the same whether a load generator, a behavioral
//! scenario, or the browser demo is sending it. Only the surrounding policy
//! differs (where the nonce comes from, whether a failure aborts the run or
//! is logged and skipped), so this returns a `Result` and leaves that to the
//! caller.

use radix_common::constants::XRD;
use radix_common::crypto::Ed25519PrivateKey;
use radix_common::math::Decimal;
use radix_common::network::NetworkDefinition;
use radix_common::types::ComponentAddress;
use radix_transactions::builder::ManifestBuilder;

use super::notarize::sign_and_notarize;
use crate::transaction::constructors::routable_from_notarized_v1;
use crate::{RoutableTransaction, TimestampRange, TransactionError};

/// The fee the payer locks. Comfortably covers a transfer at current costing;
/// the surplus refunds, so overshooting is free and underestimating aborts.
const TRANSFER_FEE: u32 = 10;

/// Build a signed, notarized XRD transfer from `from` to `to`, routable and
/// valid across `validity`.
///
/// `payer` must control `from`: it both signs the withdrawal and notarizes.
///
/// # Errors
///
/// Returns [`TransactionError`] if signing or notarization fails, which in
/// practice means a malformed manifest.
pub fn build_transfer_tx(
    payer: &Ed25519PrivateKey,
    from: ComponentAddress,
    to: ComponentAddress,
    amount: Decimal,
    network: &NetworkDefinition,
    nonce: u32,
    validity: TimestampRange,
) -> Result<RoutableTransaction, TransactionError> {
    let manifest = ManifestBuilder::new()
        .lock_fee(from, Decimal::from(TRANSFER_FEE))
        .withdraw_from_account(from, XRD, amount)
        .try_deposit_entire_worktop_or_abort(to, None)
        .build();
    let notarized = sign_and_notarize(manifest, network, nonce, payer)?;
    routable_from_notarized_v1(notarized, validity)
}
