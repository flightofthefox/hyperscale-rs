//! Package artifact fetch response (cross-shard code availability).

use hyperscale_hbor::Hbor;

use crate::network::request::MAX_PACKAGE_ARTIFACTS_PER_REQUEST;
use crate::{MAX_TX_BYTES_LEN, MessageClass, NetworkMessage};

/// Response to a package artifact fetch request.
///
/// Carries the requested artifacts the responder holds, verbatim;
/// missing entries are simply absent. The receiver identifies each
/// artifact by hashing it — the request's own ids are the only trust
/// anchor, so no ids ride back.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
#[hbor(validate = artifacts_fit)]
pub struct GetPackageArtifactsResponse {
    /// The found artifacts' bytes.
    #[hbor(max = MAX_PACKAGE_ARTIFACTS_PER_REQUEST)]
    pub artifacts: Vec<Vec<u8>>,
}

/// Each artifact is bounded by what a publish transaction can carry, so
/// a response is refused at decode before an oversized allocation.
fn artifacts_fit(response: &GetPackageArtifactsResponse) -> Result<(), &'static str> {
    if response
        .artifacts
        .iter()
        .all(|artifact| artifact.len() <= MAX_TX_BYTES_LEN)
    {
        Ok(())
    } else {
        Err("an artifact exceeds the publish byte cap")
    }
}

impl GetPackageArtifactsResponse {
    /// Build a response carrying the supplied artifacts.
    #[must_use]
    pub const fn new(artifacts: Vec<Vec<u8>>) -> Self {
        Self { artifacts }
    }
}

impl NetworkMessage for GetPackageArtifactsResponse {
    fn message_type_id() -> &'static str {
        "package_artifact.response"
    }

    fn class() -> MessageClass {
        MessageClass::CrossShardProgress
    }
}
