//! Instance record fetch response (cross-shard component resolution).

use hyperscale_hbor::Hbor;
use hyperscale_vm_types::MAX_CELL_VALUE_LEN;

use crate::network::request::MAX_INSTANCE_RECORDS_PER_REQUEST;
use crate::{MessageClass, NetworkMessage};

/// Response to an instance record fetch request.
///
/// Carries the configuration leaves the responder holds, verbatim;
/// missing entries are simply absent. The receiver identifies each
/// record by re-deriving the address its contents commit — the request's
/// own ids are the only trust anchor, so no addresses ride back.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
#[hbor(validate = records_fit)]
pub struct GetInstanceRecordsResponse {
    /// The found records, as their configuration leaves store them.
    #[hbor(max = MAX_INSTANCE_RECORDS_PER_REQUEST)]
    pub records: Vec<Vec<u8>>,
}

/// No record is larger than the cell that held it.
///
/// A protocol bound rather than a guard on what decoding allocates: a
/// claimed length the remaining input cannot satisfy is refused before
/// any collection is built. What this adds is that a well-formed frame
/// still cannot name a record no configuration leaf could have stored.
fn records_fit(response: &GetInstanceRecordsResponse) -> Result<(), &'static str> {
    if response
        .records
        .iter()
        .all(|record| record.len() <= MAX_CELL_VALUE_LEN)
    {
        Ok(())
    } else {
        Err("a record exceeds the substate value cap")
    }
}

impl GetInstanceRecordsResponse {
    /// Build a response carrying the supplied records.
    #[must_use]
    pub const fn new(records: Vec<Vec<u8>>) -> Self {
        Self { records }
    }
}

impl NetworkMessage for GetInstanceRecordsResponse {
    fn message_type_id() -> &'static str {
        "instance_record.response"
    }

    fn class() -> MessageClass {
        MessageClass::CrossShardProgress
    }
}
