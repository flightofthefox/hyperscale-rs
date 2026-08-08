//! Settled-transaction window request for the split-boundary fence.
//!
//! After a shard `P` terminates at a split, a surviving counterpart must
//! decide, for any cross-shard transaction still referencing `P`, whether `P`
//! actually settled it in its chain at or before the terminal block
//! `B`. It reads `P`'s beacon-attested `settled_txs_root` from its own
//! fold and fetches the whole window settled-transaction list in one shot: the
//! complete set `S_P` of the **cross-shard** transactions `P` settled over
//! `[B − RETENTION_HORIZON, B]` (single-shard transactions are never queried, so
//! they are excluded). The requester accepts the list iff its recomputed
//! root equals the attested one, so a withheld transaction changes the root and the
//! absence of any transaction from the verified-complete set is sound (see
//! [`GetSettledTxsResponse`]).

use hyperscale_hbor::Hbor;

use crate::network::response::GetSettledTxsResponse;
use crate::{BlockHash, BlockHeight, MessageClass, NetworkMessage, Request};

/// Request a terminated shard's complete settled-transaction window list,
/// anchored at its terminal block.
#[derive(Debug, Clone, PartialEq, Eq, Hbor)]
pub struct GetSettledTxsRequest {
    /// Height of the terminal block `B` the window ends at.
    pub terminal_height: BlockHeight,
    /// Expected hash of `B` — the beacon-attested terminal the requester
    /// reads from its fold. The server resolves `B` by height and answers
    /// `not_found` on a hash mismatch.
    pub terminal_block_hash: BlockHash,
}

impl GetSettledTxsRequest {
    /// Request the settled-transaction window ending at terminal block
    /// `(terminal_height, terminal_block_hash)`.
    #[must_use]
    pub const fn new(terminal_height: BlockHeight, terminal_block_hash: BlockHash) -> Self {
        Self {
            terminal_height,
            terminal_block_hash,
        }
    }
}

impl NetworkMessage for GetSettledTxsRequest {
    fn message_type_id() -> &'static str {
        "settled_txs.request"
    }

    fn class() -> MessageClass {
        MessageClass::Bulk
    }
}

impl Request for GetSettledTxsRequest {
    type Response = GetSettledTxsResponse;

    fn is_empty_response(response: &Self::Response) -> bool {
        response.txs.is_none()
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_hbor::{from_slice as hbor_from_slice, to_vec as hbor_to_vec};

    use super::*;
    use crate::Hash;

    #[test]
    fn test_hbor_roundtrip() {
        let request = GetSettledTxsRequest::new(
            BlockHeight::new(98),
            BlockHash::from_raw(Hash::from_bytes(b"terminal")),
        );
        let encoded = hbor_to_vec(&request).unwrap();
        let decoded: GetSettledTxsRequest = hbor_from_slice(&encoded).unwrap();
        assert_eq!(request, decoded);
    }
}
