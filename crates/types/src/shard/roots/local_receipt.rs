//! [`LocalReceiptRoot`]: the root over a block's local receipts, one
//! leaf per receipt's local hash.

use crate::{Hash, LeafRoot, LocalReceiptRoot, StoredReceipt};

impl LeafRoot for LocalReceiptRoot {
    type Leaf = StoredReceipt;

    const ZERO: Self = Self::ZERO;

    fn from_raw(raw: Hash) -> Self {
        Self::from_raw(raw)
    }

    fn leaf(receipt: &Self::Leaf) -> Hash {
        receipt.consensus.local_receipt_hash()
    }
}
