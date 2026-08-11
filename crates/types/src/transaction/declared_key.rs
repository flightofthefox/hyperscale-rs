//! [`DeclaredKey`]: the substate-granular admission key.
//!
//! A declared key names an access target in the engine's identity space.
//! It is the mempool's conflict-key domain and is always derived locally
//! from effect metadata. Nothing key-granular travels on the wire; every
//! effect set is derived from the manifest and published metadata on each
//! node.

use hyperscale_hbor::Hbor;

use crate::{Address, LocalKey, SubstateKey};

/// One declared access target: an owner prefix, or one substate cell —
/// the cell variant is exactly the state leaf key.
///
/// Two keys conflict only when equal — an owner-granular key and a cell
/// under the same owner are distinct keys, so a producer narrowing its
/// declarations must narrow them consistently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Hbor)]
pub enum DeclaredKey {
    /// Owner-granular: every cell under the owner's prefix.
    Prefix(Address),
    /// One substate cell.
    Cell(SubstateKey),
}

impl DeclaredKey {
    /// The substate-granular key for a leaf `[owner | local]`.
    #[must_use]
    pub const fn substate(owner: Address, local: [u8; 16]) -> Self {
        Self::Cell(SubstateKey {
            owner,
            local: LocalKey(local),
        })
    }

    /// The owner-granular key for a prefix.
    #[must_use]
    pub const fn prefix(owner: Address) -> Self {
        Self::Prefix(owner)
    }

    /// The owning address — the routing half either variant carries.
    #[must_use]
    pub const fn owner(&self) -> Address {
        match self {
            Self::Prefix(owner) => *owner,
            Self::Cell(key) => key.owner,
        }
    }

    /// The cell key, when declared finer than owner-granular.
    #[must_use]
    pub const fn cell(&self) -> Option<SubstateKey> {
        match self {
            Self::Prefix(_) => None,
            Self::Cell(key) => Some(*key),
        }
    }
}
