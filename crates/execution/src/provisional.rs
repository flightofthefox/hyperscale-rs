//! Cells an unresolved cross-shard leg has claimed, and the test a
//! candidate transaction has to pass to join a tick.
//!
//! A cross-shard leg's local writes are provisional until its wave
//! resolves, and nothing may read a provisional value. A transaction the
//! shard commits afterwards would therefore execute against the cell as
//! it stood *before* the leg — and since a receipt carries the absolute
//! its baseline produced, whichever of the two settles second overwrites
//! the other. So a candidate touching a claimed cell stays out of the
//! tick and enters the first one composed after the claim clears.
//!
//! The claim is the leg's *declared* mutations rather than the writes it
//! turned out to make: every substate a transaction touches is declared,
//! so the declaration covers the writes, and it is known when the leg
//! joins its tick rather than when its batch comes back.

use std::collections::BTreeSet;

use hyperscale_types::{Address, DeclaredKey, SubstateKey};

/// The cells unresolved cross-shard legs have claimed.
///
/// Claims and candidates are both [`DeclaredKey`]s, which name either one
/// cell or a whole owner prefix. Two of them overlap when they are equal
/// or when either names the prefix the other sits under — a range claim
/// covers the points inside it.
#[derive(Debug, Default)]
pub struct ProvisionalCells {
    cells: BTreeSet<SubstateKey>,
    prefixes: BTreeSet<Address>,
    owners: BTreeSet<Address>,
}

impl ProvisionalCells {
    /// Record what one unresolved leg declared it would mutate.
    pub fn claim(&mut self, keys: &[DeclaredKey]) {
        for key in keys {
            self.owners.insert(key.owner());
            match key {
                DeclaredKey::Cell(cell) => {
                    self.cells.insert(*cell);
                }
                DeclaredKey::Prefix(owner) => {
                    self.prefixes.insert(*owner);
                }
            }
        }
    }

    /// Whether nothing is claimed — the common case, and worth
    /// short-circuiting on since it spares every candidate the walk.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.owners.is_empty()
    }

    /// Whether a candidate declaring `keys` overlaps a claim.
    #[must_use]
    pub fn blocks(&self, keys: &[DeclaredKey]) -> bool {
        keys.iter().any(|key| match key {
            // A claimed prefix covers every cell under it.
            DeclaredKey::Cell(cell) => {
                self.cells.contains(cell) || self.prefixes.contains(&cell.owner)
            }
            // A candidate prefix covers every claim under it.
            DeclaredKey::Prefix(owner) => self.owners.contains(owner),
        })
    }
}

#[cfg(test)]
mod tests {
    use hyperscale_types::LocalKey;

    use super::*;

    fn cell(owner: u8, local: u8) -> DeclaredKey {
        DeclaredKey::Cell(SubstateKey {
            owner: Address([owner; 16]),
            local: LocalKey([local; 16]),
        })
    }

    fn prefix(owner: u8) -> DeclaredKey {
        DeclaredKey::Prefix(Address([owner; 16]))
    }

    #[test]
    fn nothing_is_blocked_by_an_empty_claim_set() {
        let claims = ProvisionalCells::default();
        assert!(claims.is_empty());
        assert!(!claims.blocks(&[cell(1, 1), prefix(2)]));
    }

    #[test]
    fn a_claim_blocks_its_own_cell_and_leaves_siblings_alone() {
        let mut claims = ProvisionalCells::default();
        claims.claim(&[cell(1, 1)]);
        assert!(claims.blocks(&[cell(1, 1)]));
        assert!(!claims.blocks(&[cell(1, 2)]), "a sibling cell is untouched");
        assert!(!claims.blocks(&[cell(2, 1)]), "another owner is untouched");
    }

    /// Granularity cuts both ways: a range claim covers the points inside
    /// it, and a range candidate covers the points already claimed. Equal
    /// keys alone would let a point access slip past a range claim on its
    /// own owner.
    #[test]
    fn prefixes_and_cells_overlap_in_both_directions() {
        let mut over_range = ProvisionalCells::default();
        over_range.claim(&[prefix(1)]);
        assert!(over_range.blocks(&[cell(1, 9)]));

        let mut over_cell = ProvisionalCells::default();
        over_cell.claim(&[cell(1, 9)]);
        assert!(over_cell.blocks(&[prefix(1)]));
    }
}
