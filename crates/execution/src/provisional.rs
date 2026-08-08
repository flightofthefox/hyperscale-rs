//! What unresolved cross-shard legs hold, and the test a candidate
//! transaction has to pass to join a tick.
//!
//! A cross-shard leg's local writes are provisional until its wave
//! resolves. Whether that stops a later transaction depends entirely on
//! *how* each of them reaches the cell, and the kernel already decides
//! that: [`compatible`] is the same relation it uses to schedule a batch,
//! asked here across a boundary the batch cannot see.
//!
//! The two commutative modes are what make this worth asking. A delta and
//! a reservation each say what they moved rather than what the cell ends
//! at, so neither depends on seeing the other and settlement composes
//! them in any order — two payments on one vault need not take turns. A
//! fresh read does depend on seeing it, and an exclusive write carries an
//! absolute that cannot be composed with anything, so both still wait.
//!
//! The claim is a leg's *declared* access rather than the writes it
//! turned out to make: every substate a transaction touches is declared,
//! so the declaration covers the outcome, and it is known when the leg
//! joins its tick rather than when its batch comes back.

use std::collections::{BTreeMap, BTreeSet};

use hyperscale_types::{Address, DeclaredKey, Mode, ModeKind, SubstateKey, compatible};

/// How unresolved legs are reaching the cells they claimed.
///
/// Claims and candidates are both [`DeclaredKey`]s, which name either one
/// cell or a whole owner prefix. Two of them overlap when they are equal
/// or when either names the prefix the other sits under — a range claim
/// covers the points inside it. Overlap alone decides nothing; what
/// decides is whether the modes on the two sides can be in flight
/// together.
#[derive(Debug, Default)]
pub struct ProvisionalCells {
    cells: BTreeMap<SubstateKey, BTreeSet<ModeKind>>,
    prefixes: BTreeMap<Address, BTreeSet<ModeKind>>,
    /// Every mode held anywhere under an owner, for the case where the
    /// *candidate* names the range and the claims sit inside it.
    owners: BTreeMap<Address, BTreeSet<ModeKind>>,
}

impl ProvisionalCells {
    /// Record how one unresolved leg declared it would reach each cell.
    pub fn claim(&mut self, declared: &[(DeclaredKey, Mode)]) {
        for (key, mode) in declared {
            let kind = mode.kind();
            self.owners.entry(key.owner()).or_default().insert(kind);
            match key {
                DeclaredKey::Cell(cell) => {
                    self.cells.entry(*cell).or_default().insert(kind);
                }
                DeclaredKey::Prefix(owner) => {
                    self.prefixes.entry(*owner).or_default().insert(kind);
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

    /// Whether a candidate's declared access cannot be in flight beside
    /// what is already held.
    ///
    /// Incompatible on any one cell is enough: the candidate is one
    /// transaction and it executes whole or not at all.
    #[must_use]
    pub fn blocks(&self, declared: &[(DeclaredKey, Mode)]) -> bool {
        declared.iter().any(|(key, mode)| {
            let candidate = mode.kind();
            let held: &[&BTreeSet<ModeKind>] = &match key {
                // A claimed prefix covers every cell under it, so both
                // the point's own claims and its owner's range claims
                // are in the way.
                DeclaredKey::Cell(cell) => [self.cells.get(cell), self.prefixes.get(&cell.owner)],
                // A candidate range covers every claim under its owner.
                DeclaredKey::Prefix(owner) => [self.owners.get(owner), None],
            }
            .map(Option::into_iter)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            held.iter()
                .flat_map(|modes| modes.iter())
                .any(|held| !compatible(*held, candidate))
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

    const RESERVE: Mode = Mode::Reserve { amount: 5 };

    #[test]
    fn nothing_is_blocked_by_an_empty_claim_set() {
        let claims = ProvisionalCells::default();
        assert!(claims.is_empty());
        assert!(!claims.blocks(&[(cell(1, 1), Mode::Write)]));
    }

    /// The reason the whole relation is here: payment traffic is delta
    /// and reserve, and those compose. A vault an unresolved leg is
    /// moving does not stop the next payment from moving it too.
    #[test]
    fn commutative_access_does_not_wait_on_commutative_access() {
        let mut claims = ProvisionalCells::default();
        claims.claim(&[(cell(1, 1), Mode::Delta)]);
        assert!(!claims.blocks(&[(cell(1, 1), Mode::Delta)]));
        assert!(!claims.blocks(&[(cell(1, 1), RESERVE)]));

        let mut reserved = ProvisionalCells::default();
        reserved.claim(&[(cell(1, 1), RESERVE)]);
        assert!(!reserved.blocks(&[(cell(1, 1), Mode::Delta)]));
        assert!(!reserved.blocks(&[(cell(1, 1), RESERVE)]));
    }

    /// A read depends on the value, and an exclusive write replaces it.
    /// Neither survives a change it cannot see.
    #[test]
    fn a_read_or_an_exclusive_write_still_waits() {
        let mut claims = ProvisionalCells::default();
        claims.claim(&[(cell(1, 1), Mode::Delta)]);
        assert!(claims.blocks(&[(cell(1, 1), Mode::Read)]));
        assert!(claims.blocks(&[(cell(1, 1), Mode::Write)]));
    }

    /// An exclusive claim excludes everything, commutative included: it
    /// carries an absolute, and an absolute composes with nothing.
    #[test]
    fn an_exclusive_claim_excludes_every_mode() {
        let mut claims = ProvisionalCells::default();
        claims.claim(&[(cell(1, 1), Mode::Write)]);
        for mode in [Mode::Delta, RESERVE, Mode::Read, Mode::Write] {
            assert!(
                claims.blocks(&[(cell(1, 1), mode)]),
                "{mode:?} slipped past"
            );
        }
    }

    #[test]
    fn a_claim_leaves_siblings_and_other_owners_alone() {
        let mut claims = ProvisionalCells::default();
        claims.claim(&[(cell(1, 1), Mode::Write)]);
        assert!(!claims.blocks(&[(cell(1, 2), Mode::Write)]), "sibling cell");
        assert!(!claims.blocks(&[(cell(2, 1), Mode::Write)]), "other owner");
    }

    /// Granularity cuts both ways: a range claim covers the points inside
    /// it, and a range candidate covers the points already claimed.
    #[test]
    fn prefixes_and_cells_overlap_in_both_directions() {
        let mut over_range = ProvisionalCells::default();
        over_range.claim(&[(prefix(1), Mode::Write)]);
        assert!(over_range.blocks(&[(cell(1, 9), Mode::Write)]));

        let mut over_cell = ProvisionalCells::default();
        over_cell.claim(&[(cell(1, 9), Mode::Write)]);
        assert!(over_cell.blocks(&[(prefix(1), Mode::Write)]));
    }

    /// A locked read declares nothing anywhere, so it neither claims nor
    /// is claimed against — the one mode that never contends.
    #[test]
    fn a_locked_read_contends_with_nothing() {
        let mut claims = ProvisionalCells::default();
        claims.claim(&[(cell(1, 1), Mode::Write)]);
        assert!(!claims.blocks(&[(cell(1, 1), Mode::Locked)]));
    }
}
