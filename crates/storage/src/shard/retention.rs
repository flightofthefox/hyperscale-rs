//! What history a shard keeps, and until when.
//!
//! Retention is a span of weighted time. Every window that licenses a
//! read of history — a reclaim probe's, a provision's, a state proof's —
//! is stated in milliseconds against the consensus-authenticated
//! weighted timestamp, so a count of versions answers those windows only
//! through whatever block rate the chain happens to be running at.
//! Nothing enforces a rate, which makes a count an answer to a different
//! question: whether a licensed anchor is servable would depend on how
//! fast blocks had happened to arrive.
//!
//! So the floor is [`RETENTION_HORIZON`] behind the tip's own timestamp,
//! and every version a consumer may name is servable by construction.
//! What that costs is history proportional to the block rate, and what
//! bounds it is the per-block write caps the protocol already sets: what
//! is kept is the writes of the last horizon, which is what a consumer is
//! licensed to ask about.
//!
//! # One floor, four readers
//!
//! The floor is stored rather than recomputed. A historical cell read, a
//! historical range read, `snapshot_at` and the collectors ask for it,
//! and what a reader may ask for has to be exactly what the collector has
//! not deleted. Admitting `height >= floor` and deleting below it is that
//! relationship, and it holds because there is one value rather than four
//! arithmetic expressions that have to agree — and one fold moving it,
//! [`retire_dated`], whichever backend dates the versions.
//!
//! # Versions this store never committed
//!
//! The floor is the first *dated* version at or above where it stands, so
//! a store whose history begins above zero needs nothing said about the
//! versions below. A snap-synced store dates its first committed block
//! and the floor arrives there; a split child's dates start at its
//! adoption. Neither needs a seed, because a version with no date is one
//! with no history to serve.

use hyperscale_types::{RETENTION_HORIZON, WeightedTimestamp};

/// What dating a version retires: the dated versions that fell outside
/// [`RETENTION_HORIZON`] of the tip, and the floor that leaves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Retired {
    /// The versions whose dates aged out, ascending.
    pub versions: Vec<u64>,
    /// The oldest version still answered for.
    pub floor: u64,
}

/// Move `floor` past what a commit at `version`, dated `tip_ts`, retires.
///
/// `dated` is every dated version from `floor` on, ascending, with its
/// timestamp — the scan runs forward from the stored floor, so each dated
/// version is passed once over the life of the store whatever the block
/// rate. The floor moves only past what it retires: a version with no
/// date of its own — the empty tree at zero, or anything below where a
/// snap-synced store's history begins — is left where it was rather than
/// skipped over, since nothing about it has aged out. Reading stops at
/// `version` itself, so a commit re-recording a height it already holds
/// retires nothing at or past it.
#[must_use]
pub fn retire_dated(
    floor: u64,
    version: u64,
    tip_ts: WeightedTimestamp,
    dated: impl IntoIterator<Item = (u64, u64)>,
) -> Retired {
    let cutoff = tip_ts.minus(RETENTION_HORIZON).as_millis();
    let versions: Vec<u64> = dated
        .into_iter()
        .take_while(|(dated, ts)| *dated < version && *ts < cutoff)
        .map(|(dated, _)| dated)
        .collect();
    let floor = versions.last().map_or(floor, |last| last + 1);
    Retired { versions, floor }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn horizon_ms() -> u64 {
        u64::try_from(RETENTION_HORIZON.as_millis()).expect("fits")
    }

    /// Versions age out in order until one is inside the horizon, and the
    /// floor lands just past the last one retired.
    #[test]
    fn the_floor_moves_past_what_aged_out() {
        let step = horizon_ms() / 2;
        let dated = (1..=5u64).map(|v| (v, v * step));
        let tip = WeightedTimestamp::from_millis(6 * step);
        let retired = retire_dated(1, 6, tip, dated);
        assert_eq!(retired.versions, vec![1, 2, 3]);
        assert_eq!(retired.floor, 4);
    }

    /// Nothing aged out leaves the floor where it stood, and a version at
    /// or past the one being dated is never read.
    #[test]
    fn nothing_retired_leaves_the_floor_and_the_commit_itself_is_never_read() {
        let tip = WeightedTimestamp::from_millis(horizon_ms());
        let stood = retire_dated(3, 4, tip, [(3, horizon_ms() - 1)]);
        assert_eq!(
            stood,
            Retired {
                versions: Vec::new(),
                floor: 3
            }
        );
        let stopped = retire_dated(0, 2, tip, [(2, 0), (3, 0)]);
        assert_eq!(stopped.versions, Vec::<u64>::new());
        assert_eq!(stopped.floor, 0);
    }
}
