//! Core traits for state machines.

use crate::{Action, Event};
use std::time::Duration;

/// A state machine that processes events.
///
/// This is the core abstraction for the consensus architecture.
/// All consensus logic is implemented as state machines that:
///
/// - **Synchronous**: No async, no `.await`
/// - **Deterministic**: Same state + event = same actions
/// - **Pure-ish**: Mutates self, but performs no I/O
///
/// # Example
///
/// ```ignore
/// impl StateMachine for NodeStateMachine {
///     fn handle(&mut self, event: Event) -> Vec<Action> {
///         match event {
///             Event::ProposalTimer => self.bft.on_proposal_timer(),
///             Event::BlockVoteReceived { from, vote } => {
///                 self.bft.on_block_vote(from, vote)
///             }
///             // ... etc
///         }
///     }
///
///     fn set_time(&mut self, now: Duration) {
///         self.now = now;
///     }
/// }
/// ```
pub trait StateMachine {
    /// Process an event, returning actions to perform.
    ///
    /// # Guarantees
    ///
    /// - **Synchronous**: This method never blocks or awaits
    /// - **Deterministic**: Given the same state and event, always returns the same actions
    /// - **No I/O**: All I/O is performed by the runner via the returned actions
    ///
    /// # Arguments
    ///
    /// * `event` - The event to process
    ///
    /// # Returns
    ///
    /// A list of actions for the runner to execute. Actions may include:
    /// - Sending network messages
    /// - Setting timers
    /// - Enqueueing internal events
    /// - Emitting notifications to clients
    fn handle(&mut self, event: Event) -> Vec<Action>;

    /// Set the current time.
    ///
    /// Called by the runner before each `handle()` call to provide the
    /// current simulation or wall-clock time.
    ///
    /// # Arguments
    ///
    /// * `now` - The current time as a duration since some epoch
    fn set_time(&mut self, now: Duration);

    /// Get the current time.
    ///
    /// Returns the time that was last set via `set_time()`.
    fn now(&self) -> Duration;
}

/// Marker trait for state machines that can be composed.
///
/// Sub-state machines handle a subset of events and can be combined
/// into a larger composite state machine.
///
/// # Example
///
/// ```ignore
/// struct NodeStateMachine {
///     bft: BftState,
///     execution: ExecutionState,
///     mempool: MempoolState,
/// }
///
/// impl StateMachine for NodeStateMachine {
///     fn handle(&mut self, event: Event) -> Vec<Action> {
///         // Try each sub-machine in order
///         if let Some(actions) = self.bft.try_handle(&event) {
///             return actions;
///         }
///         if let Some(actions) = self.execution.try_handle(&event) {
///             return actions;
///         }
///         if let Some(actions) = self.mempool.try_handle(&event) {
///             return actions;
///         }
///         vec![] // Unknown event
///     }
/// }
/// ```
pub trait SubStateMachine {
    /// Handle an event if it's relevant to this sub-state machine.
    ///
    /// Returns `Some(actions)` if this sub-machine handled the event,
    /// or `None` if the event should be passed to other sub-machines.
    ///
    /// # Arguments
    ///
    /// * `event` - The event to potentially handle
    ///
    /// # Returns
    ///
    /// - `Some(actions)` if this sub-machine handled the event
    /// - `None` if the event is not relevant to this sub-machine
    fn try_handle(&mut self, event: &Event) -> Option<Vec<Action>>;

    /// Set the current time on this sub-machine.
    fn set_time(&mut self, now: Duration);
}

/// Extension trait for composing multiple sub-state machines.
pub trait CompositeStateMachine {
    /// Get the sub-state machines in processing order.
    fn sub_machines(&mut self) -> Vec<&mut dyn SubStateMachine>;
}

impl<T: CompositeStateMachine> StateMachine for T {
    fn handle(&mut self, event: Event) -> Vec<Action> {
        for sub in self.sub_machines() {
            if let Some(actions) = sub.try_handle(&event) {
                return actions;
            }
        }
        vec![] // No sub-machine handled this event
    }

    fn set_time(&mut self, now: Duration) {
        for sub in self.sub_machines() {
            sub.set_time(now);
        }
    }

    fn now(&self) -> Duration {
        // This is a bit awkward - composite machines need their own time tracking
        Duration::ZERO
    }
}
