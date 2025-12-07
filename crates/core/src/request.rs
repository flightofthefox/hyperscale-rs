//! Client request tracking.

/// Opaque identifier for tracking client requests through the system.
///
/// The runner maintains a map of `RequestId` -> response channel.
/// This keeps async response handling out of the sync state machine.
///
/// # Example
///
/// ```ignore
/// // In the runner:
/// let request_id = RequestId(self.next_request_id);
/// self.next_request_id += 1;
///
/// let (response_tx, response_rx) = oneshot::channel();
/// self.pending_requests.insert(request_id, response_tx);
///
/// // Send event to state machine
/// self.event_tx.send(Event::SubmitTransaction { tx, request_id }).await;
///
/// // Later, when state machine returns Action::EmitTransactionResult:
/// if let Some(tx) = self.pending_requests.remove(&request_id) {
///     let _ = tx.send(result);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

impl RequestId {
    /// Create a new request ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "req-{}", self.0)
    }
}
