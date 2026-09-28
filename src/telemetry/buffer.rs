//! Bounded, lock-free-ish async buffer for telemetry events.
//!
//! The hot path (the DAG step loop) enqueues via [`TelemetryBuffer::enqueue`],
//! which is a **non-blocking** `try_send` into a bounded `crossbeam-channel`.
//! It never `.await`s and never blocks: if the queue is full the event is
//! dropped deterministically (drop-newest) and a counter is incremented, so the
//! core agent loop is never stalled and memory is strictly bounded by
//! `capacity`.
//!
//! The [aggregator](crate::telemetry::aggregator) is the sole consumer; it
//! drains the receiver end on its hourly tick.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use super::event::TelemetryEvent;

/// Default bounded capacity. Sized well above the 5,000-concurrent-event
/// acceptance target so a standard burst is fully captured without loss.
pub const DEFAULT_CAPACITY: usize = 8_192;

/// Result of a single enqueue attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnqueueOutcome {
    /// Event was accepted into the buffer.
    Buffered,
    /// Buffer was full; the event was dropped (drop-newest).
    DroppedFull,
    /// Consumer half was gone; the event was dropped.
    DroppedClosed,
}

impl EnqueueOutcome {
    /// Whether the event made it into the buffer.
    pub fn is_buffered(self) -> bool {
        matches!(self, EnqueueOutcome::Buffered)
    }
}

/// Shared, cheaply-cloneable handle to the telemetry buffer.
///
/// Cloning shares the same underlying queue and counters, so producers and the
/// aggregator observe a single consistent buffer.
#[derive(Clone)]
pub struct TelemetryBuffer {
    inner: Arc<BufferInner>,
}

struct BufferInner {
    sender: Sender<TelemetryEvent>,
    receiver: Receiver<TelemetryEvent>,
    capacity: usize,
    enqueued: AtomicU64,
    dropped: AtomicU64,
}

impl TelemetryBuffer {
    /// Create a buffer with [`DEFAULT_CAPACITY`].
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a buffer with an explicit bounded capacity.
    ///
    /// A capacity of `0` is promoted to `1` so the queue is always usable.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let (sender, receiver) = bounded(capacity);
        Self {
            inner: Arc::new(BufferInner {
                sender,
                receiver,
                capacity,
                enqueued: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
            }),
        }
    }

    /// Non-blocking enqueue for the hot path.
    ///
    /// Uses `try_send`, so it returns immediately whether or not the buffer has
    /// room. On overflow the event is dropped (drop-newest) and the drop
    /// counter is bumped — never blocks, never panics, never grows unbounded.
    #[inline]
    pub fn enqueue(&self, event: TelemetryEvent) -> EnqueueOutcome {
        match self.inner.sender.try_send(event) {
            Ok(()) => {
                self.inner.enqueued.fetch_add(1, Ordering::Relaxed);
                EnqueueOutcome::Buffered
            }
            Err(TrySendError::Full(_)) => {
                self.inner.dropped.fetch_add(1, Ordering::Relaxed);
                EnqueueOutcome::DroppedFull
            }
            Err(TrySendError::Disconnected(_)) => {
                self.inner.dropped.fetch_add(1, Ordering::Relaxed);
                EnqueueOutcome::DroppedClosed
            }
        }
    }

    /// Drain all currently-buffered events without blocking.
    ///
    /// Called by the aggregator on each tick. Returns the events in FIFO order.
    pub fn drain(&self) -> Vec<TelemetryEvent> {
        let mut out = Vec::new();
        while let Ok(event) = self.inner.receiver.try_recv() {
            out.push(event);
        }
        out
    }

    /// Number of events currently sitting in the buffer.
    pub fn len(&self) -> usize {
        self.inner.receiver.len()
    }

    /// Whether the buffer is currently empty.
    pub fn is_empty(&self) -> bool {
        self.inner.receiver.is_empty()
    }

    /// Bounded capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Total number of events successfully buffered over the buffer's lifetime.
    pub fn enqueued_total(&self) -> u64 {
        self.inner.enqueued.load(Ordering::Relaxed)
    }

    /// Total number of events dropped due to overflow / disconnection.
    pub fn dropped_total(&self) -> u64 {
        self.inner.dropped.load(Ordering::Relaxed)
    }
}

impl Default for TelemetryBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TelemetryBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TelemetryBuffer")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .field("enqueued_total", &self.enqueued_total())
            .field("dropped_total", &self.dropped_total())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::super::event::{RuntimeType, TelemetryEvent};
    use super::*;

    fn sample(step: &str) -> TelemetryEvent {
        TelemetryEvent::agent_invocation("tenant", "agent", step, RuntimeType::Cpu)
    }

    #[test]
    fn enqueue_and_drain_roundtrip() {
        let buf = TelemetryBuffer::with_capacity(16);
        for i in 0..5 {
            assert_eq!(
                buf.enqueue(sample(&format!("n{i}"))),
                EnqueueOutcome::Buffered
            );
        }
        assert_eq!(buf.len(), 5);
        let drained = buf.drain();
        assert_eq!(drained.len(), 5);
        assert!(buf.is_empty());
        // FIFO ordering preserved.
        assert_eq!(drained[0].metadata.step_name, "n0");
        assert_eq!(drained[4].metadata.step_name, "n4");
        assert_eq!(buf.enqueued_total(), 5);
        assert_eq!(buf.dropped_total(), 0);
    }

    #[test]
    fn overflow_drops_newest_and_counts() {
        let buf = TelemetryBuffer::with_capacity(4);
        // First 4 fit, remaining 6 are dropped — bounded memory, no panic.
        let mut buffered = 0;
        let mut dropped = 0;
        for i in 0..10 {
            match buf.enqueue(sample(&format!("n{i}"))) {
                EnqueueOutcome::Buffered => buffered += 1,
                EnqueueOutcome::DroppedFull => dropped += 1,
                EnqueueOutcome::DroppedClosed => unreachable!(),
            }
        }
        assert_eq!(buffered, 4);
        assert_eq!(dropped, 6);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.enqueued_total(), 4);
        assert_eq!(buf.dropped_total(), 6);
        // Retained events are the OLDEST (drop-newest is deterministic).
        let drained = buf.drain();
        assert_eq!(drained[0].metadata.step_name, "n0");
        assert_eq!(drained[3].metadata.step_name, "n3");
    }

    #[test]
    fn zero_capacity_is_promoted_to_one() {
        let buf = TelemetryBuffer::with_capacity(0);
        assert_eq!(buf.capacity(), 1);
        assert_eq!(buf.enqueue(sample("n")), EnqueueOutcome::Buffered);
        assert_eq!(buf.enqueue(sample("n")), EnqueueOutcome::DroppedFull);
    }
}
