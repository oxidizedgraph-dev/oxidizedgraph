//! Hourly telemetry aggregator.
//!
//! Drains the [`TelemetryBuffer`] on a fixed interval (hourly in production),
//! rolls the raw events up into per-(tenant, agent) invocation totals, and
//! writes those totals to the local persistence cache (a [`Checkpointer`]).
//!
//! The aggregator is the *only* consumer of the buffer, so draining is
//! single-writer and lock-free from the producers' perspective.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::checkpoint::{Checkpoint, Checkpointer};
use crate::state::AgentState;

use super::buffer::TelemetryBuffer;
use super::event::TelemetryEvent;

/// One hour, the production aggregation cadence.
pub const HOURLY: Duration = Duration::from_secs(3_600);

/// Thread id used to group telemetry rollup checkpoints in the cache.
pub const TELEMETRY_THREAD_ID: &str = "telemetry::invocation-totals";

/// Aggregated invocation totals for a single (tenant, agent) pair.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvocationTotal {
    /// Tenant the totals are billed to.
    pub tenant_id: String,
    /// Agent that produced the invocations.
    pub agent_id: String,
    /// Sum of `value` fields (i.e. number of billable invocations).
    pub total: u64,
}

/// Result of a single drain-and-persist cycle.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregationReport {
    /// When the rollup ran.
    pub aggregated_at: DateTime<Utc>,
    /// Number of raw events consumed in this cycle.
    pub events_processed: u64,
    /// Per-(tenant, agent) totals for this cycle.
    pub totals: Vec<InvocationTotal>,
}

impl AggregationReport {
    /// Total invocations across every tenant/agent in this cycle.
    pub fn grand_total(&self) -> u64 {
        self.totals.iter().map(|t| t.total).sum()
    }
}

/// Drains the buffer and folds the events into an [`AggregationReport`].
///
/// Pure with respect to persistence — persisting is a separate step so it can
/// be unit-tested without a checkpointer.
pub fn aggregate(buffer: &TelemetryBuffer) -> AggregationReport {
    let events = buffer.drain();
    fold_events(events)
}

fn fold_events(events: Vec<TelemetryEvent>) -> AggregationReport {
    // BTreeMap keeps the output deterministically ordered for stable snapshots.
    let mut totals: BTreeMap<(String, String), u64> = BTreeMap::new();
    let mut processed: u64 = 0;

    for event in events {
        processed += 1;
        *totals.entry((event.tenant_id, event.agent_id)).or_insert(0) += event.value;
    }

    AggregationReport {
        aggregated_at: Utc::now(),
        events_processed: processed,
        totals: totals
            .into_iter()
            .map(|((tenant_id, agent_id), total)| InvocationTotal {
                tenant_id,
                agent_id,
                total,
            })
            .collect(),
    }
}

/// Drain the buffer, aggregate, and persist the rollup to the cache.
///
/// The rollup is stored as a [`Checkpoint`] whose `metadata` carries the
/// [`AggregationReport`], keyed under [`TELEMETRY_THREAD_ID`]. Returns the
/// report so callers (and tests) can assert on the totals.
pub async fn drain_and_persist(
    buffer: &TelemetryBuffer,
    cache: &dyn Checkpointer,
) -> AggregationReport {
    let report = aggregate(buffer);

    if report.events_processed == 0 {
        debug!("telemetry aggregator: nothing to persist this cycle");
        return report;
    }

    let metadata = serde_json::to_value(&report).unwrap_or(serde_json::Value::Null);
    let checkpoint = Checkpoint::new(
        TELEMETRY_THREAD_ID,
        "telemetry_aggregator",
        AgentState::new(),
    )
    .with_metadata(metadata);

    if let Err(e) = cache.save(checkpoint).await {
        warn!(error = %e, "telemetry aggregator: failed to persist invocation totals");
    } else {
        info!(
            events = report.events_processed,
            tenants_agents = report.totals.len(),
            grand_total = report.grand_total(),
            "telemetry aggregator: persisted invocation totals to cache"
        );
    }

    report
}

/// Spawn the hourly aggregator as a background tokio task.
///
/// Ticks every `interval` (use [`HOURLY`] in production), draining the buffer
/// and writing totals to `cache`. The returned [`JoinHandle`](tokio::task::JoinHandle)
/// can be aborted to stop the task. The task runs for the lifetime of the
/// process; dropping the handle detaches it.
pub fn spawn_hourly_aggregator(
    buffer: TelemetryBuffer,
    cache: Arc<dyn Checkpointer>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // Skip the immediate tick that `interval` fires at t=0.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let report = drain_and_persist(&buffer, cache.as_ref()).await;
            debug!(
                events = report.events_processed,
                "telemetry aggregator tick complete"
            );
        }
    })
}

#[cfg(test)]
mod tests {
    use super::super::event::{RuntimeType, TelemetryEvent};
    use super::*;
    use crate::checkpoint::MemoryCheckpointer;

    fn event(tenant: &str, agent: &str, step: &str) -> TelemetryEvent {
        TelemetryEvent::agent_invocation(tenant, agent, step, RuntimeType::Cpu)
    }

    #[test]
    fn aggregates_totals_per_tenant_agent() {
        let buf = TelemetryBuffer::with_capacity(64);
        buf.enqueue(event("t1", "a1", "llm_call"));
        buf.enqueue(event("t1", "a1", "tool_call"));
        buf.enqueue(event("t1", "a2", "conditional"));
        buf.enqueue(event("t2", "a1", "human_in_loop"));

        let report = aggregate(&buf);
        assert_eq!(report.events_processed, 4);
        assert_eq!(report.grand_total(), 4);
        assert_eq!(report.totals.len(), 3);

        let t1a1 = report
            .totals
            .iter()
            .find(|t| t.tenant_id == "t1" && t.agent_id == "a1")
            .unwrap();
        assert_eq!(t1a1.total, 2);
    }

    #[tokio::test]
    async fn drain_and_persist_writes_totals_to_cache() {
        let buf = TelemetryBuffer::with_capacity(64);
        for _ in 0..7 {
            buf.enqueue(event("tenant-x", "agent-y", "llm_call"));
        }

        let cache = MemoryCheckpointer::new();
        let report = drain_and_persist(&buf, &cache).await;
        assert_eq!(report.grand_total(), 7);
        assert!(buf.is_empty());

        // The rollup was persisted and is retrievable from the cache.
        let saved = cache.load(TELEMETRY_THREAD_ID).await.unwrap().unwrap();
        let persisted: AggregationReport = serde_json::from_value(saved.metadata).unwrap();
        assert_eq!(persisted.grand_total(), 7);
        assert_eq!(persisted.totals[0].tenant_id, "tenant-x");
    }

    #[tokio::test]
    async fn empty_cycle_persists_nothing() {
        let buf = TelemetryBuffer::with_capacity(8);
        let cache = MemoryCheckpointer::new();
        let report = drain_and_persist(&buf, &cache).await;
        assert_eq!(report.events_processed, 0);
        assert!(cache.load(TELEMETRY_THREAD_ID).await.unwrap().is_none());
    }
}
