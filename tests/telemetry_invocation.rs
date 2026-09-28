//! Integration tests for agent-invocation telemetry (lornu-ai/plans#24).
//!
//! Covers the four acceptance criteria:
//!  1. Every workflow node execution triggers a telemetry event capture.
//!  2. Telemetry ingestion adds < 2ms to the DAG step loop (non-blocking).
//!  3. The in-memory queue handles up to 5,000 concurrent events without
//!     memory exhaustion or data loss.
//!  4. Standard DAG runs produce accurate invocation counts.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use oxidizedgraph::prelude::*;
use oxidizedgraph::telemetry::{aggregator, TELEMETRY_THREAD_ID};

// ── Helpers ──────────────────────────────────────────────────────────

/// A node that simply continues; represents any DAG node kind
/// (llm_call, tool_call, conditional, human_in_loop).
struct StepNode {
    id: String,
}

impl StepNode {
    fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

#[async_trait]
impl NodeExecutor for StepNode {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, _state: SharedState) -> Result<NodeOutput, NodeError> {
        Ok(NodeOutput::cont())
    }
}

/// Build a linear 4-node DAG exercising the four documented node kinds, then
/// route to END. Nodes continue in declared order via default edges.
fn four_kind_graph() -> CompiledGraph {
    GraphBuilder::new()
        .name("telemetry-dag")
        .add_node(StepNode::new("llm_call"))
        .add_node(StepNode::new("tool_call"))
        .add_node(StepNode::new("conditional"))
        .add_node(StepNode::new("human_in_loop"))
        .set_entry_point("llm_call")
        .add_edge("llm_call", "tool_call")
        .add_edge("tool_call", "conditional")
        .add_edge("conditional", "human_in_loop")
        .add_edge_to_end("human_in_loop")
        .compile()
        .unwrap()
}

fn seeded_state() -> AgentState {
    let mut state = AgentState::new();
    state.set_context(CTX_TENANT_ID, "tenant-acme".to_string());
    state.set_context(CTX_AGENT_ID, "agent-router".to_string());
    state.set_context(CTX_RUNTIME_TYPE, "GPU".to_string());
    state
}

// ── AC1 + AC4: every node fires exactly one accurately-counted event ──

#[tokio::test]
async fn every_node_completion_fires_one_event() {
    let recorder = TelemetryRecorder::new(TelemetryConfig::enabled());
    let buffer = recorder.buffer();

    let runner = TracedRunner::with_context(
        Arc::new(four_kind_graph()),
        RunContext::with_ids("run-1", "thread-1"),
        RunnerConfig::default(),
    )
    .with_telemetry(recorder);

    let result = runner.invoke(seeded_state()).await.unwrap();

    // 4 nodes executed → 4 transitions → 4 telemetry events, no loss.
    assert_eq!(result.transition_log.len(), 4);
    let events = buffer.drain();
    assert_eq!(events.len(), 4, "one AGENT_INVOCATION per node completion");
    assert_eq!(buffer.dropped_total(), 0);

    // Payload identity is threaded from state, step names match node ids.
    let steps: Vec<&str> = events
        .iter()
        .map(|e| e.metadata.step_name.as_str())
        .collect();
    assert_eq!(
        steps,
        ["llm_call", "tool_call", "conditional", "human_in_loop"]
    );
    for e in &events {
        assert_eq!(e.tenant_id, "tenant-acme");
        assert_eq!(e.agent_id, "agent-router");
        assert_eq!(e.event_type, EventType::AgentInvocation);
        assert_eq!(e.value, 1);
        assert_eq!(e.metadata.runtime_type, RuntimeType::Gpu);
    }
}

#[tokio::test]
async fn disabled_telemetry_captures_nothing() {
    // Default runner has telemetry disabled → zero overhead, zero events.
    let runner = TracedRunner::with_context(
        Arc::new(four_kind_graph()),
        RunContext::with_ids("run-off", "thread-off"),
        RunnerConfig::default(),
    );
    let result = runner.invoke(seeded_state()).await.unwrap();
    assert_eq!(result.transition_log.len(), 4);
    assert!(!runner.telemetry().is_enabled());
    assert!(runner.telemetry().buffer().is_empty());
}

#[tokio::test]
async fn aggregator_counts_invocations_across_runs() {
    let recorder = TelemetryRecorder::new(TelemetryConfig::enabled());
    let buffer = recorder.buffer();

    // Run the same 4-node DAG three times → 12 invocations for one tenant/agent.
    for i in 0..3 {
        let runner = TracedRunner::with_context(
            Arc::new(four_kind_graph()),
            RunContext::with_ids(format!("run-{i}"), format!("thread-{i}")),
            RunnerConfig::default(),
        )
        .with_telemetry(recorder.clone());
        runner.invoke(seeded_state()).await.unwrap();
    }

    let cache = Arc::new(MemoryCheckpointer::new());
    let report = aggregator::drain_and_persist(&buffer, cache.as_ref()).await;

    assert_eq!(report.events_processed, 12);
    assert_eq!(report.grand_total(), 12);
    assert_eq!(report.totals.len(), 1);
    assert_eq!(report.totals[0].tenant_id, "tenant-acme");
    assert_eq!(report.totals[0].agent_id, "agent-router");
    assert_eq!(report.totals[0].total, 12);

    // Totals were written to the local persistence cache.
    let saved = cache.load(TELEMETRY_THREAD_ID).await.unwrap().unwrap();
    let persisted: aggregator::AggregationReport = serde_json::from_value(saved.metadata).unwrap();
    assert_eq!(persisted.grand_total(), 12);
}

// ── AC3: 5,000 concurrent events, bounded memory, no data loss ──

#[tokio::test]
async fn handles_five_thousand_concurrent_events_without_loss() {
    // Default capacity (8_192) comfortably holds the 5,000-event target.
    let recorder = TelemetryRecorder::new(TelemetryConfig::enabled());
    let buffer = recorder.buffer();
    let capacity = buffer.capacity();
    assert!(
        capacity >= 5_000,
        "default capacity must absorb the AC3 burst"
    );

    // Fan out 5,000 enqueues across 50 concurrent tasks — the buffer is
    // Arc-backed and Send, so producers race on it exactly like the real
    // multi-node loop would.
    let mut handles = Vec::new();
    for t in 0..50u32 {
        let rec = recorder.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..100 {
                let mut state = AgentState::new();
                state.set_context(CTX_TENANT_ID, format!("tenant-{}", t % 5));
                state.set_context(CTX_AGENT_ID, format!("agent-{t}"));
                rec.record_node_completion(&state, "llm_call", "run-burst");
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    // All 5,000 fit within the bound: none dropped, memory bounded by capacity.
    assert_eq!(buffer.enqueued_total(), 5_000);
    assert_eq!(buffer.dropped_total(), 0);
    assert!(buffer.len() <= capacity, "buffer never exceeds its bound");

    // Draining recovers every event — no data loss.
    let drained = buffer.drain();
    assert_eq!(drained.len(), 5_000);
}

#[tokio::test]
async fn overflow_is_bounded_and_deterministic_not_lost_silently() {
    // A deliberately tiny buffer proves the drop path is bounded + counted:
    // events are never lost silently (dropped_total is exact), never panic,
    // and memory is capped at capacity.
    let recorder = TelemetryRecorder::new(TelemetryConfig::enabled().with_capacity(1_000));
    let buffer = recorder.buffer();

    let state = {
        let mut s = AgentState::new();
        s.set_context(CTX_TENANT_ID, "t".to_string());
        s
    };
    for _ in 0..5_000 {
        recorder.record_node_completion(&state, "tool_call", "run");
    }

    assert_eq!(buffer.capacity(), 1_000);
    assert_eq!(buffer.enqueued_total(), 1_000);
    assert_eq!(buffer.dropped_total(), 4_000);
    // Enqueued + dropped == total attempted: nothing vanished unaccounted.
    assert_eq!(buffer.enqueued_total() + buffer.dropped_total(), 5_000);
    assert!(buffer.len() <= 1_000);
}

// ── AC2: enqueue is non-blocking / negligible latency ──

#[tokio::test]
async fn enqueue_is_non_blocking_under_2ms() {
    let recorder = TelemetryRecorder::new(TelemetryConfig::enabled());
    let mut state = AgentState::new();
    state.set_context(CTX_TENANT_ID, "tenant-perf".to_string());

    // Even when the buffer is completely full, a single enqueue must return
    // immediately (drop-newest) rather than block the step loop.
    let capacity = recorder.buffer().capacity();
    for _ in 0..capacity {
        recorder.record_node_completion(&state, "llm_call", "run");
    }
    assert_eq!(recorder.buffer().len(), capacity);

    let start = Instant::now();
    recorder.record_node_completion(&state, "llm_call", "run"); // full → dropped
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_micros() < 2_000,
        "enqueue on a full buffer must be non-blocking (<2ms), took {elapsed:?}"
    );
    assert!(recorder.buffer().dropped_total() >= 1);

    // Amortized cost across many enqueues stays far below the 2ms/step budget.
    let start = Instant::now();
    let iterations = 10_000;
    for _ in 0..iterations {
        recorder.record_node_completion(&state, "llm_call", "run");
    }
    let per_call = start.elapsed() / iterations;
    assert!(
        per_call.as_micros() < 2_000,
        "per-enqueue cost must be well under 2ms, was {per_call:?}"
    );
}
