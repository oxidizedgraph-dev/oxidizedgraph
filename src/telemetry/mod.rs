//! Agent-invocation telemetry for usage-based billing (lornu-ai/plans#24).
//!
//! Lornu bills per agent action, so every time a DAG agent node finishes
//! executing we publish an `AGENT_INVOCATION` telemetry event. This module
//! provides:
//!
//! - [`TelemetryEvent`] — the exact billing payload shape.
//! - [`TelemetryBuffer`] — a bounded, non-blocking async buffer. The hot path
//!   enqueues with `try_send` and never blocks; on overflow events are dropped
//!   deterministically (drop-newest) and counted, so memory is bounded and the
//!   core agent loop is never stalled.
//! - [`aggregator`] — an hourly task that drains the buffer, rolls events up
//!   into per-(tenant, agent) invocation totals, and writes them to the local
//!   persistence cache (a [`Checkpointer`](crate::checkpoint::Checkpointer)).
//! - [`TelemetryRecorder`] — the glue wired into the execution layer that reads
//!   `tenant_id` / `agent_id` / `runtime_type` from the run state and enqueues
//!   one event per node completion. Gated by [`TelemetryConfig::enabled`].
//!
//! # Threading identity through state
//!
//! The recorder reads billing identity from [`AgentState`] context:
//!
//! - `tenant_id` — reuses [`CTX_TENANT_ID`](crate::enterprise::CTX_TENANT_ID).
//! - `agent_id` — [`CTX_AGENT_ID`], falling back to the run id.
//! - `runtime_type` — [`CTX_RUNTIME_TYPE`] (`"CPU"` / `"GPU"`), defaulting to
//!   CPU.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxidizedgraph::telemetry::{TelemetryRecorder, TelemetryConfig, aggregator};
//! use std::sync::Arc;
//!
//! let recorder = TelemetryRecorder::new(TelemetryConfig::enabled());
//! // Wire the aggregator to the persistence cache.
//! let cache = Arc::new(oxidizedgraph::checkpoint::MemoryCheckpointer::new());
//! aggregator::spawn_hourly_aggregator(recorder.buffer(), cache, aggregator::HOURLY);
//! ```

pub mod aggregator;
mod buffer;
mod event;

pub use aggregator::{
    aggregate, drain_and_persist, spawn_hourly_aggregator, AggregationReport, InvocationTotal,
    HOURLY, TELEMETRY_THREAD_ID,
};
pub use buffer::{EnqueueOutcome, TelemetryBuffer, DEFAULT_CAPACITY};
pub use event::{EventType, RuntimeType, TelemetryEvent, TelemetryMetadata};

use crate::enterprise::CTX_TENANT_ID;
use crate::state::AgentState;

/// Context key for the agent identifier used in telemetry.
pub const CTX_AGENT_ID: &str = "agent_id";
/// Context key for the runtime class (`"CPU"` / `"GPU"`) of the current node.
pub const CTX_RUNTIME_TYPE: &str = "runtime_type";
/// Fallback tenant id when none is present on state.
pub const UNKNOWN_TENANT: &str = "unknown";

/// Configuration for the telemetry subsystem.
///
/// Follows the codebase convention of an explicit opt-in toggle for optional
/// subsystems: telemetry is **disabled by default** and must be turned on
/// (config toggle) before events are captured, so existing workflows pay zero
/// cost unless billing telemetry is wanted.
#[derive(Clone, Debug)]
pub struct TelemetryConfig {
    /// Whether the invocation hook is active. Disabled by default.
    pub enabled: bool,
    /// Bounded capacity of the in-memory buffer.
    pub buffer_capacity: usize,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            buffer_capacity: DEFAULT_CAPACITY,
        }
    }
}

impl TelemetryConfig {
    /// Disabled config (the default).
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Enabled config with the default buffer capacity.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Set the buffer capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }
}

/// Records agent-invocation telemetry at node-completion sites.
///
/// Holds the shared [`TelemetryBuffer`] and the [`TelemetryConfig`]. Cheap to
/// clone (the buffer is `Arc`-backed). When disabled, [`record_node_completion`]
/// is a no-op returning without touching the buffer.
///
/// [`record_node_completion`]: TelemetryRecorder::record_node_completion
#[derive(Clone, Debug)]
pub struct TelemetryRecorder {
    config: TelemetryConfig,
    buffer: TelemetryBuffer,
}

impl TelemetryRecorder {
    /// Create a recorder from config, allocating a buffer of the configured
    /// capacity.
    pub fn new(config: TelemetryConfig) -> Self {
        let buffer = TelemetryBuffer::with_capacity(config.buffer_capacity);
        Self { config, buffer }
    }

    /// Create a disabled recorder (default).
    pub fn disabled() -> Self {
        Self::new(TelemetryConfig::disabled())
    }

    /// Whether telemetry capture is active.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Shared handle to the underlying buffer (for the aggregator).
    pub fn buffer(&self) -> TelemetryBuffer {
        self.buffer.clone()
    }

    /// Enqueue an already-built event (non-blocking). Respects the toggle.
    pub fn record(&self, event: TelemetryEvent) -> EnqueueOutcome {
        if !self.config.enabled {
            return EnqueueOutcome::DroppedClosed;
        }
        self.buffer.enqueue(event)
    }

    /// Capture a node completion.
    ///
    /// Reads `tenant_id` / `agent_id` / `runtime_type` from the run state,
    /// builds an `AGENT_INVOCATION` event for `step_name`, and enqueues it
    /// non-blockingly. No-op (returns `false`) when telemetry is disabled.
    ///
    /// `run_id` is used as the agent id fallback when no explicit `agent_id`
    /// is present on state.
    pub fn record_node_completion(
        &self,
        state: &AgentState,
        step_name: &str,
        run_id: &str,
    ) -> bool {
        if !self.config.enabled {
            return false;
        }

        let tenant_id = state
            .get_context::<String>(CTX_TENANT_ID)
            .unwrap_or_else(|| UNKNOWN_TENANT.to_string());
        let agent_id = state
            .get_context::<String>(CTX_AGENT_ID)
            .unwrap_or_else(|| run_id.to_string());
        let runtime_type = state
            .get_context::<String>(CTX_RUNTIME_TYPE)
            .map(|s| RuntimeType::from_str_lossy(&s))
            .unwrap_or_default();

        let event = TelemetryEvent::agent_invocation(tenant_id, agent_id, step_name, runtime_type);
        self.buffer.enqueue(event).is_buffered()
    }
}

impl Default for TelemetryRecorder {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_recorder_captures_nothing() {
        let recorder = TelemetryRecorder::disabled();
        let mut state = AgentState::new();
        state.set_context(CTX_TENANT_ID, "tenant-1".to_string());
        assert!(!recorder.record_node_completion(&state, "llm_call", "run-1"));
        assert!(recorder.buffer().is_empty());
    }

    #[test]
    fn enabled_recorder_reads_identity_from_state() {
        let recorder = TelemetryRecorder::new(TelemetryConfig::enabled());
        let mut state = AgentState::new();
        state.set_context(CTX_TENANT_ID, "tenant-42".to_string());
        state.set_context(CTX_AGENT_ID, "agent-7".to_string());
        state.set_context(CTX_RUNTIME_TYPE, "GPU".to_string());

        assert!(recorder.record_node_completion(&state, "tool_call", "run-x"));

        let events = recorder.buffer().drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].tenant_id, "tenant-42");
        assert_eq!(events[0].agent_id, "agent-7");
        assert_eq!(events[0].metadata.step_name, "tool_call");
        assert_eq!(events[0].metadata.runtime_type, RuntimeType::Gpu);
    }

    #[test]
    fn falls_back_to_run_id_and_defaults() {
        let recorder = TelemetryRecorder::new(TelemetryConfig::enabled());
        let state = AgentState::new();
        assert!(recorder.record_node_completion(&state, "conditional", "run-fallback"));
        let events = recorder.buffer().drain();
        assert_eq!(events[0].tenant_id, UNKNOWN_TENANT);
        assert_eq!(events[0].agent_id, "run-fallback");
        assert_eq!(events[0].metadata.runtime_type, RuntimeType::Cpu);
    }
}
