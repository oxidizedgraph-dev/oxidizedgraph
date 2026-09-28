//! Telemetry event payloads for usage-based billing (plans#24).
//!
//! The [`TelemetryEvent`] struct is the on-the-wire shape published whenever a
//! DAG agent node finishes executing. The JSON field names and types are a
//! contract with the downstream billing pipeline and must not drift:
//!
//! ```json
//! {
//!   "event_id": "uuid-v4",
//!   "tenant_id": "string",
//!   "agent_id": "string",
//!   "type": "AGENT_INVOCATION",
//!   "value": 1,
//!   "timestamp": "ISO-8601 UTC",
//!   "metadata": { "step_name": "string", "runtime_type": "CPU | GPU" }
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The billing event category. Currently only agent invocations are metered,
/// but this is modelled as an enum so future usage kinds serialize with the
/// same SCREAMING_SNAKE_CASE convention the billing pipeline expects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EventType {
    /// One agent node finished executing (the unit of usage-based billing).
    AgentInvocation,
}

/// Compute class the node ran on. Serializes to the exact `"CPU"` / `"GPU"`
/// strings required by the billing payload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum RuntimeType {
    /// Standard CPU worker (default).
    #[default]
    Cpu,
    /// GPU-accelerated worker (e.g. NVIDIA NIM inference nodes).
    Gpu,
}

impl RuntimeType {
    /// Parse a runtime type from a case-insensitive string, defaulting to
    /// [`RuntimeType::Cpu`] for anything that is not explicitly a GPU marker.
    pub fn from_str_lossy(value: &str) -> Self {
        match value.trim().to_ascii_uppercase().as_str() {
            "GPU" => RuntimeType::Gpu,
            _ => RuntimeType::Cpu,
        }
    }
}

/// Metadata block carried alongside every telemetry event.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryMetadata {
    /// Name of the workflow step / node that finished (e.g. `llm_call`).
    pub step_name: String,
    /// Compute class the node ran on.
    pub runtime_type: RuntimeType,
}

/// A single agent-invocation telemetry record.
///
/// Constructed at the node-completion site and enqueued into the async buffer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Unique identifier for this event (UUID v4).
    pub event_id: String,
    /// Tenant the invocation is billed to.
    pub tenant_id: String,
    /// Agent (graph/run) that produced the invocation.
    pub agent_id: String,
    /// Event category. Always [`EventType::AgentInvocation`] today.
    #[serde(rename = "type")]
    pub event_type: EventType,
    /// Billable units for this event. Always `1` for an invocation.
    pub value: u64,
    /// ISO-8601 UTC timestamp of when the node finished.
    pub timestamp: DateTime<Utc>,
    /// Structured metadata (step name, runtime class).
    pub metadata: TelemetryMetadata,
}

impl TelemetryEvent {
    /// Build an `AGENT_INVOCATION` event with a freshly generated UUID and the
    /// current UTC timestamp.
    pub fn agent_invocation(
        tenant_id: impl Into<String>,
        agent_id: impl Into<String>,
        step_name: impl Into<String>,
        runtime_type: RuntimeType,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4().to_string(),
            tenant_id: tenant_id.into(),
            agent_id: agent_id.into(),
            event_type: EventType::AgentInvocation,
            value: 1,
            timestamp: Utc::now(),
            metadata: TelemetryMetadata {
                step_name: step_name.into(),
                runtime_type,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_with_exact_billing_shape() {
        let event =
            TelemetryEvent::agent_invocation("tenant-1", "agent-9", "llm_call", RuntimeType::Gpu);
        let json = serde_json::to_value(&event).unwrap();

        // Exact field names / types are a contract with the billing pipeline.
        assert!(json["event_id"].is_string());
        assert_eq!(json["tenant_id"], "tenant-1");
        assert_eq!(json["agent_id"], "agent-9");
        assert_eq!(json["type"], "AGENT_INVOCATION");
        assert_eq!(json["value"], 1);
        assert!(json["timestamp"].is_string());
        assert_eq!(json["metadata"]["step_name"], "llm_call");
        assert_eq!(json["metadata"]["runtime_type"], "GPU");
    }

    #[test]
    fn runtime_type_defaults_to_cpu() {
        assert_eq!(RuntimeType::default(), RuntimeType::Cpu);
        assert_eq!(RuntimeType::from_str_lossy("gpu"), RuntimeType::Gpu);
        assert_eq!(RuntimeType::from_str_lossy("  GPU "), RuntimeType::Gpu);
        assert_eq!(RuntimeType::from_str_lossy("anything"), RuntimeType::Cpu);
        assert_eq!(
            serde_json::to_value(RuntimeType::Cpu).unwrap(),
            serde_json::json!("CPU")
        );
    }

    #[test]
    fn round_trips_through_json() {
        let event = TelemetryEvent::agent_invocation("t", "a", "tool_call", RuntimeType::Cpu);
        let json = serde_json::to_string(&event).unwrap();
        let parsed: TelemetryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }
}
