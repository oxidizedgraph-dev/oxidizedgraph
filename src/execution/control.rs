//! Execution control for signal-driven DAG halting (oxidizedgraph #10).
//!
//! Provides an in-memory run registry, cooperative cancellation via
//! [`CancellationToken`](tokio_util::sync::CancellationToken), and HTTP-facing
//! request/response types for halt, terminate, resume, and quarantine.
#![allow(missing_docs)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

/// How safely a node may be interrupted by an external halt request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stoppability {
    /// Safe to pause at the next node boundary.
    #[default]
    Stoppable,
    /// Finish the current unit of work, then stop.
    Drainable,
    /// Must not be auto-halted; requires HITL approval.
    Critical,
}

impl Stoppability {
    /// Whether signal-driven halt is permitted without operator approval.
    pub fn allows_auto_halt(self) -> bool {
        !matches!(self, Self::Critical)
    }
}

/// Lifecycle status of a tracked execution run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    /// Actively traversing the graph.
    Running,
    /// Halt requested; waiting for cooperative stop.
    Halting,
    /// Stopped cooperatively with a checkpoint.
    Halted,
    /// Force-stopped.
    Terminated,
    /// Completed successfully.
    Completed,
    /// Failed with an error.
    Failed,
}

impl ExecutionStatus {
    /// Whether the run has reached a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Halted | Self::Terminated | Self::Completed | Self::Failed
        )
    }
}

/// Graceful vs force stop strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HaltStrategy {
    #[default]
    Graceful,
    Force,
}

/// Request body for `POST /v1/executions/{run_id}/halt`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HaltRequest {
    /// Human-readable reason (e.g. `swarm_signal:abc-123`).
    pub reason: String,
    /// Originating swarm signal id, when applicable.
    #[serde(default)]
    pub signal_id: Option<String>,
    /// Stop strategy (default: graceful).
    #[serde(default)]
    pub strategy: HaltStrategy,
    /// Graceful timeout in milliseconds (default: 30_000).
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    /// Graph nodes implicated by the signal.
    #[serde(default)]
    pub affected_nodes: Vec<String>,
    /// Caller identity (e.g. `agent-sensing`).
    #[serde(default)]
    pub requested_by: Option<String>,
}

fn default_timeout_ms() -> u64 {
    30_000
}

/// Response for halt requests.
#[derive(Clone, Debug, Serialize)]
pub struct HaltResponse {
    pub run_id: String,
    pub status: ExecutionStatus,
    pub strategy: HaltStrategy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<String>,
}

/// Request body for `POST /v1/executions/{run_id}/terminate`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TerminateRequest {
    pub reason: String,
    #[serde(default)]
    pub signal_id: Option<String>,
    #[serde(default)]
    pub requested_by: Option<String>,
}

/// Response for terminate requests.
#[derive(Clone, Debug, Serialize)]
pub struct TerminateResponse {
    pub run_id: String,
    pub status: ExecutionStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<String>,
    pub terminated_at: DateTime<Utc>,
}

/// Public execution view returned by `GET /v1/executions/{run_id}`.
#[derive(Clone, Debug, Serialize)]
pub struct ExecutionView {
    pub run_id: String,
    pub graph_name: String,
    pub status: ExecutionStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_node: Option<String>,
    pub stoppability: Stoppability,
    pub quarantined: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<String>,
    pub signal_ids: Vec<String>,
}

/// Request body for `POST /v1/executions/{run_id}/resume`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResumeRequest {
    #[serde(default)]
    pub checkpoint_id: Option<String>,
    #[serde(default)]
    pub clear_quarantine: bool,
    #[serde(default)]
    pub approved_by: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Response for resume requests.
#[derive(Clone, Debug, Serialize)]
pub struct ResumeResponse {
    pub run_id: String,
    pub status: ExecutionStatus,
    pub resumed_at: DateTime<Utc>,
}

/// Request body for `POST /v1/executions/{run_id}/quarantine`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QuarantineRequest {
    pub signal_id: String,
    #[serde(default)]
    pub affected_nodes: Vec<String>,
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
}

/// Errors from execution control operations.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ControlError {
    #[error("execution run not found: {0}")]
    NotFound(String),
    #[error("run is not in a haltable state: {0}")]
    NotHaltable(String),
    #[error("node stoppability is critical — operator approval required")]
    CriticalNodeBlocked,
    #[error("duplicate halt for signal {signal_id} on run {run_id}")]
    DuplicateHalt { run_id: String, signal_id: String },
    #[error("run is not resumable: {0}")]
    NotResumable(String),
}

/// Internal record for a tracked run.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ExecutionRecord {
    pub run_id: String,
    pub graph_name: String,
    pub status: ExecutionStatus,
    pub current_node: Option<String>,
    pub stoppability: Stoppability,
    pub quarantined: bool,
    pub checkpoint_id: Option<String>,
    pub signal_ids: Vec<String>,
    pub worker_job: Option<String>,
    pub task_id: Option<String>,
    pub session_id: Option<String>,
    pub halt_deadline: Option<DateTime<Utc>>,
    pub cancel_token: CancellationToken,
}

impl ExecutionRecord {
    fn new(
        run_id: impl Into<String>,
        graph_name: impl Into<String>,
        stoppability: Stoppability,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            graph_name: graph_name.into(),
            status: ExecutionStatus::Running,
            current_node: None,
            stoppability,
            quarantined: false,
            checkpoint_id: None,
            signal_ids: Vec::new(),
            worker_job: None,
            task_id: None,
            session_id: None,
            halt_deadline: None,
            cancel_token: CancellationToken::new(),
        }
    }

    /// Build the public API view.
    pub fn to_view(&self) -> ExecutionView {
        ExecutionView {
            run_id: self.run_id.clone(),
            graph_name: self.graph_name.clone(),
            status: self.status,
            current_node: self.current_node.clone(),
            stoppability: self.stoppability,
            quarantined: self.quarantined,
            checkpoint_id: self.checkpoint_id.clone(),
            signal_ids: self.signal_ids.clone(),
        }
    }
}

/// In-memory registry of active and recently halted runs.
#[derive(Clone, Default)]
pub struct ExecutionRegistry {
    runs: Arc<RwLock<HashMap<String, ExecutionRecord>>>,
    task_index: Arc<RwLock<HashMap<String, String>>>,
}

impl ExecutionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new run and return its cancellation token.
    pub async fn register(
        &self,
        run_id: impl Into<String>,
        graph_name: impl Into<String>,
        stoppability: Stoppability,
    ) -> CancellationToken {
        let run_id = run_id.into();
        let record = ExecutionRecord::new(&run_id, graph_name, stoppability);
        let token = record.cancel_token.clone();
        self.runs.write().await.insert(run_id, record);
        token
    }

    /// Link an A2A task id to a run id.
    pub async fn link_task(&self, task_id: impl Into<String>, run_id: impl Into<String>) {
        self.task_index
            .write()
            .await
            .insert(task_id.into(), run_id.into());
    }

    /// Attach worker Job metadata to a run.
    pub async fn set_worker_job(&self, run_id: &str, job_name: impl Into<String>) {
        if let Some(record) = self.runs.write().await.get_mut(run_id) {
            record.worker_job = Some(job_name.into());
        }
    }

    /// Attach session id to a run.
    pub async fn set_session_id(&self, run_id: &str, session_id: impl Into<String>) {
        if let Some(record) = self.runs.write().await.get_mut(run_id) {
            record.session_id = Some(session_id.into());
        }
    }

    /// Update the node currently executing.
    pub async fn set_current_node(&self, run_id: &str, node_id: impl Into<String>) {
        if let Some(record) = self.runs.write().await.get_mut(run_id) {
            record.current_node = Some(node_id.into());
        }
    }

    /// Mark a run completed and remove it from the active registry.
    pub async fn complete(&self, run_id: &str, checkpoint_id: Option<String>) {
        let mut runs = self.runs.write().await;
        if let Some(record) = runs.get_mut(run_id) {
            record.status = ExecutionStatus::Completed;
            record.checkpoint_id = checkpoint_id;
        }
    }

    /// Mark a run failed.
    pub async fn fail(&self, run_id: &str, reason: &str) {
        warn!(run_id, reason, "execution failed");
        if let Some(record) = self.runs.write().await.get_mut(run_id) {
            record.status = ExecutionStatus::Failed;
        }
    }

    /// Fetch a run by id.
    pub async fn get(&self, run_id: &str) -> Option<ExecutionView> {
        self.runs.read().await.get(run_id).map(|r| r.to_view())
    }

    /// Resolve run id from A2A task id.
    pub async fn run_id_for_task(&self, task_id: &str) -> Option<String> {
        self.task_index.read().await.get(task_id).cloned()
    }

    /// Request cooperative halt.
    pub async fn halt(&self, run_id: &str, req: HaltRequest) -> Result<HaltResponse, ControlError> {
        let mut runs = self.runs.write().await;
        let record = runs
            .get_mut(run_id)
            .ok_or_else(|| ControlError::NotFound(run_id.to_string()))?;

        if !record.stoppability.allows_auto_halt() {
            return Err(ControlError::CriticalNodeBlocked);
        }

        // Reject terminal runs BEFORE recording the signal — otherwise a
        // rejected halt against a Completed/Terminated run still mutates
        // signal_ids, misrepresenting which signals actually acted on the run.
        if record.status.is_terminal() {
            return Err(ControlError::NotHaltable(format!(
                "status is {:?}",
                record.status
            )));
        }

        if let Some(ref signal_id) = req.signal_id {
            if record.signal_ids.contains(signal_id)
                && matches!(
                    record.status,
                    ExecutionStatus::Halting | ExecutionStatus::Halted
                )
            {
                return Err(ControlError::DuplicateHalt {
                    run_id: run_id.to_string(),
                    signal_id: signal_id.clone(),
                });
            }
            if !record.signal_ids.contains(signal_id) {
                record.signal_ids.push(signal_id.clone());
            }
        }

        let deadline = Utc::now() + chrono::Duration::milliseconds(req.timeout_ms as i64);
        record.status = ExecutionStatus::Halting;
        record.halt_deadline = Some(deadline);
        record.quarantined = true;
        record.cancel_token.cancel();

        info!(
            run_id,
            reason = %req.reason,
            strategy = ?req.strategy,
            "halt requested"
        );

        Ok(HaltResponse {
            run_id: run_id.to_string(),
            status: ExecutionStatus::Halting,
            strategy: req.strategy,
            deadline_at: Some(deadline),
            checkpoint_id: record.checkpoint_id.clone(),
        })
    }

    /// Force-terminate a run.
    pub async fn terminate(
        &self,
        run_id: &str,
        req: TerminateRequest,
    ) -> Result<TerminateResponse, ControlError> {
        let mut runs = self.runs.write().await;
        let record = runs
            .get_mut(run_id)
            .ok_or_else(|| ControlError::NotFound(run_id.to_string()))?;

        // Never force a run that has already reached a terminal state (e.g. a
        // successfully `Completed` run) into `Terminated` — that would overwrite
        // a real result. Mirrors the guard in `halt`. Checked BEFORE recording
        // the signal so a rejected terminate doesn't mutate signal_ids.
        if record.status.is_terminal() {
            return Err(ControlError::NotHaltable(format!(
                "cannot terminate run in terminal status {:?}",
                record.status
            )));
        }

        if let Some(ref signal_id) = req.signal_id {
            if !record.signal_ids.contains(signal_id) {
                record.signal_ids.push(signal_id.clone());
            }
        }

        record.status = ExecutionStatus::Terminated;
        record.quarantined = true;
        record.cancel_token.cancel();

        let checkpoint_id = record.checkpoint_id.clone().or_else(|| {
            let id = format!("ckpt-{}", &Uuid::new_v4().to_string()[..8]);
            record.checkpoint_id = Some(id.clone());
            Some(id)
        });

        info!(run_id, reason = %req.reason, "execution terminated");

        Ok(TerminateResponse {
            run_id: run_id.to_string(),
            status: ExecutionStatus::Terminated,
            checkpoint_id,
            terminated_at: Utc::now(),
        })
    }

    /// Terminate by A2A task id (used by `CancelTask`).
    pub async fn terminate_by_task(
        &self,
        task_id: &str,
        reason: &str,
    ) -> Result<Option<TerminateResponse>, ControlError> {
        let run_id = self.run_id_for_task(task_id).await;
        match run_id {
            Some(id) => {
                let resp = self
                    .terminate(
                        &id,
                        TerminateRequest {
                            reason: reason.to_string(),
                            signal_id: None,
                            requested_by: Some("a2a:CancelTask".into()),
                        },
                    )
                    .await?;
                Ok(Some(resp))
            }
            None => Ok(None),
        }
    }

    /// Worker job name for a run (for K8s deletion).
    pub async fn worker_job_for_run(&self, run_id: &str) -> Option<String> {
        self.runs
            .read()
            .await
            .get(run_id)
            .and_then(|r| r.worker_job.clone())
    }

    /// Mark halt complete (called when runner observes cancellation).
    pub async fn mark_halted(&self, run_id: &str, checkpoint_id: Option<String>) {
        if let Some(record) = self.runs.write().await.get_mut(run_id) {
            record.status = ExecutionStatus::Halted;
            if checkpoint_id.is_some() {
                record.checkpoint_id = checkpoint_id;
            }
        }
    }

    /// Apply quarantine tags without changing run status.
    pub async fn quarantine(
        &self,
        run_id: &str,
        req: QuarantineRequest,
    ) -> Result<ExecutionView, ControlError> {
        let mut runs = self.runs.write().await;
        let record = runs
            .get_mut(run_id)
            .ok_or_else(|| ControlError::NotFound(run_id.to_string()))?;

        if !record.signal_ids.contains(&req.signal_id) {
            record.signal_ids.push(req.signal_id);
        }
        record.quarantined = true;

        Ok(record.to_view())
    }

    /// Resume a halted run (clears quarantine when requested).
    pub async fn resume(
        &self,
        run_id: &str,
        req: ResumeRequest,
    ) -> Result<ResumeResponse, ControlError> {
        let mut runs = self.runs.write().await;
        let record = runs
            .get_mut(run_id)
            .ok_or_else(|| ControlError::NotFound(run_id.to_string()))?;

        if !matches!(
            record.status,
            ExecutionStatus::Halted | ExecutionStatus::Terminated
        ) {
            return Err(ControlError::NotResumable(format!(
                "status is {:?}",
                record.status
            )));
        }

        if req.clear_quarantine {
            record.quarantined = false;
        }
        record.status = ExecutionStatus::Running;
        record.halt_deadline = None;
        // A CancellationToken is permanently cancelled once `cancel()` is called
        // (halt/terminate do so); reusing it would make the resumed runner return
        // `Cancelled` on its very first loop check without executing any node.
        // Hand the resumed run a fresh token so it can actually proceed.
        record.cancel_token = CancellationToken::new();

        info!(
            run_id,
            approved_by = ?req.approved_by,
            reason = ?req.reason,
            "execution resumed"
        );

        Ok(ResumeResponse {
            run_id: run_id.to_string(),
            status: ExecutionStatus::Running,
            resumed_at: Utc::now(),
        })
    }

    /// Access the cancellation token for a run.
    pub async fn cancel_token(&self, run_id: &str) -> Option<CancellationToken> {
        self.runs
            .read()
            .await
            .get(run_id)
            .map(|r| r.cancel_token.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn halt_and_terminate_flow() {
        let registry = ExecutionRegistry::new();
        let run_id = "run-test-1";
        registry
            .register(run_id, "ci-remediation", Stoppability::Stoppable)
            .await;

        let halt = registry
            .halt(
                run_id,
                HaltRequest {
                    reason: "swarm_signal:sig-1".into(),
                    signal_id: Some("sig-1".into()),
                    strategy: HaltStrategy::Graceful,
                    timeout_ms: 30_000,
                    affected_nodes: vec!["lint".into()],
                    requested_by: Some("agent-sensing".into()),
                },
            )
            .await
            .unwrap();

        assert_eq!(halt.status, ExecutionStatus::Halting);

        let view = registry.get(run_id).await.unwrap();
        assert!(view.quarantined);
        assert_eq!(view.signal_ids, vec!["sig-1"]);

        let term = registry
            .terminate(
                run_id,
                TerminateRequest {
                    reason: "timeout".into(),
                    signal_id: Some("sig-1".into()),
                    requested_by: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(term.status, ExecutionStatus::Terminated);
        assert!(term.checkpoint_id.is_some());
    }

    #[tokio::test]
    async fn terminate_rejects_a_terminal_run() {
        let registry = ExecutionRegistry::new();
        let run_id = "run-done";
        registry
            .register(run_id, "job", Stoppability::Stoppable)
            .await;
        registry.complete(run_id, None).await;
        let res = registry
            .terminate(
                run_id,
                TerminateRequest {
                    reason: "late cancel".into(),
                    signal_id: None,
                    requested_by: None,
                },
            )
            .await;
        assert!(matches!(res, Err(ControlError::NotHaltable(_))));
        // The completed status must be preserved, not overwritten with Terminated.
        assert_eq!(
            registry.get(run_id).await.unwrap().status,
            ExecutionStatus::Completed
        );
    }

    #[tokio::test]
    async fn critical_node_blocks_auto_halt() {
        let registry = ExecutionRegistry::new();
        let run_id = "run-critical";
        registry
            .register(run_id, "deploy", Stoppability::Critical)
            .await;

        let err = registry
            .halt(
                run_id,
                HaltRequest {
                    reason: "signal".into(),
                    signal_id: None,
                    strategy: HaltStrategy::Graceful,
                    timeout_ms: 30_000,
                    affected_nodes: vec![],
                    requested_by: None,
                },
            )
            .await
            .unwrap_err();

        assert_eq!(err, ControlError::CriticalNodeBlocked);
    }

    #[tokio::test]
    async fn resume_clears_quarantine() {
        let registry = ExecutionRegistry::new();
        let run_id = "run-resume";
        registry
            .register(run_id, "test", Stoppability::Stoppable)
            .await;
        registry.mark_halted(run_id, Some("ckpt-1".into())).await;

        registry
            .quarantine(
                run_id,
                QuarantineRequest {
                    signal_id: "sig-1".into(),
                    affected_nodes: vec![],
                    ttl_seconds: None,
                },
            )
            .await
            .unwrap();

        registry
            .resume(
                run_id,
                ResumeRequest {
                    checkpoint_id: Some("ckpt-1".into()),
                    clear_quarantine: true,
                    approved_by: Some("operator".into()),
                    reason: Some("signal resolved".into()),
                },
            )
            .await
            .unwrap();

        let view = registry.get(run_id).await.unwrap();
        assert!(!view.quarantined);
        assert_eq!(view.status, ExecutionStatus::Running);
    }
}
