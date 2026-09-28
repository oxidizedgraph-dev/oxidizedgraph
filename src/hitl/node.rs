//! HITL checkpoint and resume nodes.

use async_trait::async_trait;

use crate::error::NodeError;
use crate::graph::{NodeExecutor, NodeOutput};
use crate::guardrails::RiskLevel;
use crate::state::SharedState;

use super::explain::ReviewSummaryBuilder;
use super::policy::{ApprovalAction, ApprovalPolicy};
use super::types::{
    append_approval_event, ApprovalDecision, ApprovalEvent, ApprovalRequest, ApprovalStatus,
    CTX_APPROVAL_DECISION, CTX_APPROVAL_REQUEST, CTX_EXPLANATION, CTX_HITL_PAUSED,
};

/// Pauses execution when policy requires human approval for the current risk level.
#[derive(Clone, Debug)]
pub struct ApprovalCheckpointNode {
    id: String,
    policy: ApprovalPolicy,
}

impl ApprovalCheckpointNode {
    /// Create a checkpoint with default policy.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            policy: ApprovalPolicy::default(),
        }
    }

    /// Override approval policy.
    pub fn with_policy(mut self, policy: ApprovalPolicy) -> Self {
        self.policy = policy;
        self
    }
}

#[async_trait]
impl NodeExecutor for ApprovalCheckpointNode {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, state: SharedState) -> Result<NodeOutput, NodeError> {
        let mut guard = state
            .write()
            .map_err(|e| NodeError::execution_failed(e.to_string()))?;

        // If the operator already decided *this* request, route accordingly.
        // The decision must answer the currently pending request — otherwise a
        // stale decision left over from an earlier checkpoint would silently
        // auto-approve this one (a fail-open HITL bypass). Approve/deny outcomes
        // are consumed so they cannot leak to a later checkpoint.
        if let Some(decision) = guard.get_context::<ApprovalDecision>(CTX_APPROVAL_DECISION) {
            let pending_id = guard
                .get_context::<ApprovalRequest>(CTX_APPROVAL_REQUEST)
                .map(|r| r.id);
            if pending_id.as_deref() == Some(decision.request_id.as_str()) {
                match decision.status {
                    ApprovalStatus::Approved => {
                        guard.remove_context(CTX_APPROVAL_DECISION);
                        guard.remove_context(CTX_APPROVAL_REQUEST);
                        guard.set_context(CTX_HITL_PAUSED, false);
                        return Ok(NodeOutput::transition("approved"));
                    }
                    ApprovalStatus::Denied => {
                        guard.remove_context(CTX_APPROVAL_DECISION);
                        guard.remove_context(CTX_APPROVAL_REQUEST);
                        guard.set_context(CTX_HITL_PAUSED, false);
                        return Ok(NodeOutput::transition("denied"));
                    }
                    // Still awaiting the operator — leave request/decision in place.
                    ApprovalStatus::Pending => {
                        return Ok(NodeOutput::transition("awaiting_approval"));
                    }
                }
            }
            // Leftover decision from a different request: discard and re-evaluate
            // policy fresh rather than honoring it.
            guard.remove_context(CTX_APPROVAL_DECISION);
        }

        let risk: RiskLevel = guard
            .get_context("change_risk_level")
            .unwrap_or(RiskLevel::Medium);

        let action = self.policy.matrix.action_for(risk);
        match action {
            ApprovalAction::Allow => Ok(NodeOutput::transition("approved")),
            ApprovalAction::Deny => {
                let request = ApprovalRequest::new("Policy denied action", risk).with_summary(
                    guard
                        .get_context::<String>("change_summary")
                        .unwrap_or_default(),
                );
                append_approval_event(&mut guard, ApprovalEvent::checkpoint_created(&request));
                guard.set_context(CTX_APPROVAL_REQUEST, request);
                guard.set_context(CTX_HITL_PAUSED, true);
                Ok(NodeOutput::transition("denied"))
            }
            ApprovalAction::Pause => {
                let summary = guard
                    .get_context::<String>("change_summary")
                    .unwrap_or_else(|| "Autonomous change pending review".to_string());
                let request =
                    ApprovalRequest::new(format!("{risk:?} change requires human approval"), risk)
                        .with_summary(&summary);

                let summarizer = ReviewSummaryBuilder::new();
                let review = summarizer
                    .from_state(&guard, format!("Risk level {risk:?} matched pause policy"));
                let explanation = summarizer.to_explanation(&review);
                append_approval_event(&mut guard, ApprovalEvent::checkpoint_created(&request));
                guard.set_context(CTX_APPROVAL_REQUEST, request);
                guard.set_context(CTX_EXPLANATION, explanation);
                guard.set_context(CTX_HITL_PAUSED, true);
                Ok(NodeOutput::transition("awaiting_approval"))
            }
        }
    }

    fn description(&self) -> Option<&str> {
        Some("Evaluates risk policy and pauses for human approval when required")
    }
}

/// Records an operator approval and clears the pause flag.
#[derive(Clone, Debug)]
pub struct GrantApprovalNode {
    id: String,
    approver: String,
    rationale: Option<String>,
}

impl GrantApprovalNode {
    /// Approve the pending request as the given operator.
    pub fn new(id: impl Into<String>, approver: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            approver: approver.into(),
            rationale: None,
        }
    }

    /// Attach operator rationale.
    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = Some(rationale.into());
        self
    }
}

#[async_trait]
impl NodeExecutor for GrantApprovalNode {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, state: SharedState) -> Result<NodeOutput, NodeError> {
        let mut guard = state
            .write()
            .map_err(|e| NodeError::execution_failed(e.to_string()))?;

        let request: ApprovalRequest =
            guard.get_context(CTX_APPROVAL_REQUEST).ok_or_else(|| {
                NodeError::execution_failed("no approval_request in context".to_string())
            })?;

        let mut decision = ApprovalDecision::approve(&request.id, &self.approver);
        decision.rationale = self.rationale.clone();

        append_approval_event(&mut guard, ApprovalEvent::decision_recorded(&decision));
        guard.set_context(CTX_APPROVAL_DECISION, decision);
        guard.set_context(CTX_HITL_PAUSED, false);

        Ok(NodeOutput::cont())
    }

    fn description(&self) -> Option<&str> {
        Some("Records operator approval and resumes the workflow")
    }
}

/// Applies queued operator edits before resuming execution.
#[derive(Clone, Debug, Default)]
pub struct EditInterventionNode {
    id: String,
}

impl EditInterventionNode {
    /// Create an edit intervention node.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

#[async_trait]
impl NodeExecutor for EditInterventionNode {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, state: SharedState) -> Result<NodeOutput, NodeError> {
        let mut guard = state
            .write()
            .map_err(|e| NodeError::execution_failed(e.to_string()))?;

        super::intervention::HitlController::apply_edits(&mut guard);
        Ok(NodeOutput::cont())
    }

    fn description(&self) -> Option<&str> {
        Some("Applies operator context edits queued during HITL pause")
    }
}

/// Clears pause state so execution can continue after external intervention.
#[derive(Clone, Debug)]
pub struct ResumeNode {
    id: String,
}

impl ResumeNode {
    /// Create a resume node.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

#[async_trait]
impl NodeExecutor for ResumeNode {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, state: SharedState) -> Result<NodeOutput, NodeError> {
        let mut guard = state
            .write()
            .map_err(|e| NodeError::execution_failed(e.to_string()))?;
        guard.set_context(CTX_HITL_PAUSED, false);
        Ok(NodeOutput::cont())
    }

    fn description(&self) -> Option<&str> {
        Some("Clears HITL pause flag after operator intervention")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AgentState;
    use std::sync::{Arc, RwLock};

    #[tokio::test]
    async fn checkpoint_pauses_on_high_risk() {
        let node = ApprovalCheckpointNode::new("checkpoint");
        let mut state = AgentState::new();
        state.set_context("change_risk_level", RiskLevel::High);
        state.set_context("change_summary", "Refactor auth module");
        let shared = Arc::new(RwLock::new(state));

        let output = node.execute(shared.clone()).await.unwrap();
        assert_eq!(output.target(), Some("awaiting_approval"));

        let guard = shared.read().unwrap();
        assert!(guard.get_context::<bool>(CTX_HITL_PAUSED).unwrap());
        assert!(guard
            .get_context::<ApprovalRequest>(CTX_APPROVAL_REQUEST)
            .is_some());
    }

    #[tokio::test]
    async fn stale_approval_decision_does_not_bypass_a_new_checkpoint() {
        let node = ApprovalCheckpointNode::new("cp");
        let mut state = AgentState::new();
        // A leftover Approved decision answering a DIFFERENT, older request, with
        // no matching pending request in context.
        state.set_context(
            CTX_APPROVAL_DECISION,
            ApprovalDecision::approve("old-request-id", "operator"),
        );
        state.set_context("change_risk_level", RiskLevel::High);
        let shared = Arc::new(RwLock::new(state));

        node.execute(shared.clone()).await.unwrap();

        let guard = shared.read().unwrap();
        // Must NOT auto-approve on the stale decision: high risk re-evaluates to
        // a pause, so paused is set and a fresh request is created.
        assert!(guard.get_context::<bool>(CTX_HITL_PAUSED).unwrap_or(false));
        assert!(guard
            .get_context::<ApprovalRequest>(CTX_APPROVAL_REQUEST)
            .is_some());
        // The stale decision was discarded, not honored.
        assert!(guard
            .get_context::<ApprovalDecision>(CTX_APPROVAL_DECISION)
            .is_none());
    }

    #[tokio::test]
    async fn grant_approval_records_decision() {
        let checkpoint = ApprovalCheckpointNode::new("checkpoint");
        let grant = GrantApprovalNode::new("grant", "operator@example.com");
        let mut state = AgentState::new();
        state.set_context("change_risk_level", RiskLevel::High);
        let shared = Arc::new(RwLock::new(state));

        checkpoint.execute(shared.clone()).await.unwrap();
        grant.execute(shared.clone()).await.unwrap();

        let guard = shared.read().unwrap();
        let decision = guard
            .get_context::<ApprovalDecision>(CTX_APPROVAL_DECISION)
            .unwrap();
        assert_eq!(decision.status, ApprovalStatus::Approved);
        assert!(!guard.get_context::<bool>(CTX_HITL_PAUSED).unwrap());
    }
}
