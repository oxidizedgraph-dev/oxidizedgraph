//! Brains (memory/cognition) client.
//!
//! Compiles a ranked **context pack** for a goal by calling the brains-server
//! `POST /v1/context-pack` endpoint — the single agent-facing cognition route
//! (brains-server exposes no store/index endpoint of its own; writes go to the
//! data-mesh directly).
//!
//! Env-driven, so the deployment binds the endpoint + credential (mirroring the
//! data-mesh checkpointer, #27):
//! - `BRAINS_URL` — base URL, e.g.
//!   `http://brains-server.brains.svc.cluster.local:8090`. When unset, the
//!   client is disabled ([`BrainsClient::from_env`] returns `None`).
//! - `BRAINS_TOKEN` — agent-idp-issued JWT (audience `lornu-ai-agents`) sent as
//!   `Authorization: Bearer`. brains-server validates it against agent-idp's
//!   JWKS; there is no static shared secret.
//! - `BRAINS_BRAIN_ID` — brain identifier (default `lornu-default`).
//! - `BRAINS_BACKEND` — server-side retrieval-plane selector (`mock`/`data-mesh`/
//!   `dgx`); informational to this client, logged by the server.

use crate::error::RuntimeError;
use serde::{Deserialize, Serialize};

const DEFAULT_BRAIN_ID: &str = "lornu-default";
const DEFAULT_TOKEN_BUDGET: u32 = 4096;

/// Client for the brains-server cognition API.
#[derive(Debug, Clone)]
pub struct BrainsClient {
    http: reqwest::Client,
    base_url: String,
    token: Option<String>,
    brain_id: String,
}

/// Action risk level (brains `risk_level`). Serialized snake_case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    /// Read-only action.
    Read,
    /// Mutating action.
    Write,
    /// Irreversible / destructive action.
    Destructive,
}

/// Retrieval knobs (brains `RetrievalOptions`). All fields are sent explicitly —
/// the server struct declares no serde defaults.
#[derive(Debug, Clone, Serialize)]
pub struct RetrievalOptions {
    /// Number of memories to retrieve.
    pub top_k: u32,
    /// Include stale (superseded) memories.
    pub include_stale: bool,
    /// Include memories flagged unsafe.
    pub include_unsafe: bool,
    /// Include memories with unresolved conflicts.
    pub include_conflicted: bool,
    /// Include code-graph nodes.
    pub include_code_graph: bool,
    /// Include experience cards.
    pub include_experience_cards: bool,
    /// Include stored procedures.
    pub include_procedures: bool,
}

impl Default for RetrievalOptions {
    fn default() -> Self {
        Self {
            top_k: 3,
            include_stale: false,
            include_unsafe: false,
            include_conflicted: false,
            include_code_graph: false,
            include_experience_cards: false,
            include_procedures: false,
        }
    }
}

/// Body of `POST /v1/context-pack`.
#[derive(Debug, Clone, Serialize)]
pub struct ContextPackRequest {
    /// Brain identifier.
    pub brain_id: String,
    /// Repository the goal is scoped to.
    pub repo: String,
    /// Conversation/thread id.
    pub thread_id: String,
    /// Optional run id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Optional task id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Natural-language goal to compile context for.
    pub goal: String,
    /// Risk level of the intended action.
    pub risk_level: RiskLevel,
    /// Token budget for the compiled pack.
    pub token_budget: u32,
    /// Retrieval knobs.
    pub retrieval: RetrievalOptions,
}

/// A ranked memory item in a compiled context pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedMemory {
    /// Memory id.
    pub id: String,
    /// Human-readable summary of the memory.
    #[serde(default)]
    pub summary: String,
    /// Relevance score.
    #[serde(default)]
    pub score: f64,
    /// Evidence-graph node ids supporting this memory.
    #[serde(default)]
    pub evidence_path: Vec<String>,
}

/// Policy decision block returned with a context pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyBlock {
    /// Decision: `allow` | `deny` | `escalate`.
    pub decision: String,
    /// Risk level the decision was made against.
    pub risk_level: RiskLevel,
    /// Constraints attached to an allow/escalate decision.
    #[serde(default)]
    pub constraints: Vec<String>,
}

/// Response of `POST /v1/context-pack` (`CompiledContextPack`). The
/// `evidence_graph` / `aivcs` sub-documents are passed through opaquely.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledContextPack {
    /// Query id assigned by brains.
    pub query_id: String,
    /// Tokens consumed by the pack (`<= token_budget`).
    #[serde(default)]
    pub used_tokens: u32,
    /// Ranked memories included in the pack.
    #[serde(default)]
    pub memories: Vec<RankedMemory>,
    /// Policy decision for the intended action.
    pub policy: PolicyBlock,
    /// Evidence graph (passed through opaquely).
    #[serde(default)]
    pub evidence_graph: serde_json::Value,
    /// AIVCS provenance refs (passed through opaquely).
    #[serde(default)]
    pub aivcs: serde_json::Value,
}

impl BrainsClient {
    /// Build from the environment. Returns `None` when `BRAINS_URL` is unset or
    /// empty (brains disabled — never a silent local fallback to a wrong host).
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("BRAINS_URL").ok().filter(|s| !s.is_empty())?;
        Some(Self::with_config(
            base_url,
            std::env::var("BRAINS_TOKEN").ok().filter(|s| !s.is_empty()),
            std::env::var("BRAINS_BRAIN_ID")
                .ok()
                .filter(|s| !s.is_empty()),
        ))
    }

    /// Build from an explicit endpoint (custom wiring / tests).
    pub fn with_config(
        base_url: impl Into<String>,
        token: Option<String>,
        brain_id: Option<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token,
            brain_id: brain_id.unwrap_or_else(|| DEFAULT_BRAIN_ID.to_string()),
        }
    }

    /// The configured brain id.
    pub fn brain_id(&self) -> &str {
        &self.brain_id
    }

    /// Compile a ranked context pack (`POST /v1/context-pack`).
    pub async fn context_pack(
        &self,
        req: &ContextPackRequest,
    ) -> Result<CompiledContextPack, RuntimeError> {
        let url = format!("{}/v1/context-pack", self.base_url);
        let mut builder = self.http.post(&url).json(req);
        if let Some(token) = &self.token {
            builder = builder.bearer_auth(token);
        }
        let resp = builder.send().await.map_err(|e| Self::err("request", e))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(RuntimeError::InvalidState(format!(
                "brains context-pack {status}: {}",
                body.trim()
            )));
        }
        resp.json::<CompiledContextPack>()
            .await
            .map_err(|e| Self::err("decode", e))
    }

    /// Convenience: compile a pack for a `goal` in a `repo`/`thread`, with
    /// default retrieval options.
    pub async fn compile_goal(
        &self,
        repo: &str,
        thread_id: &str,
        goal: &str,
        risk_level: RiskLevel,
        token_budget: Option<u32>,
        top_k: Option<u32>,
    ) -> Result<CompiledContextPack, RuntimeError> {
        let retrieval = RetrievalOptions {
            top_k: top_k.unwrap_or(RetrievalOptions::default().top_k),
            ..RetrievalOptions::default()
        };
        self.context_pack(&ContextPackRequest {
            brain_id: self.brain_id.clone(),
            repo: repo.to_string(),
            thread_id: thread_id.to_string(),
            run_id: None,
            task_id: None,
            goal: goal.to_string(),
            risk_level,
            token_budget: token_budget.unwrap_or(DEFAULT_TOKEN_BUDGET),
            retrieval,
        })
        .await
    }

    fn err(context: &str, e: impl std::fmt::Display) -> RuntimeError {
        RuntimeError::InvalidState(format!("brains {context}: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> ContextPackRequest {
        ContextPackRequest {
            brain_id: "lornu-default".to_string(),
            repo: "lornu-ai/oxidizedgraph".to_string(),
            thread_id: "thread-1".to_string(),
            run_id: None,
            task_id: None,
            goal: "Fix the PR pipeline".to_string(),
            risk_level: RiskLevel::Write,
            token_budget: 4096,
            retrieval: RetrievalOptions::default(),
        }
    }

    #[test]
    fn request_serializes_to_the_brains_contract() {
        let v = serde_json::to_value(sample_request()).unwrap();
        assert_eq!(v["brain_id"], "lornu-default");
        assert_eq!(v["repo"], "lornu-ai/oxidizedgraph");
        assert_eq!(v["thread_id"], "thread-1");
        assert_eq!(v["goal"], "Fix the PR pipeline");
        assert_eq!(v["risk_level"], "write");
        assert_eq!(v["token_budget"], 4096);
        // Optional ids are omitted, not null.
        assert!(v.get("run_id").is_none());
        assert!(v.get("task_id").is_none());
        // All seven retrieval knobs are present (server has no serde defaults).
        let r = &v["retrieval"];
        assert_eq!(r["top_k"], 3);
        for k in [
            "include_stale",
            "include_unsafe",
            "include_conflicted",
            "include_code_graph",
            "include_experience_cards",
            "include_procedures",
        ] {
            assert!(r.get(k).is_some(), "missing retrieval.{k}");
        }
    }

    #[test]
    fn response_deserializes_from_the_brains_contract() {
        let json = serde_json::json!({
            "query_id": "ctx_local:lornu-default",
            "used_tokens": 128,
            "memories": [{ "id": "mem-1", "summary": "s", "score": 0.87, "evidence_path": ["n1"] }],
            "policy": { "decision": "allow", "risk_level": "write", "constraints": [] }
        });
        let pack: CompiledContextPack = serde_json::from_value(json).unwrap();
        assert_eq!(pack.query_id, "ctx_local:lornu-default");
        assert_eq!(pack.used_tokens, 128);
        assert_eq!(pack.memories.len(), 1);
        assert_eq!(pack.memories[0].id, "mem-1");
        assert_eq!(pack.policy.decision, "allow");
        assert_eq!(pack.policy.risk_level, RiskLevel::Write);
    }

    #[test]
    fn risk_level_is_snake_case() {
        assert_eq!(
            serde_json::to_value(RiskLevel::Destructive).unwrap(),
            "destructive"
        );
        assert_eq!(serde_json::to_value(RiskLevel::Read).unwrap(), "read");
    }

    #[test]
    fn from_env_disabled_without_url() {
        std::env::remove_var("BRAINS_URL");
        assert!(BrainsClient::from_env().is_none());
    }

    #[test]
    fn with_config_trims_trailing_slash_and_defaults_brain_id() {
        let c = BrainsClient::with_config("http://brains:8090/", None, None);
        assert_eq!(c.base_url, "http://brains:8090");
        assert_eq!(c.brain_id(), "lornu-default");
    }
}
