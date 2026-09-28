//! Embedded JSON checkpoint database.
//!
//! Ships inside the oxidizedgraph binary. No Postgres, YSQL, SQLite daemon,
//! or extra crates. The on-disk format is a single JSON document atomically
//! replaced on each write so it behaves the same on Windows, macOS, and Linux.
//!
//! Default path: `$OG_CHECKPOINT_PATH`, else `./.og/state.json`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{Checkpoint, Checkpointer};
use crate::error::RuntimeError;

/// Default relative path when `OG_CHECKPOINT_PATH` is unset.
pub const DEFAULT_CHECKPOINT_PATH: &str = ".og/state.json";

#[derive(Default, Serialize, Deserialize)]
struct Store {
    checkpoints: HashMap<String, Checkpoint>,
    threads: HashMap<String, Vec<String>>,
}

/// File-backed checkpoint database shipped with the engine.
pub struct EmbeddedCheckpointer {
    path: PathBuf,
    store: Mutex<Store>,
}

impl EmbeddedCheckpointer {
    /// Open (or create) the default embedded database.
    pub fn new() -> Result<Self, RuntimeError> {
        Self::open(default_path())
    }

    /// Open from `OG_CHECKPOINT_PATH`, falling back to [`DEFAULT_CHECKPOINT_PATH`].
    pub fn from_env() -> Result<Self, RuntimeError> {
        Self::open(default_path())
    }

    /// Open a database at `path`. Missing files start empty.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, RuntimeError> {
        let path = path.into();
        let store = load_store(&path)?;
        Ok(Self {
            path,
            store: Mutex::new(store),
        })
    }

    /// On-disk path for this database.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn persist(&self, store: &Store) -> Result<(), RuntimeError> {
        let bytes = serde_json::to_vec_pretty(store)
            .map_err(|e| RuntimeError::InvalidState(format!("embedded checkpoint encode: {e}")))?;
        atomic_write(&self.path, &bytes)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Store>, RuntimeError> {
        self.store
            .lock()
            .map_err(|e| RuntimeError::InvalidState(format!("embedded checkpoint lock: {e}")))
    }
}

fn default_path() -> PathBuf {
    std::env::var("OG_CHECKPOINT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_CHECKPOINT_PATH))
}

fn load_store(path: &Path) -> Result<Store, RuntimeError> {
    if !path.exists() {
        return Ok(Store::default());
    }
    let bytes = fs::read(path).map_err(|e| {
        RuntimeError::InvalidState(format!("embedded checkpoint read {}: {e}", path.display()))
    })?;
    if bytes.is_empty() {
        return Ok(Store::default());
    }
    serde_json::from_slice(&bytes).map_err(|e| {
        RuntimeError::InvalidState(format!(
            "embedded checkpoint decode {}: {e}",
            path.display()
        ))
    })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), RuntimeError> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir).map_err(|e| {
                RuntimeError::InvalidState(format!(
                    "embedded checkpoint mkdir {}: {e}",
                    dir.display()
                ))
            })?;
        }
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, bytes).map_err(|e| {
        RuntimeError::InvalidState(format!("embedded checkpoint write {}: {e}", tmp.display()))
    })?;
    if path.exists() {
        fs::remove_file(path).map_err(|e| {
            RuntimeError::InvalidState(format!(
                "embedded checkpoint replace {}: {e}",
                path.display()
            ))
        })?;
    }
    fs::rename(&tmp, path).map_err(|e| {
        RuntimeError::InvalidState(format!(
            "embedded checkpoint rename {} -> {}: {e}",
            tmp.display(),
            path.display()
        ))
    })?;
    Ok(())
}

#[async_trait]
impl Checkpointer for EmbeddedCheckpointer {
    async fn save(&self, checkpoint: Checkpoint) -> Result<(), RuntimeError> {
        let id = checkpoint.id.clone();
        let thread_id = checkpoint.thread_id.clone();
        let mut store = self.lock()?;
        store.checkpoints.insert(id.clone(), checkpoint);
        store.threads.entry(thread_id).or_default().push(id);
        self.persist(&store)
    }

    async fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>, RuntimeError> {
        let store = self.lock()?;
        let Some(ids) = store.threads.get(thread_id) else {
            return Ok(None);
        };
        let latest = ids
            .iter()
            .filter_map(|id| store.checkpoints.get(id))
            .max_by_key(|cp| cp.created_at)
            .cloned();
        Ok(latest)
    }

    async fn load_by_id(&self, checkpoint_id: &str) -> Result<Option<Checkpoint>, RuntimeError> {
        let store = self.lock()?;
        Ok(store.checkpoints.get(checkpoint_id).cloned())
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<Checkpoint>, RuntimeError> {
        let store = self.lock()?;
        let Some(ids) = store.threads.get(thread_id) else {
            return Ok(Vec::new());
        };
        let mut result: Vec<Checkpoint> = ids
            .iter()
            .filter_map(|id| store.checkpoints.get(id).cloned())
            .collect();
        result.sort_by_key(|cp| std::cmp::Reverse(cp.created_at));
        Ok(result)
    }

    async fn delete(&self, checkpoint_id: &str) -> Result<(), RuntimeError> {
        let mut store = self.lock()?;
        if let Some(cp) = store.checkpoints.remove(checkpoint_id) {
            if let Some(ids) = store.threads.get_mut(&cp.thread_id) {
                ids.retain(|id| id != checkpoint_id);
            }
        }
        self.persist(&store)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<(), RuntimeError> {
        let mut store = self.lock()?;
        let ids = store.threads.remove(thread_id).unwrap_or_default();
        for id in ids {
            store.checkpoints.remove(&id);
        }
        self.persist(&store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AgentState;
    use chrono::{Duration, Utc};

    fn tmp_db() -> (tempfile::TempDir, EmbeddedCheckpointer) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json");
        let db = EmbeddedCheckpointer::open(&path).unwrap();
        (dir, db)
    }

    #[tokio::test]
    async fn save_load_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json");
        let checkpoint_id;
        {
            let db = EmbeddedCheckpointer::open(&path).unwrap();
            let mut state = AgentState::with_user_message("hello");
            state.set_context("n", 1u32);
            let cp = Checkpoint::new("thread-1", "node-a", state);
            checkpoint_id = cp.id.clone();
            db.save(cp).await.unwrap();
        }
        let db = EmbeddedCheckpointer::open(&path).unwrap();
        let loaded = db.load("thread-1").await.unwrap().unwrap();
        assert_eq!(loaded.id, checkpoint_id);
        assert_eq!(loaded.node_id, "node-a");
        assert_eq!(loaded.state.get_context::<u32>("n"), Some(1));
    }

    #[tokio::test]
    async fn load_returns_newest_by_created_at() {
        let (_dir, db) = tmp_db();
        let mut older = Checkpoint::new("t", "n1", AgentState::new());
        let mut newer = Checkpoint::new("t", "n2", AgentState::new());
        older.created_at = Utc::now() - Duration::seconds(60);
        newer.created_at = Utc::now();
        let newer_id = newer.id.clone();
        db.save(newer).await.unwrap();
        db.save(older).await.unwrap();
        let latest = db.load("t").await.unwrap().unwrap();
        assert_eq!(latest.id, newer_id);
    }

    #[tokio::test]
    async fn list_delete_and_delete_thread() {
        let (_dir, db) = tmp_db();
        for i in 0..3 {
            db.save(Checkpoint::new("t1", format!("n{i}"), AgentState::new()))
                .await
                .unwrap();
        }
        db.save(Checkpoint::new("t2", "n0", AgentState::new()))
            .await
            .unwrap();

        let list = db.list("t1").await.unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].node_id, "n2");

        db.delete(&list[0].id).await.unwrap();
        assert_eq!(db.list("t1").await.unwrap().len(), 2);

        db.delete_thread("t1").await.unwrap();
        assert!(db.list("t1").await.unwrap().is_empty());
        assert_eq!(db.list("t2").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn missing_thread_is_none() {
        let (_dir, db) = tmp_db();
        assert!(db.load("nope").await.unwrap().is_none());
    }
}
