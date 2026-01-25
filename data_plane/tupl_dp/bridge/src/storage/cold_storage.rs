//! Cold storage - SQLite database for overflow and long-term persistence.
//!
//! Used for rules that don't fit in warm storage.
//! Simple schema: rule_id (PRIMARY KEY) + anchors (BLOB)

use parking_lot::Mutex;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Arc;

use crate::rule_vector::RuleVector;
use crate::types::RuleMetadata;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS rule_anchors (
    rule_id TEXT PRIMARY KEY,
    anchors BLOB NOT NULL,
    stored_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stored_at ON rule_anchors(stored_at_ms);

CREATE TABLE IF NOT EXISTS rule_metadata (
    rule_id TEXT PRIMARY KEY,
    metadata BLOB NOT NULL,
    stored_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metadata_stored_at ON rule_metadata(stored_at_ms);
";

/// SQLite-backed cold storage.
#[derive(Debug)]
pub struct ColdStorage {
    /// Database connection
    conn: Arc<Mutex<Connection>>,
}

impl ColdStorage {
    /// Open or create cold storage database.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| format!("Open DB failed: {}", e))?;

        conn.execute_batch(SCHEMA)
            .map_err(|e| format!("Create schema failed: {}", e))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Get anchors by rule_id.
    pub fn get(&self, rule_id: &str) -> Result<Option<RuleVector>, String> {
        let conn = self.conn.lock();

        let mut stmt = conn
            .prepare("SELECT anchors FROM rule_anchors WHERE rule_id = ?1")
            .map_err(|e| format!("Prepare failed: {}", e))?;

        let result = stmt.query_row(params![rule_id], |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(blob)
        });

        match result {
            Ok(blob) => {
                let anchors: RuleVector = bincode::deserialize(&blob)
                    .map_err(|e| format!("Deserialize failed: {}", e))?;
                Ok(Some(anchors))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Query failed: {}", e)),
        }
    }

    /// Insert or update rule anchors.
    pub fn upsert(&self, rule_id: &str, anchors: &RuleVector) -> Result<(), String> {
        let conn = self.conn.lock();

        let blob = bincode::serialize(anchors).map_err(|e| format!("Serialize failed: {}", e))?;

        conn.execute(
            "INSERT OR REPLACE INTO rule_anchors (rule_id, anchors, stored_at_ms) VALUES (?1, ?2, ?3)",
            params![rule_id, blob, crate::types::now_ms()],
        )
        .map_err(|e| format!("Insert failed: {}", e))?;

        Ok(())
    }

    /// Insert or update rule metadata.
    pub fn upsert_metadata(&self, metadata: &RuleMetadata) -> Result<(), String> {
        let conn = self.conn.lock();

        let blob = serde_json::to_vec(metadata).map_err(|e| format!("Serialize failed: {}", e))?;

        conn.execute(
            "INSERT OR REPLACE INTO rule_metadata (rule_id, metadata, stored_at_ms) VALUES (?1, ?2, ?3)",
            params![metadata.rule_id, blob, crate::types::now_ms()],
        )
        .map_err(|e| format!("Insert failed: {}", e))?;

        Ok(())
    }

    /// Remove a rule.
    pub fn remove(&self, rule_id: &str) -> Result<bool, String> {
        let conn = self.conn.lock();

        let rows = conn
            .execute(
                "DELETE FROM rule_anchors WHERE rule_id = ?1",
                params![rule_id],
            )
            .map_err(|e| format!("Delete failed: {}", e))?;

        Ok(rows > 0)
    }

    /// Remove rule metadata.
    pub fn remove_metadata(&self, rule_id: &str) -> Result<bool, String> {
        let conn = self.conn.lock();

        let rows = conn
            .execute(
                "DELETE FROM rule_metadata WHERE rule_id = ?1",
                params![rule_id],
            )
            .map_err(|e| format!("Delete failed: {}", e))?;

        Ok(rows > 0)
    }

    /// Remove all rules from cold storage.
    pub fn clear(&self) -> Result<(), String> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM rule_anchors", [])
            .map_err(|e| format!("Clear failed: {}", e))?;
        Ok(())
    }

    /// Remove all metadata from cold storage.
    pub fn clear_metadata(&self) -> Result<(), String> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM rule_metadata", [])
            .map_err(|e| format!("Clear failed: {}", e))?;
        Ok(())
    }

    /// List all rule IDs.
    pub fn list_rule_ids(&self) -> Result<Vec<String>, String> {
        let conn = self.conn.lock();

        let mut stmt = conn
            .prepare("SELECT rule_id FROM rule_anchors")
            .map_err(|e| format!("Prepare failed: {}", e))?;

        let ids = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| format!("Query failed: {}", e))?
            .collect::<Result<Vec<String>, _>>()
            .map_err(|e| format!("Collect failed: {}", e))?;

        Ok(ids)
    }

    /// List all stored rule metadata.
    pub fn list_metadata(&self) -> Result<Vec<RuleMetadata>, String> {
        let conn = self.conn.lock();

        let mut stmt = conn
            .prepare("SELECT rule_id, metadata FROM rule_metadata")
            .map_err(|e| format!("Prepare failed: {}", e))?;

        let items = stmt
            .query_map([], |row| {
                let rule_id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((rule_id, blob))
            })
            .map_err(|e| format!("Query failed: {}", e))?
            .collect::<Result<Vec<(String, Vec<u8>)>, _>>()
            .map_err(|e| format!("Collect failed: {}", e))?;

        let mut metadata = Vec::with_capacity(items.len());
        for (rule_id, blob) in items {
            match serde_json::from_slice::<RuleMetadata>(&blob) {
                Ok(item) => metadata.push(item),
                Err(err) => {
                    eprintln!("Skipping invalid rule metadata for {}: {}", rule_id, err);
                    let _ = conn.execute(
                        "DELETE FROM rule_metadata WHERE rule_id = ?1",
                        params![rule_id],
                    );
                }
            }
        }

        Ok(metadata)
    }

    /// Get metadata by rule_id.
    pub fn get_metadata(&self, rule_id: &str) -> Result<Option<RuleMetadata>, String> {
        let conn = self.conn.lock();

        let mut stmt = conn
            .prepare("SELECT metadata FROM rule_metadata WHERE rule_id = ?1")
            .map_err(|e| format!("Prepare failed: {}", e))?;

        let result = stmt.query_row(params![rule_id], |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(blob)
        });

        match result {
            Ok(blob) => match serde_json::from_slice::<RuleMetadata>(&blob) {
                Ok(metadata) => Ok(Some(metadata)),
                Err(err) => {
                    eprintln!("Removing invalid rule metadata for {}: {}", rule_id, err);
                    let _ = conn.execute(
                        "DELETE FROM rule_metadata WHERE rule_id = ?1",
                        params![rule_id],
                    );
                    Ok(None)
                }
            },
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Query failed: {}", e)),
        }
    }

    /// Count total rules in cold storage.
    pub fn count(&self) -> Result<usize, String> {
        let conn = self.conn.lock();

        conn.query_row("SELECT COUNT(*) FROM rule_anchors", [], |row| row.get(0))
            .map_err(|e| format!("Count failed: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_storage_create_and_open() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.db");

        let _ = ColdStorage::open(&path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_cold_storage_upsert_and_get() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.db");

        let storage = ColdStorage::open(&path).unwrap();
        let anchors = RuleVector::default();

        storage.upsert("rule-1", &anchors).unwrap();

        let result = storage.get("rule-1").unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_cold_storage_remove() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.db");

        let storage = ColdStorage::open(&path).unwrap();
        let anchors = RuleVector::default();

        storage.upsert("rule-1", &anchors).unwrap();
        let removed = storage.remove("rule-1").unwrap();
        assert!(removed);

        let result = storage.get("rule-1").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_cold_storage_list_rule_ids() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.db");

        let storage = ColdStorage::open(&path).unwrap();
        let anchors = RuleVector::default();

        storage.upsert("rule-1", &anchors).unwrap();
        storage.upsert("rule-2", &anchors).unwrap();

        let ids = storage.list_rule_ids().unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"rule-1".to_string()));
        assert!(ids.contains(&"rule-2".to_string()));
    }

    #[test]
    fn test_cold_storage_count() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.db");

        let storage = ColdStorage::open(&path).unwrap();
        let anchors = RuleVector::default();

        storage.upsert("rule-1", &anchors).unwrap();
        storage.upsert("rule-2", &anchors).unwrap();

        let count = storage.count().unwrap();
        assert_eq!(count, 2);
    }
}
