//! # Hitlog Query Utility
//!
//! Query and analyze enforcement sessions from hitlog files.

use super::session::EnforcementSession;
use rusqlite::{params_from_iter, Connection};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Query filter for searching hitlogs
#[derive(Debug, Clone, Default)]
pub struct QueryFilter {
    /// Filter by session ID
    pub session_id: Option<String>,

    /// Filter by layer
    pub layer: Option<String>,

    /// Filter by agent ID
    pub agent_id: Option<String>,

    /// Filter by tenant ID
    pub tenant_id: Option<String>,

    /// Filter by decision (0 = BLOCK, 1 = ALLOW)
    pub decision: Option<u8>,

    /// Filter by time range (start timestamp in ms)
    pub start_time_ms: Option<u64>,

    /// Filter by time range (end timestamp in ms)
    pub end_time_ms: Option<u64>,

    /// Filter by minimum duration (microseconds)
    pub min_duration_us: Option<u64>,

    /// Filter by maximum duration (microseconds)
    pub max_duration_us: Option<u64>,

    /// Filter by rule ID (session must have evaluated this rule)
    pub rule_id: Option<String>,

    /// Maximum results to return
    pub limit: Option<usize>,

    /// Number of results to skip (for pagination)
    pub offset: Option<usize>,
}

/// Query results
#[derive(Debug)]
pub struct QueryResult {
    pub sessions: Vec<EnforcementSession>,
    pub total_matched: usize,
    pub files_searched: usize,
}

/// Hitlog query engine
pub struct HitlogQuery {
    hitlog_dir: PathBuf,
    sqlite_path: Option<PathBuf>,
}

impl HitlogQuery {
    /// Create a new query engine
    pub fn new(hitlog_dir: impl AsRef<Path>) -> Self {
        HitlogQuery {
            hitlog_dir: hitlog_dir.as_ref().to_path_buf(),
            sqlite_path: std::env::var("HITLOG_SQLITE_PATH").ok().map(PathBuf::from),
        }
    }

    /// Query hitlogs with filters
    pub fn query(&self, filter: &QueryFilter) -> Result<QueryResult, String> {
        // Prefer SQLite if configured; fall back to file-based hitlogs
        if let Some(ref sqlite_path) = self.sqlite_path {
            match self.query_sqlite(sqlite_path, filter) {
                Ok(result) => return Ok(result),
                Err(e) => eprintln!("SQLite hitlog query failed (falling back to files): {}", e),
            }
        }

        let mut sessions = Vec::new();
        let mut total_matched = 0;
        let mut files_searched = 0;

        // Get all hitlog files (current + rotated)
        let files = self.get_hitlog_files()?;

        for file_path in files {
            files_searched += 1;

            let file = File::open(&file_path)
                .map_err(|e| format!("Failed to open {}: {}", file_path.display(), e))?;

            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read line: {}", e))?;

                // Parse session
                let session: EnforcementSession = match serde_json::from_str(&line) {
                    Ok(s) => s,
                    Err(_) => continue, // Skip malformed lines
                };

                // Apply filters
                if self.matches_filter(&session, filter) {
                    total_matched += 1;

                    // Apply offset
                    let offset = filter.offset.unwrap_or(0);
                    if total_matched <= offset {
                        continue;
                    }

                    // Check limit
                    if let Some(limit) = filter.limit {
                        if sessions.len() >= limit {
                            return Ok(QueryResult {
                                sessions,
                                total_matched,
                                files_searched,
                            });
                        }
                    }

                    sessions.push(session);
                }
            }
        }

        Ok(QueryResult {
            sessions,
            total_matched,
            files_searched,
        })
    }

    fn query_sqlite(&self, path: &Path, filter: &QueryFilter) -> Result<QueryResult, String> {
        let conn = Connection::open(path)
            .map_err(|e| format!("Failed to open sqlite hitlog db: {}", e))?;

        let mut conditions: Vec<String> = Vec::new();
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(ref session_id) = filter.session_id {
            conditions.push("session_id = ?".into());
            params_vec.push(Box::new(session_id.clone()));
        }
        if let Some(ref layer) = filter.layer {
            conditions.push("layer = ?".into());
            params_vec.push(Box::new(layer.clone()));
        }
        if let Some(ref agent_id) = filter.agent_id {
            conditions.push("agent_id = ?".into());
            params_vec.push(Box::new(agent_id.clone()));
        }
        if let Some(ref tenant_id) = filter.tenant_id {
            conditions.push("tenant_id = ?".into());
            params_vec.push(Box::new(tenant_id.clone()));
        }
        if let Some(decision) = filter.decision {
            conditions.push("final_decision = ?".into());
            params_vec.push(Box::new(decision as i64));
        }
        if let Some(start_time) = filter.start_time_ms {
            conditions.push("timestamp_ms >= ?".into());
            params_vec.push(Box::new(start_time as i64));
        }
        if let Some(end_time) = filter.end_time_ms {
            conditions.push("timestamp_ms <= ?".into());
            params_vec.push(Box::new(end_time as i64));
        }

        let mut sql = "SELECT session_json FROM hitlogs".to_string();
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(" ORDER BY timestamp_ms DESC");

        if let Some(limit) = filter.limit {
            sql.push_str(" LIMIT ");
            sql.push_str(&limit.min(500).to_string());
        }
        if let Some(offset) = filter.offset {
            sql.push_str(" OFFSET ");
            sql.push_str(&offset.to_string());
        }

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("Failed to prepare sqlite query: {}", e))?;

        let rows = stmt
            .query_map(params_from_iter(params_vec.iter().map(|b| &**b)), |row| {
                let json: String = row.get(0)?;
                Ok(json)
            })
            .map_err(|e| format!("SQLite query error: {}", e))?;

        let mut sessions = Vec::new();
        let mut total_matched = 0;
        for r in rows {
            let json = r.map_err(|e| format!("SQLite row error: {}", e))?;
            if let Ok(session) = serde_json::from_str::<EnforcementSession>(&json) {
                // Apply any min/max duration or rule filters here (client-side)
                if self.matches_filter(&session, filter) {
                    total_matched += 1;
                    sessions.push(session);
                }
            }
        }

        Ok(QueryResult {
            sessions,
            total_matched,
            files_searched: 0,
        })
    }

    /// Get the most recent N sessions
    pub fn recent(&self, limit: usize) -> Result<Vec<EnforcementSession>, String> {
        let filter = QueryFilter {
            limit: Some(limit),
            ..Default::default()
        };

        let result = self.query(&filter)?;
        Ok(result.sessions)
    }

    /// Get all blocked sessions
    pub fn blocked(&self, limit: Option<usize>) -> Result<Vec<EnforcementSession>, String> {
        let filter = QueryFilter {
            decision: Some(0),
            limit,
            ..Default::default()
        };

        let result = self.query(&filter)?;
        Ok(result.sessions)
    }

    /// Get sessions for a specific agent
    pub fn by_agent(
        &self,
        agent_id: String,
        limit: Option<usize>,
    ) -> Result<Vec<EnforcementSession>, String> {
        let filter = QueryFilter {
            agent_id: Some(agent_id),
            limit,
            ..Default::default()
        };

        let result = self.query(&filter)?;
        Ok(result.sessions)
    }

    /// Get sessions by time range
    pub fn by_time_range(
        &self,
        start_ms: u64,
        end_ms: u64,
        limit: Option<usize>,
    ) -> Result<Vec<EnforcementSession>, String> {
        let filter = QueryFilter {
            start_time_ms: Some(start_ms),
            end_time_ms: Some(end_ms),
            limit,
            ..Default::default()
        };

        let result = self.query(&filter)?;
        Ok(result.sessions)
    }

    /// Check if session matches filter
    fn matches_filter(&self, session: &EnforcementSession, filter: &QueryFilter) -> bool {
        if let Some(ref session_id) = filter.session_id {
            if &session.session_id != session_id {
                return false;
            }
        }

        if let Some(ref layer) = filter.layer {
            if &session.layer != layer {
                return false;
            }
        }

        if let Some(ref agent_id) = filter.agent_id {
            if session.agent_id.as_ref() != Some(agent_id) {
                return false;
            }
        }

        if let Some(ref tenant_id) = filter.tenant_id {
            if session.tenant_id.as_ref() != Some(tenant_id) {
                return false;
            }
        }

        if let Some(decision) = filter.decision {
            if session.final_decision != decision {
                return false;
            }
        }

        if let Some(start_time) = filter.start_time_ms {
            if session.timestamp_ms < start_time {
                return false;
            }
        }

        if let Some(end_time) = filter.end_time_ms {
            if session.timestamp_ms > end_time {
                return false;
            }
        }

        if let Some(min_duration) = filter.min_duration_us {
            if session.duration_us < min_duration {
                return false;
            }
        }

        if let Some(max_duration) = filter.max_duration_us {
            if session.duration_us > max_duration {
                return false;
            }
        }

        if let Some(ref rule_id) = filter.rule_id {
            if !session
                .rules_evaluated
                .iter()
                .any(|r| &r.rule_id == rule_id)
            {
                return false;
            }
        }

        true
    }

    /// Get all hitlog files (current + rotated, sorted newest first)
    fn get_hitlog_files(&self) -> Result<Vec<PathBuf>, String> {
        use std::fs;

        let entries = fs::read_dir(&self.hitlog_dir)
            .map_err(|e| format!("Failed to read hitlog directory: {}", e))?;

        let mut files: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|name| name.starts_with("enforcement.hitlog"))
                    .unwrap_or(false)
            })
            .collect();

        // Sort by modification time (newest first)
        files.sort_by_key(|path| {
            fs::metadata(path)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        files.reverse();

        Ok(files)
    }

    /// Get aggregate statistics from hitlogs
    pub fn statistics(&self) -> Result<HitlogStatistics, String> {
        let all_sessions = self.query(&QueryFilter::default())?;

        let total_sessions = all_sessions.sessions.len();
        let blocked = all_sessions
            .sessions
            .iter()
            .filter(|s| s.final_decision == 0)
            .count();
        let allowed = all_sessions
            .sessions
            .iter()
            .filter(|s| s.final_decision == 1)
            .count();

        let total_duration_us: u64 = all_sessions.sessions.iter().map(|s| s.duration_us).sum();
        let avg_duration_us = if total_sessions > 0 {
            total_duration_us / total_sessions as u64
        } else {
            0
        };

        let total_rules_evaluated: usize = all_sessions
            .sessions
            .iter()
            .map(|s| s.rules_evaluated.len())
            .sum();
        let avg_rules_per_session = if total_sessions > 0 {
            total_rules_evaluated as f64 / total_sessions as f64
        } else {
            0.0
        };

        Ok(HitlogStatistics {
            total_sessions,
            blocked,
            allowed,
            block_rate: if total_sessions > 0 {
                blocked as f64 / total_sessions as f64
            } else {
                0.0
            },
            avg_duration_us,
            avg_rules_per_session,
        })
    }
}

/// Aggregate statistics from hitlogs
#[derive(Debug, Clone)]
pub struct HitlogStatistics {
    pub total_sessions: usize,
    pub blocked: usize,
    pub allowed: usize,
    pub block_rate: f64,
    pub avg_duration_us: u64,
    pub avg_rules_per_session: f64,
}
