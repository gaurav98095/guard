use crate::families::DesignBoundaryRule;
use crate::rule_vector::RuleVector;
use crate::storage::{ColdStorage, HotCache, StorageStats, WarmStorage};
use crate::types::{now_ms, RuleInstance, RuleMetadata};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

// ================================================================================================
// STORAGE CONFIGURATION
// ================================================================================================

/// Configuration for Bridge tiered storage.
#[derive(Clone, Debug)]
pub struct StorageConfig {
    /// Path to warm storage file (mmap)
    pub warm_storage_path: PathBuf,
    /// Path to cold storage database (SQLite)
    pub cold_storage_path: PathBuf,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            warm_storage_path: PathBuf::from("./var/data/warm_storage.bin"),
            cold_storage_path: PathBuf::from("./var/data/cold_storage.db"),
        }
    }
}

// ================================================================================================
// BRIDGE STRUCTURE
// ================================================================================================

/// The Bridge is the root data structure for storing all rules in the data plane.
///
/// All rules are stored in tiered storage only (hot → warm → cold). There are no
/// per-family tables or layer hierarchies – rules are indexed by ID plus optional
/// metadata such as layer strings.
#[derive(Debug)]
pub struct Bridge {
    active_version: Arc<RwLock<u64>>,
    staged_version: Arc<RwLock<Option<u64>>>,
    created_at: u64,
    /// Hot cache: in-memory LRU cache for rule vectors
    pub hot_cache: Arc<HotCache>,
    /// Warm storage: memory-mapped file for persistent cache
    pub warm_storage: Arc<WarmStorage>,
    /// Cold storage: SQLite database for overflow
    pub cold_storage: Arc<ColdStorage>,
    /// In-memory index of installed rules for metadata lookups
    rule_index: Arc<RwLock<HashMap<String, Arc<dyn RuleInstance>>>>,
}

impl Bridge {
    /// Initializes a new Bridge with default storage configuration.
    pub fn init() -> Result<Self, String> {
        Self::new(StorageConfig::default())
    }

    /// Creates a new Bridge with the specified storage configuration.
    pub fn new(storage_config: StorageConfig) -> Result<Self, String> {
        let hot_cache = Arc::new(HotCache::new());
        let warm_storage = Arc::new(WarmStorage::open(&storage_config.warm_storage_path)?);
        let cold_storage = Arc::new(ColdStorage::open(&storage_config.cold_storage_path)?);

        // Load warm storage into hot cache on startup for fast access
        let warm_anchors = warm_storage.load_anchors()?;
        for (rule_id, anchors) in warm_anchors {
            hot_cache.insert(rule_id, anchors)?;
        }

        let mut rule_index: HashMap<String, Arc<dyn RuleInstance>> = HashMap::new();
        for metadata in cold_storage.list_metadata()? {
            let rule_id = metadata.rule_id.clone();
            let rule: Arc<dyn RuleInstance> = Arc::new(DesignBoundaryRule::new(
                rule_id.clone(),
                metadata.priority,
                metadata.scope,
                metadata.layer,
                metadata.created_at_ms,
                metadata.enabled,
                metadata.description,
                metadata.params,
            ));
            rule_index.insert(rule_id, rule);
        }

        Ok(Bridge {
            active_version: Arc::new(RwLock::new(0)),
            staged_version: Arc::new(RwLock::new(None)),
            created_at: now_ms(),
            hot_cache,
            warm_storage,
            cold_storage,
            rule_index: Arc::new(RwLock::new(rule_index)),
        })
    }

    /// Creates a Bridge with default storage paths.
    pub fn with_defaults() -> Result<Self, String> {
        Self::new(StorageConfig::default())
    }

    // ============================================================================================
    // ACCESSORS
    // ============================================================================================

    /// Returns the current global version
    pub fn version(&self) -> u64 {
        *self.active_version.read()
    }

    /// Returns the staged version (if any)
    pub fn staged_version(&self) -> Option<u64> {
        *self.staged_version.read()
    }

    /// Returns the creation timestamp
    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    /// Returns the number of installed rules
    pub fn rule_count(&self) -> usize {
        self.rule_index.read().len()
    }

    /// Returns a clone of all installed rule instances.
    pub fn all_rules(&self) -> Vec<Arc<dyn RuleInstance>> {
        self.rule_index.read().values().cloned().collect()
    }

    /// Returns a specific rule by ID if present.
    pub fn get_rule(&self, rule_id: &str) -> Option<Arc<dyn RuleInstance>> {
        self.rule_index.read().get(rule_id).cloned()
    }

    // ============================================================================================
    // RULE OPERATIONS
    // ============================================================================================

    /// Adds a rule and stores its pre-encoded anchors with tiered persistence.
    pub fn add_rule_with_anchors(
        &self,
        rule: Arc<dyn RuleInstance>,
        anchors: RuleVector,
    ) -> Result<(), String> {
        let rule_id = rule.rule_id().to_string();

        self.hot_cache.insert(rule_id.clone(), anchors.clone())?;
        self.warm_storage
            .write_anchors(self.hot_cache.snapshot())
            .map_err(|e| format!("Warm storage sync failed: {}", e))?;
        self.cold_storage
            .upsert(&rule_id, &anchors)
            .map_err(|e| format!("Cold storage upsert failed: {}", e))?;
        let metadata = RuleMetadata::from_rule(rule.as_ref());
        self.cold_storage
            .upsert_metadata(&metadata)
            .map_err(|e| format!("Cold storage metadata upsert failed: {}", e))?;

        self.rule_index.write().insert(rule_id, Arc::clone(&rule));
        self.increment_version();

        Ok(())
    }

    /// Removes a rule by ID. Returns true if the rule was present.
    pub fn remove_rule(&self, rule_id: &str) -> Result<bool, String> {
        let removed = self.rule_index.write().remove(rule_id).is_some();
        if !removed {
            return Ok(false);
        }

        self.hot_cache.remove(rule_id);
        self.warm_storage
            .write_anchors(self.hot_cache.snapshot())
            .map_err(|e| format!("Warm storage sync failed: {}", e))?;
        self.cold_storage
            .remove(rule_id)
            .map_err(|e| format!("Cold storage removal failed: {}", e))?;
        self.cold_storage
            .remove_metadata(rule_id)
            .map_err(|e| format!("Cold storage metadata removal failed: {}", e))?;
        self.increment_version();
        Ok(true)
    }

    /// Clears all rules and tiered storage state.
    pub fn clear_all(&self) {
        self.rule_index.write().clear();
        self.hot_cache.clear();
        let _ = self
            .warm_storage
            .write_anchors(HashMap::<String, RuleVector>::new());
        let _ = self.cold_storage.clear();
        let _ = self.cold_storage.clear_metadata();
        self.increment_version();
    }

    /// Get rule anchors with tiered lookup and automatic promotion.
    pub fn get_rule_anchors(&self, rule_id: &str) -> Option<RuleVector> {
        if let Some(anchors) = self.hot_cache.get(rule_id) {
            return Some(anchors);
        }

        if let Ok(Some(anchors)) = self.warm_storage.get(rule_id) {
            let _ = self.hot_cache.insert(rule_id.to_string(), anchors.clone());
            return Some(anchors);
        }

        if let Ok(Some(anchors)) = self.cold_storage.get(rule_id) {
            let _ = self.hot_cache.insert(rule_id.to_string(), anchors.clone());
            return Some(anchors);
        }

        None
    }

    // ============================================================================================
    // STATISTICS & MONITORING
    // ============================================================================================

    /// Returns statistics about the bridge
    pub fn stats(&self) -> BridgeStats {
        let index = self.rule_index.read();
        let total_rules = index.len();
        let global_rules = index.values().filter(|rule| rule.scope().is_global).count();

        BridgeStats {
            version: self.version(),
            total_rules,
            global_rules,
            scoped_rules: total_rules.saturating_sub(global_rules),
            created_at: self.created_at,
        }
    }

    /// Returns storage statistics across all tiers.
    pub fn storage_stats(&self) -> StorageStats {
        let hot_stats = self.hot_cache.stats();

        StorageStats {
            hot_rules: hot_stats.entries,
            evictions: hot_stats.total_evictions,
            ..Default::default()
        }
    }

    // ============================================================================================
    // VERSIONING
    // ============================================================================================

    /// Increments the bridge version
    fn increment_version(&self) {
        *self.active_version.write() += 1;
    }

    /// Sets the staged version for hot-reload
    pub fn set_staged_version(&self, version: u64) {
        *self.staged_version.write() = Some(version);
    }

    /// Clears the staged version
    pub fn clear_staged_version(&self) {
        *self.staged_version.write() = None;
    }

    /// Promotes staged version to active (atomic hot-reload)
    pub fn promote_staged(&self) -> Result<(), String> {
        let staged = *self.staged_version.read();

        match staged {
            Some(v) => {
                *self.active_version.write() = v;
                self.clear_staged_version();
                Ok(())
            }
            None => Err("No staged version to promote".to_string()),
        }
    }
}

// ================================================================================================
// STATISTICS STRUCTURES
// ================================================================================================

/// Bridge-level statistics
#[derive(Debug, Clone)]
pub struct BridgeStats {
    /// Current bridge version
    pub version: u64,
    /// Total rules stored in tiered storage
    pub total_rules: usize,
    /// Number of global rules (scope.is_global)
    pub global_rules: usize,
    /// Number of scoped rules (total - global)
    pub scoped_rules: usize,
    /// Bridge creation timestamp
    pub created_at: u64,
}

// ================================================================================================
// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bridge() -> Result<Bridge, String> {
        let tmp_dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let warm_path = tmp_dir.path().join("warm.bin");
        let cold_path = tmp_dir.path().join("cold.db");

        let config = StorageConfig {
            warm_storage_path: warm_path,
            cold_storage_path: cold_path,
        };

        Bridge::new(config)
    }

    #[test]
    fn test_bridge_init_with_storage() -> Result<(), String> {
        let bridge = create_test_bridge()?;

        assert_eq!(bridge.rule_count(), 0);

        let storage_stats = bridge.storage_stats();
        assert_eq!(storage_stats.hot_rules, 0);

        Ok(())
    }

    #[test]
    fn test_hot_cache_lookup_empty() -> Result<(), String> {
        let bridge = create_test_bridge()?;

        let result = bridge.get_rule_anchors("non-existent-rule");
        assert!(result.is_none());

        Ok(())
    }

    #[test]
    fn test_warm_storage_reload_on_startup() -> Result<(), String> {
        let tmp_dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let warm_path = tmp_dir.path().join("warm.bin");
        let cold_path = tmp_dir.path().join("cold.db");

        {
            let config = StorageConfig {
                warm_storage_path: warm_path.clone(),
                cold_storage_path: cold_path.clone(),
            };
            let bridge = Bridge::new(config)?;
            let stats = bridge.storage_stats();
            assert_eq!(stats.hot_rules, 0);
        }

        {
            let config = StorageConfig {
                warm_storage_path: warm_path,
                cold_storage_path: cold_path,
            };
            let bridge = Bridge::new(config)?;
            let stats = bridge.storage_stats();
            assert_eq!(stats.hot_rules, 0);
        }

        Ok(())
    }

    #[test]
    fn test_storage_config_defaults() {
        let config = StorageConfig::default();
        assert_eq!(
            config.warm_storage_path,
            PathBuf::from("./var/data/warm_storage.bin")
        );
        assert_eq!(
            config.cold_storage_path,
            PathBuf::from("./var/data/cold_storage.db")
        );
    }

    #[test]
    fn test_bridge_init_defaults() {
        let bridge = Bridge::with_defaults();
        assert!(bridge.is_ok());
    }
}
