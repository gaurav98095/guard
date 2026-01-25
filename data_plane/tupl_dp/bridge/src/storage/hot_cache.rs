//! LRU hot cache implementation for rule vectors.
//!
//! The hot cache maintains the most frequently accessed rules in memory
//! with automatic LRU eviction when capacity is exceeded.
//!
//! **Algorithm**: Least-Recently-Used (LRU) eviction
//! - Each entry stores `(last_accessed_timestamp_ms, rule_vector)`
//! - When capacity (10K rules) is exceeded, evict 10% of oldest entries
//! - Eviction is O(n log n) due to sort, but 10% batch amortizes cost
//! - Access on lookup updates timestamp (mark_evaluated call)

use crate::rule_vector::RuleVector;
use crate::types::now_ms;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Hot cache configuration constants
const DEFAULT_CAPACITY: usize = 10_000;
const EVICTION_BATCH_PERCENT: f32 = 0.10; // Evict 10% of capacity at a time

/// Statistics about hot cache performance
#[derive(Clone, Debug)]
pub struct HotCacheStats {
    /// Number of rules currently in cache
    pub entries: usize,
    /// Total number of evictions (not per-entry, but eviction operations)
    pub total_evictions: u64,
    /// Total rules evicted across all operations
    pub total_evicted: u64,
    /// Cache capacity
    pub capacity: usize,
}

/// Hot cache with LRU eviction policy for rule vectors.
///
/// Maintains a HashMap of rule_id → (last_accessed_ms, RuleVector).
/// When inserting and at capacity, evicts least-recently-used entries.
#[derive(Debug)]
pub struct HotCache {
    /// Map of rule_id → (last_accessed_ms, rule_vector)
    cache: Arc<RwLock<HashMap<String, (u64, RuleVector)>>>,
    /// Maximum number of entries before eviction
    capacity: usize,
    /// Statistics tracking
    stats: Arc<RwLock<HotCacheStats>>,
}

impl HotCache {
    /// Create a new hot cache with default capacity (10K).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a new hot cache with specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of rule entries before eviction triggers
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            capacity,
            stats: Arc::new(RwLock::new(HotCacheStats {
                entries: 0,
                total_evictions: 0,
                total_evicted: 0,
                capacity,
            })),
        }
    }

    /// Insert or update a rule in the hot cache.
    ///
    /// If the rule already exists, updates its last_accessed timestamp.
    /// If at capacity and inserting new rule, triggers LRU eviction first.
    ///
    /// # Arguments
    /// * `rule_id` - Unique identifier for the rule
    /// * `anchors` - Pre-encoded rule vector
    ///
    /// # Returns
    /// `Ok(())` on success, `Err(String)` if eviction fails
    pub fn insert(&self, rule_id: String, anchors: RuleVector) -> Result<(), String> {
        let mut cache = self.cache.write();

        // If rule already exists, just update timestamp
        if cache.contains_key(&rule_id) {
            let now = now_ms();
            cache.insert(rule_id, (now, anchors));
            return Ok(());
        }

        // Check if we need to evict before adding new entry
        if cache.len() >= self.capacity {
            self.evict_lru(&mut cache)?;
        }

        // Insert new entry with current timestamp
        let now = now_ms();
        cache.insert(rule_id, (now, anchors));

        // Update stats
        let mut stats = self.stats.write();
        stats.entries = cache.len();

        Ok(())
    }

    /// Get a rule from the hot cache WITHOUT updating access time.
    ///
    /// This is useful for lookups that don't represent "evaluation" access.
    /// Use `get_and_mark` if you want to update the access timestamp.
    pub fn get(&self, rule_id: &str) -> Option<RuleVector> {
        let cache = self.cache.read();
        cache.get(rule_id).map(|(_, vector)| vector.clone())
    }

    /// Get a rule and update its last access time.
    ///
    /// This should be called after successful rule comparison
    /// to mark the rule as recently used (prevent eviction).
    pub fn get_and_mark(&self, rule_id: &str) -> Option<RuleVector> {
        let mut cache = self.cache.write();
        if let Some((ts, vector)) = cache.get_mut(rule_id) {
            *ts = now_ms();
            return Some(vector.clone());
        }
        None
    }

    /// Take a snapshot of the current cache entries (rule_id → RuleVector).
    pub fn snapshot(&self) -> HashMap<String, RuleVector> {
        let cache = self.cache.read();
        cache
            .iter()
            .map(|(rule_id, (_, vector))| (rule_id.clone(), vector.clone()))
            .collect()
    }

    /// Check if a rule exists in the cache.
    pub fn contains(&self, rule_id: &str) -> bool {
        self.cache.read().contains_key(rule_id)
    }

    /// Remove a rule from the cache.
    pub fn remove(&self, rule_id: &str) -> Option<RuleVector> {
        let mut cache = self.cache.write();
        cache.remove(rule_id).map(|(_, vector)| {
            let mut stats = self.stats.write();
            stats.entries = cache.len();
            vector
        })
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        let mut stats = self.stats.write();
        stats.entries = 0;
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> HotCacheStats {
        self.stats.read().clone()
    }

    /// Evict least-recently-used entries when cache is full.
    ///
    /// This is called internally when a new entry would exceed capacity.
    /// It evicts 10% of capacity (rounded up to at least 1 entry).
    ///
    /// **Algorithm**:
    /// 1. Collect all entries with their timestamps
    /// 2. Sort by timestamp (oldest first)
    /// 3. Remove oldest 10% of capacity
    /// 4. Update statistics
    /// 5. Log eviction details
    fn evict_lru(&self, cache: &mut HashMap<String, (u64, RuleVector)>) -> Result<(), String> {
        // Calculate how many entries to evict (10% of capacity, at least 1)
        let evict_count =
            std::cmp::max(1, (self.capacity as f32 * EVICTION_BATCH_PERCENT) as usize);

        // Collect entries with their rule_ids and timestamps
        let mut entries: Vec<(String, u64)> = cache
            .iter()
            .map(|(rule_id, (ts, _))| (rule_id.clone(), *ts))
            .collect();

        // Sort by timestamp (oldest first)
        entries.sort_by_key(|(_rule_id, ts)| *ts);

        // Evict the oldest entries
        for (rule_id, _ts) in entries.iter().take(evict_count) {
            cache.remove(rule_id);
        }

        // Update statistics
        let mut stats = self.stats.write();
        stats.total_evictions += 1;
        stats.total_evicted += evict_count as u64;
        stats.entries = cache.len();

        // Debug: LRU eviction occurred
        // evict_count oldest entries removed, cache now has cache.len() entries

        Ok(())
    }
}

impl Default for HotCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let cache = HotCache::with_capacity(100);
        let rule_id = "rule-1".to_string();
        let vector = RuleVector::default();

        assert!(cache.insert(rule_id.clone(), vector.clone()).is_ok());
        assert!(cache.contains(&rule_id));
        assert!(cache.get(&rule_id).is_some());
    }

    #[test]
    fn test_update_existing_rule() {
        let cache = HotCache::with_capacity(100);
        let rule_id = "rule-1".to_string();
        let vector = RuleVector::default();

        // Insert first time
        cache.insert(rule_id.clone(), vector.clone()).ok();
        let stats_after_insert = cache.stats();
        assert_eq!(stats_after_insert.entries, 1);

        // Update same rule - should not increase count
        cache.insert(rule_id.clone(), vector.clone()).ok();
        let stats_after_update = cache.stats();
        assert_eq!(stats_after_update.entries, 1);

        // No eviction should have occurred
        assert_eq!(stats_after_update.total_evictions, 0);
    }

    #[test]
    fn test_capacity_enforcement() {
        let cache = HotCache::with_capacity(10);

        // Insert 10 rules (at capacity)
        for i in 0..10 {
            let rule_id = format!("rule-{}", i);
            let vector = RuleVector::default();
            assert!(cache.insert(rule_id, vector).is_ok());
        }

        let stats = cache.stats();
        assert_eq!(stats.entries, 10);
        assert_eq!(stats.total_evictions, 0);

        // Insert 11th rule - should trigger eviction
        let vector = RuleVector::default();
        cache.insert("rule-10".to_string(), vector).ok();

        let stats = cache.stats();
        // Should have evicted 1 entry (10% of 10), then inserted new one
        assert_eq!(stats.entries, 10);
        assert!(stats.total_evictions > 0);
    }

    #[test]
    fn test_evicts_least_recently_used() {
        let cache = HotCache::with_capacity(10);

        // Insert 10 rules
        for i in 0..10 {
            let rule_id = format!("rule-{}", i);
            let vector = RuleVector::default();
            cache.insert(rule_id, vector).ok();
            // Small delay to ensure different timestamps
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // Access rule-5 to make it recently used
        cache.get_and_mark("rule-5");
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Insert new rule - should evict oldest (rule-0, not rule-5)
        let vector = RuleVector::default();
        cache.insert("rule-10".to_string(), vector).ok();

        // rule-5 should still be there (it was accessed)
        assert!(cache.contains("rule-5"));
        // rule-0 should be gone (it was oldest)
        assert!(!cache.contains("rule-0"));
    }

    #[test]
    fn test_eviction_statistics() {
        let cache = HotCache::with_capacity(5);

        // Insert 5 rules
        for i in 0..5 {
            let rule_id = format!("rule-{}", i);
            let vector = RuleVector::default();
            cache.insert(rule_id, vector).ok();
        }

        let stats_before = cache.stats();
        assert_eq!(stats_before.total_evictions, 0);
        assert_eq!(stats_before.total_evicted, 0);

        // Trigger eviction
        let vector = RuleVector::default();
        cache.insert("rule-5".to_string(), vector).ok();

        let stats_after = cache.stats();
        assert_eq!(stats_after.total_evictions, 1);
        assert!(stats_after.total_evicted > 0); // At least 1 evicted
    }

    #[test]
    fn test_mark_evaluated() {
        let cache = HotCache::with_capacity(10);
        let rule_id = "rule-1".to_string();
        let vector = RuleVector::default();

        cache.insert(rule_id.clone(), vector).ok();

        // Get without marking
        let result = cache.get(&rule_id);
        assert!(result.is_some());

        // Get and mark as recently used
        let result = cache.get_and_mark(&rule_id);
        assert!(result.is_some());

        // Both should return the same vector data
        let vec1 = cache.get(&rule_id).unwrap();
        let vec2 = cache.get_and_mark(&rule_id).unwrap();
        assert_eq!(vec1.action_count, vec2.action_count);
    }

    #[test]
    fn test_remove_and_clear() {
        let cache = HotCache::with_capacity(100);

        // Insert some rules
        for i in 0..5 {
            let rule_id = format!("rule-{}", i);
            let vector = RuleVector::default();
            cache.insert(rule_id, vector).ok();
        }

        assert_eq!(cache.stats().entries, 5);

        // Remove one
        cache.remove("rule-2");
        assert_eq!(cache.stats().entries, 4);
        assert!(!cache.contains("rule-2"));

        // Clear all
        cache.clear();
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(HotCache::with_capacity(100));

        let mut handles = vec![];

        // Spawn multiple threads
        for thread_id in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..10 {
                    let rule_id = format!("rule-{}-{}", thread_id, i);
                    let vector = RuleVector::default();
                    cache_clone.insert(rule_id.clone(), vector).ok();

                    // Sometimes access to mark as recently used
                    if i % 3 == 0 {
                        cache_clone.get_and_mark(&rule_id);
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Cache should have entries and no panics
        let stats = cache.stats();
        assert!(stats.entries > 0);
    }
}
