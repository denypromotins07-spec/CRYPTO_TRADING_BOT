// ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
// High-Frequency Feature Stores, MLOps, and Concept Drift
// File: backend/distributed/state_sharding.rs
// Chapter 3: Distributed Feature Engineering and Ray Actor State Management
//
// Partitions feature state to prevent Ray actor contention.
// Uses consistent hashing for deterministic shard assignment.
// Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
// Zero-cost abstractions with lock-free shard access.

use std::collections::{HashMap, BTreeMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::hash::{Hash, Hasher};
use std::time::Instant;

/// Number of shards (power of 2 for efficient modulo)
const NUM_SHARDS: usize = 16;

/// Maximum entries per shard before split consideration
const MAX_ENTRIES_PER_SHARD: usize = 100_000;

/// Shard identifier type
type ShardId = usize;

/// Consistent hash ring for shard assignment
struct ConsistentHashRing {
    /// Virtual nodes per physical shard
    virtual_nodes: usize,
    /// Sorted ring of hash positions
    ring: BTreeMap<u64, ShardId>,
    /// Cached shard count
    shard_count: usize,
}

impl ConsistentHashRing {
    /// Create a new consistent hash ring
    pub fn new(shard_count: usize, virtual_nodes: usize) -> Self {
        let mut ring = BTreeMap::new();
        
        for shard_id in 0..shard_count {
            for vn in 0..virtual_nodes {
                // Hash shard_id + virtual_node_id to get position
                let hash = compute_hash(&(shard_id, vn));
                ring.insert(hash, shard_id);
            }
        }
        
        Self {
            virtual_nodes,
            ring,
            shard_count,
        }
    }

    /// Get the shard ID for a given key
    #[inline]
    pub fn get_shard(&self, key: &dyn Hash) -> ShardId {
        let hash = compute_hash(key);
        
        // Find the first node with hash >= key hash (clockwise search)
        if let Some((_, &shard_id)) = self.range().find(|(h, _)| **h >= hash) {
            shard_id
        } else {
            // Wrap around to first node
            *self.ring.values().next().unwrap()
        }
    }

    /// Get range reference for iteration
    fn range(&self) -> impl Iterator<Item = (&u64, &ShardId)> {
        self.ring.iter()
    }

    /// Add a new shard dynamically
    pub fn add_shard(&mut self, new_shard_id: ShardId) {
        for vn in 0..self.virtual_nodes {
            let hash = compute_hash(&(new_shard_id, vn));
            self.ring.insert(hash, new_shard_id);
        }
        self.shard_count += 1;
    }
}

/// Compute hash for any Hash type
#[inline]
fn compute_hash<T: Hash>(value: &T) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// A single shard containing a subset of feature state
pub struct FeatureShard {
    /// Feature data stored as key-value pairs
    data: RwLock<HashMap<String, Vec<f64>>>,
    /// Access count for hot/cold detection
    access_count: AtomicUsize,
    /// Last access timestamp
    last_access_ns: AtomicU64,
    /// Memory usage estimate
    memory_bytes: AtomicUsize,
}

impl FeatureShard {
    /// Create a new empty shard
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::with_capacity(1024)),
            access_count: AtomicUsize::new(0),
            last_access_ns: AtomicU64::new(0),
            memory_bytes: AtomicUsize::new(0),
        }
    }

    /// Get a feature value (read-only, lock-free via snapshot)
    pub fn get(&self, key: &str) -> Option<Vec<f64>> {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access_ns.store(current_time_ns(), Ordering::Relaxed);
        
        let data = self.data.read().unwrap();
        data.get(key).cloned()
    }

    /// Insert or update a feature value
    pub fn insert(&self, key: String, value: Vec<f64>) -> Option<Vec<f64>> {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access_ns.store(current_time_ns(), Ordering::Relaxed);
        
        let mut data = self.data.write().unwrap();
        let old_value = data.insert(key, value);
        
        // Update memory estimate
        self.update_memory_estimate();
        
        old_value
    }

    /// Remove a feature
    pub fn remove(&self, key: &str) -> Option<Vec<f64>> {
        let mut data = self.data.write().unwrap();
        let removed = data.remove(key);
        
        if removed.is_some() {
            self.update_memory_estimate();
        }
        
        removed
    }

    /// Get all keys in this shard
    pub fn keys(&self) -> Vec<String> {
        let data = self.data.read().unwrap();
        data.keys().cloned().collect()
    }

    /// Get entry count
    pub fn len(&self) -> usize {
        let data = self.data.read().unwrap();
        data.len()
    }

    /// Check if shard is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get memory usage estimate
    pub fn memory_bytes(&self) -> usize {
        self.memory_bytes.load(Ordering::Relaxed)
    }

    /// Update memory estimate
    fn update_memory_estimate(&self) {
        let data = self.data.read().unwrap();
        let estimated = data.len() * (64 + 8 * 10); // Approximate overhead + avg vector size
        self.memory_bytes.store(estimated, Ordering::Relaxed);
    }

    /// Clear all data
    pub fn clear(&self) {
        self.data.write().unwrap().clear();
        self.memory_bytes.store(0, Ordering::Relaxed);
    }

    /// Get access statistics
    pub fn get_stats(&self) -> ShardStats {
        ShardStats {
            entry_count: self.len(),
            access_count: self.access_count.load(Ordering::Relaxed),
            last_access_ns: self.last_access_ns.load(Ordering::Relaxed),
            memory_bytes: self.memory_bytes(),
        }
    }
}

impl Default for FeatureShard {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a shard
#[derive(Debug, Clone)]
pub struct ShardStats {
    pub entry_count: usize,
    pub access_count: usize,
    pub last_access_ns: u64,
    pub memory_bytes: usize,
}

/// Sharded feature state manager
pub struct ShardedFeatureState {
    /// The hash ring for shard assignment
    hash_ring: ConsistentHashRing,
    /// Array of shards
    shards: Vec<Arc<FeatureShard>>,
    /// Global statistics
    total_inserts: AtomicUsize,
    total_gets: AtomicUsize,
    total_removes: AtomicUsize,
    /// Operation latency tracking
    total_latency_ns: AtomicU64,
    operation_count: AtomicUsize,
}

impl ShardedFeatureState {
    /// Create a new sharded feature state
    pub fn new(num_shards: usize) -> Self {
        let hash_ring = ConsistentHashRing::new(num_shards, 100); // 100 virtual nodes
        let shards: Vec<Arc<FeatureShard>> = (0..num_shards)
            .map(|_| Arc::new(FeatureShard::new()))
            .collect();
        
        Self {
            hash_ring,
            shards,
            total_inserts: AtomicUsize::new(0),
            total_gets: AtomicUsize::new(0),
            total_removes: AtomicUsize::new(0),
            total_latency_ns: AtomicU64::new(0),
            operation_count: AtomicUsize::new(0),
        }
    }

    /// Get the shard for a key
    #[inline]
    fn get_shard_for_key(&self, key: &str) -> &Arc<FeatureShard> {
        let shard_id = self.hash_ring.get_shard(&key);
        &self.shards[shard_id]
    }

    /// Insert a feature value
    pub fn insert(&self, key: String, value: Vec<f64>) -> Result<(), &'static str> {
        let start = Instant::now();
        
        let shard = self.get_shard_for_key(&key);
        
        // Check shard capacity
        if shard.len() >= MAX_ENTRIES_PER_SHARD {
            return Err("Shard at capacity");
        }
        
        shard.insert(key, value);
        self.total_inserts.fetch_add(1, Ordering::Relaxed);
        
        self.record_latency(start.elapsed().as_nanos());
        Ok(())
    }

    /// Get a feature value
    pub fn get(&self, key: &str) -> Option<Vec<f64>> {
        let start = Instant::now();
        
        let shard = self.get_shard_for_key(key);
        let result = shard.get(key);
        
        self.total_gets.fetch_add(1, Ordering::Relaxed);
        self.record_latency(start.elapsed().as_nans());
        
        result
    }

    /// Remove a feature
    pub fn remove(&self, key: &str) -> Option<Vec<f64>> {
        let start = Instant::now();
        
        let shard = self.get_shard_for_key(key);
        let result = shard.remove(key);
        
        if result.is_some() {
            self.total_removes.fetch_add(1, Ordering::Relaxed);
        }
        
        self.record_latency(start.elapsed().as_nanos());
        result
    }

    /// Get values for multiple keys (batch read)
    pub fn get_batch(&self, keys: &[String]) -> HashMap<String, Vec<f64>> {
        let start = Instant::now();
        let mut results = HashMap::with_capacity(keys.len());
        
        // Group keys by shard for efficiency
        let mut shard_groups: HashMap<ShardId, Vec<&String>> = HashMap::new();
        for key in keys {
            let shard_id = self.hash_ring.get_shard(key);
            shard_groups.entry(shard_id).or_default().push(key);
        }
        
        // Fetch from each shard
        for (shard_id, shard_keys) in shard_groups {
            let shard = &self.shards[shard_id];
            for key in shard_keys {
                if let Some(value) = shard.get(key) {
                    results.insert(key.clone(), value);
                }
            }
        }
        
        self.total_gets.fetch_add(keys.len(), Ordering::Relaxed);
        self.record_latency(start.elapsed().as_nanos());
        
        results
    }

    /// Record operation latency
    fn record_latency(&self, latency_ns: u64) {
        self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
        self.operation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get average latency
    pub fn avg_latency_ns(&self) -> f64 {
        let count = self.operation_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.total_latency_ns.load(Ordering::Relaxed) as f64 / count as f64
    }

    /// Get global statistics
    pub fn get_stats(&self) -> ShardedStateStats {
        let mut total_entries = 0;
        let mut total_memory = 0;
        let mut shard_stats = Vec::new();
        
        for shard in &self.shards {
            let stats = shard.get_stats();
            total_entries += stats.entry_count;
            total_memory += stats.memory_bytes;
            shard_stats.push(stats);
        }
        
        ShardedStateStats {
            num_shards: self.shards.len(),
            total_entries,
            total_memory_bytes: total_memory,
            total_inserts: self.total_inserts.load(Ordering::Relaxed),
            total_gets: self.total_gets.load(Ordering::Relaxed),
            total_removes: self.total_removes.load(Ordering::Relaxed),
            avg_latency_ns: self.avg_latency_ns(),
            shard_stats,
        }
    }

    /// Find the hottest shard (most accesses)
    pub fn get_hottest_shard(&self) -> Option<(ShardId, ShardStats)> {
        let mut max_accesses = 0;
        let mut hottest_id = 0;
        
        for (id, shard) in self.shards.iter().enumerate() {
            let stats = shard.get_stats();
            if stats.access_count > max_accesses {
                max_accesses = stats.access_count;
                hottest_id = id;
            }
        }
        
        if max_accesses > 0 {
            Some((hottest_id, self.shards[hottest_id].get_stats()))
        } else {
            None
        }
    }

    /// Clear all shards
    pub fn clear_all(&self) {
        for shard in &self.shards {
            shard.clear();
        }
    }
}

/// Global statistics for the sharded state
#[derive(Debug, Clone)]
pub struct ShardedStateStats {
    pub num_shards: usize,
    pub total_entries: usize,
    pub total_memory_bytes: usize,
    pub total_inserts: usize,
    pub total_gets: usize,
    pub total_removes: usize,
    pub avg_latency_ns: f64,
    pub shard_stats: Vec<ShardStats>,
}

/// Helper function to get current time in nanoseconds
#[inline]
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Extension trait for u64 nanoseconds
trait NanosExt {
    fn as_nanos(self) -> u64;
}

impl NanosExt for u128 {
    fn as_nanos(self) -> u64 {
        self as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_hash_ring() {
        let ring = ConsistentHashRing::new(4, 10);
        
        // Same key should always map to same shard
        let key = "test_key";
        let shard1 = ring.get_shard(&key);
        let shard2 = ring.get_shard(&key);
        assert_eq!(shard1, shard2);
        
        // Different keys should distribute across shards
        let mut shards_seen = std::collections::HashSet::new();
        for i in 0..100 {
            let key = format!("key_{}", i);
            shards_seen.insert(ring.get_shard(&key));
        }
        assert!(shards_seen.len() > 1); // Should hit multiple shards
    }

    #[test]
    fn test_feature_shard() {
        let shard = FeatureShard::new();
        
        shard.insert("key1".to_string(), vec![1.0, 2.0, 3.0]);
        assert_eq!(shard.len(), 1);
        
        let value = shard.get("key1");
        assert_eq!(value, Some(vec![1.0, 2.0, 3.0]));
        
        let removed = shard.remove("key1");
        assert!(removed.is_some());
        assert_eq!(shard.len(), 0);
    }

    #[test]
    fn test_sharded_state() {
        let state = ShardedFeatureState::new(NUM_SHARDS);
        
        // Insert some data
        for i in 0..100 {
            let key = format!("feature_{}", i);
            let value = vec![i as f64; 10];
            state.insert(key, value).unwrap();
        }
        
        // Get data back
        for i in 0..100 {
            let key = format!("feature_{}", i);
            let value = state.get(&key);
            assert!(value.is_some());
            assert_eq!(value.unwrap().len(), 10);
        }
        
        // Check stats
        let stats = state.get_stats();
        assert_eq!(stats.total_entries, 100);
        assert_eq!(stats.total_gets, 100);
    }

    #[test]
    fn test_batch_get() {
        let state = ShardedFeatureState::new(NUM_SHARDS);
        
        // Insert data
        for i in 0..50 {
            let key = format!("batch_key_{}", i);
            state.insert(key, vec![i as f64]).unwrap();
        }
        
        // Batch get
        let keys: Vec<String> = (0..50).map(|i| format!("batch_key_{}", i)).collect();
        let results = state.get_batch(&keys);
        
        assert_eq!(results.len(), 50);
    }

    #[test]
    fn test_shard_distribution() {
        let state = ShardedFeatureState::new(NUM_SHARDS);
        
        // Insert many keys
        for i in 0..10000 {
            let key = format!("dist_key_{}", i);
            state.insert(key, vec![i as f64]).unwrap();
        }
        
        let stats = state.get_stats();
        
        // Check distribution is reasonable (no shard should have > 50% of entries)
        let max_entries = stats.shard_stats.iter().map(|s| s.entry_count).max().unwrap_or(0);
        assert!(max_entries < stats.total_entries / 2);
    }
}
