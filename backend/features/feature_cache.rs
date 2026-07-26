// ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
// High-Frequency Feature Stores, MLOps, and Concept Drift
// File: backend/features/feature_cache.rs
// Chapter 1: Real-Time Feature Store and Low-Latency Vector Retrieval
//
// Manages TTL-based eviction for high-frequency tick features.
// Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
// Uses zero-cost abstractions and efficient time-based expiration.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use crossbeam::epoch;

/// Default TTL for tick features (in milliseconds)
const DEFAULT_TTL_MS: u64 = 100; // 100ms for high-frequency data

/// Maximum number of entries before forced eviction
const MAX_ENTRIES_PER_ASSET: usize = 10_000;

/// Feature entry with metadata
#[derive(Clone, Debug)]
pub struct FeatureEntry {
    /// Timestamp when the entry was created (nanoseconds since epoch)
    pub created_ns: u64,
    /// Last access timestamp (nanoseconds since epoch)
    pub last_access_ns: u64,
    /// TTL in milliseconds
    pub ttl_ms: u64,
    /// The actual feature data
    pub data: Vec<f64>,
    /// Feature version for schema evolution
    pub version: u32,
    /// Checksum for data integrity
    pub checksum: u64,
}

impl FeatureEntry {
    /// Create a new feature entry
    pub fn new(data: Vec<f64>, ttl_ms: u64, version: u32) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            created_ns: now,
            last_access_ns: now,
            ttl_ms,
            data,
            version,
            checksum: compute_checksum(&data),
        }
    }

    /// Check if the entry has expired
    #[inline]
    pub fn is_expired(&self, current_time_ns: u64) -> bool {
        let age_ms = (current_time_ns - self.created_ns) / 1_000_000;
        age_ms > self.ttl_ms
    }

    /// Update last access time
    #[inline]
    pub fn touch(&mut self) {
        self.last_access_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
    }

    /// Verify data integrity
    #[inline]
    pub fn verify_checksum(&self) -> bool {
        self.checksum == compute_checksum(&self.data)
    }
}

/// Compute a fast checksum for feature data
#[inline]
fn compute_checksum(data: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    let mut hasher = DefaultHasher::new();
    for value in data {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Key type for feature cache
pub type FeatureKey = String;

/// Asset identifier type
pub type AssetId = u32;

/// TTL-based feature cache with LRU eviction
pub struct FeatureCache {
    /// Per-asset feature storage
    assets: RwLock<HashMap<AssetId, BTreeMap<FeatureKey, FeatureEntry>>>,
    /// Access order tracking for LRU (timestamp -> key)
    access_order: RwLock<HashMap<AssetId, BTreeMap<u64, FeatureKey>>>,
    /// Global time source (monotonic clock)
    time_source: AtomicU64,
    /// Entry count per asset
    entry_counts: DashMap<AssetId, AtomicUsize>,
    /// Statistics
    stats: CacheStats,
    /// Memory budget in bytes
    memory_budget_bytes: AtomicUsize,
    /// Current memory usage
    memory_used_bytes: AtomicUsize,
}

impl FeatureCache {
    /// Create a new feature cache with specified memory budget
    pub fn new(memory_budget_mb: usize) -> Self {
        Self {
            assets: RwLock::new(HashMap::with_capacity(4)), // BTC, ETH, SOL, USDT
            access_order: RwLock::new(HashMap::with_capacity(4)),
            time_source: AtomicU64::new(0),
            entry_counts: DashMap::new(),
            stats: CacheStats::default(),
            memory_budget_bytes: AtomicUsize::new(memory_budget_mb * 1024 * 1024),
            memory_used_bytes: AtomicUsize::new(0),
        }
    }

    /// Insert a feature with default TTL
    pub fn insert(
        &self,
        asset_id: AssetId,
        key: FeatureKey,
        data: Vec<f64>,
        version: u32,
    ) -> Result<(), &'static str> {
        self.insert_with_ttl(asset_id, key, data, DEFAULT_TTL_MS, version)
    }

    /// Insert a feature with custom TTL
    pub fn insert_with_ttl(
        &self,
        asset_id: AssetId,
        key: FeatureKey,
        data: Vec<f64>,
        ttl_ms: u64,
        version: u32,
    ) -> Result<(), &'static str> {
        let entry = FeatureEntry::new(data, ttl_ms, version);
        let entry_size = std::mem::size_of::<FeatureEntry>() + entry.data.capacity() * 8;
        
        // Check memory budget
        let current_usage = self.memory_used_bytes.load(Ordering::Relaxed);
        let budget = self.memory_budget_bytes.load(Ordering::Relaxed);
        
        if current_usage + entry_size > budget {
            self.evict_if_needed(asset_id, entry_size);
        }

        // Get or create asset map
        {
            let mut assets = self.assets.write().unwrap();
            let asset_map = assets.entry(asset_id).or_insert_with(BTreeMap::new);
            
            // Check if key already exists
            if asset_map.contains_key(&key) {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
            }
            
            // Insert entry
            let access_time = entry.last_access_ns;
            asset_map.insert(key.clone(), entry);
            
            // Update access order
            let mut access_order = self.access_order.write().unwrap();
            let order_map = access_order.entry(asset_id).or_insert_with(BTreeMap::new);
            order_map.insert(access_time, key);
        }

        // Update counts
        let count = self.entry_counts
            .entry(asset_id)
            .or_insert_with(|| AtomicUsize::new(0));
        count.fetch_add(1, Ordering::Relaxed);

        // Update memory usage
        self.memory_used_bytes.fetch_add(entry_size, Ordering::Relaxed);
        self.stats.insertions.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get a feature by key
    pub fn get(&self, asset_id: AssetId, key: &str) -> Option<Vec<f64>> {
        let current_time_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut assets = self.assets.write().unwrap();
        let asset_map = assets.get_mut(&asset_id)?;
        
        // Check if entry exists and is not expired
        if let Some(entry) = asset_map.get_mut(key) {
            if entry.is_expired(current_time_ns) {
                // Entry expired, remove it
                drop(assets);
                self.remove(asset_id, key);
                self.stats.expirations.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Verify checksum
            if !entry.verify_checksum() {
                self.stats.corruptions.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Update access time
            entry.touch();
            
            // Update access order
            {
                let mut access_order = self.access_order.write().unwrap();
                if let Some(order_map) = access_order.get_mut(&asset_id) {
                    order_map.remove(&entry.last_access_ns);
                    order_map.insert(entry.last_access_ns, key.to_string());
                }
            }

            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.data.clone())
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Remove a feature by key
    pub fn remove(&self, asset_id: AssetId, key: &str) -> bool {
        let mut assets = self.assets.write().unwrap();
        if let Some(asset_map) = assets.get_mut(&asset_id) {
            if let Some(entry) = asset_map.remove(key) {
                let entry_size = std::mem::size_of::<FeatureEntry>() + entry.data.capacity() * 8;
                self.memory_used_bytes.fetch_sub(entry_size, Ordering::Relaxed);
                
                // Update count
                if let Some(count) = self.entry_counts.get(&asset_id) {
                    count.fetch_sub(1, Ordering::Relaxed);
                }
                
                self.stats.removals.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// Evict expired and LRU entries if needed
    pub fn evict_if_needed(&self, asset_id: AssetId, required_bytes: usize) {
        let current_usage = self.memory_used_bytes.load(Ordering::Relaxed);
        let budget = self.memory_budget_bytes.load(Ordering::Relaxed);
        
        // First, evict expired entries
        self.evict_expired(asset_id);
        
        // Check if we freed enough space
        if current_usage + required_bytes <= budget {
            return;
        }

        // Evict LRU entries until we have space
        let entries_to_evict = ((required_bytes as f64) / 1024.0).ceil() as usize;
        
        for _ in 0..entries_to_evict.max(10) {
            if self.memory_used_bytes.load(Ordering::Relaxed) + required_bytes <= budget {
                break;
            }
            self.evict_lru_entry(asset_id);
        }
    }

    /// Evict all expired entries for an asset
    pub fn evict_expired(&self, asset_id: AssetId) -> usize {
        let current_time_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut assets = self.assets.write().unwrap();
        let asset_map = match assets.get_mut(&asset_id) {
            Some(map) => map,
            None => return 0,
        };

        let mut expired_keys = Vec::new();
        for (key, entry) in asset_map.iter() {
            if entry.is_expired(current_time_ns) {
                expired_keys.push(key.clone());
            }
        }

        let evicted_count = expired_keys.len();
        for key in expired_keys {
            if let Some(entry) = asset_map.remove(&key) {
                let entry_size = std::mem::size_of::<FeatureEntry>() + entry.data.capacity() * 8;
                self.memory_used_bytes.fetch_sub(entry_size, Ordering::Relaxed);
            }
        }

        // Update count
        if let Some(count) = self.entry_counts.get(&asset_id) {
            count.fetch_sub(evicted_count, Ordering::Relaxed);
        }

        self.stats.evictions.fetch_add(evicted_count, Ordering::Relaxed);
        evicted_count
    }

    /// Evict the least recently used entry
    fn evict_lru_entry(&self, asset_id: AssetId) -> bool {
        let mut access_order = self.access_order.write().unwrap();
        let order_map = match access_order.get_mut(&asset_id) {
            Some(map) if !map.is_empty() => map,
            _ => return false,
        };

        // Get the oldest access time
        if let Some((access_time, key)) = order_map.iter().next() {
            let key = key.clone();
            let access_time = *access_time;
            
            drop(access_order);
            
            // Remove from access order
            {
                let mut ao = self.access_order.write().unwrap();
                if let Some(map) = ao.get_mut(&asset_id) {
                    map.remove(&access_time);
                }
            }
            
            // Remove from assets
            return self.remove(asset_id, &key);
        }
        
        false
    }

    /// Get statistics about the cache
    pub fn get_stats(&self) -> CacheStatsSnapshot {
        let total_entries: usize = self.entry_counts
            .iter()
            .map(|e| e.value().load(Ordering::Relaxed))
            .sum();

        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };

        CacheStatsSnapshot {
            total_entries,
            hits,
            misses,
            hit_rate,
            insertions: self.stats.insertions.load(Ordering::Relaxed),
            removals: self.stats.removals.load(Ordering::Relaxed),
            expirations: self.stats.expirations.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            corruptions: self.stats.corruptions.load(Ordering::Relaxed),
            memory_used_mb: self.memory_used_bytes.load(Ordering::Relaxed) / (1024 * 1024),
            memory_budget_mb: self.memory_budget_bytes.load(Ordering::Relaxed) / (1024 * 1024),
        }
    }

    /// Clear all entries for an asset
    pub fn clear_asset(&self, asset_id: AssetId) {
        {
            let mut assets = self.assets.write().unwrap();
            if let Some(map) = assets.get_mut(&asset_id) {
                let total_size: usize = map.values()
                    .map(|e| std::mem::size_of::<FeatureEntry>() + e.data.capacity() * 8)
                    .sum();
                self.memory_used_bytes.fetch_sub(total_size, Ordering::Relaxed);
                map.clear();
            }
        }
        
        {
            let mut access_order = self.access_order.write().unwrap();
            if let Some(map) = access_order.get_mut(&asset_id) {
                map.clear();
            }
        }
        
        if let Some(count) = self.entry_counts.get(&asset_id) {
            count.store(0, Ordering::Relaxed);
        }
    }

    /// Clear all entries
    pub fn clear_all(&self) {
        self.assets.write().unwrap().clear();
        self.access_order.write().unwrap().clear();
        self.entry_counts.clear();
        self.memory_used_bytes.store(0, Ordering::Relaxed);
    }
}

/// Cache statistics (atomic counters)
struct CacheStats {
    hits: AtomicUsize,
    misses: AtomicUsize,
    insertions: AtomicUsize,
    removals: AtomicUsize,
    expirations: AtomicUsize,
    evictions: AtomicUsize,
    corruptions: AtomicUsize,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            insertions: AtomicUsize::new(0),
            removals: AtomicUsize::new(0),
            expirations: AtomicUsize::new(0),
            evictions: AtomicUsize::new(0),
            corruptions: AtomicUsize::new(0),
        }
    }
}

/// Snapshot of cache statistics (for reporting)
#[derive(Debug, Clone)]
pub struct CacheStatsSnapshot {
    pub total_entries: usize,
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f64,
    pub insertions: usize,
    pub removals: usize,
    pub expirations: usize,
    pub evictions: usize,
    pub corruptions: usize,
    pub memory_used_mb: usize,
    pub memory_budget_mb: usize,
}

/// Builder for creating feature cache entries
pub struct FeatureEntryBuilder {
    data: Vec<f64>,
    ttl_ms: u64,
    version: u32,
}

impl FeatureEntryBuilder {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            ttl_ms: DEFAULT_TTL_MS,
            version: 1,
        }
    }

    pub fn with_data(mut self, data: Vec<f64>) -> Self {
        self.data = data;
        self
    }

    pub fn with_ttl(mut self, ttl_ms: u64) -> Self {
        self.ttl_ms = ttl_ms;
        self
    }

    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    pub fn build(self) -> FeatureEntry {
        FeatureEntry::new(self.data, self.ttl_ms, self.version)
    }
}

impl Default for FeatureEntryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_feature_entry_creation() {
        let entry = FeatureEntry::new(vec![1.0, 2.0, 3.0], 100, 1);
        assert_eq!(entry.data.len(), 3);
        assert_eq!(entry.ttl_ms, 100);
        assert_eq!(entry.version, 1);
        assert!(entry.verify_checksum());
    }

    #[test]
    fn test_cache_insert_get() {
        let cache = FeatureCache::new(100);
        let data = vec![1.0, 2.0, 3.0];
        
        cache.insert(0, "test_key".to_string(), data.clone(), 1).unwrap();
        
        let retrieved = cache.get(0, "test_key");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);
    }

    #[test]
    fn test_cache_miss() {
        let cache = FeatureCache::new(100);
        let result = cache.get(0, "nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = FeatureCache::new(100);
        
        cache.insert(0, "key1".to_string(), vec![1.0], 100, 1).unwrap();
        cache.insert(0, "key2".to_string(), vec![2.0], 100, 1).unwrap();
        
        cache.get(0, "key1");
        cache.get(0, "key1");
        cache.get(0, "nonexistent");
        
        let stats = cache.get_stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 2.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_budget_enforcement() {
        let cache = FeatureCache::new(1); // 1MB budget
        
        // Insert many entries
        for i in 0..10000 {
            let data = vec![i as f64; 100];
            let _ = cache.insert(0, format!("key_{}", i), data, 1);
        }
        
        let stats = cache.get_stats();
        assert!(stats.memory_used_mb <= stats.memory_budget_mb);
    }
}
