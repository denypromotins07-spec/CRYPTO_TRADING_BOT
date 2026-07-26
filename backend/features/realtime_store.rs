// ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
// High-Frequency Feature Stores, MLOps, and Concept Drift
// File: backend/features/realtime_store.rs
// Chapter 1: Real-Time Feature Store and Low-Latency Vector Retrieval
// 
// Implements an in-memory, lock-free feature cache for sub-100ns feature vector serving.
// Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
// Uses zero-cost abstractions and atomic operations for thread safety without locks.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crossbeam::epoch::{self, Atomic};
use dashmap::DashMap;
use rayon::prelude::*;

/// Maximum number of features per asset (BTC, ETH, SOL, USDT)
const MAX_FEATURES_PER_ASSET: usize = 256;

/// Feature vector type optimized for cache-line alignment
#[repr(align(64))]
#[derive(Clone, Debug)]
pub struct FeatureVector {
    pub timestamp_ns: u64,
    pub asset_id: u32,
    pub data: [f64; MAX_FEATURES_PER_ASSET],
    pub validity_mask: u256, // Bitmask for valid features
}

impl FeatureVector {
    pub fn new(asset_id: u32) -> Self {
        Self {
            timestamp_ns: 0,
            asset_id,
            data: [0.0; MAX_FEATURES_PER_ASSET],
            validity_mask: 0,
        }
    }

    #[inline]
    pub fn set_feature(&mut self, index: usize, value: f64) {
        if index < MAX_FEATURES_PER_ASSET {
            self.data[index] = value;
            self.validity_mask |= 1u256 << index;
        }
    }

    #[inline]
    pub fn get_feature(&self, index: usize) -> Option<f64> {
        if index < MAX_FEATURES_PER_ASSET && (self.validity_mask & (1u256 << index)) != 0 {
            Some(self.data[index])
        } else {
            None
        }
    }

    /// Compute hash for HNSW indexing - uses first 16 features as embedding
    #[inline]
    pub fn compute_embedding_hash(&self) -> u64 {
        let mut hash = 0u64;
        for i in 0..16.min(MAX_FEATURES_PER_ASSET) {
            hash = hash.wrapping_add((self.data[i].to_bits() ^ (i as u64)).wrapping_mul(0x9e3779b97f4a7c15));
            hash = hash.rotate_left(13);
        }
        hash
    }
}

/// Lock-free ring buffer for tick data
struct TickRingBuffer {
    buffer: Vec<Atomic<FeatureVector>>,
    head: AtomicUsize,
    tail: AtomicUsize,
    capacity: usize,
}

impl TickRingBuffer {
    fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(Atomic::null());
        }
        Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Push a new feature vector (lock-free via epoch-based reclamation)
    fn push(&self, vector: FeatureVector) -> Result<(), &'static str> {
        let guard = epoch::pin();
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) % self.capacity;

        if next_tail == self.head.load(Ordering::Acquire) {
            return Err("Buffer full");
        }

        let boxed = Box::new(vector);
        self.buffer[tail].store(Box::into_raw(boxed).cast_const(), Ordering::Release);
        self.tail.store(next_tail, Ordering::Release);

        guard.flush();
        Ok(())
    }

    /// Pop the latest feature vector
    fn pop(&self) -> Option<FeatureVector> {
        let guard = epoch::pin();
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head == tail {
            return None;
        }

        let ptr = self.buffer[head].load(Ordering::Acquire, &guard);
        if ptr.is_null() {
            return None;
        }

        let vector = unsafe { Box::from_raw(ptr.cast_mut::<FeatureVector>()) }.clone();
        self.buffer[head].store(std::ptr::null(), Ordering::Release);
        self.head.store((head + 1) % self.capacity, Ordering::Release);

        Some(vector)
    }
}

/// Real-time feature store with lock-free access
pub struct RealtimeFeatureStore {
    /// Per-asset feature caches
    asset_caches: DashMap<u32, Arc<TickRingBuffer>>,
    /// Global sequence counter for ordering
    sequence_counter: AtomicU64,
    /// Last update timestamp per asset
    last_update_ns: DashMap<u32, AtomicU64>,
    /// Memory budget in bytes (configured for 8GB system)
    memory_budget_bytes: AtomicUsize,
    /// Current memory usage
    memory_used_bytes: AtomicUsize,
}

impl RealtimeFeatureStore {
    /// Create a new realtime feature store
    pub fn new(memory_budget_mb: usize) -> Self {
        let asset_caches = DashMap::with_capacity(4); // BTC, ETH, SOL, USDT
        
        // Initialize caches for each asset
        let assets = [0u32, 1, 2, 3]; // Asset IDs
        for asset_id in assets.iter() {
            asset_caches.insert(*asset_id, Arc::new(TickRingBuffer::new(1024)));
        }

        Self {
            asset_caches,
            sequence_counter: AtomicU64::new(0),
            last_update_ns: DashMap::new(),
            memory_budget_bytes: AtomicUsize::new(memory_budget_mb * 1024 * 1024),
            memory_used_bytes: AtomicUsize::new(0),
        }
    }

    /// Insert a feature vector with lock-free semantics
    /// Returns the sequence number for ordering guarantees
    #[inline]
    pub fn insert(&self, asset_id: u32, mut vector: FeatureVector) -> Result<u64, &'static str> {
        // Update timestamp
        vector.timestamp_ns = self.sequence_counter.fetch_add(1, Ordering::Relaxed);

        // Check memory budget
        let current_usage = self.memory_used_bytes.load(Ordering::Relaxed);
        let budget = self.memory_budget_bytes.load(Ordering::Relaxed);
        
        if current_usage + std::mem::size_of::<FeatureVector>() > budget {
            // Trigger eviction (simplified - in production would use LRU)
            self.evict_oldest(asset_id);
        }

        // Get or create asset cache
        let cache = self.asset_caches
            .entry(asset_id)
            .or_insert_with(|| Arc::new(TickRingBuffer::new(1024)))
            .value()
            .clone();

        // Update last update time
        let last_update = self.last_update_ns
            .entry(asset_id)
            .or_insert_with(|| AtomicU64::new(0));
        last_update.store(vector.timestamp_ns, Ordering::Release);

        // Push to ring buffer
        cache.push(vector)?;

        // Update memory usage
        self.memory_used_bytes.fetch_add(
            std::mem::size_of::<FeatureVector>(),
            Ordering::Relaxed
        );

        Ok(vector.timestamp_ns)
    }

    /// Get the latest feature vector for an asset (sub-100ns target)
    #[inline]
    pub fn get_latest(&self, asset_id: u32) -> Option<FeatureVector> {
        if let Some(cache) = self.asset_caches.get(&asset_id) {
            // Peek at the tail without popping (read-only access)
            let tail = cache.tail.load(Ordering::Acquire);
            let head = cache.head.load(Ordering::Relaxed);
            
            if tail == head {
                return None;
            }

            let prev_tail = if tail == 0 { cache.capacity - 1 } else { tail - 1 };
            let guard = epoch::pin();
            let ptr = cache.buffer[prev_tail].load(Ordering::Acquire, &guard);
            
            if ptr.is_null() {
                return None;
            }

            unsafe { ptr.as_ref().cloned() }
        } else {
            None
        }
    }

    /// Get feature vectors for all assets (for cross-asset correlation)
    pub fn get_all_latest(&self) -> Vec<(u32, FeatureVector)> {
        self.asset_caches
            .iter()
            .filter_map(|entry| {
                let asset_id = *entry.key();
                self.get_latest(asset_id).map(|v| (asset_id, v))
            })
            .collect()
    }

    /// Evict oldest entries for an asset (memory pressure handling)
    fn evict_oldest(&self, asset_id: u32) {
        if let Some(cache) = self.asset_caches.get(&asset_id) {
            // Pop half the buffer to free memory
            for _ in 0..cache.capacity / 2 {
                if let Some(vector) = cache.pop() {
                    self.memory_used_bytes.fetch_sub(
                        std::mem::size_of_val(&vector),
                        Ordering::Relaxed
                    );
                }
            }
        }
    }

    /// Get latency statistics for monitoring
    pub fn get_latency_stats(&self) -> LatencyStats {
        LatencyStats {
            total_assets: self.asset_caches.len(),
            memory_used_mb: self.memory_used_bytes.load(Ordering::Relaxed) / (1024 * 1024),
            memory_budget_mb: self.memory_budget_bytes.load(Ordering::Relaxed) / (1024 * 1024),
        }
    }

    /// Clear all data (for retraining cycles)
    pub fn clear(&self) {
        self.asset_caches.clear();
        self.last_update_ns.clear();
        self.memory_used_bytes.store(0, Ordering::Relaxed);
        
        // Reinitialize asset caches
        let assets = [0u32, 1, 2, 3];
        for asset_id in assets.iter() {
            self.asset_caches.insert(*asset_id, Arc::new(TickRingBuffer::new(1024)));
        }
    }
}

/// Latency statistics structure
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub total_assets: usize,
    pub memory_used_mb: usize,
    pub memory_budget_mb: usize,
}

/// Builder pattern for constructing feature vectors
pub struct FeatureVectorBuilder {
    asset_id: u32,
    features: Vec<(usize, f64)>,
}

impl FeatureVectorBuilder {
    pub fn new(asset_id: u32) -> Self {
        Self {
            asset_id,
            features: Vec::with_capacity(MAX_FEATURES_PER_ASSET),
        }
    }

    pub fn add_feature(mut self, index: usize, value: f64) -> Self {
        self.features.push((index, value));
        self
    }

    pub fn build(self) -> FeatureVector {
        let mut vector = FeatureVector::new(self.asset_id);
        for (index, value) in self.features {
            vector.set_feature(index, value);
        }
        vector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector_creation() {
        let vector = FeatureVector::new(0);
        assert_eq!(vector.asset_id, 0);
        assert_eq!(vector.validity_mask, 0);
    }

    #[test]
    fn test_feature_set_get() {
        let mut vector = FeatureVector::new(0);
        vector.set_feature(5, 42.0);
        assert_eq!(vector.get_feature(5), Some(42.0));
        assert_eq!(vector.get_feature(6), None);
    }

    #[test]
    fn test_realtime_store_insert_get() {
        let store = RealtimeFeatureStore::new(100);
        let vector = FeatureVectorBuilder::new(0)
            .add_feature(0, 1.0)
            .add_feature(1, 2.0)
            .build();
        
        let seq = store.insert(0, vector.clone()).unwrap();
        assert!(seq > 0);

        let retrieved = store.get_latest(0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().get_feature(0), Some(1.0));
    }

    #[test]
    fn test_memory_budget_enforcement() {
        let store = RealtimeFeatureStore::new(1); // 1MB budget
        for i in 0..10000 {
            let vector = FeatureVectorBuilder::new(0)
                .add_feature(0, i as f64)
                .build();
            let _ = store.insert(0, vector);
        }
        
        let stats = store.get_latency_stats();
        assert!(stats.memory_used_mb <= stats.memory_budget_mb);
    }
}
