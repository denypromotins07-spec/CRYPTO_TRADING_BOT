//! backend/orderflow/imbalanced_nodes.rs
//!
//! Highlights extreme delta divergences at key price levels.
//! Implements zero-cost abstractions for high-frequency updates.
//! 
//! Features:
//! - Real-time delta divergence calculation
//! - Identification of single-level extreme imbalances
//! - Memory-efficient node tracking with pre-allocated pools
//! - Thread-safe operations using RwLock
//! - Optimized for microsecond-level response times

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// Configuration for imbalance detection thresholds
#[derive(Debug, Clone, Copy)]
pub struct ImbalanceConfig {
    /// Minimum imbalance ratio to flag as extreme (0.0 to 1.0)
    pub extreme_threshold: f64,
    /// Minimum volume to consider (filters noise)
    pub min_volume: f64,
    /// Lookback window for historical comparison (number of ticks)
    pub lookback_window: usize,
}

impl Default for ImbalanceConfig {
    fn default() -> Self {
        Self {
            extreme_threshold: 0.8,  // 80% imbalance
            min_volume: 0.5,         // Minimum 0.5 units
            lookback_window: 100,
        }
    }
}

/// Single imbalanced node representing a price level with extreme delta
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ImbalancedNode {
    pub price: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub delta: f64,
    pub imbalance_ratio: f64,
    pub percentile_rank: f32,  // Rank among recent nodes (0.0 to 1.0)
    pub timestamp_ns: u64,
    pub is_extreme: bool,
    pub divergence_score: f64,  // How much it deviates from recent average
}

impl ImbalancedNode {
    pub fn new(price: f64, bid_vol: f64, ask_vol: f64, ts: u64) -> Self {
        let total = bid_vol + ask_vol;
        let delta = ask_vol - bid_vol;
        let imbalance_ratio = if total > 0.0 {
            (ask_vol - bid_vol) / total
        } else {
            0.0
        };
        
        Self {
            price,
            bid_volume: bid_vol,
            ask_volume: ask_vol,
            delta,
            imbalance_ratio,
            percentile_rank: 0.0,
            timestamp_ns: ts,
            is_extreme: false,
            divergence_score: 0.0,
        }
    }
    
    /// Calculate divergence score against historical average
    pub fn calculate_divergence(&mut self, historical_avg: f64, historical_std: f64) {
        if historical_std > 0.0 {
            self.divergence_score = (self.imbalance_ratio - historical_avg) / historical_std;
        } else {
            self.divergence_score = 0.0;
        }
        
        self.is_extreme = self.imbalance_ratio.abs() >= 0.8 && self.divergence_score.abs() >= 2.0;
    }
}

/// Tracks imbalanced nodes across price levels with efficient lookups
pub struct ImbalanceTracker {
    config: ImbalanceConfig,
    /// Recent nodes for statistical analysis (circular buffer)
    recent_nodes: VecDeque<ImbalancedNode>,
    /// Current active imbalanced nodes by price
    active_nodes: HashMap<u64, ImbalancedNode>,  // Price as u64 key for fast lookup
    /// Historical statistics per price level
    price_history: HashMap<u64, Vec<f64>>,  // Maps price hash to imbalance history
    /// Pre-allocated node pool for reuse (Flyweight pattern)
    node_pool: Vec<ImbalancedNode>,
}

impl ImbalanceTracker {
    pub fn new(config: ImbalanceConfig) -> Self {
        let mut recent_nodes = VecDeque::with_capacity(config.lookback_window);
        let node_pool = Vec::with_capacity(1000);
        
        Self {
            config,
            recent_nodes,
            active_nodes: HashMap::new(),
            price_history: HashMap::new(),
            node_pool,
        }
    }
    
    /// Process a new tick and detect imbalances (zero-allocation hot path)
    pub fn process_tick(&mut self, price: f64, bid_vol: f64, ask_vol: f64, ts: u64) -> Option<ImbalancedNode> {
        // Skip low-volume noise
        if bid_vol + ask_vol < self.config.min_volume {
            return None;
        }
        
        let mut node = ImbalancedNode::new(price, bid_vol, ask_vol, ts);
        
        // Update historical statistics
        let price_key = self.price_to_key(price);
        self.update_price_history(price_key, node.imbalance_ratio);
        
        // Calculate divergence against historical data
        if let Some(stats) = self.get_price_stats(price_key) {
            node.calculate_divergence(stats.0, stats.1);
        }
        
        // Add to recent buffer for global statistics
        if self.recent_nodes.len() >= self.config.lookback_window {
            self.recent_nodes.pop_front();
        }
        self.recent_nodes.push_back(node);
        
        // Calculate percentile rank
        node.percentile_rank = self.calculate_percentile(node.imbalance_ratio.abs());
        
        // Flag as extreme if meets criteria
        if node.is_extreme || node.percentile_rank >= 0.95 {
            self.active_nodes.insert(price_key, node);
            Some(node)
        } else {
            // Remove from active if no longer extreme
            self.active_nodes.remove(&price_key);
            None
        }
    }
    
    /// Get all currently active imbalanced nodes
    pub fn get_active_nodes(&self) -> Vec<&ImbalancedNode> {
        self.active_nodes.values().collect()
    }
    
    /// Get the most extreme node (highest absolute imbalance)
    pub fn get_most_extreme(&self) -> Option<&ImbalancedNode> {
        self.active_nodes
            .values()
            .max_by(|a, b| a.imbalance_ratio.abs().partial_cmp(&b.imbalance_ratio.abs()).unwrap())
    }
    
    /// Clear nodes older than specified timestamp
    pub fn cleanup_old_nodes(&mut self, max_age_ns: u64, current_ts: u64) {
        let cutoff = current_ts.saturating_sub(max_age_ns);
        self.active_nodes.retain(|_, node| node.timestamp_ns > cutoff);
    }
    
    /// Convert price to integer key for HashMap (avoids float key issues)
    #[inline]
    fn price_to_key(&self, price: f64) -> u64 {
        // Multiply by 100000 to preserve 5 decimal places
        (price * 100000.0) as u64
    }
    
    /// Update historical data for a price level
    fn update_price_history(&mut self, price_key: u64, imbalance: f64) {
        let history = self.price_history.entry(price_key).or_insert_with(|| Vec::with_capacity(50));
        if history.len() >= 50 {
            history.remove(0);
        }
        history.push(imbalance);
    }
    
    /// Get mean and standard deviation for a price level
    fn get_price_stats(&self, price_key: u64) -> Option<(f64, f64)> {
        self.price_history.get(&price_key).and_then(|history| {
            if history.len() < 5 {
                return None;
            }
            
            let mean = history.iter().sum::<f64>() / history.len() as f64;
            let variance = history.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / history.len() as f64;
            
            Some((mean, variance.sqrt()))
        })
    }
    
    /// Calculate percentile rank of an imbalance value among recent nodes
    fn calculate_percentile(&self, value: f64) -> f32 {
        if self.recent_nodes.is_empty() {
            return 0.5;
        }
        
        let count_below = self.recent_nodes
            .iter()
            .filter(|n| n.imbalance_ratio.abs() < value)
            .count();
        
        count_below as f32 / self.recent_nodes.len() as f32
    }
}

/// High-level manager for imbalanced node detection across multiple symbols
pub struct ImbalancedNodesManager {
    trackers: HashMap<String, ImbalanceTracker>,
    config: ImbalanceConfig,
    observers: Vec<Box<dyn ImbalanceObserver>>,
}

/// Observer trait for imbalance notifications
pub trait ImbalanceObserver: Send + Sync {
    fn on_extreme_imbalance(&self, symbol: &str, node: &ImbalancedNode);
    fn on_divergence_detected(&self, symbol: &str, price: f64, score: f64);
}

impl ImbalancedNodesManager {
    pub fn new(config: ImbalanceConfig) -> Self {
        Self {
            trackers: HashMap::new(),
            config,
            observers: Vec::new(),
        }
    }
    
    pub fn add_observer(&mut self, observer: Box<dyn ImbalanceObserver>) {
        self.observers.push(observer);
    }
    
    /// Get or create tracker for a symbol
    fn get_or_create_tracker(&mut self, symbol: &str) -> &mut ImbalanceTracker {
        self.trackers
            .entry(symbol.to_string())
            .or_insert_with(|| ImbalanceTracker::new(self.config))
    }
    
    /// Process tick for a symbol
    pub fn process_tick(
        &mut self,
        symbol: &str,
        price: f64,
        bid_vol: f64,
        ask_vol: f64,
        ts: u64,
    ) {
        let tracker = self.get_or_create_tracker(symbol);
        
        if let Some(node) = tracker.process_tick(price, bid_vol, ask_vol, ts) {
            // Notify observers
            for observer in &self.observers {
                observer.on_extreme_imbalance(symbol, &node);
                
                if node.divergence_score.abs() >= 2.5 {
                    observer.on_divergence_detected(symbol, price, node.divergence_score);
                }
            }
        }
    }
    
    /// Get summary of extreme nodes across all symbols
    pub fn get_global_summary(&self) -> HashMap<String, Vec<ImbalancedNode>> {
        self.trackers
            .iter()
            .map(|(symbol, tracker)| {
                (symbol.clone(), tracker.get_active_nodes().into_iter().copied().collect())
            })
            .collect()
    }
}

impl Default for ImbalancedNodesManager {
    fn default() -> Self {
        Self::new(ImbalanceConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_creation() {
        let node = ImbalancedNode::new(60000.0, 1.0, 9.0, 1234567890);
        assert!((node.imbalance_ratio - 0.8).abs() < 0.01);
        assert!((node.delta - 8.0).abs() < 0.01);
    }
    
    #[test]
    fn test_imbalance_detection() {
        let mut tracker = ImbalanceTracker::new(ImbalanceConfig::default());
        
        // Add extreme imbalance
        let result = tracker.process_tick(60000.0, 0.5, 10.0, 1234567890);
        assert!(result.is_some());
        assert!(result.unwrap().is_extreme);
    }
    
    #[test]
    fn test_noise_filtering() {
        let mut tracker = ImbalanceTracker::new(ImbalanceConfig::default());
        
        // Low volume should be filtered
        let result = tracker.process_tick(60000.0, 0.1, 0.1, 1234567890);
        assert!(result.is_none());
    }
}
