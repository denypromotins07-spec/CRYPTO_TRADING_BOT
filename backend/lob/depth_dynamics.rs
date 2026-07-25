//! Depth Dynamics Tracker for Order Book Shape Analysis
//!
//! This module tracks the elasticity and shape changes of the L2 order book,
//! measuring how depth evolves over time and responds to market events.
//!
//! Key Features:
//! - Real-time depth profile tracking
//! - Elasticity measurement (depth response to price)
//! - Shape anomaly detection
//! - Zero-cost abstractions with strict borrowing checks

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for depth dynamics tracking
#[derive(Debug, Clone)]
pub struct DepthConfig {
    /// Number of price levels to track
    pub n_levels: usize,
    /// Tick size for the asset
    pub tick_size: f64,
    /// History window for dynamics calculation
    pub history_window: Duration,
    /// Anomaly detection threshold (standard deviations)
    pub anomaly_threshold: f64,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            n_levels: 20,
            tick_size: 0.01,
            history_window: Duration::from_secs(60),
            anomaly_threshold: 3.0,
        }
    }
}

/// Order book depth at a single level
#[derive(Debug, Clone)]
pub struct DepthLevel {
    /// Price level
    pub price: f64,
    /// Volume at this level
    pub volume: f64,
    /// Number of orders at this level
    pub order_count: u32,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Snapshot of full order book depth
#[derive(Debug, Clone)]
pub struct DepthSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// Bid side depth (top to bottom)
    pub bids: Vec<DepthLevel>,
    /// Ask side depth (top to bottom)
    pub asks: Vec<DepthLevel>,
    /// Total bid volume
    pub total_bid_volume: f64,
    /// Total ask volume
    pub total_ask_volume: f64,
    /// Weighted average price (bid side)
    pub vwap_bid: f64,
    /// Weighted average price (ask side)
    pub vwap_ask: f64,
}

impl DepthSnapshot {
    /// Calculate spread from snapshot
    pub fn spread(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return f64::INFINITY;
        }
        self.asks[0].price - self.bids[0].price
    }
    
    /// Calculate mid price
    pub fn mid_price(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        (self.bids[0].price + self.asks[0].price) / 2.0
    }
    
    /// Calculate depth imbalance
    pub fn imbalance(&self) -> f64 {
        let total = self.total_bid_volume + self.total_ask_volume;
        if total == 0.0 {
            return 0.0;
        }
        (self.total_bid_volume - self.total_ask_volume) / total
    }
}

/// Elasticity measurement result
#[derive(Debug, Clone)]
pub struct ElasticityMetrics {
    /// Bid side elasticity (volume change per price unit)
    pub bid_elasticity: f64,
    /// Ask side elasticity
    pub ask_elasticity: f64,
    /// Combined elasticity
    pub combined_elasticity: f64,
    /// Elasticity ratio (bid/ask)
    pub elasticity_ratio: f64,
    /// Classification (rigid, normal, elastic)
    pub classification: ElasticityClass,
}

/// Elasticity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticityClass {
    /// Very rigid (low depth response)
    Rigid,
    /// Normal elasticity
    Normal,
    /// Highly elastic (high depth response)
    Elastic,
}

/// Depth dynamics tracker
pub struct DepthDynamicsTracker {
    config: DepthConfig,
    /// Historical snapshots
    history: VecDeque<DepthSnapshot>,
    /// Running statistics for anomaly detection
    depth_mean: f64,
    depth_variance: f64,
    /// Current elasticity estimate
    current_elasticity: Option<ElasticityMetrics>,
    /// Last update time
    last_update: Instant,
}

impl DepthDynamicsTracker {
    /// Create new depth dynamics tracker
    pub fn new(config: DepthConfig) -> Self {
        Self {
            config,
            history: VecDeque::with_capacity(1000),
            depth_mean: 0.0,
            depth_variance: 0.0,
            current_elasticity: None,
            last_update: Instant::now(),
        }
    }
    
    /// Update with new depth snapshot
    pub fn update(&mut self, snapshot: DepthSnapshot) {
        let now = Instant::now();
        
        // Add to history
        self.history.push_back(snapshot);
        
        // Prune old snapshots
        self.prune_history();
        
        // Update running statistics
        self.update_statistics();
        
        // Calculate elasticity
        self.calculate_elasticity();
        
        self.last_update = now;
    }
    
    /// Prune snapshots outside history window
    fn prune_history(&mut self) {
        let cutoff = self.last_update - self.config.history_window;
        
        while let Some(front) = self.history.front() {
            if front.timestamp < cutoff {
                self.history.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Update running statistics for anomaly detection
    fn update_statistics(&mut self) {
        if self.history.len() < 2 {
            return;
        }
        
        // Welford's online algorithm for mean and variance
        let total_volume: f64 = self.history.iter()
            .map(|s| s.total_bid_volume + s.total_ask_volume)
            .sum();
        
        self.depth_mean = total_volume / self.history.len() as f64;
        
        let variance_sum: f64 = self.history.iter()
            .map(|s| {
                let vol = s.total_bid_volume + s.total_ask_volume;
                (vol - self.depth_mean).powi(2)
            })
            .sum();
        
        self.depth_variance = variance_sum / (self.history.len() - 1) as f64;
    }
    
    /// Calculate elasticity from recent depth changes
    fn calculate_elasticity(&mut self) {
        if self.history.len() < 5 {
            return;
        }
        
        // Calculate depth change vs price change correlation
        let mut bid_depth_changes = Vec::new();
        let mut ask_depth_changes = Vec::new();
        let mut price_changes = Vec::new();
        
        let windows: Vec<_> = self.history.iter().collect();
        for i in 1..windows.len().min(10) {
            let prev = windows[i - 1];
            let curr = windows[i];
            
            let depth_change_bid = curr.total_bid_volume - prev.total_bid_volume;
            let depth_change_ask = curr.total_ask_volume - prev.total_ask_volume;
            let price_change = curr.mid_price() - prev.mid_price();
            
            if price_change.abs() > 0.0 {
                bid_depth_changes.push(depth_change_bid / price_change.abs());
                ask_depth_changes.push(depth_change_ask / price_change.abs());
                price_changes.push(price_change);
            }
        }
        
        if bid_depth_changes.is_empty() {
            return;
        }
        
        let bid_elasticity = bid_depth_changes.iter().sum::<f64>() / bid_depth_changes.len() as f64;
        let ask_elasticity = ask_depth_changes.iter().sum::<f64>() / ask_depth_changes.len() as f64;
        let combined = (bid_elasticity.abs() + ask_elasticity.abs()) / 2.0;
        
        // Classify elasticity
        let classification = if combined < 100.0 {
            ElasticityClass::Rigid
        } else if combined < 1000.0 {
            ElasticityClass::Normal
        } else {
            ElasticityClass::Elastic
        };
        
        self.current_elasticity = Some(ElasticityMetrics {
            bid_elasticity,
            ask_elasticity,
            combined_elasticity: combined,
            elasticity_ratio: if ask_elasticity != 0.0 { 
                bid_elasticity / ask_elasticity 
            } else { 
                1.0 
            },
            classification,
        });
    }
    
    /// Check for depth anomaly
    pub fn is_anomaly(&self, current_volume: f64) -> bool {
        if self.depth_variance <= 0.0 {
            return false;
        }
        
        let std_dev = self.depth_variance.sqrt();
        let z_score = (current_volume - self.depth_mean).abs() / std_dev;
        
        z_score > self.config.anomaly_threshold
    }
    
    /// Get current elasticity metrics
    pub fn elasticity(&self) -> Option<&ElasticityMetrics> {
        self.current_elasticity.as_ref()
    }
    
    /// Get depth trend (increasing, stable, decreasing)
    pub fn depth_trend(&self) -> DepthTrend {
        if self.history.len() < 5 {
            return DepthTrend::Unknown;
        }
        
        let recent: Vec<_> = self.history.iter().take(5).collect();
        let older: Vec<_> = self.history.iter().skip(5).take(5).collect();
        
        let recent_avg: f64 = recent.iter()
            .map(|s| s.total_bid_volume + s.total_ask_volume)
            .sum::<f64>() / recent.len() as f64;
        
        let older_avg: f64 = older.iter()
            .map(|s| s.total_bid_volume + s.total_ask_volume)
            .sum::<f64>() / older.len() as f64;
        
        let change_pct = (recent_avg - older_avg) / older_avg.max(1.0);
        
        if change_pct > 0.1 {
            DepthTrend::Increasing
        } else if change_pct < -0.1 {
            DepthTrend::Decreasing
        } else {
            DepthTrend::Stable
        }
    }
    
    /// Get best bid depth (top 3 levels)
    pub fn best_bid_depth(&self) -> f64 {
        self.history.back()
            .map(|s| s.bids.iter().take(3).map(|l| l.volume).sum())
            .unwrap_or(0.0)
    }
    
    /// Get best ask depth (top 3 levels)
    pub fn best_ask_depth(&self) -> f64 {
        self.history.back()
            .map(|s| s.asks.iter().take(3).map(|l| l.volume).sum())
            .unwrap_or(0.0)
    }
}

/// Depth trend classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthTrend {
    Increasing,
    Stable,
    Decreasing,
    Unknown,
}

/// Shape analyzer for detecting order book patterns
pub struct ShapeAnalyzer {
    config: DepthConfig,
}

impl ShapeAnalyzer {
    /// Create new shape analyzer
    pub fn new(config: DepthConfig) -> Self {
        Self { config }
    }
    
    /// Detect spoon pattern (large orders at edges)
    pub fn detect_spoon(&self, snapshot: &DepthSnapshot) -> bool {
        if snapshot.bids.len() < 3 || snapshot.asks.len() < 3 {
            return false;
        }
        
        // Check if outer levels have significantly more volume
        let inner_bid: f64 = snapshot.bids.iter().take(3).map(|l| l.volume).sum();
        let outer_bid: f64 = snapshot.bids.iter().skip(3).take(3).map(|l| l.volume).sum();
        
        outer_bid > inner_bid * 2.0
    }
    
    /// Detect wall pattern (concentrated large order)
    pub fn detect_wall(&self, snapshot: &DepthSnapshot, level: usize) -> bool {
        let levels = if level < snapshot.bids.len() {
            &snapshot.bids
        } else {
            &snapshot.asks
        };
        
        if levels.is_empty() {
            return false;
        }
        
        let target_level = level.min(levels.len() - 1);
        let avg_neighbor: f64 = if target_level > 0 && target_level < levels.len() - 1 {
            (levels[target_level - 1].volume + levels[target_level + 1].volume) / 2.0
        } else {
            levels[target_level].volume
        };
        
        levels[target_level].volume > avg_neighbor * 5.0
    }
    
    /// Detect thinning (decreasing depth toward edges)
    pub fn detect_thinning(&self, snapshot: &DepthSnapshot) -> bool {
        if snapshot.bids.len() < 5 {
            return false;
        }
        
        let inner: f64 = snapshot.bids.iter().take(3).map(|l| l.volume).sum();
        let outer: f64 = snapshot.bids.iter().skip(snapshot.bids.len() - 3).map(|l| l.volume).sum();
        
        inner > outer * 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_depth_snapshot() {
        let now = Instant::now();
        let snapshot = DepthSnapshot {
            timestamp: now,
            bids: vec![
                DepthLevel { price: 99.0, volume: 100.0, order_count: 5, last_update: now },
                DepthLevel { price: 98.0, volume: 200.0, order_count: 10, last_update: now },
            ],
            asks: vec![
                DepthLevel { price: 100.0, volume: 150.0, order_count: 7, last_update: now },
                DepthLevel { price: 101.0, volume: 250.0, order_count: 12, last_update: now },
            ],
            total_bid_volume: 300.0,
            total_ask_volume: 400.0,
            vwap_bid: 98.5,
            vwap_ask: 100.5,
        };
        
        assert!((snapshot.spread() - 1.0).abs() < 1e-10);
        assert!((snapshot.mid_price() - 99.5).abs() < 1e-10);
        assert!(snapshot.imbalance().abs() > 0.1);
    }
}
