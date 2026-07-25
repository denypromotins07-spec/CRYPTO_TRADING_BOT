//! Depth Dynamics Tracker for Order Book Shape Analysis
//!
//! This module tracks the elasticity and shape changes of the L2 order book,
//! providing insights into liquidity distribution and market microstructure.
//!
//! # Features
//! - Real-time depth profile tracking
//! - Elasticity measurement (responsiveness to trades)
//! - Shape factor analysis (concentration vs dispersion)
//! - Memory-efficient sliding window implementation

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;

/// Configuration for depth dynamics tracking
#[derive(Debug, Clone)]
pub struct DepthConfig {
    /// Number of price levels to track on each side
    pub n_levels: usize,
    /// Maximum events in history
    pub max_history: usize,
    /// Tick size for the asset
    pub tick_size: f64,
    /// Minimum depth threshold
    pub min_depth: f64,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            n_levels: 20,
            max_history: 1000,
            tick_size: 0.01,
            min_depth: 0.001,
        }
    }
}

/// Snapshot of order book depth at a point in time
#[derive(Debug, Clone)]
pub struct DepthSnapshot {
    /// Timestamp in microseconds
    pub timestamp: u128,
    /// Bid depths at each level (volume)
    pub bid_depths: Vec<f64>,
    /// Ask depths at each level (volume)
    pub ask_depths: Vec<f64>,
    /// Best bid price
    pub best_bid: f64,
    /// Best ask price
    pub best_ask: f64,
    /// Mid price
    pub mid_price: f64,
    /// Spread in ticks
    pub spread_ticks: u64,
}

impl DepthSnapshot {
    /// Create a new depth snapshot
    pub fn new(
        bid_depths: Vec<f64>,
        ask_depths: Vec<f64>,
        best_bid: f64,
        best_ask: f64,
    ) -> Self {
        let mid_price = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;
        let spread_ticks = ((spread / 0.01).round() as u64).max(1);
        
        Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros(),
            bid_depths,
            ask_depths,
            best_bid,
            best_ask,
            mid_price,
            spread_ticks,
        }
    }
    
    /// Calculate total depth within N levels
    pub fn total_depth(&self, levels: usize) -> (f64, f64) {
        let levels = levels.min(self.bid_depths.len()).min(self.ask_depths.len());
        let bid_total: f64 = self.bid_depths[..levels].iter().sum();
        let ask_total: f64 = self.ask_depths[..levels].iter().sum();
        (bid_total, ask_total)
    }
    
    /// Calculate depth imbalance
    pub fn imbalance(&self, levels: usize) -> f64 {
        let (bid_total, ask_total) = self.total_depth(levels);
        let sum = bid_total + ask_total;
        if sum > 0.0 {
            (bid_total - ask_total) / sum
        } else {
            0.0
        }
    }
    
    /// Calculate weighted average price (WAP)
    pub fn wap(&self, levels: usize) -> (f64, f64) {
        let levels = levels.min(self.bid_depths.len()).min(self.ask_depths.len());
        
        let mut bid_wap = 0.0;
        let mut bid_volume = 0.0;
        for i in 0..levels {
            let price = self.best_bid - (i as f64 * 0.01);
            let volume = self.bid_depths[i];
            bid_wap += price * volume;
            bid_volume += volume;
        }
        if bid_volume > 0.0 {
            bid_wap /= bid_volume;
        }
        
        let mut ask_wap = 0.0;
        let mut ask_volume = 0.0;
        for i in 0..levels {
            let price = self.best_ask + (i as f64 * 0.01);
            let volume = self.ask_depths[i];
            ask_wap += price * volume;
            ask_volume += volume;
        }
        if ask_volume > 0.0 {
            ask_wap /= ask_volume;
        }
        
        (bid_wap, ask_wap)
    }
}

/// Metrics derived from depth analysis
#[derive(Debug, Clone)]
pub struct DepthMetrics {
    /// Total bid depth
    pub total_bid_depth: f64,
    /// Total ask depth
    pub total_ask_depth: f64,
    /// Depth ratio (bid/ask)
    pub depth_ratio: f64,
    /// Imbalance [-1, 1]
    pub imbalance: f64,
    /// Concentration factor (higher = more concentrated near top)
    pub concentration: f64,
    /// Elasticity estimate
    pub elasticity: f64,
    /// Shape factor (logarithmic slope)
    pub shape_factor: f64,
}

/// Depth dynamics tracker with historical analysis
pub struct DepthDynamicsTracker {
    config: DepthConfig,
    /// History of depth snapshots
    history: VecDeque<DepthSnapshot>,
    /// Current snapshot
    current: Option<DepthSnapshot>,
    /// Running statistics
    avg_concentration: f64,
    avg_elasticity: f64,
    sample_count: usize,
}

impl DepthDynamicsTracker {
    /// Create a new depth dynamics tracker
    pub fn new(config: DepthConfig) -> Self {
        Self {
            config,
            history: VecDeque::with_capacity(config.max_history),
            current: None,
            avg_concentration: 0.0,
            avg_elasticity: 0.0,
            sample_count: 0,
        }
    }
    
    /// Update with new depth snapshot
    pub fn update(&mut self, snapshot: DepthSnapshot) {
        // Calculate metrics before updating
        if let Some(ref prev) = self.current {
            self.update_statistics(prev, &snapshot);
        }
        
        self.current = Some(snapshot);
        
        // Add to history
        if self.history.len() >= self.config.max_history {
            self.history.pop_front();
        }
        if let Some(ref curr) = self.current {
            self.history.push_back(curr.clone());
        }
    }
    
    /// Update running statistics
    fn update_statistics(&mut self, prev: &DepthSnapshot, curr: &DepthSnapshot) {
        self.sample_count += 1;
        
        // Calculate concentration (how much depth is near the top)
        let (bid_top, bid_total) = curr.total_depth(5);
        let bid_concentration = if bid_total > 0.0 { bid_top / bid_total } else { 0.0 };
        
        let (ask_top, ask_total) = curr.total_depth(5);
        let ask_concentration = if ask_total > 0.0 { ask_top / ask_total } else { 0.0 };
        
        let concentration = (bid_concentration + ask_concentration) / 2.0;
        
        // Calculate elasticity (change in depth relative to price change)
        let price_change = (curr.mid_price - prev.mid_price).abs();
        let (_, prev_total_bid) = prev.total_depth(10);
        let (_, curr_total_bid) = curr.total_depth(10);
        let depth_change = (curr_total_bid - prev_total_bid).abs();
        
        let elasticity = if prev_total_bid > 0.0 && price_change > 0.0 {
            (depth_change / prev_total_bid) / (price_change / prev.mid_price)
        } else {
            0.0
        };
        
        // Update running averages with exponential weighting
        let alpha = 0.1;
        self.avg_concentration = (1.0 - alpha) * self.avg_concentration + alpha * concentration;
        self.avg_elasticity = (1.0 - alpha) * self.avg_elasticity + alpha * elasticity;
    }
    
    /// Calculate shape factor (slope of log-depth vs distance)
    pub fn calculate_shape_factor(&self) -> f64 {
        if let Some(ref snapshot) = self.current {
            // Fit a line to log(depth) vs level
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_xx = 0.0;
            let n = snapshot.bid_depths.len().min(10) as f64;
            
            for i in 0..(n as usize) {
                let x = i as f64;
                let y = if snapshot.bid_depths[i] > 0.0 {
                    snapshot.bid_depths[i].ln()
                } else {
                    0.0
                };
                
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
            }
            
            if n * sum_xx - sum_x * sum_x > 0.0 {
                (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// Get current depth metrics
    pub fn get_metrics(&self) -> Option<DepthMetrics> {
        self.current.as_ref().map(|snapshot| {
            let (bid_total, ask_total) = snapshot.total_depth(self.config.n_levels);
            let sum = bid_total + ask_total;
            
            DepthMetrics {
                total_bid_depth: bid_total,
                total_ask_depth: ask_total,
                depth_ratio: if ask_total > 0.0 { bid_total / ask_total } else { f64::INFINITY },
                imbalance: snapshot.imbalance(self.config.n_levels),
                concentration: self.avg_concentration,
                elasticity: self.avg_elasticity,
                shape_factor: self.calculate_shape_factor(),
            }
        })
    }
    
    /// Check if depth is abnormal compared to recent history
    pub fn is_depth_anomaly(&self, threshold: f64) -> bool {
        if self.history.len() < 10 {
            return false;
        }
        
        if let Some(ref curr) = self.current {
            let (curr_bid, curr_ask) = curr.total_depth(10);
            
            let mut prev_bids = Vec::new();
            let mut prev_asks = Vec::new();
            for snap in &self.history {
                let (b, a) = snap.total_depth(10);
                prev_bids.push(b);
                prev_asks.push(a);
            }
            
            let mean_bid = prev_bids.iter().sum::<f64>() / prev_bids.len() as f64;
            let std_bid = (prev_bids.iter().map(|x| (x - mean_bid).powi(2)).sum::<f64>() 
                          / prev_bids.len() as f64).sqrt();
            
            let mean_ask = prev_asks.iter().sum::<f64>() / prev_asks.len() as f64;
            let std_ask = (prev_asks.iter().map(|x| (x - mean_ask).powi(2)).sum::<f64>() 
                          / prev_asks.len() as f64).sqrt();
            
            let bid_z = if std_bid > 0.0 { (curr_bid - mean_bid).abs() / std_bid } else { 0.0 };
            let ask_z = if std_ask > 0.0 { (curr_ask - mean_ask).abs() / std_ask } else { 0.0 };
            
            bid_z > threshold || ask_z > threshold
        } else {
            false
        }
    }
    
    /// Get recent snapshots
    pub fn get_recent_snapshots(&self, count: usize) -> Vec<DepthSnapshot> {
        self.history.iter().rev().take(count).cloned().collect()
    }
    
    /// Clear history
    pub fn clear(&mut self) {
        self.history.clear();
        self.current = None;
        self.avg_concentration = 0.0;
        self.avg_elasticity = 0.0;
        self.sample_count = 0;
    }
}

/// Thread-safe wrapper
pub struct ThreadSafeDepthTracker {
    inner: Arc<RwLock<DepthDynamicsTracker>>,
}

impl ThreadSafeDepthTracker {
    pub fn new(tracker: DepthDynamicsTracker) -> Self {
        Self {
            inner: Arc::new(RwLock::new(tracker)),
        }
    }
    
    pub fn update(&self, snapshot: DepthSnapshot) {
        let mut t = self.inner.write();
        t.update(snapshot);
    }
    
    pub fn get_metrics(&self) -> Option<DepthMetrics> {
        let t = self.inner.read();
        t.get_metrics()
    }
    
    pub fn is_depth_anomaly(&self, threshold: f64) -> bool {
        let t = self.inner.read();
        t.is_depth_anomaly(threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_depth_snapshot() {
        let snapshot = DepthSnapshot::new(
            vec![10.0, 8.0, 6.0, 4.0, 2.0],
            vec![10.0, 8.0, 6.0, 4.0, 2.0],
            50000.0,
            50000.01,
        );
        
        assert_eq!(snapshot.spread_ticks, 1);
        assert!((snapshot.mid_price - 50000.005).abs() < 0.001);
        
        let (bid, ask) = snapshot.total_depth(3);
        assert!((bid - 24.0).abs() < 0.001);
        assert!((ask - 24.0).abs() < 0.001);
    }
    
    #[test]
    fn test_tracker_update() {
        let config = DepthConfig::default();
        let mut tracker = DepthDynamicsTracker::new(config);
        
        let snapshot1 = DepthSnapshot::new(
            vec![10.0, 8.0, 6.0],
            vec![10.0, 8.0, 6.0],
            50000.0,
            50000.01,
        );
        tracker.update(snapshot1);
        
        let snapshot2 = DepthSnapshot::new(
            vec![12.0, 9.0, 7.0],
            vec![11.0, 8.5, 6.5],
            50000.005,
            50000.015,
        );
        tracker.update(snapshot2);
        
        let metrics = tracker.get_metrics();
        assert!(metrics.is_some());
        let m = metrics.unwrap();
        assert!(m.total_bid_depth > 0.0);
    }
}
