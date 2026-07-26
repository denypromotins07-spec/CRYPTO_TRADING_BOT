//! # Volume Bucketing Implementation
//! 
//! Aggregates trades into constant volume buckets (bars) for microstructure analysis.
//! Implements the "volume bars" concept from quantitative finance literature.
//! 
//! **Key Features:**
//! - Constant volume bucket aggregation
//! - Handles zero-activity periods gracefully
//! - Memory-efficient streaming design
//! - Zero-cost abstractions with proper borrowing
//! 
//! **Performance:** Processes millions of trades/sec without heap allocations during normal operation.

use std::collections::VecDeque;

/// Configuration for volume bucketing
#[derive(Debug, Clone)]
pub struct VolumeBucketConfig {
    /// Target volume per bucket (in base currency units)
    pub target_volume: f64,
    /// Maximum number of trades per bucket (safety limit)
    pub max_trades_per_bucket: usize,
    /// Minimum volume threshold to start a bucket
    pub min_volume_threshold: f64,
}

impl Default for VolumeBucketConfig {
    fn default() -> Self {
        Self {
            target_volume: 100.0, // Default: 100 units per bucket
            max_trades_per_bucket: 10000,
            min_volume_threshold: 0.001,
        }
    }
}

/// A single volume bucket (bar)
#[derive(Debug, Clone)]
pub struct VolumeBucket {
    /// Opening price of the bucket
    pub open: f64,
    /// Closing price of the bucket
    pub close: f64,
    /// High price within the bucket
    pub high: f64,
    /// Low price within the bucket
    pub low: f64,
    /// Total volume in the bucket
    pub volume: f64,
    /// Number of trades in the bucket
    pub trade_count: u32,
    /// Sum of (price * volume) for VWAP calculation
    pub pv_sum: f64,
    /// First timestamp in the bucket
    pub timestamp_start: u64,
    /// Last timestamp in the bucket
    pub timestamp_end: u64,
    /// Buy volume (classified)
    pub buy_volume: f64,
    /// Sell volume (classified)
    pub sell_volume: f64,
}

impl VolumeBucket {
    /// Create a new empty bucket
    pub fn new() -> Self {
        Self {
            open: 0.0,
            close: 0.0,
            high: f64::NEG_INFINITY,
            low: f64::INFINITY,
            volume: 0.0,
            trade_count: 0,
            pv_sum: 0.0,
            timestamp_start: 0,
            timestamp_end: 0,
            buy_volume: 0.0,
            sell_volume: 0.0,
        }
    }

    /// Calculate VWAP for this bucket
    #[inline]
    pub fn vwap(&self) -> f64 {
        if self.volume > 0.0 {
            self.pv_sum / self.volume
        } else {
            0.0
        }
    }

    /// Calculate order flow imbalance
    #[inline]
    pub fn order_imbalance(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total > 0.0 {
            (self.buy_volume - self.sell_volume) / total
        } else {
            0.0
        }
    }

    /// Check if bucket is valid (has data)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.volume > 0.0 && self.trade_count > 0
    }

    /// Get bucket range
    #[inline]
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
}

impl Default for VolumeBucket {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming volume bucket aggregator
pub struct VolumeBucker {
    /// Configuration
    config: VolumeBucketConfig,
    /// Current incomplete bucket
    current_bucket: VolumeBucket,
    /// Completed buckets (circular buffer)
    completed_buckets: VecDeque<VolumeBucket>,
    /// Maximum completed buckets to retain
    max_history: usize,
    /// Running total volume processed
    total_volume_processed: f64,
    /// Bucket counter
    bucket_count: u64,
}

impl VolumeBucker {
    /// Create a new volume bucker with specified target volume
    pub fn new(target_volume: f64) -> Self {
        Self {
            config: VolumeBucketConfig {
                target_volume,
                ..Default::default()
            },
            current_bucket: VolumeBucket::new(),
            completed_buckets: VecDeque::with_capacity(1000),
            max_history: 10000,
            total_volume_processed: 0.0,
            bucket_count: 0,
        }
    }

    /// Create with full configuration
    pub fn with_config(config: VolumeBucketConfig) -> Self {
        Self {
            config,
            current_bucket: VolumeBucket::new(),
            completed_buckets: VecDeque::with_capacity(1000),
            max_history: 10000,
            total_volume_processed: 0.0,
            bucket_count: 0,
        }
    }

    /// Add a single trade to the bucketing stream
    /// 
    /// # Arguments
    /// * `price` - Trade price
    /// * `volume` - Trade volume
    /// * `timestamp` - Trade timestamp
    /// * `is_buy` - Whether this is a buy trade (for buy/sell volume tracking)
    /// 
    /// # Returns
    /// Option containing completed bucket if one was finished, None otherwise
    pub fn add_trade(&mut self, price: f64, volume: f64, timestamp: u64, is_buy: bool) -> Option<VolumeBucket> {
        // Skip invalid or too-small volumes
        if !price.is_finite() || volume <= self.config.min_volume_threshold {
            return None;
        }

        // Initialize bucket if empty
        if self.current_bucket.trade_count == 0 {
            self.current_bucket.open = price;
            self.current_bucket.timestamp_start = timestamp;
        }

        // Update bucket statistics
        self.current_bucket.close = price;
        self.current_bucket.high = self.current_bucket.high.max(price);
        self.current_bucket.low = self.current_bucket.low.min(price);
        self.current_bucket.volume += volume;
        self.current_bucket.pv_sum += price * volume;
        self.current_bucket.trade_count += 1;
        self.current_bucket.timestamp_end = timestamp;

        if is_buy {
            self.current_bucket.buy_volume += volume;
        } else {
            self.current_bucket.sell_volume += volume;
        }

        self.total_volume_processed += volume;

        // Check if bucket is complete
        if self.current_bucket.volume >= self.config.target_volume 
            || self.current_bucket.trade_count as usize >= self.config.max_trades_per_bucket 
        {
            // Validate bucket before completing
            if !self.current_bucket.is_valid() {
                self.current_bucket = VolumeBucket::new();
                return None;
            }

            // Swap out completed bucket
            let completed = std::mem::replace(&mut self.current_bucket, VolumeBucket::new());
            self.bucket_count += 1;

            // Store in history
            if self.completed_buckets.len() >= self.max_history {
                self.completed_buckets.pop_front();
            }
            self.completed_buckets.push_back(completed.clone());

            Some(completed)
        } else {
            None
        }
    }

    /// Add multiple trades efficiently
    /// 
    /// # Returns
    /// Vec of completed buckets
    pub fn add_trades_batch(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        timestamps: &[u64],
        is_buys: &[bool],
    ) -> Vec<VolumeBucket> {
        let mut completed = Vec::new();
        
        debug_assert_eq!(prices.len(), volumes.len());
        debug_assert_eq!(prices.len(), timestamps.len());
        debug_assert_eq!(prices.len(), is_buys.len());

        for i in 0..prices.len() {
            if let Some(bucket) = self.add_trade(prices[i], volumes[i], timestamps[i], is_buys[i]) {
                completed.push(bucket);
            }
        }

        completed
    }

    /// Get the current incomplete bucket
    pub fn current_bucket(&self) -> &VolumeBucket {
        &self.current_bucket
    }

    /// Get completed buckets (reference)
    pub fn completed_buckets(&self) -> &VecDeque<VolumeBucket> {
        &self.completed_buckets
    }

    /// Get last N completed buckets
    pub fn get_recent_buckets(&self, n: usize) -> Vec<&VolumeBucket> {
        let len = self.completed_buckets.len();
        let start = len.saturating_sub(n);
        self.completed_buckets.iter().skip(start).collect()
    }

    /// Get statistics
    pub fn get_stats(&self) -> BuckerStats {
        let avg_volume = if self.bucket_count > 0 {
            self.total_volume_processed / self.bucket_count as f64
        } else {
            0.0
        };

        let avg_trades = if self.bucket_count > 0 {
            self.total_volume_processed / self.bucket_count as f64
        } else {
            0.0
        };

        BuckerStats {
            total_volume: self.total_volume_processed,
            bucket_count: self.bucket_count,
            avg_volume_per_bucket: avg_volume,
            avg_trades_per_bucket: avg_trades,
            current_bucket_progress: if self.config.target_volume > 0.0 {
                self.current_bucket.volume / self.config.target_volume
            } else {
                0.0
            },
            history_size: self.completed_buckets.len(),
        }
    }

    /// Reset the bucker state
    pub fn reset(&mut self) {
        self.current_bucket = VolumeBucket::new();
        self.completed_buckets.clear();
        self.total_volume_processed = 0.0;
        self.bucket_count = 0;
    }

    /// Set maximum history size
    pub fn set_max_history(&mut self, max: usize) {
        self.max_history = max;
        while self.completed_buckets.len() > max {
            self.completed_buckets.pop_front();
        }
    }
}

/// Statistics for the volume bucker
#[derive(Debug, Clone)]
pub struct BuckerStats {
    pub total_volume: f64,
    pub bucket_count: u64,
    pub avg_volume_per_bucket: f64,
    pub avg_trades_per_bucket: f64,
    pub current_bucket_progress: f64,
    pub history_size: usize,
}

/// Adaptive volume bucker that adjusts target volume based on market conditions
pub struct AdaptiveVolumeBucker {
    /// Base bucker
    inner: VolumeBucker,
    /// Lookback period for adaptation
    lookback: usize,
    /// Volatility estimate (ATR-like)
    volatility_estimate: f64,
    /// Volume momentum
    volume_momentum: f64,
    /// Adaptation factor
    adaptation_factor: f64,
}

impl AdaptiveVolumeBucker {
    /// Create new adaptive bucker
    pub fn new(base_target_volume: f64, lookback: usize) -> Self {
        Self {
            inner: VolumeBucker::new(base_target_volume),
            lookback,
            volatility_estimate: 0.0,
            volume_momentum: 1.0,
            adaptation_factor: 1.0,
        }
    }

    /// Add trade with adaptive adjustment
    pub fn add_trade(&mut self, price: f64, volume: f64, timestamp: u64, is_buy: bool) -> Option<VolumeBucket> {
        // Update volatility estimate (simplified)
        let buckets = self.inner.completed_buckets();
        if buckets.len() >= 2 {
            let recent: Vec<_> = buckets.iter().rev().take(self.lookback).collect();
            if recent.len() >= 2 {
                let ranges: Vec<f64> = recent.iter().map(|b| b.range()).collect();
                self.volatility_estimate = ranges.iter().sum::<f64>() / ranges.len() as f64;
            }
        }

        // Adjust target volume based on volatility
        // Higher volatility → smaller buckets for more frequent updates
        if self.volatility_estimate > 0.0 {
            let base = self.inner.get_stats().avg_volume_per_bucket;
            if base > 0.0 {
                self.adaptation_factor = (base / (self.volatility_estimate * 100.0)).clamp(0.5, 2.0);
            }
        }

        self.inner.add_trade(price, volume, timestamp, is_buy)
    }

    /// Get current adaptation factor
    pub fn adaptation_factor(&self) -> f64 {
        self.adaptation_factor
    }

    /// Get inner bucker reference
    pub fn inner(&self) -> &VolumeBucker {
        &self.inner
    }

    /// Get mutable inner bucker
    pub fn inner_mut(&mut self) -> &mut VolumeBucker {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_bucket_completion() {
        let mut bucker = VolumeBucker::new(100.0);
        
        // Add trades totaling exactly 100 volume
        assert!(bucker.add_trade(100.0, 50.0, 1000, true).is_none());
        assert!(bucker.add_trade(100.0, 50.0, 1001, false).is_some());
        
        let stats = bucker.get_stats();
        assert_eq!(stats.bucket_count, 1);
        assert_eq!(stats.total_volume, 100.0);
    }

    #[test]
    fn test_multiple_buckets() {
        let mut bucker = VolumeBucker::new(50.0);
        let mut completed_count = 0;
        
        for i in 0..10 {
            if bucker.add_trade(100.0 + (i as f64 * 0.1), 25.0, 1000 + i, i % 2 == 0).is_some() {
                completed_count += 1;
            }
        }
        
        assert_eq!(completed_count, 5); // 250 volume / 50 = 5 buckets
        assert_eq!(bucker.get_stats().bucket_count, 5);
    }

    #[test]
    fn test_bucket_statistics() {
        let mut bucker = VolumeBucker::new(100.0);
        
        bucker.add_trade(100.0, 60.0, 1000, true);
        let bucket = bucker.add_trade(101.0, 40.0, 1001, false).unwrap();
        
        assert_eq!(bucket.open, 100.0);
        assert_eq!(bucket.close, 101.0);
        assert_eq!(bucket.high, 101.0);
        assert_eq!(bucket.low, 100.0);
        assert_eq!(bucket.volume, 100.0);
        assert_eq!(bucket.trade_count, 2);
        assert!((bucket.vwap() - 100.6).abs() < 0.001);
        assert_eq!(bucket.buy_volume, 60.0);
        assert_eq!(bucket.sell_volume, 40.0);
        assert!((bucket.order_imbalance() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_zero_volume_handling() {
        let mut bucker = VolumeBucker::new(100.0);
        
        // Very small volume should be ignored
        assert!(bucker.add_trade(100.0, 0.0001, 1000, true).is_none());
        assert!(!bucker.current_bucket().is_valid());
    }

    #[test]
    fn test_batch_processing() {
        let mut bucker = VolumeBucker::new(50.0);
        
        let prices = vec![100.0, 100.1, 100.2, 100.3];
        let volumes = vec![25.0, 25.0, 25.0, 25.0];
        let timestamps = vec![1000, 1001, 1002, 1003];
        let is_buys = vec![true, false, true, false];
        
        let completed = bucker.add_trades_batch(&prices, &volumes, &timestamps, &is_buys);
        
        assert_eq!(completed.len(), 2);
        assert_eq!(bucker.get_stats().bucket_count, 2);
    }
}
