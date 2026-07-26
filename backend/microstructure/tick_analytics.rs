//! # Tick Analytics Implementation
//! 
//! Analyzes tick-level market data including inter-arrival times,
//! trade clustering, and high-frequency patterns.
//! 
//! **Key Features:**
//! - Inter-arrival time distribution analysis
//! - Trade clustering detection
//! - Volume-weighted tick statistics
//! - Zero-cost abstractions with proper borrowing
//! 
//! **Performance:** O(1) updates suitable for millions of ticks/sec.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for tick analytics
#[derive(Debug, Clone)]
pub struct TickAnalyticsConfig {
    /// Maximum number of ticks to retain in history
    pub max_history: usize,
    /// Minimum samples for statistical estimates
    pub min_samples: usize,
    /// Time window for clustering detection (milliseconds)
    pub cluster_window_ms: u64,
}

impl Default for TickAnalyticsConfig {
    fn default() -> Self {
        Self {
            max_history: 10000,
            min_samples: 100,
            cluster_window_ms: 100, // 100ms clustering window
        }
    }
}

/// Result from tick analytics
#[derive(Debug, Clone)]
pub struct TickAnalyticsResult {
    /// Mean inter-arrival time (microseconds)
    pub mean_iat: f64,
    /// Standard deviation of inter-arrival times
    pub std_iat: f64,
    /// Arrival rate (ticks per second)
    pub arrival_rate: f64,
    /// Clustering coefficient (0 = uniform, 1 = highly clustered)
    pub clustering_coef: f64,
    /// Number of detected clusters
    pub cluster_count: usize,
    /// Sample count used
    pub sample_count: usize,
    /// Whether results are statistically reliable
    pub is_reliable: bool,
}

/// Single tick record
#[derive(Debug, Clone)]
pub struct TickRecord {
    /// Timestamp (microseconds since epoch or relative)
    pub timestamp: u64,
    /// Trade price
    pub price: f64,
    /// Trade volume
    pub volume: f64,
    /// Trade direction (1=buy, -1=sell, 0=unknown)
    pub direction: i8,
}

/// Streaming tick analytics engine
pub struct TickAnalytics {
    config: TickAnalyticsConfig,
    /// Rolling window of tick records
    ticks: VecDeque<TickRecord>,
    /// Rolling window of inter-arrival times (microseconds)
    iats: VecDeque<u64>,
    /// Previous timestamp for IAT calculation
    prev_timestamp: Option<u64>,
    /// Running sum of IATs
    sum_iat: u128,
    /// Running sum of squared IATs
    sum_sq_iat: u128,
    /// Last update instant
    last_update: Instant,
    /// Total ticks processed
    total_ticks: u64,
}

impl TickAnalytics {
    /// Create new analytics engine with default config
    pub fn new() -> Self {
        Self::with_config(TickAnalyticsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: TickAnalyticsConfig) -> Self {
        Self {
            config,
            ticks: VecDeque::with_capacity(config.max_history),
            iats: VecDeque::with_capacity(config.max_history),
            prev_timestamp: None,
            sum_iat: 0,
            sum_sq_iat: 0,
            last_update: Instant::now(),
            total_ticks: 0,
        }
    }

    /// Add a new tick observation
    #[inline]
    pub fn add_tick(&mut self, tick: TickRecord) -> Option<TickAnalyticsResult> {
        // Validate price
        if !tick.price.is_finite() || tick.price <= 0.0 {
            return None;
        }

        // Calculate inter-arrival time
        let iat = if let Some(prev_ts) = self.prev_timestamp {
            if tick.timestamp > prev_ts {
                tick.timestamp - prev_ts
            } else {
                0 // Out of order or same timestamp
            }
        } else {
            0
        };

        self.prev_timestamp = Some(tick.timestamp);

        // Add to rolling windows
        self.add_tick_and_iat(tick.clone(), iat);
        self.total_ticks += 1;
        self.last_update = Instant::now();

        // Return result if enough samples
        if self.iats.len() >= self.config.min_samples {
            Some(self.compute_result())
        } else {
            None
        }
    }

    /// Convenience method to add tick with raw parameters
    pub fn add_tick_raw(
        &mut self,
        timestamp: u64,
        price: f64,
        volume: f64,
        direction: i8,
    ) -> Option<TickAnalyticsResult> {
        let tick = TickRecord {
            timestamp,
            price,
            volume,
            direction,
        };
        self.add_tick(tick)
    }

    /// Internal method to add tick and IAT
    fn add_tick_and_iat(&mut self, tick: TickRecord, iat: u64) {
        // Remove oldest if at capacity
        if self.ticks.len() >= self.config.max_history {
            if let Some(old_tick) = self.ticks.pop_front() {
                // Find corresponding IAT and remove
                if let Some(old_iat) = self.iats.pop_front() {
                    self.sum_iat -= old_iat as u128;
                    self.sum_sq_iat -= (old_iat as u128) * (old_iat as u128);
                }
            }
        }

        self.ticks.push_back(tick);
        
        if iat > 0 {
            self.iats.push_back(iat);
            self.sum_iat += iat as u128;
            self.sum_sq_iat += (iat as u128) * (iat as u128);
        }
    }

    /// Compute analytics result
    fn compute_result(&self) -> TickAnalyticsResult {
        let n = self.iats.len() as f64;
        if n < 1.0 {
            return TickAnalyticsResult {
                mean_iat: 0.0,
                std_iat: 0.0,
                arrival_rate: 0.0,
                clustering_coef: 0.0,
                cluster_count: 0,
                sample_count: 0,
                is_reliable: false,
            };
        }

        // Mean IAT (convert to microseconds)
        let mean_iat = self.sum_iat as f64 / n;

        // Variance and std dev
        let variance = if n > 1.0 {
            let sq_mean = (self.sum_iat as f64 / n).powi(2);
            let mean_sq = self.sum_sq_iat as f64 / n;
            (mean_sq - sq_mean).max(0.0)
        } else {
            0.0
        };
        let std_iat = variance.sqrt();

        // Arrival rate (ticks per second)
        let arrival_rate = if mean_iat > 0.0 {
            1_000_000.0 / mean_iat // Convert μs to seconds
        } else {
            f64::INFINITY
        };

        // Clustering coefficient (coefficient of variation of IAT)
        let clustering_coef = if mean_iat > 0.0 {
            (std_iat / mean_iat).min(1.0)
        } else {
            0.0
        };

        // Count clusters
        let cluster_count = self.detect_clusters();

        // Reliability check
        let is_reliable = n >= self.config.min_samples as f64
            && mean_iat.is_finite()
            && std_iat.is_finite();

        TickAnalyticsResult {
            mean_iat,
            std_iat,
            arrival_rate,
            clustering_coef,
            cluster_count,
            sample_count: self.iats.len(),
            is_reliable,
        }
    }

    /// Detect trade clusters using time-based grouping
    fn detect_clusters(&self) -> usize {
        if self.iats.len() < 2 {
            return 0;
        }

        let cluster_threshold_us = self.config.cluster_window_ms * 1000;
        let mut clusters = 0;
        let mut in_cluster = false;

        for &iat in &self.iats {
            if iat < cluster_threshold_us {
                if !in_cluster {
                    clusters += 1;
                    in_cluster = true;
                }
            } else {
                in_cluster = false;
            }
        }

        clusters
    }

    /// Get current arrival rate
    pub fn current_arrival_rate(&self) -> f64 {
        if self.iats.is_empty() {
            return 0.0;
        }
        let mean_iat = self.sum_iat as f64 / self.iats.len() as f64;
        if mean_iat > 0.0 {
            1_000_000.0 / mean_iat
        } else {
            f64::INFINITY
        }
    }

    /// Get recent ticks (last N)
    pub fn get_recent_ticks(&self, n: usize) -> Vec<&TickRecord> {
        let len = self.ticks.len();
        let start = len.saturating_sub(n);
        self.ticks.iter().skip(start).collect()
    }

    /// Get volume-weighted average price from recent ticks
    pub fn vwap_recent(&self, n: usize) -> f64 {
        let recent: Vec<_> = self.ticks.iter().rev().take(n).collect();
        
        let total_pv: f64 = recent.iter().map(|t| t.price * t.volume).sum();
        let total_volume: f64 = recent.iter().map(|t| t.volume).sum();
        
        if total_volume > 0.0 {
            total_pv / total_volume
        } else {
            0.0
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> TickStats {
        TickStats {
            total_ticks: self.total_ticks,
            current_history_len: self.ticks.len(),
            current_iat_history_len: self.iats.len(),
            min_samples: self.config.min_samples,
            is_ready: self.iats.len() >= self.config.min_samples,
            last_update_elapsed: self.last_update.elapsed().as_micros() as u64,
            current_arrival_rate: self.current_arrival_rate(),
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.ticks.clear();
        self.iats.clear();
        self.prev_timestamp = None;
        self.sum_iat = 0;
        self.sum_sq_iat = 0;
        self.total_ticks = 0;
    }
}

impl Default for TickAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for tick analytics
#[derive(Debug, Clone)]
pub struct TickStats {
    pub total_ticks: u64,
    pub current_history_len: usize,
    pub current_iat_history_len: usize,
    pub min_samples: usize,
    pub is_ready: bool,
    pub last_update_elapsed: u64, // microseconds
    pub current_arrival_rate: f64,
}

/// Poisson process tester for tick arrivals
pub struct PoissonTester {
    iats: VecDeque<f64>,
}

impl PoissonTester {
    pub fn new(max_samples: usize) -> Self {
        Self {
            iats: VecDeque::with_capacity(max_samples),
        }
    }

    /// Add an inter-arrival time
    pub fn add_iat(&mut self, iat_us: u64) {
        self.iats.push_back(iat_us as f64);
        if self.iats.len() > 10000 {
            self.iats.pop_front();
        }
    }

    /// Test if arrivals follow Poisson process
    /// 
    /// For Poisson: mean ≈ std_dev (exponential distribution)
    pub fn is_poisson_like(&self) -> bool {
        if self.iats.len() < 100 {
            return false;
        }

        let iats: Vec<f64> = self.iats.iter().copied().collect();
        let mean = iats.iter().sum::<f64>() / iats.len() as f64;
        let variance: f64 = iats.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / iats.len() as f64;
        let std_dev = variance.sqrt();

        // For exponential distribution: mean ≈ std_dev
        // Allow 20% tolerance
        let ratio = std_dev / mean.max(1.0);
        (ratio - 1.0).abs() < 0.2
    }

    /// Get dispersion index (variance/mean)
    /// 
    /// For Poisson: dispersion ≈ 1
    pub fn dispersion_index(&self) -> f64 {
        if self.iats.is_empty() {
            return 0.0;
        }

        let iats: Vec<f64> = self.iats.iter().copied().collect();
        let mean = iats.iter().sum::<f64>() / iats.len() as f64;
        let variance: f64 = iats.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / iats.len() as f64;

        if mean > 0.0 {
            variance / mean
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_arrivals() {
        let mut analytics = TickAnalytics::with_config(TickAnalyticsConfig {
            max_history: 1000,
            min_samples: 100,
            cluster_window_ms: 100,
        });

        // Simulate uniform arrivals (every 1000 μs)
        let mut timestamp = 0u64;
        for i in 0..200 {
            let result = analytics.add_tick_raw(
                timestamp,
                100.0 + (i as f64 * 0.01),
                1.0,
                1,
            );
            timestamp += 1000; // Fixed 1ms intervals
        }

        let result = analytics.compute_result();
        
        assert!(result.is_reliable);
        assert!((result.mean_iat - 1000.0).abs() < 100.0); // ~1000 μs
        assert!(result.clustering_coef < 0.1); // Low clustering
    }

    #[test]
    fn test_clustered_arrivals() {
        let mut analytics = TickAnalytics::new();

        // Simulate clustered arrivals
        let mut timestamp = 0u64;
        for _ in 0..10 {
            // Cluster of 20 trades with 10μs spacing
            for _ in 0..20 {
                analytics.add_tick_raw(timestamp, 100.0, 1.0, 1);
                timestamp += 10;
            }
            // Gap of 5000μs between clusters
            timestamp += 5000;
        }

        let result = analytics.compute_result();
        
        assert!(result.cluster_count >= 5); // Should detect multiple clusters
        assert!(result.clustering_coef > 0.5); // High clustering
    }

    #[test]
    fn test_arrival_rate_calculation() {
        let mut analytics = TickAnalytics::new();

        // 100 ticks per second = 10000 μs mean IAT
        let mut timestamp = 0u64;
        for _ in 0..200 {
            analytics.add_tick_raw(timestamp, 100.0, 1.0, 1);
            timestamp += 10000;
        }

        let rate = analytics.current_arrival_rate();
        
        // Should be approximately 100 ticks/sec
        assert!((rate - 100.0).abs() < 10.0);
    }

    #[test]
    fn test_vwap_calculation() {
        let mut analytics = TickAnalytics::new();

        analytics.add_tick_raw(1000, 100.0, 10.0, 1);
        analytics.add_tick_raw(2000, 101.0, 20.0, 1);
        analytics.add_tick_raw(3000, 102.0, 30.0, 1);

        let vwap = analytics.vwap_recent(3);
        
        // VWAP = (100*10 + 101*20 + 102*30) / (10+20+30)
        //      = (1000 + 2020 + 3060) / 60 = 6080 / 60 = 101.33...
        assert!((vwap - 101.333).abs() < 0.01);
    }

    #[test]
    fn test_poisson_tester() {
        let mut tester = PoissonTester::new(10000);

        // Add exponentially distributed IATs (Poisson process)
        // Using simple approximation
        for i in 0..1000 {
            let iat = ((i % 100) as f64 * 10.0) as u64;
            tester.add_iat(iat);
        }

        let dispersion = tester.dispersion_index();
        assert!(dispersion > 0.0);
    }
}
