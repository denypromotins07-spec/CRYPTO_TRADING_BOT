//! VPIN (Volume-Synchronized Probability of Informed Trading) Calculator
//! Measures order flow toxicity to detect informed trading and widen spreads.
//! Zero-cost abstractions for real-time toxicity detection.
//!
//! VPIN is a key metric for market makers to identify when they are
//! trading against informed participants ("toxic flow").
//!
//! High VPIN indicates:
//! - Informed traders are active
//! - Spreads should be widened
//! - Inventory risk is elevated
//!
//! Formula: VPIN = (1/n) * Σ|V_buy - V_sell| / (V_buy + V_sell)
//! Where volumes are bucketed by equal volume intervals.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Configuration for VPIN calculation
#[derive(Debug, Clone, Copy)]
pub struct VPINConfig {
    /// Size of each volume bucket (in base currency units, scaled by 1e9)
    pub bucket_size: u64,
    /// Number of buckets for VPIN calculation
    pub num_buckets: usize,
    /// Maximum trade history size
    pub max_history: usize,
    /// VPIN threshold for toxic flow detection
    pub toxic_threshold: f64,
    /// Minimum trades per bucket for valid calculation
    pub min_trades_per_bucket: usize,
}

impl Default for VPINConfig {
    fn default() -> Self {
        Self {
            bucket_size: (10.0 * 1e9) as u64, // 10 BTC equivalent per bucket
            num_buckets: 50,
            max_history: 10000,
            toxic_threshold: 0.7, // VPIN > 0.7 is considered toxic
            min_trades_per_bucket: 5,
        }
    }
}

/// Represents a single trade
#[derive(Debug, Clone, Copy)]
pub struct Trade {
    pub timestamp_ms: u64,
    pub price: f64,
    pub volume: u64, // Scaled by 1e9 for precision
    pub is_buy: bool, // True = buyer-initiated, False = seller-initiated
}

/// Volume bucket for VPIN calculation
#[derive(Debug, Clone)]
pub struct VolumeBucket {
    pub buy_volume: u64,
    pub sell_volume: u64,
    pub trade_count: usize,
    pub start_time_ms: u64,
    pub end_time_ms: u64,
}

impl VolumeBucket {
    pub fn new(start_time_ms: u64) -> Self {
        Self {
            buy_volume: 0,
            sell_volume: 0,
            trade_count: 0,
            start_time_ms,
            end_time_ms: start_time_ms,
        }
    }

    #[inline]
    pub fn add_trade(&mut self, trade: &Trade) {
        if trade.is_buy {
            self.buy_volume += trade.volume;
        } else {
            self.sell_volume += trade.volume;
        }
        self.trade_count += 1;
        self.end_time_ms = trade.timestamp_ms;
    }

    #[inline]
    pub fn total_volume(&self) -> u64 {
        self.buy_volume + self.sell_volume
    }

    #[inline]
    pub fn imbalance(&self) -> f64 {
        if self.total_volume() == 0 {
            return 0.0;
        }
        let diff = if self.buy_volume > self.sell_volume {
            self.buy_volume - self.sell_volume
        } else {
            self.sell_volume - self.buy_volume
        };
        diff as f64 / self.total_volume() as f64
    }

    #[inline]
    pub fn is_full(&self, bucket_size: u64) -> bool {
        self.total_volume() >= bucket_size
    }
}

/// VPIN calculator with rolling bucket management
pub struct VPINCalculator {
    config: VPINConfig,
    /// Current active bucket
    current_bucket: parking_lot::Mutex<Option<VolumeBucket>>,
    /// Rolling window of completed buckets
    buckets: parking_lot::Mutex<VecDeque<VolumeBucket>>,
    /// Running VPIN value (scaled by 1e9)
    current_vpin: AtomicU64,
    /// Toxic flow flag
    is_toxic: AtomicBool,
    /// Last update timestamp
    last_update_ms: AtomicU64,
    /// Trade count for statistics
    trade_count: AtomicU64,
    /// Validity flag
    is_valid: AtomicBool,
}

impl VPINCalculator {
    /// Create a new VPIN calculator with default configuration
    pub fn new() -> Self {
        Self::with_config(VPINConfig::default())
    }

    /// Create a new VPIN calculator with custom configuration
    pub fn with_config(config: VPINConfig) -> Self {
        Self {
            config,
            current_bucket: parking_lot::Mutex::new(None),
            buckets: parking_lot::Mutex::new(VecDeque::with_capacity(config.num_buckets)),
            current_vpin: AtomicU64::new(0),
            is_toxic: AtomicBool::new(false),
            last_update_ms: AtomicU64::new(0),
            trade_count: AtomicU64::new(0),
            is_valid: AtomicBool::new(false),
        }
    }

    /// Process a single trade and update VPIN
    #[inline]
    pub fn process_trade(&self, trade: Trade) {
        let start = Instant::now();
        
        self.trade_count.fetch_add(1, Ordering::Relaxed);
        
        // Get or create current bucket
        let mut current_guard = self.current_bucket.lock();
        
        if current_guard.is_none() {
            *current_guard = Some(VolumeBucket::new(trade.timestamp_ms));
        }
        
        let bucket = current_guard.as_mut().unwrap();
        bucket.add_trade(&trade);
        
        // Check if bucket is full
        if bucket.is_full(self.config.bucket_size) && bucket.trade_count >= self.config.min_trades_per_bucket {
            // Move to completed buckets
            let completed_bucket = current_guard.take().unwrap();
            
            let mut buckets_guard = self.buckets.lock();
            buckets_guard.push_back(completed_bucket);
            
            // Maintain rolling window
            while buckets_guard.len() > self.config.num_buckets {
                buckets_guard.pop_front();
            }
            
            // Recalculate VPIN
            let vpin = self.calculate_vpin_internal(&buckets_guard);
            self.current_vpin.store((vpin * 1e9) as u64, Ordering::Relaxed);
            self.is_toxic.store(vpin > self.config.toxic_threshold, Ordering::Relaxed);
        }
        
        self.last_update_ms.store(trade.timestamp_ms, Ordering::Relaxed);
        self.is_valid.store(true, Ordering::Relaxed);
    }

    /// Internal VPIN calculation from buckets
    #[inline]
    fn calculate_vpin_internal(&self, buckets: &VecDeque<VolumeBucket>) -> f64 {
        if buckets.len() < self.config.num_buckets / 2 {
            return 0.0; // Not enough data
        }

        let mut sum_imbalance = 0.0;
        let mut total_volume = 0u64;

        for bucket in buckets.iter() {
            if bucket.total_volume() > 0 {
                let bucket_imbalance = bucket.imbalance();
                sum_imbalance += bucket_imbalance * bucket.total_volume() as f64;
                total_volume += bucket.total_volume();
            }
        }

        if total_volume == 0 {
            return 0.0;
        }

        sum_imbalance / total_volume as f64
    }

    /// Get current VPIN value (0.0 to 1.0)
    #[inline]
    pub fn get_vpin(&self) -> f64 {
        self.current_vpin.load(Ordering::Relaxed) as f64 / 1e9
    }

    /// Check if current flow is toxic
    #[inline]
    pub fn is_toxic_flow(&self) -> bool {
        self.is_toxic.load(Ordering::Relaxed)
    }

    /// Get recommended spread multiplier based on VPIN
    /// Higher VPIN -> Wider spreads to protect against informed traders
    #[inline]
    pub fn get_spread_multiplier(&self) -> f64 {
        let vpin = self.get_vpin();
        
        // Linear scaling: 1.0x at VPIN=0, up to 5.0x at VPIN=1.0
        1.0 + 4.0 * vpin
    }

    /// Get toxicity level category
    #[inline]
    pub fn get_toxicity_level(&self) -> ToxicityLevel {
        let vpin = self.get_vpin();
        
        if vpin >= self.config.toxic_threshold {
            ToxicityLevel::Critical
        } else if vpin >= self.config.toxic_threshold * 0.7 {
            ToxicityLevel::High
        } else if vpin >= self.config.toxic_threshold * 0.4 {
            ToxicityLevel::Medium
        } else {
            ToxicityLevel::Low
        }
    }

    /// Get statistics about the calculation
    #[inline]
    pub fn get_stats(&self) -> VPINStats {
        let buckets_guard = self.buckets.lock();
        
        let avg_bucket_volume = if !buckets_guard.is_empty() {
            let total: u64 = buckets_guard.iter().map(|b| b.total_volume()).sum();
            total as f64 / buckets_guard.len() as f64
        } else {
            0.0
        };

        VPINStats {
            current_vpin: self.get_vpin(),
            is_toxic: self.is_toxic_flow(),
            bucket_count: buckets_guard.len(),
            total_trades: self.trade_count.load(Ordering::Relaxed),
            avg_bucket_volume: avg_bucket_volume / 1e9, // Convert back to base units
            is_valid: self.is_valid.load(Ordering::Relaxed),
        }
    }

    /// Reset all state (e.g., for new trading session)
    #[inline]
    pub fn reset(&self) {
        *self.current_bucket.lock() = None;
        self.buckets.lock().clear();
        self.current_vpin.store(0, Ordering::Relaxed);
        self.is_toxic.store(false, Ordering::Relaxed);
        self.is_valid.store(false, Ordering::Relaxed);
    }

    /// Invalidate calculator (e.g., during data feed issues)
    #[inline]
    pub fn invalidate(&self) {
        self.is_valid.store(false, Ordering::Relaxed);
    }
}

impl Default for VPINCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Toxicity level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityLevel {
    Low,      // VPIN < 0.28
    Medium,   // VPIN 0.28-0.49
    High,     // VPIN 0.49-0.70
    Critical, // VPIN > 0.70
}

/// VPIN statistics
#[derive(Debug, Clone, Copy)]
pub struct VPINStats {
    pub current_vpin: f64,
    pub is_toxic: bool,
    pub bucket_count: usize,
    pub total_trades: u64,
    pub avg_bucket_volume: f64,
    pub is_valid: bool,
}

/// Builder pattern for VPINConfig
pub struct VPINBuilder {
    config: VPINConfig,
}

impl VPINBuilder {
    pub fn new() -> Self {
        Self {
            config: VPINConfig::default(),
        }
    }

    pub fn bucket_size(mut self, size: f64) -> Self {
        self.config.bucket_size = (size * 1e9) as u64;
        self
    }

    pub fn num_buckets(mut self, count: usize) -> Self {
        self.config.num_buckets = count;
        self
    }

    pub fn toxic_threshold(mut self, threshold: f64) -> Self {
        self.config.toxic_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> VPINCalculator {
        VPINCalculator::with_config(self.config)
    }
}

impl Default for VPINBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_balanced_flow() {
        let calc = VPINCalculator::new();
        
        // Add balanced buy/sell trades
        for i in 0..100 {
            let trade = Trade {
                timestamp_ms: 1000 + i * 10,
                price: 50000.0,
                volume: (1.0 * 1e9) as u64,
                is_buy: i % 2 == 0,
            };
            calc.process_trade(trade);
        }

        let vpin = calc.get_vpin();
        // With balanced flow, VPIN should be low
        assert!(vpin < 0.3);
        assert!(!calc.is_toxic_flow());
    }

    #[test]
    fn test_vpin_unbalanced_flow() {
        let calc = VPINCalculator::new();
        
        // Add mostly buy trades (informed buying)
        for i in 0..100 {
            let trade = Trade {
                timestamp_ms: 1000 + i * 10,
                price: 50000.0,
                volume: (1.0 * 1e9) as u64,
                is_buy: i < 90, // 90% buys
            };
            calc.process_trade(trade);
        }

        let vpin = calc.get_vpin();
        // With unbalanced flow, VPIN should be high
        assert!(vpin > 0.5);
    }

    #[test]
    fn test_spread_multiplier() {
        let calc = VPINCalculator::new();
        
        // Low VPIN -> normal spread
        assert!((calc.get_spread_multiplier() - 1.0).abs() < 0.1);
        
        // Simulate high VPIN
        calc.current_vpin.store((0.8 * 1e9) as u64, Ordering::Relaxed);
        calc.is_toxic.store(true, Ordering::Relaxed);
        
        // High VPIN -> wider spread
        assert!(calc.get_spread_multiplier() > 4.0);
    }

    #[test]
    fn test_toxicity_levels() {
        let calc = VPINCalculator::new();
        
        assert_eq!(calc.get_toxicity_level(), ToxicityLevel::Low);
        
        calc.current_vpin.store((0.5 * 1e9) as u64, Ordering::Relaxed);
        assert_eq!(calc.get_toxicity_level(), ToxicityLevel::High);
        
        calc.current_vpin.store((0.8 * 1e9) as u64, Ordering::Relaxed);
        assert_eq!(calc.get_toxicity_level(), ToxicityLevel::Critical);
    }

    #[test]
    fn test_bucket_rollover() {
        let mut config = VPINConfig::default();
        config.bucket_size = (5.0 * 1e9) as u64; // Smaller buckets for testing
        config.num_buckets = 10;
        
        let calc = VPINCalculator::with_config(config);
        
        // Fill multiple buckets
        for i in 0..200 {
            let trade = Trade {
                timestamp_ms: 1000 + i * 10,
                price: 50000.0,
                volume: (0.5 * 1e9) as u64, // Each trade is 0.5, need 10 per bucket
                is_buy: i % 2 == 0,
            };
            calc.process_trade(trade);
        }

        let stats = calc.get_stats();
        assert!(stats.bucket_count > 0);
        assert!(stats.bucket_count <= config.num_buckets);
    }
}
