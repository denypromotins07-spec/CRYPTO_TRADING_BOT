//! Information Asymmetry Calculator
//! Calculates probability of trading against a whale (informed trader).
//! Zero-cost abstractions for real-time asymmetry detection.
//!
//! Information asymmetry occurs when one party has superior knowledge
//! about an asset's true value. This module detects such situations by:
//! - Analyzing trade size distributions
//! - Measuring price impact asymmetry
//! - Tracking order flow toxicity
//! - Identifying whale signatures
//!
//! High asymmetry = widen spreads, reduce quote sizes

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Configuration for information asymmetry calculation
#[derive(Debug, Clone, Copy)]
pub struct AsymmetryConfig {
    /// Volume threshold for whale detection (scaled by 1e9)
    pub whale_volume_threshold: u64,
    /// Price impact threshold for informed detection (basis points * 100)
    pub impact_threshold_bps: u64,
    /// Lookback window for analysis (milliseconds)
    pub lookback_window_ms: u64,
    /// Minimum samples for valid calculation
    pub min_samples: usize,
    /// Asymmetry threshold for high-risk flag
    pub high_risk_threshold: f64,
}

impl Default for AsymmetryConfig {
    fn default() -> Self {
        Self {
            whale_volume_threshold: (5.0 * 1e9) as u64, // 5 BTC
            impact_threshold_bps: 500, // 5 basis points
            lookback_window_ms: 60000, // 1 minute
            min_samples: 20,
            high_risk_threshold: 0.7,
        }
    }
}

/// Trade record for asymmetry analysis
#[derive(Debug, Clone, Copy)]
pub struct AsymmetryTrade {
    pub timestamp_ms: u64,
    pub price: f64,
    pub volume: u64, // Scaled by 1e9
    pub is_buy: bool,
    pub price_change_bps: i64, // Price change after trade (basis points * 100)
}

/// Rolling statistics tracker
struct RollingStats {
    values: parking_lot::Mutex<Vec<f64>>,
    max_size: usize,
    sum: AtomicU64,
    sum_sq: AtomicU64,
    count: AtomicU64,
}

impl RollingStats {
    fn new(max_size: usize) -> Self {
        Self {
            values: parking_lot::Mutex::new(Vec::with_capacity(max_size)),
            max_size,
            sum: AtomicU64::new(0),
            sum_sq: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    fn add(&self, value: f64) {
        let mut values = self.values.lock();
        
        if values.len() >= self.max_size {
            let old = values.remove(0);
            let old_scaled = (old * 1e6) as u64;
            let old_sq_scaled = ((old * old) * 1e12) as u64;
            
            self.sum.fetch_sub(old_scaled, Ordering::Relaxed);
            self.sum_sq.fetch_sub(old_sq_scaled, Ordering::Relaxed);
            self.count.fetch_sub(1, Ordering::Relaxed);
        }
        
        values.push(value);
        let scaled = (value * 1e6) as u64;
        let sq_scaled = ((value * value) * 1e12) as u64;
        
        self.sum.fetch_add(scaled, Ordering::Relaxed);
        self.sum_sq.fetch_add(sq_scaled, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn mean(&self) -> Option<f64> {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }
        Some(self.sum.load(Ordering::Relaxed) as f64 / 1e6 / count as f64)
    }

    #[inline]
    fn std_dev(&self) -> Option<f64> {
        let count = self.count.load(Ordering::Relaxed);
        if count < 2 {
            return None;
        }
        
        let sum = self.sum.load(Ordering::Relaxed) as f64 / 1e6;
        let sum_sq = self.sum_sq.load(Ordering::Relaxed) as f64 / 1e12;
        
        let variance = (sum_sq - (sum * sum / count as f64)) / (count as f64 - 1.0);
        Some(variance.sqrt())
    }
}

/// Information asymmetry calculator
pub struct InformationAsymmetryCalculator {
    config: AsymmetryConfig,
    /// Recent trades buffer
    trades: parking_lot::Mutex<Vec<AsymmetryTrade>>,
    /// Whale trade counter
    whale_count: AtomicU64,
    /// Informed trade counter (trades followed by favorable price move)
    informed_count: AtomicU64,
    /// Total trade counter
    total_count: AtomicU64,
    /// Current asymmetry score (scaled by 1e9)
    asymmetry_score: AtomicU64,
    /// High risk flag
    is_high_risk: AtomicBool,
    /// Last update timestamp
    last_update_ms: AtomicU64,
    /// Validity flag
    is_valid: AtomicBool,
    /// Rolling price impact stats
    impact_stats: RollingStats,
}

impl InformationAsymmetryCalculator {
    /// Create a new asymmetry calculator with default configuration
    pub fn new() -> Self {
        Self::with_config(AsymmetryConfig::default())
    }

    /// Create a new asymmetry calculator with custom configuration
    pub fn with_config(config: AsymmetryConfig) -> Self {
        Self {
            config,
            trades: parking_lot::Mutex::new(Vec::with_capacity(1000)),
            whale_count: AtomicU64::new(0),
            informed_count: AtomicU64::new(0),
            total_count: AtomicU64::new(0),
            asymmetry_score: AtomicU64::new(0),
            is_high_risk: AtomicBool::new(false),
            last_update_ms: AtomicU64::new(0),
            is_valid: AtomicBool::new(false),
            impact_stats: RollingStats::new(1000),
        }
    }

    /// Process a trade and update asymmetry metrics
    #[inline]
    pub fn process_trade(&self, trade: AsymmetryTrade) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
        
        // Add to rolling impact stats
        self.impact_stats.add(trade.price_change_bps as f64 / 100.0);
        
        // Check if whale trade
        let is_whale = trade.volume >= self.config.whale_volume_threshold;
        if is_whale {
            self.whale_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Check if informed (price moved in direction of trade)
        let is_informed = if trade.is_buy {
            trade.price_change_bps > self.config.impact_threshold_bps as i64
        } else {
            trade.price_change_bps < -(self.config.impact_threshold_bps as i64)
        };
        
        if is_informed {
            self.informed_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update trades buffer
        let mut trades_guard = self.trades.lock();
        trades_guard.push(trade);
        
        // Trim old trades
        let cutoff = trade.timestamp_ms.saturating_sub(self.config.lookback_window_ms);
        while let Some(first) = trades_guard.first() {
            if first.timestamp_ms < cutoff {
                trades_guard.remove(0);
            } else {
                break;
            }
        }
        
        // Recalculate asymmetry score
        let score = self.calculate_asymmetry_internal(&trades_guard);
        self.asymmetry_score.store((score * 1e9) as u64, Ordering::Relaxed);
        self.is_high_risk.store(score > self.config.high_risk_threshold, Ordering::Relaxed);
        
        self.last_update_ms.store(trade.timestamp_ms, Ordering::Relaxed);
        self.is_valid.store(true, Ordering::Relaxed);
    }

    /// Internal asymmetry calculation
    #[inline]
    fn calculate_asymmetry_internal(&self, trades: &[AsymmetryTrade]) -> f64 {
        if trades.len() < self.config.min_samples {
            return 0.0;
        }

        let whale_trades = trades.iter()
            .filter(|t| t.volume >= self.config.whale_volume_threshold)
            .count();
        
        let informed_trades = trades.iter()
            .filter(|t| {
                if t.is_buy {
                    t.price_change_bps > self.config.impact_threshold_bps as i64
                } else {
                    t.price_change_bps < -(self.config.impact_threshold_bps as i64)
                }
            })
            .count();

        // Calculate components
        let whale_ratio = whale_trades as f64 / trades.len() as f64;
        let informed_ratio = informed_trades as f64 / trades.len() as f64;
        
        // Weighted combination
        let asymmetry = whale_ratio * 0.4 + informed_ratio * 0.6;
        
        asymmetry.min(1.0)
    }

    /// Get current asymmetry score (0.0 to 1.0)
    #[inline]
    pub fn get_asymmetry_score(&self) -> f64 {
        self.asymmetry_score.load(Ordering::Relaxed) as f64 / 1e9
    }

    /// Check if currently in high-risk regime
    #[inline]
    pub fn is_high_risk(&self) -> bool {
        self.is_high_risk.load(Ordering::Relaxed)
    }

    /// Get recommended spread multiplier based on asymmetry
    #[inline]
    pub fn get_spread_multiplier(&self) -> f64 {
        let score = self.get_asymmetry_score();
        1.0 + score * 3.0 // 1x to 4x multiplier
    }

    /// Get whale probability for next trade
    #[inline]
    pub fn get_whale_probability(&self) -> f64 {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let whales = self.whale_count.load(Ordering::Relaxed);
        whales as f64 / total as f64
    }

    /// Get informed trader probability
    #[inline]
    pub fn get_informed_probability(&self) -> f64 {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let informed = self.informed_count.load(Ordering::Relaxed);
        informed as f64 / total as f64
    }

    /// Get comprehensive statistics
    #[inline]
    pub fn get_stats(&self) -> AsymmetryStats {
        let total = self.total_count.load(Ordering::Relaxed);
        
        AsymmetryStats {
            asymmetry_score: self.get_asymmetry_score(),
            is_high_risk: self.is_high_risk(),
            total_trades: total,
            whale_count: self.whale_count.load(Ordering::Relaxed),
            informed_count: self.informed_count.load(Ordering::Relaxed),
            whale_probability: self.get_whale_probability(),
            informed_probability: self.get_informed_probability(),
            avg_price_impact: self.impact_stats.mean().unwrap_or(0.0),
            price_impact_std: self.impact_stats.std_dev().unwrap_or(0.0),
            is_valid: self.is_valid.load(Ordering::Relaxed),
        }
    }

    /// Reset all state
    #[inline]
    pub fn reset(&self) {
        self.trades.lock().clear();
        self.whale_count.store(0, Ordering::Relaxed);
        self.informed_count.store(0, Ordering::Relaxed);
        self.total_count.store(0, Ordering::Relaxed);
        self.asymmetry_score.store(0, Ordering::Relaxed);
        self.is_high_risk.store(false, Ordering::Relaxed);
        self.is_valid.store(false, Ordering::Relaxed);
    }

    /// Invalidate calculator
    #[inline]
    pub fn invalidate(&self) {
        self.is_valid.store(false, Ordering::Relaxed);
    }
}

impl Default for InformationAsymmetryCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Asymmetry statistics
#[derive(Debug, Clone, Copy)]
pub struct AsymmetryStats {
    pub asymmetry_score: f64,
    pub is_high_risk: bool,
    pub total_trades: u64,
    pub whale_count: u64,
    pub informed_count: u64,
    pub whale_probability: f64,
    pub informed_probability: f64,
    pub avg_price_impact: f64,
    pub price_impact_std: f64,
    pub is_valid: bool,
}

/// Builder pattern for AsymmetryConfig
pub struct AsymmetryBuilder {
    config: AsymmetryConfig,
}

impl AsymmetryBuilder {
    pub fn new() -> Self {
        Self {
            config: AsymmetryConfig::default(),
        }
    }

    pub fn whale_threshold(mut self, volume: f64) -> Self {
        self.config.whale_volume_threshold = (volume * 1e9) as u64;
        self
    }

    pub fn impact_threshold_bps(mut self, bps: f64) -> Self {
        self.config.impact_threshold_bps = (bps * 100.0) as u64;
        self
    }

    pub fn lookback_window_ms(mut self, ms: u64) -> Self {
        self.config.lookback_window_ms = ms;
        self
    }

    pub fn high_risk_threshold(mut self, threshold: f64) -> Self {
        self.config.high_risk_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> InformationAsymmetryCalculator {
        InformationAsymmetryCalculator::with_config(self.config)
    }
}

impl Default for AsymmetryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_flow() {
        let calc = InformationAsymmetryCalculator::new();
        
        // Add normal retail trades
        for i in 0..50 {
            let trade = AsymmetryTrade {
                timestamp_ms: 1000 + i * 100,
                price: 50000.0,
                volume: (0.1 * 1e9) as u64,
                is_buy: i % 2 == 0,
                price_change_bps: 50, // Small random movement
            };
            calc.process_trade(trade);
        }

        let score = calc.get_asymmetry_score();
        assert!(score < 0.3);
        assert!(!calc.is_high_risk());
    }

    #[test]
    fn test_whale_detection() {
        let calc = InformationAsymmetryCalculator::new();
        
        // Add whale trades
        for i in 0..30 {
            let trade = AsymmetryTrade {
                timestamp_ms: 1000 + i * 100,
                price: 50000.0,
                volume: (10.0 * 1e9) as u64, // Large whale trade
                is_buy: true,
                price_change_bps: 800, // Significant impact
            };
            calc.process_trade(trade);
        }

        let prob = calc.get_whale_probability();
        assert!(prob > 0.5);
    }

    #[test]
    fn test_informed_detection() {
        let calc = InformationAsymmetryCalculator::new();
        
        // Add informed trades (always correct direction)
        for i in 0..30 {
            let trade = AsymmetryTrade {
                timestamp_ms: 1000 + i * 100,
                price: 50000.0,
                volume: (1.0 * 1e9) as u64,
                is_buy: true,
                price_change_bps: 1000, // Price went up after buy
            };
            calc.process_trade(trade);
        }

        let prob = calc.get_informed_probability();
        assert!(prob > 0.5);
        assert!(calc.get_asymmetry_score() > 0.3);
    }

    #[test]
    fn test_spread_multiplier() {
        let calc = InformationAsymmetryCalculator::new();
        
        // Low asymmetry -> normal spread
        assert!((calc.get_spread_multiplier() - 1.0).abs() < 0.5);
        
        // Simulate high asymmetry
        calc.asymmetry_score.store((0.9 * 1e9) as u64, Ordering::Relaxed);
        calc.is_high_risk.store(true, Ordering::Relaxed);
        
        // High asymmetry -> wide spread
        assert!(calc.get_spread_multiplier() > 3.0);
    }

    #[test]
    fn test_rolling_stats() {
        let stats = RollingStats::new(10);
        
        for i in 0..15 {
            stats.add(i as f64);
        }
        
        let mean = stats.mean().unwrap();
        // Should be mean of [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        assert!((mean - 9.5).abs() < 0.1);
    }
}
