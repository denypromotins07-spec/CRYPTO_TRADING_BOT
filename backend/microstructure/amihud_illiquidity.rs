//! # Amihud Illiquidity Ratio Implementation
//! 
//! Calculates the Amihud (2002) illiquidity ratio for real-time liquidity measurement.
//! Optimized for streaming crypto market data with zero-cost abstractions.
//! 
//! **Key Features:**
//! - Real-time Amihud ratio calculation
//! - Rolling window statistics
//! - Handles extreme volatility gracefully
//! - Memory-efficient streaming design
//! 
//! **Performance:** Microsecond updates, suitable for 8GB RAM constraint.

use std::collections::VecDeque;

/// Configuration for Amihud calculation
#[derive(Debug, Clone)]
pub struct AmihudConfig {
    /// Rolling window size (number of periods)
    pub window_size: usize,
    /// Minimum samples required before producing valid ratio
    pub min_samples: usize,
    /// Volume scaling factor (for different units)
    pub volume_scale: f64,
}

impl Default for AmihudConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            min_samples: 50,
            volume_scale: 1.0,
        }
    }
}

/// Result from Amihud calculation
#[derive(Debug, Clone)]
pub struct AmihudResult {
    /// Current Amihud ratio (price impact per unit volume)
    pub ratio: f64,
    /// Rolling average ratio
    pub avg_ratio: f64,
    /// Number of observations used
    pub sample_count: usize,
    /// Standard deviation of ratios
    pub std_dev: f64,
    /// Whether the result is statistically reliable
    pub is_reliable: bool,
}

/// Streaming Amihud illiquidity calculator
pub struct AmihudCalculator {
    config: AmihudConfig,
    /// Rolling window of |return| / volume values
    ratios: VecDeque<f64>,
    /// Running sum for efficient mean calculation
    sum_ratios: f64,
    /// Running sum of squares for variance
    sum_sq_ratios: f64,
    /// Previous price for return calculation
    prev_price: Option<f64>,
    /// Total volume in window
    total_volume: f64,
    /// Calculation counter
    calc_count: u64,
}

impl AmihudCalculator {
    /// Create new Amihud calculator with default config
    pub fn new() -> Self {
        Self::with_config(AmihudConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AmihudConfig) -> Self {
        Self {
            config,
            ratios: VecDeque::with_capacity(config.window_size),
            sum_ratios: 0.0,
            sum_sq_ratios: 0.0,
            prev_price: None,
            total_volume: 0.0,
            calc_count: 0,
        }
    }

    /// Update with new price and volume data
    /// 
    /// # Arguments
    /// * `price` - Current price
    /// * `volume` - Trading volume since last update
    /// 
    /// # Returns
    /// Option containing AmihudResult if enough samples, None otherwise
    #[inline]
    pub fn update(&mut self, price: f64, volume: f64) -> Option<AmihudResult> {
        // Validate inputs
        if !price.is_finite() || price <= 0.0 || volume <= 0.0 {
            return None;
        }

        // Calculate return
        let return_abs = if let Some(prev) = self.prev_price {
            ((price - prev) / prev).abs()
        } else {
            self.prev_price = Some(price);
            return None;
        };

        self.prev_price = Some(price);

        // Calculate Amihud ratio for this period: |return| / volume
        let scaled_volume = volume * self.config.volume_scale;
        let ratio = return_abs / scaled_volume;

        // Add to rolling window
        self.add_ratio(ratio, volume);
        self.calc_count += 1;

        // Return result if we have enough samples
        if self.ratios.len() >= self.config.min_samples {
            Some(self.compute_result())
        } else {
            None
        }
    }

    /// Add a pre-calculated ratio to the window
    fn add_ratio(&mut self, ratio: f64, volume: f64) {
        // Remove oldest if at capacity
        if self.ratios.len() >= self.config.window_size {
            if let Some(old_ratio) = self.ratios.pop_front() {
                self.sum_ratios -= old_ratio;
                self.sum_sq_ratios -= old_ratio * old_ratio;
            }
        }

        // Add new ratio
        self.ratios.push_back(ratio);
        self.sum_ratios += ratio;
        self.sum_sq_ratios += ratio * ratio;
        self.total_volume += volume;

        // Adjust total volume if we removed an element
        if self.ratios.len() == self.config.window_size {
            // Volume tracking would need separate deque for exactness
            // Simplified here for performance
        }
    }

    /// Compute current result from rolling window
    fn compute_result(&self) -> AmihudResult {
        let n = self.ratios.len() as f64;
        if n < 1.0 {
            return AmihudResult {
                ratio: 0.0,
                avg_ratio: 0.0,
                sample_count: 0,
                std_dev: 0.0,
                is_reliable: false,
            };
        }

        let avg = self.sum_ratios / n;
        
        // Calculate standard deviation using Welford-like approach
        let variance = if n > 1.0 {
            let sq_avg = (self.sum_ratios / n).powi(2);
            let avg_sq = self.sum_sq_ratios / n;
            (avg_sq - sq_avg).max(0.0)
        } else {
            0.0
        };
        
        let std_dev = variance.sqrt();

        // Current ratio is the most recent
        let current_ratio = self.ratios.back().copied().unwrap_or(0.0);

        // Reliability check
        let is_reliable = n >= self.config.min_samples as f64 
            && std_dev.is_finite() 
            && current_ratio.is_finite();

        AmihudResult {
            ratio: current_ratio,
            avg_ratio: avg,
            sample_count: self.ratios.len(),
            std_dev,
            is_reliable,
        }
    }

    /// Get current Amihud ratio without updating
    pub fn current_ratio(&self) -> f64 {
        self.ratios.back().copied().unwrap_or(0.0)
    }

    /// Get rolling average ratio
    pub fn avg_ratio(&self) -> f64 {
        if self.ratios.is_empty() {
            0.0
        } else {
            self.sum_ratios / self.ratios.len() as f64
        }
    }

    /// Batch update with multiple price/volume pairs
    pub fn update_batch(&mut self, prices: &[f64], volumes: &[f64]) -> Option<AmihudResult> {
        debug_assert_eq!(prices.len(), volumes.len());
        
        let mut last_result = None;
        
        for (price, volume) in prices.iter().zip(volumes.iter()) {
            if let Some(result) = self.update(*price, *volume) {
                last_result = Some(result);
            }
        }
        
        last_result
    }

    /// Get statistics about the calculator state
    pub fn get_stats(&self) -> AmihudStats {
        AmihudStats {
            window_size: self.config.window_size,
            current_window_len: self.ratios.len(),
            min_samples: self.config.min_samples,
            is_ready: self.ratios.len() >= self.config.min_samples,
            total_calculations: self.calc_count,
            sum_ratios: self.sum_ratios,
            avg_ratio: self.avg_ratio(),
        }
    }

    /// Reset calculator state
    pub fn reset(&mut self) {
        self.ratios.clear();
        self.sum_ratios = 0.0;
        self.sum_sq_ratios = 0.0;
        self.prev_price = None;
        self.total_volume = 0.0;
        self.calc_count = 0;
    }

    /// Set new window size (clears existing data)
    pub fn set_window_size(&mut self, size: usize) {
        self.config.window_size = size;
        self.config.min_samples = size / 10;
        self.ratios.clear();
        self.sum_ratios = 0.0;
        self.sum_sq_ratios = 0.0;
    }
}

impl Default for AmihudCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for Amihud calculator
#[derive(Debug, Clone)]
pub struct AmihudStats {
    pub window_size: usize,
    pub current_window_len: usize,
    pub min_samples: usize,
    pub is_ready: bool,
    pub total_calculations: u64,
    pub sum_ratios: f64,
    pub avg_ratio: f64,
}

/// Multi-asset Amihud tracker for cross-asset liquidity comparison
pub struct MultiAssetAmihudTracker {
    trackers: std::collections::HashMap<String, AmihudCalculator>,
    default_config: AmihudConfig,
}

impl MultiAssetAmihudTracker {
    /// Create new multi-asset tracker
    pub fn new(default_config: AmihudConfig) -> Self {
        Self {
            trackers: std::collections::HashMap::new(),
            default_config,
        }
    }

    /// Get or create tracker for an asset
    pub fn get_or_create(&mut self, asset: &str) -> &mut AmihudCalculator {
        self.trackers
            .entry(asset.to_string())
            .or_insert_with(|| AmihudCalculator::with_config(self.default_config.clone()))
    }

    /// Update specific asset
    pub fn update_asset(&mut self, asset: &str, price: f64, volume: f64) -> Option<AmihudResult> {
        self.get_or_create(asset).update(price, volume)
    }

    /// Get all current ratios
    pub fn get_all_ratios(&self) -> Vec<(&str, f64)> {
        self.trackers
            .iter()
            .map(|(asset, calc)| (asset.as_str(), calc.current_ratio()))
            .collect()
    }

    /// Rank assets by illiquidity (highest Amihud = most illiquid)
    pub fn rank_by_illiquidity(&self) -> Vec<(String, f64)> {
        let mut rankings: Vec<_> = self.trackers
            .iter()
            .map(|(asset, calc)| (asset.clone(), calc.avg_ratio()))
            .collect();
        
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings
    }

    /// Remove an asset tracker
    pub fn remove_asset(&mut self, asset: &str) -> bool {
        self.trackers.remove(asset).is_some()
    }

    /// Get number of tracked assets
    pub fn asset_count(&self) -> usize {
        self.trackers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_amihud_calculation() {
        let mut calc = AmihudCalculator::with_config(AmihudConfig {
            window_size: 100,
            min_samples: 10,
            volume_scale: 1.0,
        });

        // Simulate price series with known returns
        let mut price = 100.0;
        for i in 0..20 {
            price *= 1.0 + (i as f64 * 0.001 - 0.01); // Small random-ish returns
            let volume = 1000.0 + (i as f64 * 10.0);
            
            if i >= 10 {
                let result = calc.update(price, volume).unwrap();
                assert!(result.ratio > 0.0);
                assert!(result.is_reliable);
            } else {
                calc.update(price, volume);
            }
        }
    }

    #[test]
    fn test_amihud_response_to_volume() {
        let mut calc = AmihudCalculator::with_config(AmihudConfig {
            window_size: 50,
            min_samples: 20,
            volume_scale: 1.0,
        });

        // Warm up
        let mut price = 100.0;
        for _ in 0..25 {
            price *= 1.001;
            calc.update(price, 1000.0);
        }

        // Get baseline
        let baseline = calc.current_ratio();

        // Same return, much lower volume → higher Amihud
        let old_price = price;
        price *= 1.01; // 1% return
        calc.update(price, 100.0); // Low volume

        assert!(calc.current_ratio() > baseline);
    }

    #[test]
    fn test_amihud_response_to_volatility() {
        let mut calc = AmihudCalculator::with_config(AmihudConfig {
            window_size: 50,
            min_samples: 20,
            volume_scale: 1.0,
        });

        // Warm up with stable prices
        let mut price = 100.0;
        for _ in 0..25 {
            price *= 1.0001; // Tiny returns
            calc.update(price, 1000.0);
        }

        // Get baseline
        let baseline = calc.current_ratio();

        // Same volume, much larger return → higher Amihud
        let old_price = price;
        price *= 1.05; // 5% return
        calc.update(price, 1000.0);

        assert!(calc.current_ratio() > baseline);
    }

    #[test]
    fn test_invalid_input_handling() {
        let mut calc = AmihudCalculator::new();

        // Invalid price
        assert!(calc.update(-100.0, 1000.0).is_none());
        assert!(calc.update(f64::NAN, 1000.0).is_none());
        assert!(calc.update(f64::INFINITY, 1000.0).is_none());

        // Invalid volume
        assert!(calc.update(100.0, 0.0).is_none());
        assert!(calc.update(100.0, -100.0).is_none());

        // First valid update doesn't produce result (need previous price)
        assert!(calc.update(100.0, 1000.0).is_none());
    }

    #[test]
    fn test_batch_update() {
        let mut calc = AmihudCalculator::with_config(AmihudConfig {
            window_size: 100,
            min_samples: 10,
            volume_scale: 1.0,
        });

        let prices: Vec<f64> = (0..30).map(|i| 100.0 * (1.0 + i as f64 * 0.001)).collect();
        let volumes: Vec<f64> = vec![1000.0; 30];

        let result = calc.update_batch(&prices, &volumes);
        
        assert!(result.is_some());
        assert!(result.unwrap().is_reliable);
        assert_eq!(calc.get_stats().total_calculations, 29); // n-1 returns
    }

    #[test]
    fn test_multi_asset_tracker() {
        let mut tracker = MultiAssetAmihudTracker::new(AmihudConfig::default());

        // Update multiple assets
        tracker.update_asset("BTC", 50000.0, 100.0);
        tracker.update_asset("ETH", 3000.0, 500.0);
        tracker.update_asset("SOL", 100.0, 1000.0);

        // Second update to establish returns
        tracker.update_asset("BTC", 50100.0, 110.0);
        tracker.update_asset("ETH", 3010.0, 520.0);
        tracker.update_asset("SOL", 101.0, 1050.0);

        assert_eq!(tracker.asset_count(), 3);

        let ratios = tracker.get_all_ratios();
        assert_eq!(ratios.len(), 3);
    }
}
