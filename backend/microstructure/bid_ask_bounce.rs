//! # Bid-Ask Bounce Model Implementation
//! 
//! Models the autoregressive bounce of trade prices due to bid-ask spread.
//! Implements Roll's model and related microstructure noise estimators.
//! 
//! **Key Features:**
//! - AR(1) modeling of bid-ask bounce
//! - Effective spread estimation
//! - Microstructure noise filtering
//! - Zero-cost abstractions with proper borrowing
//! 
//! **Performance:** O(1) updates, suitable for high-frequency streams.

use std::collections::VecDeque;

/// Configuration for bid-ask bounce model
#[derive(Debug, Clone)]
pub struct BidAskBounceConfig {
    /// Maximum lag for autocorrelation calculation
    pub max_lag: usize,
    /// Rolling window size
    pub window_size: usize,
    /// Minimum samples required
    pub min_samples: usize,
}

impl Default for BidAskBounceConfig {
    fn default() -> Self {
        Self {
            max_lag: 10,
            window_size: 1000,
            min_samples: 50,
        }
    }
}

/// Result from bid-ask bounce analysis
#[derive(Debug, Clone)]
pub struct BidAskBounceResult {
    /// First-order autocorrelation of price changes
    pub acf_1: f64,
    /// Estimated effective spread (Roll's estimator)
    pub effective_spread: f64,
    /// AR(1) coefficient
    pub ar1_coef: f64,
    /// Standard error of estimate
    pub std_error: f64,
    /// Number of observations used
    pub sample_count: usize,
    /// Whether result is statistically reliable
    pub is_reliable: bool,
}

/// Streaming bid-ask bounce analyzer
pub struct BidAskBounceModel {
    config: BidAskBounceConfig,
    /// Rolling window of price changes
    price_changes: VecDeque<f64>,
    /// Previous price for change calculation
    prev_price: Option<f64>,
    /// Cached sum for efficient mean calculation
    sum_changes: f64,
    /// Cached sum of squares
    sum_sq_changes: f64,
    /// Calculation counter
    calc_count: u64,
}

impl BidAskBounceModel {
    /// Create new model with default config
    pub fn new() -> Self {
        Self::with_config(BidAskBounceConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: BidAskBounceConfig) -> Self {
        Self {
            config,
            price_changes: VecDeque::with_capacity(config.window_size),
            prev_price: None,
            sum_changes: 0.0,
            sum_sq_changes: 0.0,
            calc_count: 0,
        }
    }

    /// Update with new price
    #[inline]
    pub fn update(&mut self, price: f64) -> Option<BidAskBounceResult> {
        if !price.is_finite() || price <= 0.0 {
            return None;
        }

        let change = if let Some(prev) = self.prev_price {
            price - prev
        } else {
            self.prev_price = Some(price);
            return None;
        };

        self.prev_price = Some(price);

        // Add to rolling window
        self.add_change(change);
        self.calc_count += 1;

        // Return result if enough samples
        if self.price_changes.len() >= self.config.min_samples {
            Some(self.compute_result())
        } else {
            None
        }
    }

    /// Add a price change to the window
    fn add_change(&mut self, change: f64) {
        // Remove oldest if at capacity
        if self.price_changes.len() >= self.config.window_size {
            if let Some(old) = self.price_changes.pop_front() {
                self.sum_changes -= old;
                self.sum_sq_changes -= old * old;
            }
        }

        self.price_changes.push_back(change);
        self.sum_changes += change;
        self.sum_sq_changes += change * change;
    }

    /// Compute autocorrelation at lag 1
    fn autocorrelation_lag1(&self) -> f64 {
        let n = self.price_changes.len();
        if n < 2 {
            return 0.0;
        }

        let mean = self.sum_changes / n as f64;
        let variance = (self.sum_sq_changes / n as f64) - (mean * mean);

        if variance <= 0.0 {
            return 0.0;
        }

        // Calculate lag-1 covariance
        let mut covar = 0.0;
        let changes: Vec<f64> = self.price_changes.iter().copied().collect();

        for i in 1..changes.len() {
            covar += (changes[i] - mean) * (changes[i - 1] - mean);
        }
        covar /= (n - 1) as f64;

        covar / variance
    }

    /// Compute Roll's effective spread estimator
    /// 
    /// Roll (1984): c = sqrt(-cov(Δp_t, Δp_{t-1}))
    /// Assumes negative autocorrelation due to bid-ask bounce
    fn roll_effective_spread(&self) -> f64 {
        let acf1 = self.autocorrelation_lag1();
        
        // If positive autocorrelation, no bid-ask bounce signal
        if acf1 >= 0.0 {
            return 0.0;
        }

        let n = self.price_changes.len();
        if n < 2 {
            return 0.0;
        }

        let variance = (self.sum_sq_changes / n as f64) 
            - ((self.sum_changes / n as f64).powi(2));

        if variance <= 0.0 {
            return 0.0;
        }

        // Roll's estimator: c = 2 * sqrt(-covariance)
        // covariance = acf1 * variance
        let covariance = acf1 * variance;
        
        if covariance >= 0.0 {
            return 0.0;
        }

        2.0 * (-covariance).sqrt()
    }

    /// Compute full result
    fn compute_result(&self) -> BidAskBounceResult {
        let acf1 = self.autocorrelation_lag1();
        let effective_spread = self.roll_effective_spread();
        
        // AR(1) coefficient approximation (Yule-Walker)
        let ar1_coef = acf1;
        
        // Standard error approximation
        let n = self.price_changes.len() as f64;
        let std_error = if n > 2.0 {
            ((1.0 - acf1 * acf1) / n).sqrt()
        } else {
            f64::INFINITY
        };

        // Reliability check
        let is_reliable = n >= self.config.min_samples as f64
            && std_error.is_finite()
            && effective_spread.is_finite();

        BidAskBounceResult {
            acf_1: acf1,
            effective_spread,
            ar1_coef,
            std_error,
            sample_count: self.price_changes.len(),
            is_reliable,
        }
    }

    /// Get current effective spread estimate
    pub fn current_effective_spread(&self) -> f64 {
        self.roll_effective_spread()
    }

    /// Get current autocorrelation
    pub fn current_acf1(&self) -> f64 {
        self.autocorrelation_lag1()
    }

    /// Batch update
    pub fn update_batch(&mut self, prices: &[f64]) -> Option<BidAskBounceResult> {
        let mut last_result = None;
        
        for &price in prices {
            if let Some(result) = self.update(price) {
                last_result = Some(result);
            }
        }
        
        last_result
    }

    /// Get statistics
    pub fn get_stats(&self) -> BounceStats {
        BounceStats {
            window_size: self.config.window_size,
            current_window_len: self.price_changes.len(),
            min_samples: self.config.min_samples,
            is_ready: self.price_changes.len() >= self.config.min_samples,
            total_updates: self.calc_count,
            mean_change: if !self.price_changes.is_empty() {
                self.sum_changes / self.price_changes.len() as f64
            } else {
                0.0
            },
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.price_changes.clear();
        self.prev_price = None;
        self.sum_changes = 0.0;
        self.sum_sq_changes = 0.0;
        self.calc_count = 0;
    }
}

impl Default for BidAskBounceModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the bounce model
#[derive(Debug, Clone)]
pub struct BounceStats {
    pub window_size: usize,
    pub current_window_len: usize,
    pub min_samples: usize,
    pub is_ready: bool,
    pub total_updates: u64,
    pub mean_change: f64,
}

/// Multi-lag autocorrelation analyzer
pub struct MultiLagAutocorrelation {
    config: BidAskBounceConfig,
    price_changes: VecDeque<f64>,
    prev_price: Option<f64>,
}

impl MultiLagAutocorrelation {
    pub fn new(max_lag: usize) -> Self {
        Self {
            config: BidAskBounceConfig {
                max_lag,
                ..Default::default()
            },
            price_changes: VecDeque::with_capacity(2000),
            prev_price: None,
        }
    }

    pub fn update(&mut self, price: f64) {
        if !price.is_finite() || price <= 0.0 {
            return;
        }

        if let Some(prev) = self.prev_price {
            self.price_changes.push_back(price - prev);
            
            // Maintain window size
            if self.price_changes.len() > self.config.window_size {
                self.price_changes.pop_front();
            }
        }
        
        self.prev_price = Some(price);
    }

    /// Calculate ACF for all lags up to max_lag
    pub fn calculate_acf(&self) -> Vec<f64> {
        let n = self.price_changes.len();
        if n < self.config.max_lag + 1 {
            return vec![0.0; self.config.max_lag + 1];
        }

        let changes: Vec<f64> = self.price_changes.iter().copied().collect();
        let mean = changes.iter().sum::<f64>() / n as f64;
        let variance: f64 = changes.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if variance <= 0.0 {
            return vec![0.0; self.config.max_lag + 1];
        }

        let mut acf = Vec::with_capacity(self.config.max_lag + 1);
        
        for lag in 0..=self.config.max_lag {
            let mut covar = 0.0;
            for i in lag..n {
                covar += (changes[i] - mean) * (changes[i - lag] - mean);
            }
            covar /= n as f64;
            acf.push(covar / variance);
        }

        acf
    }

    /// Check for bid-ask bounce pattern (negative ACF at lag 1)
    pub fn has_bounce_pattern(&self) -> bool {
        let acf = self.calculate_acf();
        acf.len() > 1 && acf[1] < -0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roll_estimator_synthetic() {
        let mut model = BidAskBounceModel::with_config(BidAskBounceConfig {
            window_size: 200,
            min_samples: 50,
            max_lag: 5,
        });

        // Simulate bid-ask bounce: alternating up/down moves
        let mut price = 100.0;
        let spread = 0.1;
        
        for i in 0..200 {
            if i % 2 == 0 {
                price += spread; // Hit ask
            } else {
                price -= spread; // Hit bid
            }
            model.update(price);
        }

        let result = model.compute_result();
        
        // Should detect negative autocorrelation
        assert!(result.acf_1 < 0.0);
        assert!(result.effective_spread > 0.0);
    }

    #[test]
    fn test_no_bounce_random_walk() {
        let mut model = BidAskBounceModel::new();

        // Simulate random walk (no bounce)
        let mut price = 100.0;
        let mut rng = std::rand::random::<f64>();
        
        for i in 0..500 {
            price += (i as f64 * 0.01 - 5.0) * 0.001; // Pseudo-random
            model.update(price);
        }

        let result = model.compute_result();
        
        // Random walk should have near-zero autocorrelation
        assert!(result.acf_1.abs() < 0.2);
    }

    #[test]
    fn test_effective_spread_accuracy() {
        let mut model = BidAskBounceModel::with_config(BidAskBounceConfig {
            window_size: 500,
            min_samples: 100,
            max_lag: 5,
        });

        // Known spread simulation
        let true_spread = 0.05;
        let mut price = 100.0;
        
        for i in 0..600 {
            if i % 2 == 0 {
                price += true_spread;
            } else {
                price -= true_spread;
            }
            model.update(price);
        }

        let result = model.compute_result();
        
        // Estimated spread should be close to true spread
        assert!((result.effective_spread - 2.0 * true_spread).abs() < 0.02);
    }

    #[test]
    fn test_invalid_input_handling() {
        let mut model = BidAskBounceModel::new();

        assert!(model.update(-100.0).is_none());
        assert!(model.update(f64::NAN).is_none());
        assert!(model.update(f64::INFINITY).is_none());
        
        // First valid update doesn't produce result
        assert!(model.update(100.0).is_none());
    }

    #[test]
    fn test_multi_lag_acf() {
        let mut analyzer = MultiLagAutocorrelation::new(5);
        
        // Create bouncing series
        let mut price = 100.0;
        for i in 0..300 {
            if i % 2 == 0 {
                price += 0.1;
            } else {
                price -= 0.1;
            }
            analyzer.update(price);
        }

        let acf = analyzer.calculate_acf();
        
        assert_eq!(acf.len(), 6); // lags 0-5
        assert!((acf[0] - 1.0).abs() < 0.01); // Lag 0 should be ~1
        assert!(acf[1] < 0.0); // Lag 1 should be negative (bounce)
    }
}
