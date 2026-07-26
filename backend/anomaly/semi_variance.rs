//! Realized Semi-Variance Module: Isolating Downside Risk for Asymmetric Analysis
//!
//! This module implements realized semi-variance calculations to separate
//! upside and downside volatility, critical for asymmetric risk management
//! in crypto markets where crash risk differs significantly from rally risk.
//!
//! Key Features:
//! - Realized semi-variance decomposition
//! - Downside deviation tracking
//! - Upside potential ratio calculation
//! - Memory-efficient streaming updates
//! - Integration with bipower variation for jump-robust estimates

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, AtomicF64, Ordering};

/// Configuration for semi-variance calculations
#[derive(Debug, Clone)]
pub struct SemiVarianceConfig {
    /// Rolling window size for calculations
    pub window_size: usize,
    /// Target return for downside deviation (often 0 or risk-free rate)
    pub target_return: f64,
    /// Sampling frequency in milliseconds
    pub sampling_freq_ms: u64,
    /// Use intraday data flag
    pub use_intraday: bool,
}

impl Default for SemiVarianceConfig {
    fn default() -> Self {
        Self {
            window_size: 1440, // 24 hours at 1-minute intervals
            target_return: 0.0,
            sampling_freq_ms: 60000,
            use_intraday: true,
        }
    }
}

/// Result of semi-variance analysis
#[derive(Debug, Clone, Copy)]
pub struct SemiVarianceResult {
    /// Timestamp of calculation
    pub timestamp_ms: u64,
    /// Total realized variance
    pub total_variance: f64,
    /// Realized semi-variance (downside only)
    pub downside_variance: f64,
    /// Realized semi-variance (upside only)
    pub upside_variance: f64,
    /// Downside deviation (sqrt of downside variance)
    pub downside_deviation: f64,
    /// Upside deviation (sqrt of upside variance)
    pub upside_deviation: f64,
    /// Semi-variance ratio (downside / total)
    pub semi_variance_ratio: f64,
    /// Sortino ratio component
    pub sortino_component: f64,
    /// Number of negative returns in window
    pub negative_count: usize,
    /// Number of positive returns in window
    pub positive_count: usize,
}

/// Streaming realized semi-variance calculator
pub struct RealizedSemiVariance {
    config: SemiVarianceConfig,
    /// Rolling window of returns
    returns: VecDeque<f64>,
    /// Sum of squared negative returns (for downside variance)
    sum_sq_negative: AtomicF64,
    /// Sum of squared positive returns (for upside variance)
    sum_sq_positive: AtomicF64,
    /// Count of negative returns
    negative_count: AtomicUsize,
    /// Count of positive returns
    positive_count: AtomicUsize,
    /// Current mean return (for target adjustment)
    mean_return: AtomicF64,
    /// Total observations processed
    total_count: usize,
}

impl RealizedSemiVariance {
    /// Create a new semi-variance calculator
    pub fn new(config: SemiVarianceConfig) -> Self {
        Self {
            config,
            returns: VecDeque::with_capacity(config.window_size),
            sum_sq_negative: AtomicF64::new(0.0),
            sum_sq_positive: AtomicF64::new(0.0),
            negative_count: AtomicUsize::new(0),
            positive_count: AtomicUsize::new(0),
            mean_return: AtomicF64::new(0.0),
            total_count: 0,
        }
    }

    /// Update with a new return observation
    pub fn update(&mut self, timestamp_ms: u64, log_return: f64) -> SemiVarianceResult {
        // Calculate excess return over target
        let excess_return = log_return - self.config.target_return;

        // Update rolling window
        if self.returns.len() >= self.config.window_size {
            if let Some(old_return) = self.returns.pop_front() {
                self._remove_observation(old_return);
            }
        }

        self.returns.push_back(log_return);
        self._add_observation(excess_return);
        self.total_count += 1;

        // Calculate results
        self._calculate_result(timestamp_ms)
    }

    /// Add an observation to the running sums
    fn _add_observation(&self, excess_return: f64) {
        let sq_return = excess_return * excess_return;

        if excess_return < 0.0 {
            // Downside return
            let current = self.sum_sq_negative.load(Ordering::Relaxed);
            self.sum_sq_negative.store(current + sq_return, Ordering::Relaxed);

            let count = self.negative_count.load(Ordering::Relaxed);
            self.negative_count.store(count + 1, Ordering::Relaxed);
        } else {
            // Upside return
            let current = self.sum_sq_positive.load(Ordering::Relaxed);
            self.sum_sq_positive.store(current + sq_return, Ordering::Relaxed);

            let count = self.positive_count.load(Ordering::Relaxed);
            self.positive_count.store(count + 1, Ordering::Relaxed);
        }

        // Update running mean
        let old_mean = self.mean_return.load(Ordering::Relaxed);
        let n = self.returns.len() as f64;
        let new_mean = old_mean + (excess_return - old_mean) / n;
        self.mean_return.store(new_mean, Ordering::Relaxed);
    }

    /// Remove an old observation from the running sums
    fn _remove_observation(&mut self, old_return: f64) {
        let excess_return = old_return - self.config.target_return;
        let sq_return = excess_return * excess_return;

        if excess_return < 0.0 {
            let current = self.sum_sq_negative.load(Ordering::Relaxed);
            self.sum_sq_negative.store((current - sq_return).max(0.0), Ordering::Relaxed);

            let count = self.negative_count.load(Ordering::Relaxed);
            if count > 0 {
                self.negative_count.store(count - 1, Ordering::Relaxed);
            }
        } else {
            let current = self.sum_sq_positive.load(Ordering::Relaxed);
            self.sum_sq_positive.store((current - sq_return).max(0.0), Ordering::Relaxed);

            let count = self.positive_count.load(Ordering::Relaxed);
            if count > 0 {
                self.positive_count.store(count - 1, Ordering::Relaxed);
            }
        }

        // Update running mean
        let old_mean = self.mean_return.load(Ordering::Relaxed);
        let n = self.returns.len() as f64;
        if n > 0.0 {
            let new_mean = (old_mean * (n + 1.0) - excess_return) / n;
            self.mean_return.store(new_mean, Ordering::Relaxed);
        }
    }

    /// Calculate the semi-variance result
    fn _calculate_result(&self, timestamp_ms: u64) -> SemiVarianceResult {
        let downside_var = self.sum_sq_negative.load(Ordering::Relaxed);
        let upside_var = self.sum_sq_positive.load(Ordering::Relaxed);
        let total_var = downside_var + upside_var;

        let neg_count = self.negative_count.load(Ordering::Relaxed);
        let pos_count = self.positive_count.load(Ordering::Relaxed);

        // Calculate deviations (annualized assuming daily data)
        let annualization_factor = if self.config.use_intraday {
            // For intraday: scale to daily then annualize
            (1440.0 / self.returns.len().max(1) as f64) * 365.0
        } else {
            252.0 // Trading days per year
        };

        let downside_dev = (downside_var * annualization_factor).sqrt();
        let upside_dev = (upside_var * annualization_factor).sqrt();

        // Semi-variance ratio: proportion of total variance from downside
        let semi_var_ratio = if total_var > 0.0 {
            downside_var / total_var
        } else {
            0.0
        };

        // Sortino component: excess return / downside deviation
        let mean_excess = self.mean_return.load(Ordering::Relaxed);
        let sortino_component = if downside_dev > 0.0 {
            mean_excess * annualization_factor.sqrt() / downside_dev
        } else {
            0.0
        };

        SemiVarianceResult {
            timestamp_ms,
            total_variance: total_var,
            downside_variance: downside_var,
            upside_variance: upside_var,
            downside_deviation: downside_dev,
            upside_deviation: upside_dev,
            semi_variance_ratio: semi_var_ratio,
            sortino_component,
            negative_count: neg_count,
            positive_count: pos_count,
        }
    }

    /// Get the current downside risk measure
    pub fn get_downside_risk(&self) -> f64 {
        let downside_var = self.sum_sq_negative.load(Ordering::Relaxed);
        let annualization_factor = if self.config.use_intraday { 365.0 } else { 252.0 };
        (downside_var * annualization_factor).sqrt()
    }

    /// Get the upside potential ratio
    pub fn get_upside_potential_ratio(&self) -> f64 {
        let downside_var = self.sum_sq_negative.load(Ordering::Relaxed);
        let upside_var = self.sum_sq_positive.load(Ordering::Relaxed);

        if downside_var <= 0.0 {
            return f64::INFINITY;
        }

        (upside_var / downside_var).sqrt()
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.returns.clear();
        self.sum_sq_negative.store(0.0, Ordering::Relaxed);
        self.sum_sq_positive.store(0.0, Ordering::Relaxed);
        self.negative_count.store(0, Ordering::Relaxed);
        self.positive_count.store(0, Ordering::Relaxed);
        self.mean_return.store(0.0, Ordering::Relaxed);
        self.total_count = 0;
    }

    /// Get window statistics
    pub fn get_window_stats(&self) -> (usize, usize, usize) {
        (
            self.returns.len(),
            self.negative_count.load(Ordering::Relaxed),
            self.positive_count.load(Ordering::Relaxed),
        )
    }
}

/// Multi-asset semi-variance analyzer for portfolio risk
pub struct PortfolioSemiVariance {
    assets: std::collections::HashMap<String, RealizedSemiVariance>,
    correlations: std::collections::HashMap<(String, String), f64>,
}

impl PortfolioSemiVariance {
    pub fn new(config: SemiVarianceConfig) -> Self {
        Self {
            assets: std::collections::HashMap::new(),
            correlations: std::collections::HashMap::new(),
        }
    }

    pub fn add_asset(&mut self, symbol: &str, config: Option<SemiVarianceConfig>) {
        let cfg = config.unwrap_or_else(SemiVarianceConfig::default);
        self.assets.insert(symbol.to_string(), RealizedSemiVariance::new(cfg));
    }

    pub fn update_asset(&mut self, symbol: &str, timestamp_ms: u64, log_return: f64) -> Option<SemiVarianceResult> {
        self.assets.get_mut(symbol).map(|a| a.update(timestamp_ms, log_return))
    }

    /// Calculate portfolio-level downside risk
    pub fn calculate_portfolio_downside(
        &self,
        weights: &std::collections::HashMap<String, f64>
    ) -> f64 {
        // Simplified: sum of weighted individual downside risks
        // Full implementation would include correlation structure
        let mut portfolio_downside = 0.0;

        for (symbol, weight) in weights {
            if let Some(asset) = self.assets.get(symbol) {
                let asset_downside = asset.get_downside_risk();
                portfolio_downside += weight * weight * asset_downside * asset_downside;
            }
        }

        portfolio_downside.sqrt()
    }

    /// Update correlation between two assets
    pub fn update_correlation(&mut self, asset1: &str, asset2: &str, correlation: f64) {
        let key = if asset1 < asset2 {
            (asset1.to_string(), asset2.to_string())
        } else {
            (asset2.to_string(), asset1.to_string())
        };
        self.correlations.insert(key, correlation);
    }
}

/// Jump-robust semi-variance using bipower variation
pub struct JumpRobustSemiVariance {
    base_calculator: RealizedSemiVariance,
    /// Bipower variation for jump adjustment
    bp_sum: AtomicF64,
    last_abs_return: AtomicF64,
}

impl JumpRobustSemiVariance {
    pub fn new(config: SemiVarianceConfig) -> Self {
        Self {
            base_calculator: RealizedSemiVariance::new(config.clone()),
            bp_sum: AtomicF64::new(0.0),
            last_abs_return: AtomicF64::new(0.0),
        }
    }

    pub fn update(&mut self, timestamp_ms: u64, log_return: f64) -> SemiVarianceResult {
        // First get base result
        let mut result = self.base_calculator.update(timestamp_ms, log_return);

        // Update bipower variation
        let abs_return = log_return.abs();
        let last_abs = self.last_abs_return.load(Ordering::Relaxed);

        if last_abs > 0.0 {
            let bp_product = last_abs * abs_return;
            let current_bp = self.bp_sum.load(Ordering::Relaxed);
            self.bp_sum.store(current_bp + bp_product, Ordering::Relaxed);
        }
        self.last_abs_return.store(abs_return, Ordering::Relaxed);

        // Adjust downside variance for jumps
        // If bipower suggests continuous process, reduce attributed downside
        let bp_current = self.bp_sum.load(Ordering::Relaxed);
        if bp_current > 0.0 && result.downside_variance > 0.0 {
            // Scale factor based on bipower ratio
            let pi_half = std::f64::consts::FRAC_PI_2;
            let adjusted_downside = result.downside_variance * (1.0 - (bp_current * pi_half / result.total_variance).min(1.0));
            result.downside_variance = adjusted_downside.max(0.0);
            result.downside_deviation = result.downside_variance.sqrt();
        }

        result
    }

    pub fn get_continuous_downside(&self) -> f64 {
        let bp_current = self.bp_sum.load(Ordering::Relaxed);
        let pi_half = std::f64::consts::FRAC_PI_2;
        (bp_current * pi_half).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semi_variance_calculation() {
        let config = SemiVarianceConfig::default();
        let mut calculator = RealizedSemiVariance::new(config);

        // Add mixed returns
        let returns = vec![-0.02, 0.01, -0.03, 0.02, -0.01, 0.03, -0.04, 0.01];
        
        for (i, ret) in returns.iter().enumerate() {
            let result = calculator.update(i as u64 * 60000, *ret);
            
            if i >= 2 {
                // Should have both upside and downside variance
                assert!(result.downside_variance > 0.0);
                assert!(result.upside_variance > 0.0);
            }
        }

        let final_result = calculator._calculate_result(8 * 60000);
        println!("Downside variance: {}", final_result.downside_variance);
        println!("Upside variance: {}", final_result.upside_variance);
        println!("Semi-variance ratio: {}", final_result.semi_variance_ratio);
    }

    #[test]
    fn test_upside_potential_ratio() {
        let config = SemiVarianceConfig::default();
        let mut calculator = RealizedSemiVariance::new(config);

        // Add more positive than negative returns
        for i in 0..100 {
            let ret = if i % 3 == 0 { -0.01 } else { 0.02 };
            calculator.update(i as u64 * 60000, ret);
        }

        let upr = calculator.get_upside_potential_ratio();
        assert!(upr > 1.0, "UPR should be > 1 with more upside");
    }

    #[test]
    fn test_reset_functionality() {
        let config = SemiVarianceConfig::default();
        let mut calculator = RealizedSemiVariance::new(config);

        // Add some data
        for i in 0..10 {
            calculator.update(i as u64 * 60000, 0.01);
        }

        assert!(calculator.get_downside_risk() >= 0.0);

        // Reset
        calculator.reset();

        let result = calculator._calculate_result(0);
        assert_eq!(result.downside_variance, 0.0);
        assert_eq!(result.upside_variance, 0.0);
    }
}
