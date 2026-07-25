//! Cointegration Tracker - Rolling Engle-Granger Tests in Rust
//!
//! This module implements rolling Engle-Granger cointegration tests to identify
//! pairs of assets that maintain a long-term equilibrium relationship.
//! Automatically halts pairs trading if p-value exceeds 0.05 threshold.
//!
//! Chapter 3: Statistical Arbitrage, Pairs Trading, and Cointegration Execution

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum window size for rolling calculations
pub const MAX_WINDOW_SIZE: usize = 256;

/// Minimum window size for valid statistical tests
pub const MIN_WINDOW_SIZE: usize = 30;

/// Cointegration test result
#[derive(Clone, Copy, Debug)]
pub struct CointegrationResult {
    /// Test statistic (ADF t-statistic)
    pub test_statistic: f64,
    /// P-value from the test
    pub p_value: f64,
    /// Critical value at 5% significance
    pub critical_value_5pct: f64,
    /// Whether series are cointegrated (p < 0.05)
    pub is_cointegrated: bool,
    /// Hedge ratio (beta coefficient)
    pub hedge_ratio: f64,
    /// Timestamp of calculation
    pub timestamp_ns: u64,
    /// Window size used
    pub window_size: usize,
}

/// Price pair observation
#[derive(Clone, Copy, Debug)]
pub struct PricePair {
    pub price_a: f64,
    pub price_b: f64,
    pub timestamp_ns: u64,
}

/// Rolling Engle-Granger cointegration tracker
pub struct CointegrationTracker {
    /// Symbol A name
    symbol_a: &'static str,
    /// Symbol B name
    symbol_b: &'static str,
    /// Rolling window of price pairs
    price_history: VecDeque<PricePair>,
    /// Maximum window size
    window_size: usize,
    /// Last cointegration result
    last_result: Option<CointegrationResult>,
    /// Whether trading should be halted (p > 0.05)
    trading_halted: AtomicBool,
    /// Consecutive non-cointegrated count
    non_cointegrated_count: AtomicU64,
    /// Total tests performed
    total_tests: AtomicU64,
    /// Cointegration failures (for monitoring)
    cointegration_failures: AtomicU64,
}

impl CointegrationTracker {
    /// Create a new cointegration tracker for a pair
    pub fn new(symbol_a: &'static str, symbol_b: &'static str, window_size: usize) -> Self {
        let window_size = window_size.min(MAX_WINDOW_SIZE).max(MIN_WINDOW_SIZE);
        
        Self {
            symbol_a,
            symbol_b,
            price_history: VecDeque::with_capacity(window_size),
            window_size,
            last_result: None,
            trading_halted: AtomicBool::new(true), // Start halted until we have data
            non_cointegrated_count: AtomicU64::new(0),
            total_tests: AtomicU64::new(0),
            cointegration_failures: AtomicU64::new(0),
        }
    }

    /// Add a new price observation
    #[inline]
    pub fn add_price_observation(&mut self, price_a: f64, price_b: f64) {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let pair = PricePair {
            price_a,
            price_b,
            timestamp_ns: now_ns,
        };
        
        self.price_history.push_back(pair);
        
        // Maintain window size
        while self.price_history.len() > self.window_size {
            self.price_history.pop_front();
        }
    }

    /// Perform Engle-Granger cointegration test
    /// Returns CointegrationResult or None if insufficient data
    #[inline]
    pub fn test_cointegration(&mut self) -> Option<CointegrationResult> {
        if self.price_history.len() < MIN_WINDOW_SIZE {
            return None;
        }

        let prices_a: Vec<f64> = self.price_history.iter().map(|p| p.price_a).collect();
        let prices_b: Vec<f64> = self.price_history.iter().map(|p| p.price_b).collect();

        // Step 1: Calculate hedge ratio (beta) using OLS
        let hedge_ratio = Self::calculate_hedge_ratio(&prices_a, &prices_b);
        
        if hedge_ratio.abs() < 1e-10 || !hedge_ratio.is_finite() {
            return None;
        }

        // Step 2: Calculate spread (residuals): spread = price_a - beta * price_b
        let spread: Vec<f64> = prices_a.iter()
            .zip(prices_b.iter())
            .map(|(&a, &b)| a - hedge_ratio * b)
            .collect();

        // Step 3: Perform ADF test on spread
        let (test_stat, p_value) = Self::adf_test_simple(&spread);

        // Critical value for ADF test at 5% (approximate, depends on sample size)
        let critical_value = -2.86; // Standard critical value for large samples

        let is_cointegrated = p_value < 0.05 && test_stat < critical_value;

        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let result = CointegrationResult {
            test_statistic: test_stat,
            p_value,
            critical_value_5pct: critical_value,
            is_cointegrated,
            hedge_ratio,
            timestamp_ns: now_ns,
            window_size: self.price_history.len(),
        };

        self.last_result = Some(result);
        self.total_tests.fetch_add(1, Ordering::SeqCst);

        // Update trading halt status
        if !is_cointegrated {
            self.non_cointegrated_count.fetch_add(1, Ordering::SeqCst);
            self.cointegration_failures.fetch_add(1, Ordering::SeqCst);
            
            // Halt trading if p-value exceeds 0.05
            self.trading_halted.store(true, Ordering::SeqCst);
        } else {
            // Reset non-cointegrated counter on success
            self.non_cointegrated_count.store(0, Ordering::SeqCst);
            
            // Allow trading if we have consistent cointegration
            if self.non_cointegrated_count.load(Ordering::Acquire) == 0 {
                self.trading_halted.store(false, Ordering::SeqCst);
            }
        }

        self.last_result
    }

    /// Check if trading is currently allowed
    #[inline]
    pub fn is_trading_allowed(&self) -> bool {
        !self.trading_halted.load(Ordering::Acquire)
    }

    /// Get the last cointegration result
    #[inline]
    pub fn last_result(&self) -> Option<CointegrationResult> {
        self.last_result
    }

    /// Get current hedge ratio
    #[inline]
    pub fn current_hedge_ratio(&self) -> Option<f64> {
        self.last_result.map(|r| r.hedge_ratio)
    }

    /// Reset the tracker (clear history)
    #[inline]
    pub fn reset(&mut self) {
        self.price_history.clear();
        self.last_result = None;
        self.trading_halted.store(true, Ordering::Release);
        self.non_cointegrated_count.store(0, Ordering::Release);
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.total_tests.load(Ordering::Acquire),
            self.cointegration_failures.load(Ordering::Acquire),
            self.non_cointegrated_count.load(Ordering::Acquire),
        )
    }

    /// Calculate hedge ratio using simple linear regression (OLS)
    /// beta = Cov(A, B) / Var(B)
    #[inline]
    fn calculate_hedge_ratio(prices_a: &[f64], prices_b: &[f64]) -> f64 {
        let n = prices_a.len() as f64;
        
        if n < 2.0 {
            return 0.0;
        }

        // Calculate means
        let mean_a: f64 = prices_a.iter().sum::<f64>() / n;
        let mean_b: f64 = prices_b.iter().sum::<f64>() / n;

        // Calculate covariance and variance
        let mut cov = 0.0;
        let mut var_b = 0.0;

        for (&a, &b) in prices_a.iter().zip(prices_b.iter()) {
            let dev_a = a - mean_a;
            let dev_b = b - mean_b;
            cov += dev_a * dev_b;
            var_b += dev_b * dev_b;
        }

        if var_b.abs() < 1e-10 {
            return 0.0;
        }

        cov / var_b
    }

    /// Simplified ADF test implementation
    /// Returns (test_statistic, approximate_p_value)
    /// 
    /// Note: This is a simplified version for performance.
    /// Production systems should use a more accurate implementation
    /// or pre-computed critical values.
    #[inline]
    fn adf_test_simple(series: &[f64]) -> (f64, f64) {
        if series.len() < 3 {
            return (0.0, 1.0);
        }

        // Create differenced series: Δy_t = y_t - y_{t-1}
        let diffs: Vec<f64> = series.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        // Lagged series (excluding last point)
        let lagged: Vec<f64> = series[..series.len()-1].to_vec();

        // Run regression: Δy_t = α + β * y_{t-1} + ε_t
        // We care about the t-statistic for β
        
        let n = diffs.len() as f64;
        
        // Calculate means
        let mean_diff: f64 = diffs.iter().sum::<f64>() / n;
        let mean_lag: f64 = lagged.iter().sum::<f64>() / n;

        // Calculate regression coefficients
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (&d, &l) in diffs.iter().zip(lagged.iter()) {
            numerator += (l - mean_lag) * (d - mean_diff);
            denominator += (l - mean_lag).powi(2);
        }

        if denominator.abs() < 1e-10 {
            return (0.0, 1.0);
        }

        let beta = numerator / denominator;
        let alpha = mean_diff - beta * mean_lag;

        // Calculate residuals and standard error
        let mut ss_res = 0.0;
        for (&d, &l) in diffs.iter().zip(lagged.iter()) {
            let predicted = alpha + beta * l;
            ss_res += (d - predicted).powi(2);
        }

        let mse = ss_res / (n - 2.0);
        
        // Standard error of beta
        let se_beta = (mse / denominator).sqrt();

        // T-statistic for beta (testing H0: beta = 0, i.e., unit root)
        let t_stat = if se_beta.abs() > 1e-10 {
            beta / se_beta
        } else {
            0.0
        };

        // Approximate p-value using normal approximation
        // For ADF test, we use one-tailed test (left tail)
        let p_value = Self::normal_cdf(t_stat);

        (t_stat, p_value)
    }

    /// Standard normal CDF approximation (Abramowitz and Stegun)
    #[inline]
    fn normal_cdf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        
        if x > 0.0 {
            1.0 - prob
        } else {
            prob
        }
    }
}

/// Multi-pair cointegration monitor
pub struct CointegrationMonitor {
    /// Active pair trackers
    trackers: std::collections::HashMap<(&'static str, &'static str), CointegrationTracker>,
    /// Default window size
    default_window: usize,
}

impl CointegrationMonitor {
    pub fn new(default_window: usize) -> Self {
        Self {
            trackers: std::collections::HashMap::new(),
            default_window,
        }
    }

    /// Add a new pair to monitor
    pub fn add_pair(&mut self, symbol_a: &'static str, symbol_b: &'static str) {
        let key = (symbol_a, symbol_b);
        if !self.trackers.contains_key(&key) {
            let tracker = CointegrationTracker::new(symbol_a, symbol_b, self.default_window);
            self.trackers.insert(key, tracker);
        }
    }

    /// Update prices for a pair
    pub fn update_prices(&mut self, symbol_a: &str, symbol_b: &str, price_a: f64, price_b: f64) {
        // Try both orderings
        let key = (symbol_a, symbol_b);
        if let Some(tracker) = self.trackers.get_mut(&key) {
            tracker.add_price_observation(price_a, price_b);
            return;
        }
        
        let key_reverse = (symbol_b, symbol_a);
        if let Some(tracker) = self.trackers.get_mut(&key_reverse) {
            tracker.add_price_observation(price_b, price_a);
        }
    }

    /// Test all pairs and return results
    pub fn test_all_pairs(&mut self) -> Vec<((&'static str, &'static str), CointegrationResult)> {
        let mut results = Vec::new();
        
        for ((sym_a, sym_b), tracker) in &mut self.trackers {
            if let Some(result) = tracker.test_cointegration() {
                results.push(((sym_a, sym_b), result));
            }
        }
        
        results
    }

    /// Check if a specific pair can be traded
    pub fn can_trade_pair(&self, symbol_a: &str, symbol_b: &str) -> bool {
        let key = (symbol_a, symbol_b);
        if let Some(tracker) = self.trackers.get(&key) {
            return tracker.is_trading_allowed();
        }
        
        let key_reverse = (symbol_b, symbol_a);
        if let Some(tracker) = self.trackers.get(&key_reverse) {
            return tracker.is_trading_allowed();
        }
        
        false
    }

    /// Get all tradeable pairs
    pub fn get_tradeable_pairs(&self) -> Vec<(&'static str, &'static str)> {
        self.trackers.iter()
            .filter(|(_, tracker)| tracker.is_trading_allowed())
            .map(|(&(a, b), _)| (a, b))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hedge_ratio_calculation() {
        // Perfect correlation: A = 2 * B
        let prices_a = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let prices_b = vec![50.0, 51.0, 52.0, 53.0, 54.0];
        
        let ratio = CointegrationTracker::calculate_hedge_ratio(&prices_a, &prices_b);
        assert!((ratio - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_cointegration_tracker() {
        let mut tracker = CointegrationTracker::new("BTC", "ETH", 50);
        
        // Initially trading should be halted (no data)
        assert!(!tracker.is_trading_allowed());
        
        // Add some correlated data
        for i in 0..60 {
            let price_a = 50000.0 + (i as f64 * 100.0);
            let price_b = 3000.0 + (i as f64 * 6.0); // Roughly correlated
            tracker.add_price_observation(price_a, price_b);
        }
        
        // Run test
        let result = tracker.test_cointegration();
        assert!(result.is_some());
        
        println!("Test result: {:?}", result);
    }
}
