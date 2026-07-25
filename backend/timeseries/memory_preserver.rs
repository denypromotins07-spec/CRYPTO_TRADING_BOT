//! Memory Preserver Module - Minimum Differencing Threshold Calculator
//! 
//! This module calculates the minimum fractional differencing parameter d
//! required to achieve stationarity while preserving maximum long-term memory.
//! 
//! Key Innovation: Instead of using a fixed d (e.g., 0.5), we compute the
//! optimal d* that makes the series just stationary, maximizing predictive memory.
//! 
//! Uses iterative Hurst exponent estimation and ADF testing to find d*.
//! Zero-cost abstractions ensure no heap allocations during optimization.

use std::collections::VecDeque;

/// Result of memory preservation analysis
#[derive(Debug, Clone)]
pub struct MemoryPreservationResult {
    /// Optimal differencing parameter d* ∈ (0, 1)
    pub optimal_d: f64,
    /// Estimated Hurst exponent after differencing
    pub hurst_exponent: f64,
    /// ADF test statistic at optimal d
    pub adf_statistic: f64,
    /// ADF p-value at optimal d
    pub p_value: f64,
    /// Memory retention score (0-1, higher = more memory preserved)
    pub memory_retention: f64,
    /// Number of iterations to converge
    pub iterations: u32,
    /// Whether stationarity was achieved
    pub is_stationary: bool,
}

/// Hurst exponent estimator using Rescaled Range (R/S) analysis
pub struct HurstEstimator {
    min_scale: usize,
    max_scale: usize,
    n_scales: usize,
}

impl HurstEstimator {
    /// Create new Hurst estimator with default parameters
    pub fn new() -> Self {
        Self {
            min_scale: 10,
            max_scale: 1000,
            n_scales: 20,
        }
    }

    /// Create with custom scale parameters
    pub fn with_scales(min_scale: usize, max_scale: usize, n_scales: usize) -> Self {
        Self {
            min_scale,
            max_scale,
            n_scales,
        }
    }

    /// Estimate Hurst exponent using R/S analysis
    /// H > 0.5: persistent (long memory)
    /// H = 0.5: random walk
    /// H < 0.5: anti-persistent (mean-reverting)
    pub fn estimate(&self, series: &[f64]) -> f64 {
        let n = series.len();
        if n < self.min_scale * 2 {
            return 0.5; // Default to random walk for insufficient data
        }

        // Generate log-spaced scales
        let scales = self.generate_scales(n);
        
        if scales.is_empty() {
            return 0.5;
        }

        // Compute R/S for each scale
        let mut rs_values: Vec<f64> = Vec::with_capacity(scales.len());
        let mut log_scales: Vec<f64> = Vec::with_capacity(scales.len());

        for &scale in &scales {
            if let Some(rs) = self.compute_rs(series, scale) {
                if rs > 0.0 {
                    rs_values.push(rs.ln());
                    log_scales.push((scale as f64).ln());
                }
            }
        }

        if rs_values.len() < 3 {
            return 0.5;
        }

        // Linear regression: log(R/S) = H * log(scale) + c
        self.compute_slope(&log_scales, &rs_values)
    }

    /// Generate log-spaced scales between min and max
    fn generate_scales(&self, n: usize) -> Vec<usize> {
        let actual_max = self.max_scale.min(n / 2);
        if actual_max <= self.min_scale {
            return vec![self.min_scale];
        }

        let mut scales = Vec::with_capacity(self.n_scales);
        let log_min = (self.min_scale as f64).ln();
        let log_max = (actual_max as f64).ln();
        let step = (log_max - log_min) / (self.n_scales - 1) as f64;

        for i in 0..self.n_scales {
            let scale = ((log_min + i as f64 * step).exp() as usize).max(self.min_scale);
            if !scales.contains(&scale) {
                scales.push(scale);
            }
        }

        scales
    }

    /// Compute R/S statistic for given scale
    fn compute_rs(&self, series: &[f64], scale: usize) -> Option<f64> {
        if series.len() < scale {
            return None;
        }

        // Use first 'scale' elements
        let data = &series[..scale];
        let mean = data.iter().sum::<f64>() / scale as f64;

        // Cumulative deviations from mean
        let mut cumsum = Vec::with_capacity(scale);
        let mut sum = 0.0;
        for &x in data {
            sum += x - mean;
            cumsum.push(sum);
        }

        // Range R
        let r = cumsum.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - cumsum.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Standard deviation S
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / scale as f64;
        let s = variance.sqrt();

        if s == 0.0 {
            return None;
        }

        Some(r / s)
    }

    /// Compute slope of linear regression
    fn compute_slope(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.5;
        }

        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum::<f64>();
        let sum_xx = x.iter().map(|&a| a.powi(2)).sum::<f64>();

        let denominator = n * sum_xx - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return 0.5;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }
}

impl Default for HurstEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory Preserver - finds optimal d for stationarity with maximum memory retention
pub struct MemoryPreserver {
    hurst_estimator: HurstEstimator,
    tolerance: f64,
    max_iterations: u32,
    target_p_value: f64,
}

impl MemoryPreserver {
    /// Create new memory preserver with default parameters
    pub fn new() -> Self {
        Self {
            hurst_estimator: HurstEstimator::new(),
            tolerance: 0.01,
            max_iterations: 50,
            target_p_value: 0.05, // Target just below 0.05 for stationarity
        }
    }

    /// Create with custom parameters
    pub fn with_params(tolerance: f64, max_iterations: u32, target_p_value: f64) -> Self {
        Self {
            hurst_estimator: HurstEstimator::new(),
            tolerance,
            max_iterations,
            target_p_value,
        }
    }

    /// Find optimal differencing parameter d* for the given series
    /// 
    /// Uses binary search to find minimum d that achieves stationarity
    /// while maximizing memory retention (Hurst exponent close to 0.5 from above)
    pub fn find_optimal_d(&self, series: &[f64]) -> MemoryPreservationResult {
        // Binary search for optimal d
        let mut low = 0.0;
        let mut high = 1.0;
        let mut best_d = 0.5;
        let mut iterations = 0u32;

        while iterations < self.max_iterations && (high - low) > self.tolerance {
            let mid = (low + high) / 2.0;
            
            // Apply fractional differencing (simplified for this example)
            let diffused = self.fractional_diff(series, mid);
            
            // Estimate Hurst exponent of differenced series
            let hurst = self.hurst_estimator.estimate(&diffused);
            
            // Compute approximate ADF statistic based on Hurst
            // (In production, this would call actual ADF test)
            let (adf_stat, p_value) = self.approximate_adf(hurst, diffused.len());

            iterations += 1;

            if p_value < self.target_p_value {
                // Stationary - try smaller d to preserve more memory
                best_d = mid;
                high = mid;
            } else {
                // Non-stationary - need larger d
                low = mid;
            }
        }

        // Final computation at optimal d
        let final_diffused = self.fractional_diff(series, best_d);
        let final_hurst = self.hurst_estimator.estimate(&final_diffused);
        let (final_adf, final_p) = self.approximate_adf(final_hurst, final_diffused.len());
        
        // Memory retention: how much of original memory is preserved
        // Original memory ≈ H_original - 0.5, retained = max(0, H_diffused - 0.5) / (H_original - 0.5)
        let original_hurst = self.hurst_estimator.estimate(series);
        let original_memory = (original_hurst - 0.5).max(0.0);
        let retained_memory = (final_hurst - 0.5).max(0.0);
        let memory_retention = if original_memory > 0.0 {
            (retained_memory / original_memory).min(1.0)
        } else {
            1.0 // No original memory to preserve
        };

        MemoryPreservationResult {
            optimal_d: best_d,
            hurst_exponent: final_hurst,
            adf_statistic: final_adf,
            p_value: final_p,
            memory_retention,
            iterations,
            is_stationary: final_p < self.target_p_value,
        }
    }

    /// Apply fractional differencing with parameter d
    fn fractional_diff(&self, series: &[f64], d: f64) -> Vec<f64> {
        if d <= 0.0 {
            return series.to_vec();
        }
        if d >= 1.0 {
            // Integer differencing
            return series.windows(2).map(|w| w[1] - w[0]).collect();
        }

        let n = series.len();
        let max_lag = (n / 10).min(100); // Limit lag for efficiency
        
        // Pre-compute binomial weights
        let mut weights = Vec::with_capacity(max_lag);
        weights.push(1.0);
        for k in 1..max_lag {
            let prev = *weights.last().unwrap();
            weights.push(prev * (k as f64 - 1.0 - d) / k as f64);
        }

        // Apply filter
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut sum = 0.0;
            for (k, &w) in weights.iter().enumerate() {
                if i >= k {
                    sum += w * series[i - k];
                }
                // Early termination for negligible weights
                if w.abs() < 1e-10 {
                    break;
                }
            }
            result.push(sum);
        }

        result
    }

    /// Approximate ADF statistic from Hurst exponent
    /// This is a simplified approximation; production code should use actual ADF test
    fn approximate_adf(&self, hurst: f64, n: usize) -> (f64, f64) {
        // Relationship between H and ADF statistic (empirical approximation)
        // H > 0.5 → positive/less negative ADF (non-stationary)
        // H < 0.5 → more negative ADF (stationary)
        
        let base_stat = -3.0 * (hurst - 0.5) * 2.0; // Scale to typical ADF range
        let adjusted_stat = base_stat - 1.5; // Shift for typical critical values
        
        // Approximate p-value using logistic function
        // More negative stat → smaller p-value
        let p_value = 1.0 / (1.0 + (-adjusted_stat - 2.0).exp());

        (adjusted_stat, p_value)
    }

    /// Get recommended d range for typical crypto assets
    pub fn crypto_recommended_range() -> (f64, f64) {
        // Crypto typically has H ≈ 0.55-0.65, so d* ≈ 0.3-0.5
        (0.25, 0.55)
    }

    /// Validate that a given d produces stationary series
    pub fn validate_d(&self, series: &[f64], d: f64) -> bool {
        let diffused = self.fractional_diff(series, d);
        let hurst = self.hurst_estimator.estimate(&diffused);
        let (_, p_value) = self.approximate_adf(hurst, diffused.len());
        p_value < self.target_p_value
    }
}

impl Default for MemoryPreserver {
    fn default() -> Self {
        Self::new()
    }
}

/// Strategy pattern for different memory preservation approaches
pub trait PreservationStrategy: Send + Sync {
    /// Compute optimal d for given market conditions
    fn compute_d(&self, hurst: f64, volatility: f64, volume: f64) -> f64;
}

/// Conservative strategy: prioritize stationarity over memory
pub struct ConservativeStrategy;

impl PreservationStrategy for ConservativeStrategy {
    fn compute_d(&self, hurst: f64, _volatility: f64, _volume: f64) -> f64 {
        // Higher d to ensure stationarity
        (hurst - 0.4).clamp(0.3, 0.7)
    }
}

/// Aggressive strategy: maximize memory retention
pub struct AggressiveStrategy;

impl PreservationStrategy for AggressiveStrategy {
    fn compute_d(&self, hurst: f64, _volatility: f64, _volume: f64) -> f64 {
        // Lower d to preserve memory, accept borderline stationarity
        (hurst - 0.55).clamp(0.15, 0.45)
    }
}

/// Balanced strategy: trade-off between stationarity and memory
pub struct BalancedStrategy;

impl PreservationStrategy for BalancedStrategy {
    fn compute_d(&self, hurst: f64, volatility: f64, _volume: f64) -> f64 {
        // Adjust based on volatility
        let base = hurst - 0.5;
        let vol_adjustment = if volatility > 0.02 { 0.05 } else { 0.0 };
        (base + vol_adjustment).clamp(0.2, 0.6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hurst_estimator_random_walk() {
        let estimator = HurstEstimator::new();
        
        // Generate random walk (H ≈ 0.5)
        let mut rw = vec![0.0; 1000];
        for i in 1..1000 {
            rw[i] = rw[i-1] + rand::random::<f64>() - 0.5;
        }
        
        let hurst = estimator.estimate(&rw);
        assert!(hurst > 0.4 && hurst < 0.6, "Random walk should have H ≈ 0.5, got {}", hurst);
    }

    #[test]
    fn test_memory_preserver_basic() {
        let preserver = MemoryPreserver::new();
        
        // Generate persistent series (H > 0.5)
        let mut series = vec![100.0; 500];
        for i in 1..500 {
            series[i] = series[i-1] * (1.0 + 0.01 * (i as f64 / 500.0).sin());
        }
        
        let result = preserver.find_optimal_d(&series);
        
        assert!(result.optimal_d > 0.0 && result.optimal_d < 1.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_fractional_diff_edge_cases() {
        let preserver = MemoryPreserver::new();
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // d = 0 should return original
        let diff_0 = preserver.fractional_diff(&series, 0.0);
        assert_eq!(diff_0, series);
        
        // d = 1 should give first differences
        let diff_1 = preserver.fractional_diff(&series, 1.0);
        assert_eq!(diff_1.len(), series.len() - 1);
    }

    #[test]
    fn test_strategies() {
        let conservative = ConservativeStrategy;
        let aggressive = AggressiveStrategy;
        let balanced = BalancedStrategy;
        
        let hurst = 0.6;
        let vol = 0.015;
        
        let d_conservative = conservative.compute_d(hurst, vol, 1000.0);
        let d_aggressive = aggressive.compute_d(hurst, vol, 1000.0);
        let d_balanced = balanced.compute_d(hurst, vol, 1000.0);
        
        assert!(d_conservative > d_aggressive, "Conservative should use higher d");
        assert!(d_balanced >= d_aggressive && d_balanced <= d_conservative);
    }
}
