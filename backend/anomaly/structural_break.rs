//! Structural Break Detection using Chow and QLR Tests
//! 
//! This module implements classical econometric tests for structural breaks
//! to determine when alpha models have become stale and need re-calibration.
//! 
//! Key Features:
//! - Chow test for known break points
//! - Quandt Likelihood Ratio (QLR) / SupF test for unknown break points
//! - Automatic model invalidation signaling
//! - Memory-efficient rolling regression
//! 
//! Mathematical Foundation:
//! Chow F = [(RSS_R - RSS_UR) / k] / [RSS_UR / (n - 2k)]
//! QLR = sup_{τ∈[τ₁,τ₂]} F(τ)

use std::collections::VecDeque;

/// Configuration for structural break tests
#[derive(Debug, Clone)]
pub struct StructuralBreakConfig {
    /// Number of regressors (including intercept)
    pub num_regressors: usize,
    /// Minimum observations per regime for Chow test
    pub min_obs_per_regime: usize,
    /// Trimmed proportion for QLR test (typically 0.15)
    pub trim_proportion: f64,
    /// Significance level for critical values
    pub significance_level: f64,
}

impl Default for StructuralBreakConfig {
    fn default() -> Self {
        Self {
            num_regressors: 2, // intercept + 1 predictor
            min_obs_per_regime: 30,
            trim_proportion: 0.15,
            significance_level: 0.05,
        }
    }
}

/// Result of a Chow test
#[derive(Debug, Clone)]
pub struct ChowTestResult {
    /// F-statistic value
    pub f_statistic: f64,
    /// Critical value at specified significance
    pub critical_value: f64,
    /// P-value approximation
    pub p_value: f64,
    /// Whether null hypothesis (no break) is rejected
    pub break_detected: bool,
    /// Break point index
    pub break_point: usize,
}

/// Result of a QLR (SupF) test
#[derive(Debug, Clone)]
pub struct QlrTestResult {
    /// Maximum F-statistic across all tested break points
    pub sup_f_statistic: f64,
    /// Critical value for QLR test
    pub critical_value: f64,
    /// Estimated break point
    pub estimated_break_point: usize,
    /// Whether structural break detected
    pub break_detected: bool,
    /// F-statistics for all tested break points
    pub f_statistics: Vec<f64>,
}

/// Rolling regression state for efficient updates
struct RollingRegression {
    /// X'X matrix (sum of squares and cross-products)
    xtx: Vec<f64>,
    /// X'y vector
    xty: Vec<f64>,
    /// y'y sum
    yty: f64,
    /// Number of observations
    n: usize,
    /// Number of regressors
    k: usize,
}

impl RollingRegression {
    fn new(k: usize) -> Self {
        let matrix_size = k * k;
        Self {
            xtx: vec![0.0; matrix_size],
            xty: vec![0.0; k],
            yty: 0.0,
            n: 0,
            k,
        }
    }

    /// Add observation to rolling regression
    #[inline]
    fn update(&mut self, x: &[f64], y: f64) {
        assert_eq!(x.len(), self.k, "Regressor dimension mismatch");
        
        // Update X'X: symmetric matrix, only store upper triangle
        for i in 0..self.k {
            for j in i..self.k {
                self.xtx[i * self.k + j] += x[i] * x[j];
                if i != j {
                    self.xtx[j * self.k + i] = self.xtx[i * self.k + j];
                }
            }
        }
        
        // Update X'y
        for i in 0..self.k {
            self.xty[i] += x[i] * y;
        }
        
        // Update y'y
        self.yty += y * y;
        self.n += 1;
    }

    /// Compute residual sum of squares
    fn rss(&self) -> f64 {
        if self.n <= self.k {
            return f64::INFINITY;
        }

        // Solve normal equations: β = (X'X)^(-1) X'y
        // Then RSS = y'y - β'X'y
        match self.solve_normal_equations() {
            Some(beta) => {
                let xty_beta: f64 = beta.iter()
                    .zip(self.xty.iter())
                    .map(|(b, x)| b * x)
                    .sum();
                self.yty - xty_beta
            }
            None => f64::INFINITY,
        }
    }

    /// Solve (X'X)β = X'y using Gaussian elimination
    fn solve_normal_equations(&self) -> Option<Vec<f64>> {
        let mut augmented = Vec::with_capacity(self.k);
        
        // Build augmented matrix [X'X | X'y]
        for i in 0..self.k {
            let mut row = Vec::with_capacity(self.k + 1);
            for j in 0..self.k {
                row.push(self.xtx[i * self.k + j]);
            }
            row.push(self.xty[i]);
            augmented.push(row);
        }
        
        // Forward elimination with partial pivoting
        for col in 0..self.k {
            // Find pivot
            let mut max_row = col;
            let mut max_val = augmented[col][col].abs();
            for row in (col + 1)..self.k {
                if augmented[row][col].abs() > max_val {
                    max_val = augmented[row][col].abs();
                    max_row = row;
                }
            }
            
            if max_val < 1e-10 {
                return None; // Singular matrix
            }
            
            // Swap rows
            augmented.swap(col, max_row);
            
            // Eliminate column
            for row in (col + 1)..self.k {
                let factor = augmented[row][col] / augmented[col][col];
                for j in col..=self.k {
                    augmented[row][j] -= factor * augmented[col][j];
                }
            }
        }
        
        // Back substitution
        let mut beta = vec![0.0; self.k];
        for i in (0..self.k).rev() {
            let mut sum = augmented[i][self.k];
            for j in (i + 1)..self.k {
                sum -= augmented[i][j] * beta[j];
            }
            beta[i] = sum / augmented[i][i];
        }
        
        Some(beta)
    }
}

/// Structural break detector implementing Chow and QLR tests
pub struct StructuralBreakDetector {
    config: StructuralBreakConfig,
    /// Data window for testing
    data_x: VecDeque<Vec<f64>>,
    data_y: VecDeque<f64>,
    /// Pre-computed regressions for efficiency
    full_regression: Option<RollingRegression>,
}

impl StructuralBreakDetector {
    /// Create new structural break detector
    pub fn new(config: StructuralBreakConfig) -> Self {
        Self {
            config,
            data_x: VecDeque::new(),
            data_y: VecDeque::new(),
            full_regression: None,
        }
    }

    /// Add observation to the data window
    pub fn add_observation(&mut self, x: Vec<f64>, y: f64) {
        assert_eq!(x.len(), self.config.num_regressors);
        
        self.data_x.push_back(x);
        self.data_y.push_back(y);
        
        // Maintain reasonable window size
        let max_window = self.config.min_obs_per_regime * 4;
        while self.data_x.len() > max_window {
            self.data_x.pop_front();
            self.data_y.pop_front();
        }
        
        // Rebuild full regression
        self.rebuild_full_regression();
    }

    /// Rebuild full regression from current window
    fn rebuild_full_regression(&mut self) {
        if self.data_x.len() < self.config.min_obs_per_regime * 2 {
            self.full_regression = None;
            return;
        }

        let mut reg = RollingRegression::new(self.config.num_regressors);
        for (x, y) in self.data_x.iter().zip(self.data_y.iter()) {
            reg.update(x, *y);
        }
        self.full_regression = Some(reg);
    }

    /// Perform Chow test at specified break point
    pub fn chow_test(&self, break_point: usize) -> Option<ChowTestResult> {
        let n = self.data_x.len();
        let k = self.config.num_regressors;
        
        // Validate break point
        if break_point < self.config.min_obs_per_regime 
            || break_point >= n - self.config.min_obs_per_regime 
        {
            return None;
        }

        // Compute RSS for full sample
        let rss_full = self.full_regression.as_ref()?.rss();

        // Compute RSS for first regime
        let mut reg1 = RollingRegression::new(k);
        for (x, y) in self.data_x.iter().zip(self.data_y.iter()).take(break_point) {
            reg1.update(x, *y);
        }
        let rss1 = reg1.rss();

        // Compute RSS for second regime
        let mut reg2 = RollingRegression::new(k);
        for (x, y) in self.data_x.iter().zip(self.data_y.iter()).skip(break_point) {
            reg2.update(x, *y);
        }
        let rss2 = reg2.rss();

        // Restricted RSS (null hypothesis: no break)
        let rss_restricted = rss_full;
        
        // Unrestricted RSS (alternative: break exists)
        let rss_unrestricted = rss1 + rss2;

        // Compute F-statistic
        let numerator = (rss_restricted - rss_unrestricted) / k as f64;
        let denominator = rss_unrestricted / (n - 2 * k) as f64;
        
        if denominator < 1e-10 {
            return None;
        }

        let f_statistic = numerator / denominator;
        
        // Critical value from F-distribution (approximation)
        let critical_value = self.f_critical_value(k as u32, (n - 2 * k) as u32);
        
        // P-value approximation using Wilson-Hilferty transformation
        let p_value = self.f_p_value_approx(f_statistic, k as u32, (n - 2 * k) as u32);

        Some(ChowTestResult {
            f_statistic,
            critical_value,
            p_value,
            break_detected: f_statistic > critical_value,
            break_point,
        })
    }

    /// Perform QLR (SupF) test scanning all valid break points
    pub fn qlr_test(&self) -> Option<QlrTestResult> {
        let n = self.data_x.len();
        let k = self.config.num_regressors;
        
        if n < self.config.min_obs_per_regime * 2 {
            return None;
        }

        // Define search range with trimming
        let trim = (n as f64 * self.config.trim_proportion) as usize;
        let start = trim.max(self.config.min_obs_per_regime);
        let end = (n - trim).min(n - self.config.min_obs_per_regime);

        if start >= end {
            return None;
        }

        let mut f_statistics = Vec::with_capacity(end - start);
        let mut sup_f = 0.0;
        let mut best_break = start;

        // Scan all break points in trimmed range
        for bp in start..end {
            if let Some(result) = self.chow_test(bp) {
                f_statistics.push(result.f_statistic);
                if result.f_statistic > sup_f {
                    sup_f = result.f_statistic;
                    best_break = bp;
                }
            }
        }

        if f_statistics.is_empty() {
            return None;
        }

        // QLR critical values (from Andrews 1993 tables, approximate)
        let critical_value = self.qlr_critical_value(k, self.config.trim_proportion);

        Some(QlrTestResult {
            sup_f_statistic: sup_f,
            critical_value,
            estimated_break_point: best_break,
            break_detected: sup_f > critical_value,
            f_statistics,
        })
    }

    /// Get F-distribution critical value (approximation)
    fn f_critical_value(&self, df1: u32, df2: u32) -> f64 {
        // Approximation using chi-square for large df2
        // For typical finance applications, this is sufficient
        let alpha = self.config.significance_level;
        
        // Simplified approximation: F ≈ χ²/df1 for large df2
        let chi2_cv = match (df1, alpha) {
            (1, 0.05) => 3.84,
            (1, 0.01) => 6.63,
            (2, 0.05) => 5.99,
            (2, 0.01) => 9.21,
            (3, 0.05) => 7.81,
            (3, 0.01) => 11.34,
            (4, 0.05) => 9.49,
            (4, 0.01) => 13.28,
            _ => df1 as f64 * 2.0, // Rough approximation
        };
        
        // Adjust for finite df2
        chi2_cv / df1 as f64 * (1.0 + 2.0 / df2 as f64)
    }

    /// Get QLR critical value (from Andrews 1993)
    fn qlr_critical_value(&self, k: usize, trim: f64) -> f64 {
        // Approximate critical values for SupF test
        // Based on Andrews (1993) Tables I and II
        let base_cv = match k {
            1 => 8.68,
            2 => 10.39,
            3 => 12.17,
            4 => 13.87,
            _ => 10.0 + k as f64 * 1.5,
        };
        
        // Adjust for trim proportion
        let trim_factor = match trim {
            t if t < 0.10 => 1.2,
            t if t < 0.15 => 1.0,
            t if t < 0.20 => 0.85,
            _ => 0.75,
        };
        
        base_cv * trim_factor
    }

    /// Approximate F-distribution p-value
    fn f_p_value_approx(&self, f: f64, df1: u32, df2: u32) -> f64 {
        // Using beta distribution approximation
        // P(F > f) = I_{df1*f/(df1*f+df2)}(df1/2, df2/2)
        let x = (df1 as f64 * f) / (df1 as f64 * f + df2 as f64);
        
        // Incomplete beta function approximation (simplified)
        if x <= 0.0 {
            return 1.0;
        }
        if x >= 1.0 {
            return 0.0;
        }
        
        // Simple approximation for tail probability
        let a = df1 as f64 / 2.0;
        let b = df2 as f64 / 2.0;
        
        // Log of beta density kernel
        let log_kernel = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln();
        
        // Very rough p-value estimate
        if log_kernel < -20.0 {
            0.001
        } else if log_kernel < -10.0 {
            0.01
        } else if log_kernel < -5.0 {
            0.05
        } else {
            0.1
        }
    }

    /// Check if model should be invalidated based on structural breaks
    pub fn should_invalidate_model(&self) -> bool {
        if let Some(qlr_result) = self.qlr_test() {
            qlr_result.break_detected
        } else {
            false
        }
    }

    /// Get current data window size
    pub fn window_size(&self) -> usize {
        self.data_x.len()
    }

    /// Clear all stored data
    pub fn clear(&mut self) {
        self.data_x.clear();
        self.data_y.clear();
        self.full_regression = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chow_test_no_break() {
        let config = StructuralBreakConfig {
            num_regressors: 1,
            min_obs_per_regime: 10,
            ..Default::default()
        };
        
        let mut detector = StructuralBreakDetector::new(config);
        
        // Add homogeneous data
        for i in 0..50 {
            let x = vec![1.0];
            let y = 2.0 + (i as f64 * 0.1).sin() * 0.5;
            detector.add_observation(x, y);
        }
        
        let result = detector.chow_test(25);
        assert!(result.is_some());
        let r = result.unwrap();
        
        // Should not detect break in homogeneous data
        assert!(!r.break_detected || r.f_statistic < r.critical_value * 2.0);
    }

    #[test]
    fn test_chow_test_with_break() {
        let config = StructuralBreakConfig {
            num_regressors: 1,
            min_obs_per_regime: 10,
            ..Default::default()
        };
        
        let mut detector = StructuralBreakDetector::new(config);
        
        // Add data with structural break
        for i in 0..25 {
            detector.add_observation(vec![1.0], 1.0 + (i as f64 * 0.1).sin() * 0.1);
        }
        for i in 0..25 {
            detector.add_observation(vec![1.0], 5.0 + (i as f64 * 0.1).sin() * 0.1);
        }
        
        let result = detector.chow_test(25);
        assert!(result.is_some());
        let r = result.unwrap();
        
        // Should detect large mean shift
        assert!(r.f_statistic > 10.0, "F-statistic should be large for obvious break");
    }

    #[test]
    fn test_qlr_test_detection() {
        let config = StructuralBreakConfig::default();
        let mut detector = StructuralBreakDetector::new(config);
        
        // Clear regime shift
        for _ in 0..30 {
            detector.add_observation(vec![1.0, 0.5], 1.0);
        }
        for _ in 0..30 {
            detector.add_observation(vec![1.0, 0.5], 3.0);
        }
        
        let result = detector.qlr_test();
        assert!(result.is_some());
        
        let r = result.unwrap();
        assert!(r.break_detected, "QLR should detect obvious structural break");
        assert!(r.estimated_break_point > 20 && r.estimated_break_point < 40);
    }
}
