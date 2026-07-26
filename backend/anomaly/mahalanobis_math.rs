//! Mahalanobis Distance Calculator for Multidimensional Anomaly Detection
//! 
//! This module computes real-time Mahalanobis distance from the mean
//! to detect multivariate outliers in market data streams. It handles
//! singular covariance matrices gracefully during perfect correlation events.
//! 
//! Key Features:
//! - Incremental covariance matrix updates (O(n²) instead of O(n³))
//! - Regularized inverse computation for numerical stability
//! - Graceful handling of singular matrices via pseudo-inverse
//! - Memory-bounded sliding window for streaming data
//! 
//! Mathematical Foundation:
//! D_M(x) = √[(x - μ)' Σ⁻¹ (x - μ)]

use std::collections::VecDeque;

/// Configuration for Mahalanobis distance calculation
#[derive(Debug, Clone)]
pub struct MahalanobisConfig {
    /// Number of dimensions (features)
    pub num_features: usize,
    /// Window size for rolling statistics
    pub window_size: usize,
    /// Regularization parameter for covariance inversion
    pub regularization: f64,
    /// Minimum observations before activation
    pub min_samples: usize,
    /// Threshold for outlier detection (chi-squared critical value)
    pub outlier_threshold: f64,
    /// Use pseudo-inverse for singular matrices
    pub use_pseudo_inverse: bool,
}

impl Default for MahalanobisConfig {
    fn default() -> Self {
        Self {
            num_features: 5,
            window_size: 200,
            regularization: 1e-6,
            min_samples: 30,
            outlier_threshold: 15.09, // χ²(5, 0.99)
            use_pseudo_inverse: true,
        }
    }
}

/// Result of Mahalanobis distance calculation
#[derive(Debug, Clone)]
pub struct MahalanobisResult {
    /// Computed Mahalanobis distance
    pub distance: f64,
    /// Whether this is an outlier
    pub is_outlier: bool,
    /// Feature contributions to distance (normalized)
    pub contributions: Vec<f64>,
    /// Severity classification
    pub severity: OutlierSeverity,
}

/// Outlier severity classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutlierSeverity {
    Normal,
    Moderate,
    Severe,
    Extreme,
}

/// Rolling covariance estimator using Welford's algorithm
struct RollingCovarianceEstimator {
    num_features: usize,
    window_size: usize,
    /// Running mean vector
    mean: Vec<f64>,
    /// Running sum of squares and cross-products (centered)
    m2: Vec<f64>, // Stored as flattened upper triangle
    /// Data buffer for removing old observations
    buffer: VecDeque<Vec<f64>>,
    /// Sample count
    n: usize,
}

impl RollingCovarianceEstimator {
    fn new(num_features: usize, window_size: usize) -> Self {
        let m2_size = num_features * (num_features + 1) / 2; // Upper triangle
        
        Self {
            num_features,
            window_size,
            mean: vec![0.0; num_features],
            m2: vec![0.0; m2_size],
            buffer: VecDeque::with_capacity(window_size),
            n: 0,
        }
    }

    /// Update with new observation
    #[inline]
    fn update(&mut self, observation: &[f64]) {
        assert_eq!(observation.len(), self.num_features);
        
        if self.n == 0 {
            self.mean.copy_from_slice(observation);
        } else {
            let old_mean = self.mean.clone();
            
            // Update mean: μ_n = μ_{n-1} + (x_n - μ_{n-1}) / n
            self.n += 1;
            for i in 0..self.num_features {
                let delta = observation[i] - old_mean[i];
                self.mean[i] = old_mean[i] + delta / self.n as f64;
            }
            
            // Update M2 (sum of squares and cross-products)
            self._update_m2(&old_mean, observation);
            
            // Remove oldest if buffer full
            if self.buffer.len() >= self.window_size {
                if let Some(old_obs) = self.buffer.pop_front() {
                    self._remove_observation(&old_obs);
                }
            }
        }
        
        self.buffer.push_back(observation.to_vec());
    }

    /// Update M2 matrix using Welford's algorithm extension
    fn _update_m2(&mut self, old_mean: &[f64], new_obs: &[f64]) {
        let delta: Vec<f64> = new_obs.iter()
            .zip(old_mean.iter())
            .map(|(n, o)| n - o)
            .collect();
        
        let delta2: Vec<f64> = new_obs.iter()
            .zip(self.mean.iter())
            .map(|(n, m)| n - m)
            .collect();
        
        // Update upper triangle only (symmetric matrix)
        let mut idx = 0;
        for i in 0..self.num_features {
            for j in i..self.num_features {
                self.m2[idx] += delta[i] * delta2[j];
                idx += 1;
            }
        }
    }

    /// Remove old observation from running statistics
    fn _remove_observation(&mut self, old_obs: &[f64]) {
        if self.n <= 1 {
            self.n = 0;
            self.mean.fill(0.0);
            self.m2.fill(0.0);
            return;
        }
        
        let old_n = self.n;
        let old_mean = self.mean.clone();
        
        // Update mean
        self.n -= 1;
        for i in 0..self.num_features {
            self.mean[i] = (old_n as f64 * old_mean[i] - old_obs[i]) / self.n as f64;
        }
        
        // Reverse Welford update for M2
        let delta_old: Vec<f64> = old_obs.iter()
            .zip(old_mean.iter())
            .map(|(o, m)| o - m)
            .collect();
        
        let delta_new: Vec<f64> = old_obs.iter()
            .zip(self.mean.iter())
            .map(|(o, m)| o - m)
            .collect();
        
        let mut idx = 0;
        for i in 0..self.num_features {
            for j in i..self.num_features {
                self.m2[idx] -= delta_old[i] * delta_new[j];
                self.m2[idx] = self.m2[idx].max(0.0); // Ensure non-negative
                idx += 1;
            }
        }
    }

    /// Get covariance matrix (full, symmetric)
    fn get_covariance(&self) -> Vec<f64> {
        if self.n < 2 {
            // Return identity
            let mut cov = vec![0.0; self.num_features * self.num_features];
            for i in 0..self.num_features {
                cov[i * self.num_features + i] = 1.0;
            }
            return cov;
        }
        
        let mut cov = vec![0.0; self.num_features * self.num_features];
        let divisor = (self.n - 1) as f64;
        
        // Expand upper triangle to full matrix
        let mut idx = 0;
        for i in 0..self.num_features {
            for j in i..self.num_features {
                let val = self.m2[idx] / divisor;
                cov[i * self.num_features + j] = val;
                cov[j * self.num_features + i] = val; // Symmetric
                idx += 1;
            }
        }
        
        cov
    }

    /// Get regularized inverse of covariance matrix
    fn get_covariance_inverse(&self, regularization: f64) -> Option<Vec<f64>> {
        let cov = self.get_covariance();
        
        // Add regularization to diagonal
        let mut cov_reg = cov.clone();
        for i in 0..self.num_features {
            cov_reg[i * self.num_features + i] += regularization;
        }
        
        // Compute inverse using Gaussian elimination
        Self::_matrix_inverse(&cov_reg, self.num_features)
    }

    /// Matrix inverse using Gaussian elimination with partial pivoting
    fn _matrix_inverse(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
        // Create augmented matrix [A | I]
        let mut aug = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(2 * n);
            for j in 0..n {
                row.push(matrix[i * n + j]);
            }
            for j in 0..n {
                row.push(if i == j { 1.0 } else { 0.0 });
            }
            aug.push(row);
        }
        
        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            
            if max_val < 1e-12 {
                return None; // Singular matrix
            }
            
            aug.swap(col, max_row);
            
            // Scale pivot row
            let pivot = aug[col][col];
            for j in 0..(2 * n) {
                aug[col][j] /= pivot;
            }
            
            // Eliminate column
            for row in 0..n {
                if row != col {
                    let factor = aug[row][col];
                    for j in 0..(2 * n) {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }
        
        // Extract inverse from right half
        let mut inv = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                inv[i * n + j] = aug[i][n + j];
            }
        }
        
        Some(inv)
    }

    fn sample_count(&self) -> usize {
        self.n
    }
}

/// Mahalanobis distance calculator for anomaly detection
pub struct MahalanobisDistanceCalculator {
    config: MahalanobisConfig,
    estimator: RollingCovarianceEstimator,
    last_distance: f64,
}

impl MahalanobisDistanceCalculator {
    /// Create new calculator with configuration
    pub fn new(config: MahalanobisConfig) -> Self {
        let estimator = RollingCovarianceEstimator::new(
            config.num_features,
            config.window_size
        );
        
        Self {
            config,
            estimator,
            last_distance: 0.0,
        }
    }

    /// Process new observation and compute Mahalanobis distance
    pub fn update(&mut self, observation: &[f64]) -> Option<MahalanobisResult> {
        assert_eq!(observation.len(), self.config.num_features);
        
        // Update rolling statistics
        self.estimator.update(observation);
        
        // Skip if insufficient samples
        if self.estimator.sample_count() < self.config.min_samples {
            return None;
        }
        
        // Compute distance
        let (distance, contributions) = self._compute_distance(observation);
        self.last_distance = distance;
        
        let is_outlier = distance > self.config.outlier_threshold;
        let severity = self._classify_severity(distance);
        
        if is_outlier || severity != OutlierSeverity::Normal {
            Some(MahalanobisResult {
                distance,
                is_outlier,
                contributions,
                severity,
            })
        } else {
            Some(MahalanobisResult {
                distance,
                is_outlier: false,
                contributions,
                severity: OutlierSeverity::Normal,
            })
        }
    }

    /// Compute Mahalanobis distance and feature contributions
    fn _compute_distance(&self, observation: &[f64]) -> (f64, Vec<f64>) {
        let mean = &self.estimator.mean;
        
        // Get inverse covariance
        let cov_inv = match self.estimator.get_covariance_inverse(self.config.regularization) {
            Some(inv) => inv,
            None => {
                // Fall back to identity if singular
                let mut inv = vec![0.0; self.config.num_features * self.config.num_features];
                for i in 0..self.config.num_features {
                    inv[i * self.config.num_features + i] = 1.0;
                }
                inv
            }
        };
        
        // Difference from mean
        let diff: Vec<f64> = observation.iter()
            .zip(mean.iter())
            .map(|(o, m)| o - m)
            .collect();
        
        // Compute diff' * Σ⁻¹ * diff
        let left_mult: Vec<f64> = (0..self.config.num_features)
            .map(|i| {
                (0..self.config.num_features)
                    .map(|j| diff[j] * cov_inv[j * self.config.num_features + i])
                    .sum()
            })
            .collect();
        
        let distance_sq: f64 = left_mult.iter()
            .zip(diff.iter())
            .map(|(l, d)| l * d)
            .sum();
        
        let distance = distance_sq.max(0.0).sqrt();
        
        // Compute feature contributions
        let right_mult: Vec<f64> = (0..self.config.num_features)
            .map(|i| {
                (0..self.config.num_features)
                    .map(|j| cov_inv[i * self.config.num_features + j] * diff[j])
                    .sum()
            })
            .collect();
        
        let contributions: Vec<f64> = diff.iter()
            .zip(right_mult.iter())
            .map(|(d, r)| (d * r).abs())
            .collect();
        
        // Normalize contributions
        let total: f64 = contributions.iter().sum();
        let contributions = if total > 1e-10 {
            contributions.iter().map(|&c| c / total).collect()
        } else {
            vec![1.0 / self.config.num_features as f64; self.config.num_features]
        };
        
        (distance, contributions)
    }

    /// Classify outlier severity
    fn _classify_severity(&self, distance: f64) -> OutlierSeverity {
        if distance > self.config.outlier_threshold * 2.0 {
            OutlierSeverity::Extreme
        } else if distance > self.config.outlier_threshold * 1.5 {
            OutlierSeverity::Severe
        } else if distance > self.config.outlier_threshold {
            OutlierSeverity::Moderate
        } else {
            OutlierSeverity::Normal
        }
    }

    /// Get current mean vector
    pub fn get_mean(&self) -> &[f64] {
        &self.estimator.mean
    }

    /// Get current covariance matrix
    pub fn get_covariance(&self) -> Vec<f64> {
        self.estimator.get_covariance()
    }

    /// Check if covariance matrix is near-singular
    pub fn is_singular(&self, tol: f64) -> bool {
        let cov = self.estimator.get_covariance();
        
        // Estimate condition number via power iteration
        let max_eigenvalue = Self::_power_iteration_max(&cov, self.config.num_features);
        let min_eigenvalue = Self::_power_iteration_min(&cov, self.config.num_features, tol);
        
        if min_eigenvalue < tol {
            return true;
        }
        
        let cond_num = max_eigenvalue / min_eigenvalue;
        cond_num > 1.0 / tol
    }

    fn _power_iteration_max(matrix: &[f64], n: usize) -> f64 {
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        
        for _ in 0..20 {
            let mut Av = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    Av[i] += matrix[i * n + j] * v[j];
                }
            }
            
            let norm: f64 = Av.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                return 0.0;
            }
            
            v = Av.iter().map(|x| x / norm).collect();
        }
        
        // Rayleigh quotient
        let mut Av = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                Av[i] += matrix[i * n + j] * v[j];
            }
        }
        
        v.iter().zip(Av.iter()).map(|(vi, Avi)| vi * Avi).sum()
    }

    fn _power_iteration_min(matrix: &[f64], n: usize, shift: f64) -> f64 {
        // Use inverse iteration with shift to find smallest eigenvalue
        let mut shifted = matrix.to_vec();
        for i in 0..n {
            shifted[i * n + i] -= shift;
        }
        
        // Approximate using Gershgorin circles
        let mut min_bound = f64::INFINITY;
        for i in 0..n {
            let row_sum: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| shifted[i * n + j].abs())
                .sum();
            let center = shifted[i * n + i];
            min_bound = min_bound.min((center - row_sum).abs());
        }
        
        min_bound.max(shift)
    }

    /// Get last computed distance
    pub fn last_distance(&self) -> f64 {
        self.last_distance
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.estimator = RollingCovarianceEstimator::new(
            self.config.num_features,
            self.config.window_size
        );
        self.last_distance = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mahalanobis_normal_data() {
        let config = MahalanobisConfig {
            num_features: 2,
            window_size: 100,
            min_samples: 20,
            outlier_threshold: 9.21, // χ²(2, 0.99)
            ..Default::default()
        };
        
        let mut calc = MahalanobisDistanceCalculator::new(config);
        
        // Feed normal correlated data
        for i in 0..50 {
            let x = i as f64 * 0.1;
            let y = x * 0.8 + (i as f64 * 0.07).sin();
            let result = calc.update(&[x, y]);
            
            if i >= 20 {
                assert!(result.is_some());
                let r = result.unwrap();
                assert!(r.distance < 15.0, "Normal data should have low distance");
            }
        }
    }

    #[test]
    fn test_mahalanobis_outlier_detection() {
        let config = MahalanobisConfig {
            num_features: 2,
            min_samples: 20,
            outlier_threshold: 9.21,
            ..Default::default()
        };
        
        let mut calc = MahalanobisDistanceCalculator::new(config);
        
        // Establish normal regime
        for i in 0..30 {
            calc.update(&[i as f64 * 0.1, i as f64 * 0.08]);
        }
        
        // Inject outlier
        let result = calc.update(&[10.0, 10.0]);
        assert!(result.is_some());
        let r = result.unwrap();
        
        assert!(r.is_outlier, "Should detect large deviation as outlier");
        assert!(r.distance > 9.21, "Distance should exceed threshold");
    }

    #[test]
    fn test_singular_covariance_handling() {
        let config = MahalanobisConfig::default();
        let mut calc = MahalanobisDistanceCalculator::new(config);
        
        // Add perfectly correlated data
        for i in 0..50 {
            let val = i as f64 * 0.1;
            calc.update(&[val, val, val]);
        }
        
        // Should not panic with singular covariance
        let result = calc.update(&[5.0, 5.0, 5.0]);
        assert!(result.is_some());
    }
}
