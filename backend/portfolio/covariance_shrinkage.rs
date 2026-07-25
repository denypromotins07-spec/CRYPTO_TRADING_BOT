//! Ledoit-Wolf Covariance Shrinkage Estimator
//!
//! Implements robust covariance estimation for crypto portfolios.
//! Addresses non-stationarity and extreme correlation spikes in crypto markets.
//! Uses linear shrinkage toward structured target for numerical stability.
//!
//! # Key Features:
//! - Handles singular/near-singular sample covariance matrices
//! - Adapts shrinkage intensity based on data characteristics
//! - Essential for stable Markowitz optimization during volatility clusters
//!
//! # Mathematical Background:
//! The Ledoit-Wolf estimator shrinks the sample covariance S toward a target T:
//! Σ_shrunk = (1 - δ) * S + δ * T
//! where δ is the optimal shrinkage intensity computed from data.

use std::f64;

/// Fixed portfolio size (BTC, SOL, ETH, USDT)
const N_ASSETS: usize = 4;

/// Result of Ledoit-Wolf shrinkage estimation
#[derive(Debug, Clone)]
pub struct ShrinkageResult {
    /// Shrunk covariance matrix (row-major 4x4)
    pub covariance: [[f64; N_ASSETS]; N_ASSETS],
    /// Optimal shrinkage intensity (0 = no shrinkage, 1 = full shrinkage)
    pub shrinkage_intensity: f64,
    /// Original sample covariance (for comparison)
    pub sample_covariance: [[f64; N_ASSETS]; N_ASSETS],
    /// Target matrix used for shrinkage
    pub target_matrix: [[f64; N_ASSETS]; N_ASSETS],
}

/// Ledoit-Wolf Shrinkage Estimator
///
/// Provides robust covariance estimation by shrinking toward:
/// 1. Constant correlation model (default)
/// 2. Diagonal model (variances only)
/// 3. Identity matrix (scaled)
pub struct LedoitWolfEstimator {
    /// Number of observations used in estimation
    n_observations: usize,
    /// Shrinkage target type
    target_type: ShrinkageTarget,
}

/// Type of shrinkage target matrix
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShrinkageTarget {
    /// Constant correlation model (recommended for crypto)
    ConstantCorrelation,
    /// Diagonal matrix (variances only)
    Diagonal,
    /// Scaled identity matrix
    Identity,
}

impl LedoitWolfEstimator {
    /// Create new estimator with default settings
    pub fn new(n_observations: usize) -> Self {
        Self {
            n_observations,
            target_type: ShrinkageTarget::ConstantCorrelation,
        }
    }

    /// Create estimator with custom target type
    pub fn with_target(n_observations: usize, target_type: ShrinkageTarget) -> Self {
        Self {
            n_observations,
            target_type,
        }
    }

    /// Compute shrunk covariance from return data
    ///
    /// # Arguments
    /// * `returns` - Matrix of returns [n_observations x n_assets]
    ///
    /// # Returns
    /// ShrinkageResult with optimal shrunk covariance matrix
    pub fn compute(&self, returns: &[[f64; N_ASSETS]]) -> ShrinkageResult {
        let n = returns.len();
        if n < 2 {
            // Not enough data, return identity-like matrix
            return self._fallback_result();
        }

        // Step 1: Compute sample covariance matrix
        let sample_cov = self._sample_covariance(returns);

        // Step 2: Compute shrinkage target matrix
        let target = self._compute_target(&sample_cov);

        // Step 3: Compute optimal shrinkage intensity
        let delta = self._compute_shrinkage_intensity(returns, &sample_cov, &target);

        // Step 4: Apply shrinkage
        let mut shrunk_cov = [[0.0; N_ASSETS]; N_ASSETS];
        for i in 0..N_ASSETS {
            for j in 0..N_ASSETS {
                shrunk_cov[i][j] = (1.0 - delta) * sample_cov[i][j] + delta * target[i][j];
            }
        }

        // Ensure symmetry (numerical precision)
        for i in 0..N_ASSETS {
            for j in (i + 1)..N_ASSETS {
                let avg = (shrunk_cov[i][j] + shrunk_cov[j][i]) / 2.0;
                shrunk_cov[i][j] = avg;
                shrunk_cov[j][i] = avg;
            }
        }

        ShrinkageResult {
            covariance: shrunk_cov,
            shrinkage_intensity: delta,
            sample_covariance: sample_cov,
            target_matrix: target,
        }
    }

    /// Compute sample covariance matrix from returns
    fn _sample_covariance(&self, returns: &[[f64; N_ASSETS]]) -> [[f64; N_ASSETS]; N_ASSETS] {
        let n = returns.len() as f64;
        
        // Compute means
        let mut means = [0.0; N_ASSETS];
        for r in returns {
            for j in 0..N_ASSETS {
                means[j] += r[j];
            }
        }
        for m in &mut means {
            *m /= n;
        }

        // Compute covariance using Welford-style online algorithm for stability
        let mut cov = [[0.0; N_ASSETS]; N_ASSETS];
        for r in returns {
            for i in 0..N_ASSETS {
                for j in 0..N_ASSETS {
                    cov[i][j] += (r[i] - means[i]) * (r[j] - means[j]);
                }
            }
        }

        // Bessel's correction for unbiased estimate
        let denom = n - 1.0;
        for i in 0..N_ASSETS {
            for j in 0..N_ASSETS {
                cov[i][j] /= denom;
            }
        }

        cov
    }

    /// Compute shrinkage target matrix based on selected type
    fn _compute_target(&self, sample_cov: &[[f64; N_ASSETS]; N_ASSETS]) -> [[f64; N_ASSETS]; N_ASSETS] {
        match self.target_type {
            ShrinkageTarget::ConstantCorrelation => self._constant_correlation_target(sample_cov),
            ShrinkageTarget::Diagonal => self._diagonal_target(sample_cov),
            ShrinkageTarget::Identity => self._identity_target(sample_cov),
        }
    }

    /// Constant correlation target (recommended for crypto)
    ///
    /// Preserves individual variances but assumes equal pairwise correlations.
    /// This is particularly effective for crypto assets which often move together
    /// during market stress periods.
    fn _constant_correlation_target(&self, sample_cov: &[[f64; N_ASSETS]; N_ASSETS]) -> [[f64; N_ASSETS]; N_ASSETS] {
        let mut target = [[0.0; N_ASSETS]; N_ASSETS];

        // Extract variances (diagonal elements)
        let variances: [f64; N_ASSETS] = [
            sample_cov[0][0],
            sample_cov[1][1],
            sample_cov[2][2],
            sample_cov[3][3],
        ];

        // Compute average correlation
        let mut sum_corr = 0.0;
        let mut count = 0;
        for i in 0..N_ASSETS {
            for j in (i + 1)..N_ASSETS {
                let vol_i = variances[i].sqrt();
                let vol_j = variances[j].sqrt();
                if vol_i > 1e-10 && vol_j > 1e-10 {
                    let corr = sample_cov[i][j] / (vol_i * vol_j);
                    // Clip correlation to [-1, 1] for numerical stability
                    let clipped_corr = corr.max(-1.0).min(1.0);
                    sum_corr += clipped_corr;
                    count += 1;
                }
            }
        }

        let avg_corr = if count > 0 { sum_corr / count as f64 } else { 0.0 };

        // Build target matrix with constant correlation
        for i in 0..N_ASSETS {
            target[i][i] = variances[i]; // Preserve variances
            for j in (i + 1)..N_ASSETS {
                let cov_ij = avg_corr * variances[i].sqrt() * variances[j].sqrt();
                target[i][j] = cov_ij;
                target[j][i] = cov_ij;
            }
        }

        target
    }

    /// Diagonal target (only variances, zero covariances)
    fn _diagonal_target(&self, sample_cov: &[[f64; N_ASSETS]; N_ASSETS]) -> [[f64; N_ASSETS]; N_ASSETS] {
        let mut target = [[0.0; N_ASSETS]; N_ASSETS];
        for i in 0..N_ASSETS {
            target[i][i] = sample_cov[i][i];
        }
        target
    }

    /// Scaled identity target
    fn _identity_target(&self, sample_cov: &[[f64; N_ASSETS]; N_ASSETS]) -> [[f64; N_ASSETS]; N_ASSETS] {
        let mut target = [[0.0; N_ASSETS]; N_ASSETS];
        
        // Average variance
        let avg_var = (0..N_ASSETS).map(|i| sample_cov[i][i]).sum::<f64>() / N_ASSETS as f64;
        
        for i in 0..N_ASSETS {
            target[i][i] = avg_var;
        }
        
        target
    }

    /// Compute optimal shrinkage intensity using Ledoit-Wolf formula
    ///
    /// δ* = max(0, min(1, (sum of squared errors) / (sum of squared deviations from target)))
    fn _compute_shrinkage_intensity(
        &self,
        returns: &[[f64; N_ASSETS]],
        sample_cov: &[[f64; N_ASSETS]; N_ASSETS],
        target: &[[f64; N_ASSETS]; N_ASSETS],
    ) -> f64 {
        let n = returns.len() as f64;
        if n <= 1.0 {
            return 1.0; // Full shrinkage when no data
        }

        // Compute means
        let mut means = [0.0; N_ASSETS];
        for r in returns {
            for j in 0..N_ASSETS {
                means[j] += r[j];
            }
        }
        for m in &mut means {
            *m /= n;
        }

        // Compute sum of squared fourth moments (numerator)
        let mut sum_sq_errors = 0.0;
        for r in returns {
            // Centered outer product for this observation
            let mut centered: [f64; N_ASSETS] = [0.0; N_ASSETS];
            for j in 0..N_ASSETS {
                centered[j] = r[j] - means[j];
            }

            // Outer product
            let mut obs_cov = [[0.0; N_ASSETS]; N_ASSETS];
            for i in 0..N_ASSETS {
                for j in 0..N_ASSETS {
                    obs_cov[i][j] = centered[i] * centered[j];
                }
            }

            // Squared deviation from sample covariance
            for i in 0..N_ASSETS {
                for j in 0..N_ASSETS {
                    let diff = obs_cov[i][j] - sample_cov[i][j];
                    sum_sq_errors += diff * diff;
                }
            }
        }

        // Normalize
        let gamma = sum_sq_errors / (n * n);

        // Compute sum of squared deviations from target (denominator)
        let mut delta_sq = 0.0;
        for i in 0..N_ASSETS {
            for j in 0..N_ASSETS {
                let diff = sample_cov[i][j] - target[i][j];
                delta_sq += diff * diff;
            }
        }

        // Optimal shrinkage intensity
        let delta = gamma / delta_sq.max(1e-12);

        // Clamp to [0, 1] and apply small-sample correction
        let corrected_delta = delta.min(1.0).max(0.0);
        
        // Additional adjustment for very small samples
        let adjusted_delta = corrected_delta.min((N_ASSETS as f64 + 2.0) / n);

        adjusted_delta
    }

    /// Fallback result when insufficient data
    fn _fallback_result(&self) -> ShrinkageResult {
        let mut cov = [[0.0; N_ASSETS]; N_ASSETS];
        let mut target = [[0.0; N_ASSETS]; N_ASSETS];
        
        // Use typical crypto volatilities as fallback
        let typical_vars = [0.04, 0.09, 0.05, 0.0001]; // BTC, SOL, ETH, USDT
        for i in 0..N_ASSETS {
            cov[i][i] = typical_vars[i];
            target[i][i] = typical_vars[i];
        }

        ShrinkageResult {
            covariance: cov,
            shrinkage_intensity: 0.5,
            sample_covariance: cov,
            target_matrix: target,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_correlation_target() {
        // Generate synthetic correlated returns
        let returns = vec![
            [0.01, 0.02, 0.015, 0.001],
            [-0.02, -0.03, -0.025, 0.000],
            [0.015, 0.025, 0.02, 0.001],
            [0.005, 0.01, 0.008, 0.000],
            [-0.01, -0.015, -0.012, 0.001],
        ];

        let estimator = LedoitWolfEstimator::new(returns.len());
        let result = estimator.compute(&returns);

        // Shrinkage intensity should be between 0 and 1
        assert!(result.shrinkage_intensity >= 0.0);
        assert!(result.shrinkage_intensity <= 1.0);

        // Result should be symmetric
        for i in 0..N_ASSETS {
            for j in (i + 1)..N_ASSETS {
                let diff = (result.covariance[i][j] - result.covariance[j][i]).abs();
                assert!(diff < 1e-10, "Covariance should be symmetric");
            }
        }

        // Diagonal elements should be positive (variances)
        for i in 0..N_ASSETS {
            assert!(result.covariance[i][i] > 0.0, "Variance should be positive");
        }
    }

    #[test]
    fn test_handles_near_singular_data() {
        // Highly correlated returns (near-singular covariance)
        let returns = vec![
            [0.01, 0.01, 0.01, 0.001],
            [0.01, 0.01, 0.01, 0.001],
            [0.01, 0.01, 0.01, 0.001],
            [0.01, 0.01, 0.01, 0.001],
        ];

        let estimator = LedoitWolfEstimator::with_target(
            returns.len(),
            ShrinkageTarget::ConstantCorrelation,
        );
        let result = estimator.compute(&returns);

        // Should not panic and should produce valid matrix
        assert!(result.shrinkage_intensity.is_finite());
        
        // High shrinkage expected for near-singular data
        assert!(result.shrinkage_intensity > 0.1);
    }

    #[test]
    fn test_different_targets() {
        let returns = vec![
            [0.02, 0.03, 0.025, 0.001],
            [-0.01, -0.02, -0.015, 0.000],
            [0.015, 0.02, 0.018, 0.001],
            [0.01, 0.015, 0.012, 0.000],
        ];

        let targets = [
            ShrinkageTarget::ConstantCorrelation,
            ShrinkageTarget::Diagonal,
            ShrinkageTarget::Identity,
        ];

        for target_type in targets {
            let estimator = LedoitWolfEstimator::with_target(returns.len(), target_type);
            let result = estimator.compute(&returns);
            
            assert!(result.shrinkage_intensity >= 0.0);
            assert!(result.shrinkage_intensity <= 1.0);
        }
    }
}
