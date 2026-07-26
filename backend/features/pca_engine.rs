//! PCA Engine - Principal Component Analysis for Orthogonal Market Factors
//! 
//! This module performs dimensionality reduction on high-dimensional crypto
//! market data to isolate orthogonal (uncorrelated) factors driving returns.
//! 
//! Key Features:
//! - Zero-cost abstractions with pre-allocated buffers
//! - Incremental PCA for streaming data (memory efficient)
//! - Dynamic eigenvector updates as new regimes emerge
//! - Handles singular covariance matrices gracefully

use std::collections::VecDeque;

/// Result from PCA decomposition
#[derive(Debug, Clone)]
pub struct PCAResult {
    /// Eigenvalues (variance explained by each component)
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (principal components as columns)
    pub eigenvectors: Vec<Vec<f64>>,
    /// Explained variance ratio for each component
    pub explained_variance_ratio: Vec<f64>,
    /// Cumulative explained variance ratio
    pub cumulative_variance_ratio: Vec<f64>,
    /// Number of original features
    pub n_features: usize,
    /// Number of components retained
    pub n_components: usize,
    /// Mean of each feature (used for centering)
    pub means: Vec<f64>,
}

impl PCAResult {
    /// Get number of components needed to explain target variance
    pub fn components_for_variance(&self, target: f64) -> usize {
        self.cumulative_variance_ratio
            .iter()
            .position(|&v| v >= target)
            .unwrap_or(self.n_components)
            + 1
    }
}

/// Covariance matrix estimator with online updates
pub struct CovarianceEstimator {
    /// Running mean
    mean: Vec<f64>,
    /// Running covariance (upper triangular stored)
    cov_upper: Vec<f64>,
    /// Number of observations seen
    n_obs: u64,
    /// Number of features
    n_features: usize,
    /// Decay factor for exponential weighting (1.0 = no decay)
    decay: f64,
}

impl CovarianceEstimator {
    /// Create new estimator
    pub fn new(n_features: usize, decay: f64) -> Self {
        let n_cov = n_features * (n_features + 1) / 2;
        Self {
            mean: vec![0.0; n_features],
            cov_upper: vec![0.0; n_cov],
            n_obs: 0,
            n_features,
            decay: decay.clamp(0.9, 1.0),
        }
    }

    /// Update with new observation
    pub fn update(&mut self, observation: &[f64]) {
        if observation.len() != self.n_features {
            panic!("Observation dimension mismatch");
        }

        self.n_obs += 1;
        let alpha = if self.decay < 1.0 {
            1.0 - self.decay
        } else {
            1.0 / self.n_obs as f64
        };

        // Update mean
        let old_mean = self.mean.clone();
        for (i, &x) in observation.iter().enumerate() {
            self.mean[i] += alpha * (x - self.mean[i]);
        }

        // Update covariance using Welford-like algorithm
        let idx = |i: usize, j: usize| {
            if i <= j {
                i * (2 * self.n_features - i + 1) / 2 + (j - i)
            } else {
                j * (2 * self.n_features - j + 1) / 2 + (i - j)
            }
        };

        for i in 0..self.n_features {
            for j in i..self.n_features {
                let diff_i = observation[i] - old_mean[i];
                let diff_j = observation[j] - old_mean[j];
                let k = idx(i, j);
                self.cov_upper[k] += alpha * (diff_i * diff_j - self.cov_upper[k]);
            }
        }
    }

    /// Get full covariance matrix
    pub fn covariance_matrix(&self) -> Vec<Vec<f64>> {
        let idx = |i: usize, j: usize| {
            if i <= j {
                i * (2 * self.n_features - i + 1) / 2 + (j - i)
            } else {
                j * (2 * self.n_features - j + 1) / 2 + (i - j)
            }
        };

        (0..self.n_features)
            .map(|i| {
                (0..self.n_features)
                    .map(|j| self.cov_upper[idx(i, j)])
                    .collect()
            })
            .collect()
    }

    /// Get number of observations
    pub fn n_obs(&self) -> u64 {
        self.n_obs
    }
}

/// Eigenvalue decomposition using Jacobi method
pub struct JacobiEigenSolver;

impl JacobiEigenSolver {
    /// Compute eigenvalues and eigenvectors of symmetric matrix
    pub fn solve(matrix: &[Vec<f64>], max_iter: usize, tol: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = matrix.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        // Initialize eigenvectors as identity
        let mut eigenvectors = vec![vec![0.0; n]; n];
        for i in 0..n {
            eigenvectors[i][i] = 1.0;
        }

        // Copy matrix to working array
        let mut a = matrix.clone();

        // Jacobi iteration
        for _ in 0..max_iter {
            // Find largest off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 1;

            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            // Check convergence
            if max_val < tol {
                break;
            }

            // Compute rotation angle
            let theta = if (a[p][p] - a[q][q]).abs() < 1e-10 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * ((2.0 * a[p][q]) / (a[p][p] - a[q][q])).atan()
            };

            let c = theta.cos();
            let s = theta.sin();

            // Apply rotation to matrix
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];

            a[p][p] = c * c * app + s * s * aqq - 2.0 * s * c * apq;
            a[q][q] = s * s * app + c * c * aqq + 2.0 * s * c * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for i in 0..n {
                if i != p && i != q {
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[p][i] = a[i][p];
                    a[i][q] = s * aip + c * aiq;
                    a[q][i] = a[i][q];
                }
            }

            // Update eigenvectors
            for i in 0..n {
                let vip = eigenvectors[i][p];
                let viq = eigenvectors[i][q];
                eigenvectors[i][p] = c * vip - s * viq;
                eigenvectors[i][q] = s * vip + c * viq;
            }
        }

        // Extract eigenvalues from diagonal
        let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();

        (eigenvalues, eigenvectors)
    }
}

/// Main PCA engine with incremental updates
pub struct PCAEngine {
    estimator: CovarianceEstimator,
    result: Option<PCAResult>,
    n_components: Option<usize>,
    variance_threshold: Option<f64>,
}

impl PCAEngine {
    /// Create new PCA engine
    pub fn new(n_features: usize) -> Self {
        Self {
            estimator: CovarianceEstimator::new(n_features, 1.0),
            result: None,
            n_components: None,
            variance_threshold: Some(0.95),
        }
    }

    /// Create with exponential decay weighting
    pub fn with_decay(n_features: usize, decay: f64) -> Self {
        Self {
            estimator: CovarianceEstimator::new(n_features, decay),
            result: None,
            n_components: None,
            variance_threshold: Some(0.95),
        }
    }

    /// Set number of components to retain
    pub fn set_n_components(&mut self, n: usize) {
        self.n_components = Some(n);
        self.variance_threshold = None;
    }

    /// Set variance threshold for component selection
    pub fn set_variance_threshold(&mut self, threshold: f64) {
        self.variance_threshold = Some(threshold.clamp(0.0, 1.0));
        self.n_components = None;
    }

    /// Add observation for incremental PCA
    pub fn partial_fit(&mut self, observation: &[f64]) {
        self.estimator.update(observation);
    }

    /// Fit PCA on batch data
    pub fn fit(&mut self, data: &[Vec<f64>]) -> &PCAResult {
        if data.is_empty() {
            panic!("Empty data for PCA fitting");
        }

        let n_features = data[0].len();
        self.estimator = CovarianceEstimator::new(n_features, 1.0);

        for obs in data {
            self.estimator.update(obs);
        }

        self.compute_pca()
    }

    /// Compute PCA decomposition
    fn compute_pca(&mut self) -> &PCAResult {
        let cov = self.estimator.covariance_matrix();
        let n = cov.len();

        // Handle edge case
        if n == 0 {
            self.result = Some(PCAResult {
                eigenvalues: vec![],
                eigenvectors: vec![],
                explained_variance_ratio: vec![],
                cumulative_variance_ratio: vec![],
                n_features: 0,
                n_components: 0,
                means: vec![],
            });
            return self.result.as_ref().unwrap();
        }

        // Eigen decomposition
        let (mut eigenvalues, eigenvectors) = JacobiEigenSolver::solve(&cov, 100, 1e-8);

        // Sort by eigenvalue (descending)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        eigenvalues = indices.iter().map(|&i| eigenvalues[i].max(0.0)).collect();
        let sorted_eigenvectors: Vec<Vec<f64>> = indices
            .iter()
            .map(|&i| eigenvectors.iter().map(|v| v[i]).collect())
            .collect();

        // Compute explained variance ratios
        let total_var: f64 = eigenvalues.iter().sum();
        let explained_ratio: Vec<f64> = eigenvalues
            .iter()
            .map(|&ev| if total_var > 0.0 { ev / total_var } else { 0.0 })
            .collect();

        // Cumulative variance
        let mut cumulative = Vec::with_capacity(n);
        let mut sum = 0.0;
        for &r in &explained_ratio {
            sum += r;
            cumulative.push(sum);
        }

        // Determine number of components
        let n_components = if let Some(nc) = self.n_components {
            nc.min(n)
        } else if let Some(threshold) = self.variance_threshold {
            cumulative
                .iter()
                .position(|&v| v >= threshold)
                .unwrap_or(n)
                + 1
        } else {
            n
        };

        self.result = Some(PCAResult {
            eigenvalues: eigenvalues[..n_components].to_vec(),
            eigenvectors: sorted_eigenvectors[..n_components]
                .iter()
                .map(|v| v.clone())
                .collect(),
            explained_variance_ratio: explained_ratio[..n_components].to_vec(),
            cumulative_variance_ratio: cumulative[..n_components].to_vec(),
            n_features: n,
            n_components,
            means: self.estimator.mean.clone(),
        });

        self.result.as_ref().unwrap()
    }

    /// Transform data to principal component space
    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let result = match &self.result {
            Some(r) => r,
            None => panic!("Must fit PCA before transforming"),
        };

        data.iter()
            .map(|obs| {
                // Center observation
                let centered: Vec<f64> = obs
                    .iter()
                    .zip(result.means.iter())
                    .map(|(&x, &m)| x - m)
                    .collect();

                // Project onto principal components
                result
                    .eigenvectors
                    .iter()
                    .map(|ev| centered.iter().zip(ev.iter()).map(|(&c, &e)| c * e).sum())
                    .collect()
            })
            .collect()
    }

    /// Transform single observation
    pub fn transform_one(&self, observation: &[f64]) -> Vec<f64> {
        let result = match &self.result {
            Some(r) => r,
            None => panic!("Must fit PCA before transforming"),
        };

        let centered: Vec<f64> = observation
            .iter()
            .zip(result.means.iter())
            .map(|(&x, &m)| x - m)
            .collect();

        result
            .eigenvectors
            .iter()
            .map(|ev| centered.iter().zip(ev.iter()).map(|(&c, &e)| c * e).sum())
            .collect()
    }

    /// Get current PCA result
    pub fn result(&self) -> Option<&PCAResult> {
        self.result.as_ref()
    }

    /// Reset the PCA (clear accumulated statistics)
    pub fn reset(&mut self) {
        let n_features = self.estimator.n_features;
        self.estimator = CovarianceEstimator::new(n_features, self.estimator.decay);
        self.result = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_basic() {
        let mut pca = PCAEngine::new(3);
        pca.set_n_components(2);

        // Generate correlated data
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
            vec![4.0, 8.0, 12.0],
        ];

        let result = pca.fit(&data);
        assert_eq!(result.n_components, 2);
        assert!(result.explained_variance_ratio[0] > 0.9); // First PC should explain most variance
    }

    #[test]
    fn test_incremental_pca() {
        let mut pca = PCAEngine::new(2);

        // Add observations incrementally
        for i in 0..100 {
            let x = i as f64;
            let y = 2.0 * x + (i as f64 % 10 - 5) as f64 * 0.1;
            pca.partial_fit(&[x, y]);
        }

        let result = pca.compute_pca();
        assert!(result.n_components > 0);
    }

    #[test]
    fn test_transform() {
        let mut pca = PCAEngine::new(2);
        let data = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];

        pca.fit(&data);
        let transformed = pca.transform(&data);

        assert_eq!(transformed.len(), data.len());
        assert_eq!(transformed[0].len(), pca.result().unwrap().n_components);
    }
}
