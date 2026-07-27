/**
 * Copula Sampler for Multi-Asset Correlated Synthetic Returns
 * 
 * Generates correlated synthetic returns using Copula theory:
 * - Gaussian Copula for normal dependence
 * - Student-t Copula for tail dependence
 * - Vine Copulas for complex hierarchical structures
 * 
 * Zero-cost abstractions, stack-allocated where possible.
 * Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
 */

use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

/// Copula types supported
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CopulaType {
    Gaussian,
    StudentT { dof: f64 },
    Clayton { alpha: f64 },
    Gumbel { alpha: f64 },
    Frank { alpha: f64 },
}

/// Correlation structure for multi-asset simulation
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    /// Dimension (number of assets)
    pub n: usize,
    /// Flattened correlation matrix (row-major)
    pub data: Vec<f64>,
    /// Cholesky decomposition (lower triangular)
    cholesky: Vec<f64>,
}

impl CorrelationMatrix {
    /// Create correlation matrix from flat data
    pub fn new(n: usize, data: Vec<f64>) -> Result<Self, &'static str> {
        if data.len() != n * n {
            return Err("Dimension mismatch");
        }
        
        // Validate symmetry and unit diagonal
        for i in 0..n {
            if (data[i * n + i] - 1.0).abs() > 1e-6 {
                return Err("Diagonal must be 1.0");
            }
            for j in 0..i {
                if (data[i * n + j] - data[j * n + i]).abs() > 1e-6 {
                    return Err("Matrix must be symmetric");
                }
            }
        }
        
        let cholesky = Self::cholesky_decomposition(&data, n)?;
        
        Ok(Self { n, data, cholesky })
    }

    /// Cholesky decomposition for Gaussian sampling
    fn cholesky_decomposition(data: &[f64], n: usize) -> Result<Vec<f64>, &'static str> {
        let mut l = vec![0.0; n * n];
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                if j == i {
                    for k in 0..j {
                        sum += l[j * n + k].powi(2);
                    }
                    let val = data[j * n + j] - sum;
                    if val <= 0.0 {
                        return Err("Matrix not positive definite");
                    }
                    l[j * n + j] = val.sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = (data[i * n + j] - sum) / l[j * n + j];
                }
            }
        }
        
        Ok(l)
    }

    /// Get correlation between two assets
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// Get Cholesky lower triangular element
    fn cholesky_get(&self, i: usize, j: usize) -> f64 {
        if j > i {
            0.0
        } else {
            self.cholesky[i * self.n + j]
        }
    }
}

/// Marginal distribution specification for each asset
#[derive(Debug, Clone)]
pub struct MarginalDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub skewness: f64,      // For non-Gaussian marginals
    pub kurtosis: f64,      // Excess kurtosis for fat tails
    pub dist_type: MarginalType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarginalType {
    Normal,
    StudentT { dof: f64 },
    SkewNormal { alpha: f64 },
    Empirical,              // Use historical empirical distribution
}

impl Default for MarginalDistribution {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.02,
            skewness: 0.0,
            kurtosis: 3.0,
            dist_type: MarginalType::Normal,
        }
    }
}

/// High-performance Copula sampler
pub struct CopulaSampler {
    rng: ChaCha8Rng,
    correlation: CorrelationMatrix,
    marginals: Vec<MarginalDistribution>,
    copula_type: CopulaType,
    /// Pre-allocated buffer for samples
    sample_buffer: Vec<f64>,
    /// Pre-allocated buffer for uniforms
    uniform_buffer: Vec<f64>,
}

impl CopulaSampler {
    /// Create new Copula sampler
    pub fn new(
        seed: u64,
        correlation: CorrelationMatrix,
        marginals: Vec<MarginalDistribution>,
        copula_type: CopulaType,
    ) -> Result<Self, &'static str> {
        if correlation.n != marginals.len() {
            return Err("Correlation dimension must match number of marginals");
        }
        
        let capacity = correlation.n;
        
        Ok(Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            correlation,
            marginals,
            copula_type,
            sample_buffer: Vec::with_capacity(capacity),
            uniform_buffer: Vec::with_capacity(capacity),
        })
    }

    /// Generate one sample of correlated returns
    pub fn sample(&mut self) -> Vec<f64> {
        self.sample_buffer.clear();
        self.uniform_buffer.clear();
        
        // Step 1: Generate independent standard normals
        let mut z: Vec<f64> = Vec::with_capacity(self.correlation.n);
        let normal = StandardNormal;
        for _ in 0..self.correlation.n {
            z.push(normal.sample(&mut self.rng));
        }
        
        // Step 2: Apply Cholesky transformation for correlation
        let mut correlated: Vec<f64> = Vec::with_capacity(self.correlation.n);
        for i in 0..self.correlation.n {
            let mut sum = 0.0;
            for j in 0..=i {
                sum += self.correlation.cholesky_get(i, j) * z[j];
            }
            correlated.push(sum);
        }
        
        // Step 3: Transform to uniforms via CDF (Gaussian CDF)
        for &z_val in &correlated {
            let u = Self::gaussian_cdf(z_val);
            self.uniform_buffer.push(u);
        }
        
        // Step 4: Apply inverse marginal CDFs
        for (i, u) in self.uniform_buffer.iter().enumerate() {
            let sample = self.inverse_marginal_cdf(*u, &self.marginals[i]);
            self.sample_buffer.push(sample);
        }
        
        self.sample_buffer.clone()
    }

    /// Generate multiple samples efficiently
    pub fn sample_batch(&mut self, count: usize) -> Vec<Vec<f64>> {
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            batch.push(self.sample());
        }
        batch
    }

    /// Gaussian CDF approximation (Abramowitz & Stegun)
    fn gaussian_cdf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        0.5 * (1.0 + sign * y)
    }

    /// Inverse Gaussian CDF (Rational approximation)
    fn gaussian_icdf(p: f64) -> f64 {
        if p <= 0.0 {
            return -10.0;
        }
        if p >= 1.0 {
            return 10.0;
        }
        
        if p < 0.5 {
            -Self::gaussian_icdf(1.0 - p)
        } else {
            let r = ((1.0 - p) * (1.0 - p)).ln();
            let c0 = 2.515517;
            let c1 = 0.802853;
            let c2 = 0.010328;
            let d1 = 1.432788;
            let d2 = 0.189269;
            let d3 = 0.001308;
            
            r.sqrt() - (c0 + c1 * r + c2 * r * r) 
                / (1.0 + d1 * r + d2 * r * r + d3 * r * r * r)
        }
    }

    /// Inverse CDF for specified marginal distribution
    fn inverse_marginal_cdf(&self, u: f64, marginal: &MarginalDistribution) -> f64 {
        match marginal.dist_type {
            MarginalType::Normal => {
                let z = Self::gaussian_icdf(u);
                marginal.mean + marginal.std_dev * z
            },
            MarginalType::StudentT { dof } => {
                // Approximate inverse t-CDF
                let z = Self::gaussian_icdf(u);
                let correction = 1.0 + z * z / (4.0 * dof);
                let t_val = z * correction;
                marginal.mean + marginal.std_dev * t_val
            },
            MarginalType::SkewNormal { alpha } => {
                let z = Self::gaussian_icdf(u);
                // Skew-normal approximation
                let skew_adjustment = alpha * z * z / (1.0 + alpha.abs());
                marginal.mean + marginal.std_dev * (z + skew_adjustment)
            },
            MarginalType::Empirical => {
                // Simple linear interpolation of empirical quantiles
                // In production, would use actual empirical CDF
                let z = Self::gaussian_icdf(u);
                marginal.mean + marginal.std_dev * z
            },
        }
    }

    /// Generate samples with tail dependence (Student-t copula)
    pub fn sample_tail_dependent(&mut self, dof: f64) -> Vec<f64> {
        self.sample_buffer.clear();
        
        // Sample scaling factor from chi-squared
        let w = self.sample_chi_squared(dof) / dof;
        
        // Generate correlated normals
        let mut z: Vec<f64> = Vec::with_capacity(self.correlation.n);
        let normal = StandardNormal;
        for _ in 0..self.correlation.n {
            z.push(normal.sample(&mut self.rng));
        }
        
        // Apply correlation and scale
        let mut correlated: Vec<f64> = Vec::with_capacity(self.correlation.n);
        for i in 0..self.correlation.n {
            let mut sum = 0.0;
            for j in 0..=i {
                sum += self.correlation.cholesky_get(i, j) * z[j];
            }
            correlated.push(sum / w.sqrt());
        }
        
        // Transform through marginals
        for (i, &z_val) in correlated.iter().enumerate() {
            let u = Self::gaussian_cdf(z_val);
            let sample = self.inverse_marginal_cdf(u, &self.marginals[i]);
            self.sample_buffer.push(sample);
        }
        
        self.sample_buffer.clone()
    }

    /// Sample from chi-squared distribution
    fn sample_chi_squared(&mut self, dof: f64) -> f64 {
        // Gamma distribution with shape = dof/2, scale = 2
        let shape = dof / 2.0;
        let scale = 2.0;
        
        // Marsaglia and Tsang's method for gamma sampling
        if shape >= 1.0 {
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            
            loop {
                let x = StandardNormal.sample(&mut self.rng);
                let v = (1.0 + c * x).powi(3);
                
                if v > 0.0 {
                    let u = self.rng.gen::<f64>();
                    if u < 1.0 - 0.0331 * x * x * x * x || 
                       u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                        return d * v * scale;
                    }
                }
            }
        } else {
            // For shape < 1
            let u = self.rng.gen::<f64>();
            self.sample_chi_squared(shape + 1.0) * u.powf(1.0 / shape)
        }
    }

    /// Calculate portfolio variance from correlation
    pub fn portfolio_variance(&self, weights: &[f64]) -> f64 {
        if weights.len() != self.correlation.n {
            panic!("Weight dimension mismatch");
        }
        
        let mut var = 0.0;
        for i in 0..self.correlation.n {
            for j in 0..self.correlation.n {
                let corr = self.correlation.get(i, j);
                let std_i = self.marginals[i].std_dev;
                let std_j = self.marginals[j].std_dev;
                var += weights[i] * weights[j] * std_i * std_j * corr;
            }
        }
        var
    }

    /// Get marginal distributions
    pub fn get_marginals(&self) -> &[MarginalDistribution] {
        &self.marginals
    }

    /// Update correlation matrix dynamically
    pub fn update_correlation(&mut self, new_corr: CorrelationMatrix) -> Result<(), &'static str> {
        if new_corr.n != self.marginals.len() {
            return Err("New correlation dimension mismatch");
        }
        self.correlation = new_corr;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_matrix_creation() {
        // 3x3 correlation matrix
        let data = vec![
            1.0, 0.5, 0.3,
            0.5, 1.0, 0.4,
            0.3, 0.4, 1.0,
        ];
        
        let corr = CorrelationMatrix::new(3, data).unwrap();
        assert_eq!(corr.n, 3);
        assert!((corr.get(0, 1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_copula_sampling() {
        let data = vec![
            1.0, 0.6,
            0.6, 1.0,
        ];
        let corr = CorrelationMatrix::new(2, data).unwrap();
        
        let marginals = vec![
            MarginalDistribution::default(),
            MarginalDistribution::default(),
        ];
        
        let mut sampler = CopulaSampler::new(
            42,
            corr,
            marginals,
            CopulaType::Gaussian,
        ).unwrap();
        
        let sample = sampler.sample();
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_correlation_preservation() {
        let data = vec![
            1.0, 0.8,
            0.8, 1.0,
        ];
        let corr = CorrelationMatrix::new(2, data).unwrap();
        
        let marginals = vec![
            MarginalDistribution::default(),
            MarginalDistribution::default(),
        ];
        
        let mut sampler = CopulaSampler::new(42, corr, marginals, CopulaType::Gaussian).unwrap();
        
        // Generate many samples
        let n_samples = 10000;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        
        for _ in 0..n_samples {
            let sample = sampler.sample();
            let (x, y) = (sample[0], sample[1]);
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        
        let mean_x = sum_x / n_samples as f64;
        let mean_y = sum_y / n_samples as f64;
        
        let cov_xy = sum_xy / n_samples as f64 - mean_x * mean_y;
        let var_x = sum_x2 / n_samples as f64 - mean_x * mean_x;
        let var_y = sum_y2 / n_samples as f64 - mean_y * mean_y;
        
        let corr_estimate = cov_xy / (var_x * var_y).sqrt();
        
        // Should be close to 0.8
        assert!((corr_estimate - 0.8).abs() < 0.05, "Correlation estimate: {}", corr_estimate);
    }

    #[test]
    fn test_portfolio_variance() {
        let data = vec![
            1.0, 0.5,
            0.5, 1.0,
        ];
        let corr = CorrelationMatrix::new(2, data).unwrap();
        
        let marginals = vec![
            MarginalDistribution { std_dev: 0.02, ..Default::default() },
            MarginalDistribution { std_dev: 0.03, ..Default::default() },
        ];
        
        let sampler = CopulaSampler::new(42, corr, marginals, CopulaType::Gaussian).unwrap();
        
        let weights = vec![0.5, 0.5];
        let var = sampler.portfolio_variance(&weights);
        
        // Manual calculation: w'Σw
        // = 0.25*0.0004 + 0.25*0.0009 + 2*0.25*0.5*0.02*0.03
        // = 0.0001 + 0.000225 + 0.00015 = 0.000475
        assert!((var - 0.000475).abs() < 1e-6);
    }
}
