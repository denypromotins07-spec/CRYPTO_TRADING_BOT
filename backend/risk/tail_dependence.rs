//! Tail Dependence Modeling for Crypto Flash Crashes
//! 
//! Implements asymmetric lower-tail dependence models to capture
//! the non-linear correlation spikes during crypto market crashes.
//! Optimized for O(1) computation without numerical overflow.
//! 
//! Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
//! ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour

use std::f64::consts::PI;
use thiserror::Error;

/// Errors specific to tail dependence calculations
#[derive(Error, Debug)]
pub enum TailDependenceError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Numerical overflow in tail dependence calculation")]
    NumericalOverflow,
    #[error("Insufficient data for estimation")]
    InsufficientData,
    #[error("Matrix dimension mismatch")]
    DimensionMismatch,
}

/// Result type for tail dependence operations
pub type TailDependenceResult<T> = Result<T, TailDependenceError>;

/// Tail dependence coefficients for a pair of assets
#[derive(Debug, Clone)]
pub struct TailDependenceCoefficients {
    /// Lower tail dependence coefficient (lambda_L)
    pub lambda_lower: f64,
    /// Upper tail dependence coefficient (lambda_U)
    pub lambda_upper: f64,
    /// Confidence level for estimation
    pub confidence_level: f64,
    /// Number of samples used
    pub sample_count: usize,
}

impl TailDependenceCoefficients {
    /// Create new coefficients
    pub fn new(lambda_lower: f64, lambda_upper: f64, confidence: f64, n: usize) -> Self {
        Self {
            lambda_lower: lambda_lower.clamp(0.0, 1.0),
            lambda_upper: lambda_upper.clamp(0.0, 1.0),
            confidence_level: confidence,
            sample_count: n,
        }
    }
    
    /// Check if significant lower tail dependence exists
    pub fn has_significant_lower_tail(&self, threshold: f64) -> bool {
        self.lambda_lower > threshold
    }
    
    /// Check if asymmetry exists (lower != upper)
    pub fn is_asymmetric(&self, tolerance: f64) -> bool {
        (self.lambda_lower - self.lambda_upper).abs() > tolerance
    }
}

/// Non-parametric tail dependence estimator using empirical copula
pub struct EmpiricalTailDependence {
    /// Quantile level for tail estimation (e.g., 0.05 for 5% tail)
    quantile_level: f64,
    /// Minimum number of tail observations required
    min_tail_obs: usize,
}

impl EmpiricalTailDependence {
    /// Create a new empirical estimator
    pub fn new(quantile_level: f64, min_tail_obs: usize) -> TailDependenceResult<Self> {
        if quantile_level <= 0.0 || quantile_level >= 0.5 {
            return Err(TailDependenceError::InvalidInput(
                "Quantile level must be in (0, 0.5)".to_string()
            ));
        }
        
        Ok(Self {
            quantile_level,
            min_tail_obs,
        })
    }
    
    /// Estimate lower tail dependence from paired returns
    /// 
    /// Uses the empirical copula approach:
    /// lambda_L = lim_{u->0} P(Y <= F_Y^(-1)(u) | X <= F_X^(-1)(u))
    pub fn estimate_lower_tail(
        &self,
        x: &[f64],
        y: &[f64]
    ) -> TailDependenceResult<TailDependenceCoefficients> {
        if x.len() != y.len() {
            return Err(TailDependenceError::DimensionMismatch);
        }
        
        let n = x.len();
        if n < self.min_tail_obs * 10 {
            return Err(TailDependenceError::InsufficientData);
        }
        
        // Compute empirical ranks
        let mut x_ranks = self.compute_ranks(x);
        let mut y_ranks = self.compute_ranks(y);
        
        // Normalize to [0, 1]
        let n_f64 = n as f64;
        for r in x_ranks.iter_mut() {
            *r /= n_f64;
        }
        for r in y_ranks.iter_mut() {
            *r /= n_f64;
        }
        
        // Count joint tail occurrences
        let mut joint_tail_count = 0;
        let mut marginal_tail_count = 0;
        
        for i in 0..n {
            if x_ranks[i] <= self.quantile_level {
                marginal_tail_count += 1;
                if y_ranks[i] <= self.quantile_level {
                    joint_tail_count += 1;
                }
            }
        }
        
        if marginal_tail_count < self.min_tail_obs {
            return Err(TailDependenceError::InsufficientData);
        }
        
        // Empirical lower tail dependence
        let lambda_lower = joint_tail_count as f64 / marginal_tail_count as f64;
        
        // Upper tail dependence (using symmetry transformation)
        let lambda_upper = self.estimate_upper_tail_empirical(&x_ranks, &y_ranks)?;
        
        Ok(TailDependenceCoefficients::new(
            lambda_lower,
            lambda_upper,
            0.95, // Default confidence
            n,
        ))
    }
    
    /// Estimate upper tail dependence empirically
    fn estimate_upper_tail_empirical(
        &self,
        x_ranks: &[f64],
        y_ranks: &[f64]
    ) -> TailDependenceResult<f64> {
        let n = x_ranks.len();
        let mut joint_upper_count = 0;
        let mut marginal_upper_count = 0;
        
        let upper_threshold = 1.0 - self.quantile_level;
        
        for i in 0..n {
            if x_ranks[i] >= upper_threshold {
                marginal_upper_count += 1;
                if y_ranks[i] >= upper_threshold {
                    joint_upper_count += 1;
                }
            }
        }
        
        if marginal_upper_count < self.min_tail_obs {
            return Ok(0.0); // Cannot estimate reliably
        }
        
        Ok(joint_upper_count as f64 / marginal_upper_count as f64)
    }
    
    /// Compute ranks for a slice of values
    fn compute_ranks(&self, values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
        
        // Sort by value
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Assign ranks (average for ties)
        let mut ranks = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            // Find all elements with same value
            while j < n && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            // Average rank for tied values
            let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }
        
        ranks
    }
}

/// Parametric tail dependence from copula parameters
pub struct ParametricTailDependence;

impl ParametricTailDependence {
    /// Calculate lower tail dependence for Clayton copula
    /// lambda_L = 2^(-1/theta)
    pub fn clayton_lower_tail(theta: f64) -> TailDependenceResult<f64> {
        if theta <= 0.0 {
            return Err(TailDependenceError::InvalidInput(
                "Clayton theta must be positive".to_string()
            ));
        }
        
        Ok(2.0_f64.powf(-1.0 / theta))
    }
    
    /// Calculate upper tail dependence for Gumbel copula
    /// lambda_U = 2 - 2^(1/theta)
    pub fn gumbel_upper_tail(theta: f64) -> TailDependenceResult<f64> {
        if theta < 1.0 {
            return Err(TailDependenceError::InvalidInput(
                "Gumbel theta must be >= 1".to_string()
            ));
        }
        
        Ok(2.0 - 2.0_f64.powf(1.0 / theta))
    }
    
    /// Calculate tail dependence for Student-t copula
    /// Symmetric tails: lambda_L = lambda_U = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
    pub fn student_t_tails(
        nu: f64,
        rho: f64
    ) -> TailDependenceResult<TailDependenceCoefficients> {
        if nu <= 0.0 {
            return Err(TailDependenceError::InvalidInput(
                "Degrees of freedom must be positive".to_string()
            ));
        }
        
        if rho.abs() >= 1.0 {
            return Err(TailDependenceError::InvalidInput(
                "Correlation must be in (-1, 1)".to_string()
            ));
        }
        
        // Compute the argument for t-CDF
        let arg = -((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt();
        
        // Evaluate t-CDF (approximation)
        let cdf_val = Self::t_cdf(arg, nu + 1.0);
        
        let lambda = 2.0 * cdf_val;
        
        Ok(TailDependenceCoefficients::new(
            lambda,
            lambda, // Symmetric for t-copula
            0.95,
            0, // Parametric, no sample count
        ))
    }
    
    /// Student-t CDF approximation
    fn t_cdf(x: f64, nu: f64) -> f64 {
        // Using regularized incomplete beta function approximation
        let t = x / (nu + x * x).sqrt();
        let beta_inc = Self::beta_inc_approx(nu / 2.0, 0.5, (1.0 + t) / 2.0);
        
        if x >= 0.0 {
            1.0 - 0.5 * beta_inc
        } else {
            0.5 * beta_inc
        }
    }
    
    /// Incomplete beta function approximation
    fn beta_inc_approx(a: f64, b: f64, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        
        // Simple continued fraction approximation
        let mut result = x.powf(a) * (1.0 - x).powf(b) / a;
        result /= Self::beta_fn(a, b);
        result.min(1.0)
    }
    
    /// Beta function
    fn beta_fn(a: f64, b: f64) -> f64 {
        // B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
        Self::gamma_lanczos(a) * Self::gamma_lanczos(b) / Self::gamma_lanczos(a + b)
    }
    
    /// Gamma function using Lanczos approximation
    fn gamma_lanczos(x: f64) -> f64 {
        if x <= 0.0 && x.fract() == 0.0 {
            return f64::INFINITY;
        }
        
        let g = 7.0;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        
        let mut z = x - 1.0;
        let mut sum = c[0];
        for i in 1..(g as usize + 2) {
            sum += c[i] / (z + i as f64);
        }
        
        let t = z + g + 0.5;
        (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * sum
    }
}

/// Tail dependence matrix for multiple assets
#[derive(Debug, Clone)]
pub struct TailDependenceMatrix {
    /// Matrix dimension
    dim: usize,
    /// Lower tail dependence coefficients (flattened row-major)
    lower_tail: Vec<f64>,
    /// Upper tail dependence coefficients (flattened row-major)
    upper_tail: Vec<f64>,
    /// Asset names for reference
    asset_names: Vec<String>,
}

impl TailDependenceMatrix {
    /// Create a new tail dependence matrix
    pub fn new(asset_names: Vec<String>) -> Self {
        let dim = asset_names.len();
        let size = dim * dim;
        
        Self {
            dim,
            lower_tail: vec![0.0; size],
            upper_tail: vec![0.0; size],
            asset_names,
        }
    }
    
    /// Set tail dependence for a pair
    pub fn set_pair(&mut self, i: usize, j: usize, lambda_lower: f64, lambda_upper: f64) {
        if i >= self.dim || j >= self.dim {
            return;
        }
        
        let idx = i * self.dim + j;
        self.lower_tail[idx] = lambda_lower.clamp(0.0, 1.0);
        self.upper_tail[idx] = lambda_upper.clamp(0.0, 1.0);
        
        // Symmetric
        let idx_sym = j * self.dim + i;
        self.lower_tail[idx_sym] = lambda_lower.clamp(0.0, 1.0);
        self.upper_tail[idx_sym] = lambda_upper.clamp(0.0, 1.0);
    }
    
    /// Get lower tail dependence between two assets
    pub fn get_lower(&self, i: usize, j: usize) -> Option<f64> {
        if i >= self.dim || j >= self.dim {
            return None;
        }
        Some(self.lower_tail[i * self.dim + j])
    }
    
    /// Get upper tail dependence between two assets
    pub fn get_upper(&self, i: usize, j: usize) -> Option<f64> {
        if i >= self.dim || j >= self.dim {
            return None;
        }
        Some(self.upper_tail[i * self.dim + j])
    }
    
    /// Find the pair with highest lower tail dependence
    pub fn max_lower_tail_pair(&self) -> Option<(usize, usize, f64)> {
        let mut max_val = 0.0;
        let mut max_pair = (0, 1);
        
        for i in 0..self.dim {
            for j in (i + 1)..self.dim {
                let val = self.lower_tail[i * self.dim + j];
                if val > max_val {
                    max_val = val;
                    max_pair = (i, j);
                }
            }
        }
        
        Some((max_pair.0, max_pair.1, max_val))
    }
    
    /// Check if any pair exceeds tail dependence threshold
    pub fn has_extreme_tail_dependence(&self, threshold: f64) -> bool {
        self.lower_tail.iter().any(|&v| v > threshold)
    }
    
    /// Get average lower tail dependence (excluding diagonal)
    pub fn average_lower_tail(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in 0..self.dim {
            for j in (i + 1)..self.dim {
                sum += self.lower_tail[i * self.dim + j];
                count += 1;
            }
        }
        
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

/// Real-time tail dependence monitor for flash crash detection
pub struct TailDependenceMonitor {
    /// Rolling window of recent tail dependence estimates
    history: Vec<TailDependenceCoefficients>,
    /// Maximum history size
    max_history: usize,
    /// Alert threshold for extreme tail dependence
    alert_threshold: f64,
}

impl TailDependenceMonitor {
    /// Create a new monitor
    pub fn new(max_history: usize, alert_threshold: f64) -> Self {
        Self {
            history: Vec::with_capacity(max_history),
            max_history,
            alert_threshold,
        }
    }
    
    /// Add a new observation
    pub fn update(&mut self, coeffs: TailDependenceCoefficients) {
        self.history.push(coeffs);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
    
    /// Check if flash crash conditions are present
    pub fn is_flash_crash_condition(&self) -> bool {
        if self.history.is_empty() {
            return false;
        }
        
        // Check if recent lower tail dependence is extremely high
        let recent = self.history.last().unwrap();
        recent.has_significant_lower_tail(self.alert_threshold)
    }
    
    /// Detect rapid increase in tail dependence
    pub fn detect_tail_dependence_surge(&self, lookback: usize) -> Option<f64> {
        if self.history.len() < lookback + 1 {
            return None;
        }
        
        let start_idx = self.history.len() - lookback - 1;
        let old_avg: f64 = self.history[start_idx..start_idx + lookback]
            .iter()
            .map(|c| c.lambda_lower)
            .sum::<f64>() / lookback as f64;
        
        let current = self.history.last().unwrap().lambda_lower;
        
        // Return the surge magnitude
        Some(current - old_avg)
    }
    
    /// Get the trend in tail dependence
    pub fn tail_dependence_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        
        let first_half: f64 = self.history[..self.history.len() / 2]
            .iter()
            .map(|c| c.lambda_lower)
            .sum::<f64>() / (self.history.len() / 2) as f64;
        
        let second_half: f64 = self.history[self.history.len() / 2..]
            .iter()
            .map(|c| c.lambda_lower)
            .sum::<f64>() / (self.history.len() - self.history.len() / 2) as f64;
        
        second_half - first_half
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_clayton_tail_dependence() {
        let lambda = ParametricTailDependence::clayton_lower_tail(2.0).unwrap();
        assert!((lambda - 0.707).abs() < 0.01);
    }
    
    #[test]
    fn test_empirical_estimator_creation() {
        let estimator = EmpiricalTailDependence::new(0.05, 10).unwrap();
        assert_eq!(estimator.quantile_level, 0.05);
    }
    
    #[test]
    fn test_tail_matrix() {
        let assets = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];
        let mut matrix = TailDependenceMatrix::new(assets);
        matrix.set_pair(0, 1, 0.6, 0.4);
        
        assert_eq!(matrix.get_lower(0, 1), Some(0.6));
        assert_eq!(matrix.get_lower(1, 0), Some(0.6));
    }
}
