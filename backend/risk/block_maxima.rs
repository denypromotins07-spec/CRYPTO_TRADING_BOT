//! Block Maxima Method for Extreme Value Theory
//! 
//! Applies Generalized Extreme Value (GEV) distribution to model
//! maximum drawdowns and extreme losses over fixed time blocks.
//! Optimized for accurate fitting with limited extreme data.
//! 
//! Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
//! ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour

use std::f64::consts::PI;
use thiserror::Error;

/// Errors specific to block maxima / GEV calculations
#[derive(Error, Debug)]
pub enum BlockMaximaError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Numerical overflow in GEV calculation")]
    NumericalOverflow,
    #[error("Insufficient blocks for estimation (need at least {min}, got {actual})")]
    InsufficientBlocks { min: usize, actual: usize },
    #[error("Optimization failed to converge")]
    OptimizationFailed,
    #[error("Invalid GEV parameters: xi={xi}, mu={mu}, sigma={sigma}")]
    InvalidParameters { xi: f64, mu: f64, sigma: f64 },
}

/// Result type for block maxima operations
pub type BlockMaximaResult<T> = Result<T, BlockMaximaError>;

/// Fitted GEV distribution parameters
#[derive(Debug, Clone)]
pub struct GEVParameters {
    /// Shape parameter (tail index) - xi > 0 means heavy tail (Fréchet)
    pub xi: f64,
    /// Location parameter
    pub mu: f64,
    /// Scale parameter (must be positive)
    pub sigma: f64,
    /// Number of blocks used
    pub n_blocks: usize,
    /// Log-likelihood at optimum
    pub log_likelihood: f64,
    /// Block size (observations per block)
    pub block_size: usize,
}

impl GEVParameters {
    /// Create new GEV parameters with validation
    pub fn new(
        xi: f64,
        mu: f64,
        sigma: f64,
        n_blocks: usize,
        log_likelihood: f64,
        block_size: usize,
    ) -> BlockMaximaResult<Self> {
        if sigma <= 0.0 {
            return Err(BlockMaximaError::InvalidParameters { xi, mu, sigma });
        }
        
        Ok(Self {
            xi,
            mu,
            sigma,
            n_blocks,
            log_likelihood,
            block_size,
        })
    }
    
    /// Check if distribution has heavy tail (Fréchet type)
    pub fn is_frechet(&self) -> bool {
        self.xi > 0.0
    }
    
    /// Check if distribution has bounded upper tail (Weibull type)
    pub fn is_weibull(&self) -> bool {
        self.xi < 0.0
    }
    
    /// Check if distribution is Gumbel type (light tail)
    pub fn is_gumbel(&self, tolerance: f64) -> bool {
        self.xi.abs() < tolerance
    }
    
    /// Get the upper endpoint (for Weibull type)
    pub fn upper_endpoint(&self) -> Option<f64> {
        if self.xi < 0.0 {
            Some(self.mu + self.sigma / (-self.xi))
        } else {
            None // Unbounded
        }
    }
    
    /// Get the lower endpoint (for Fréchet type with xi > 0)
    pub fn lower_endpoint(&self) -> Option<f64> {
        if self.xi > 0.0 {
            Some(self.mu - self.sigma / self.xi)
        } else {
            None // Unbounded below
        }
    }
}

/// Block maxima estimator using GEV distribution
pub struct BlockMaximaEstimator {
    /// Minimum number of blocks required
    min_blocks: usize,
    /// Default block size
    default_block_size: usize,
}

impl BlockMaximaEstimator {
    /// Create a new block maxima estimator
    pub fn new(min_blocks: usize, default_block_size: usize) -> Self {
        Self {
            min_blocks,
            default_block_size,
        }
    }
    
    /// Compute block maxima from returns data
    pub fn compute_block_maxima(
        &self,
        returns: &[f64],
        block_size: Option<usize>
    ) -> BlockMaximaResult<Vec<f64>> {
        let bs = block_size.unwrap_or(self.default_block_size);
        
        if returns.len() < bs * 2 {
            return Err(BlockMaximaError::InsufficientBlocks {
                min: 2,
                actual: returns.len() / bs,
            });
        }
        
        let n_blocks = returns.len() / bs;
        if n_blocks < self.min_blocks {
            return Err(BlockMaximaError::InsufficientBlocks {
                min: self.min_blocks,
                actual: n_blocks,
            });
        }
        
        // Compute maxima of negative returns (losses) for each block
        let mut maxima = Vec::with_capacity(n_blocks);
        
        for i in 0..n_blocks {
            let start = i * bs;
            let end = start + bs;
            
            // Find maximum loss in block (minimum return)
            let block_min = returns[start..end]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            
            // Store as positive loss
            maxima.push(-block_min);
        }
        
        Ok(maxima)
    }
    
    /// Fit GEV distribution to block maxima using Maximum Likelihood
    pub fn fit_gev_mle(&self, block_maxima: &[f64]) -> BlockMaximaResult<GEVParameters> {
        let n = block_maxima.len();
        
        if n < self.min_blocks {
            return Err(BlockMaximaError::InsufficientBlocks {
                min: self.min_blocks,
                actual: n,
            });
        }
        
        // Initial parameter estimates using method of moments
        let (xi_init, mu_init, sigma_init) = self.method_of_moments_init(block_maxima)?;
        
        // Optimize using Nelder-Mead style approach
        let (xi, mu, sigma) = self.optimize_mle(block_maxima, xi_init, mu_init, sigma_init)?;
        
        // Compute final log-likelihood
        let log_likelihood = self.gev_log_likelihood(block_maxima, xi, mu, sigma)?;
        
        GEVParameters::new(
            xi,
            mu,
            sigma,
            n,
            log_likelihood,
            self.default_block_size,
        )
    }
    
    /// Method of moments initial estimates
    fn method_of_moments_init(
        &self,
        data: &[f64]
    ) -> BlockMaximaResult<(f64, f64, f64)> {
        let n = data.len();
        if n < 3 {
            return Err(BlockMaximaError::InsufficientBlocks {
                min: 3,
                actual: n,
            });
        }
        
        // Sample moments
        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        
        // Sample skewness
        let skewness = if std_dev > 1e-10 {
            data.iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>() / n as f64
        } else {
            0.0
        };
        
        // Approximate xi from skewness (empirical relationship)
        let xi = skewness / 6.0;
        let xi = xi.clamp(-0.5, 0.5);
        
        // Initial mu and sigma
        let mu = mean;
        let sigma = std_dev * (1.0 + xi.powi(2)).sqrt();
        
        Ok((xi, mu, sigma.max(1e-6)))
    }
    
    /// Optimize MLE using coordinate descent
    fn optimize_mle(
        &self,
        data: &[f64],
        xi_init: f64,
        mu_init: f64,
        sigma_init: f64,
    ) -> BlockMaximaResult<(f64, f64, f64)> {
        let mut xi = xi_init;
        let mut mu = mu_init;
        let mut sigma = sigma_init;
        
        let mut best_ll = self.gev_log_likelihood(data, xi, mu, sigma)
            .unwrap_or(f64::NEG_INFINITY);
        
        // Coordinate descent iterations
        for _iteration in 0..100 {
            let mut improved = false;
            
            // Optimize xi
            for delta_xi in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1] {
                let test_xi = (xi + delta_xi).clamp(-0.5, 0.5);
                if let Ok(ll) = self.gev_log_likelihood(data, test_xi, mu, sigma) {
                    if ll > best_ll {
                        best_ll = ll;
                        xi = test_xi;
                        improved = true;
                    }
                }
            }
            
            // Optimize mu
            let sigma_step = sigma * 0.1;
            for delta_mu in [-sigma_step, -sigma_step * 0.5, sigma_step * 0.5, sigma_step] {
                let test_mu = mu + delta_mu;
                if let Ok(ll) = self.gev_log_likelihood(data, xi, test_mu, sigma) {
                    if ll > best_ll {
                        best_ll = ll;
                        mu = test_mu;
                        improved = true;
                    }
                }
            }
            
            // Optimize sigma
            for factor in [0.8, 0.9, 1.1, 1.2] {
                let test_sigma = sigma * factor;
                if let Ok(ll) = self.gev_log_likelihood(data, xi, mu, test_sigma) {
                    if ll > best_ll {
                        best_ll = ll;
                        sigma = test_sigma;
                        improved = true;
                    }
                }
            }
            
            if !improved {
                break;
            }
        }
        
        Ok((xi, mu, sigma))
    }
    
    /// Compute GEV log-likelihood
    fn gev_log_likelihood(
        &self,
        data: &[f64],
        xi: f64,
        mu: f64,
        sigma: f64,
    ) -> BlockMaximaResult<f64> {
        if sigma <= 0.0 {
            return Err(BlockMaximaError::InvalidParameters { xi, mu, sigma });
        }
        
        let mut ll = 0.0;
        
        for &x in data {
            let z = (x - mu) / sigma;
            
            // Check support condition
            if xi > 0.0 && z <= -1.0 / xi {
                return Ok(f64::NEG_INFINITY);
            }
            if xi < 0.0 && z >= -1.0 / xi {
                return Ok(f64::NEG_INFINITY);
            }
            
            let one_plus_xi_z = 1.0 + xi * z;
            
            if one_plus_xi_z <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            
            // GEV log-density
            let log_term = if xi.abs() < 1e-10 {
                // Gumbel limit
                z + (-z).exp()
            } else {
                -(1.0 / xi).ln_mul_add(one_plus_xi_z.ln(), one_plus_xi_z.powf(-1.0 / xi).ln())
            };
            
            ll -= sigma.ln() + log_term;
        }
        
        Ok(ll)
    }
    
    /// Compute return level for given probability
    pub fn return_level(&self, params: &GEVParameters, p: f64) -> BlockMaximaResult<f64> {
        if p <= 0.0 || p >= 1.0 {
            return Err(BlockMaximaError::InvalidInput(
                "Probability must be in (0, 1)".to_string()
            ));
        }
        
        let y_p = -p.ln(); // Return period
        
        if params.xi.abs() < 1e-10 {
            // Gumbel case
            Ok(params.mu - params.sigma * y_p.ln())
        } else {
            // General GEV
            let result = params.mu + params.sigma * (y_p.powf(-params.xi) - 1.0) / params.xi;
            Ok(result)
        }
    }
    
    /// Compute Expected Shortfall (Conditional Tail Expectation)
    pub fn expected_shortfall(
        &self,
        params: &GEVParameters,
        confidence: f64,
    ) -> BlockMaximaResult<f64> {
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(BlockMaximaError::InvalidInput(
                "Confidence must be in (0, 1)".to_string()
            ));
        }
        
        if params.xi >= 1.0 {
            // ES undefined (infinite mean)
            return Ok(f64::INFINITY);
        }
        
        let var = self.return_level(params, 1.0 - confidence)?;
        
        if params.xi.abs() < 1e-10 {
            // Gumbel case
            let es = var + params.sigma;
            Ok(es)
        } else {
            // General GEV ES formula
            let one_minus_xi = 1.0 - params.xi;
            let p = 1.0 - confidence;
            let y_p = -p.ln();
            
            let term1 = var / one_minus_xi;
            let term2 = params.sigma * (y_p.powf(-params.xi) - 1.0) / (params.xi * one_minus_xi);
            
            Ok(term1 + term2)
        }
    }
}

/// Helper trait for ln_mul_add operation
trait LnMulAdd {
    fn ln_mul_add(self, ln_val: f64, other: f64) -> f64;
}

impl LnMulAdd for f64 {
    fn ln_mul_add(self, ln_val: f64, other: f64) -> f64 {
        self * ln_val + other
    }
}

/// Real-time block maxima monitor for max drawdown tracking
pub struct MaxDrawdownMonitor {
    /// Rolling window of recent block maxima
    history: Vec<f64>,
    /// Maximum history size
    max_history: usize,
    /// Alert threshold for extreme drawdown
    alert_threshold: f64,
    /// Latest GEV fit
    latest_params: Option<GEVParameters>,
}

impl MaxDrawdownMonitor {
    /// Create a new monitor
    pub fn new(max_history: usize, alert_threshold: f64) -> Self {
        Self {
            history: Vec::with_capacity(max_history),
            max_history,
            alert_threshold,
            latest_params: None,
        }
    }
    
    /// Add a new block maximum observation
    pub fn update(&mut self, block_max: f64) {
        self.history.push(block_max);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
    
    /// Refit GEV distribution to current history
    pub fn refit_gev(&mut self, estimator: &BlockMaximaEstimator) -> BlockMaximaResult<&GEVParameters> {
        if self.history.len() < estimator.min_blocks {
            return Err(BlockMaximaError::InsufficientBlocks {
                min: estimator.min_blocks,
                actual: self.history.len(),
            });
        }
        
        let params = estimator.fit_gev_mle(&self.history)?;
        self.latest_params = Some(params);
        Ok(self.latest_params.as_ref().unwrap())
    }
    
    /// Check if current drawdown exceeds alert threshold
    pub fn is_extreme_drawdown(&self, current_drawdown: f64) -> bool {
        current_drawdown > self.alert_threshold
    }
    
    /// Get probability of observing a drawdown at least as extreme
    pub fn exceedance_probability(&self, drawdown: f64) -> Option<f64> {
        let params = self.latest_params.as_ref()?;
        
        let z = (drawdown - params.mu) / params.sigma;
        
        if params.xi.abs() < 1e-10 {
            // Gumbel
            Some((-z).exp())
        } else {
            let one_plus_xi_z = 1.0 + params.xi * z;
            if one_plus_xi_z <= 0.0 {
                Some(1.0)
            } else {
                Some(-one_plus_xi_z.powf(-1.0 / params.xi).exp())
            }
        }
    }
    
    /// Get the estimated 1-in-N block event level
    pub fn return_level_for_period(&self, n_blocks: f64) -> Option<f64> {
        let params = self.latest_params.as_ref()?;
        let p = 1.0 / n_blocks;
        
        // Inverse of survival function
        if params.xi.abs() < 1e-10 {
            Some(params.mu - params.sigma * p.ln())
        } else {
            Some(params.mu + params.sigma * (p.powf(-params.xi) - 1.0) / params.xi)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gev_parameter_creation() {
        let params = GEVParameters::new(0.3, 0.0, 1.0, 100, -150.0, 20).unwrap();
        assert!(params.is_frechet());
        assert_eq!(params.upper_endpoint(), None);
    }
    
    #[test]
    fn test_block_maxima_computation() {
        let estimator = BlockMaximaEstimator::new(10, 20);
        let returns: Vec<f64> = (0..200).map(|i| (i as f64 * 0.01).sin() * 0.05).collect();
        
        let maxima = estimator.compute_block_maxima(&returns, None).unwrap();
        assert_eq!(maxima.len(), 10);
    }
    
    #[test]
    fn test_weibull_upper_endpoint() {
        let params = GEVParameters::new(-0.2, 0.0, 1.0, 100, -150.0, 20).unwrap();
        assert!(params.is_weibull());
        assert!(params.upper_endpoint().is_some());
    }
}
