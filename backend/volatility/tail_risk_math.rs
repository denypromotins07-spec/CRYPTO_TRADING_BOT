//! Tail Risk Mathematics: Extreme Value Theory and Pareto Tails
//!
//! This module implements Extreme Value Theory (EVT) for modeling tail risk
//! in cryptocurrency returns. Essential for accurate VaR/CVaR estimation
//! and understanding extreme market movements beyond normal distribution assumptions.
//!
//! Features:
//! - Generalized Pareto Distribution (GPD) fitting
//! - Peaks Over Threshold (POT) method
//! - Hill estimator for tail index
//! - Return level calculations
//! - Tail risk metrics (VaR, CVaR/Expected Shortfall)
//!
//! Optimized for zero-cost abstractions and numerical stability.

use std::f64::consts::PI;

/// Numerical constants for stability
const MIN_TAIL_INDEX: f64 = 0.1;
const MAX_TAIL_INDEX: f64 = 10.0;
const MIN_THRESHOLD_PROB: f64 = 0.90;
const MAX_THRESHOLD_PROB: f64 = 0.999;

/// Result of GPD parameter estimation
#[derive(Debug, Clone)]
pub struct GpdFit {
    /// Shape parameter (xi) - tail index
    /// xi > 0 indicates heavy tails (Fréchet domain)
    /// xi = 0 indicates exponential tails (Gumbel domain)
    /// xi < 0 indicates bounded tails (Weibull domain)
    pub shape: f64,
    
    /// Scale parameter (sigma) - spread of exceedances
    pub scale: f64,
    
    /// Threshold used for fitting
    pub threshold: f64,
    
    /// Number of exceedances above threshold
    pub n_exceedances: usize,
    
    /// Standard error of shape estimate
    pub shape_se: f64,
    
    /// Goodness of fit statistic (Anderson-Darling)
    pub ad_statistic: f64,
}

impl GpdFit {
    /// Check if the fit indicates heavy tails
    pub fn is_heavy_tailed(&self) -> bool {
        self.shape > 0.0
    }
    
    /// Check if variance is finite (requires xi < 0.5)
    pub fn has_finite_variance(&self) -> bool {
        self.shape < 0.5
    }
    
    /// Check if mean is finite (requires xi < 1)
    pub fn has_finite_mean(&self) -> bool {
        self.shape < 1.0
    }
}

/// Generalized Pareto Distribution utilities
pub struct GeneralizedPareto;

impl GeneralizedPareto {
    /// Probability density function of GPD
    /// 
    /// f(x) = (1/sigma) * (1 + xi * x / sigma)^(-1/xi - 1)
    /// 
    /// # Arguments
    /// * `x` - Value (exceedance above threshold)
    /// * `shape` - Shape parameter (xi)
    /// * `scale` - Scale parameter (sigma)
    pub fn pdf(x: f64, shape: f64, scale: f64) -> f64 {
        if x < 0.0 || scale <= 0.0 {
            return 0.0;
        }
        
        let z = 1.0 + shape * x / scale;
        if z <= 0.0 {
            return 0.0;
        }
        
        if shape.abs() < 1e-10 {
            // Limit as xi -> 0 (exponential distribution)
            return (-x / scale).exp() / scale;
        }
        
        z.powf(-1.0 / shape - 1.0) / scale
    }
    
    /// Cumulative distribution function of GPD
    /// 
    /// F(x) = 1 - (1 + xi * x / sigma)^(-1/xi)
    pub fn cdf(x: f64, shape: f64, scale: f64) -> f64 {
        if x < 0.0 || scale <= 0.0 {
            return 0.0;
        }
        
        let z = 1.0 + shape * x / scale;
        if z <= 0.0 {
            return 1.0;
        }
        
        if shape.abs() < 1e-10 {
            // Limit as xi -> 0
            return 1.0 - (-x / scale).exp();
        }
        
        1.0 - z.powf(-1.0 / shape)
    }
    
    /// Quantile function (inverse CDF) of GPD
    /// 
    /// Q(p) = (sigma / xi) * ((1 - p)^(-xi) - 1)
    pub fn quantile(p: f64, shape: f64, scale: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 || scale <= 0.0 {
            return f64::NAN;
        }
        
        if shape.abs() < 1e-10 {
            return -scale * (1.0 - p).ln();
        }
        
        scale / shape * ((1.0 - p).powf(-shape) - 1.0)
    }
    
    /// Calculate Value at Risk (VaR) at confidence level alpha
    /// 
    /// For POT method: VaR = u + GPD_quantile((alpha - F(u)) / (1 - F(u)))
    /// 
    /// # Arguments
    /// * `threshold` - Threshold u
    /// * `prob_above` - Proportion of data above threshold
    /// * `alpha` - Confidence level (e.g., 0.99 for 99% VaR)
    pub fn var(threshold: f64, prob_above: f64, alpha: f64, shape: f64, scale: f64) -> f64 {
        if alpha <= prob_above || alpha >= 1.0 {
            return f64::NAN;
        }
        
        // Conditional probability in the tail
        let cond_prob = (alpha - prob_above) / (1.0 - prob_above);
        
        threshold + Self::quantile(cond_prob, shape, scale)
    }
    
    /// Calculate Expected Shortfall (CVaR) at confidence level alpha
    /// 
    /// ES = VaR / (1 - xi) + (scale - xi * VaR) / (1 - xi)
    /// 
    /// Only valid for xi < 1 (finite mean)
    pub fn expected_shortfall(var: f64, threshold: f64, shape: f64, scale: f64, alpha: f64) -> f64 {
        if shape >= 1.0 || shape <= 0.0 {
            return f64::NAN;
        }
        
        let excess_var = var - threshold;
        
        // ES formula for GPD
        (var + scale - shape * excess_var) / (1.0 - shape)
    }
    
    /// Calculate return level for return period T
    /// 
    /// The level expected to be exceeded once every T observations
    pub fn return_level(threshold: f64, prob_above: f64, t: f64, shape: f64, scale: f64) -> f64 {
        if t <= 0.0 {
            return f64::NAN;
        }
        
        // Exceedance probability for return period T
        let p = 1.0 / (t * prob_above);
        
        if p >= 1.0 || p <= 0.0 {
            return f64::NAN;
        }
        
        threshold + Self::quantile(1.0 - p, shape, scale)
    }
}

/// Hill Estimator for tail index
pub struct HillEstimator;

impl HillEstimator {
    /// Calculate Hill estimator for tail index
    /// 
    /// gamma_H = (1/k) * sum_{i=1}^{k} log(X_{(i)} / X_{(k+1)})
    /// 
    /// where X_{(i)} are the k largest order statistics
    /// 
    /// # Arguments
    /// * `sorted_data` - Data sorted in descending order
    /// * `k` - Number of top order statistics to use
    /// 
    /// # Returns
    /// Tail index estimate (gamma_H), which equals 1/xi
    pub fn estimate(sorted_data: &[f64], k: usize) -> Option<f64> {
        if k == 0 || k >= sorted_data.len() {
            return None;
        }
        
        let threshold = sorted_data[k]; // (k+1)-th largest
        
        let sum_logs: f64 = sorted_data[..k]
            .iter()
            .filter(|&&x| x > threshold && threshold > 0.0)
            .map(|&x| (x / threshold).ln())
            .sum();
        
        Some(sum_logs / k as f64)
    }
    
    /// Calculate optimal k using bootstrap or heuristic methods
    /// 
    /// Uses the "square root rule" as default: k ≈ sqrt(n)
    pub fn optimal_k(n: usize) -> usize {
        (n as f64).sqrt() as usize
    }
    
    /// Asymptotic standard error of Hill estimator
    pub fn standard_error(hill_estimate: f64, k: usize) -> f64 {
        hill_estimate / (k as f64).sqrt()
    }
}

/// Peaks Over Threshold (POT) analysis
pub struct PotAnalysis {
    data: Vec<f64>,
    threshold: f64,
    exceedances: Vec<f64>,
}

impl PotAnalysis {
    /// Create new POT analysis
    /// 
    /// # Arguments
    /// * `data` - Full dataset (typically negative returns for left tail)
    /// * `threshold` - Threshold for exceedances (can be percentile-based)
    pub fn new(data: Vec<f64>, threshold: f64) -> Self {
        let exceedances: Vec<f64> = data
            .iter()
            .filter(|&&x| x > threshold)
            .map(|&x| x - threshold)
            .collect();
        
        Self {
            data,
            threshold,
            exceedances,
        }
    }
    
    /// Create POT analysis with threshold based on quantile
    /// 
    /// # Arguments
    /// * `data` - Full dataset
    /// * `quantile` - Quantile for threshold (e.g., 0.95 for 95th percentile)
    pub fn from_quantile(mut data: Vec<f64>, quantile: f64) -> Option<Self> {
        if quantile < MIN_THRESHOLD_PROB || quantile > MAX_THRESHOLD_PROB {
            return None;
        }
        
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((data.len() as f64) * quantile) as usize;
        let threshold = *data.get(idx)?;
        
        Some(Self::new(data, threshold))
    }
    
    /// Fit GPD to exceedances using Maximum Likelihood Estimation
    pub fn fit_gpd(&self) -> Option<GpdFit> {
        if self.exceedances.len() < 10 {
            return None;
        }
        
        // Method of moments initial estimates
        let mean_exc: f64 = self.exceedances.iter().sum::<f64>() / self.exceedances.len() as f64;
        let var_exc: f64 = self.exceedances.iter()
            .map(|&x| (x - mean_exc).powi(2))
            .sum::<f64>() / self.exceedances.len() as f64;
        
        // Initial shape estimate from coefficient of variation
        let cv = var_exc.sqrt() / mean_exc;
        let shape_init = (cv - 1.0) / (cv + 1.0);
        let shape_init = shape_init.clamp(MIN_TAIL_INDEX, MAX_TAIL_INDEX);
        
        // Initial scale estimate
        let scale_init = mean_exc * (1.0 - shape_init);
        
        // Simple iterative refinement (for production, use proper MLE optimization)
        let (shape, scale) = self.refine_mle(shape_init, scale_init, 100);
        
        // Calculate standard error (simplified)
        let shape_se = shape / (self.exceedances.len() as f64).sqrt();
        
        // Anderson-Darling statistic for goodness of fit
        let ad_stat = self.anderson_darling(shape, scale);
        
        Some(GpdFit {
            shape,
            scale,
            threshold: self.threshold,
            n_exceedances: self.exceedances.len(),
            shape_se,
            ad_statistic: ad_stat,
        })
    }
    
    /// Refine MLE estimates using iterative method
    fn refine_mle(&self, shape_init: f64, scale_init: f64, max_iter: usize) -> (f64, f64) {
        let mut shape = shape_init;
        let mut scale = scale_init;
        
        for _ in 0..max_iter {
            // Update scale given shape
            let new_scale = shape * self.exceedances.iter()
                .map(|&x| {
                    let z = 1.0 + shape * x / scale;
                    if z <= 0.0 { return 0.0; }
                    x / z
                })
                .sum::<f64>() / self.exceedances.len() as f64;
            
            // Update shape given scale (simplified Newton step)
            let sum_log_z: f64 = self.exceedances.iter()
                .map(|&x| {
                    let z = 1.0 + shape * x / new_scale;
                    if z <= 0.0 { return 0.0; }
                    z.ln()
                })
                .sum();
            
            let new_shape = shape + (1.0 / self.exceedances.len() as f64) * sum_log_z;
            
            // Check convergence
            if (new_shape - shape).abs() < 1e-6 && (new_scale - scale).abs() < 1e-6 {
                break;
            }
            
            shape = new_shape.clamp(MIN_TAIL_INDEX, MAX_TAIL_INDEX);
            scale = new_scale.max(1e-10);
        }
        
        (shape, scale)
    }
    
    /// Calculate Anderson-Darling statistic for GPD fit
    fn anderson_darling(&self, shape: f64, scale: f64) -> f64 {
        let n = self.exceedances.len() as f64;
        let mut sorted_exc = self.exceedances.clone();
        sorted_exc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut ad = 0.0;
        for (i, &x) in sorted_exc.iter().enumerate() {
            let cdf_val = GeneralizedPareto::cdf(x, shape, scale);
            let i_f64 = (i + 1) as f64;
            
            let term1 = -(2.0 * i_f64 - 1.0) / n * cdf_val.ln();
            let term2 = -(2.0 * (n - i_f64) + 1.0) / n * (1.0 - cdf_val).ln();
            
            ad += term1 + term2;
        }
        
        ad - n
    }
    
    /// Get proportion of data above threshold
    pub fn prob_above_threshold(&self) -> f64 {
        self.exceedances.len() as f64 / self.data.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpd_basic() {
        // Test basic GPD functions
        let shape = 0.3;
        let scale = 1.0;
        
        // PDF should be positive
        assert!(GeneralizedPareto::pdf(0.5, shape, scale) > 0.0);
        
        // CDF should be increasing
        let cdf1 = GeneralizedPareto::cdf(0.5, shape, scale);
        let cdf2 = GeneralizedPareto::cdf(1.0, shape, scale);
        assert!(cdf2 > cdf1);
        
        // Quantile should invert CDF
        let q = GeneralizedPareto::quantile(0.5, shape, scale);
        let cdf_at_q = GeneralizedPareto::cdf(q, shape, scale);
        assert!((cdf_at_q - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_hill_estimator() {
        // Generate Pareto-distributed data (known tail index)
        let mut data: Vec<f64> = (0..1000)
            .map(|i| {
                let u = (i as f64 + 1.0) / 1001.0;
                (1.0 - u).powf(-1.0 / 2.5) // Tail index = 2.5
            })
            .collect();
        
        data.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending
        
        let k = HillEstimator::optimal_k(data.len());
        let hill = HillEstimator::estimate(&data, k);
        
        assert!(hill.is_some());
        let hill_val = hill.unwrap();
        
        // Hill estimator gives 1/xi, so should be close to 2.5
        assert!(hill_val > 1.5 && hill_val < 3.5);
    }
    
    #[test]
    fn test_pot_analysis() {
        // Generate some heavy-tailed data
        let mut data: Vec<f64> = (0..500)
            .map(|_| {
                // Mixture: mostly normal with some extreme values
                if rand_bool() {
                    rand_normal() * 0.01
                } else {
                    rand_normal() * 0.05
                }
            })
            .collect();
        
        let pot = PotAnalysis::from_quantile(data, 0.95);
        assert!(pot.is_some());
        
        let pot = pot.unwrap();
        assert!(pot.exceedances.len() > 0);
        assert!(pot.exceedances.len() < 50); // ~5% of 500
    }
    
    fn rand_bool() -> bool {
        // Simple deterministic "random" for testing
        static mut COUNTER: usize = 0;
        unsafe {
            COUNTER += 1;
            COUNTER % 10 != 0 // 90% true, 10% false
        }
    }
    
    fn rand_normal() -> f64 {
        // Box-Muller approximation
        use std::f64::consts::PI;
        static mut COUNTER: usize = 0;
        unsafe {
            COUNTER += 1;
            let u1 = (COUNTER as f64 * 0.1) % 1.0;
            let u2 = (COUNTER as f64 * 0.3) % 1.0;
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }
    
    #[test]
    fn test_tail_properties() {
        // Heavy-tailed distribution (xi > 0)
        let heavy_fit = GpdFit {
            shape: 0.4,
            scale: 1.0,
            threshold: 0.0,
            n_exceedances: 100,
            shape_se: 0.04,
            ad_statistic: 0.5,
        };
        
        assert!(heavy_fit.is_heavy_tailed());
        assert!(heavy_fit.has_finite_variance()); // xi < 0.5
        assert!(heavy_fit.has_finite_mean()); // xi < 1
        
        // Very heavy-tailed (infinite variance)
        let very_heavy = GpdFit {
            shape: 0.7,
            ..heavy_fit.clone()
        };
        
        assert!(very_heavy.is_heavy_tailed());
        assert!(!very_heavy.has_finite_variance()); // xi > 0.5
        assert!(very_heavy.has_finite_mean()); // xi < 1
    }
}
