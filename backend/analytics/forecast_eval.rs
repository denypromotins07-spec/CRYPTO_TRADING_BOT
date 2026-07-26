//! Forecast Evaluation Engine - Diebold-Mariano Test for Model Comparison
//! 
//! This module implements the Diebold-Mariano statistical test to compare
//! forecasting model accuracy and determine if one model significantly
//! outperforms another.
//! 
//! Key Features:
//! - DM test with HAC (Heteroskedasticity and Autocorrelation Consistent) errors
//! - Multi-horizon evaluation
//! - Automatic model disabling when significance is lost
//! - Memory-efficient rolling window computation

use std::collections::VecDeque;

/// Result from Diebold-Mariano test
#[derive(Debug, Clone)]
pub struct DMTestResult {
    /// DM test statistic
    pub dm_statistic: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// Critical value at specified alpha
    pub critical_value: f64,
    /// Whether model 1 significantly outperforms model 2
    pub model_1_better: bool,
    /// Whether model 2 significantly outperforms model 1
    pub model_2_better: bool,
    /// Significance level used
    pub alpha: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Loss differential mean
    pub mean_diff: f64,
    /// Long-run variance estimate
    pub lr_variance: f64,
}

impl DMTestResult {
    /// Check if the test is significant (either direction)
    pub fn is_significant(&self) -> bool {
        self.model_1_better || self.model_2_better
    }
    
    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Diebold-Mariano Test Result\n\
             ===========================\n\
             DM Statistic: {:.4f}\n\
             P-value: {:.4f}\n\
             Alpha: {:.2f}\n\
             Critical Value: ±{:.4f}\n\
             Model 1 Better: {}\n\
             Model 2 Better: {}\n\
             Significant: {}\n\
             Observations: {}",
            self.dm_statistic,
            self.p_value,
            self.alpha,
            self.critical_value,
            self.model_1_better,
            self.model_2_better,
            self.is_significant(),
            self.n_observations
        )
    }
}

/// Loss function types for forecast evaluation
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    /// Squared Error Loss: (y - ŷ)²
    MSE,
    /// Absolute Error Loss: |y - ŷ|
    MAE,
    /// Huber Loss (robust to outliers)
    Huber(f64), // delta parameter
}

impl LossFunction {
    /// Compute loss for a single prediction
    pub fn compute(&self, actual: f64, predicted: f64) -> f64 {
        let error = actual - predicted;
        match self {
            LossFunction::MSE => error * error,
            LossFunction::MAE => error.abs(),
            LossFunction::Huber(delta) => {
                if error.abs() <= *delta {
                    0.5 * error * error
                } else {
                    *delta * (error.abs() - 0.5 * *delta)
                }
            }
        }
    }
}

/// HAC (Newey-West) estimator for long-run variance
struct HACEstimator {
    max_lag: usize,
}

impl HACEstimator {
    fn new(max_lag: usize) -> Self {
        Self { max_lag }
    }
    
    /// Estimate long-run variance using Newey-West HAC estimator
    fn estimate(&self, series: &[f64]) -> f64 {
        let n = series.len();
        if n < 2 {
            return 0.0;
        }
        
        // Mean of series
        let mean = series.iter().sum::<f64>() / n as f64;
        
        // Centered series
        let centered: Vec<f64> = series.iter().map(|&x| x - mean).collect();
        
        // Variance (lag 0)
        let gamma_0: f64 = centered.iter().map(|&x| x * x).sum::<f64>() / n as f64;
        
        // Autocovariances up to max_lag
        let mut lr_var = gamma_0;
        
        for lag in 1..=self.max_lag.min(n - 1) {
            let gamma_k: f64 = centered[..n - lag]
                .iter()
                .zip(centered[lag..].iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>() / n as f64;
            
            // Bartlett kernel weight
            let weight = 1.0 - (lag as f64) / (self.max_lag as f64 + 1.0);
            lr_var += 2.0 * weight * gamma_k;
        }
        
        lr_var.max(0.0) // Ensure non-negative
    }
}

/// Forecast evaluation engine with rolling DM tests
pub struct ForecastEvaluator {
    loss_function: LossFunction,
    alpha: f64,
    max_lag: usize,
    min_observations: usize,
    window_size: Option<usize>,
    results_history: VecDeque<DMTestResult>,
}

impl ForecastEvaluator {
    /// Create new evaluator with default settings
    pub fn new() -> Self {
        Self {
            loss_function: LossFunction::MSE,
            alpha: 0.05,
            max_lag: 10,
            min_observations: 30,
            window_size: None,
            results_history: VecDeque::with_capacity(100),
        }
    }
    
    /// Create with custom loss function
    pub fn with_loss(loss_function: LossFunction) -> Self {
        Self {
            loss_function,
            ..Default::default()
        }
    }
    
    /// Set significance level
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha.clamp(0.001, 0.5);
    }
    
    /// Set rolling window size (None for expanding window)
    pub fn set_window_size(&mut self, size: Option<usize>) {
        self.window_size = size;
    }
    
    /// Perform Diebold-Mariano test comparing two models
    /// 
    /// Null hypothesis: Both models have equal forecast accuracy
    pub fn diebold_mariano(&self, 
                           actuals: &[f64],
                           predictions_1: &[f64],
                           predictions_2: &[f64]) -> DMTestResult {
        let n = actuals.len().min(predictions_1.len()).min(predictions_2.len());
        
        if n < self.min_observations {
            return DMTestResult {
                dm_statistic: 0.0,
                p_value: 1.0,
                critical_value: 0.0,
                model_1_better: false,
                model_2_better: false,
                alpha: self.alpha,
                n_observations: n,
                mean_diff: 0.0,
                lr_variance: 0.0,
            };
        }
        
        // Compute loss differentials: d_t = L1_t - L2_t
        let loss_diff: Vec<f64> = (0..n)
            .map(|i| {
                let l1 = self.loss_function.compute(actuals[i], predictions_1[i]);
                let l2 = self.loss_function.compute(actuals[i], predictions_2[i]);
                l1 - l2
            })
            .collect();
        
        // Mean of loss differential
        let mean_diff = loss_diff.iter().sum::<f64>() / n as f64;
        
        // HAC estimator for long-run variance
        let hac = HACEstimator::new(self.max_lag);
        let lr_variance = hac.estimate(&loss_diff);
        
        // DM statistic: mean(d) / sqrt(var(d)/n)
        let dm_statistic = if lr_variance > 1e-10 {
            mean_diff / (lr_variance / n as f64).sqrt()
        } else {
            0.0
        };
        
        // Two-tailed p-value using normal approximation
        let p_value = 2.0 * (1.0 - self.normal_cdf(dm_statistic.abs()));
        
        // Critical value
        let critical_value = self.normal_quantile(1.0 - self.alpha / 2.0);
        
        // Determine which model is better
        let model_1_better = dm_statistic > critical_value; // L1 < L2 means M1 is better
        let model_2_better = dm_statistic < -critical_value;
        
        DMTestResult {
            dm_statistic,
            p_value,
            critical_value,
            model_1_better,
            model_2_better,
            alpha: self.alpha,
            n_observations: n,
            mean_diff,
            lr_variance,
        }
    }
    
    /// Standard normal CDF approximation (Abramowitz and Stegun)
    fn normal_cdf(&self, x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        
        if x > 0.0 {
            1.0 - prob
        } else {
            prob
        }
    }
    
    /// Standard normal quantile (inverse CDF) approximation
    fn normal_quantile(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }
        
        // Rational approximation
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];
        
        let p_low = 0.02425;
        let p_high = 1.0 - p_low;
        
        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        }
    }
    
    /// Rolling DM test with automatic model tracking
    pub fn rolling_test(&mut self,
                        actuals: &[f64],
                        predictions_1: &[f64],
                        predictions_2: &[f64]) -> Vec<DMTestResult> {
        let n = actuals.len().min(predictions_1.len()).min(predictions_2.len());
        let mut results = Vec::new();
        
        let start = self.min_observations;
        let window = self.window_size.unwrap_or(n);
        
        for end in start..=n {
            let win_start = (end - window).max(start);
            
            let result = self.diebold_mariano(
                &actuals[win_start..end],
                &predictions_1[win_start..end],
                &predictions_2[win_start..end],
            );
            
            results.push(result.clone());
            self.results_history.push_back(result);
            
            // Keep history bounded
            while self.results_history.len() > 100 {
                self.results_history.pop_front();
            }
        }
        
        results
    }
    
    /// Check if a model should be disabled based on recent performance
    pub fn should_disable_model(&self, 
                                 model_id: usize,
                                 threshold_p_value: f64) -> bool {
        // Check last N results for loss of significance
        let recent: Vec<_> = self.results_history.iter().rev().take(10).collect();
        
        if recent.is_empty() {
            return false;
        }
        
        // Count how many recent tests show the model as worse
        let worse_count = recent.iter().filter(|r| {
            if model_id == 1 {
                r.model_2_better && r.p_value < threshold_p_value
            } else {
                r.model_1_better && r.p_value < threshold_p_value
            }
        }).count();
        
        // Disable if significantly worse in majority of recent tests
        worse_count > recent.len() / 2
    }
}

impl Default for ForecastEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dm_test_equal_models() {
        let evaluator = ForecastEvaluator::new();
        
        // Generate identical predictions (should not reject null)
        let actuals: Vec<f64> = (0..100).map(|i| i as f64 + (i % 10) as f64).collect();
        let preds1: Vec<f64> = actuals.iter().map(|&x| x + (x % 5) as f64).collect();
        let preds2 = preds1.clone();
        
        let result = evaluator.diebold_mariano(&actuals, &preds1, &preds2);
        
        assert!(!result.model_1_better);
        assert!(!result.model_2_better);
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_dm_test_different_models() {
        let evaluator = ForecastEvaluator::new();
        
        // Model 1 is clearly better (closer to actuals)
        let actuals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let preds1: Vec<f64> = actuals.iter().map(|&x| x + 1.0).collect(); // Small error
        let preds2: Vec<f64> = actuals.iter().map(|&x| x + 10.0).collect(); // Large error
        
        let result = evaluator.diebold_mariano(&actuals, &preds1, &preds2);
        
        assert!(result.model_1_better); // Model 1 should be better
    }

    #[test]
    fn test_loss_functions() {
        let mse = LossFunction::MSE;
        let mae = LossFunction::MAE;
        let huber = LossFunction::Huber(1.0);
        
        let actual = 10.0;
        let predicted = 12.0;
        let error = 2.0;
        
        assert_eq!(mse.compute(actual, predicted), error * error);
        assert_eq!(mae.compute(actual, predicted), error);
        assert!(huber.compute(actual, predicted) < mse.compute(actual, predicted));
    }
}
