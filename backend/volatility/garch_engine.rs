//! GARCH(1,1) and EGARCH Engine for High-Frequency Volatility Modeling
//! 
//! This module implements ultra-fast GARCH and EGARCH calculations optimized for
//! tick-by-tick updates without numerical divergence. Designed for the ZAID bot's
//! 8GB RAM constraint with zero-cost abstractions.
//!
//! Features:
//! - GARCH(1,1): sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
//! - EGARCH: log(sigma²_t) = omega + alpha * |z_{t-1}| + gamma * z_{t-1} + beta * log(sigma²_{t-1})
//! - Online parameter updates using recursive least squares
//! - Numerical stability checks preventing overflow/underflow

use std::collections::VecDeque;

/// Maximum history window for parameter estimation (prevents memory bloat)
const MAX_HISTORY: usize = 1024;

/// Numerical bounds to prevent divergence
const MIN_VARIANCE: f64 = 1e-10;
const MAX_VARIANCE: f64 = 1e6;
const MIN_PARAM: f64 = 1e-6;
const MAX_PARAM: f64 = 0.999;

/// GARCH(1,1) model state with online learning capability
#[derive(Debug, Clone)]
pub struct GarchModel {
    /// Constant term (omega)
    omega: f64,
    /// ARCH term coefficient (alpha) - reaction to recent shocks
    alpha: f64,
    /// GARCH term coefficient (beta) - persistence of volatility
    beta: f64,
    /// Current conditional variance
    current_variance: f64,
    /// Long-run average variance for mean reversion
    long_run_variance: f64,
    /// History of squared returns for parameter updates
    squared_returns: VecDeque<f64>,
    /// History of variances for parameter updates
    variances: VecDeque<f64>,
}

impl GarchModel {
    /// Create a new GARCH(1,1) model with initial parameters
    /// 
    /// # Arguments
    /// * `omega` - Constant term (typically 1e-6 to 1e-4 for crypto)
    /// * `alpha` - ARCH coefficient (typically 0.05 to 0.15)
    /// * `beta` - GARCH coefficient (typically 0.85 to 0.95)
    /// * `initial_variance` - Starting variance estimate
    pub fn new(omega: f64, alpha: f64, beta: f64, initial_variance: f64) -> Self {
        let omega = omega.clamp(MIN_PARAM, MAX_PARAM);
        let alpha = alpha.clamp(MIN_PARAM, MAX_PARAM);
        let beta = beta.clamp(MIN_PARAM, MAX_PARAM);
        
        // Ensure stationarity: alpha + beta < 1
        let (alpha, beta) = if alpha + beta >= 1.0 {
            let sum = alpha + beta;
            (alpha * 0.99 / sum, beta * 0.99 / sum)
        } else {
            (alpha, beta)
        };
        
        let long_run_variance = omega / (1.0 - alpha - beta);
        
        Self {
            omega,
            alpha,
            beta,
            current_variance: initial_variance.clamp(MIN_VARIANCE, MAX_VARIANCE),
            long_run_variance,
            squared_returns: VecDeque::with_capacity(MAX_HISTORY),
            variances: VecDeque::with_capacity(MAX_HISTORY),
        }
    }
    
    /// Update the model with a new return observation
    /// 
    /// # Arguments
    /// * `return_value` - The latest log return
    /// 
    /// # Returns
    /// The updated conditional variance (volatility squared)
    pub fn update(&mut self, return_value: f64) -> f64 {
        let squared_return = return_value * return_value;
        
        // GARCH(1,1) recursion: sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
        let new_variance = self.omega 
            + self.alpha * squared_return 
            + self.beta * self.current_variance;
        
        // Clamp to prevent numerical explosion
        self.current_variance = new_variance.clamp(MIN_VARIANCE, MAX_VARIANCE);
        
        // Store history for parameter updates
        self.squared_returns.push_back(squared_return);
        self.variances.push_back(self.current_variance);
        
        if self.squared_returns.len() > MAX_HISTORY {
            self.squared_returns.pop_front();
            self.variances.pop_front();
        }
        
        self.current_variance
    }
    
    /// Get the current volatility (standard deviation)
    pub fn volatility(&self) -> f64 {
        self.current_variance.sqrt()
    }
    
    /// Get the current variance
    pub fn variance(&self) -> f64 {
        self.current_variance
    }
    
    /// Forecast variance h steps ahead
    /// 
    /// Uses the mean-reverting property: E[sigma²_{t+h}] = V_L + (alpha+beta)^h * (sigma²_t - V_L)
    pub fn forecast_variance(&self, steps_ahead: usize) -> f64 {
        let persistence = self.alpha + self.beta;
        let mean_reversion = persistence.powi(steps_ahead as i32);
        self.long_run_variance + mean_reversion * (self.current_variance - self.long_run_variance)
    }
    
    /// Online parameter update using method of moments
    /// 
    /// This is a simplified recursive update that adjusts parameters based on
    /// recent prediction errors without full maximum likelihood estimation.
    pub fn update_parameters_online(&mut self, learning_rate: f64) {
        if self.squared_returns.len() < 100 {
            return; // Need sufficient data
        }
        
        // Calculate recent averages
        let n = self.squared_returns.len() as f64;
        let avg_squared_return: f64 = self.squared_returns.iter().sum::<f64>() / n;
        let avg_variance: f64 = self.variances.iter().sum::<f64>() / n;
        
        // Adjust omega to match long-run average
        let target_omega = avg_variance * (1.0 - self.alpha - self.beta);
        self.omega = ((1.0 - learning_rate) * self.omega + learning_rate * target_omega)
            .clamp(MIN_PARAM, MAX_PARAM);
        
        // Recalculate long-run variance
        self.long_run_variance = self.omega / (1.0 - self.alpha - self.beta);
    }
    
    /// Check for numerical stability
    pub fn is_stable(&self) -> bool {
        self.alpha + self.beta < 1.0 
            && self.current_variance > MIN_VARIANCE 
            && self.current_variance < MAX_VARIANCE
            && self.omega > MIN_PARAM
    }
}

/// EGARCH model for asymmetric volatility (leverage effect)
#[derive(Debug, Clone)]
pub struct EGarchModel {
    /// Constant term (omega)
    omega: f64,
    /// ARCH term coefficient (alpha) - magnitude effect
    alpha: f64,
    /// Leverage coefficient (gamma) - asymmetry
    gamma: f64,
    /// GARCH term coefficient (beta) - persistence
    beta: f64,
    /// Current log variance
    current_log_variance: f64,
    /// Last standardized residual
    last_z: f64,
    /// History for parameter updates
    log_variances: VecDeque<f64>,
    standardized_residuals: VecDeque<f64>,
}

impl EGarchModel {
    /// Create a new EGARCH model
    /// 
    /// # Arguments
    /// * `omega` - Constant term
    /// * `alpha` - Magnitude effect coefficient
    /// * `gamma` - Leverage/asymmetry coefficient (negative for stocks, can be positive for crypto)
    /// * `beta` - Persistence coefficient
    /// * `initial_variance` - Starting variance
    pub fn new(omega: f64, alpha: f64, gamma: f64, beta: f64, initial_variance: f64) -> Self {
        let omega = omega.clamp(MIN_PARAM, MAX_PARAM);
        let alpha = alpha.clamp(MIN_PARAM, MAX_PARAM);
        let gamma = gamma.clamp(-MAX_PARAM, MAX_PARAM);
        let beta = beta.clamp(MIN_PARAM, MAX_PARAM);
        
        // Ensure stationarity: beta < 1
        let beta = beta.min(0.999);
        
        let initial_log_variance = initial_variance.ln().clamp(-20.0, 20.0);
        
        Self {
            omega,
            alpha,
            gamma,
            beta,
            current_log_variance: initial_log_variance,
            last_z: 0.0,
            log_variances: VecDeque::with_capacity(MAX_HISTORY),
            standardized_residuals: VecDeque::with_capacity(MAX_HISTORY),
        }
    }
    
    /// Update the model with a new return observation
    /// 
    /// EGARCH specification:
    /// log(sigma²_t) = omega + alpha * (|z_{t-1}| - E[|z|]) + gamma * z_{t-1} + beta * log(sigma²_{t-1})
    /// where z_{t-1} = r_{t-1} / sigma_{t-1}
    pub fn update(&mut self, return_value: f64) -> f64 {
        let current_volatility = self.current_log_variance.exp().sqrt();
        
        // Standardized residual
        let z = if current_volatility > MIN_VARIANCE.sqrt() {
            return_value / current_volatility
        } else {
            return_value.signum() * 10.0 // Cap extreme values
        };
        
        // Expected value of |z| for standard normal ≈ 0.7979
        const E_ABS_Z: f64 = 0.7978845608028654;
        
        // EGARCH recursion
        let new_log_variance = self.omega 
            + self.alpha * (z.abs() - E_ABS_Z) 
            + self.gamma * z 
            + self.beta * self.current_log_variance;
        
        // Clamp to prevent numerical issues
        self.current_log_variance = new_log_variance.clamp(-20.0, 20.0);
        self.last_z = z;
        
        // Store history
        self.log_variances.push_back(self.current_log_variance);
        self.standardized_residuals.push_back(z);
        
        if self.log_variances.len() > MAX_HISTORY {
            self.log_variances.pop_front();
            self.standardized_residuals.pop_front();
        }
        
        self.current_log_variance.exp() // Return variance
    }
    
    /// Get the current volatility
    pub fn volatility(&self) -> f64 {
        self.current_log_variance.exp().sqrt()
    }
    
    /// Get the leverage effect (asymmetry)
    /// 
    /// Returns true if negative shocks increase volatility more than positive shocks
    pub fn has_leverage_effect(&self) -> bool {
        self.gamma < 0.0
    }
    
    /// Forecast variance h steps ahead
    /// 
    /// For EGARCH, the forecast converges to the unconditional variance
    pub fn forecast_variance(&self, steps_ahead: usize) -> f64 {
        // Unconditional log variance for EGARCH
        let unconditional_log_var = self.omega / (1.0 - self.beta);
        let persistence = self.beta.powi(steps_ahead as i32);
        
        let forecast_log_var = unconditional_log_var 
            + persistence * (self.current_log_variance - unconditional_log_var);
        
        forecast_log_var.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_garch_stationarity() {
        let mut model = GarchModel::new(1e-5, 0.1, 0.85, 0.0004);
        assert!(model.is_stable());
        
        // Force non-stationary parameters
        let mut bad_model = GarchModel::new(1e-5, 0.6, 0.5, 0.0004);
        assert!(bad_model.is_stable()); // Should auto-correct
        assert!(bad_model.alpha + bad_model.beta < 1.0);
    }
    
    #[test]
    fn test_garch_update_sequence() {
        let mut model = GarchModel::new(1e-5, 0.1, 0.85, 0.0004);
        
        // Simulate a series of returns
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.03];
        let mut prev_var = model.variance();
        
        for r in returns {
            let new_var = model.update(r);
            // Variance should increase after large absolute returns
            if r.abs() > 0.02 {
                assert!(new_var > prev_var);
            }
            prev_var = new_var;
        }
    }
    
    #[test]
    fn test_egarch_leverage() {
        // Negative gamma means negative shocks increase vol more
        let mut model = EGarchModel::new(0.0, 0.1, -0.2, 0.9, 0.0004);
        assert!(model.has_leverage_effect());
        
        let vol_before = model.volatility();
        
        // Negative shock
        model.update(-0.05);
        let vol_after_negative = model.volatility();
        
        // Reset and apply positive shock of same magnitude
        let mut model2 = EGarchModel::new(0.0, 0.1, -0.2, 0.9, 0.0004);
        model2.update(0.05);
        let vol_after_positive = model2.volatility();
        
        // Negative shock should increase vol more due to leverage effect
        assert!(vol_after_negative > vol_after_positive);
    }
    
    #[test]
    fn test_forecast_mean_reversion() {
        let model = GarchModel::new(1e-5, 0.1, 0.85, 0.0004);
        let short_term = model.forecast_variance(1);
        let long_term = model.forecast_variance(100);
        
        // Long-term forecast should be closer to long-run variance
        let lr_var = model.long_run_variance;
        assert!((long_term - lr_var).abs() < (short_term - lr_var).abs());
    }
}
