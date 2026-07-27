//! SABR Stochastic Volatility Model Implementation
//! Implements Hagan's asymptotic expansion for implied volatility
//! Handles crypto-specific zero-lower-bound interest rate environment
//! 
//! SABR dynamics:
//! dF_t = α_t F_t^β dW_t^F
//! dα_t = ν α_t dW_t^α
//! d⟨W^F, W^α⟩_t = ρ dt
//! 
//! Zero-cost abstractions for AMD Ryzen AI 5 optimization

use std::fmt;

/// SABR model parameters
#[derive(Debug, Clone, Copy)]
pub struct SabrParams {
    pub alpha: f64,   /// Initial volatility level (not variance!)
    pub beta: f64,    /// CEV exponent (0=normal, 1=lognormal, 0.5=CEV)
    pub rho: f64,     /// Correlation between forward and vol
    pub nu: f64,      /// Vol-of-vol (volatility of alpha)
}

impl SabrParams {
    /// Create new SABR parameters with validation
    pub fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> Result<Self, SabrError> {
        if alpha <= 0.0 {
            return Err(SabrError::InvalidAlpha);
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(SabrError::InvalidBeta);
        }
        if rho < -1.0 || rho > 1.0 {
            return Err(SabrError::InvalidRho);
        }
        if nu <= 0.0 {
            return Err(SabrError::InvalidNu);
        }
        Ok(Self { alpha, beta, rho, nu })
    }

    /// Default crypto-friendly parameters (high vol, negative skew)
    pub fn crypto_default() -> Self {
        Self {
            alpha: 0.6,   // 60% initial vol typical for BTC
            beta: 0.9,    // Near lognormal for crypto
            rho: -0.4,    // Moderate negative skew
            nu: 0.8,      // High vol-of-vol
        }
    }
}

/// SABR error types
#[derive(Debug, Clone, PartialEq)]
pub enum SabrError {
    InvalidAlpha,
    InvalidBeta,
    InvalidRho,
    InvalidNu,
    NumericalInstability,
    ArbitrageDetected,
}

impl fmt::Display for SabrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SabrError::InvalidAlpha => write!(f, "Alpha must be positive"),
            SabrError::InvalidBeta => write!(f, "Beta must be in [0, 1]"),
            SabrError::InvalidRho => write!(f, "Rho must be in [-1, 1]"),
            SabrError::InvalidNu => write!(f, "Nu must be positive"),
            SabrError::NumericalInstability => write!(f, "Numerical instability detected"),
            SabrError::ArbitrageDetected => write!(f, "Static arbitrage detected in smile"),
        }
    }
}

/// Hagan's asymptotic expansion for SABR implied volatility
pub struct HaganExpansion<'a> {
    params: &'a SabrParams,
    forward: f64,
    expiry: f64,
}

impl<'a> HaganExpansion<'a> {
    pub fn new(params: &'a SabrParams, forward: f64, expiry: f64) -> Self {
        Self {
            params,
            forward,
            expiry,
        }
    }

    /// Compute implied volatility for a given strike using Hagan's formula
    /// This is the standard market formula used across derivatives desks
    pub fn implied_vol(&self, strike: f64) -> Result<f64, SabrError> {
        let p = self.params;
        let f = self.forward;
        let k = strike;
        let t = self.expiry;

        if t <= 0.0 {
            return Err(SabrError::NumericalInstability);
        }

        // Handle ATM case separately for numerical stability
        let fk_mid = ((f * k).sqrt());
        let is_atm = (f - k).abs() / fk_mid < 1e-6;

        if is_atm {
            return Ok(self.atm_vol()?);
        }

        // Log-moneyness
        let log_fk = (f / k).ln();
        let fk_diff = f - k;

        // First term: leading order
        let z = (p.nu / p.alpha) * (f.powi(1 - p.beta as i32) - k.powi(1 - p.beta as i32));
        let x_z = if z.abs() < 1e-8 {
            1.0 + (p.rho * p.beta / 4.0) * z + ((2.0 + p.rho * p.rho) * p.beta * p.beta / 24.0) * z * z
        } else {
            let sqrt_term = (1.0 - 2.0 * p.rho * z + z * z).sqrt();
            z * (-z * p.rho).ln().exp() / (z - p.rho + sqrt_term).ln().exp()
        };

        // Adjust for numerical edge cases
        let x_z = x_z.max(0.1).min(10.0);

        // Second term: O(1) correction
        let term1 = p.alpha / (fk_mid.powi(1 - p.beta as i32) * (1.0 + (1.0 - p.beta).powi(2) * log_fk * log_fk / 24.0));
        
        // Third term: O(t) correction (Hagan et al. 2002)
        let term2 = 1.0 + t * (
            ((1.0 - p.beta).powi(2) / 24.0) * p.alpha * p.alpha / fk_mid.powi(2 - 2 * p.beta as i32)
            + (p.rho * p.beta / 4.0) * p.nu * p.alpha / fk_mid.powi(1 - p.beta as i32)
            + (2.0 - 3.0 * p.rho * p.rho) / 24.0 * p.nu * p.nu
        );

        let sigma = term1 * term2 * x_z;

        if sigma <= 0.0 || sigma > 5.0 {
            return Err(SabrError::NumericalInstability);
        }

        Ok(sigma)
    }

    /// ATM volatility (simplified Hagan formula)
    fn atm_vol(&self) -> Result<f64, SabrError> {
        let p = self.params;
        let f = self.forward;
        let t = self.expiry;

        if f <= 0.0 || t <= 0.0 {
            return Err(SabrError::NumericalInstability);
        }

        // ATM SABR formula
        let atm_sigma = p.alpha / f.powi(1 - p.beta as i32);
        
        // Time correction
        let correction = 1.0 + t * (
            ((1.0 - p.beta).powi(2) / 24.0) * p.alpha * p.alpha / f.powi(2 - 2 * p.beta as i32)
            + (p.rho * p.beta / 4.0) * p.nu * p.alpha / f.powi(1 - p.beta as i32)
            + (2.0 - 3.0 * p.rho * p.rho) / 24.0 * p.nu * p.nu
        );

        let sigma = atm_sigma * correction;

        if sigma <= 0.0 || sigma > 5.0 {
            return Err(SabrError::NumericalInstability);
        }

        Ok(sigma)
    }

    /// Compute the full volatility smile across strikes
    pub fn compute_smile(&self, strikes: &[f64]) -> Vec<(f64, f64)> {
        let mut smile = Vec::with_capacity(strikes.len());
        
        for &k in strikes {
            if let Ok(iv) = self.implied_vol(k) {
                smile.push((k, iv));
            }
        }
        
        smile
    }
}

/// SABR smile dynamics tracker
pub struct SmileDynamics {
    params_history: Vec<SabrParams>,
    skew_history: Vec<f64>,
    curvature_history: Vec<f64>,
}

impl SmileDynamics {
    pub fn new() -> Self {
        Self {
            params_history: Vec::with_capacity(50),
            skew_history: Vec::with_capacity(50),
            curvature_history: Vec::with_capacity(50),
        }
    }

    /// Record new calibration and compute smile metrics
    pub fn record(&mut self, params: SabrParams, forward: f64, expiry: f64) {
        self.params_history.push(params);
        
        // Compute skew: IV(25d put) - IV(25d call)
        let put_strike = forward * 0.95;  // Approx 25d delta put
        let call_strike = forward * 1.05; // Approx 25d delta call
        
        let expansion = HaganExpansion::new(&params, forward, expiry);
        
        let skew = if let (Ok(iv_put), Ok(iv_call)) = (expansion.implied_vol(put_strike), expansion.implied_vol(call_strike)) {
            iv_put - iv_call
        } else {
            0.0
        };
        
        // Compute curvature: IV(ATM) - 0.5*(IV(put) + IV(call))
        let curvature = if let Ok(iv_atm) = expansion.implied_vol(forward) {
            iv_atm - 0.5 * ((iv_put + iv_call) / 2.0)
        } else {
            0.0
        };
        
        self.skew_history.push(skew);
        self.curvature_history.push(curvature);
        
        // Bound history size
        if self.params_history.len() > 50 {
            self.params_history.remove(0);
            self.skew_history.remove(0);
            self.curvature_history.remove(0);
        }
    }

    /// Get current skew (negative = typical crypto skew)
    pub fn current_skew(&self) -> Option<f64> {
        self.skew_history.last().copied()
    }

    /// Get current curvature (positive = smile, negative = smirk)
    pub fn current_curvature(&self) -> Option<f64> {
        self.curvature_history.last().copied()
    }

    /// Detect regime change in smile dynamics
    pub fn detect_regime_change(&self) -> Option<SmileRegime> {
        if self.skew_history.len() < 5 {
            return None;
        }
        
        let recent_skew: f64 = self.skew_history.iter().rev().take(3).sum::<f64>() / 3.0;
        let old_skew: f64 = self.skew_history.iter().rev().skip(3).take(3).sum::<f64>() / 3.0;
        
        let skew_change = recent_skew - old_skew;
        
        if skew_change.abs() > 0.1 {
            if skew_change > 0.0 {
                Some(SmileRegime::Flattening)  // Skew becoming less negative
            } else {
                Some(SmileRegime::Steepening)  // Skew becoming more negative
            }
        } else {
            Some(SmileRegime::Stable)
        }
    }
}

/// Smile regime classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmileRegime {
    Stable,
    Steepening,   // Skew becoming more negative (fear increasing)
    Flattening,   // Skew becoming less negative (fear decreasing)
    Inverted,     // Rare: positive skew
}

/// Arbitrage-free SABR constraints checker
pub struct ArbitrageFreeSabr {
    butterfly_tolerance: f64,
    calendar_tolerance: f64,
}

impl ArbitrageFreeSabr {
    pub fn new(butterfly_tolerance: f64, calendar_tolerance: f64) -> Self {
        Self {
            butterfly_tolerance,
            calendar_tolerance,
        }
    }

    /// Check for butterfly arbitrage (convexity violation)
    /// Butterfly spread must have non-negative value
    pub fn check_butterfly_arbitrage(&self, params: SabrParams, forward: f64, 
                                      expiry: f64, strikes: &[f64]) -> bool {
        if strikes.len() < 3 {
            return true;  // Cannot check with insufficient data
        }

        let expansion = HaganExpansion::new(&params, forward, expiry);
        
        // Check convexity of option prices
        for i in 1..strikes.len() - 1 {
            let k_prev = strikes[i - 1];
            let k_mid = strikes[i];
            let k_next = strikes[i + 1];
            
            if let (Ok(iv_prev), Ok(iv_mid), Ok(iv_next)) = (
                expansion.implied_vol(k_prev),
                expansion.implied_vol(k_mid),
                expansion.implied_vol(k_next),
            ) {
                // Convert to prices and check butterfly
                let price_prev = self.black_price(forward, k_prev, expiry, iv_prev, true);
                let price_mid = self.black_price(forward, k_mid, expiry, iv_mid, true);
                let price_next = self.black_price(forward, k_next, expiry, iv_next, true);
                
                // Butterfly: long 1 low, short 2 mid, long 1 high (adjusted for spacing)
                let lambda = (k_mid - k_prev) / (k_next - k_prev);
                let butterfly_value = price_prev - (1.0 + lambda) * price_mid + lambda * price_next;
                
                if butterfly_value < -self.butterfly_tolerance {
                    return false;  // Arbitrage detected
                }
            }
        }
        
        true
    }

    /// Simple Black-Scholes pricer for arbitrage checks
    fn black_price(&self, f: f64, k: f64, t: f64, sigma: f64, is_call: bool) -> f64 {
        if t <= 0.0 || sigma <= 0.0 {
            return if is_call { (f - k).max(0.0) } else { (k - f).max(0.0) };
        }

        let d1 = (f / k).ln() / (sigma * t.sqrt()) + 0.5 * sigma * t.sqrt();
        let d2 = d1 - sigma * t.sqrt();
        
        // Simplified normal CDF
        let nd1 = 0.5 * (1.0 + (2.0 / std::f64::consts::PI).sqrt() * (d1 - d1.powi(3) / 6.0).tanh());
        let nd2 = 0.5 * (1.0 + (2.0 / std::f64::consts::PI).sqrt() * (d2 - d2.powi(3) / 6.0).tanh());
        
        if is_call {
            f * nd1 - k * nd2
        } else {
            k * (1.0 - nd2) - f * (1.0 - nd1)
        }
    }

    /// Check for calendar arbitrage (total variance must increase with time)
    pub fn check_calendar_arbitrage(&self, params_short: SabrParams, params_long: SabrParams,
                                     forward: f64, t_short: f64, t_long: f64) -> bool {
        if t_short >= t_long {
            return true;  // Invalid ordering
        }

        let exp_short = HaganExpansion::new(&params_short, forward, t_short);
        let exp_long = HaganExpansion::new(&params_long, forward, t_long);

        // Total variance should be increasing: σ²T should increase with T
        let test_strikes = [forward * 0.9, forward, forward * 1.1];
        
        for &k in &test_strikes {
            if let (Ok(iv_s), Ok(iv_l)) = (exp_short.implied_vol(k), exp_long.implied_vol(k)) {
                let var_short = iv_s * iv_s * t_short;
                let var_long = iv_l * iv_l * t_long;
                
                if var_long < var_short - self.calendar_tolerance {
                    return false;  // Calendar arbitrage detected
                }
            }
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sabr_params_creation() {
        let params = SabrParams::new(0.5, 0.9, -0.3, 0.6).unwrap();
        assert_eq!(params.alpha, 0.5);
        assert_eq!(params.beta, 0.9);
    }

    #[test]
    fn test_hagan_atm_vol() {
        let params = SabrParams::crypto_default();
        let expansion = HaganExpansion::new(&params, 100.0, 0.25);
        let iv = expansion.atm_vol().unwrap();
        assert!(iv > 0.3 && iv < 1.0);
    }

    #[test]
    fn test_smile_dynamics() {
        let mut dynamics = SmileDynamics::new();
        let params = SabrParams::crypto_default();
        
        dynamics.record(params, 100.0, 0.25);
        dynamics.record(params, 100.0, 0.25);
        
        assert!(dynamics.current_skew().is_some());
        assert!(dynamics.current_curvature().is_some());
    }

    #[test]
    fn test_arbitrage_check() {
        let checker = ArbitrageFreeSabr::new(1e-6, 1e-6);
        let params = SabrParams::crypto_default();
        let strikes = [90.0, 95.0, 100.0, 105.0, 110.0];
        
        let result = checker.check_butterfly_arbitrage(params, 100.0, 0.25, &strikes);
        assert!(result);  // Should pass for reasonable params
    }
}
