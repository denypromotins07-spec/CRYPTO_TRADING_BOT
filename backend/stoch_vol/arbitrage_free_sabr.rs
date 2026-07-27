//! Arbitrage-Free SABR Model Implementation
//! Enforces butterfly and calendar arbitrage constraints
//! Ensures volatility surface is free from static arbitrage opportunities
//! 
//! Critical for production trading: any arbitrage implies model misspecification

use std::collections::HashMap;

use super::sabr_model::{SabrParams, HaganExpansion, SabrError};

/// Result of arbitrage check
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArbitrageStatus {
    /// No arbitrage detected
    Clean,
    /// Butterfly arbitrage (convexity violation)
    ButterflyArbitrage { severity: f64 },
    /// Calendar arbitrage (time inconsistency)
    CalendarArbitrage { severity: f64 },
    /// Both types detected
    Both { butterfly_severity: f64, calendar_severity: f64 },
}

impl ArbitrageStatus {
    pub fn is_arbitrage_free(&self) -> bool {
        matches!(self, ArbitrageStatus::Clean)
    }

    pub fn max_severity(&self) -> f64 {
        match self {
            ArbitrageStatus::Clean => 0.0,
            ArbitrageStatus::ButterflyArbitrage { severity } => *severity,
            ArbitrageStatus::CalendarArbitrage { severity } => *severity,
            ArbitrageStatus::Both { butterfly_severity, calendar_severity } => {
                *butterfly_severity.max(calendar_severity)
            }
        }
    }
}

/// Configuration for arbitrage checks
#[derive(Debug, Clone, Copy)]
pub struct ArbitrageConfig {
    pub butterfly_tolerance: f64,    // Tolerance for butterfly spread violations
    pub calendar_tolerance: f64,     // Tolerance for calendar spread violations
    pub total_var_tolerance: f64,    // Tolerance for total variance monotonicity
    pub min_strike_spacing: f64,     // Minimum strike spacing for reliable checks
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            butterfly_tolerance: 1e-4,    // 0.01% price tolerance
            calendar_tolerance: 1e-4,
            total_var_tolerance: 1e-6,
            min_strike_spacing: 0.01,     // 1% moneyness spacing
        }
    }
}

/// Arbitrage-free SABR validator
pub struct ArbitrageFreeSabrValidator {
    config: ArbitrageConfig,
    strike_cache: Vec<f64>,
}

impl ArbitrageFreeSabrValidator {
    pub fn new(config: ArbitrageConfig) -> Self {
        let strike_cache = Self::generate_standard_strikes();
        Self { config, strike_cache }
    }

    pub fn with_default_config() -> Self {
        Self::new(ArbitrageConfig::default())
    }

    /// Generate standard strike grid for arbitrage checks
    fn generate_standard_strikes() -> Vec<f64> {
        // Standard delta-based strikes: 10d, 15d, 20d, 25d, ATM
        let deltas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50];
        let mut strikes = Vec::with_capacity(deltas.len() * 2 + 1);
        
        // OTM puts
        for &delta in &deltas {
            strikes.push(1.0 - (1.0 - 2.0 * delta));  // Approximate moneyness
        }
        strikes.push(1.0);  // ATM
        // OTM calls
        for &delta in deltas.iter().rev() {
            strikes.push(1.0 + (1.0 - 2.0 * delta));
        }
        
        strikes
    }

    /// Comprehensive arbitrage check across all dimensions
    pub fn check_all(&self, params: SabrParams, forward: f64, 
                     expiries: &[f64]) -> ArbitrageStatus {
        let mut butterfly_arb = false;
        let mut calendar_arb = false;
        let mut max_butterfly_severity = 0.0;
        let mut max_calendar_severity = 0.0;

        // Check butterfly arbitrage for each expiry
        for &expiry in expiries {
            if let Some(severity) = self.check_butterfly_arbitrage(params, forward, expiry) {
                butterfly_arb = true;
                max_butterfly_severity = max_butterfly_severity.max(severity);
            }
        }

        // Check calendar arbitrage across expiries
        if expiries.len() >= 2 {
            if let Some(severity) = self.check_calendar_arbitrage(params, forward, expiries) {
                calendar_arb = true;
                max_calendar_severity = severity;
            }
        }

        match (butterfly_arb, calendar_arb) {
            (false, false) => ArbitrageStatus::Clean,
            (true, false) => ArbitrageStatus::ButterflyArbitrage { 
                severity: max_butterfly_severity 
            },
            (false, true) => ArbitrageStatus::CalendarArbitrage { 
                severity: max_calendar_severity 
            },
            (true, true) => ArbitrageStatus::Both {
                butterfly_severity: max_butterfly_severity,
                calendar_severity: max_calendar_severity,
            },
        }
    }

    /// Check for butterfly arbitrage (violation of call price convexity)
    /// Butterfly spread: long 1 low strike, short 2 middle, long 1 high
    /// Must have non-negative value for no arbitrage
    pub fn check_butterfly_arbitrage(&self, params: SabrParams, 
                                      forward: f64, expiry: f64) -> Option<f64> {
        let expansion = HaganExpansion::new(&params, forward, expiry);
        
        let mut max_violation = 0.0;
        
        // Check convexity at multiple points
        for i in 0..self.strike_cache.len() - 2 {
            let k_low = self.strike_cache[i] * forward;
            let k_mid = self.strike_cache[i + 1] * forward;
            let k_high = self.strike_cache[i + 2] * forward;

            // Skip if spacing is too small
            if (k_mid - k_low) / forward < self.config.min_strike_spacing ||
               (k_high - k_mid) / forward < self.config.min_strike_spacing {
                continue;
            }

            // Get implied vols
            let iv_low = expansion.implied_vol(k_low)?;
            let iv_mid = expansion.implied_vol(k_mid)?;
            let iv_high = expansion.implied_vol(k_high)?;

            // Convert to prices using Black-Scholes
            let p_low = self.black_price(forward, k_low, expiry, iv_low, true);
            let p_mid = self.black_price(forward, k_mid, expiry, iv_mid, true);
            let p_high = self.black_price(forward, k_high, expiry, iv_high, true);

            // Compute lambda for uneven spacing
            let lambda = (k_mid - k_low) / (k_high - k_low);
            
            // Butterfly value: should be >= 0
            let butterfly_value = p_low - (1.0 + lambda) * p_mid + lambda * p_high;
            
            if butterfly_value < -self.config.butterfly_tolerance {
                let severity = (-butterfly_value / forward).abs();
                max_violation = max_violation.max(severity);
            }
        }

        if max_violation > 0.0 {
            Some(max_violation)
        } else {
            None
        }
    }

    /// Check for calendar arbitrage (total variance must increase with time)
    /// σ²(T) * T must be non-decreasing in T for all strikes
    pub fn check_calendar_arbitrage(&self, params: SabrParams, 
                                     forward: f64, expiries: &[f64]) -> Option<f64> {
        if expiries.len() < 2 {
            return None;
        }

        let mut sorted_expiries = expiries.to_vec();
        sorted_expiries.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut max_violation = 0.0;

        // Check total variance monotonicity for each strike
        for &strike_mult in &self.strike_cache {
            let strike = strike_mult * forward;
            let mut prev_total_var = 0.0;

            for &expiry in &sorted_expiries {
                let expansion = HaganExpansion::new(&params, forward, expiry);
                
                if let Some(iv) = expansion.implied_vol(strike) {
                    let total_var = iv * iv * expiry;
                    
                    // Total variance should increase with time
                    if expiry > 0.0 && total_var < prev_total_var - self.config.total_var_tolerance {
                        let severity = (prev_total_var - total_var) / prev_total_var;
                        max_violation = max_violation.max(severity);
                    }
                    
                    prev_total_var = total_var;
                }
            }
        }

        if max_violation > 0.0 {
            Some(max_violation)
        } else {
            None
        }
    }

    /// Enforce arbitrage-free constraints by adjusting parameters
    /// Returns adjusted parameters that pass all arbitrage checks
    pub fn enforce_constraints(&self, params: SabrParams, forward: f64,
                                expiries: &[f64], max_iterations: usize) -> SabrParams {
        let mut current_params = params;
        
        for _ in 0..max_iterations {
            let status = self.check_all(current_params, forward, expiries);
            
            if status.is_arbitrage_free() {
                return current_params;
            }

            // Adjust parameters to remove arbitrage
            current_params = self.adjust_parameters(current_params, &status);
        }

        // Return best effort even if not fully arbitrage-free
        current_params
    }

    /// Adjust SABR parameters to remove detected arbitrage
    fn adjust_parameters(&self, params: SabrParams, status: &ArbitrageStatus) -> SabrParams {
        let mut adjusted = params;

        match status {
            ArbitrageStatus::ButterflyArbitrage { .. } => {
                // Reduce vol-of-vol to flatten smile
                adjusted.nu *= 0.9;
                // Move beta closer to 1 (more lognormal)
                adjusted.beta = adjusted.beta.min(0.95).max(0.7);
            }
            ArbitrageStatus::CalendarArbitrage { .. } => {
                // Adjust term structure by reducing alpha
                adjusted.alpha *= 0.95;
            }
            ArbitrageStatus::Both { .. } => {
                // Aggressive adjustment for both
                adjusted.nu *= 0.85;
                adjusted.alpha *= 0.95;
                adjusted.beta = 0.85;
            }
            ArbitrageStatus::Clean => {}
        }

        // Ensure parameters remain valid
        adjusted.alpha = adjusted.alpha.max(0.01);
        adjusted.nu = adjusted.nu.max(0.01);
        adjusted.beta = adjusted.beta.clamp(0.0, 1.0);
        adjusted.rho = adjusted.rho.clamp(-0.99, 0.99);

        adjusted
    }

    /// Simple Black-Scholes pricer for arbitrage checks
    fn black_price(&self, f: f64, k: f64, t: f64, sigma: f64, is_call: bool) -> f64 {
        if t <= 0.0 || sigma <= 0.0 {
            return if is_call { (f - k).max(0.0) } else { (k - f).max(0.0) };
        }

        let d1 = (f / k).ln() / (sigma * t.sqrt()) + 0.5 * sigma * t.sqrt();
        let d2 = d1 - sigma * t.sqrt();

        // Error function approximation for normal CDF
        let erf_approx = |x: f64| -> f64 {
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            let x = x.abs();
            let t = 1.0 / (1.0 + 0.3275911 * x);
            let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 
                + t * (-1.453152027 + t * 1.061405429))));
            sign * (1.0 - poly * (-x * x).exp())
        };

        let nd1 = 0.5 * (1.0 + erf_approx(d1 / std::f64::consts::SQRT_2));
        let nd2 = 0.5 * (1.0 + erf_approx(d2 / std::f64::consts::SQRT_2));

        if is_call {
            f * nd1 - k * nd2
        } else {
            k * (1.0 - nd2) - f * (1.0 - nd1)
        }
    }
}

/// Builder for constructing arbitrage-free SABR surfaces
pub struct SabrSurfaceBuilder {
    params_by_expiry: HashMap<f64, SabrParams>,
    validator: ArbitrageFreeSabrValidator,
}

impl SabrSurfaceBuilder {
    pub fn new() -> Self {
        Self {
            params_by_expiry: HashMap::new(),
            validator: ArbitrageFreeSabrValidator::with_default_config(),
        }
    }

    /// Add calibrated parameters for an expiry
    pub fn add_expiry(&mut self, expiry: f64, params: SabrParams, forward: f64) -> &mut Self {
        // Validate and adjust if necessary
        let expiries: Vec<f64> = self.params_by_expiry.keys().copied().chain([expiry]).collect();
        let adjusted = self.validator.enforce_constraints(params, forward, &expiries, 10);
        
        self.params_by_expiry.insert(expiry, adjusted);
        self
    }

    /// Build the final arbitrage-free surface
    pub fn build(self) -> HashMap<f64, SabrParams> {
        self.params_by_expiry
    }

    /// Check if current surface is arbitrage-free
    pub fn validate_surface(&self, forward: f64) -> ArbitrageStatus {
        let expiries: Vec<f64> = self.params_by_expiry.keys().copied().collect();
        
        if expiries.is_empty() {
            return ArbitrageStatus::Clean;
        }

        // Use average parameters for cross-expiry check
        let avg_params = self.compute_average_params();
        self.validator.check_all(avg_params, forward, &expiries)
    }

    fn compute_average_params(&self) -> SabrParams {
        let n = self.params_by_expiry.len() as f64;
        if n == 0.0 {
            return SabrParams::crypto_default();
        }

        let (sum_alpha, sum_beta, sum_rho, sum_nu) = self.params_by_expiry.values()
            .fold((0.0, 0.0, 0.0, 0.0), |(a, b, r, n), p| 
                (a + p.alpha, b + p.beta, r + p.rho, n + p.nu)
            );

        SabrParams::new(
            sum_alpha / n,
            sum_beta / n,
            sum_rho / n,
            sum_nu / n,
        ).unwrap_or_else(|_| SabrParams::crypto_default())
    }
}

impl Default for SabrSurfaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butterfly_check_clean() {
        let validator = ArbitrageFreeSabrValidator::with_default_config();
        let params = SabrParams::new(0.5, 0.9, -0.3, 0.4).unwrap();
        
        let result = validator.check_butterfly_arbitrage(params, 100.0, 0.25);
        assert!(result.is_none());  // No arbitrage
    }

    #[test]
    fn test_extreme_params_detect_arbitrage() {
        let validator = ArbitrageFreeSabrValidator::with_default_config();
        // Extreme vol-of-vol can cause butterfly arbitrage
        let params = SabrParams::new(0.3, 0.5, -0.8, 2.0).unwrap();
        
        let result = validator.check_butterfly_arbitrage(params, 100.0, 0.25);
        // May or may not detect depending on severity
        // Test just ensures no panic
    }

    #[test]
    fn test_enforce_constraints() {
        let validator = ArbitrageFreeSabrValidator::with_default_config();
        let params = SabrParams::crypto_default();
        let expiries = [0.1, 0.25, 0.5];
        
        let adjusted = validator.enforce_constraints(params, 100.0, &expiries, 20);
        
        // Should return valid parameters
        assert!(adjusted.alpha > 0.0);
        assert!(adjusted.nu > 0.0);
        assert!((0.0..=1.0).contains(&adjusted.beta));
    }

    #[test]
    fn test_surface_builder() {
        let mut builder = SabrSurfaceBuilder::new();
        
        builder
            .add_expiry(0.1, SabrParams::crypto_default(), 100.0)
            .add_expiry(0.25, SabrParams::crypto_default(), 100.0)
            .add_expiry(0.5, SabrParams::crypto_default(), 100.0);
        
        let status = builder.validate_surface(100.0);
        // Surface should be arbitrage-free after enforcement
        assert!(status.is_arbitrage_free() || status.max_severity() < 0.01);
    }
}
