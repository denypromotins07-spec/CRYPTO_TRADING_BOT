//! Feller Condition Validator for Heston Model
//! Ensures variance process remains strictly positive: 2κθ > ξ²
//! Critical for numerical stability during flash crashes and extreme volatility
//! 
//! Zero-cost abstractions with compile-time checks where possible

use std::fmt;

/// Feller condition validation result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FellerStatus {
    /// Strong condition satisfied: 2κθ >> ξ² (very stable)
    StrongSatisfied { ratio: f64 },
    /// Weakly satisfied: 2κθ > ξ² but close to boundary
    WeakSatisfied { ratio: f64 },
    /// Violated: variance can reach zero (numerical instability risk)
    Violated { ratio: f64 },
    /// Critical: parameters are invalid (negative or zero)
    Invalid { reason: &'static str },
}

impl fmt::Display for FellerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FellerStatus::StrongSatisfied { ratio } => 
                write!(f, "Feller STRONGLY satisfied (ratio={:.2})", ratio),
            FellerStatus::WeakSatisfied { ratio } => 
                write!(f, "Feller weakly satisfied (ratio={:.2})", ratio),
            FellerStatus::Violated { ratio } => 
                write!(f, "Feller VIOLATED (ratio={:.2}) - variance may hit zero!", ratio),
            FellerStatus::Invalid { reason } => 
                write!(f, "Invalid parameters: {}", reason),
        }
    }
}

/// Heston parameter set for Feller validation
#[derive(Debug, Clone, Copy)]
pub struct FellerParams {
    pub kappa: f64,   // Mean reversion speed
    pub theta: f64,   // Long-term variance mean
    pub xi: f64,      // Volatility of variance (vol-of-vol)
}

impl FellerParams {
    /// Create new parameters with basic validation
    pub fn new(kappa: f64, theta: f64, xi: f64) -> Result<Self, &'static str> {
        if kappa <= 0.0 {
            return Err("kappa must be positive");
        }
        if theta <= 0.0 {
            return Err("theta must be positive");
        }
        if xi <= 0.0 {
            return Err("xi must be positive");
        }
        Ok(Self { kappa, theta, xi })
    }

    /// Compute the Feller ratio: 2κθ / ξ²
    /// Ratio > 1: condition satisfied (variance stays positive)
    /// Ratio < 1: condition violated (variance can reach zero)
    #[inline]
    pub fn feller_ratio(&self) -> f64 {
        (2.0 * self.kappa * self.theta) / (self.xi * self.xi)
    }

    /// Comprehensive Feller condition check with status
    pub fn check_feller(&self) -> FellerStatus {
        let ratio = self.feller_ratio();
        
        // Thresholds for classification
        const STRONG_THRESHOLD: f64 = 2.0;  // Very safe zone
        const WEAK_THRESHOLD: f64 = 1.0;    // Boundary
        
        if ratio >= STRONG_THRESHOLD {
            FellerStatus::StrongSatisfied { ratio }
        } else if ratio >= WEAK_THRESHOLD {
            FellerStatus::WeakSatisfied { ratio }
        } else {
            FellerStatus::Violated { ratio }
        }
    }

    /// Adjust parameters to satisfy Feller condition
    /// Returns new parameters that guarantee 2κθ > ξ²
    pub fn adjust_to_satisfy(&self, target_ratio: f64) -> Self {
        let current_ratio = self.feller_ratio();
        
        if current_ratio >= target_ratio {
            return *self;  // Already satisfied
        }
        
        // Strategy: increase theta (most natural adjustment)
        let required_theta = (target_ratio * self.xi * self.xi) / (2.0 * self.kappa);
        
        Self {
            kappa: self.kappa,
            theta: required_theta.max(self.theta * 1.1),  // At least 10% increase
            xi: self.xi,
        }
    }

    /// Check if parameters are in the "danger zone" for numerical schemes
    /// Even if Feller is satisfied, low ratios can cause issues with Euler discretization
    pub fn is_numerically_stable(&self, scheme: DiscretizationScheme) -> bool {
        let ratio = self.feller_ratio();
        
        match scheme {
            DiscretizationScheme::Euler => {
                // Euler requires stronger condition due to bias
                ratio > 3.0
            }
            DiscretizationScheme::Milstein => {
                // Milstein is more robust
                ratio > 1.5
            }
            DiscretizationScheme::QuadraticExponential => {
                // QE scheme handles low ratios well
                ratio > 0.5
            }
            DiscretizationScheme::FullTruncation => {
                // Full truncation is very robust
                ratio > 0.3
            }
        }
    }
}

/// Discretization schemes for Heston variance process
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiscretizationScheme {
    /// Simple Euler-Maruyama (biased, requires strong Feller)
    Euler,
    /// Milstein scheme (better accuracy)
    Milstein,
    /// Quadratic Exponential (Andersen 2008) - excellent for low Feller ratios
    QuadraticExponential,
    /// Full Truncation (Lord et al. 2010) - prevents negative variance
    FullTruncation,
}

/// Runtime monitor for Feller condition during calibration
pub struct FellerMonitor {
    params_history: Vec<(FellerParams, FellerStatus)>,
    alert_threshold: f64,
}

impl FellerMonitor {
    pub fn new(alert_threshold: f64) -> Self {
        Self {
            params_history: Vec::with_capacity(100),
            alert_threshold: alert_threshold.max(1.0),
        }
    }

    /// Record and validate new parameters
    pub fn record(&mut self, params: FellerParams) -> FellerStatus {
        let status = params.check_feller();
        self.params_history.push((params, status));
        
        // Keep history bounded for memory efficiency
        if self.params_history.len() > 100 {
            self.params_history.remove(0);
        }
        
        status
    }

    /// Check if any recent calibrations violated Feller
    pub fn has_recent_violations(&self, lookback: usize) -> bool {
        let start = self.params_history.len().saturating_sub(lookback);
        self.params_history[start..]
            .iter()
            .any(|(_, status)| matches!(status, FellerStatus::Violated { .. }))
    }

    /// Get the worst (lowest) Feller ratio in recent history
    pub fn worst_recent_ratio(&self, lookback: usize) -> Option<f64> {
        if self.params_history.is_empty() {
            return None;
        }
        
        let start = self.params_history.len().saturating_sub(lookback);
        self.params_history[start..]
            .iter()
            .map(|(_, status)| match status {
                FellerStatus::StrongSatisfied { ratio } |
                FellerStatus::WeakSatisfied { ratio } |
                FellerStatus::Violated { ratio } => *ratio,
                FellerStatus::Invalid { .. } => 0.0,
            })
            .fold(None, |min, r| Some(min.map_or(r, |m: f64| m.min(r))))
    }

    /// Generate alert message if Feller condition is at risk
    pub fn generate_alert(&self) -> Option<String> {
        if let Some(worst) = self.worst_recent_ratio(10) {
            if worst < self.alert_threshold {
                return Some(format!(
                    "WARNING: Feller ratio dropped to {:.3} (threshold: {})",
                    worst, self.alert_threshold
                ));
            }
        }
        None
    }
}

/// Compile-time Feller checker for constant parameters
/// Uses const fn for zero-runtime-overhead validation
pub mod const_checks {
    use super::*;

    /// Const Feller ratio calculation
    pub const fn feller_ratio_const(kappa: f64, theta: f64, xi: f64) -> f64 {
        if xi == 0.0 {
            return f64::INFINITY;  // Degenerate case
        }
        (2.0 * kappa * theta) / (xi * xi)
    }

    /// Const validation check
    pub const fn is_feller_satisfied_const(kappa: f64, theta: f64, xi: f64) -> bool {
        feller_ratio_const(kappa, theta, xi) > 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feller_satisfied() {
        let params = FellerParams::new(2.0, 0.04, 0.3).unwrap();
        let status = params.check_feller();
        assert!(matches!(status, FellerStatus::StrongSatisfied { .. }));
        assert!(params.feller_ratio() > 1.0);
    }

    #[test]
    fn test_feller_violated() {
        let params = FellerParams::new(0.1, 0.01, 0.5).unwrap();
        let status = params.check_feller();
        assert!(matches!(status, FellerStatus::Violated { .. }));
        assert!(params.feller_ratio() < 1.0);
    }

    #[test]
    fn test_adjustment() {
        let params = FellerParams::new(0.5, 0.02, 0.4).unwrap();
        assert!(params.feller_ratio() < 1.0);
        
        let adjusted = params.adjust_to_satisfy(1.5);
        assert!(adjusted.feller_ratio() >= 1.5);
        assert!(adjusted.theta > params.theta);
    }

    #[test]
    fn test_const_checks() {
        assert!(const_checks::is_feller_satisfied_const(2.0, 0.04, 0.3));
        assert!(!const_checks::is_feller_satisfied_const(0.1, 0.01, 0.5));
    }

    #[test]
    fn test_monitor() {
        let mut monitor = FellerMonitor::new(1.2);
        
        let good_params = FellerParams::new(2.0, 0.04, 0.3).unwrap();
        let bad_params = FellerParams::new(0.1, 0.01, 0.5).unwrap();
        
        monitor.record(good_params);
        assert!(!monitor.has_recent_violations(5));
        
        monitor.record(bad_params);
        assert!(monitor.has_recent_violations(5));
        assert!(monitor.generate_alert().is_some());
    }
}
