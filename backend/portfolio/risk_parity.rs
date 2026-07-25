//! Risk Parity Portfolio Allocator
//!
//! Equalizes risk contributions across BTC, SOL, ETH, USDT.
//! Unlike mean-variance optimization, focuses on risk budgeting rather than returns.
//! Ideal for crypto portfolios where return estimation is highly uncertain.
//!
//! # Key Principles:
//! - Each asset contributes equally to total portfolio risk
//! - More robust to estimation error than Markowitz optimization
//! - Naturally diversifies across volatility regimes
//!
//! # Algorithm:
//! Uses iterative optimization to find weights where:
//! RC_i = w_i * (Σw)_i / σ_p = constant for all i

use std::f64;

/// Fixed portfolio size
const N_ASSETS: usize = 4;

/// Result of risk parity optimization
#[derive(Debug, Clone)]
pub struct RiskParityResult {
    /// Optimal weights for each asset
    pub weights: [f64; N_ASSETS],
    /// Risk contribution of each asset (should be equal)
    pub risk_contributions: [f64; N_ASSETS],
    /// Total portfolio volatility
    pub portfolio_volatility: f64,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Maximum deviation from equal risk (measure of optimality)
    pub max_risk_deviation: f64,
}

/// Risk Parity Optimizer
///
/// Finds portfolio weights that equalize risk contributions.
/// Supports optional risk budgets for asymmetric allocation.
pub struct RiskParityOptimizer {
    /// Covariance matrix
    covariance: [[f64; N_ASSETS]; N_ASSETS],
    /// Optional custom risk budgets (default: equal 25% each)
    risk_budgets: [f64; N_ASSETS],
}

impl RiskParityOptimizer {
    /// Create new optimizer with equal risk budgets
    pub fn new(covariance: [[f64; N_ASSETS]; N_ASSETS]) -> Self {
        Self {
            covariance,
            risk_budgets: [0.25, 0.25, 0.25, 0.25],
        }
    }

    /// Create optimizer with custom risk budgets
    ///
    /// # Arguments
    /// * `risk_budgets` - Target risk contribution for each asset (must sum to 1)
    pub fn with_budgets(
        covariance: [[f64; N_ASSETS]; N_ASSETS],
        risk_budgets: [f64; N_ASSETS],
    ) -> Self {
        // Validate budgets sum to ~1
        let sum: f64 = risk_budgets.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Risk budgets must sum to 1.0, got {}",
            sum
        );

        Self {
            covariance,
            risk_budgets,
        }
    }

    /// Compute risk parity portfolio
    ///
    /// Uses cyclical coordinate descent for fast convergence.
    pub fn optimize(&self) -> RiskParityResult {
        // Initialize with inverse volatility weights (good starting point)
        let mut weights = self._inverse_volatility_init();
        
        let max_iterations = 1000;
        let tolerance = 1e-8;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_iterations {
            iterations = iter;

            // Cyclical coordinate descent update
            let new_weights = self._coordinate_descent_step(&weights);
            
            // Check convergence
            let max_change: f64 = weights
                .iter()
                .zip(new_weights.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            weights = new_weights;

            if max_change < tolerance {
                converged = true;
                break;
            }
        }

        // Compute final metrics
        let (risk_contributions, portfolio_vol) = self._compute_risk_metrics(&weights);
        let max_deviation = self._compute_max_risk_deviation(&risk_contributions);

        RiskParityResult {
            weights,
            risk_contributions,
            portfolio_volatility: portfolio_vol,
            converged,
            iterations,
            max_risk_deviation: max_deviation,
        }
    }

    /// Initialize weights inversely proportional to volatility
    fn _inverse_volatility_init(&self) -> [f64; N_ASSETS] {
        let mut weights = [0.0; N_ASSETS];
        let mut inv_vol_sum = 0.0;

        for i in 0..N_ASSETS {
            let vol = self.covariance[i][i].sqrt();
            if vol > 1e-10 {
                weights[i] = 1.0 / vol;
                inv_vol_sum += weights[i];
            } else {
                weights[i] = 1.0;
                inv_vol_sum += 1.0;
            }
        }

        // Normalize
        for w in &mut weights {
            *w /= inv_vol_sum;
        }

        weights
    }

    /// Single coordinate descent step
    fn _coordinate_descent_step(&self, weights: &[f64; N_ASSETS]) -> [f64; N_ASSETS] {
        let mut new_weights = *weights;

        // Update each weight cyclically
        for i in 0..N_ASSETS {
            // Compute marginal risk contribution
            let marginal_risk = self._marginal_risk(&new_weights, i);
            
            if marginal_risk > 1e-10 {
                // Target: RC_i = budget_i * portfolio_vol
                // RC_i = w_i * marginal_risk_i
                // So: w_i = budget_i * portfolio_vol / marginal_risk_i
                
                let portfolio_vol = self._portfolio_volatility(&new_weights);
                let target_weight = self.risk_budgets[i] * portfolio_vol / marginal_risk;
                
                // Damped update for stability
                new_weights[i] = 0.7 * new_weights[i] + 0.3 * target_weight.max(1e-6);
            }
        }

        // Normalize to sum to 1
        let sum: f64 = new_weights.iter().sum();
        if sum > 1e-10 {
            for w in &mut new_weights {
                *w /= sum;
            }
        }

        new_weights
    }

    /// Compute marginal risk contribution for asset i
    fn _marginal_risk(&self, weights: &[f64; N_ASSETS], i: usize) -> f64 {
        let portfolio_var = self._portfolio_variance(weights);
        if portfolio_var < 1e-12 {
            return 0.0;
        }
        
        let portfolio_vol = portfolio_var.sqrt();
        
        // Marginal risk = (Σw)_i / σ_p
        let mut cov_w = 0.0;
        for j in 0..N_ASSETS {
            cov_w += self.covariance[i][j] * weights[j];
        }
        
        cov_w / portfolio_vol
    }

    /// Compute portfolio variance
    fn _portfolio_variance(&self, weights: &[f64; N_ASSETS]) -> f64 {
        let mut var = 0.0;
        for i in 0..N_ASSETS {
            for j in 0..N_ASSETS {
                var += weights[i] * self.covariance[i][j] * weights[j];
            }
        }
        var
    }

    /// Compute portfolio volatility
    fn _portfolio_volatility(&self, weights: &[f64; N_ASSETS]) -> f64 {
        self._portfolio_variance(weights).sqrt()
    }

    /// Compute risk contributions and portfolio volatility
    fn _compute_risk_metrics(&self, weights: &[f64; N_ASSETS]) -> ([f64; N_ASSETS], f64) {
        let portfolio_vol = self._portfolio_volatility(weights);
        let mut risk_contributions = [0.0; N_ASSETS];

        if portfolio_vol > 1e-10 {
            for i in 0..N_ASSETS {
                let marginal = self._marginal_risk(weights, i);
                risk_contributions[i] = weights[i] * marginal;
            }
        }

        (risk_contributions, portfolio_vol)
    }

    /// Compute maximum deviation from target risk budgets
    fn _compute_max_risk_deviation(&self, risk_contributions: &[f64; N_ASSETS]) -> f64 {
        let mut max_dev = 0.0;
        for i in 0..N_ASSETS {
            let deviation = (risk_contributions[i] - self.risk_budgets[i]).abs();
            max_dev = max_dev.max(deviation);
        }
        max_dev
    }

    /// Compute implied risk budgets from given weights
    pub fn compute_risk_contributions(&self, weights: &[f64; N_ASSETS]) -> [f64; N_ASSETS] {
        let (rc, _) = self._compute_risk_metrics(weights);
        rc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_risk_contribution() {
        // Sample covariance matrix
        let covariance = [
            [0.04, 0.02, 0.025, 0.0001],
            [0.02, 0.09, 0.04, 0.0002],
            [0.025, 0.04, 0.05, 0.0001],
            [0.0001, 0.0002, 0.0001, 0.0001],
        ];

        let optimizer = RiskParityOptimizer::new(covariance);
        let result = optimizer.optimize();

        assert!(result.converged, "Should converge");
        
        // Weights should sum to 1
        let weight_sum: f64 = result.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6, "Weights should sum to 1");

        // All weights should be positive
        for w in &result.weights {
            assert!(*w > 0.0, "All weights should be positive");
        }

        // Risk contributions should be approximately equal (within 1%)
        let target_rc = 0.25;
        for rc in &result.risk_contributions {
            assert!(
                (rc - target_rc).abs() < 0.01,
                "Risk contribution {} should be close to 0.25",
                rc
            );
        }

        // Max deviation should be small
        assert!(result.max_risk_deviation < 0.01);
    }

    #[test]
    fn test_custom_risk_budgets() {
        let covariance = [
            [0.04, 0.02, 0.025, 0.0001],
            [0.02, 0.09, 0.04, 0.0002],
            [0.025, 0.04, 0.05, 0.0001],
            [0.0001, 0.0002, 0.0001, 0.0001],
        ];

        // Custom budgets: more risk to BTC, less to USDT
        let budgets = [0.35, 0.25, 0.30, 0.10];

        let optimizer = RiskParityOptimizer::with_budgets(covariance, budgets);
        let result = optimizer.optimize();

        assert!(result.converged);

        // Risk contributions should match custom budgets
        for (i, &budget) in budgets.iter().enumerate() {
            let deviation = (result.risk_contributions[i] - budget).abs();
            assert!(deviation < 0.01, "RC[{}] = {} should be close to {}", i, result.risk_contributions[i], budget);
        }
    }

    #[test]
    fn test_high_correlation_handling() {
        // Highly correlated assets (stress scenario)
        let covariance = [
            [0.04, 0.035, 0.035, 0.0001],
            [0.035, 0.09, 0.08, 0.0002],
            [0.035, 0.08, 0.05, 0.0001],
            [0.0001, 0.0002, 0.0001, 0.0001],
        ];

        let optimizer = RiskParityOptimizer::new(covariance);
        let result = optimizer.optimize();

        // Should still converge even with high correlation
        assert!(result.converged);
        assert!(result.weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_performance() {
        let covariance = [
            [0.04, 0.02, 0.025, 0.0001],
            [0.02, 0.09, 0.04, 0.0002],
            [0.025, 0.04, 0.05, 0.0001],
            [0.0001, 0.0002, 0.0001, 0.0001],
        ];

        let optimizer = RiskParityOptimizer::new(covariance);
        
        let start = std::time::Instant::now();
        let _result = optimizer.optimize();
        let elapsed = start.elapsed();

        // Should complete very quickly (< 1ms for 4 assets)
        assert!(elapsed.as_micros() < 1000, "Should complete in under 1ms");
    }
}
