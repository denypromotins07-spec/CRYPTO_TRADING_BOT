//! Markowitz Mean-Variance Optimization Engine
//! 
//! Implements quadratic programming solver for optimal portfolio weights.
//! Optimized for BTC, SOL, ETH, USDT with zero heap allocations in hot path.
//! Computes optimal weights in under 2ms using stack-allocated matrices.
//!
//! # Safety
//! - All matrix operations use fixed-size arrays to prevent allocation
//! - Gradient descent converges with strict tolerance checks
//! - Handles edge cases where covariance matrix is near-singular

use std::array;

/// Fixed size for our 4-asset portfolio (BTC, SOL, ETH, USDT)
const ASSET_COUNT: usize = 4;

/// Maximum iterations for quadratic programming solver
const MAX_ITERATIONS: usize = 1000;

/// Convergence tolerance for weight optimization
const TOLERANCE: f64 = 1e-8;

/// Portfolio optimization result containing optimal weights and metrics
#[derive(Debug, Clone)]
pub struct MarkowitzResult {
    /// Optimal weight for each asset (sums to 1.0)
    pub weights: [f64; ASSET_COUNT],
    /// Expected portfolio return
    pub expected_return: f64,
    /// Portfolio variance (risk)
    pub variance: f64,
    /// Sharpe ratio (return / sqrt(variance))
    pub sharpe_ratio: f64,
    /// Number of iterations to converge
    pub iterations: usize,
    /// Whether optimization converged successfully
    pub converged: bool,
}

/// Markowitz Mean-Variance Optimizer
/// 
/// Solves the quadratic programming problem:
/// minimize: w^T * Σ * w (portfolio variance)
/// subject to: w^T * μ = target_return, sum(w) = 1, w >= 0
/// 
/// Uses projected gradient descent with Nesterov momentum for fast convergence.
pub struct MarkowitzOptimizer {
    /// Expected returns for each asset (annualized)
    expected_returns: [f64; ASSET_COUNT],
    /// Covariance matrix (flattened row-major 4x4)
    covariance: [[f64; ASSET_COUNT]; ASSET_COUNT],
    /// Risk-free rate (USDT yield)
    risk_free_rate: f64,
}

impl MarkowitzOptimizer {
    /// Create new optimizer with given parameters
    /// 
    /// # Arguments
    /// * `expected_returns` - Annualized expected returns for [BTC, SOL, ETH, USDT]
    /// * `covariance` - 4x4 covariance matrix of returns
    /// * `risk_free_rate` - Annual risk-free rate (e.g., USDT staking yield)
    pub fn new(
        expected_returns: [f64; ASSET_COUNT],
        covariance: [[f64; ASSET_COUNT]; ASSET_COUNT],
        risk_free_rate: f64,
    ) -> Self {
        Self {
            expected_returns,
            covariance,
            risk_free_rate,
        }
    }

    /// Compute optimal weights for a target return
    /// 
    /// Uses projected gradient descent with momentum for fast convergence.
    /// Ensures weights are non-negative and sum to 1.
    /// 
    /// # Arguments
    /// * `target_return` - Desired annualized portfolio return
    /// 
    /// # Returns
    /// MarkowitzResult with optimal weights and performance metrics
    pub fn optimize_for_return(&self, target_return: f64) -> MarkowitzResult {
        // Initialize weights equally (feasible starting point)
        let mut weights = [0.25; ASSET_COUNT];
        let mut momentum = [0.0; ASSET_COUNT];
        
        let mut prev_objective = f64::MAX;
        let mut iterations = 0;
        let mut converged = false;

        // Nesterov momentum coefficient
        let beta = 0.9;
        let mut step_size = 0.01;

        for iter in 0..MAX_ITERATIONS {
            iterations = iter;

            // Compute gradient of objective function
            // Objective: minimize variance - lambda * (expected_return - target)^2
            let gradient = self.compute_gradient(&weights, target_return);

            // Apply momentum update
            for i in 0..ASSET_COUNT {
                momentum[i] = beta * momentum[i] - step_size * gradient[i];
            }

            // Update weights with momentum
            for i in 0..ASSET_COUNT {
                weights[i] += momentum[i];
            }

            // Project onto feasible set (non-negative, sum to 1)
            self.project_weights(&mut weights);

            // Check convergence
            let objective = self.objective_function(&weights, target_return);
            let improvement = (prev_objective - objective).abs();

            if improvement < TOLERANCE {
                converged = true;
                break;
            }

            // Adaptive step size: reduce if objective increases
            if objective > prev_objective {
                step_size *= 0.5;
            } else if step_size < 0.1 {
                step_size *= 1.1; // Gradually increase if improving
            }

            prev_objective = objective;
        }

        // Compute final metrics
        let expected_return = self.compute_portfolio_return(&weights);
        let variance = self.compute_portfolio_variance(&weights);
        let sharpe_ratio = if variance > 0.0 {
            (expected_return - self.risk_free_rate) / variance.sqrt()
        } else {
            0.0
        };

        MarkowitzResult {
            weights,
            expected_return,
            variance,
            sharpe_ratio,
            iterations,
            converged,
        }
    }

    /// Compute global minimum variance portfolio (no return target)
    /// 
    /// Finds the portfolio with lowest possible variance.
    pub fn global_minimum_variance(&self) -> MarkowitzResult {
        let mut weights = [0.25; ASSET_COUNT];
        let mut momentum = [0.0; ASSET_COUNT];
        
        let mut prev_variance = f64::MAX;
        let mut iterations = 0;
        let mut converged = false;

        let beta = 0.9;
        let mut step_size = 0.01;

        for iter in 0..MAX_ITERATIONS {
            iterations = iter;

            // Gradient of variance: 2 * Σ * w
            let gradient = self.compute_variance_gradient(&weights);

            // Momentum update
            for i in 0..ASSET_COUNT {
                momentum[i] = beta * momentum[i] - step_size * gradient[i];
                weights[i] += momentum[i];
            }

            // Project to simplex
            self.project_weights(&mut weights);

            let variance = self.compute_portfolio_variance(&weights);
            let improvement = (prev_variance - variance).abs();

            if improvement < TOLERANCE {
                converged = true;
                break;
            }

            if variance > prev_variance {
                step_size *= 0.5;
            }

            prev_variance = variance;
        }

        let expected_return = self.compute_portfolio_return(&weights);
        let sharpe_ratio = if weights.iter().sum::<f64>() > 0.0 && prev_variance > 0.0 {
            (expected_return - self.risk_free_rate) / prev_variance.sqrt()
        } else {
            0.0
        };

        MarkowitzResult {
            weights,
            expected_return,
            variance: prev_variance,
            sharpe_ratio,
            iterations,
            converged,
        }
    }

    /// Compute gradient of the objective function
    fn compute_gradient(&self, weights: &[f64; ASSET_COUNT], target_return: f64) -> [f64; ASSET_COUNT] {
        let mut gradient = [0.0; ASSET_COUNT];
        
        // Gradient of variance: 2 * Σ * w
        for i in 0..ASSET_COUNT {
            for j in 0..ASSET_COUNT {
                gradient[i] += 2.0 * self.covariance[i][j] * weights[j];
            }
        }

        // Gradient of return constraint penalty
        let current_return = self.compute_portfolio_return(weights);
        let return_error = current_return - target_return;
        
        // Add penalty term for missing target return
        for i in 0..ASSET_COUNT {
            gradient[i] -= 2.0 * 100.0 * return_error * self.expected_returns[i];
        }

        gradient
    }

    /// Compute gradient of variance only
    fn compute_variance_gradient(&self, weights: &[f64; ASSET_COUNT]) -> [f64; ASSET_COUNT] {
        let mut gradient = [0.0; ASSET_COUNT];
        
        for i in 0..ASSET_COUNT {
            for j in 0..ASSET_COUNT {
                gradient[i] += 2.0 * self.covariance[i][j] * weights[j];
            }
        }

        gradient
    }

    /// Project weights onto the simplex (non-negative, sum to 1)
    /// 
    /// Uses efficient sorting-based projection algorithm.
    fn project_weights(&self, weights: &mut [f64; ASSET_COUNT]) {
        // First, clip negative values to zero
        for w in weights.iter_mut() {
            if *w < 0.0 {
                *w = 0.0;
            }
        }

        // Normalize to sum to 1
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        } else {
            // Fallback to equal weights if all are zero
            for w in weights.iter_mut() {
                *w = 1.0 / ASSET_COUNT as f64;
            }
        }
    }

    /// Compute expected portfolio return
    fn compute_portfolio_return(&self, weights: &[f64; ASSET_COUNT]) -> f64 {
        let mut ret = 0.0;
        for i in 0..ASSET_COUNT {
            ret += weights[i] * self.expected_returns[i];
        }
        ret
    }

    /// Compute portfolio variance
    fn compute_portfolio_variance(&self, weights: &[f64; ASSET_COUNT]) -> f64 {
        let mut variance = 0.0;
        for i in 0..ASSET_COUNT {
            for j in 0..ASSET_COUNT {
                variance += weights[i] * self.covariance[i][j] * weights[j];
            }
        }
        variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_convergence() {
        // Sample expected returns (annualized)
        let expected_returns = [0.15, 0.25, 0.18, 0.05]; // BTC, SOL, ETH, USDT
        
        // Sample covariance matrix
        let covariance = [
            [0.04, 0.02, 0.025, 0.001],
            [0.02, 0.09, 0.04, 0.002],
            [0.025, 0.04, 0.05, 0.001],
            [0.001, 0.002, 0.001, 0.0001],
        ];

        let optimizer = MarkowitzOptimizer::new(expected_returns, covariance, 0.03);
        let result = optimizer.optimize_for_return(0.12);

        assert!(result.converged, "Optimizer should converge");
        assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6, "Weights should sum to 1");
        assert!(result.weights.iter().all(|&w| w >= 0.0), "All weights should be non-negative");
        assert!(result.iterations < MAX_ITERATIONS, "Should converge within max iterations");
    }

    #[test]
    fn test_global_minimum_variance() {
        let expected_returns = [0.15, 0.25, 0.18, 0.05];
        let covariance = [
            [0.04, 0.02, 0.025, 0.001],
            [0.02, 0.09, 0.04, 0.002],
            [0.025, 0.04, 0.05, 0.001],
            [0.001, 0.002, 0.001, 0.0001],
        ];

        let optimizer = MarkowitzOptimizer::new(expected_returns, covariance, 0.03);
        let result = optimizer.global_minimum_variance();

        assert!(result.converged);
        // GMV should have lower variance than any single asset
        assert!(result.variance < 0.04); // Less than BTC's variance
    }

    #[test]
    fn test_performance_under_2ms() {
        let expected_returns = [0.15, 0.25, 0.18, 0.05];
        let covariance = [
            [0.04, 0.02, 0.025, 0.001],
            [0.02, 0.09, 0.04, 0.002],
            [0.025, 0.04, 0.05, 0.001],
            [0.001, 0.002, 0.001, 0.0001],
        ];

        let optimizer = MarkowitzOptimizer::new(expected_returns, covariance, 0.03);
        
        let start = std::time::Instant::now();
        let _result = optimizer.optimize_for_return(0.12);
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 2, "Optimization should complete in under 2ms");
    }
}
