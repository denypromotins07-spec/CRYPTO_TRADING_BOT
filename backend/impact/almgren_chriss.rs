//! Almgren-Chriss Optimal Execution Model
//!
//! This module implements the classic Almgren-Chriss model for optimal
//! trade execution, solving for the optimal trajectory to execute large
//! block trades while minimizing market impact and timing risk.
//!
//! Key Features:
//! - Closed-form solution for optimal execution trajectory
//! - Dynamic adjustment based on volatility and risk aversion
//! - Separation of permanent and temporary market impact
//! - Time-constrained optimization for 4-hour trading windows
//! - Zero-cost abstractions with strict borrowing checks
//!
//! Mathematical Foundation:
//! The model minimizes: E[Cost] + λ * Var[Cost]
//! where:
//!   E[Cost] = η * X²/T + γ * X²/2 (permanent + temporary impact)
//!   Var[Cost] = σ² * ∫₀ᵀ x(t)² dt (timing risk)
//!
//! Optimal trajectory: x*(t) = X * sinh(κ(T-t)) / sinh(κT)
//! where κ = √(λσ²/η)

use std::time::Duration;

/// Configuration for Almgren-Chriss model
#[derive(Debug, Clone)]
pub struct AlmgrenChrissConfig {
    /// Permanent impact coefficient (γ) - price move per unit volume
    pub gamma: f64,
    /// Temporary impact coefficient (η) - cost per unit rate
    pub eta: f64,
    /// Volatility (σ) - price volatility per sqrt(time)
    pub sigma: f64,
    /// Risk aversion parameter (λ)
    pub lambda: f64,
    /// Minimum trade size to use AC model
    pub min_trade_size: f64,
}

impl Default for AlmgrenChrissConfig {
    fn default() -> Self {
        Self {
            gamma: 1e-5,      // 0.001% permanent impact per unit
            eta: 1e-4,        // 0.01% temporary impact per unit rate
            sigma: 0.02,      // 2% hourly volatility
            lambda: 1e-6,     // Risk aversion
            min_trade_size: 10.0,
        }
    }
}

impl AlmgrenChrissConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), AlmgrenChrissError> {
        if self.gamma < 0.0 {
            return Err(AlmgrenChrissError::NegativeGamma);
        }
        if self.eta <= 0.0 {
            return Err(AlmgrenChrissError::NonPositiveEta);
        }
        if self.sigma <= 0.0 {
            return Err(AlmgrenChrissError::NonPositiveSigma);
        }
        if self.lambda < 0.0 {
            return Err(AlmgrenChrissError::NegativeLambda);
        }
        Ok(())
    }
    
    /// Calculate characteristic decay parameter κ
    #[inline]
    pub fn kappa(&self) -> f64 {
        (self.lambda * self.sigma.powi(2) / self.eta).sqrt()
    }
}

/// Errors specific to Almgren-Chriss calculations
#[derive(Debug, Clone, PartialEq)]
pub enum AlmgrenChrissError {
    NegativeGamma,
    NonPositiveEta,
    NonPositiveSigma,
    NegativeLambda,
    InvalidTimeHorizon,
    InvalidTradeSize,
    NumericalOverflow,
}

/// Result of optimal trajectory calculation
#[derive(Debug, Clone)]
pub struct ExecutionTrajectory {
    /// Total quantity to execute
    pub total_quantity: f64,
    /// Time horizon in seconds
    pub time_horizon_s: f64,
    /// Number of intervals
    pub n_intervals: usize,
    /// Interval duration in seconds
    pub interval_duration_s: f64,
    /// Quantities to trade in each interval
    pub quantities: Vec<f64>,
    /// Cumulative quantities traded
    pub cumulative: Vec<f64>,
    /// Expected cost
    pub expected_cost: f64,
    /// Cost variance
    pub cost_variance: f64,
    /// Implementation shortfall
    pub implementation_shortfall: f64,
}

impl ExecutionTrajectory {
    /// Get remaining quantity at interval i
    pub fn remaining_at(&self, i: usize) -> f64 {
        if i >= self.cumulative.len() {
            return 0.0;
        }
        self.total_quantity - self.cumulative[i]
    }
    
    /// Get participation rate at interval i
    pub fn participation_rate_at(&self, i: usize, market_volume: f64) -> f64 {
        if market_volume <= 0.0 || i >= self.quantities.len() {
            return 0.0;
        }
        self.quantities[i] / market_volume
    }
}

/// Almgren-Chriss optimal execution solver
pub struct AlmgrenChrissSolver {
    config: AlmgrenChrissConfig,
}

impl AlmgrenChrissSolver {
    /// Create new solver with given configuration
    pub fn new(config: AlmgrenChrissConfig) -> Result<Self, AlmgrenChrissError> {
        config.validate()?;
        Ok(Self { config })
    }
    
    /// Calculate optimal execution trajectory
    ///
    /// Solves for the optimal trading schedule that minimizes:
    /// Expected Cost + λ * Variance
    ///
    /// Args:
    ///   total_quantity: Total amount to trade (positive=sell, negative=buy)
    ///   time_horizon: Maximum time allowed for execution
    ///   n_intervals: Number of trading intervals
    ///
    /// Returns:
    ///   ExecutionTrajectory with optimal quantities per interval
    pub fn solve(
        &self,
        total_quantity: f64,
        time_horizon: Duration,
        n_intervals: usize,
    ) -> Result<ExecutionTrajectory, AlmgrenChrissError> {
        if total_quantity.abs() < self.config.min_trade_size {
            // For small trades, use TWAP (equal intervals)
            return self.solve_twap(total_quantity, time_horizon, n_intervals);
        }
        
        let time_horizon_s = time_horizon.as_secs_f64();
        if time_horizon_s <= 0.0 {
            return Err(AlmgrenChrissError::InvalidTimeHorizon);
        }
        
        let n = n_intervals.max(2);
        let dt = time_horizon_s / n as f64;
        let kappa = self.config.kappa();
        
        // Calculate hyperbolic terms
        let kappa_t = kappa * time_horizon_s;
        let sinh_kappa_t = kappa_t.sinh();
        
        if !sinh_kappa_t.is_finite() || sinh_kappa_t.abs() < 1e-10 {
            // Fall back to TWAP for numerical stability
            return self.solve_twap(total_quantity, time_horizon, n_intervals);
        }
        
        // Calculate optimal trajectory using closed-form solution
        // x*(t) = X * sinh(κ(T-t)) / sinh(κT)
        let mut quantities = Vec::with_capacity(n);
        let mut cumulative = Vec::with_capacity(n);
        let mut cum_sum = 0.0;
        
        for i in 0..n {
            let t_i = i as f64 * dt;
            let t_next = (i + 1) as f64 * dt;
            
            // Remaining quantity at start of interval
            let remaining_start = total_quantity * (kappa * (time_horizon_s - t_i)).sinh() / sinh_kappa_t;
            let remaining_end = total_quantity * (kappa * (time_horizon_s - t_next)).sinh() / sinh_kappa_t;
            
            // Trade quantity in this interval
            let q_i = (remaining_start - remaining_end).max(0.0);
            quantities.push(q_i);
            cum_sum += q_i;
            cumulative.push(cum_sum);
        }
        
        // Normalize to ensure total is executed
        let sum_q: f64 = quantities.iter().sum();
        if sum_q > 0.0 {
            let scale = total_quantity.abs() / sum_q;
            for q in &mut quantities {
                *q *= scale;
            }
            cumulative.clear();
            cum_sum = 0.0;
            for &q in &quantities {
                cum_sum += q;
                cumulative.push(cum_sum);
            }
        }
        
        // Calculate expected cost
        let expected_cost = self.calculate_expected_cost(total_quantity, time_horizon_s, &quantities, dt);
        
        // Calculate cost variance
        let cost_variance = self.calculate_cost_variance(&quantities, dt);
        
        // Implementation shortfall
        let implementation_shortfall = expected_cost + 3.0 * cost_variance.sqrt();
        
        Ok(ExecutionTrajectory {
            total_quantity: total_quantity.abs(),
            time_horizon_s,
            n_intervals: n,
            interval_duration_s: dt,
            quantities,
            cumulative,
            expected_cost,
            cost_variance,
            implementation_shortfall,
        })
    }
    
    /// Solve using TWAP (Time-Weighted Average Price) as fallback
    fn solve_twap(
        &self,
        total_quantity: f64,
        time_horizon: Duration,
        n_intervals: usize,
    ) -> Result<ExecutionTrajectory, AlmgrenChrissError> {
        let time_horizon_s = time_horizon.as_secs_f64();
        let n = n_intervals.max(1);
        let dt = time_horizon_s / n as f64;
        let q_per_interval = total_quantity.abs() / n as f64;
        
        let quantities = vec![q_per_interval; n];
        let cumulative: Vec<f64> = quantities
            .iter()
            .scan(0.0, |acc, &q| {
                *acc += q;
                Some(*acc)
            })
            .collect();
        
        // TWAP expected cost (only permanent + temporary impact, no timing risk optimization)
        let expected_cost = self.config.gamma * total_quantity.powi(2) / 2.0
            + self.config.eta * total_quantity.powi(2) / time_horizon_s;
        
        // TWAP variance (higher than optimized)
        let cost_variance = self.config.sigma.powi(2) * total_quantity.powi(2) * time_horizon_s / 3.0;
        
        Ok(ExecutionTrajectory {
            total_quantity: total_quantity.abs(),
            time_horizon_s,
            n_intervals: n,
            interval_duration_s: dt,
            quantities,
            cumulative,
            expected_cost,
            cost_variance,
            implementation_shortfall: expected_cost + 3.0 * cost_variance.sqrt(),
        })
    }
    
    /// Calculate expected execution cost
    fn calculate_expected_cost(
        &self,
        total_quantity: f64,
        time_horizon_s: f64,
        quantities: &[f64],
        dt: f64,
    ) -> f64 {
        let mut cost = 0.0;
        let mut remaining = total_quantity.abs();
        
        for &q in quantities {
            // Permanent impact: γ * q * remaining
            cost += self.config.gamma * q * remaining;
            
            // Temporary impact: η * q² / dt
            cost += self.config.eta * q.powi(2) / dt;
            
            remaining -= q;
        }
        
        cost
    }
    
    /// Calculate variance of execution cost
    fn calculate_cost_variance(&self, quantities: &[f64], dt: f64) -> f64 {
        let mut variance = 0.0;
        let mut remaining = 0.0;
        
        // Accumulate remaining position variance
        for &q in quantities {
            remaining += q;
            variance += self.config.sigma.powi(2) * remaining.powi(2) * dt;
        }
        
        variance * self.config.lambda
    }
    
    /// Get optimal time horizon for given trade size
    ///
    /// Balances market impact (increases with speed) vs timing risk (increases with time)
    pub fn optimal_time_horizon(&self, quantity: f64, avg_daily_volume: f64) -> Duration {
        // Rule of thumb: T* = √(2ηQ / (λσ²ADV))
        // Simplified: execute over fraction of daily volume
        
        let participation_limit = 0.1; // Max 10% of ADV
        let min_time_s = quantity / (avg_daily_volume * participation_limit) * 86400.0;
        
        // Cap at 4 hours for our trading window
        let capped_time = min_time_s.min(4.0 * 3600.0);
        
        Duration::from_secs_f64(capped_time.max(60.0)) // Minimum 1 minute
    }
    
    /// Update configuration dynamically based on market conditions
    pub fn update_config(&mut self, config: AlmgrenChrissConfig) -> Result<(), AlmgrenChrissError> {
        config.validate()?;
        self.config = config;
        Ok(())
    }
    
    /// Get current configuration
    pub fn config(&self) -> &AlmgrenChrissConfig {
        &self.config
    }
}

/// Impact-aware order scheduler for real-time execution
pub struct ExecutionScheduler {
    solver: AlmgrenChrissSolver,
    current_trajectory: Option<ExecutionTrajectory>,
    current_interval: usize,
    executed_quantity: f64,
}

impl ExecutionScheduler {
    /// Create new scheduler
    pub fn new(config: AlmgrenChrissConfig) -> Result<Self, AlmgrenChrissError> {
        let solver = AlmgrenChrissSolver::new(config)?;
        Ok(Self {
            solver,
            current_trajectory: None,
            current_interval: 0,
            executed_quantity: 0.0,
        })
    }
    
    /// Initialize execution for a new order
    pub fn start_execution(
        &mut self,
        quantity: f64,
        time_horizon: Duration,
        n_intervals: usize,
    ) -> Result<&ExecutionTrajectory, AlmgrenChrissError> {
        let trajectory = self.solver.solve(quantity, time_horizon, n_intervals)?;
        self.executed_quantity = 0.0;
        self.current_interval = 0;
        self.current_trajectory = Some(trajectory);
        self.current_trajectory.as_ref().unwrap()
    }
    
    /// Get next order quantity
    pub fn next_order(&mut self) -> Option<f64> {
        if let Some(traj) = &self.current_trajectory {
            if self.current_interval < traj.quantities.len() {
                let q = traj.quantities[self.current_interval];
                self.current_interval += 1;
                self.executed_quantity += q;
                return Some(q);
            }
        }
        None
    }
    
    /// Get progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        if let Some(traj) = &self.current_trajectory {
            self.executed_quantity / traj.total_quantity
        } else {
            0.0
        }
    }
    
    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        self.progress() >= 1.0
    }
    
    /// Reset scheduler
    pub fn reset(&mut self) {
        self.current_trajectory = None;
        self.current_interval = 0;
        self.executed_quantity = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solver_creation() {
        let config = AlmgrenChrissConfig::default();
        let solver = AlmgrenChrissSolver::new(config);
        assert!(solver.is_ok());
    }
    
    #[test]
    fn test_twap_fallback() {
        let config = AlmgrenChrissConfig::default();
        let solver = AlmgrenChrissSolver::new(config).unwrap();
        
        // Small trade should use TWAP
        let result = solver.solve(5.0, Duration::from_secs(3600), 10);
        assert!(result.is_ok());
        
        let traj = result.unwrap();
        // TWAP should have equal quantities
        let first = traj.quantities[0];
        for &q in &traj.quantities {
            assert!((q - first).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_optimal_trajectory() {
        let config = AlmgrenChrissConfig {
            gamma: 1e-5,
            eta: 1e-4,
            sigma: 0.02,
            lambda: 1e-5, // Higher risk aversion
            ..Default::default()
        };
        let solver = AlmgrenChrissSolver::new(config).unwrap();
        
        let result = solver.solve(100.0, Duration::from_secs(3600), 10);
        assert!(result.is_ok());
        
        let traj = result.unwrap();
        // First interval should trade more than last (front-loaded due to risk)
        assert!(traj.quantities[0] > traj.quantities[9]);
    }
}
