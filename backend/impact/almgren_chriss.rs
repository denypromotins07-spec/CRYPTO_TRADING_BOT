//! Almgren-Chriss Optimal Execution Model
//!
//! This module implements the Almgren-Chriss model for optimal trade execution,
//! solving for the optimal trajectory to execute large block trades while minimizing
//! market impact and timing risk. Critical for executing within the 4-hour trading window.
//!
//! # Features
//! - Closed-form solution for optimal execution trajectory
//! - Linear and temporary/permanent impact separation
//! - Risk-aversion parameter tuning
//! - Time-constrained optimization for 4-hour windows
//! - Zero-cost abstractions for memory efficiency

use std::f64::consts::SQRT_2;

/// Configuration for Almgren-Chriss model
#[derive(Debug, Clone)]
pub struct AlmgrenChrissConfig {
    /// Total shares to execute
    pub total_shares: f64,
    /// Total time horizon in seconds
    pub time_horizon: f64,
    /// Number of time steps
    pub n_steps: usize,
    /// Price volatility (annualized)
    pub volatility: f64,
    /// Temporary impact coefficient (eta)
    pub temp_impact: f64,
    /// Permanent impact coefficient (gamma)
    pub perm_impact: f64,
    /// Risk aversion parameter (lambda)
    pub risk_aversion: f64,
    /// Initial price
    pub initial_price: f64,
}

impl Default for AlmgrenChrissConfig {
    fn default() -> Self {
        Self {
            total_shares: 100.0,
            time_horizon: 14400.0, // 4 hours in seconds
            n_steps: 100,
            volatility: 0.04, // 4% daily vol for BTC
            temp_impact: 1e-5,
            perm_impact: 1e-6,
            risk_aversion: 1e-7,
            initial_price: 50000.0,
        }
    }
}

/// Single step in the execution trajectory
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    /// Time index
    pub step: usize,
    /// Time in seconds from start
    pub time: f64,
    /// Shares to trade in this step (negative for sell)
    pub trade_size: f64,
    /// Remaining shares after this step
    pub remaining: f64,
    /// Expected price impact for this step
    pub expected_impact: f64,
    /// Cumulative cost so far
    pub cumulative_cost: f64,
}

/// Result of Almgren-Chriss optimization
#[derive(Debug, Clone)]
pub struct ExecutionTrajectory {
    /// Sequence of execution steps
    pub steps: Vec<ExecutionStep>,
    /// Total expected cost
    pub total_expected_cost: f64,
    /// Total variance of cost
    pub total_variance: f64,
    /// Implementation shortfall
    pub implementation_shortfall: f64,
    /// Average execution price
    pub avg_execution_price: f64,
}

impl ExecutionTrajectory {
    /// Get the trade size for a specific step
    pub fn get_trade_size(&self, step: usize) -> Option<f64> {
        self.steps.get(step).map(|s| s.trade_size)
    }
    
    /// Get remaining shares at a specific step
    pub fn get_remaining(&self, step: usize) -> Option<f64> {
        self.steps.get(step).map(|s| s.remaining)
    }
}

/// Almgren-Chriss optimizer for optimal execution
pub struct AlmgrenChrissOptimizer {
    config: AlmgrenChrissConfig,
    /// Time step size
    dt: f64,
    /// Trading rate factor (kappa)
    kappa: f64,
}

impl AlmgrenChrissOptimizer {
    /// Create a new optimizer with given configuration
    pub fn new(config: AlmgrenChrissConfig) -> Self {
        let dt = config.time_horizon / config.n_steps as f64;
        
        // Calculate kappa: sqrt(lambda * sigma^2 / (2 * eta))
        let kappa = ((config.risk_aversion * config.volatility.powi(2)) 
            / (2.0 * config.temp_impact)).sqrt();
        
        Self { config, dt, kappa }
    }
    
    /// Compute the optimal execution trajectory using closed-form solution
    /// 
    /// The optimal strategy is: x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)
    /// where X is total shares, T is time horizon, kappa is the trading rate
    pub fn compute_trajectory(&self) -> ExecutionTrajectory {
        let X = self.config.total_shares;
        let T = self.config.time_horizon;
        let kappa = self.kappa;
        let eta = self.config.temp_impact;
        let gamma = self.config.perm_impact;
        let lambda = self.config.risk_aversion;
        let sigma = self.config.volatility;
        
        // Precompute sinh(kappa * T)
        let kappa_T = kappa * T;
        let sinh_kappa_T = kappa_T.sinh();
        
        let mut steps = Vec::with_capacity(self.config.n_steps);
        let mut cumulative_cost = 0.0;
        let mut remaining = X;
        
        // Calculate expected cost components
        let mut total_temporary_impact = 0.0;
        let mut total_permanent_impact = 0.0;
        let mut timing_risk_variance = 0.0;
        
        for i in 0..self.config.n_steps {
            let t = i as f64 * self.dt;
            let t_next = (i + 1) as f64 * self.dt;
            
            // Optimal remaining shares at time t
            let x_t = if kappa_T > 0.001 {
                X * (kappa * (T - t)).sinh() / sinh_kappa_T
            } else {
                // Linear strategy when kappa is very small
                X * (1.0 - t / T)
            };
            
            // Optimal remaining shares at time t+1
            let x_next = if kappa_T > 0.001 {
                X * (kappa * (T - t_next)).sinh() / sinh_kappa_T
            } else {
                X * (1.0 - t_next / T)
            };
            
            // Trade size (positive for buying, negative for selling)
            let trade_size = x_t - x_next;
            remaining = x_next;
            
            // Temporary impact cost: eta * (trade_size / dt)^2 * dt
            let trading_rate = trade_size / self.dt;
            let temp_cost = eta * trading_rate.powi(2) * self.dt;
            total_temporary_impact += temp_cost;
            
            // Permanent impact cost: gamma * x_t * trade_size
            let perm_cost = gamma * x_t * trade_size;
            total_permanent_impact += perm_cost;
            
            // Timing risk contribution: lambda * sigma^2 * x_t^2 * dt
            let risk_contribution = lambda * sigma.powi(2) * x_t.powi(2) * self.dt;
            timing_risk_variance += risk_contribution;
            
            cumulative_cost += temp_cost + perm_cost.abs();
            
            steps.push(ExecutionStep {
                step: i,
                time: t,
                trade_size,
                remaining,
                expected_impact: temp_cost + perm_cost.abs(),
                cumulative_cost,
            });
        }
        
        // Total expected cost = temporary + permanent impact
        let total_expected_cost = total_temporary_impact + total_permanent_impact.abs();
        
        // Total variance from timing risk
        let total_variance = timing_risk_variance;
        
        // Implementation shortfall (cost relative to arrival price)
        let implementation_shortfall = total_expected_cost;
        
        // Average execution price
        let avg_execution_price = if X > 0.0 {
            self.config.initial_price + implementation_shortfall / X
        } else {
            self.config.initial_price
        };
        
        ExecutionTrajectory {
            steps,
            total_expected_cost,
            total_variance,
            implementation_shortfall,
            avg_execution_price,
        }
    }
    
    /// Compute optimal liquidation time given risk parameters
    /// 
    /// Returns the optimal time horizon for executing the given quantity
    pub fn optimal_time_horizon(&self) -> f64 {
        let X = self.config.total_shares;
        let eta = self.config.temp_impact;
        let lambda = self.config.risk_aversion;
        let sigma = self.config.volatility;
        
        // Optimal time: T* = sqrt(2 * eta * X^2 / (lambda * sigma^2))
        // But bounded by our 4-hour window
        let optimal = ((2.0 * eta * X.powi(2)) / (lambda * sigma.powi(2))).sqrt();
        
        // Constrain to maximum 4 hours
        optimal.min(14400.0)
    }
    
    /// Get the trading rate factor (kappa)
    pub fn kappa(&self) -> f64 {
        self.kappa
    }
    
    /// Update configuration and recalculate parameters
    pub fn update_config(&mut self, config: AlmgrenChrissConfig) {
        self.config = config;
        self.dt = self.config.time_horizon / self.config.n_steps as f64;
        self.kappa = ((self.config.risk_aversion * self.config.volatility.powi(2)) 
            / (2.0 * self.config.temp_impact)).sqrt();
    }
    
    /// Adjust for asset-specific parameters (BTC vs SOL)
    pub fn adjust_for_asset(&mut self, asset_type: AssetType) {
        match asset_type {
            AssetType::BTC => {
                self.config.temp_impact = 1e-5;
                self.config.perm_impact = 1e-6;
                self.config.volatility = 0.04;
            }
            AssetType::ETH => {
                self.config.temp_impact = 2e-5;
                self.config.perm_impact = 2e-6;
                self.config.volatility = 0.06;
            }
            AssetType::SOL => {
                self.config.temp_impact = 5e-5;
                self.config.perm_impact = 5e-6;
                self.config.volatility = 0.10;
            }
        }
    }
}

/// Asset type for parameter calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetType {
    BTC,
    ETH,
    SOL,
}

/// Extended model with non-linear impact (square-root law)
pub struct NonLinearAlmgrenChriss {
    base_optimizer: AlmgrenChrissOptimizer,
    /// Square-root impact exponent
    impact_exponent: f64,
    /// Impact coefficient for square-root law
    impact_coefficient: f64,
}

impl NonLinearAlmgrenChriss {
    /// Create with square-root impact (exponent = 0.5)
    pub fn new_square_root(base_config: AlmgrenChrissConfig) -> Self {
        Self {
            base_optimizer: AlmgrenChrissOptimizer::new(base_config.clone()),
            impact_exponent: 0.5,
            impact_coefficient: base_config.temp_impact,
        }
    }
    
    /// Create with custom impact exponent
    pub fn new(base_config: AlmgrenChrissConfig, exponent: f64, coefficient: f64) -> Self {
        Self {
            base_optimizer: AlmgrenChrissOptimizer::new(base_config),
            impact_exponent: exponent,
            impact_coefficient: coefficient,
        }
    }
    
    /// Compute trajectory with non-linear impact
    pub fn compute_trajectory(&self) -> ExecutionTrajectory {
        // Use base optimizer for initial trajectory
        let mut trajectory = self.base_optimizer.compute_trajectory();
        
        // Adjust costs for non-linear impact
        let mut adjusted_cost = 0.0;
        for step in &mut trajectory.steps {
            // Non-linear temporary impact: c * |trade_size|^alpha
            let nonlinear_temp = self.impact_coefficient 
                * step.trade_size.abs().powf(self.impact_exponent);
            
            step.expected_impact = nonlinear_temp;
            adjusted_cost += nonlinear_temp;
        }
        
        trajectory.total_expected_cost = adjusted_cost;
        trajectory.implementation_shortfall = adjusted_cost;
        
        trajectory
    }
    
    /// Dynamically switch between linear and square-root based on order size
    pub fn compute_adaptive_trajectory(&self, threshold: f64) -> ExecutionTrajectory {
        let X = self.base_optimizer.config.total_shares;
        
        if X.abs() > threshold {
            // Large order: use square-root impact
            self.compute_trajectory()
        } else {
            // Small order: use linear impact
            self.base_optimizer.compute_trajectory()
        }
    }
}

/// Utility functions for Almgren-Chriss calculations
pub mod utils {
    use super::*;
    
    /// Calculate efficient frontier for different risk aversion values
    pub fn efficient_frontier(
        config: AlmgrenChrissConfig,
        risk_range: &[f64],
    ) -> Vec<(f64, f64, f64)> {
        // Returns (risk_aversion, expected_cost, cost_variance)
        risk_range.iter().map(|&lambda| {
            let mut cfg = config.clone();
            cfg.risk_aversion = lambda;
            let optimizer = AlmgrenChrissOptimizer::new(cfg);
            let trajectory = optimizer.compute_trajectory();
            (lambda, trajectory.total_expected_cost, trajectory.total_variance)
        }).collect()
    }
    
    /// Calculate optimal risk aversion for given constraints
    pub fn optimal_risk_aversion(
        config: &AlmgrenChrissConfig,
        max_cost: f64,
    ) -> f64 {
        // Binary search for optimal lambda
        let mut low = 1e-10;
        let mut high = 1e-4;
        
        for _ in 0..50 {
            let mid = (low + high) / 2.0;
            let mut cfg = config.clone();
            cfg.risk_aversion = mid;
            let optimizer = AlmgrenChrissOptimizer::new(cfg);
            let trajectory = optimizer.compute_trajectory();
            
            if trajectory.total_expected_cost > max_cost {
                high = mid;
            } else {
                low = mid;
            }
        }
        
        (low + high) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_trajectory() {
        let config = AlmgrenChrissConfig::default();
        let optimizer = AlmgrenChrissOptimizer::new(config);
        let trajectory = optimizer.compute_trajectory();
        
        assert_eq!(trajectory.steps.len(), 100);
        assert!(trajectory.total_expected_cost > 0.0);
        
        // Verify remaining shares approaches zero
        let last_remaining = trajectory.steps.last().unwrap().remaining;
        assert!(last_remaining.abs() < 1.0);
    }
    
    #[test]
    fn test_asset_adjustment() {
        let mut config = AlmgrenChrissConfig::default();
        let mut optimizer = AlmgrenChrissOptimizer::new(config);
        
        optimizer.adjust_for_asset(AssetType::SOL);
        
        // SOL should have higher impact and volatility
        assert!(optimizer.config.temp_impact > 1e-5);
        assert!(optimizer.config.volatility > 0.04);
    }
    
    #[test]
    fn test_nonlinear_trajectory() {
        let config = AlmgrenChrissConfig::default();
        let nonlinear = NonLinearAlmgrenChriss::new_square_root(config);
        let trajectory = nonlinear.compute_trajectory();
        
        assert!(trajectory.total_expected_cost > 0.0);
    }
}
