//! Almgren-Chriss Optimal Execution Model
//! 
//! This module implements the Almgren-Chriss model for optimal trade execution,
//! which balances market impact against timing risk to minimize implementation shortfall.
//! 
//! The model solves for the optimal trading trajectory that minimizes:
//! - Permanent market impact (information leakage)
//! - Temporary market impact (liquidity consumption)
//! - Timing risk (price volatility during execution)
//! 
//! Key Features:
//! - Closed-form solution for linear impact
//! - Dynamic adjustment for 4-hour trading window constraints
//! - Asset-specific parameter calibration (BTC vs SOL)
//! - Zero-cost abstractions for high-frequency recalibration

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for Almgren-Chriss model parameters
#[derive(Debug, Clone)]
pub struct AlmgrenChrissConfig {
    /// Total shares to execute
    pub total_shares: f64,
    /// Total time horizon for execution (seconds)
    pub time_horizon_secs: f64,
    /// Number of trading intervals
    pub num_intervals: usize,
    /// Price volatility (annualized)
    pub volatility: f64,
    /// Linear temporary impact coefficient (eta)
    pub temp_impact_coeff: f64,
    /// Linear permanent impact coefficient (gamma)
    pub perm_impact_coeff: f64,
    /// Risk aversion parameter (lambda)
    pub risk_aversion: f64,
    /// Minimum trade size (to avoid dust)
    pub min_trade_size: f64,
}

impl Default for AlmgrenChrissConfig {
    fn default() -> Self {
        Self {
            total_shares: 1.0, // 1 BTC
            time_horizon_secs: 14400.0, // 4 hours
            num_intervals: 240, // Every minute
            volatility: 0.60, // 60% annualized for crypto
            temp_impact_coeff: 1e-5,
            perm_impact_coeff: 1e-6,
            risk_aversion: 0.5,
            min_trade_size: 0.0001,
        }
    }
}

/// Single trade in the execution schedule
#[derive(Debug, Clone)]
pub struct TradeScheduleItem {
    /// Time offset from start (seconds)
    pub time_offset_secs: f64,
    /// Quantity to trade
    pub quantity: f64,
    /// Cumulative quantity traded
    pub cumulative_quantity: f64,
    /// Expected price impact (bps)
    pub expected_impact_bps: f64,
    /// Confidence interval for impact (lower bound)
    pub impact_lower_bound: f64,
    /// Confidence interval for impact (upper bound)
    pub impact_upper_bound: f64,
}

/// Complete execution trajectory
#[derive(Debug, Clone)]
pub struct ExecutionTrajectory {
    /// List of trades in chronological order
    pub trades: Vec<TradeScheduleItem>,
    /// Total expected cost
    pub total_expected_cost: f64,
    /// Total variance of cost
    pub total_variance: f64,
    /// Utility function value (cost + risk_penalty * variance)
    pub utility: f64,
    /// Start timestamp
    pub start_time: Instant,
    /// Asset symbol
    pub asset: String,
}

impl ExecutionTrajectory {
    /// Get remaining quantity to execute at given time
    pub fn remaining_quantity(&self, elapsed_secs: f64) -> f64 {
        let executed = self.trades
            .iter()
            .filter(|t| t.time_offset_secs <= elapsed_secs)
            .map(|t| t.quantity)
            .sum();
        
        self.trades.last()
            .map(|t| t.cumulative_quantity)
            .unwrap_or(0.0) - executed
    }
    
    /// Get next trade to execute
    pub fn get_next_trade(&self, elapsed_secs: f64) -> Option<&TradeScheduleItem> {
        self.trades.iter().find(|t| t.time_offset_secs > elapsed_secs)
    }
}

/// Almgren-Chriss optimal execution solver
pub struct AlmgrenChrissSolver {
    config: AlmgrenChrissConfig,
    /// Current asset being traded
    current_asset: String,
    /// Cache for computed trajectories
    trajectory_cache: Option<ExecutionTrajectory>,
    /// Last computation timestamp
    last_computation: Instant,
}

impl AlmgrenChrissSolver {
    /// Create a new solver with given configuration
    pub fn new(config: AlmgrenChrissConfig) -> Self {
        Self {
            config,
            current_asset: "BTC".to_string(),
            trajectory_cache: None,
            last_computation: Instant::now(),
        }
    }
    
    /// Update configuration and invalidate cache
    pub fn update_config(&mut self, config: AlmgrenChrissConfig) {
        self.config = config;
        self.trajectory_cache = None;
    }
    
    /// Set current asset for asset-specific calibration
    pub fn set_asset(&mut self, asset: &str) {
        self.current_asset = asset.to_string();
        self.trajectory_cache = None; // Recalibrate for new asset
        
        // Apply asset-specific adjustments
        self.apply_asset_calibration();
    }
    
    /// Apply asset-specific parameter calibration
    fn apply_asset_calibration(&mut self) {
        match self.current_asset.as_str() {
            "BTC" => {
                // Bitcoin: lower impact due to deep liquidity
                self.config.temp_impact_coeff = 1e-5;
                self.config.perm_impact_coeff = 1e-6;
                self.config.volatility = 0.60;
            }
            "ETH" => {
                // Ethereum: moderate impact
                self.config.temp_impact_coeff = 1.5e-5;
                self.config.perm_impact_coeff = 1.5e-6;
                self.config.volatility = 0.70;
            }
            "SOL" => {
                // Solana: higher impact due to lower liquidity
                self.config.temp_impact_coeff = 3e-5;
                self.config.perm_impact_coeff = 2e-6;
                self.config.volatility = 0.90;
            }
            "USDT" => {
                // Stablecoin: minimal volatility
                self.config.temp_impact_coeff = 5e-6;
                self.config.perm_impact_coeff = 5e-7;
                self.config.volatility = 0.05;
            }
            _ => {}
        }
    }
    
    /// Compute optimal execution trajectory
    /// 
    /// Uses the closed-form solution from Almgren-Chriss (2001)
    pub fn compute_trajectory(&mut self) -> ExecutionTrajectory {
        // Return cached result if still valid (within 1 second)
        if let Some(ref cached) = self.trajectory_cache {
            if self.last_computation.elapsed() < Duration::from_secs(1) {
                return cached.clone();
            }
        }
        
        let dt = self.config.time_horizon_secs / self.config.num_intervals as f64;
        let tau = self.config.time_horizon_secs;
        
        // Calculate key parameters
        let kappa = self.calculate_kappa(dt);
        let kappa_tilde = self.calculate_kappa_tilde(dt);
        
        // Compute optimal trading schedule
        let mut trades = Vec::with_capacity(self.config.num_intervals);
        let mut cumulative = 0.0;
        let mut total_expected_cost = 0.0;
        let mut total_variance = 0.0;
        
        for k in 0..self.config.num_intervals {
            let t_k = k as f64 * dt;
            
            // Optimal position at time t_k
            let q_k = self.optimal_position(t_k, kappa);
            let q_k_plus_1 = self.optimal_position(t_k + dt, kappa);
            
            // Trade size
            let trade_size = (q_k - q_k_plus_1).max(self.config.min_trade_size);
            cumulative += trade_size;
            
            // Expected impact for this trade
            let expected_impact = self.calculate_trade_impact(trade_size, dt);
            total_expected_cost += expected_impact;
            
            // Variance contribution
            let variance_contrib = self.calculate_variance_contribution(t_k, dt, kappa);
            total_variance += variance_contrib;
            
            trades.push(TradeScheduleItem {
                time_offset_secs: t_k,
                quantity: trade_size,
                cumulative_quantity: cumulative,
                expected_impact_bps: expected_impact * 10000.0, // Convert to bps
                impact_lower_bound: expected_impact * 0.8,
                impact_upper_bound: expected_impact * 1.2,
            });
        }
        
        // Calculate utility
        let utility = total_expected_cost + self.config.risk_aversion * total_variance;
        
        let trajectory = ExecutionTrajectory {
            trades,
            total_expected_cost,
            total_variance,
            utility,
            start_time: Instant::now(),
            asset: self.current_asset.clone(),
        };
        
        // Cache result
        self.trajectory_cache = Some(trajectory.clone());
        self.last_computation = Instant::now();
        
        trajectory
    }
    
    /// Calculate optimal position at time t
    fn optimal_position(&self, t: f64, kappa: f64) -> f64 {
        let tau = self.config.time_horizon_secs;
        
        if kappa.abs() < 1e-10 {
            // Risk-neutral limit: linear schedule
            return self.config.total_shares * (1.0 - t / tau);
        }
        
        // General solution with risk aversion
        let sinh_kappa_tau = (kappa * tau).sinh();
        let sinh_kappa_t = (kappa * t).sinh();
        let sinh_kappa_tau_minus_t = (kappa * (tau - t)).sinh();
        
        if sinh_kappa_tau.abs() < 1e-10 {
            return self.config.total_shares * (1.0 - t / tau);
        }
        
        self.config.total_shares * sinh_kappa_tau_minus_t / sinh_kappa_tau
    }
    
    /// Calculate kappa parameter
    fn calculate_kappa(&self, dt: f64) -> f64 {
        let eta = self.config.temp_impact_coeff;
        let gamma = self.config.perm_impact_coeff;
        let sigma = self.config.volatility / std::f64::consts::SQRT_252.0; // Daily vol
        let lambda = self.config.risk_aversion;
        
        let numerator = lambda * sigma * sigma * dt;
        let denominator = 2.0 * eta + gamma * dt;
        
        if denominator.abs() < 1e-10 {
            return 0.0;
        }
        
        (numerator / denominator).sqrt()
    }
    
    /// Calculate kappa_tilde (adjusted for discrete time)
    fn calculate_kappa_tilde(&self, dt: f64) -> f64 {
        let kappa = self.calculate_kappa(dt);
        let eta = self.config.temp_impact_coeff;
        let gamma = self.config.perm_impact_coeff;
        
        let numerator = 2.0 * eta * kappa * kappa * dt;
        let denominator = 2.0 * eta + gamma * dt;
        
        if denominator.abs() < 1e-10 {
            return kappa;
        }
        
        (kappa * kappa + numerator / denominator).sqrt()
    }
    
    /// Calculate expected impact for a single trade
    fn calculate_trade_impact(&self, quantity: f64, dt: f64) -> f64 {
        let eta = self.config.temp_impact_coeff;
        let gamma = self.config.perm_impact_coeff;
        
        // Temporary impact (linear)
        let temp_impact = eta * quantity / dt;
        
        // Permanent impact (linear)
        let perm_impact = gamma * quantity;
        
        temp_impact + perm_impact
    }
    
    /// Calculate variance contribution at time t
    fn calculate_variance_contribution(&self, t: f64, dt: f64, kappa: f64) -> f64 {
        let sigma = self.config.volatility / std::f64::consts::SQRT_252.0;
        let q_t = self.optimal_position(t, kappa);
        
        // Variance from holding position q_t over interval dt
        sigma * sigma * q_t * q_t * dt
    }
    
    /// Get recommended execution speed (shares per second)
    pub fn get_execution_rate(&mut self, elapsed_secs: f64) -> f64 {
        let trajectory = self.compute_trajectory();
        
        if let Some(trade) = trajectory.get_next_trade(elapsed_secs) {
            let dt = self.config.time_horizon_secs / self.config.num_intervals as f64;
            trade.quantity / dt
        } else {
            0.0
        }
    }
    
    /// Check if execution is on track
    pub fn check_execution_progress(
        &mut self,
        elapsed_secs: f64,
        executed_quantity: f64,
    ) -> ExecutionStatus {
        let trajectory = self.compute_trajectory();
        
        let expected_executed = trajectory.trades
            .iter()
            .filter(|t| t.time_offset_secs <= elapsed_secs)
            .map(|t| t.quantity)
            .sum::<f64>();
        
        let deviation = executed_quantity - expected_executed;
        let deviation_pct = if expected_executed > 0.0 {
            deviation / expected_executed
        } else {
            0.0
        };
        
        let status = if deviation_pct.abs() < 0.05 {
            ExecutionStatus::OnTrack
        } else if deviation < 0.0 {
            ExecutionStatus::BehindSchedule
        } else {
            ExecutionStatus::AheadOfSchedule
        };
        
        ExecutionStatus {
            status,
            expected_executed,
            actual_executed: executed_quantity,
            deviation,
            deviation_pct,
            recommendation: self.get_adjustment_recommendation(status, deviation_pct),
        }
    }
    
    /// Get recommendation for adjusting execution
    fn get_adjustment_recommendation(&self, status: ExecutionState, deviation_pct: f64) -> &'static str {
        match status {
            ExecutionState::OnTrack => "Continue current execution pace",
            ExecutionState::BehindSchedule => "Increase trade sizes or reduce interval",
            ExecutionState::AheadOfSchedule => "Consider slowing down to reduce impact",
        }
    }
    
    /// Reset solver state
    pub fn reset(&mut self) {
        self.trajectory_cache = None;
        self.last_computation = Instant::now();
    }
}

/// Execution status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionState {
    OnTrack,
    BehindSchedule,
    AheadOfSchedule,
}

/// Detailed execution status
#[derive(Debug, Clone)]
pub struct ExecutionStatus {
    pub status: ExecutionState,
    pub expected_executed: f64,
    pub actual_executed: f64,
    pub deviation: f64,
    pub deviation_pct: f64,
    pub recommendation: &'static str,
}

/// Builder pattern for constructing Almgren-Chriss configurations
pub struct AlmgrenChrissBuilder {
    config: AlmgrenChrissConfig,
}

impl AlmgrenChrissBuilder {
    pub fn new() -> Self {
        Self {
            config: AlmgrenChrissConfig::default(),
        }
    }
    
    pub fn total_shares(mut self, shares: f64) -> Self {
        self.config.total_shares = shares;
        self
    }
    
    pub fn time_horizon_hours(mut self, hours: f64) -> Self {
        self.config.time_horizon_secs = hours * 3600.0;
        self
    }
    
    pub fn volatility(mut self, vol: f64) -> Self {
        self.config.volatility = vol;
        self
    }
    
    pub fn risk_aversion(mut self, lambda: f64) -> Self {
        self.config.risk_aversion = lambda;
        self
    }
    
    pub fn temp_impact(mut self, eta: f64) -> Self {
        self.config.temp_impact_coeff = eta;
        self
    }
    
    pub fn perm_impact(mut self, gamma: f64) -> Self {
        self.config.perm_impact_coeff = gamma;
        self
    }
    
    pub fn build(self) -> AlmgrenChrissConfig {
        self.config
    }
}

impl Default for AlmgrenChrissBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trajectory_computation() {
        let config = AlmgrenChrissConfig {
            total_shares: 10.0,
            time_horizon_secs: 3600.0, // 1 hour
            num_intervals: 60,
            ..Default::default()
        };
        
        let mut solver = AlmgrenChrissSolver::new(config);
        solver.set_asset("BTC");
        
        let trajectory = solver.compute_trajectory();
        
        // Verify total quantity matches
        let total_traded: f64 = trajectory.trades.iter().map(|t| t.quantity).sum();
        assert!((total_traded - 10.0).abs() < 0.1);
        
        // Verify positive costs
        assert!(trajectory.total_expected_cost > 0.0);
        assert!(trajectory.total_variance > 0.0);
    }
    
    #[test]
    fn test_asset_calibration() {
        let mut solver = AlmgrenChrissSolver::new(AlmgrenChrissConfig::default());
        
        solver.set_asset("BTC");
        let btc_impact = solver.config.temp_impact_coeff;
        
        solver.set_asset("SOL");
        let sol_impact = solver.config.temp_impact_coeff;
        
        // SOL should have higher impact than BTC
        assert!(sol_impact > btc_impact);
    }
    
    #[test]
    fn test_builder_pattern() {
        let config = AlmgrenChrissBuilder::new()
            .total_shares(5.0)
            .time_horizon_hours(2.0)
            .volatility(0.8)
            .risk_aversion(1.0)
            .build();
        
        assert_eq!(config.total_shares, 5.0);
        assert_eq!(config.time_horizon_secs, 7200.0);
        assert_eq!(config.volatility, 0.8);
    }
}
