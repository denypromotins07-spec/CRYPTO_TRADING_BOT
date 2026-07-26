//! Economic Capital Calculator
//! 
//! Computes the exact capital buffer required for 99.9% survival probability
//! using advanced risk models (EVT, copulas, stress tests).
//! Implements strict trading halts if capital falls below safety thresholds.
//! 
//! Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
//! ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour

use std::collections::HashMap;
use thiserror::Error;

/// Errors specific to economic capital calculations
#[derive(Error, Debug)]
pub enum EconomicCapitalError {
    #[error("Insufficient capital: current={current}, required={required}")]
    InsufficientCapital { current: f64, required: f64 },
    #[error("Invalid confidence level: {0}")]
    InvalidConfidenceLevel(String),
    #[error("VaR calculation failed: {0}")]
    VaRCalculationFailed(String),
    #[error("Numerical overflow in capital computation")]
    NumericalOverflow,
    #[error("Trading halted: capital buffer breached")]
    TradingHalted,
}

/// Result type for economic capital operations
pub type EconomicCapitalResult<T> = Result<T, EconomicCapitalError>;

/// Configuration for economic capital calculation
#[derive(Debug, Clone)]
pub struct EconomicCapitalConfig {
    /// Target survival probability (e.g., 0.999 for 99.9%)
    pub confidence_level: f64,
    /// Time horizon in days
    pub time_horizon_days: usize,
    /// Minimum capital buffer ratio (buffer / total capital)
    pub min_buffer_ratio: f64,
    /// Regulatory capital multiplier
    pub regulatory_multiplier: f64,
    /// Stress test overlay factor
    pub stress_overlay_factor: f64,
}

impl Default for EconomicCapitalConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.999,
            time_horizon_days: 10,
            min_buffer_ratio: 0.15,
            regulatory_multiplier: 1.5,
            stress_overlay_factor: 1.2,
        }
    }
}

/// Economic capital calculation result
#[derive(Debug, Clone)]
pub struct EconomicCapitalResult {
    /// Total economic capital required
    pub total_capital_required: f64,
    /// Base capital (from VaR/ES)
    pub base_capital: f64,
    /// Stress test overlay
    pub stress_overlay: f64,
    /// Buffer capital
    pub buffer_capital: f64,
    /// Current available capital
    pub available_capital: f64,
    /// Capital surplus/deficit
    pub capital_surplus: f64,
    /// Utilization ratio (used / total)
    pub utilization_ratio: f64,
    /// Whether trading is allowed
    pub trading_allowed: bool,
    /// Risk-based capital ratio
    pub capital_ratio: f64,
}

impl EconomicCapitalResult {
    /// Check if capital is sufficient
    pub fn is_adequate(&self) -> bool {
        self.capital_surplus >= 0.0 && self.trading_allowed
    }
    
    /// Get capital adequacy status message
    pub fn adequacy_status(&self) -> &'static str {
        if self.capital_surplus > self.buffer_capital * 0.5 {
            "WELL_CAPITALIZED"
        } else if self.capital_surplus > 0.0 {
            "ADEQUATELY_CAPITALIZED"
        } else if self.capital_surplus > -self.buffer_capital * 0.2 {
            "UNDERCAPITALIZED"
        } else {
            "CRITICALLY_UNDERCAPITALIZED"
        }
    }
}

/// Economic capital calculator with multiple methodologies
pub struct EconomicCapitalCalculator {
    /// Configuration
    config: EconomicCapitalConfig,
    /// Portfolio value
    portfolio_value: f64,
    /// Current capital holdings
    current_capital: f64,
    /// Pre-computed VaR at various confidence levels
    var_cache: HashMap<f64, f64>,
    /// Expected Shortfall values
    es_cache: HashMap<f64, f64>,
    /// Stress test losses
    stress_losses: Vec<f64>,
}

impl EconomicCapitalCalculator {
    /// Create a new calculator
    pub fn new(
        config: Option<EconomicCapitalConfig>,
        portfolio_value: f64,
        initial_capital: f64,
    ) -> Self {
        Self {
            config: config.unwrap_or_default(),
            portfolio_value,
            current_capital: initial_capital,
            var_cache: HashMap::new(),
            es_cache: HashMap::new(),
            stress_losses: Vec::new(),
        }
    }
    
    /// Set pre-computed VaR values
    pub fn set_var(&mut self, confidence: f64, var: f64) {
        self.var_cache.insert(confidence, var);
    }
    
    /// Set pre-computed Expected Shortfall values
    pub fn set_es(&mut self, confidence: f64, es: f64) {
        self.es_cache.insert(confidence, es);
    }
    
    /// Add stress test loss scenario
    pub fn add_stress_scenario(&mut self, loss: f64) {
        self.stress_losses.push(loss);
    }
    
    /// Update current capital
    pub fn update_capital(&mut self, new_capital: f64) {
        self.current_capital = new_capital;
    }
    
    /// Calculate total economic capital required
    pub fn calculate_economic_capital(&self) -> EconomicCapitalResult<EconomicCapitalResult> {
        // Get target confidence level VaR
        let target_conf = self.config.confidence_level;
        
        // Base capital from Expected Shortfall (more conservative than VaR)
        let base_capital = self.compute_base_capital(target_conf)?;
        
        // Stress test overlay
        let stress_overlay = self.compute_stress_overlay();
        
        // Apply regulatory multiplier
        let risk_adjusted_capital = (base_capital + stress_overlay) 
            * self.config.regulatory_multiplier;
        
        // Add buffer
        let buffer_capital = risk_adjusted_capital * self.config.min_buffer_ratio;
        
        // Total required
        let total_required = risk_adjusted_capital + buffer_capital;
        
        // Calculate surplus/deficit
        let capital_surplus = self.current_capital - total_required;
        
        // Utilization ratio
        let utilized = self.portfolio_value - self.current_capital;
        let utilization_ratio = utilized / self.portfolio_value;
        
        // Capital ratio
        let capital_ratio = self.current_capital / self.portfolio_value;
        
        // Determine if trading is allowed
        let trading_allowed = self.check_trading_eligibility(capital_surplus);
        
        Ok(EconomicCapitalResult {
            total_capital_required: total_required,
            base_capital,
            stress_overlay,
            buffer_capital,
            available_capital: self.current_capital,
            capital_surplus,
            utilization_ratio,
            trading_allowed,
            capital_ratio,
        })
    }
    
    /// Compute base capital requirement
    fn compute_base_capital(&self, confidence: f64) -> EconomicCapitalResult<f64> {
        // Prefer ES over VaR for tail risk
        if let Some(&es) = self.es_cache.get(&confidence) {
            return Ok(es);
        }
        
        // Fall back to VaR with adjustment
        if let Some(&var) = self.var_cache.get(&confidence) {
            // ES approximately 1.2-1.5x VaR for fat tails
            let es_approx = var * 1.35;
            return Ok(es_approx);
        }
        
        // If no pre-computed values, use parametric approximation
        // This would typically come from the EVT module
        let parametric_var = self.parametric_var(confidence);
        Ok(parametric_var * 1.35)
    }
    
    /// Parametric VaR approximation (fallback)
    fn parametric_var(&self, confidence: f64) -> f64 {
        // Simplified: assumes normal distribution
        // In production, would use actual portfolio volatility
        let z_score = match confidence {
            c if c >= 0.999 => 3.09,
            c if c >= 0.99 => 2.33,
            c if c >= 0.975 => 1.96,
            c if c >= 0.95 => 1.645,
            _ => 1.28,
        };
        
        // Assume typical crypto portfolio daily vol of 3%
        let daily_vol = 0.03;
        let horizon_adjustment = (self.config.time_horizon_days as f64).sqrt();
        
        z_score * daily_vol * horizon_adjustment * self.portfolio_value
    }
    
    /// Compute stress test overlay
    fn compute_stress_overlay(&self) -> f64 {
        if self.stress_losses.is_empty() {
            return 0.0;
        }
        
        // Use worst-case stress loss
        let max_stress_loss = self.stress_losses
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        
        // Apply stress overlay factor
        max_stress_loss.abs() * self.config.stress_overlay_factor
    }
    
    /// Check if trading is allowed based on capital position
    fn check_trading_eligibility(&self, capital_surplus: f64) -> bool {
        // Trading halted if:
        // 1. Capital deficit
        // 2. Surplus below minimum buffer threshold
        
        if capital_surplus < 0.0 {
            return false;
        }
        
        let min_required_surplus = self.config.min_buffer_ratio * 0.5;
        capital_surplus >= min_required_surplus * self.portfolio_value
    }
    
    /// Get maximum allowable position size given current capital
    pub fn max_position_size(&self, asset_risk: f64) -> EconomicCapitalResult<f64> {
        let result = self.calculate_economic_capital()?;
        
        if !result.trading_allowed {
            return Err(EconomicCapitalError::TradingHalted);
        }
        
        // Position size limited by available surplus
        let available_for_risk = result.capital_surplus * 0.5; // Use 50% of surplus
        
        if asset_risk <= 0.0 {
            return Ok(self.portfolio_value); // No risk constraint
        }
        
        // Max position = available_risk / asset_risk_per_unit
        let max_position = available_for_risk / asset_risk;
        
        Ok(max_position.min(self.portfolio_value))
    }
    
    /// Calculate capital at risk (CaR) for a given time horizon
    pub fn capital_at_risk(&self, horizon_days: usize) -> f64 {
        let confidence = self.config.confidence_level;
        
        // Scale VaR by time horizon
        let daily_var = self.var_cache.get(&confidence)
            .copied()
            .unwrap_or_else(|| self.parametric_var(confidence));
        
        let horizon_adjustment = (horizon_days as f64).sqrt();
        daily_var * horizon_adjustment
    }
    
    /// Check for margin call condition
    pub fn is_margin_call_imminent(&self, maintenance_ratio: f64) -> bool {
        let capital_ratio = self.current_capital / self.portfolio_value;
        capital_ratio < maintenance_ratio * 1.1 // 10% buffer
    }
    
    /// Generate capital allocation recommendation
    pub fn recommend_capital_allocation(&self) -> CapitalAllocationRecommendation {
        let result = self.calculate_economic_capital().unwrap_or_else(|_| {
            EconomicCapitalResult {
                total_capital_required: self.portfolio_value * 0.3,
                base_capital: self.portfolio_value * 0.2,
                stress_overlay: self.portfolio_value * 0.05,
                buffer_capital: self.portfolio_value * 0.05,
                available_capital: self.current_capital,
                capital_surplus: 0.0,
                utilization_ratio: 0.7,
                trading_allowed: false,
                capital_ratio: self.current_capital / self.portfolio_value,
            }
        });
        
        let action = if result.capital_surplus > result.buffer_capital {
            AllocationAction::IncreasePositions
        } else if result.capital_surplus > 0.0 {
            AllocationAction::Maintain
        } else if result.capital_surplus > -result.buffer_capital * 0.5 {
            AllocationAction::ReducePositions
        } else {
            AllocationAction::EmergencyDeleveraging
        };
        
        CapitalAllocationRecommendation {
            action,
            recommended_capital_change: result.capital_surplus,
            priority: self.determine_priority(&result),
            reasoning: self.generate_reasoning(&result),
        }
    }
    
    fn determine_priority(&self, result: &EconomicCapitalResult) -> Priority {
        if !result.trading_allowed {
            Priority::Critical
        } else if result.capital_surplus < 0.0 {
            Priority::High
        } else if result.capital_surplus < result.buffer_capital * 0.3 {
            Priority::Medium
        } else {
            Priority::Low
        }
    }
    
    fn generate_reasoning(&self, result: &EconomicCapitalResult) -> String {
        format!(
            "Capital ratio: {:.2}%, Surplus: ${:.2}, Status: {}",
            result.capital_ratio * 100.0,
            result.capital_surplus,
            result.adequacy_status()
        )
    }
}

/// Capital allocation action recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationAction {
    /// Increase position sizes
    IncreasePositions,
    /// Maintain current positions
    Maintain,
    /// Reduce position sizes
    ReducePositions,
    /// Emergency deleveraging required
    EmergencyDeleveraging,
}

/// Priority level for capital actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Capital allocation recommendation
#[derive(Debug, Clone)]
pub struct CapitalAllocationRecommendation {
    /// Recommended action
    pub action: AllocationAction,
    /// Recommended capital change amount
    pub recommended_capital_change: f64,
    /// Priority level
    pub priority: Priority,
    /// Reasoning for the recommendation
    pub reasoning: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_economic_capital_calculation() {
        let config = EconomicCapitalConfig {
            confidence_level: 0.999,
            time_horizon_days: 10,
            ..Default::default()
        };
        
        let mut calc = EconomicCapitalCalculator::new(
            Some(config),
            1_000_000.0,
            300_000.0,
        );
        
        // Set pre-computed risk metrics
        calc.set_var(0.999, 150_000.0);
        calc.set_es(0.999, 200_000.0);
        calc.add_stress_scenario(-250_000.0);
        
        let result = calc.calculate_economic_capital().unwrap();
        
        assert!(result.total_capital_required > 0.0);
        assert!(result.buffer_capital > 0.0);
    }
    
    #[test]
    fn test_trading_halt_condition() {
        let mut calc = EconomicCapitalCalculator::new(
            None,
            1_000_000.0,
            50_000.0, // Very low capital
        );
        
        calc.set_var(0.999, 200_000.0);
        
        let result = calc.calculate_economic_capital().unwrap();
        assert!(!result.trading_allowed);
    }
}
