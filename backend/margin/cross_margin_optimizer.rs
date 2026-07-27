//! Cross-Margin Optimizer - Optimal Collateral Distribution
//! 
//! This module calculates optimal collateral distribution across accounts
//! to maximize capital efficiency while maintaining a 20% buffer above
//! maintenance margin requirements to prevent margin calls.
//! 
//! Key Features:
//! - Portfolio margin calculation across multiple accounts
//! - Optimal collateral allocation algorithm
//! - 20% safety buffer enforcement
//! - API delay handling without liquidation risk
//! - Zero-cost abstractions for memory efficiency
//! 
//! Target: Prevent margin calls by maintaining 20% buffer above maintenance

use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Account identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccountId(pub u64);

impl Display for AccountId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ACC-{}", self.0)
    }
}

/// Asset types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Asset {
    BTC,
    ETH,
    SOL,
    USDT,
}

impl Display for Asset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Asset::BTC => write!(f, "BTC"),
            Asset::ETH => write!(f, "ETH"),
            Asset::SOL => write!(f, "SOL"),
            Asset::USDT => write!(f, "USDT"),
        }
    }
}

/// Position with margin requirements
#[derive(Debug, Clone)]
pub struct MarginPosition {
    pub asset: Asset,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub position_type: PositionType,
    pub notional_value: f64,
    pub unrealized_pnl: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionType {
    Spot,
    Perpetual,
    Futures,
    Options,
}

/// Account margin state
#[derive(Debug, Clone)]
pub struct AccountMarginState {
    pub account_id: AccountId,
    pub total_collateral: f64,
    pub available_collateral: f64,
    pub used_collateral: f64,
    pub initial_margin: f64,
    pub maintenance_margin: f64,
    pub positions: Vec<MarginPosition>,
    pub margin_ratio: f64,
    pub liquidation_price: Option<f64>,
    pub last_updated: Instant,
}

impl AccountMarginState {
    /// Check if account is safe (above 20% buffer)
    pub fn is_safe_with_buffer(&self, buffer_pct: f64) -> bool {
        if self.maintenance_margin == 0.0 {
            return true;
        }
        
        let required_with_buffer = self.maintenance_margin * (1.0 + buffer_pct);
        self.total_collateral >= required_with_buffer
    }
    
    /// Get margin utilization percentage
    pub fn margin_utilization(&self) -> f64 {
        if self.total_collateral == 0.0 {
            return 0.0;
        }
        self.used_collateral / self.total_collateral
    }
    
    /// Calculate distance to liquidation
    pub fn distance_to_liquidation(&self) -> f64 {
        if self.maintenance_margin == 0.0 || self.total_collateral == 0.0 {
            return f64::INFINITY;
        }
        (self.total_collateral - self.maintenance_margin) / self.maintenance_margin
    }
}

/// Collateral allocation recommendation
#[derive(Debug, Clone)]
pub struct CollateralAllocation {
    pub account_id: AccountId,
    pub current_collateral: f64,
    pub recommended_collateral: f64,
    pub adjustment: f64,  // Positive = add, Negative = remove
    pub priority: Priority,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Critical,  // Immediate action needed
    High,      // Action needed soon
    Medium,    // Should address
    Low,       // Optional optimization
}

/// Optimization result
#[derive(Debug)]
pub struct OptimizationResult {
    pub success: bool,
    pub execution_time_us: u128,
    pub allocations: Vec<CollateralAllocation>,
    pub total_collateral: f64,
    pub total_required: f64,
    pub buffer_maintained: bool,
    pub accounts_at_risk: Vec<AccountId>,
    pub recommendations: Vec<String>,
}

/// Main cross-margin optimizer
pub struct CrossMarginOptimizer {
    /// Account states
    accounts: HashMap<AccountId, AccountMarginState>,
    
    /// Asset prices for valuation
    prices: HashMap<Asset, f64>,
    
    /// Configuration
    safety_buffer: f64,  // 20% default
    max_concentration: f64,  // Max % in single asset
    
    /// Optimization history
    optimization_count: u64,
    last_optimization: Option<Instant>,
    
    /// Risk metrics
    total_portfolio_margin: f64,
    portfolio_margin_ratio: f64,
}

impl CrossMarginOptimizer {
    /// Create new optimizer with default settings
    pub fn new(safety_buffer: f64, max_concentration: f64) -> Self {
        Self {
            accounts: HashMap::new(),
            prices: HashMap::new(),
            safety_buffer,
            max_concentration,
            optimization_count: 0,
            last_optimization: None,
            total_portfolio_margin: 0.0,
            portfolio_margin_ratio: 0.0,
        }
    }
    
    /// Add or update account state
    pub fn update_account(&mut self, state: AccountMarginState) {
        self.accounts.insert(state.account_id, state);
    }
    
    /// Update asset price
    pub fn update_price(&mut self, asset: Asset, price: f64) {
        self.prices.insert(asset, price);
    }
    
    /// Get account state
    pub fn get_account(&self, account_id: AccountId) -> Option<&AccountMarginState> {
        self.accounts.get(&account_id)
    }
    
    /// Execute optimization to find best collateral distribution
    pub fn optimize(&mut self) -> OptimizationResult {
        let start_time = Instant::now();
        
        let mut allocations = Vec::new();
        let mut accounts_at_risk = Vec::new();
        let mut recommendations = Vec::new();
        
        // Calculate total collateral and requirements
        let mut total_collateral = 0.0;
        let mut total_required = 0.0;
        
        for (account_id, account) in &self.accounts {
            total_collateral += account.total_collateral;
            total_required += account.maintenance_margin * (1.0 + self.safety_buffer);
            
            // Check if account is at risk
            if !account.is_safe_with_buffer(self.safety_buffer) {
                accounts_at_risk.push(*account_id);
            }
        }
        
        // Sort accounts by risk level (distance to liquidation)
        let mut sorted_accounts: Vec<_> = self.accounts.iter().collect();
        sorted_accounts.sort_by(|a, b| {
            a.1.distance_to_liquidation()
                .partial_cmp(&b.1.distance_to_liquidation())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Generate allocations
        let mut surplus = 0.0;
        let mut deficit = 0.0;
        
        for (account_id, account) in &sorted_accounts {
            let required_with_buffer = account.maintenance_margin * (1.0 + self.safety_buffer);
            let gap = required_with_buffer - account.total_collateral;
            
            if gap > 0.0 {
                deficit += gap;
                
                allocations.push(CollateralAllocation {
                    account_id: **account_id,
                    current_collateral: account.total_collateral,
                    recommended_collateral: required_with_buffer,
                    adjustment: gap,
                    priority: if gap > account.total_collateral * 0.5 {
                        Priority::Critical
                    } else if gap > account.total_collateral * 0.2 {
                        Priority::High
                    } else {
                        Priority::Medium
                    },
                    reason: format!(
                        "Account needs {:.2} more to maintain {}% buffer",
                        gap,
                        self.safety_buffer * 100.0
                    ),
                });
            } else {
                surplus += -gap;
                
                // Only recommend removing excess if other accounts need it
                if deficit > 0.0 {
                    let transfer_amount = (-gap).min(deficit);
                    
                    allocations.push(CollateralAllocation {
                        account_id: **account_id,
                        current_collateral: account.total_collateral,
                        recommended_collateral: account.total_collateral - transfer_amount,
                        adjustment: -transfer_amount,
                        priority: Priority::Low,
                        reason: format!(
                            "Can transfer {:.2} to undercollateralized accounts",
                            transfer_amount
                        ),
                    });
                }
            }
        }
        
        // Sort allocations by priority
        allocations.sort_by(|a, b| a.priority.cmp(&b.priority));
        
        // Generate recommendations
        if !accounts_at_risk.is_empty() {
            recommendations.push(format!(
                "CRITICAL: {} account(s) below safety buffer - immediate action required",
                accounts_at_risk.len()
            ));
        }
        
        if deficit > surplus && surplus > 0.0 {
            recommendations.push(format!(
                "WARNING: Total deficit ({:.2}) exceeds available surplus ({:.2})",
                deficit, surplus
            ));
            recommendations.push("Consider adding external collateral or reducing positions");
        }
        
        // Check concentration risk
        self.check_concentration_risk(&mut recommendations);
        
        let execution_time = start_time.elapsed().as_micros();
        
        // Update metrics
        self.optimization_count += 1;
        self.last_optimization = Some(Instant::now());
        self.total_portfolio_margin = total_required;
        self.portfolio_margin_ratio = if total_collateral > 0.0 {
            total_required / total_collateral
        } else {
            0.0
        };
        
        OptimizationResult {
            success: accounts_at_risk.is_empty() || surplus >= deficit,
            execution_time_us: execution_time,
            allocations,
            total_collateral,
            total_required,
            buffer_maintained: accounts_at_risk.is_empty(),
            accounts_at_risk,
            recommendations,
        }
    }
    
    /// Check concentration risk across portfolio
    fn check_concentration_risk(&self, recommendations: &mut Vec<String>) {
        // Calculate total exposure by asset
        let mut exposure_by_asset: HashMap<Asset, f64> = HashMap::new();
        let mut total_exposure = 0.0;
        
        for account in self.accounts.values() {
            for position in &account.positions {
                let exposure = position.notional_value.abs();
                *exposure_by_asset.entry(position.asset).or_insert(0.0) += exposure;
                total_exposure += exposure;
            }
        }
        
        // Check for concentration violations
        for (asset, exposure) in &exposure_by_asset {
            if total_exposure > 0.0 {
                let concentration = exposure / total_exposure;
                if concentration > self.max_concentration {
                    recommendations.push(format!(
                        "CONCENTRATION RISK: {:.1}% in {} exceeds {:.1}% limit",
                        concentration * 100.0,
                        asset,
                        self.max_concentration * 100.0
                    ));
                }
            }
        }
    }
    
    /// Simulate API delay handling - ensures no liquidation during delays
    pub fn simulate_delay_safety(&self, delay_ms: u64) -> DelaySafetyReport {
        let mut report = DelaySafetyReport {
            max_safe_delay_ms: u64::MAX,
            accounts_safe_during_delay: 0,
            accounts_at_risk_during_delay: 0,
            worst_case_margin_call: None,
        };
        
        let delay_seconds = delay_ms as f64 / 1000.0;
        
        for account in self.accounts.values() {
            // Estimate how much margin could erode during delay
            // Based on position volatility (simplified)
            let estimated_margin_erosion = account.maintenance_margin * 0.01 * delay_seconds;
            
            let projected_collateral = account.total_collateral - estimated_margin_erosion;
            let projected_buffer = if account.maintenance_margin > 0.0 {
                (projected_collateral - account.maintenance_margin) / account.maintenance_margin
            } else {
                f64::INFINITY
            };
            
            if projected_buffer >= self.safety_buffer {
                report.accounts_safe_during_delay += 1;
            } else {
                report.accounts_at_risk_during_delay += 1;
                
                if report.worst_case_margin_call.is_none()
                    || projected_buffer < report.worst_case_margin_call.unwrap()
                {
                    report.worst_case_margin_call = Some(projected_buffer);
                }
            }
            
            // Calculate max safe delay for this account
            if account.maintenance_margin > 0.0 {
                let max_erosion = account.total_collateral - account.maintenance_margin * (1.0 + self.safety_buffer);
                if max_erosion > 0.0 {
                    let max_erosion_rate = account.maintenance_margin * 0.01;
                    let max_safe_delay = (max_erosion / max_erosion_rate * 1000.0) as u64;
                    report.max_safe_delay_ms = report.max_safe_delay_ms.min(max_safe_delay);
                }
            }
        }
        
        report
    }
    
    /// Get portfolio summary
    pub fn get_portfolio_summary(&self) -> PortfolioSummary {
        let total_collateral: f64 = self.accounts.values()
            .map(|a| a.total_collateral)
            .sum();
        
        let total_maintenance: f64 = self.accounts.values()
            .map(|a| a.maintenance_margin)
            .sum();
        
        let total_initial: f64 = self.accounts.values()
            .map(|a| a.initial_margin)
            .sum();
        
        let accounts_at_risk: usize = self.accounts.values()
            .filter(|a| !a.is_safe_with_buffer(self.safety_buffer))
            .count();
        
        PortfolioSummary {
            total_collateral,
            total_maintenance_margin: total_maintenance,
            total_initial_margin: total_initial,
            portfolio_margin_ratio: if total_maintenance > 0.0 {
                total_collateral / total_maintenance
            } else {
                0.0
            },
            accounts_count: self.accounts.len(),
            accounts_at_risk,
            average_buffer: if accounts_at_risk < self.accounts.len() {
                (total_collateral - total_maintenance) / total_maintenance
            } else {
                0.0
            },
        }
    }
}

impl Default for CrossMarginOptimizer {
    fn default() -> Self {
        Self::new(0.20, 0.50) // 20% buffer, 50% max concentration
    }
}

/// Delay safety report
#[derive(Debug)]
pub struct DelaySafetyReport {
    pub max_safe_delay_ms: u64,
    pub accounts_safe_during_delay: usize,
    pub accounts_at_risk_during_delay: usize,
    pub worst_case_margin_call: Option<f64>,
}

/// Portfolio summary
#[derive(Debug)]
pub struct PortfolioSummary {
    pub total_collateral: f64,
    pub total_maintenance_margin: f64,
    pub total_initial_margin: f64,
    pub portfolio_margin_ratio: f64,
    pub accounts_count: usize,
    pub accounts_at_risk: usize,
    pub average_buffer: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_account_safety_buffer() {
        let account = AccountMarginState {
            account_id: AccountId(1),
            total_collateral: 12000.0,
            available_collateral: 2000.0,
            used_collateral: 10000.0,
            initial_margin: 10000.0,
            maintenance_margin: 8000.0,
            positions: Vec::new(),
            margin_ratio: 1.5,
            liquidation_price: None,
            last_updated: Instant::now(),
        };
        
        assert!(account.is_safe_with_buffer(0.20)); // 12000 >= 8000 * 1.2 = 9600
        assert!(!account.is_safe_with_buffer(0.50)); // 12000 < 8000 * 1.5 = 12000
    }
    
    #[test]
    fn test_optimizer_allocation() {
        let mut optimizer = CrossMarginOptimizer::new(0.20, 0.50);
        
        // Add account with sufficient collateral
        optimizer.update_account(AccountMarginState {
            account_id: AccountId(1),
            total_collateral: 15000.0,
            available_collateral: 5000.0,
            used_collateral: 10000.0,
            initial_margin: 10000.0,
            maintenance_margin: 8000.0,
            positions: Vec::new(),
            margin_ratio: 1.5,
            liquidation_price: None,
            last_updated: Instant::now(),
        });
        
        // Add account needing collateral
        optimizer.update_account(AccountMarginState {
            account_id: AccountId(2),
            total_collateral: 8000.0,
            available_collateral: 0.0,
            used_collateral: 8000.0,
            initial_margin: 8000.0,
            maintenance_margin: 7000.0,
            positions: Vec::new(),
            margin_ratio: 1.14,
            liquidation_price: None,
            last_updated: Instant::now(),
        });
        
        let result = optimizer.optimize();
        
        assert!(result.success);
        assert!(!result.allocations.is_empty());
    }
    
    #[test]
    fn test_delay_safety() {
        let mut optimizer = CrossMarginOptimizer::default();
        
        optimizer.update_account(AccountMarginState {
            account_id: AccountId(1),
            total_collateral: 20000.0,
            available_collateral: 10000.0,
            used_collateral: 10000.0,
            initial_margin: 10000.0,
            maintenance_margin: 8000.0,
            positions: Vec::new(),
            margin_ratio: 2.5,
            liquidation_price: None,
            last_updated: Instant::now(),
        });
        
        let report = optimizer.simulate_delay_safety(1000); // 1 second delay
        
        assert_eq!(report.accounts_at_risk_during_delay, 0);
        assert!(report.max_safe_delay_ms > 0);
    }
}
