//! Cross-Isolated Margin Router
//! 
//! Intelligent capital allocation system that routes margin between
//! cross and isolated modes based on risk/reward optimization.
//! Implements dynamic mode switching to maximize capital efficiency
//! while minimizing liquidation risk.
//!
//! Features:
//! - Risk-based margin mode selection
//! - Capital reallocation optimizer
//! - Liquidation risk-aware routing
//! - Zero-cost abstractions for high-frequency updates

use std::collections::HashMap;

/// Margin mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginMode {
    /// Shared margin across all positions
    Cross,
    /// Dedicated margin per position
    Isolated,
}

/// Position metadata for routing decisions
#[derive(Debug, Clone)]
pub struct PositionInfo {
    pub symbol: String,
    pub side: String,  // "long" or "short"
    pub size: f64,
    pub entry_price: f64,
    pub leverage: f64,
    pub current_mode: MarginMode,
    pub unrealized_pnl: f64,
    pub volatility_24h: f64,  // 24h realized volatility
}

impl PositionInfo {
    /// Calculate notional value
    pub fn notional(&self) -> f64 {
        self.size * self.entry_price
    }
    
    /// Calculate risk score (higher = more risky)
    pub fn risk_score(&self) -> f64 {
        let base_risk = self.leverage / 10.0;  // Normalize around 10x
        let vol_adjustment = self.volatility_24h / 0.5;  // Normalize around 50% vol
        
        base_risk * vol_adjustment
    }
}

/// Routing decision result
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub symbol: String,
    pub recommended_mode: MarginMode,
    pub reason: String,
    pub capital_efficiency_gain: f64,  // Percentage improvement
    pub risk_change: f64,  // Change in risk score
}

/// Cross-Isolated margin router
pub struct CrossIsolatedRouter {
    /// Minimum leverage for isolated mode recommendation
    isolated_leverage_threshold: f64,
    /// Maximum acceptable portfolio risk in cross mode
    max_cross_risk: f64,
    /// Volatility threshold for mode switch
    volatility_threshold: f64,
    /// Current positions
    positions: HashMap<String, PositionInfo>,
    /// Total available capital
    total_capital: f64,
    /// Capital allocated to isolated positions
    isolated_capital: f64,
}

impl CrossIsolatedRouter {
    /// Create a new router with default thresholds
    pub fn new(total_capital: f64) -> Self {
        Self {
            isolated_leverage_threshold: 15.0,
            max_cross_risk: 0.3,  // 30% of capital at risk
            volatility_threshold: 0.8,  // 80% annualized vol
            positions: HashMap::new(),
            total_capital,
            isolated_capital: 0.0,
        }
    }
    
    /// Set the leverage threshold for isolated mode
    pub fn set_leverage_threshold(&mut self, threshold: f64) {
        self.isolated_leverage_threshold = threshold.max(1.0);
    }
    
    /// Set maximum cross mode risk tolerance
    pub fn set_max_cross_risk(&mut self, risk: f64) {
        self.max_cross_risk = risk.clamp(0.1, 0.9);
    }
    
    /// Add or update a position
    pub fn update_position(&mut self, info: PositionInfo) {
        let notional = info.notional();
        
        // Track capital allocation
        if let Some(existing) = self.positions.get(&info.symbol) {
            if existing.current_mode == MarginMode::Isolated {
                self.isolated_capital -= existing.notional() / existing.leverage;
            }
        }
        
        if info.current_mode == MarginMode::Isolated {
            self.isolated_capital += notional / info.leverage;
        }
        
        self.positions.insert(info.symbol.clone(), info);
    }
    
    /// Remove a position
    pub fn remove_position(&mut self, symbol: &str) {
        if let Some(pos) = self.positions.remove(symbol) {
            if pos.current_mode == MarginMode::Isolated {
                self.isolated_capital -= pos.notional() / pos.leverage;
            }
        }
    }
    
    /// Determine optimal margin mode for a position
    pub fn recommend_mode(&self, position: &PositionInfo) -> RoutingDecision {
        let mut recommended = MarginMode::Cross;
        let mut reasons = Vec::new();
        let mut efficiency_gain = 0.0;
        let mut risk_change = 0.0;
        
        // Rule 1: High leverage positions should use isolated
        if position.leverage >= self.isolated_leverage_threshold {
            recommended = MarginMode::Isolated;
            reasons.push(format!(
                "High leverage ({:.1}x) exceeds threshold ({:.1}x)",
                position.leverage, self.isolated_leverage_threshold
            ));
            risk_change = -0.1;  // Reduces portfolio risk
        }
        
        // Rule 2: High volatility assets prefer isolated
        if position.volatility_24h >= self.volatility_threshold {
            recommended = MarginMode::Isolated;
            reasons.push(format!(
                "High volatility ({:.1}%) increases liquidation risk",
                position.volatility_24h * 100.0
            ));
            risk_change -= 0.05;
        }
        
        // Rule 3: Check cross mode capacity
        let cross_utilization = self.calculate_cross_utilization();
        if cross_utilization > 0.7 && recommended == MarginMode::Cross {
            // Cross mode is getting full, prefer isolated for new risk
            recommended = MarginMode::Isolated;
            reasons.push("Cross margin utilization above 70%".to_string());
        }
        
        // Rule 4: Low risk positions can share cross margin
        if position.risk_score() < 0.5 && position.leverage < 5.0 {
            if recommended == MarginMode::Isolated {
                efficiency_gain = 0.02;  // Small efficiency gain from sharing
            }
            recommended = MarginMode::Cross;
            reasons.push("Low risk profile suitable for cross margin".to_string());
            risk_change += 0.02;  // Slight increase in shared risk
        }
        
        // If no specific reason, default to cross for capital efficiency
        if reasons.is_empty() {
            reasons.push("Default recommendation for capital efficiency".to_string());
        }
        
        RoutingDecision {
            symbol: position.symbol.clone(),
            recommended_mode: recommended,
            reason: reasons.join("; "),
            capital_efficiency_gain: efficiency_gain,
            risk_change,
        }
    }
    
    /// Calculate cross margin utilization ratio
    fn calculate_cross_utilization(&self) -> f64 {
        let mut cross_margin_used = 0.0;
        
        for position in self.positions.values() {
            if position.current_mode == MarginMode::Cross {
                cross_margin_used += position.notional() / position.leverage;
            }
        }
        
        if self.total_capital <= 0.0 {
            return 0.0;
        }
        
        (cross_margin_used / self.total_capital).min(1.0)
    }
    
    /// Get all routing recommendations
    pub fn get_all_recommendations(&self) -> Vec<RoutingDecision> {
        self.positions
            .values()
            .map(|pos| self.recommend_mode(pos))
            .collect()
    }
    
    /// Find positions that should switch modes
    pub fn get_switch_candidates(&self) -> Vec<RoutingDecision> {
        self.get_all_recommendations()
            .into_iter()
            .filter(|decision| {
                let position = self.positions.get(&decision.symbol);
                if let Some(pos) = position {
                    pos.current_mode != decision.recommended_mode
                } else {
                    false
                }
            })
            .collect()
    }
    
    /// Execute a mode switch (returns simulation result)
    pub fn simulate_switch(&self, symbol: &str, new_mode: MarginMode) -> SwitchSimulation {
        let position = self.positions.get(symbol);
        
        match position {
            Some(pos) => {
                let old_margin = pos.notional() / pos.leverage;
                
                let new_margin = match new_mode {
                    MarginMode::Isolated => old_margin,  // Same margin, just isolated
                    MarginMode::Cross => old_margin,     // Released to cross pool
                };
                
                let capital_freed = if pos.current_mode == MarginMode::Isolated && new_mode == MarginMode::Cross {
                    old_margin
                } else if pos.current_mode == MarginMode::Cross && new_mode == MarginMode::Isolated {
                    -old_margin  // Capital now locked
                } else {
                    0.0
                };
                
                SwitchSimulation {
                    symbol: symbol.to_string(),
                    from_mode: pos.current_mode,
                    to_mode: new_mode,
                    capital_impact: capital_freed,
                    risk_reduction: pos.risk_score() * 0.1,
                    success: true,
                }
            }
            None => SwitchSimulation {
                symbol: symbol.to_string(),
                from_mode: MarginMode::Cross,
                to_mode: new_mode,
                capital_impact: 0.0,
                risk_reduction: 0.0,
                success: false,
            },
        }
    }
    
    /// Get portfolio summary
    pub fn get_portfolio_summary(&self) -> PortfolioSummary {
        let mut cross_positions = 0;
        let mut isolated_positions = 0;
        let mut cross_notional = 0.0;
        let mut isolated_notional = 0.0;
        
        for position in self.positions.values() {
            match position.current_mode {
                MarginMode::Cross => {
                    cross_positions += 1;
                    cross_notional += position.notional();
                }
                MarginMode::Isolated => {
                    isolated_positions += 1;
                    isolated_notional += position.notional();
                }
            }
        }
        
        PortfolioSummary {
            total_positions: self.positions.len(),
            cross_positions,
            isolated_positions,
            cross_notional,
            isolated_notional,
            cross_utilization: self.calculate_cross_utilization(),
            isolated_capital_allocated: self.isolated_capital,
            available_capital: self.total_capital - self.isolated_capital,
        }
    }
    
    /// Optimize capital allocation across all positions
    pub fn optimize_allocation(&self) -> OptimizationResult {
        let mut switches_needed = Vec::new();
        let mut total_risk_reduction = 0.0;
        let mut capital_efficiency_change = 0.0;
        
        for decision in self.get_switch_candidates() {
            switches_needed.push(decision.clone());
            total_risk_reduction += decision.risk_change;
            capital_efficiency_change += decision.capital_efficiency_gain;
        }
        
        OptimizationResult {
            switches: switches_needed,
            total_risk_reduction,
            capital_efficiency_change,
            recommended_order: self.prioritize_switches(&switches_needed),
        }
    }
    
    /// Prioritize switches by urgency (highest risk reduction first)
    fn prioritize_switches(&self, switches: &[RoutingDecision]) -> Vec<String> {
        let mut sorted: Vec<_> = switches.iter().collect();
        sorted.sort_by(|a, b| {
            b.risk_change.partial_cmp(&a.risk_change).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        sorted.iter().map(|s| s.symbol.clone()).collect()
    }
}

/// Simulation result for a mode switch
#[derive(Debug, Clone)]
pub struct SwitchSimulation {
    pub symbol: String,
    pub from_mode: MarginMode,
    pub to_mode: MarginMode,
    pub capital_impact: f64,  // Positive = freed, negative = locked
    pub risk_reduction: f64,
    pub success: bool,
}

/// Portfolio summary statistics
#[derive(Debug, Clone)]
pub struct PortfolioSummary {
    pub total_positions: usize,
    pub cross_positions: usize,
    pub isolated_positions: usize,
    pub cross_notional: f64,
    pub isolated_notional: f64,
    pub cross_utilization: f64,
    pub isolated_capital_allocated: f64,
    pub available_capital: f64,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub switches: Vec<RoutingDecision>,
    pub total_risk_reduction: f64,
    pub capital_efficiency_change: f64,
    pub recommended_order: Vec<String>,
}

impl Default for CrossIsolatedRouter {
    fn default() -> Self {
        Self::new(100000.0)  // Default 100k capital
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_high_leverage_isolated() {
        let router = CrossIsolatedRouter::new(100000.0);
        
        let position = PositionInfo {
            symbol: "BTCUSDT".to_string(),
            side: "long".to_string(),
            size: 1.0,
            entry_price: 50000.0,
            leverage: 20.0,
            current_mode: MarginMode::Cross,
            unrealized_pnl: 0.0,
            volatility_24h: 0.5,
        };
        
        let decision = router.recommend_mode(&position);
        
        // High leverage should recommend isolated
        assert_eq!(decision.recommended_mode, MarginMode::Isolated);
    }
    
    #[test]
    fn test_low_risk_cross() {
        let router = CrossIsolatedRouter::new(100000.0);
        
        let position = PositionInfo {
            symbol: "ETHUSDT".to_string(),
            side: "long".to_string(),
            size: 10.0,
            entry_price: 3000.0,
            leverage: 3.0,
            current_mode: MarginMode::Isolated,
            unrealized_pnl: 0.0,
            volatility_24h: 0.4,
        };
        
        let decision = router.recommend_mode(&position);
        
        // Low risk should recommend cross
        assert_eq!(decision.recommended_mode, MarginMode::Cross);
    }
}
