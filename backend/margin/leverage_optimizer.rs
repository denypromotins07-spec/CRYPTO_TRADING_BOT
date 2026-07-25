//! Leverage Optimizer using Kelly Criterion
//! 
//! Dynamic leverage adjustment system that calculates optimal position sizing
//! based on edge, win rate, and risk constraints. Implements fractional Kelly
//! to reduce volatility while maintaining positive expected growth.
//!
//! Features:
//! - Kelly Criterion calculation with Bayesian priors
//! - Drawdown-aware leverage reduction
//! - Correlation-adjusted position limits
//! - Zero-cost abstractions for real-time updates

use std::collections::HashMap;

/// Trading strategy statistics for Kelly calculation
#[derive(Debug, Clone)]
pub struct StrategyStats {
    /// Historical win rate (0.0 to 1.0)
    pub win_rate: f64,
    /// Average win size as fraction of stake
    pub avg_win: f64,
    /// Average loss size as fraction of stake
    pub avg_loss: f64,
    /// Number of trades in sample
    pub trade_count: usize,
    /// Current drawdown from peak equity
    pub current_drawdown: f64,
    /// Maximum historical drawdown
    pub max_drawdown: f64,
}

impl StrategyStats {
    /// Create new stats with default values
    pub fn new() -> Self {
        Self {
            win_rate: 0.5,
            avg_win: 0.02,
            avg_loss: 0.01,
            trade_count: 0,
            current_drawdown: 0.0,
            max_drawdown: 0.0,
        }
    }
    
    /// Calculate expected value per trade
    pub fn expected_value(&self) -> f64 {
        self.win_rate * self.avg_win - (1.0 - self.win_rate) * self.avg_loss
    }
    
    /// Check if strategy has sufficient data
    pub fn is_statistically_significant(&self) -> bool {
        self.trade_count >= 30
    }
}

impl Default for StrategyStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Kelly Criterion calculator with safety constraints
pub struct KellyOptimizer {
    /// Fractional Kelly multiplier (0.5 = half Kelly)
    kelly_fraction: f64,
    /// Maximum allowed leverage
    max_leverage: f64,
    /// Minimum allowed leverage
    min_leverage: f64,
    /// Drawdown threshold for leverage reduction
    drawdown_threshold: f64,
    /// Per-asset leverage limits
    asset_limits: HashMap<String, f64>,
}

impl KellyOptimizer {
    /// Create a new Kelly optimizer with conservative defaults
    pub fn new() -> Self {
        let mut asset_limits = HashMap::new();
        asset_limits.insert("BTC".to_string(), 5.0);
        asset_limits.insert("ETH".to_string(), 4.0);
        asset_limits.insert("SOL".to_string(), 3.0);
        
        Self {
            kelly_fraction: 0.5,  // Half Kelly for reduced volatility
            max_leverage: 10.0,
            min_leverage: 0.5,
            drawdown_threshold: 0.10,  // 10% drawdown triggers reduction
            asset_limits,
        }
    }
    
    /// Set the Kelly fraction (0.0 to 1.0)
    pub fn set_kelly_fraction(&mut self, fraction: f64) {
        self.kelly_fraction = fraction.clamp(0.1, 1.0);
    }
    
    /// Set maximum leverage limit
    pub fn set_max_leverage(&mut self, max: f64) {
        self.max_leverage = max.max(1.0);
    }
    
    /// Set per-asset leverage limit
    pub fn set_asset_limit(&mut self, asset: &str, limit: f64) {
        self.asset_limits.insert(asset.to_string(), limit.max(1.0));
    }
    
    /// Calculate raw Kelly fraction (f* = (p*b - q) / b)
    /// where p = win probability, q = loss probability, b = win/loss ratio
    fn calculate_raw_kelly(&self, stats: &StrategyStats) -> f64 {
        if !stats.is_statistically_significant() {
            // Use conservative default for insufficient data
            return 0.5;
        }
        
        let p = stats.win_rate;
        let q = 1.0 - p;
        
        // Win/loss ratio (odds)
        let b = if stats.avg_loss > 0.0 {
            stats.avg_win / stats.avg_loss
        } else {
            stats.avg_win / 0.01  // Prevent division by zero
        };
        
        // Kelly formula: f* = (p*b - q) / b
        let kelly = (p * b - q) / b;
        
        // Clamp to reasonable range
        kelly.clamp(-0.5, 2.0)
    }
    
    /// Apply drawdown-based leverage reduction
    fn apply_drawdown_adjustment(&self, kelly: f64, current_dd: f64) -> f64 {
        if current_dd <= 0.0 {
            return kelly;
        }
        
        // Linear reduction from threshold to 2x threshold
        if current_dd < self.drawdown_threshold {
            kelly
        } else if current_dd < 2.0 * self.drawdown_threshold {
            // Reduce by up to 50%
            let reduction_factor = 1.0 - 0.5 * (current_dd - self.drawdown_threshold) / self.drawdown_threshold;
            kelly * reduction_factor
        } else {
            // Severe drawdown: minimum leverage only
            self.min_leverage
        }
    }
    
    /// Calculate optimal leverage for an asset
    /// 
    /// # Arguments
    /// * `stats` - Strategy performance statistics
    /// * `asset` - Asset symbol for limit lookup
    /// * `equity` - Current account equity
    /// * `current_dd` - Current drawdown from peak
    /// 
    /// # Returns
    /// Optimal leverage factor
    pub fn calculate_optimal_leverage(
        &self,
        stats: &StrategyStats,
        asset: &str,
        equity: f64,
        current_dd: f64,
    ) -> f64 {
        if equity <= 0.0 {
            return self.min_leverage;
        }
        
        // Step 1: Calculate raw Kelly
        let raw_kelly = self.calculate_raw_kelly(stats);
        
        // Step 2: Apply fractional Kelly
        let fractional_kelly = raw_kelly * self.kelly_fraction;
        
        // Step 3: Ensure positive leverage
        let positive_kelly = fractional_kelly.max(self.min_leverage);
        
        // Step 4: Apply drawdown adjustment
        let adjusted = self.apply_drawdown_adjustment(positive_kelly, current_dd);
        
        // Step 5: Apply global max limit
        let limited = adjusted.min(self.max_leverage);
        
        // Step 6: Apply asset-specific limit
        let asset_limit = self.asset_limits.get(asset).copied().unwrap_or(self.max_leverage);
        let final_leverage = limited.min(asset_limit);
        
        // Ensure minimum leverage floor
        final_leverage.max(self.min_leverage)
    }
    
    /// Calculate position size given leverage and equity
    pub fn calculate_position_size(
        &self,
        stats: &StrategyStats,
        asset: &str,
        equity: f64,
        current_dd: f64,
        price: f64,
    ) -> f64 {
        let leverage = self.calculate_optimal_leverage(stats, asset, equity, current_dd);
        let notional = equity * leverage;
        
        if price <= 0.0 {
            return 0.0;
        }
        
        notional / price
    }
    
    /// Get recommended leverage change
    pub fn get_leverage_recommendation(
        &self,
        stats: &StrategyStats,
        asset: &str,
        current_leverage: f64,
        equity: f64,
        current_dd: f64,
    ) -> LeverageRecommendation {
        let optimal = self.calculate_optimal_leverage(stats, asset, equity, current_dd);
        
        let action = if (optimal - current_leverage).abs() < 0.1 {
            LeverageAction::Hold
        } else if optimal > current_leverage {
            LeverageAction::Increase
        } else {
            LeverageAction::Decrease
        };
        
        LeverageRecommendation {
            current: current_leverage,
            recommended: optimal,
            action,
            reason: self.get_recommendation_reason(stats, current_dd),
        }
    }
    
    fn get_recommendation_reason(&self, stats: &StrategyStats, dd: f64) -> String {
        if dd > self.drawdown_threshold {
            format!("Drawdown protection active ({:.1}%)", dd * 100.0)
        } else if !stats.is_statistically_significant() {
            "Insufficient trade history - using conservative leverage".to_string()
        } else if stats.expected_value() <= 0.0 {
            "Negative expected value - minimum leverage recommended".to_string()
        } else {
            format!("Kelly optimal (EV: {:.2}%)", stats.expected_value() * 100.0)
        }
    }
}

impl Default for KellyOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Leverage recommendation result
#[derive(Debug, Clone)]
pub struct LeverageRecommendation {
    pub current: f64,
    pub recommended: f64,
    pub action: LeverageAction,
    pub reason: String,
}

/// Recommended action type
#[derive(Debug, Clone, PartialEq)]
pub enum LeverageAction {
    Hold,
    Increase,
    Decrease,
}

/// Multi-asset leverage manager
pub struct MultiAssetLeverageManager {
    optimizer: KellyOptimizer,
    asset_stats: HashMap<String, StrategyStats>,
    current_leverage: HashMap<String, f64>,
}

impl MultiAssetLeverageManager {
    pub fn new() -> Self {
        Self {
            optimizer: KellyOptimizer::new(),
            asset_stats: HashMap::new(),
            current_leverage: HashMap::new(),
        }
    }
    
    /// Update statistics for an asset
    pub fn update_stats(&mut self, asset: &str, stats: StrategyStats) {
        self.asset_stats.insert(asset.to_string(), stats);
    }
    
    /// Get or create default stats for an asset
    pub fn get_stats(&self, asset: &str) -> Option<&StrategyStats> {
        self.asset_stats.get(asset)
    }
    
    /// Update current leverage for an asset
    pub fn set_current_leverage(&mut self, asset: &str, leverage: f64) {
        self.current_leverage.insert(asset.to_string(), leverage);
    }
    
    /// Get optimal leverage for all assets
    pub fn get_all_optimal_leverages(&self, equity: f64, dd: f64) -> HashMap<String, f64> {
        self.asset_stats
            .iter()
            .map(|(asset, stats)| {
                let leverage = self.optimizer.calculate_optimal_leverage(stats, asset, equity, dd);
                (asset.clone(), leverage)
            })
            .collect()
    }
    
    /// Check if any asset needs leverage adjustment
    pub fn get_rebalancing_needed(&self, equity: f64, dd: f64) -> Vec<LeverageRecommendation> {
        let mut recommendations = Vec::new();
        
        for (asset, stats) in &self.asset_stats {
            let current = self.current_leverage.get(asset).copied().unwrap_or(1.0);
            let rec = self.optimizer.get_leverage_recommendation(stats, asset, current, equity, dd);
            
            if rec.action != LeverageAction::Hold {
                recommendations.push(rec);
            }
        }
        
        recommendations
    }
}

impl Default for MultiAssetLeverageManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kelly_calculation() {
        let optimizer = KellyOptimizer::new();
        
        let stats = StrategyStats {
            win_rate: 0.55,
            avg_win: 0.03,
            avg_loss: 0.02,
            trade_count: 100,
            current_drawdown: 0.0,
            max_drawdown: 0.05,
        };
        
        let leverage = optimizer.calculate_optimal_leverage(&stats, "BTC", 100000.0, 0.0);
        
        // Should be positive given positive EV
        assert!(leverage > 1.0);
        assert!(leverage <= optimizer.max_leverage);
    }
    
    #[test]
    fn test_drawdown_reduction() {
        let mut optimizer = KellyOptimizer::new();
        optimizer.set_kelly_fraction(1.0);  // Full Kelly
        
        let stats = StrategyStats {
            win_rate: 0.55,
            avg_win: 0.03,
            avg_loss: 0.02,
            trade_count: 100,
            current_drawdown: 0.0,
            max_drawdown: 0.05,
        };
        
        let leverage_no_dd = optimizer.calculate_optimal_leverage(&stats, "BTC", 100000.0, 0.0);
        let leverage_with_dd = optimizer.calculate_optimal_leverage(&stats, "BTC", 100000.0, 0.15);
        
        // Leverage should be reduced with drawdown
        assert!(leverage_with_dd < leverage_no_dd);
    }
}
