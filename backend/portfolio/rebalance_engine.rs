//! Portfolio Rebalancing Engine
//!
//! Executes threshold-based and time-based portfolio rebalancing.
//! Minimizes transaction costs while maintaining target allocations.
//! Critical for maintaining optimal risk exposure across BTC, SOL, ETH, USDT.
//!
//! # Rebalancing Triggers:
//! - Threshold breach: Any asset deviates > X% from target
//! - Time-based: Regular interval rebalancing (e.g., daily, weekly)
//! - Volatility-triggered: Rebalance after major market moves
//! - Cost-aware: Only rebalance if benefit exceeds transaction costs
//!
//! # Key Features:
//! - Smart order generation with cost optimization
//! - Drift monitoring and alerting
//! - Transaction cost modeling

use std::f64;

/// Fixed portfolio size
const N_ASSETS: usize = 4;

/// Asset labels
pub const ASSET_LABELS: [&str; N_ASSETS] = ["BTC", "SOL", "ETH", "USDT"];

/// Type of rebalancing trigger
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RebalanceTrigger {
    /// Threshold breach (drift exceeded)
    ThresholdBreach,
    /// Scheduled time-based rebalance
    Scheduled,
    /// Volatility event triggered
    VolatilityEvent,
    /// Manual/forced rebalance
    Manual,
    /// Cash flow triggered (deposit/withdrawal)
    CashFlow,
}

/// Result of a rebalance analysis
#[derive(Debug, Clone)]
pub struct RebalanceAnalysis {
    /// Whether rebalancing is recommended
    pub should_rebalance: bool,
    /// Trigger type if rebalancing recommended
    pub trigger: Option<RebalanceTrigger>,
    /// Current weights
    pub current_weights: [f64; N_ASSETS],
    /// Target weights
    pub target_weights: [f64; N_ASSETS],
    /// Weight drift (absolute difference)
    pub weight_drift: [f64; N_ASSETS],
    /// Maximum drift across all assets
    pub max_drift: f64,
    /// Estimated transaction cost (in basis points)
    pub estimated_cost_bps: f64,
    /// Expected benefit from rebalancing (risk reduction)
    pub expected_benefit: f64,
    /// Net benefit (benefit - cost)
    pub net_benefit: f64,
    /// Recommended trades to execute
    pub recommended_trades: Vec<RebalanceTrade>,
}

/// Individual trade recommendation
#[derive(Debug, Clone)]
pub struct RebalanceTrade {
    /// Asset to trade
    pub asset: String,
    /// Trade direction (positive = buy, negative = sell)
    pub quantity: f64,
    /// Current weight
    pub current_weight: f64,
    /// Target weight
    pub target_weight: f64,
    /// Estimated slippage in bps
    pub estimated_slippage_bps: f64,
}

/// Configuration for rebalancing logic
#[derive(Debug, Clone)]
pub struct RebalanceConfig {
    /// Drift threshold for triggering rebalance (default 5%)
    pub drift_threshold: f64,
    /// Minimum time between rebalances (hours)
    pub min_rebalance_interval_hours: u64,
    /// Maximum allowed drift before forced rebalance (default 10%)
    pub max_drift_threshold: f64,
    /// Transaction cost per trade in basis points
    pub transaction_cost_bps: f64,
    /// Minimum trade size as % of portfolio
    pub min_trade_size_pct: f64,
    /// Whether to use cost-benefit analysis
    pub use_cost_benefit: bool,
    /// Benefit/cost ratio threshold for rebalancing
    pub benefit_cost_threshold: f64,
}

impl Default for RebalanceConfig {
    fn default() -> Self {
        Self {
            drift_threshold: 0.05,      // 5% drift triggers review
            min_rebalance_interval_hours: 24,  // At most once per day
            max_drift_threshold: 0.10,  // Force rebalance at 10% drift
            transaction_cost_bps: 10.0, // 0.10% per trade
            min_trade_size_pct: 0.01,   // 1% minimum trade
            use_cost_benefit: true,
            benefit_cost_threshold: 2.0, // Benefit must be 2x cost
        }
    }
}

/// Portfolio Rebalancing Engine
///
/// Monitors portfolio drift and generates optimal rebalancing trades.
pub struct RebalanceEngine {
    /// Configuration
    config: RebalanceConfig,
    /// Target weights for each asset
    target_weights: [f64; N_ASSETS],
    /// Last rebalance timestamp (Unix epoch seconds)
    last_rebalance_time: u64,
    /// Portfolio value
    portfolio_value: f64,
}

impl RebalanceEngine {
    /// Create new engine with given targets
    pub fn new(target_weights: [f64; N_ASSETS], portfolio_value: f64) -> Self {
        // Validate targets sum to ~1
        let sum: f64 = target_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Target weights must sum to 1.0, got {}",
            sum
        );

        Self {
            config: RebalanceConfig::default(),
            target_weights,
            last_rebalance_time: 0,
            portfolio_value,
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        target_weights: [f64; N_ASSETS],
        portfolio_value: f64,
        config: RebalanceConfig,
    ) -> Self {
        Self {
            config,
            target_weights,
            last_rebalance_time: 0,
            portfolio_value,
        }
    }

    /// Update portfolio value
    pub fn set_portfolio_value(&mut self, value: f64) {
        if value > 0.0 {
            self.portfolio_value = value;
        }
    }

    /// Record that a rebalance was executed
    pub fn record_rebalance(&mut self, timestamp: u64) {
        self.last_rebalance_time = timestamp;
    }

    /// Analyze whether rebalancing is needed
    ///
    /// # Arguments
    /// * `current_weights` - Current portfolio weights
    /// * `current_time` - Current Unix timestamp
    ///
    /// # Returns
    /// RebalanceAnalysis with recommendation and trade list
    pub fn analyze(&self, current_weights: [f64; N_ASSETS], current_time: u64) -> RebalanceAnalysis {
        // Compute drift
        let mut weight_drift = [0.0; N_ASSETS];
        let mut max_drift = 0.0;

        for i in 0..N_ASSETS {
            weight_drift[i] = (current_weights[i] - self.target_weights[i]).abs();
            max_drift = max_drift.max(weight_drift[i]);
        }

        // Check time since last rebalance
        let hours_since_rebalance = if self.last_rebalance_time > 0 {
            (current_time - self.last_rebalance_time) / 3600
        } else {
            u64::MAX // Never rebalanced, allow immediate
        };

        // Determine if rebalancing is triggered
        let mut should_rebalance = false;
        let mut trigger: Option<RebalanceTrigger> = None;

        // Check maximum drift breach
        if max_drift >= self.config.max_drift_threshold {
            should_rebalance = true;
            trigger = Some(RebalanceTrigger::ThresholdBreach);
        }
        // Check regular threshold with time constraint
        else if max_drift >= self.config.drift_threshold
            && hours_since_rebalance >= self.config.min_rebalance_interval_hours
        {
            should_rebalance = true;
            trigger = Some(RebalanceTrigger::ThresholdBreach);
        }

        // Generate recommended trades
        let trades = self._generate_trades(&current_weights, &weight_drift);

        // Estimate costs and benefits
        let estimated_cost = self._estimate_transaction_cost(&trades);
        let expected_benefit = self._estimate_rebalance_benefit(&current_weights, &weight_drift);

        // Apply cost-benefit analysis if enabled
        if self.config.use_cost_benefit && should_rebalance {
            let benefit_cost_ratio = if estimated_cost > 0.0 {
                expected_benefit / estimated_cost
            } else {
                f64::MAX
            };

            if benefit_cost_ratio < self.config.benefit_cost_threshold {
                should_rebalance = false;
                trigger = None;
            }
        }

        RebalanceAnalysis {
            should_rebalance,
            trigger,
            current_weights,
            target_weights: self.target_weights,
            weight_drift,
            max_drift,
            estimated_cost_bps: estimated_cost,
            expected_benefit,
            net_benefit: expected_benefit - estimated_cost,
            recommended_trades: trades,
        }
    }

    /// Generate trade list to restore target weights
    fn _generate_trades(
        &self,
        current_weights: &[f64; N_ASSETS],
        drift: &[f64; N_ASSETS],
    ) -> Vec<RebalanceTrade> {
        let mut trades = Vec::with_capacity(N_ASSETS);

        for i in 0..N_ASSETS {
            let diff = self.target_weights[i] - current_weights[i];
            
            // Skip small trades below minimum size
            if diff.abs() < self.config.min_trade_size_pct {
                continue;
            }

            let quantity = diff * self.portfolio_value;
            
            // Estimate slippage based on trade size
            let slippage_bps = self._estimate_slippage(diff.abs());

            trades.push(RebalanceTrade {
                asset: ASSET_LABELS[i].to_string(),
                quantity,
                current_weight: current_weights[i],
                target_weight: self.target_weights[i],
                estimated_slippage_bps: slippage_bps,
            });
        }

        trades
    }

    /// Estimate slippage for a trade size
    fn _estimate_slippage(&self, trade_size_pct: f64) -> f64 {
        // Simple linear model: larger trades have more slippage
        // Base slippage + size-dependent component
        let base_bps = 2.0;
        let size_factor = trade_size_pct * 100.0; // Convert to percentage
        base_bps + size_factor * 0.5
    }

    /// Estimate total transaction cost
    fn _estimate_transaction_cost(&self, trades: &[RebalanceTrade]) -> f64 {
        let total_turnover: f64 = trades.iter().map(|t| t.quantity.abs()).sum();
        let turnover_pct = total_turnover / self.portfolio_value;
        
        // Cost = turnover * transaction_cost_bps
        turnover_pct * self.config.transaction_cost_bps
    }

    /// Estimate benefit from rebalancing (simplified model)
    ///
    /// Benefit is approximated as risk reduction from restoring targets
    fn _estimate_rebalance_benefit(
        &self,
        current_weights: &[f64; N_ASSETS],
        drift: &[f64; N_ASSETS],
    ) -> f64 {
        // Simplified: benefit proportional to squared drift
        // This captures that larger deviations are more costly
        let sum_sq_drift: f64 = drift.iter().map(|d| d * d).sum();
        
        // Scale to comparable units with costs (basis points)
        sum_sq_drift * 1000.0
    }

    /// Get current drift status without full analysis
    pub fn get_drift_status(&self, current_weights: [f64; N_ASSETS]) -> DriftStatus {
        let mut max_drift = 0.0;
        let mut max_drift_asset = 0;

        for i in 0..N_ASSETS {
            let drift = (current_weights[i] - self.target_weights[i]).abs();
            if drift > max_drift {
                max_drift = drift;
                max_drift_asset = i;
            }
        }

        let status_level = if max_drift >= self.config.max_drift_threshold {
            DriftLevel::Critical
        } else if max_drift >= self.config.drift_threshold {
            DriftLevel::Warning
        } else if max_drift >= self.config.drift_threshold * 0.5 {
            DriftLevel::Monitoring
        } else {
            DriftLevel::Normal
        };

        DriftStatus {
            max_drift,
            max_drift_asset: ASSET_LABELS[max_drift_asset].to_string(),
            level: status_level,
            all_within_bounds: max_drift < self.config.drift_threshold,
        }
    }

    /// Update target weights (e.g., from strategic allocation change)
    pub fn update_targets(&mut self, new_targets: [f64; N_ASSETS]) {
        let sum: f64 = new_targets.iter().sum();
        if (sum - 1.0).abs() < 0.01 {
            self.target_weights = new_targets;
        }
    }
}

/// Drift severity level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftLevel {
    Normal,      // < 2.5%
    Monitoring,  // 2.5-5%
    Warning,     // 5-10%
    Critical,    // > 10%
}

/// Summary of portfolio drift
#[derive(Debug, Clone)]
pub struct DriftStatus {
    pub max_drift: f64,
    pub max_drift_asset: String,
    pub level: DriftLevel,
    pub all_within_bounds: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_rebalance_needed() {
        let targets = [0.25, 0.25, 0.25, 0.25];
        let engine = RebalanceEngine::new(targets, 100_000.0);

        // Small drift - no rebalance
        let current = [0.26, 0.24, 0.25, 0.25];
        let analysis = engine.analyze(current, 3600 * 48); // 48 hours

        assert!(!analysis.should_rebalance);
        assert!(analysis.max_drift < engine.config.drift_threshold);
    }

    #[test]
    fn test_threshold_breach() {
        let targets = [0.25, 0.25, 0.25, 0.25];
        let engine = RebalanceEngine::new(targets, 100_000.0);

        // Large drift - should trigger rebalance
        let current = [0.35, 0.20, 0.25, 0.20];
        let analysis = engine.analyze(current, 3600 * 48);

        assert!(analysis.should_rebalance);
        assert_eq!(analysis.trigger, Some(RebalanceTrigger::ThresholdBreach));
        assert!(analysis.max_drift >= engine.config.drift_threshold);
    }

    #[test]
    fn test_trade_generation() {
        let targets = [0.25, 0.25, 0.25, 0.25];
        let engine = RebalanceEngine::new(targets, 100_000.0);

        let current = [0.35, 0.20, 0.25, 0.20];
        let analysis = engine.analyze(current, 3600 * 48);

        // Should have trades to restore balance
        assert!(!analysis.recommended_trades.is_empty());
        
        // BTC should be sold (overweight)
        let btc_trade = analysis.recommended_trades.iter()
            .find(|t| t.asset == "BTC");
        assert!(btc_trade.is_some());
        assert!(btc_trade.unwrap().quantity < 0.0); // Negative = sell
    }

    #[test]
    fn test_time_constraint() {
        let targets = [0.25, 0.25, 0.25, 0.25];
        let mut engine = RebalanceEngine::new(targets, 100_000.0);
        
        // Set recent rebalance time
        engine.last_rebalance_time = 1000000;

        // Drift exists but too soon since last rebalance
        let current = [0.32, 0.22, 0.24, 0.22];
        let analysis = engine.analyze(current, 1000000 + 3600); // 1 hour later

        // Should not rebalance due to time constraint
        // (unless max drift threshold breached)
        if analysis.max_drift < engine.config.max_drift_threshold {
            assert!(!analysis.should_rebalance);
        }
    }

    #[test]
    fn test_max_drift_forced_rebalance() {
        let targets = [0.25, 0.25, 0.25, 0.25];
        let mut engine = RebalanceEngine::new(targets, 100_000.0);
        
        // Set very recent rebalance
        engine.last_rebalance_time = 1000000;

        // Extreme drift - should force rebalance regardless of timing
        let current = [0.45, 0.15, 0.20, 0.20];
        let analysis = engine.analyze(current, 1000000 + 1800); // 30 min later

        assert!(analysis.should_rebalance);
        assert_eq!(analysis.trigger, Some(RebalanceTrigger::ThresholdBreach));
    }
}
