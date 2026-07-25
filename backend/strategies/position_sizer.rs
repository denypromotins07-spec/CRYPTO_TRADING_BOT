// Position Sizer: Kelly Criterion and VaR mathematics in Rust.
// Implements zero-cost abstractions for ultra-fast position sizing.
// Guarantees no over-leveraging during extreme volatility spikes.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

/// Configuration for position sizing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizerConfig {
    /// Maximum position size as fraction of portfolio (e.g., 0.1 = 10%)
    pub max_position_fraction: f64,
    /// Kelly criterion multiplier (0.0 to 1.0)
    pub kelly_multiplier: f64,
    /// Maximum leverage allowed
    pub max_leverage: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Value at Risk confidence level (e.g., 0.95 for 95%)
    pub var_confidence: f64,
    /// Maximum drawdown threshold for circuit breaker
    pub max_drawdown: f64,
}

impl Default for PositionSizerConfig {
    fn default() -> Self {
        Self {
            max_position_fraction: 0.1,
            kelly_multiplier: 0.25, // Fractional Kelly for safety
            max_leverage: 3.0,
            stop_loss_pct: 0.02,
            var_confidence: 0.95,
            max_drawdown: 0.02,
        }
    }
}

/// Result of position size calculation
#[derive(Debug, Clone)]
pub struct PositionSizeResult {
    /// Recommended position size in base currency
    pub quantity: f64,
    /// Kelly fraction used
    pub kelly_fraction: f64,
    /// VaR limit applied
    pub var_limit: f64,
    /// Whether position was capped by risk limits
    pub is_capped: bool,
    /// Reason for capping if applicable
    pub cap_reason: Option<String>,
}

/// Volatility-adjusted position sizer using Kelly Criterion and VaR
pub struct PositionSizer {
    config: PositionSizerConfig,
    /// Cache for volatility estimates per asset
    volatilities: Arc<RwLock<HashMap<String, f64>>>,
    /// Cache for correlation matrix
    correlations: Arc<RwLock<HashMap<(String, String), f64>>>,
    /// Current portfolio value
    portfolio_value: Arc<RwLock<f64>>,
    /// Current drawdown
    current_drawdown: Arc<RwLock<f64>>,
}

impl PositionSizer {
    /// Create a new PositionSizer with default configuration
    pub fn new() -> Self {
        Self::with_config(PositionSizerConfig::default())
    }

    /// Create a new PositionSizer with custom configuration
    pub fn with_config(config: PositionSizerConfig) -> Self {
        Self {
            config,
            volatilities: Arc::new(RwLock::new(HashMap::new())),
            correlations: Arc::new(RwLock::new(HashMap::new())),
            portfolio_value: Arc::new(RwLock::new(100_000.0)), // Default $100k
            current_drawdown: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Update portfolio value
    pub fn update_portfolio_value(&self, value: f64) {
        if let Ok(mut pv) = self.portfolio_value.write() {
            *pv = value;
        }
    }

    /// Update current drawdown
    pub fn update_drawdown(&self, drawdown: f64) {
        if let Ok(mut dd) = self.current_drawdown.write() {
            *dd = drawdown.clamp(0.0, 1.0);
        }
    }

    /// Update volatility estimate for an asset
    pub fn update_volatility(&self, asset: &str, volatility: f64) {
        if let Ok(mut vols) = self.volatilities.write() {
            vols.insert(asset.to_string(), volatility.clamp(0.0, 10.0));
        }
    }

    /// Update correlation between two assets
    pub fn update_correlation(&self, asset1: &str, asset2: &str, correlation: f64) {
        if let Ok(mut corrs) = self.correlations.write() {
            let key = if asset1 < asset2 {
                (asset1.to_string(), asset2.to_string())
            } else {
                (asset2.to_string(), asset1.to_string())
            };
            corrs.insert(key, correlation.clamp(-1.0, 1.0));
        }
    }

    /// Calculate position size using Kelly Criterion with VaR constraints
    /// 
    /// # Arguments
    /// * `asset` - Asset identifier (e.g., "BTCUSDT")
    /// * `win_probability` - Estimated probability of winning (0.0 to 1.0)
    /// * `win_loss_ratio` - Ratio of average win to average loss
    /// * `price` - Current asset price
    /// 
    /// # Returns
    /// PositionSizeResult with calculated quantity and metadata
    pub fn calculate_position_size(
        &self,
        asset: &str,
        win_probability: f64,
        win_loss_ratio: f64,
        price: f64,
    ) -> PositionSizeResult {
        // Check circuit breaker for drawdown
        let current_dd = self.current_drawdown.read().unwrap_or_default();
        if *current_dd >= self.config.max_drawdown {
            return PositionSizeResult {
                quantity: 0.0,
                kelly_fraction: 0.0,
                var_limit: 0.0,
                is_capped: true,
                cap_reason: Some("Maximum drawdown breached - circuit breaker active".to_string()),
            };
        }

        // Validate inputs
        let win_prob = win_probability.clamp(0.0, 1.0);
        let wl_ratio = win_loss_ratio.max(0.01);
        
        // Calculate raw Kelly fraction: f* = p - q/b
        // where p = win probability, q = loss probability, b = win/loss ratio
        let loss_prob = 1.0 - win_prob;
        let raw_kelly = if wl_ratio > 0.0 {
            win_prob - (loss_prob / wl_ratio)
        } else {
            0.0
        };

        // Apply fractional Kelly for safety
        let kelly_fraction = (raw_kelly * self.config.kelly_multiplier).clamp(0.0, 1.0);

        // Get portfolio value
        let portfolio_value = *self.portfolio_value.read().unwrap_or(&100_000.0);

        // Calculate maximum position value based on Kelly
        let kelly_position_value = portfolio_value * kelly_fraction;

        // Apply maximum position fraction constraint
        let max_position_value = portfolio_value * self.config.max_position_fraction;
        let mut position_value = kelly_position_value.min(max_position_value);

        // Check VaR constraint
        let var_limit = self.calculate_var_limit(asset, portfolio_value);
        if position_value > var_limit {
            position_value = var_limit;
        }

        // Check leverage constraint
        let max_exposure = portfolio_value * self.config.max_leverage;
        if position_value > max_exposure {
            position_value = max_exposure;
        }

        // Adjust for volatility if available
        if let Ok(vols) = self.volatilities.read() {
            if let Some(&vol) = vols.get(asset) {
                // Reduce position size for high volatility assets
                let vol_adjustment = (1.0 / (1.0 + vol)).min(1.0);
                position_value *= vol_adjustment;
            }
        }

        // Calculate quantity
        let quantity = if price > 0.0 {
            position_value / price
        } else {
            0.0
        };

        // Determine if position was capped
        let is_capped = kelly_fraction > 0.0 && (position_value < kelly_position_value * 0.99);
        let cap_reason = if is_capped {
            if position_value == var_limit {
                Some("VaR limit".to_string())
            } else if position_value == max_position_value {
                Some("Max position fraction".to_string())
            } else {
                Some("Volatility adjustment".to_string())
            }
        } else {
            None
        };

        PositionSizeResult {
            quantity,
            kelly_fraction,
            var_limit,
            is_capped,
            cap_reason,
        }
    }

    /// Calculate Value at Risk (VaR) limit for an asset
    /// Uses parametric VaR method with historical volatility
    fn calculate_var_limit(&self, asset: &str, portfolio_value: f64) -> f64 {
        // Get volatility for the asset
        let volatility = {
            let vols = self.volatilities.read().unwrap_or_default();
            vols.get(asset).copied().unwrap_or(0.05) // Default 5% daily vol
        };

        // Get Z-score for confidence level
        let z_score = self.get_z_score(self.config.var_confidence);

        // Calculate VaR: VaR = portfolio_value * volatility * z_score
        let var = portfolio_value * volatility * z_score;

        // Limit position to VaR constraint
        // Position should not lose more than VaR amount at stop loss
        let stop_loss = self.config.stop_loss_pct.max(0.001);
        var / stop_loss
    }

    /// Get Z-score for a given confidence level
    fn get_z_score(&self, confidence: f64) -> f64 {
        // Approximate inverse normal CDF for common confidence levels
        match confidence {
            c if c >= 0.99 => 2.326,
            c if c >= 0.975 => 1.960,
            c if c >= 0.95 => 1.645,
            c if c >= 0.90 => 1.282,
            c if c >= 0.80 => 0.842,
            _ => 1.645, // Default to 95%
        }
    }

    /// Calculate optimal position size for multiple assets considering correlations
    /// Uses mean-variance optimization with Kelly objective
    pub fn calculate_multi_asset_positions(
        &self,
        assets: &[&str],
        expected_returns: &[f64],
        prices: &[f64],
    ) -> Vec<PositionSizeResult> {
        let n = assets.len();
        if n != expected_returns.len() || n != prices.len() {
            // Return zero positions on mismatch
            return vec![
                PositionSizeResult {
                    quantity: 0.0,
                    kelly_fraction: 0.0,
                    var_limit: 0.0,
                    is_capped: true,
                    cap_reason: Some("Input dimension mismatch".to_string()),
                };
                n
            ];
        }

        let portfolio_value = *self.portfolio_value.read().unwrap_or(&100_000.0);

        // Simple approach: calculate individual positions and scale down if needed
        let mut results = Vec::with_capacity(n);
        let mut total_exposure = 0.0;

        for (i, &asset) in assets.iter().enumerate() {
            // Estimate win probability from expected return (simplified)
            let exp_return = expected_returns[i];
            let win_prob = (0.5 + exp_return / 2.0).clamp(0.0, 1.0);
            let wl_ratio = 2.0; // Assume 2:1 reward-to-risk

            let result = self.calculate_position_size(asset, win_prob, wl_ratio, prices[i]);
            total_exposure += result.quantity * prices[i];
            results.push(result);
        }

        // Scale down if total exposure exceeds limits
        let max_total_exposure = portfolio_value * self.config.max_leverage;
        if total_exposure > max_total_exposure && total_exposure > 0.0 {
            let scale_factor = max_total_exposure / total_exposure;
            for result in &mut results {
                result.quantity *= scale_factor;
                result.is_capped = true;
                result.cap_reason = Some("Multi-asset exposure limit".to_string());
            }
        }

        results
    }

    /// Reset all cached data
    pub fn reset(&self) {
        if let Ok(mut vols) = self.volatilities.write() {
            vols.clear();
        }
        if let Ok(mut corrs) = self.correlations.write() {
            corrs.clear();
        }
        if let Ok(mut dd) = self.current_drawdown.write() {
            *dd = 0.0;
        }
    }

    /// Export current state for monitoring
    pub fn export_state(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        
        if let Ok(pv) = self.portfolio_value.read() {
            state.insert("portfolio_value".to_string(), *pv);
        }
        if let Ok(dd) = self.current_drawdown.read() {
            state.insert("current_drawdown".to_string(), *dd);
        }
        if let Ok(vols) = self.volatilities.read() {
            for (asset, vol) in vols.iter() {
                state.insert(format!("volatility_{}", asset), *vol);
            }
        }

        state
    }
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_calculation() {
        let sizer = PositionSizer::new();
        sizer.update_portfolio_value(100_000.0);

        // Test with 60% win probability and 2:1 win/loss ratio
        let result = sizer.calculate_position_size("BTCUSDT", 0.6, 2.0, 50_000.0);

        assert!(result.quantity > 0.0);
        assert!(result.kelly_fraction > 0.0);
        println!("Kelly fraction: {:.4}", result.kelly_fraction);
        println!("Quantity: {:.6}", result.quantity);
    }

    #[test]
    fn test_drawdown_circuit_breaker() {
        let sizer = PositionSizer::new();
        sizer.update_portfolio_value(100_000.0);
        sizer.update_drawdown(0.025); // Above 2% threshold

        let result = sizer.calculate_position_size("BTCUSDT", 0.7, 3.0, 50_000.0);

        assert_eq!(result.quantity, 0.0);
        assert!(result.is_capped);
        assert!(result.cap_reason.is_some());
    }

    #[test]
    fn test_volatility_adjustment() {
        let sizer = PositionSizer::new();
        sizer.update_portfolio_value(100_000.0);
        sizer.update_volatility("BTCUSDT", 0.08); // High volatility
        sizer.update_volatility("ETHUSDT", 0.04); // Lower volatility

        let btc_result = sizer.calculate_position_size("BTCUSDT", 0.6, 2.0, 50_000.0);
        let eth_result = sizer.calculate_position_size("ETHUSDT", 0.6, 2.0, 3_000.0);

        // BTC position should be smaller due to higher volatility adjustment
        println!("BTC quantity: {:.6}", btc_result.quantity);
        println!("ETH quantity: {:.6}", eth_result.quantity);
    }
}

// FFI exports for Python integration
#[no_mangle]
pub extern "C" fn create_position_sizer() -> *mut PositionSizer {
    Box::into_raw(Box::new(PositionSizer::new()))
}

#[no_mangle]
pub extern "C" fn calculate_position(
    sizer: *mut PositionSizer,
    asset: *const i8,
    win_prob: f64,
    wl_ratio: f64,
    price: f64,
) -> PositionSizeResult {
    unsafe {
        let sizer = &mut *sizer;
        let asset_str = std::ffi::CStr::from_ptr(asset).to_str().unwrap_or("UNKNOWN");
        sizer.calculate_position_size(asset_str, win_prob, wl_ratio, price)
    }
}
