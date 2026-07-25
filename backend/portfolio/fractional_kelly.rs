//! Fractional Kelly Criterion Position Sizer
//!
//! Implements half-Kelly and fractional Kelly strategies for optimal bet sizing.
//! Prevents ruin during drawdowns while maximizing long-term growth.
//! Critical for crypto trading where win rates fluctuate rapidly.
//!
//! # Key Features:
//! - Full Kelly, Half-Kelly, and custom fraction modes
//! - Automatic position scaling based on historical win rate
//! - Drawdown-aware size reduction
//! - Maximum exposure limits per asset
//!
//! # Mathematical Background:
//! Kelly fraction: f* = (p * b - q) / b
//! where p = win probability, q = 1-p, b = odds (win/loss ratio)

use std::f64;

/// Fixed portfolio size
const N_ASSETS: usize = 4;

/// Asset labels for the portfolio
pub const ASSET_LABELS: [&str; N_ASSETS] = ["BTC", "SOL", "ETH", "USDT"];

/// Kelly criterion mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KellyMode {
    /// Full Kelly (aggressive, maximum growth)
    Full,
    /// Half Kelly (recommended, balances growth and volatility)
    Half,
    /// Quarter Kelly (conservative)
    Quarter,
    /// Custom fraction of Kelly
    Custom(f64),
}

impl KellyMode {
    /// Get the fraction multiplier for this mode
    pub fn fraction(&self) -> f64 {
        match self {
            KellyMode::Full => 1.0,
            KellyMode::Half => 0.5,
            KellyMode::Quarter => 0.25,
            KellyMode::Custom(f) => f.clamp(0.0, 1.0),
        }
    }
}

/// Result of Kelly position sizing calculation
#[derive(Debug, Clone)]
pub struct KellyResult {
    /// Optimal fraction to bet for each asset
    pub kelly_fractions: [f64; N_ASSETS],
    /// Actual position sizes (after applying limits)
    pub position_sizes: [f64; N_ASSETS],
    /// Expected growth rate with these fractions
    pub expected_growth_rate: f64,
    /// Whether any positions were capped by limits
    pub was_capped: bool,
    /// Recommended action based on edge analysis
    pub recommendation: KellyRecommendation,
}

/// Trading recommendation based on Kelly analysis
#[derive(Debug, Clone, PartialEq)]
pub enum KellyRecommendation {
    /// Strong buy signal with positive edge
    AggressiveBuy,
    /// Moderate buy with reduced size
    ModerateBuy,
    /// Hold current positions
    Hold,
    /// Reduce exposure
    Reduce,
    /// Exit position (negative edge)
    Exit,
}

/// Fractional Kelly Position Sizer
///
/// Computes optimal position sizes using Kelly criterion with
/// safety constraints for crypto trading.
pub struct FractionalKellySizer {
    /// Kelly mode (Full, Half, Quarter, or Custom)
    mode: KellyMode,
    /// Maximum position size per asset (as fraction of portfolio)
    max_position: [f64; N_ASSETS],
    /// Minimum win rate to take positions
    min_win_rate: f64,
    /// Current drawdown level (0.0 to 1.0)
    drawdown: f64,
    /// Drawdown sensitivity (how much to reduce during drawdowns)
    drawdown_sensitivity: f64,
}

impl FractionalKellySizer {
    /// Create new sizer with default parameters
    pub fn new() -> Self {
        Self {
            mode: KellyMode::Half, // Default to half-Kelly for safety
            max_position: [0.30, 0.25, 0.30, 0.10], // BTC, SOL, ETH, USDT limits
            min_win_rate: 0.45,
            drawdown: 0.0,
            drawdown_sensitivity: 0.5, // Reduce by 50% at max drawdown
        }
    }

    /// Set custom maximum positions
    pub fn with_max_positions(mut self, max_pos: [f64; N_ASSETS]) -> Self {
        self.max_position = max_pos;
        self
    }

    /// Set Kelly mode
    pub fn with_mode(mut self, mode: KellyMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set minimum win rate threshold
    pub fn with_min_win_rate(mut self, min_rate: f64) -> Self {
        self.min_win_rate = min_rate.clamp(0.0, 1.0);
        self
    }

    /// Update current drawdown level
    pub fn set_drawdown(&mut self, drawdown: f64) {
        self.drawdown = drawdown.clamp(0.0, 1.0);
    }

    /// Compute Kelly-optimal position sizes
    ///
    /// # Arguments
    /// * `win_rates` - Historical win rate for each asset [0, 1]
    /// * `payoff_ratios` - Average win/loss ratio for each asset
    /// * `portfolio_value` - Total portfolio value in USDT
    ///
    /// # Returns
    /// KellyResult with optimal fractions and position sizes
    pub fn compute_positions(
        &self,
        win_rates: [f64; N_ASSETS],
        payoff_ratios: [f64; N_ASSETS],
        portfolio_value: f64,
    ) -> KellyResult {
        let kelly_fraction = self.mode.fraction();
        let mut kelly_fractions = [0.0; N_ASSETS];
        let mut position_sizes = [0.0; N_ASSETS];
        let mut was_capped = false;
        let mut worst_recommendation = KellyRecommendation::Hold;

        for i in 0..N_ASSETS {
            let win_rate = win_rates[i];
            let payoff_ratio = payoff_ratios[i].max(0.01); // Avoid division by zero
            
            // Compute raw Kelly fraction
            // f* = (p * b - q) / b where q = 1-p
            let loss_rate = 1.0 - win_rate;
            let numerator = win_rate * payoff_ratio - loss_rate;
            let raw_kelly = if payoff_ratio > 1e-10 {
                numerator / payoff_ratio
            } else {
                0.0
            };

            // Apply fractional Kelly
            let adjusted_kelly = raw_kelly * kelly_fraction;

            // Apply drawdown scaling
            let drawdown_factor = 1.0 - (self.drawdown * self.drawdown_sensitivity);
            let scaled_kelly = adjusted_kelly * drawdown_factor;

            // Enforce minimum win rate
            let final_kelly = if win_rate < self.min_win_rate {
                0.0
            } else {
                scaled_kelly.max(0.0) // No short positions
            };

            // Apply maximum position limit
            let capped_kelly = final_kelly.min(self.max_position[i]);
            if capped_kelly < final_kelly && final_kelly > 0.0 {
                was_capped = true;
            }

            kelly_fractions[i] = final_kelly;
            position_sizes[i] = capped_kelly * portfolio_value;

            // Determine recommendation
            let rec = self._get_recommendation(win_rate, payoff_ratio, final_kelly);
            worst_recommendation = self._worst_recommendation(worst_recommendation, rec);
        }

        // Compute expected growth rate
        let expected_growth = self._compute_growth_rate(&kelly_fractions, &win_rates, &payoff_ratios);

        KellyResult {
            kelly_fractions,
            position_sizes,
            expected_growth_rate: expected_growth,
            was_capped,
            recommendation: worst_recommendation,
        }
    }

    /// Compute Kelly fraction for a single trade
    pub fn single_trade_kelly(&self, win_rate: f64, payoff_ratio: f64) -> f64 {
        if win_rate < self.min_win_rate || payoff_ratio <= 0.0 {
            return 0.0;
        }

        let loss_rate = 1.0 - win_rate;
        let raw_kelly = (win_rate * payoff_ratio - loss_rate) / payoff_ratio;
        
        let adjusted = raw_kelly * self.mode.fraction();
        let drawdown_factor = 1.0 - (self.drawdown * self.drawdown_sensitivity);
        
        adjusted.max(0.0) * drawdown_factor
    }

    /// Get recommendation based on edge analysis
    fn _get_recommendation(
        &self,
        win_rate: f64,
        payoff_ratio: f64,
        kelly_fraction: f64,
    ) -> KellyRecommendation {
        let edge = win_rate * payoff_ratio - (1.0 - win_rate);
        
        if edge < -0.1 {
            KellyRecommendation::Exit
        } else if edge < 0.0 || kelly_fraction < 0.02 {
            KellyRecommendation::Reduce
        } else if edge < 0.1 {
            KellyRecommendation::Hold
        } else if kelly_fraction > 0.20 {
            KellyRecommendation::AggressiveBuy
        } else {
            KellyRecommendation::ModerateBuy
        }
    }

    /// Get the more conservative recommendation
    fn _worst_recommendation(
        &self,
        current: KellyRecommendation,
        new: KellyRecommendation,
    ) -> KellyRecommendation {
        // Order from most to least aggressive
        let aggressiveness = |r: &KellyRecommendation| match r {
            KellyRecommendation::AggressiveBuy => 5,
            KellyRecommendation::ModerateBuy => 4,
            KellyRecommendation::Hold => 3,
            KellyRecommendation::Reduce => 2,
            KellyRecommendation::Exit => 1,
        };

        if aggressiveness(&new) < aggressiveness(&current) {
            new
        } else {
            current
        }
    }

    /// Compute expected geometric growth rate
    fn _compute_growth_rate(
        &self,
        fractions: &[f64; N_ASSETS],
        win_rates: &[f64; N_ASSETS],
        payoff_ratios: &[f64; N_ASSETS],
    ) -> f64 {
        let mut growth = 0.0;

        for i in 0..N_ASSETS {
            let f = fractions[i];
            let p = win_rates[i];
            let b = payoff_ratios[i];

            if f > 0.0 && p > 0.0 && b > 0.0 {
                // Expected log growth: p * ln(1 + f*b) + (1-p) * ln(1 - f)
                let win_growth = (1.0 + f * b).max(1e-10).ln();
                let loss_growth = (1.0 - f).max(1e-10).ln();
                growth += p * win_growth + (1.0 - p) * loss_growth;
            }
        }

        growth
    }

    /// Instantly scale down all positions based on win rate drop
    pub fn emergency_scale_down(&mut self, severity: f64) {
        // Increase drawdown based on severity
        self.drawdown = (self.drawdown + severity * 0.2).min(1.0);
        
        // Optionally switch to more conservative mode
        if severity > 0.5 {
            self.mode = KellyMode::Quarter;
        } else if severity > 0.2 {
            self.mode = KellyMode::Half;
        }
    }
}

impl Default for FractionalKellySizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_kelly_computation() {
        let sizer = FractionalKellySizer::new()
            .with_mode(KellyMode::Half);

        // Good opportunity: 60% win rate, 2:1 payoff
        let win_rates = [0.60, 0.55, 0.50, 0.40];
        let payoffs = [2.0, 1.5, 1.0, 0.5];
        
        let result = sizer.compute_positions(win_rates, payoffs, 100_000.0);

        // BTC should have positive allocation (good edge)
        assert!(result.position_sizes[0] > 0.0, "BTC should have positive position");
        
        // USDT should have zero allocation (negative edge, below min win rate)
        assert!(result.position_sizes[3] == 0.0, "USDT should have zero position");

        // Positions should be within limits
        for i in 0..N_ASSETS {
            let fraction = result.position_sizes[i] / 100_000.0;
            assert!(fraction <= sizer.max_position[i] + 1e-6);
        }
    }

    #[test]
    fn test_drawdown_scaling() {
        let mut sizer = FractionalKellySizer::new()
            .with_mode(KellyMode::Full);

        let win_rates = [0.60, 0.55, 0.58, 0.50];
        let payoffs = [2.0, 1.5, 1.8, 1.0];

        // Compute without drawdown
        sizer.set_drawdown(0.0);
        let result_no_dd = sizer.compute_positions(win_rates, payoffs, 100_000.0);

        // Compute with 50% drawdown
        sizer.set_drawdown(0.5);
        let result_with_dd = sizer.compute_positions(win_rates, payoffs, 100_000.0);

        // Positions should be smaller with drawdown
        for i in 0..N_ASSETS {
            assert!(
                result_with_dd.position_sizes[i] <= result_no_dd.position_sizes[i] + 1e-6,
                "Position {} should be smaller during drawdown",
                i
            );
        }
    }

    #[test]
    fn test_win_rate_threshold() {
        let sizer = FractionalKellySizer::new()
            .with_min_win_rate(0.50);

        // Edge case: exactly at threshold
        let result_at = sizer.single_trade_kelly(0.50, 2.0);
        assert!(result_at > 0.0, "Should allow trades at threshold");

        // Below threshold
        let result_below = sizer.single_trade_kelly(0.49, 2.0);
        assert!(result_below == 0.0, "Should reject trades below threshold");
    }

    #[test]
    fn test_emergency_scale_down() {
        let mut sizer = FractionalKellySizer::new()
            .with_mode(KellyMode::Full);

        assert_eq!(sizer.mode, KellyMode::Full);
        
        // Trigger severe emergency
        sizer.emergency_scale_down(0.6);
        
        // Should switch to quarter Kelly
        assert_eq!(sizer.mode, KellyMode::Quarter);
        assert!(sizer.drawdown > 0.0);
    }

    #[test]
    fn test_position_caps() {
        let max_positions = [0.10, 0.10, 0.10, 0.10]; // Tight limits
        let sizer = FractionalKellySizer::new()
            .with_max_positions(max_positions);

        // Very attractive opportunity
        let win_rates = [0.70, 0.70, 0.70, 0.70];
        let payoffs = [3.0, 3.0, 3.0, 3.0];
        
        let result = sizer.compute_positions(win_rates, payoffs, 100_000.0);

        // Should be capped
        assert!(result.was_capped, "Positions should be capped");
        
        // Verify caps are respected
        for i in 0..N_ASSETS {
            let fraction = result.position_sizes[i] / 100_000.0;
            assert!(fraction <= max_positions[i] + 1e-6);
        }
    }
}
