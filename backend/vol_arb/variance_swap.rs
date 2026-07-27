//! Variance Swap Pricer and Hedger
//! Prices variance swaps using the volatility surface
//! Dynamically delta-hedges using underlying perpetual swaps
//! 
//! Variance swap payoff: Notional * (σ_realized² - σ_strike²)

use std::fmt;

/// Variance swap contract specification
#[derive(Debug, Clone)]
pub struct VarianceSwap {
    pub notional: f64,         // Per vol point (e.g., $100 per vol sq)
    pub strike: f64,           // Agreed variance strike
    pub maturity: f64,         // Time to maturity in years
    pub sampling_frequency: SamplingFrequency,
}

/// Sampling frequency for realized variance calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingFrequency {
    Daily,
    Hourly,
    Minute,
    Tick,
}

impl SamplingFrequency {
    /// Number of samples per year
    pub fn samples_per_year(&self) -> f64 {
        match self {
            SamplingFrequency::Daily => 252.0,
            SamplingFrequency::Hourly => 252.0 * 24.0,
            SamplingFrequency::Minute => 252.0 * 24.0 * 60.0,
            SamplingFrequency::Tick => f64::MAX,  // Continuous limit
        }
    }
}

/// Variance swap valuation result
#[derive(Debug, Clone)]
pub struct VarianceSwapValuation {
    pub fair_value: f64,       // Current fair value of the swap
    pub pv: f64,              // Present value to long position
    pub delta: f64,           // Delta hedge ratio
    pub vega: f64,            // Vega exposure
    pub gamma: f64,           // Gamma exposure
}

/// Realized variance calculator
pub struct RealizedVarianceCalculator {
    log_returns: Vec<f64>,
    sampling: SamplingFrequency,
}

impl RealizedVarianceCalculator {
    pub fn new(sampling: SamplingFrequency) -> Self {
        Self {
            log_returns: Vec::new(),
            sampling,
        }
    }

    /// Add a price observation
    pub fn add_price(&mut self, price: f64) {
        if let Some(last_price) = self.log_returns.last().map(|_| price).or(None) {
            // This is simplified - in production would track actual prices
        }
    }

    /// Add a log return directly
    pub fn add_log_return(&mut self, ret: f64) {
        self.log_returns.push(ret);
    }

    /// Calculate annualized realized variance
    pub fn realized_variance(&self) -> f64 {
        if self.log_returns.is_empty() {
            return 0.0;
        }

        let n = self.log_returns.len() as f64;
        let mean_ret = self.log_returns.iter().sum::<f64>() / n;
        
        // Sample variance with annualization
        let var = self.log_returns.iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>() / (n - 1.0);
        
        let annualization_factor = self.sampling.samples_per_year();
        var * annualization_factor
    }

    /// Reset for new period
    pub fn reset(&mut self) {
        self.log_returns.clear();
    }
}

/// Variance swap pricer using volatility surface integration
pub struct VarianceSwapPricer {
    risk_free_rate: f64,
}

impl VarianceSwapPricer {
    pub fn new(risk_free_rate: f64) -> Self {
        Self { risk_free_rate }
    }

    /// Price variance swap using replication portfolio
    /// Fair variance strike = 2/T * ∫(O(K)/K²) dK over all strikes
    pub fn fair_variance_strike(&self, option_prices: &[(f64, f64)], 
                                 forward: f64, t: f64) -> Option<f64> {
        if option_prices.is_empty() || t <= 0.0 {
            return None;
        }

        // Sort by strike
        let mut sorted_prices = option_prices.to_vec();
        sorted_prices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut integral = 0.0;

        // Trapezoidal integration of O(K)/K²
        for i in 0..sorted_prices.len() - 1 {
            let (k1, o1) = sorted_prices[i];
            let (k2, o2) = sorted_prices[i + 1];

            if k1 <= 0.0 || k2 <= 0.0 {
                continue;
            }

            // Average of O(K)/K² at endpoints
            let f1 = o1 / (k1 * k1);
            let f2 = o2 / (k2 * k2);
            
            // Trapezoidal rule
            integral += 0.5 * (f1 + f2) * (k2 - k1);
        }

        // Carr-Madan formula: fair_var = 2/T * integral
        let fair_var = 2.0 / t * integral;

        if fair_var > 0.0 && fair_var < 10.0 {
            Some(fair_var)
        } else {
            None
        }
    }

    /// Value an existing variance swap
    pub fn value_swap(&self, swap: &VarianceSwap, current_forward_var: f64,
                      realized_var_so_far: f64, elapsed_fraction: f64) -> VarianceSwapValuation {
        // Expected final variance = weighted average of realized and forward
        let expected_final_var = realized_var_so_far * elapsed_fraction 
            + current_forward_var * (1.0 - elapsed_fraction);

        // PV = notional * (expected_final_var - strike) * exp(-rT)
        let discount = (-self.risk_free_rate * swap.maturity).exp();
        let pv = swap.notional * (expected_final_var - swap.strike) * discount;

        // Greeks
        let delta = self.compute_delta(swap, elapsed_fraction);
        let vega = swap.notional * (1.0 - elapsed_fraction) * discount;
        let gamma = 0.0;  // Variance swaps are linear in variance

        VarianceSwapValuation {
            fair_value: expected_final_var,
            pv,
            delta,
            vega,
            gamma,
        }
    }

    /// Compute delta hedge ratio for variance swap
    /// Delta ≈ 2 * notional * (σ_realized - σ_strike) / σ_spot
    fn compute_delta(&self, swap: &VarianceSwap, elapsed: f64) -> f64 {
        // Simplified: delta scales with remaining exposure
        swap.notional * (1.0 - elapsed) * 2.0
    }

    /// Generate dynamic hedging instructions
    pub fn generate_hedge_instructions(&self, valuation: &VarianceSwapValuation,
                                        spot_price: f64, current_position: f64) -> HedgeInstruction {
        let target_delta = valuation.delta * spot_price;
        let trade_qty = target_delta - current_position;

        HedgeInstruction {
            action: if trade_qty.abs() > 0.01 {
                if trade_qty > 0.0 { Action::Buy } else { Action::Sell }
            } else {
                Action::Hold
            },
            quantity: trade_qty.abs(),
            instrument: "PERP".to_string(),
            urgency: if trade_qty.abs() > 1.0 { Urgency::High } else { Urgency::Normal },
        }
    }
}

/// Hedging instruction
#[derive(Debug, Clone)]
pub struct HedgeInstruction {
    pub action: Action,
    pub quantity: f64,
    pub instrument: String,
    pub urgency: Urgency,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Urgency {
    High,
    Normal,
    Low,
}

/// Jump risk premium estimator
/// Crypto markets have significant jump risk that must be priced
pub struct JumpRiskPremium {
    historical_jumps: Vec<f64>,
    avg_jump_size: f64,
    jump_frequency: f64,
}

impl JumpRiskPremium {
    pub fn new() -> Self {
        Self {
            historical_jumps: Vec::new(),
            avg_jump_size: 0.0,
            jump_frequency: 0.0,
        }
    }

    /// Record a detected jump
    pub fn record_jump(&mut self, jump_size: f64) {
        self.historical_jumps.push(jump_size.abs());
        
        // Update statistics
        if !self.historical_jumps.is_empty() {
            self.avg_jump_size = self.historical_jumps.iter().sum::<f64>() 
                / self.historical_jumps.len() as f64;
        }
    }

    /// Estimate jump risk premium to add to variance strike
    pub fn estimate_premium(&self) -> f64 {
        if self.historical_jumps.is_empty() {
            return 0.0;
        }

        // Premium = λ * E[J²] where λ is jump frequency
        let jump_intensity = self.historical_jumps.len() as f64 / 252.0;  // Per year
        let second_moment = self.historical_jumps.iter()
            .map(|j| j.powi(2))
            .sum::<f64>() / self.historical_jumps.len() as f64;

        jump_intensity * second_moment
    }
}

impl Default for JumpRiskPremium {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realized_variance() {
        let mut calc = RealizedVarianceCalculator::new(SamplingFrequency::Daily);
        
        // Add some returns
        calc.add_log_return(0.01);
        calc.add_log_return(-0.02);
        calc.add_log_return(0.015);
        
        let rv = calc.realized_variance();
        assert!(rv > 0.0);
    }

    #[test]
    fn test_fair_strike() {
        let pricer = VarianceSwapPricer::new(0.0);
        
        // Synthetic option prices around forward
        let prices = vec![
            (90.0, 12.0),
            (95.0, 8.0),
            (100.0, 5.0),
            (105.0, 3.0),
            (110.0, 1.5),
        ];
        
        let fair_var = pricer.fair_variance_strike(&prices, 100.0, 0.25);
        assert!(fair_var.is_some());
        assert!(fair_var.unwrap() > 0.0);
    }

    #[test]
    fn test_swap_valuation() {
        let pricer = VarianceSwapPricer::new(0.0);
        let swap = VarianceSwap {
            notional: 100.0,
            strike: 0.04,  // 20% vol squared
            maturity: 0.25,
            sampling: SamplingFrequency::Daily,
        };
        
        let val = pricer.value_swap(&swap, 0.05, 0.03, 0.5);
        assert!(val.pv > 0.0);  // Should be positive (long variance)
    }
}
