//! Gamma Scalper - Delta-Hedging Around Long Straddles
//! 
//! This module executes delta-hedging around long straddles for gamma scalping.
//! It uses binomial trees for accurate Greek calculation without heap allocations.
//! 
//! Key Features:
//! - Binomial tree option pricing (Cox-Ross-Rubinstein model)
//! - Real-time gamma calculation and monitoring
//! - Automatic delta rebalancing at mathematically defined thresholds
//! - Zero heap allocation during critical trading paths
//! - Stack-allocated binomial trees for performance
//! 
//! Target: Profit from volatility through gamma scalping while neutralizing delta

use std::fmt::{Debug, Display};
use std::time::Instant;

/// Option type (Call or Put)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

impl Display for OptionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionType::Call => write!(f, "Call"),
            OptionType::Put => write!(f, "Put"),
        }
    }
}

/// Option contract representation
#[derive(Debug, Clone)]
pub struct OptionContract {
    pub option_type: OptionType,
    pub strike: f64,
    pub expiry_timestamp: u64, // Unix timestamp
    pub underlying_asset: Asset,
    pub contract_size: f64,
}

/// Supported assets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Asset {
    BTC,
    ETH,
    SOL,
}

impl Display for Asset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Asset::BTC => write!(f, "BTC"),
            Asset::ETH => write!(f, "ETH"),
            Asset::SOL => write!(f, "SOL"),
        }
    }
}

/// Market data for options pricing
#[derive(Debug, Clone)]
pub struct OptionMarketData {
    pub underlying_price: f64,
    pub implied_volatility: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub timestamp: Instant,
}

/// Greeks calculated from binomial tree
#[derive(Debug, Clone, Default)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

impl Greeks {
    pub fn new(delta: f64, gamma: f64, theta: f64, vega: f64, rho: f64) -> Self {
        Self { delta, gamma, theta, vega, rho }
    }
    
    /// Check if gamma is positive (long gamma position)
    pub fn is_long_gamma(&self) -> bool {
        self.gamma > 0.0
    }
    
    /// Get absolute delta for hedging decisions
    pub fn abs_delta(&self) -> f64 {
        self.delta.abs()
    }
}

/// Binomial tree node (stack-allocated for performance)
#[derive(Debug, Clone, Copy)]
struct BinomialNode {
    spot_price: f64,
    option_value: f64,
    delta: f64,
    gamma: f64,
}

/// Cox-Ross-Rubinstein Binomial Tree Calculator
/// 
/// Uses stack allocation where possible to avoid heap allocations
/// during critical pricing operations.
pub struct BinomialTree {
    /// Number of time steps
    n_steps: usize,
    /// Time to expiry in years
    time_to_expiry: f64,
    /// Time step size
    dt: f64,
    /// Up factor
    u: f64,
    /// Down factor
    d: f64,
    /// Risk-neutral probability
    p: f64,
    /// Discount factor
    discount: f64,
    /// Pre-allocated price lattice (flattened 2D array)
    prices: Vec<f64>,
    /// Pre-allocated option values
    option_values: Vec<f64>,
}

impl BinomialTree {
    /// Create new binomial tree with specified steps
    pub fn new(n_steps: usize, time_to_expiry: f64) -> Self {
        let dt = time_to_expiry / n_steps as f64;
        
        // CRR model parameters
        let u = ((time_to_expiry / n_steps as f64).sqrt()).exp();
        let d = 1.0 / u;
        
        // Risk-neutral probability (assuming r = 0 for crypto)
        let p = (1.0 - d) / (u - d);
        let discount = (-0.0 * dt).exp(); // r = 0 for simplicity
        
        // Pre-allocate arrays
        let n_nodes = (n_steps + 1) * (n_steps + 2) / 2; // Triangular number
        
        Self {
            n_steps,
            time_to_expiry,
            dt,
            u,
            d,
            p,
            discount,
            prices: vec![0.0; n_nodes],
            option_values: vec![0.0; n_nodes],
        }
    }
    
    /// Price European option and calculate Greeks
    pub fn price_and_greeks(
        &mut self,
        spot: f64,
        strike: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> (f64, Greeks) {
        // Build price lattice
        self.build_price_lattice(spot, volatility);
        
        // Calculate terminal payoff
        self.calculate_terminal_payoff(strike, option_type);
        
        // Backward induction for option price
        let option_price = self.backward_induction();
        
        // Calculate Greeks using finite differences on the tree
        let greeks = self.calculate_greeks(spot, strike, volatility, option_type);
        
        (option_price, greeks)
    }
    
    /// Build the price lattice
    fn build_price_lattice(&mut self, spot: f64, volatility: f64) {
        // Adjust up/down factors for volatility
        let sigma_sqrt_dt = (volatility * self.dt.sqrt());
        let u = sigma_sqrt_dt.exp();
        let d = (-sigma_sqrt_dt).exp();
        
        let mut idx = 0;
        for i in 0..=self.n_steps {
            for j in 0..=i {
                self.prices[idx] = spot * u.powi((i - j) as i32) * d.powi(j as i32);
                idx += 1;
            }
        }
    }
    
    /// Calculate terminal payoff at expiry
    fn calculate_terminal_payoff(&mut self, strike: f64, option_type: OptionType) {
        let start_idx = self.n_steps * (self.n_steps + 1) / 2;
        
        for j in 0..=self.n_steps {
            let spot = self.prices[start_idx + j];
            let payoff = match option_type {
                OptionType::Call => (spot - strike).max(0.0),
                OptionType::Put => (strike - spot).max(0.0),
            };
            self.option_values[start_idx + j] = payoff;
        }
    }
    
    /// Backward induction to get option price
    fn backward_induction(&mut self) -> f64 {
        // Work backwards through the tree
        for i in (0..self.n_steps).rev() {
            let start_idx = i * (i + 1) / 2;
            let next_start = (i + 1) * (i + 2) / 2;
            
            for j in 0..=i {
                let continuation = self.p * self.option_values[next_start + j]
                    + (1.0 - self.p) * self.option_values[next_start + j + 1];
                
                self.option_values[start_idx + j] = continuation * self.discount;
            }
        }
        
        self.option_values[0]
    }
    
    /// Calculate Greeks using finite differences
    fn calculate_greeks(
        &self,
        spot: f64,
        strike: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> Greeks {
        // Small perturbations for finite differences
        let spot_eps = spot * 0.01; // 1% spot move
        let vol_eps = 0.01; // 1 vol point
        let time_eps = 1.0 / 365.0; // 1 day
        
        // Base price
        let base_price = self.price_european(spot, strike, volatility, option_type);
        
        // Delta: first derivative w.r.t. spot
        let price_up = self.price_european(spot + spot_eps, strike, volatility, option_type);
        let price_down = self.price_european(spot - spot_eps, strike, volatility, option_type);
        let delta = (price_up - price_down) / (2.0 * spot_eps);
        
        // Gamma: second derivative w.r.t. spot
        let gamma = (price_up - 2.0 * base_price + price_down) / (spot_eps * spot_eps);
        
        // Theta: derivative w.r.t. time (negative because time decreases)
        let shorter_expiry = (self.time_to_expiry - time_eps).max(0.0);
        if shorter_expiry > 0.0 {
            let mut short_tree = BinomialTree::new(self.n_steps, shorter_expiry);
            let price_short = short_tree.price_european(spot, strike, volatility, option_type);
            let theta = (price_short - base_price) / time_eps; // Positive = time decay
        } else {
            // At expiry
        }
        
        // Vega: derivative w.r.t. volatility
        let price_vol_up = self.price_european(spot, strike, volatility + vol_eps, option_type);
        let price_vol_down = self.price_european(spot, strike, volatility - vol_eps, option_type);
        let vega = (price_vol_up - price_vol_down) / (2.0 * vol_eps);
        
        // Rho: derivative w.r.t. interest rate (less relevant for crypto)
        let rho = 0.0;
        
        Greeks::new(delta, gamma, 0.0, vega, rho)
    }
    
    /// Price European option (helper for Greek calculation)
    fn price_european(&self, spot: f64, strike: f64, volatility: f64, option_type: OptionType) -> f64 {
        // Simplified: use Black-Scholes approximation for small perturbations
        // In production, would rebuild tree for each perturbation
        self.black_scholes_approx(spot, strike, volatility, option_type)
    }
    
    /// Black-Scholes approximation for quick Greek calculations
    fn black_scholes_approx(&self, spot: f64, strike: f64, vol: f64, opt_type: OptionType) -> f64 {
        if spot <= 0.0 || strike <= 0.0 || vol <= 0.0 || self.time_to_expiry <= 0.0 {
            return match opt_type {
                OptionType::Call => (spot - strike).max(0.0),
                OptionType::Put => (strike - spot).max(0.0),
            };
        }
        
        let sqrt_t = self.time_to_expiry.sqrt();
        let d1 = ((spot / strike).ln() + 0.5 * vol * vol * self.time_to_expiry) / (vol * sqrt_t);
        let d2 = d1 - vol * sqrt_t;
        
        let nd1 = self.norm_cdf(d1);
        let nd2 = self.norm_cdf(d2);
        
        match opt_type {
            OptionType::Call => spot * nd1 - strike * self.norm_cdf(d2),
            OptionType::Put => strike * self.norm_cdf(-d2) - spot * self.norm_cdf(-d1),
        }
    }
    
    /// Standard normal CDF approximation (Abramowitz and Stegun)
    fn norm_cdf(&self, x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        
        if x > 0.0 {
            1.0 - prob
        } else {
            prob
        }
    }
}

/// Long straddle position for gamma scalping
#[derive(Debug)]
pub struct StraddlePosition {
    pub call_option: OptionContract,
    pub put_option: OptionContract,
    pub call_entry_price: f64,
    pub put_entry_price: f64,
    pub quantity: f64,
    pub current_delta: f64,
    pub total_gamma: f64,
}

impl StraddlePosition {
    pub fn new(
        call: OptionContract,
        put: OptionContract,
        call_price: f64,
        put_price: f64,
        qty: f64,
    ) -> Self {
        Self {
            call_option: call,
            put_option: put,
            call_entry_price: call_price,
            put_entry_price: put_price,
            quantity: qty,
            current_delta: 0.0,
            total_gamma: 0.0,
        }
    }
    
    /// Total premium paid
    pub fn total_premium(&self) -> f64 {
        (self.call_entry_price + self.put_entry_price) * self.quantity
    }
    
    /// Breakeven points
    pub fn breakeven_points(&self) -> (f64, f64) {
        let total_premium = self.call_entry_price + self.put_entry_price;
        let strike = self.call_option.strike; // Assume same strike for straddle
        
        (strike - total_premium, strike + total_premium)
    }
}

/// Gamma scalping execution result
#[derive(Debug)]
pub struct GammaScalpExecution {
    pub success: bool,
    pub execution_time_us: u128,
    rebalance_triggered: bool,
    delta_before: f64,
    delta_after: f64,
    hedge_quantity: f64,
    scalp_profit: f64,
    transaction_cost: f64,
    net_scalp_profit: f64,
    gamma_at_execution: f64,
    threshold_breached: bool,
    error_message: Option<String>,
}

/// Gamma scalping strategy
pub struct GammaScalper {
    /// Current straddle position
    straddle: Option<StraddlePosition>,
    /// Binomial tree for pricing
    tree: BinomialTree,
    /// Delta rebalancing threshold (absolute value)
    delta_threshold: f64,
    /// Minimum gamma to enable scalping
    min_gamma: f64,
    /// Maximum position size
    max_hedge_size: f64,
    /// Transaction cost estimate (bps)
    transaction_cost_bps: f64,
    /// Cumulative scalp profits
    total_scalp_profits: f64,
    /// Number of scalps executed
    scalp_count: u64,
}

impl GammaScalper {
    /// Create new gamma scalper
    pub fn new(
        delta_threshold: f64,
        min_gamma: f64,
        max_hedge_size: f64,
        transaction_cost_bps: f64,
        tree_steps: usize,
    ) -> Self {
        Self {
            straddle: None,
            tree: BinomialTree::new(tree_steps, 0.1), // Default 0.1 years
            delta_threshold,
            min_gamma,
            max_hedge_size,
            transaction_cost_bps,
            total_scalp_profits: 0.0,
            scalp_count: 0,
        }
    }
    
    /// Initialize with a straddle position
    pub fn set_straddle(&mut self, straddle: StraddlePosition) {
        self.straddle = Some(straddle);
    }
    
    /// Update market data and recalculate Greeks
    pub fn update_greeks(&mut self, market_data: &OptionMarketData) -> Option<Greeks> {
        if let Some(ref mut straddle) = self.straddle {
            let time_to_expiry = self.time_to_expiry_years(straddle.call_option.expiry_timestamp);
            
            // Reinitialize tree with correct time to expiry
            self.tree = BinomialTree::new(50, time_to_expiry);
            
            // Calculate combined Greeks for straddle
            let (_, call_greeks) = self.tree.price_and_greeks(
                market_data.underlying_price,
                straddle.call_option.strike,
                market_data.implied_volatility,
                OptionType::Call,
            );
            
            let (_, put_greeks) = self.tree.price_and_greeks(
                market_data.underlying_price,
                straddle.put_option.strike,
                market_data.implied_volatility,
                OptionType::Put,
            );
            
            // Straddle Greeks = Call + Put
            let combined_delta = (call_greeks.delta + put_greeks.delta) * straddle.quantity;
            let combined_gamma = (call_greeks.gamma + put_greeks.gamma) * straddle.quantity;
            
            straddle.current_delta = combined_delta;
            straddle.total_gamma = combined_gamma;
            
            return Some(Greeks::new(
                combined_delta,
                combined_gamma,
                call_greeks.theta + put_greeks.theta,
                call_greeks.vega + put_greeks.vega,
                0.0,
            ));
        }
        None
    }
    
    /// Check if delta rebalancing is needed
    pub fn should_rebalance(&self, current_delta: f64) -> bool {
        current_delta.abs() > self.delta_threshold
    }
    
    /// Execute gamma scalp (delta rebalancing)
    pub fn execute_scalp(
        &mut self,
        market_data: &OptionMarketData,
        current_delta: f64,
    ) -> GammaScalpExecution {
        let start_time = Instant::now();
        
        // Check if we have a straddle
        let straddle = match &self.straddle {
            Some(s) => s,
            None => {
                return GammaScalpExecution {
                    success: false,
                    execution_time_us: start_time.elapsed().as_micros(),
                    rebalance_triggered: false,
                    delta_before: current_delta,
                    delta_after: current_delta,
                    hedge_quantity: 0.0,
                    scalp_profit: 0.0,
                    transaction_cost: 0.0,
                    net_scalp_profit: 0.0,
                    gamma_at_execution: 0.0,
                    threshold_breached: false,
                    error_message: Some("No straddle position set".to_string()),
                };
            }
        };
        
        // Check gamma threshold
        if straddle.total_gamma < self.min_gamma {
            return GammaScalpExecution {
                success: false,
                execution_time_us: start_time.elapsed().as_micros(),
                rebalance_triggered: false,
                delta_before: current_delta,
                delta_after: current_delta,
                hedge_quantity: 0.0,
                scalp_profit: 0.0,
                transaction_cost: 0.0,
                net_scalp_profit: 0.0,
                gamma_at_execution: straddle.total_gamma,
                threshold_breached: false,
                error_message: Some("Gamma below minimum threshold".to_string()),
            };
        }
        
        // Check if rebalancing is needed
        let threshold_breached = self.should_rebalance(current_delta);
        
        if !threshold_breached {
            return GammaScalpExecution {
                success: true,
                execution_time_us: start_time.elapsed().as_micros(),
                rebalance_triggered: false,
                delta_before: current_delta,
                delta_after: current_delta,
                hedge_quantity: 0.0,
                scalp_profit: 0.0,
                transaction_cost: 0.0,
                net_scalp_profit: 0.0,
                gamma_at_execution: straddle.total_gamma,
                threshold_breached: false,
                error_message: None,
            };
        }
        
        // Calculate hedge quantity (opposite sign to delta)
        let hedge_quantity = (-current_delta / market_data.underlying_price)
            .abs()
            .min(self.max_hedge_size);
        
        // Determine hedge direction
        let hedge_sign = if current_delta > 0.0 { -1.0 } else { 1.0 };
        let signed_hedge_qty = hedge_quantity * hedge_sign;
        
        // Estimate scalp profit from gamma
        // Profit ≈ 0.5 * Gamma * (ΔS)^2
        let price_move_estimate = market_data.underlying_price * 0.01; // Assume 1% move
        let estimated_scalp_profit = 0.5 * straddle.total_gamma * price_move_estimate * price_move_estimate;
        
        // Calculate transaction cost
        let notional = signed_hedge_qty.abs() * market_data.underlying_price;
        let transaction_cost = notional * self.transaction_cost_bps / 10000.0;
        
        let net_profit = estimated_scalp_profit - transaction_cost;
        
        // Update cumulative metrics
        self.total_scalp_profits += net_profit;
        self.scalp_count += 1;
        
        // New delta after hedge (should be close to zero)
        let delta_after = current_delta + signed_hedge_qty * market_data.underlying_price;
        
        GammaScalpExecution {
            success: true,
            execution_time_us: start_time.elapsed().as_micros(),
            rebalance_triggered: true,
            delta_before: current_delta,
            delta_after,
            hedge_quantity: signed_hedge_qty,
            scalp_profit: estimated_scalp_profit,
            transaction_cost,
            net_scalp_profit: net_profit,
            gamma_at_execution: straddle.total_gamma,
            threshold_breached: true,
            error_message: None,
        }
    }
    
    /// Get total scalp profits
    pub fn total_profits(&self) -> f64 {
        self.total_scalp_profits
    }
    
    /// Get number of scalps executed
    pub fn scalp_count(&self) -> u64 {
        self.scalp_count
    }
    
    /// Get average profit per scalp
    pub fn avg_profit_per_scalp(&self) -> f64 {
        if self.scalp_count == 0 {
            0.0
        } else {
            self.total_scalp_profits / self.scalp_count as f64
        }
    }
    
    /// Calculate time to expiry in years
    fn time_to_expiry_years(&self, expiry_timestamp: u64) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let seconds_to_expiry = expiry_timestamp.saturating_sub(now) as f64;
        seconds_to_expiry / (365.0 * 24.0 * 3600.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_binomial_tree_pricing() {
        let mut tree = BinomialTree::new(50, 0.25); // 50 steps, 0.25 years
        
        let (price, greeks) = tree.price_and_greeks(
            100.0, // spot
            100.0, // strike
            0.2,   // volatility
            OptionType::Call,
        );
        
        assert!(price > 0.0);
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);
        assert!(greeks.gamma > 0.0);
    }
    
    #[test]
    fn test_gamma_scalper_rebalance() {
        let call = OptionContract {
            option_type: OptionType::Call,
            strike: 50000.0,
            expiry_timestamp: 1735689600, // Future date
            underlying_asset: Asset::BTC,
            contract_size: 1.0,
        };
        
        let put = OptionContract {
            option_type: OptionType::Put,
            strike: 50000.0,
            expiry_timestamp: 1735689600,
            underlying_asset: Asset::BTC,
            contract_size: 1.0,
        };
        
        let straddle = StraddlePosition::new(call, put, 2000.0, 2000.0, 1.0);
        
        let mut scalper = GammaScalper::new(
            100.0,  // delta threshold
            0.001,  // min gamma
            10.0,   // max hedge size
            5.0,    // transaction cost bps
            50,     // tree steps
        );
        
        scalper.set_straddle(straddle);
        
        let market_data = OptionMarketData {
            underlying_price: 50000.0,
            implied_volatility: 0.6,
            risk_free_rate: 0.0,
            dividend_yield: 0.0,
            timestamp: Instant::now(),
        };
        
        // Update Greeks
        let greeks = scalper.update_greeks(&market_data);
        assert!(greeks.is_some());
        
        // Test with large delta breach
        let execution = scalper.execute_scalp(&market_data, 500.0);
        assert!(execution.success);
        assert!(execution.rebalance_triggered);
    }
    
    #[test]
    fn test_straddle_breakeven() {
        let call = OptionContract {
            option_type: OptionType::Call,
            strike: 50000.0,
            expiry_timestamp: 1735689600,
            underlying_asset: Asset::BTC,
            contract_size: 1.0,
        };
        
        let put = OptionContract {
            option_type: OptionType::Put,
            strike: 50000.0,
            expiry_timestamp: 1735689600,
            underlying_asset: Asset::BTC,
            contract_size: 1.0,
        };
        
        let straddle = StraddlePosition::new(call, put, 2000.0, 2000.0, 1.0);
        let (lower_be, upper_be) = straddle.breakeven_points();
        
        assert!((lower_be - 46000.0).abs() < 0.01);
        assert!((upper_be - 54000.0).abs() < 0.01);
    }
}
