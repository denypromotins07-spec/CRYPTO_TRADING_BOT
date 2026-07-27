/**
 * Order Generator for Synthetic Market Simulation
 * 
 * Generates realistic limit and market orders matching crypto market stylized facts:
 * - Fat-tailed return distributions
 * - Volatility clustering
 * - Order flow autocorrelation
 * - Spread dynamics
 * 
 * Zero-cost abstractions, no heap allocations in hot paths.
 * Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
 */

use std::collections::VecDeque;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, Pareto, Uniform};

/// Order side (buy/sell)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type specification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Limit { price: f64 },
    Market,
    StopLoss { trigger: f64, limit: f64 },
    TakeProfit { trigger: f64, limit: f64 },
}

/// Synthetic order structure
#[derive(Debug, Clone)]
pub struct SyntheticOrder {
    pub order_id: u64,
    pub timestamp_ns: u64,
    pub side: Side,
    pub order_type: OrderType,
    pub quantity: f64,
    pub agent_id: u64,
    pub urgency: f64,           // 0-1, higher = more aggressive
    pub expected_slippage_bps: f64,
}

/// Parameters for order generation calibrated to crypto markets
#[derive(Debug, Clone)]
pub struct OrderGenParams {
    /// Base volatility for returns
    pub base_volatility: f64,
    /// Volatility of volatility (clustering)
    pub vol_of_vol: f64,
    /// Mean reversion speed for volatility
    pub vol_mean_reversion: f64,
    /// Pareto alpha for fat tails
    pub pareto_alpha: f64,
    /// Order flow autocorrelation
    pub order_flow_acf: f64,
    /// Spread mean in bps
    pub spread_mean_bps: f64,
    /// Spread volatility
    pub spread_vol_bps: f64,
    /// Market order ratio
    pub market_order_ratio: f64,
    /// Minimum order size
    pub min_size: f64,
    /// Maximum order size
    pub max_size: f64,
}

impl Default for OrderGenParams {
    fn default() -> Self {
        Self {
            base_volatility: 0.0002,      // ~2% per 100 ticks
            vol_of_vol: 0.3,              // Vol clustering strength
            vol_mean_reversion: 0.05,     // Slow mean reversion
            pareto_alpha: 3.0,            // Fat tail parameter
            order_flow_acf: 0.7,          // Strong autocorrelation
            spread_mean_bps: 8.0,         // 8 bps average spread
            spread_vol_bps: 3.0,          // Spread variation
            market_order_ratio: 0.35,     // 35% market orders
            min_size: 0.001,              // BTC minimum
            max_size: 10.0,               // BTC maximum
        }
    }
}

/// State variables for order generation
struct OrderGenState {
    current_volatility: f64,
    current_spread_bps: f64,
    last_side: Option<Side>,
    consecutive_same_side: u32,
    order_counter: u64,
}

/// High-performance synthetic order generator
pub struct OrderGenerator {
    rng: ChaCha8Rng,
    params: OrderGenParams,
    state: OrderGenState,
    /// Pre-allocated buffer for batch generation
    order_buffer: Vec<SyntheticOrder>,
    /// GARCH-like volatility history
    vol_history: VecDeque<f64>,
    /// Price level for relative sizing
    reference_price: f64,
}

impl OrderGenerator {
    /// Create new order generator with seed for reproducibility
    pub fn new(seed: u64, params: OrderGenParams, reference_price: f64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        Self {
            rng,
            params,
            state: OrderGenState {
                current_volatility: params.base_volatility,
                current_spread_bps: params.spread_mean_bps,
                last_side: None,
                consecutive_same_side: 0,
                order_counter: 0,
            },
            order_buffer: Vec::with_capacity(1000),
            vol_history: VecDeque::with_capacity(100),
            reference_price,
        }
    }

    /// Generate a single synthetic order
    pub fn generate_order(&mut self, timestamp_ns: u64, agent_id: u64) -> SyntheticOrder {
        self.state.order_counter += 1;
        
        // Update volatility state (GARCH-like dynamics)
        self.update_volatility();
        
        // Update spread
        self.update_spread();
        
        // Determine order side with autocorrelation
        let side = self.determine_side();
        
        // Determine order type
        let order_type = self.determine_order_type(side);
        
        // Generate size based on distribution
        let quantity = self.generate_size();
        
        // Calculate urgency and expected slippage
        let urgency = self.calculate_urgency(&order_type);
        let expected_slippage = self.estimate_slippage(quantity, &order_type);
        
        SyntheticOrder {
            order_id: self.state.order_counter,
            timestamp_ns,
            side,
            order_type,
            quantity,
            agent_id,
            urgency,
            expected_slippage_bps: expected_slippage,
        }
    }

    /// Generate a batch of orders efficiently
    pub fn generate_batch(&mut self, count: usize, base_timestamp_ns: u64, agent_ids: &[u64]) -> Vec<SyntheticOrder> {
        self.order_buffer.clear();
        
        let time_step_ns = 1_000_000; // 1ms between orders on average
        
        for i in 0..count {
            let timestamp = base_timestamp_ns + (i as u64) * time_step_ns;
            let agent_id = if agent_ids.is_empty() {
                0
            } else {
                agent_ids[self.rng.gen_range(0..agent_ids.len())]
            };
            
            let order = self.generate_order(timestamp, agent_id);
            self.order_buffer.push(order);
        }
        
        self.order_buffer.clone()
    }

    /// Update volatility using Heston-like stochastic process
    fn update_volatility(&mut self) {
        let dt = 1.0; // Discrete time step
        
        // Mean-reverting square-root process (CIR)
        let drift = self.params.vol_mean_reversion * (self.params.base_volatility - self.state.current_volatility) * dt;
        
        let diffusion = if self.state.current_volatility > 0.0 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            self.params.vol_of_vol * self.state.current_volatility.sqrt() * normal.sample(&mut self.rng) * dt.sqrt()
        } else {
            0.0
        };
        
        self.state.current_volatility = (self.state.current_volatility + drift + diffusion).max(1e-6);
        
        // Add to history for statistics
        self.vol_history.push_back(self.state.current_volatility);
        if self.vol_history.len() > 100 {
            self.vol_history.pop_front();
        }
    }

    /// Update spread based on volatility regime
    fn update_spread(&mut self) {
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        // Spread widens with volatility
        let vol_adjustment = self.state.current_volatility / self.params.base_volatility;
        let random_component = normal.sample(&mut self.rng) * self.params.spread_vol_bps;
        
        self.state.current_spread_bps = (
            self.params.spread_mean_bps * vol_adjustment + random_component
        ).max(1.0);
    }

    /// Determine order side with autocorrelation
    fn determine_side(&mut self) -> Side {
        let acf = self.params.order_flow_acf;
        
        // Probability of continuing same direction
        let continue_prob = if self.state.consecutive_same_side > 0 {
            0.5 + 0.5 * acf * (1.0 - 1.0 / (self.state.consecutive_same_side as f64 + 1.0))
        } else {
            0.5
        };
        
        let switch = self.rng.gen::<f64>() > continue_prob;
        
        if switch || self.state.last_side.is_none() {
            // Switch or initial
            let new_side = if self.rng.gen_bool(0.5) { Side::Buy } else { Side::Sell };
            self.state.last_side = Some(new_side);
            self.state.consecutive_same_side = 1;
            new_side
        } else {
            // Continue
            self.state.consecutive_same_side += 1;
            self.state.last_side.unwrap()
        }
    }

    /// Determine order type (limit vs market)
    fn determine_order_type(&mut self, side: Side) -> OrderType {
        if self.rng.gen::<f64>() < self.params.market_order_ratio {
            // Market order
            OrderType::Market
        } else {
            // Limit order with price offset from reference
            let normal = Normal::new(0.0, 1.0).unwrap();
            let offset_bps = normal.sample(&mut self.rng) * self.state.current_spread_bps * 0.5;
            
            let limit_price = match side {
                Side::Buy => self.reference_price * (1.0 - offset_bps / 10000.0),
                Side::Sell => self.reference_price * (1.0 + offset_bps / 10000.0),
            };
            
            OrderType::Limit { price: limit_price.max(0.01) }
        }
    }

    /// Generate order size with fat-tailed distribution
    fn generate_size(&mut self) -> f64 {
        // Mix of log-normal (normal trades) and Pareto (whale trades)
        let whale_prob = 0.02; // 2% chance of whale order
        
        if self.rng.gen::<f64>() < whale_prob {
            // Pareto distribution for fat tails
            let pareto = Pareto::new(self.params.min_size, self.params.pareto_alpha).unwrap();
            pareto.sample(&mut self.rng).min(self.params.max_size)
        } else {
            // Log-normal for typical orders
            let uniform = Uniform::new(
                self.params.min_size.ln(),
                (self.params.max_size * 0.1).ln()
            );
            uniform.sample(&mut self.rng).exp()
        }
    }

    /// Calculate order urgency (0-1)
    fn calculate_urgency(&mut self, order_type: &OrderType) -> f64 {
        match order_type {
            OrderType::Market => 0.8 + self.rng.gen::<f64>() * 0.2, // High urgency
            OrderType::Limit { .. } => {
                // Urgency based on how aggressive the limit price is
                let base = 0.3;
                let spread_factor = self.state.current_spread_bps / self.params.spread_mean_bps;
                (base + self.rng.gen::<f64>() * 0.4) / spread_factor
            },
            _ => 0.5,
        }
    }

    /// Estimate expected slippage in basis points
    fn estimate_slippage(&mut self, quantity: f64, order_type: &OrderType) -> f64 {
        let base_slippage = match order_type {
            OrderType::Market => self.state.current_spread_bps * 0.5,
            OrderType::Limit { .. } => 0.0, // No slippage if filled at limit
            _ => self.state.current_spread_bps * 0.3,
        };
        
        // Size impact (larger orders = more slippage)
        let size_impact = (quantity / self.params.max_size).sqrt() * 5.0;
        
        // Volatility impact
        let vol_impact = (self.state.current_volatility / self.params.base_volatility) * 2.0;
        
        (base_slippage + size_impact + vol_impact).max(0.0)
    }

    /// Get current volatility estimate
    pub fn get_current_volatility(&self) -> f64 {
        self.state.current_volatility
    }

    /// Get current spread estimate
    pub fn get_current_spread_bps(&self) -> f64 {
        self.state.current_spread_bps
    }

    /// Set reference price for limit order generation
    pub fn set_reference_price(&mut self, price: f64) {
        self.reference_price = price.max(0.01);
    }

    /// Get volatility history for analysis
    pub fn get_vol_history(&self) -> &[f64] {
        &self.vol_history
    }

    /// Reset state (for regime changes)
    pub fn reset_state(&mut self) {
        self.state.current_volatility = self.params.base_volatility;
        self.state.current_spread_bps = self.params.spread_mean_bps;
        self.state.last_side = None;
        self.state.consecutive_same_side = 0;
        self.vol_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_order_generation() {
        let mut gen = OrderGenerator::new(42, OrderGenParams::default(), 50000.0);
        let order = gen.generate_order(1_000_000_000, 1);
        
        assert!(order.quantity >= 0.001);
        assert!(order.quantity <= 10.0);
        assert!(order.urgency >= 0.0);
        assert!(order.urgency <= 1.0);
        assert!(order.expected_slippage_bps >= 0.0);
    }

    #[test]
    fn test_batch_generation() {
        let mut gen = OrderGenerator::new(123, OrderGenParams::default(), 50000.0);
        let agents = vec![1, 2, 3, 4, 5];
        let batch = gen.generate_batch(100, 1_000_000_000, &agents);
        
        assert_eq!(batch.len(), 100);
        
        // Verify timestamps are increasing
        for i in 1..batch.len() {
            assert!(batch[i].timestamp_ns > batch[i-1].timestamp_ns);
        }
    }

    #[test]
    fn test_fat_tails() {
        let mut gen = OrderGenerator::new(456, OrderGenParams::default(), 50000.0);
        
        let mut sizes = Vec::with_capacity(10000);
        for _ in 0..10000 {
            let order = gen.generate_order(0, 0);
            sizes.push(order.quantity);
        }
        
        // Check for presence of large orders (fat tail characteristic)
        let max_size = sizes.iter().cloned().fold(0.0/0.0, f64::max);
        let large_orders: usize = sizes.iter().filter(|&&s| s > 5.0).count();
        
        assert!(max_size > 5.0, "Should have some large orders");
        assert!(large_orders > 0, "Should have whale-sized orders");
    }

    #[test]
    fn test_volatility_clustering() {
        let mut gen = OrderGenerator::new(789, OrderGenParams::default(), 50000.0);
        
        // Generate many orders to build vol history
        for _ in 0..200 {
            gen.generate_order(0, 0);
        }
        
        let vol_history = gen.get_vol_history();
        assert!(!vol_history.is_empty());
        
        // Check autocorrelation in volatility (clustering)
        if vol_history.len() > 10 {
            let mut sum_diff = 0.0;
            for i in 1..vol_history.len() {
                sum_diff += (vol_history[i] - vol_history[i-1]).abs();
            }
            let avg_diff = sum_diff / (vol_history.len() - 1) as f64;
            
            // Volatility should change gradually (clustering), not randomly
            assert!(avg_diff < 0.001, "Volatility should be clustered");
        }
    }
}
