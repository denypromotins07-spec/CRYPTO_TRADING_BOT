//! Avellaneda-Stoikov Market Making Model
//! Solves the Hamilton-Jacobi-Bellman (HJB) equation for optimal quote placement.
//! Optimized for zero-cost abstractions and microsecond-level execution.
//! 
//! This implementation calculates optimal bid/ask spreads based on:
//! - Current inventory risk
//! - Volatility estimates
//! - Risk aversion parameter (gamma)
//! - Time to horizon (T)
//!
//! Target: 8k-20k INR/hour via spread capture on BTC, ETH, SOL, USDT pairs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Configuration parameters for the Avellaneda-Stoikov model
#[derive(Debug, Clone, Copy)]
pub struct ASConfig {
    /// Risk aversion parameter (gamma). Higher = tighter spreads, more aggressive mean reversion.
    pub gamma: f64,
    /// Volatility of the mid-price (sigma)
    pub volatility: f64,
    /// Time horizon in seconds (T)
    pub time_horizon: f64,
    /// Order book depth sensitivity (kappa)
    pub kappa: f64,
    /// Maximum inventory position allowed
    pub max_inventory: i64,
}

impl Default for ASConfig {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            volatility: 0.0001, // ~1% daily vol scaled to tick
            time_horizon: 300.0, // 5 minutes
            kappa: 1.0,
            max_inventory: 100,
        }
    }
}

/// Represents the current state of the market maker
#[derive(Debug, Clone, Copy)]
pub struct MarketMakerState {
    /// Current mid-price
    pub mid_price: f64,
    /// Current inventory (positive = long, negative = short)
    pub inventory: i64,
    /// Current timestamp
    pub timestamp: u64,
}

/// Optimal quote result
#[derive(Debug, Clone, Copy)]
pub struct OptimalQuotes {
    /// Optimal bid price
    pub bid: f64,
    /// Optimal ask price
    pub ask: f64,
    /// Reservation price (indifference price)
    pub reservation_price: f64,
    /// Spread width
    pub spread: f64,
}

/// Avellaneda-Stoikov solver using closed-form approximation
/// Avoids numerical PDE solving for microsecond performance
pub struct AvellanedaStoikovSolver {
    config: ASConfig,
    /// Cached volatility estimate (atomic for lock-free updates)
    cached_volatility: AtomicU64,
    /// Last update timestamp
    last_update: AtomicU64,
}

impl AvellanedaStoikovSolver {
    /// Create a new solver with given configuration
    pub fn new(config: ASConfig) -> Self {
        Self {
            config,
            cached_volatility: AtomicU64::new((config.volatility * 1e9) as u64),
            last_update: AtomicU64::new(0),
        }
    }

    /// Update volatility estimate (thread-safe, lock-free)
    #[inline]
    pub fn update_volatility(&self, new_vol: f64) {
        self.cached_volatility.store((new_vol * 1e9) as u64, Ordering::Relaxed);
        self.last_update.store(Instant::now().elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Get current volatility estimate
    #[inline]
    fn get_volatility(&self) -> f64 {
        self.cached_volatility.load(Ordering::Relaxed) as f64 / 1e9
    }

    /// Calculate reservation price (indifference price)
    /// r(s, t) = s - q * gamma * sigma^2 * (T - t)
    #[inline]
    pub fn calculate_reservation_price(&self, state: &MarketMakerState) -> f64 {
        let vol = self.get_volatility();
        let inventory_f64 = state.inventory as f64;
        
        // Avoid overflow in extreme inventory scenarios
        if inventory_f64.abs() > self.config.max_inventory as f64 {
            // Push reservation price aggressively to force liquidation
            let direction = if inventory_f64 > 0.0 { -1.0 } else { 1.0 };
            return state.mid_price + direction * self.config.gamma * vol.powi(2) * self.config.time_horizon * self.config.max_inventory as f64;
        }

        state.mid_price - inventory_f64 * self.config.gamma * vol.powi(2) * self.config.time_horizon
    }

    /// Calculate optimal spread width
    /// delta = (1/gamma) * ln(1 + gamma/kappa) + (inventory adjustment)
    #[inline]
    fn calculate_optimal_spread(&self, inventory: i64) -> f64 {
        let vol = self.get_volatility();
        let base_spread = (1.0 / self.config.gamma) * (1.0 + self.config.gamma / self.config.kappa).ln();
        
        // Inventory penalty: widen spread on the side we want to reduce exposure
        let inventory_adjustment = self.config.gamma * vol.powi(2) * self.config.time_horizon * inventory.abs() as f64;
        
        base_spread + inventory_adjustment
    }

    /// Calculate optimal bid and ask quotes
    /// Returns quotes that maximize expected utility under the HJB framework
    #[inline]
    pub fn calculate_optimal_quotes(&self, state: &MarketMakerState) -> OptimalQuotes {
        let reservation_price = self.calculate_reservation_price(state);
        let half_spread = self.calculate_optimal_spread(state.inventory) / 2.0;

        let mut bid = reservation_price - half_spread;
        let mut ask = reservation_price + half_spread;

        // Ensure bid < mid < ask to avoid crossed quotes
        if bid >= state.mid_price {
            bid = state.mid_price - 0.5 * (ask - bid);
        }
        if ask <= state.mid_price {
            ask = state.mid_price + 0.5 * (ask - bid);
        }

        // Enforce minimum tick size (exchange-specific, using generic 0.01 for now)
        const MIN_TICK: f64 = 0.01;
        bid = (bid / MIN_TICK).floor() * MIN_TICK;
        ask = (ask / MIN_TICK).ceil() * MIN_TICK;

        OptimalQuotes {
            bid,
            ask,
            reservation_price,
            spread: ask - bid,
        }
    }

    /// Check if inventory limit is breached
    #[inline]
    pub fn is_inventory_breached(&self, inventory: i64) -> bool {
        inventory.abs() > self.config.max_inventory
    }

    /// Get recommended action when inventory is breached
    #[inline]
    pub fn get_liquidation_quote(&self, state: &MarketMakerState) -> OptimalQuotes {
        let vol = self.get_volatility();
        let aggressive_factor = 2.0; // Aggressively price to liquidate
        
        let (bid, ask) = if state.inventory > 0 {
            // Long inventory: aggressive ask to sell
            (state.mid_price * 0.999, state.mid_price * (1.0 - aggressive_factor * vol))
        } else {
            // Short inventory: aggressive bid to buy
            (state.mid_price * (1.0 + aggressive_factor * vol), state.mid_price * 1.001)
        };

        OptimalQuotes {
            bid,
            ask,
            reservation_price: state.mid_price,
            spread: ask - bid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_initialization() {
        let config = ASConfig::default();
        let solver = AvellanedaStoikovSolver::new(config);
        assert!(!solver.is_inventory_breached(0));
    }

    #[test]
    fn test_reservation_price_calculation() {
        let config = ASConfig::default();
        let solver = AvellanedaStoikovSolver::new(config);
        
        let state = MarketMakerState {
            mid_price: 50000.0,
            inventory: 10,
            timestamp: 0,
        };

        let res_price = solver.calculate_reservation_price(&state);
        // With positive inventory, reservation price should be below mid-price
        assert!(res_price < state.mid_price);
    }

    #[test]
    fn test_optimal_quotes_no_crossing() {
        let config = ASConfig::default();
        let solver = AvellanedaStoikovSolver::new(config);
        
        let state = MarketMakerState {
            mid_price: 50000.0,
            inventory: 0,
            timestamp: 0,
        };

        let quotes = solver.calculate_optimal_quotes(&state);
        assert!(quotes.bid < state.mid_price);
        assert!(quotes.ask > state.mid_price);
        assert!(quotes.bid < quotes.ask);
    }

    #[test]
    fn test_inventory_breach_detection() {
        let mut config = ASConfig::default();
        config.max_inventory = 50;
        let solver = AvellanedaStoikovSolver::new(config);
        
        assert!(!solver.is_inventory_breached(49));
        assert!(solver.is_inventory_breached(51));
        assert!(solver.is_inventory_breached(-51));
    }
}
