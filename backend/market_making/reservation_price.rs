//! Reservation Price Calculator
//! Computes the indifference price based on current inventory risk.
//! Zero-cost abstractions for microsecond-level quote updates.
//!
//! The reservation price represents the price at which the market maker
//! is indifferent between holding the current inventory or being flat.
//! It serves as the center point around which bid/ask quotes are placed.
//!
//! Formula: r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
//! Where:
//!   s = mid-price
//!   q = inventory quantity
//!   gamma = risk aversion parameter
//!   sigma = volatility
//!   T - t = time horizon remaining

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Configuration for reservation price calculation
#[derive(Debug, Clone, Copy)]
pub struct ReservationPriceConfig {
    /// Risk aversion coefficient (gamma)
    pub gamma: f64,
    /// Default volatility estimate (sigma)
    pub default_volatility: f64,
    /// Time horizon in seconds (T)
    pub time_horizon: f64,
    /// Minimum tick size for price rounding
    pub min_tick: f64,
    /// Maximum inventory adjustment factor
    pub max_adjustment_factor: f64,
}

impl Default for ReservationPriceConfig {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            default_volatility: 0.0001, // ~1% daily vol
            time_horizon: 300.0, // 5 minutes
            min_tick: 0.01,
            max_adjustment_factor: 0.05, // Max 5% adjustment from mid
        }
    }
}

/// Thread-safe reservation price calculator
pub struct ReservationPriceCalculator {
    config: ReservationPriceConfig,
    /// Cached volatility per symbol (stored as u64 bits for atomic access)
    volatility_cache: parking_lot::RwLock<Vec<(String, f64)>>,
    /// Last calculation timestamp
    last_calculation: AtomicU64,
    /// Calculation latency tracking (nanoseconds)
    avg_latency_ns: AtomicU64,
    /// Validity flag
    is_valid: AtomicBool,
}

impl ReservationPriceCalculator {
    /// Create a new calculator with default configuration
    pub fn new() -> Self {
        Self::with_config(ReservationPriceConfig::default())
    }

    /// Create a new calculator with custom configuration
    pub fn with_config(config: ReservationPriceConfig) -> Self {
        Self {
            config,
            volatility_cache: parking_lot::RwLock::new(Vec::new()),
            last_calculation: AtomicU64::new(0),
            avg_latency_ns: AtomicU64::new(0),
            is_valid: AtomicBool::new(true),
        }
    }

    /// Update volatility estimate for a symbol (thread-safe)
    #[inline]
    pub fn update_volatility(&self, symbol: &str, volatility: f64) {
        let mut cache = self.volatility_cache.write();
        
        // Find existing entry or add new one
        if let Some(entry) = cache.iter_mut().find(|(s, _)| s == symbol) {
            entry.1 = volatility;
        } else {
            cache.push((symbol.to_string(), volatility));
        }
    }

    /// Get volatility for a symbol
    #[inline]
    fn get_volatility(&self, symbol: &str) -> f64 {
        let cache = self.volatility_cache.read();
        
        cache
            .iter()
            .find(|(s, _)| s == symbol)
            .map(|(_, v)| *v)
            .unwrap_or(self.config.default_volatility)
    }

    /// Calculate raw reservation price without rounding
    /// This is the core HJB-derived formula
    #[inline]
    pub fn calculate_raw(&self, mid_price: f64, inventory: f64, volatility: f64) -> f64 {
        // r = s - q * gamma * sigma^2 * T
        let adjustment = inventory * self.config.gamma * volatility.powi(2) * self.config.time_horizon;
        
        // Clamp adjustment to prevent extreme deviations
        let max_adjustment = mid_price * self.config.max_adjustment_factor;
        let clamped_adjustment = adjustment.clamp(-max_adjustment, max_adjustment);
        
        mid_price - clamped_adjustment
    }

    /// Calculate reservation price with proper rounding to tick size
    #[inline]
    pub fn calculate(&self, mid_price: f64, inventory: f64, symbol: &str) -> f64 {
        let start = Instant::now();
        
        let volatility = self.get_volatility(symbol);
        let raw_price = self.calculate_raw(mid_price, inventory, volatility);
        
        // Round to nearest tick
        let rounded_price = (raw_price / self.config.min_tick).round() * self.config.min_tick;
        
        // Update timing statistics
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.update_timing_stats(elapsed_ns);
        
        self.last_calculation.store(Instant::now().elapsed().as_nanos() as u64, Ordering::Relaxed);
        
        rounded_price
    }

    /// Calculate reservation price using cached volatility
    #[inline]
    pub fn calculate_with_cached_vol(&self, mid_price: f64, inventory: f64, symbol: &str) -> f64 {
        self.calculate(mid_price, inventory, symbol)
    }

    /// Update timing statistics (exponential moving average)
    #[inline]
    fn update_timing_stats(&self, new_latency_ns: u64) {
        let current_avg = self.avg_latency_ns.load(Ordering::Relaxed);
        // EMA with alpha = 0.1
        let new_avg = ((current_avg as f64 * 0.9) + (new_latency_ns as f64 * 0.1)) as u64;
        self.avg_latency_ns.store(new_avg, Ordering::Relaxed);
    }

    /// Get average calculation latency in nanoseconds
    #[inline]
    pub fn get_avg_latency_ns(&self) -> u64 {
        self.avg_latency_ns.load(Ordering::Relaxed)
    }

    /// Get the inventory adjustment component separately
    #[inline]
    pub fn calculate_inventory_adjustment(&self, inventory: f64, volatility: f64) -> f64 {
        inventory * self.config.gamma * volatility.powi(2) * self.config.time_horizon
    }

    /// Calculate symmetric spread around reservation price
    /// Returns (bid, ask) pair
    #[inline]
    pub fn calculate_symmetric_quotes(
        &self,
        mid_price: f64,
        inventory: f64,
        symbol: &str,
        half_spread: f64,
    ) -> (f64, f64) {
        let res_price = self.calculate(mid_price, inventory, symbol);
        
        let mut bid = res_price - half_spread;
        let mut ask = res_price + half_spread;
        
        // Ensure no crossed quotes
        if bid >= ask {
            let midpoint = (bid + ask) / 2.0;
            bid = midpoint - half_spread / 2.0;
            ask = midpoint + half_spread / 2.0;
        }
        
        // Round to ticks
        bid = (bid / self.config.min_tick).floor() * self.config.min_tick;
        ask = (ask / self.config.min_tick).ceil() * self.config.min_tick;
        
        (bid, ask)
    }

    /// Calculate asymmetric quotes based on inventory skew
    /// Wider spread on the side we want to reduce exposure
    #[inline]
    pub fn calculate_asymmetric_quotes(
        &self,
        mid_price: f64,
        inventory: f64,
        symbol: &str,
        base_half_spread: f64,
        skew_factor: f64,
    ) -> (f64, f64) {
        let res_price = self.calculate(mid_price, inventory, symbol);
        
        // Adjust spreads based on inventory direction
        // Positive inventory: wider ask (want to sell), tighter bid (don't want to buy more)
        // Negative inventory: wider bid (want to buy), tighter ask (don't want to sell more)
        let bid_spread = base_half_spread * (1.0 + skew_factor);
        let ask_spread = base_half_spread * (1.0 - skew_factor);
        
        let bid = res_price - bid_spread;
        let ask = res_price + ask_spread;
        
        // Round to ticks
        let bid = (bid / self.config.min_tick).floor() * self.config.min_tick;
        let ask = (ask / self.config.min_tick).ceil() * self.config.min_tick;
        
        (bid, ask)
    }

    /// Check if calculator is in valid state
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.is_valid.load(Ordering::Relaxed)
    }

    /// Invalidate calculator (e.g., during market halt)
    #[inline]
    pub fn invalidate(&self) {
        self.is_valid.store(false, Ordering::Relaxed);
    }

    /// Revalidate calculator
    #[inline]
    pub fn validate(&self) {
        self.is_valid.store(true, Ordering::Relaxed);
    }

    /// Get configuration
    #[inline]
    pub fn config(&self) -> &ReservationPriceConfig {
        &self.config
    }
}

impl Default for ReservationPriceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder pattern for constructing ReservationPriceCalculator
pub struct ReservationPriceBuilder {
    config: ReservationPriceConfig,
}

impl ReservationPriceBuilder {
    pub fn new() -> Self {
        Self {
            config: ReservationPriceConfig::default(),
        }
    }

    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    pub fn volatility(mut self, vol: f64) -> Self {
        self.config.default_volatility = vol;
        self
    }

    pub fn time_horizon(mut self, horizon: f64) -> Self {
        self.config.time_horizon = horizon;
        self
    }

    pub fn min_tick(mut self, tick: f64) -> Self {
        self.config.min_tick = tick;
        self
    }

    pub fn max_adjustment_factor(mut self, factor: f64) -> Self {
        self.config.max_adjustment_factor = factor;
        self
    }

    pub fn build(self) -> ReservationPriceCalculator {
        ReservationPriceCalculator::with_config(self.config)
    }
}

impl Default for ReservationPriceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_calculator() {
        let calc = ReservationPriceCalculator::new();
        assert!(calc.is_valid());
        
        let res_price = calc.calculate(50000.0, 0.0, "BTC");
        assert!((res_price - 50000.0).abs() < 0.02); // Should be very close to mid with zero inventory
    }

    #[test]
    fn test_inventory_adjustment() {
        let calc = ReservationPriceCalculator::new();
        
        // Positive inventory should lower reservation price
        let res_long = calc.calculate(50000.0, 1.0, "BTC");
        assert!(res_long < 50000.0);
        
        // Negative inventory should raise reservation price
        let res_short = calc.calculate(50000.0, -1.0, "BTC");
        assert!(res_short > 50000.0);
        
        // Long should be below short
        assert!(res_long < res_short);
    }

    #[test]
    fn test_symmetric_quotes() {
        let calc = ReservationPriceCalculator::new();
        
        let (bid, ask) = calc.calculate_symmetric_quotes(50000.0, 0.0, "BTC", 10.0);
        
        assert!(bid < 50000.0);
        assert!(ask > 50000.0);
        assert!(bid < ask);
        assert!((ask - bid - 20.0).abs() < 0.02); // Spread should be ~20
    }

    #[test]
    fn test_builder_pattern() {
        let calc = ReservationPriceBuilder::new()
            .gamma(0.2)
            .volatility(0.0002)
            .time_horizon(600.0)
            .min_tick(0.01)
            .build();
        
        assert_eq!(calc.config().gamma, 0.2);
        assert_eq!(calc.config().default_volatility, 0.0002);
    }

    #[test]
    fn test_latency_tracking() {
        let calc = ReservationPriceCalculator::new();
        
        // Perform several calculations
        for i in 0..10 {
            calc.calculate(50000.0, i as f64 * 0.1, "BTC");
        }
        
        let avg_latency = calc.get_avg_latency_ns();
        assert!(avg_latency > 0);
        assert!(avg_latency < 1_000_000); // Should be sub-millisecond
    }
}
