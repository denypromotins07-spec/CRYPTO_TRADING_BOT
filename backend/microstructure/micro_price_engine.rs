//! Micro-Price Engine
//! Computes the true fair value using L2 order book queue dynamics.
//! Zero-cost abstractions for nanosecond-level micro-price updates.
//!
//! The micro-price (or fair value) is calculated as a volume-weighted
//! midpoint that accounts for order book imbalance and queue positions.
//!
//! Formula: micro_price = bid + (ask - bid) * (bid_volume / (bid_volume + ask_volume))
//! Advanced: Incorporates queue decay factors and aggressive flow detection.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Represents a single price level in the order book
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    pub price: f64,
    pub volume: f64,
    pub order_count: u32,
}

/// Order book state for micro-price calculation
#[derive(Debug, Clone)]
pub struct OrderBookState {
    /// Best bid price level
    pub best_bid: PriceLevel,
    /// Best ask price level
    pub best_ask: PriceLevel,
    /// Second best bid (for depth analysis)
    pub bid_depth: Vec<PriceLevel>,
    /// Second best ask (for depth analysis)
    pub ask_depth: Vec<PriceLevel>,
    /// Timestamp of last update
    pub timestamp: u64,
}

impl OrderBookState {
    pub fn new(best_bid: PriceLevel, best_ask: PriceLevel) -> Self {
        Self {
            best_bid,
            best_ask,
            bid_depth: Vec::with_capacity(5),
            ask_depth: Vec::with_capacity(5),
            timestamp: 0,
        }
    }

    #[inline]
    pub fn mid_price(&self) -> f64 {
        (self.best_bid.price + self.best_ask.price) / 2.0
    }

    #[inline]
    pub fn spread(&self) -> f64 {
        self.best_ask.price - self.best_bid.price
    }

    #[inline]
    pub fn spread_bps(&self) -> f64 {
        if self.mid_price() == 0.0 {
            return 0.0;
        }
        (self.spread() / self.mid_price()) * 10000.0
    }
}

/// Configuration for micro-price calculation
#[derive(Debug, Clone, Copy)]
pub struct MicroPriceConfig {
    /// Weight given to bid volume in micro-price calculation (0.0 to 1.0)
    pub bid_weight: f64,
    /// Enable queue decay adjustment
    pub use_queue_decay: bool,
    /// Queue decay factor per millisecond
    pub decay_rate_ms: f64,
    /// Maximum age for order book data (milliseconds)
    pub max_age_ms: u64,
    /// Enable depth-adjusted micro-price
    pub use_depth_adjustment: bool,
    /// Number of levels to consider for depth adjustment
    pub depth_levels: usize,
}

impl Default for MicroPriceConfig {
    fn default() -> Self {
        Self {
            bid_weight: 0.5,
            use_queue_decay: true,
            decay_rate_ms: 0.001, // 0.1% per ms
            max_age_ms: 100, // 100ms max age
            use_depth_adjustment: true,
            depth_levels: 5,
        }
    }
}

/// Micro-price calculator with zero allocations in hot path
pub struct MicroPriceEngine {
    config: MicroPriceConfig,
    /// Cached micro-price value
    cached_micro_price: AtomicU64,
    /// Last update timestamp (nanoseconds since epoch)
    last_update_ns: AtomicU64,
    /// Cache validity flag
    is_cache_valid: AtomicBool,
    /// Calculation count for statistics
    calculation_count: AtomicU64,
    /// Average calculation latency (nanoseconds)
    avg_latency_ns: AtomicU64,
}

impl MicroPriceEngine {
    /// Create a new micro-price engine with default configuration
    pub fn new() -> Self {
        Self::with_config(MicroPriceConfig::default())
    }

    /// Create a new micro-price engine with custom configuration
    pub fn with_config(config: MicroPriceConfig) -> Self {
        Self {
            config,
            cached_micro_price: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            is_cache_valid: AtomicBool::new(false),
            calculation_count: AtomicU64::new(0),
            avg_latency_ns: AtomicU64::new(0),
        }
    }

    /// Calculate basic volume-weighted micro-price
    /// micro_price = bid + spread * (bid_vol / (bid_vol + ask_vol))
    #[inline]
    pub fn calculate_basic_micro_price(&self, book: &OrderBookState) -> f64 {
        let bid_vol = book.best_bid.volume;
        let ask_vol = book.best_ask.volume;
        
        if bid_vol + ask_vol == 0.0 {
            return book.mid_price();
        }

        // Volume-weighted position within spread
        let weight = bid_vol / (bid_vol + ask_vol);
        book.best_bid.price + book.spread() * weight
    }

    /// Calculate queue-decay adjusted volumes
    /// Older orders have reduced effective volume
    #[inline]
    fn apply_queue_decay(&self, volume: f64, age_ms: u64) -> f64 {
        if !self.config.use_queue_decay {
            return volume;
        }
        
        let decay_factor = (-self.config.decay_rate_ms * age_ms as f64).exp();
        volume * decay_factor.max(0.1) // Floor at 10% of original
    }

    /// Calculate depth-adjusted volume imbalance
    /// Considers multiple levels for more robust micro-price
    #[inline]
    fn calculate_depth_imbalance(&self, book: &OrderBookState) -> f64 {
        if !self.config.use_depth_adjustment || book.bid_depth.is_empty() {
            return book.best_bid.volume / (book.best_bid.volume + book.best_ask.volume);
        }

        let mut total_bid_vol = 0.0;
        let mut total_ask_vol = 0.0;
        
        // Include best bid/ask
        total_bid_vol += book.best_bid.volume;
        total_ask_vol += book.best_ask.volume;
        
        // Include additional depth levels with exponential decay weighting
        let levels = self.config.depth_levels.min(book.bid_depth.len().min(book.ask_depth.len()));
        for i in 0..levels {
            let weight = 1.0 / ((i + 1) as f64); // Exponential decay
            
            if i < book.bid_depth.len() {
                total_bid_vol += book.bid_depth[i].volume * weight;
            }
            if i < book.ask_depth.len() {
                total_ask_vol += book.ask_depth[i].volume * weight;
            }
        }

        if total_bid_vol + total_ask_vol == 0.0 {
            return 0.5;
        }

        total_bid_vol / (total_bid_vol + total_ask_vol)
    }

    /// Calculate advanced micro-price with all adjustments
    #[inline]
    pub fn calculate_micro_price(&self, book: &OrderBookState, current_time_ms: u64) -> f64 {
        let start = Instant::now();

        // Check if order book data is stale
        let age_ms = current_time_ms.saturating_sub(book.timestamp);
        if age_ms > self.config.max_age_ms {
            // Data too stale, fall back to mid-price
            self.update_cache_stats(start.elapsed().as_nanos(), false);
            return book.mid_price();
        }

        // Apply queue decay to volumes
        let effective_bid_vol = self.apply_queue_decay(book.best_bid.volume, age_ms);
        let effective_ask_vol = self.apply_queue_decay(book.best_ask.volume, age_ms);

        // Calculate base weight from best bid/ask
        let base_weight = if effective_bid_vol + effective_ask_vol == 0.0 {
            0.5
        } else {
            effective_bid_vol / (effective_bid_vol + effective_ask_vol)
        };

        // Blend with depth-adjusted weight
        let final_weight = if self.config.use_depth_adjustment {
            let depth_weight = self.calculate_depth_imbalance(book);
            // Weighted average of base and depth weights
            base_weight * 0.6 + depth_weight * 0.4
        } else {
            base_weight
        };

        let micro_price = book.best_bid.price + book.spread() * final_weight;

        // Update cache
        self.cached_micro_price.store(micro_price.to_bits(), Ordering::Relaxed);
        self.last_update_ns.store(current_time_ms * 1_000_000, Ordering::Relaxed);
        self.is_cache_valid.store(true, Ordering::Relaxed);

        self.update_cache_stats(start.elapsed().as_nanos(), true);

        micro_price
    }

    /// Get cached micro-price (fast path)
    #[inline]
    pub fn get_cached_micro_price(&self) -> Option<f64> {
        if self.is_cache_valid.load(Ordering::Relaxed) {
            let bits = self.cached_micro_price.load(Ordering::Relaxed);
            Some(f64::from_bits(bits))
        } else {
            None
        }
    }

    /// Update statistics after calculation
    #[inline]
    fn update_cache_stats(&self, latency_ns: u128, success: bool) {
        if success {
            let count = self.calculation_count.fetch_add(1, Ordering::Relaxed);
            let current_avg = self.avg_latency_ns.load(Ordering::Relaxed);
            
            // EMA update
            let new_avg = ((current_avg as f64 * 0.95) + (latency_ns as f64 * 0.05)) as u64;
            self.avg_latency_ns.store(new_avg, Ordering::Relaxed);
        }
    }

    /// Get calculation statistics
    #[inline]
    pub fn get_stats(&self) -> MicroPriceStats {
        MicroPriceStats {
            calculation_count: self.calculation_count.load(Ordering::Relaxed),
            avg_latency_ns: self.avg_latency_ns.load(Ordering::Relaxed),
            is_cache_valid: self.is_cache_valid.load(Ordering::Relaxed),
        }
    }

    /// Calculate order book imbalance ratio
    /// Returns value in [-1, 1]: positive = bid-heavy, negative = ask-heavy
    #[inline]
    pub fn calculate_imbalance(&self, book: &OrderBookState) -> f64 {
        let bid_vol = book.best_bid.volume;
        let ask_vol = book.best_ask.volume;
        
        if bid_vol + ask_vol == 0.0 {
            return 0.0;
        }

        (bid_vol - ask_vol) / (bid_vol + ask_vol)
    }

    /// Predict short-term price direction based on micro-price vs mid-price
    /// Returns: positive = upward pressure, negative = downward pressure
    #[inline]
    pub fn predict_direction(&self, book: &OrderBookState, current_time_ms: u64) -> f64 {
        let micro_price = self.calculate_micro_price(book, current_time_ms);
        let mid_price = book.mid_price();
        
        if mid_price == 0.0 {
            return 0.0;
        }

        // Normalized difference in basis points
        ((micro_price - mid_price) / mid_price) * 10000.0
    }
}

impl Default for MicroPriceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about micro-price calculations
#[derive(Debug, Clone, Copy)]
pub struct MicroPriceStats {
    pub calculation_count: u64,
    pub avg_latency_ns: u64,
    pub is_cache_valid: bool,
}

/// Builder pattern for MicroPriceEngine
pub struct MicroPriceBuilder {
    config: MicroPriceConfig,
}

impl MicroPriceBuilder {
    pub fn new() -> Self {
        Self {
            config: MicroPriceConfig::default(),
        }
    }

    pub fn bid_weight(mut self, weight: f64) -> Self {
        self.config.bid_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn use_queue_decay(mut self, enabled: bool) -> Self {
        self.config.use_queue_decay = enabled;
        self
    }

    pub fn decay_rate_ms(mut self, rate: f64) -> Self {
        self.config.decay_rate_ms = rate;
        self
    }

    pub fn max_age_ms(mut self, age: u64) -> Self {
        self.config.max_age_ms = age;
        self
    }

    pub fn use_depth_adjustment(mut self, enabled: bool) -> Self {
        self.config.use_depth_adjustment = enabled;
        self
    }

    pub fn depth_levels(mut self, levels: usize) -> Self {
        self.config.depth_levels = levels;
        self
    }

    pub fn build(self) -> MicroPriceEngine {
        MicroPriceEngine::with_config(self.config)
    }
}

impl Default for MicroPriceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_micro_price_balanced() {
        let engine = MicroPriceEngine::new();
        
        let book = OrderBookState::new(
            PriceLevel { price: 99.0, volume: 100.0, order_count: 5 },
            PriceLevel { price: 101.0, volume: 100.0, order_count: 5 },
        );

        let micro_price = engine.calculate_basic_micro_price(&book);
        
        // With equal volumes, micro-price should be at mid
        assert!((micro_price - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_basic_micro_price_imbalanced() {
        let engine = MicroPriceEngine::new();
        
        // More bid volume -> micro-price closer to ask (upward pressure)
        let book = OrderBookState::new(
            PriceLevel { price: 99.0, volume: 300.0, order_count: 10 },
            PriceLevel { price: 101.0, volume: 100.0, order_count: 5 },
        );

        let micro_price = engine.calculate_basic_micro_price(&book);
        
        // With 3:1 bid:ask ratio, micro-price should be 75% into spread
        // 99 + 2 * 0.75 = 100.5
        assert!((micro_price - 100.5).abs() < 0.001);
        assert!(micro_price > book.mid_price());
    }

    #[test]
    fn test_imbalance_calculation() {
        let engine = MicroPriceEngine::new();
        
        let balanced_book = OrderBookState::new(
            PriceLevel { price: 99.0, volume: 100.0, order_count: 5 },
            PriceLevel { price: 101.0, volume: 100.0, order_count: 5 },
        );
        assert!((engine.calculate_imbalance(&balanced_book)).abs() < 0.001);

        let bid_heavy_book = OrderBookState::new(
            PriceLevel { price: 99.0, volume: 300.0, order_count: 10 },
            PriceLevel { price: 101.0, volume: 100.0, order_count: 5 },
        );
        let imbalance = engine.calculate_imbalance(&bid_heavy_book);
        assert!(imbalance > 0.5); // Should be (300-100)/(300+100) = 0.5
    }

    #[test]
    fn test_stale_data_fallback() {
        let engine = MicroPriceEngine::new();
        
        let mut book = OrderBookState::new(
            PriceLevel { price: 99.0, volume: 100.0, order_count: 5 },
            PriceLevel { price: 101.0, volume: 100.0, order_count: 5 },
        );
        book.timestamp = 0; // Very old

        let current_time_ms = 1000; // 1 second later, beyond max_age_ms
        
        let micro_price = engine.calculate_micro_price(&book, current_time_ms);
        
        // Should fall back to mid-price for stale data
        assert!((micro_price - book.mid_price()).abs() < 0.001);
    }

    #[test]
    fn test_builder_pattern() {
        let engine = MicroPriceBuilder::new()
            .bid_weight(0.7)
            .use_queue_decay(false)
            .max_age_ms(50)
            .build();

        assert_eq!(engine.config.bid_weight, 0.7);
        assert!(!engine.config.use_queue_decay);
        assert_eq!(engine.config.max_age_ms, 50);
    }
}
