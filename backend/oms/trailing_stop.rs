//! ZAID PERSONAL CRYPTO TRADING BOT - Stage 32
//! Advanced Order Management System (OMS) - Chapter 2
//! File: backend/oms/trailing_stop.rs
//!
//! Calculates real-time trailing distances based on ATR and volatility.
//! Prevents trailing stops from being whipsawed by microsecond bid-ask spread noise.
//! Optimized for AMD Ryzen AI 5, strictly respecting 8GB RAM limit.
//! Targets 8k-20k INR/hour in a 4hr trading window.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;

/// Type of trailing stop calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrailingMethod {
    /// Fixed percentage distance
    FixedPercentage,
    /// Based on Average True Range (ATR)
    AtrBased,
    /// Based on standard deviation of returns
    VolatilityBased,
    /// Step-trailing that only moves at fixed intervals
    StepTrailing,
    /// Hybrid approach combining multiple methods
    Hybrid,
}

/// Configuration for trailing stop calculation
#[derive(Debug, Clone)]
pub struct TrailingConfig {
    /// Method for calculating trailing distance
    pub method: TrailingMethod,
    /// Fixed percentage (for FixedPercentage method)
    pub fixed_pct: f64,
    /// ATR multiplier (for AtrBased method)
    pub atr_multiplier: f64,
    /// ATR period for calculation
    pub atr_period: usize,
    /// Volatility multiplier (for VolatilityBased method)
    pub vol_multiplier: f64,
    /// Volatility lookback period
    pub vol_period: usize,
    /// Minimum tick size for the symbol
    pub min_tick_size: f64,
    /// Minimum distance to prevent whipsaws
    pub min_distance_pct: f64,
    /// Maximum distance cap
    pub max_distance_pct: f64,
    /// Step size for step-trailing (percentage)
    pub step_size_pct: f64,
}

impl Default for TrailingConfig {
    fn default() -> Self {
        TrailingConfig {
            method: TrailingMethod::AtrBased,
            fixed_pct: 0.02, // 2%
            atr_multiplier: 2.0,
            atr_period: 14,
            vol_multiplier: 2.5,
            vol_period: 20,
            min_tick_size: 0.01,
            min_distance_pct: 0.005, // 0.5% minimum
            max_distance_pct: 0.10,  // 10% maximum
            step_size_pct: 0.01,     // 1% steps
        }
    }
}

/// Candle data for ATR and volatility calculations
#[derive(Debug, Clone, Copy)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    /// Calculate True Range for this candle given previous close
    #[inline]
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let high_low = self.high - self.low;
        let high_prev = (self.high - prev_close).abs();
        let low_prev = (self.low - prev_close).abs();
        
        high_low.max(high_prev).max(low_prev)
    }
    
    /// Calculate return for this candle
    #[inline]
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 {
            return 0.0;
        }
        (self.close - self.open) / self.open * 100.0
    }
}

/// Trailing stop calculator with ATR and volatility support
pub struct TrailingStopCalculator {
    /// Configuration
    config: TrailingConfig,
    /// Rolling window of candles for ATR calculation
    candles: VecDeque<Candle>,
    /// Current ATR value
    current_atr: f64,
    /// Current volatility (std dev of returns)
    current_volatility: f64,
    /// Last calculated trailing distance
    last_distance: f64,
    /// Last price that triggered a trailing update
    last_trigger_price: f64,
    /// For step-trailing: accumulated move since last step
    step_accumulator: f64,
    /// Symbol this calculator is for
    symbol: String,
}

impl TrailingStopCalculator {
    /// Create a new trailing stop calculator
    pub fn new(symbol: String, config: TrailingConfig) -> Self {
        let max_period = config.atr_period.max(config.vol_period) + 10;
        
        TrailingStopCalculator {
            config,
            candles: VecDeque::with_capacity(max_period),
            current_atr: 0.0,
            current_volatility: 0.0,
            last_distance: 0.0,
            last_trigger_price: 0.0,
            step_accumulator: 0.0,
            symbol,
        }
    }

    /// Add a new candle and update indicators
    pub fn add_candle(&mut self, candle: Candle) {
        self.candles.push_back(candle);
        
        // Maintain window size
        let max_len = self.config.atr_period.max(self.config.vol_period) + 10;
        while self.candles.len() > max_len {
            self.candles.pop_front();
        }
        
        // Update indicators
        self.update_atr();
        self.update_volatility();
    }

    /// Update ATR using Wilder's smoothing method
    fn update_atr(&mut self) {
        if self.candles.len() < 2 {
            return;
        }

        let period = self.config.atr_period.min(self.candles.len() - 1);
        
        // Calculate initial ATR as simple average of first 'period' true ranges
        if self.candles.len() == period + 1 && self.current_atr == 0.0 {
            let mut sum_tr = 0.0;
            for i in 1..=period {
                let prev_close = self.candles[i - 1].close;
                sum_tr += self.candles[i].true_range(prev_close);
            }
            self.current_atr = sum_tr / period as f64;
            return;
        }

        // Wilder's smoothing: ATR = (prev_ATR * (n-1) + current_TR) / n
        if let Some(last) = self.candles.back() {
            if let Some(second_last) = self.candles.iter().rev().nth(1) {
                let current_tr = last.true_range(second_last.close);
                self.current_atr = (self.current_atr * (period - 1) as f64 + current_tr) 
                    / period as f64;
            }
        }
    }

    /// Update volatility (standard deviation of returns)
    fn update_volatility(&mut self) {
        if self.candles.len() < 2 {
            return;
        }

        let period = self.config.vol_period.min(self.candles.len());
        
        // Calculate returns
        let returns: Vec<f64> = self.candles.iter()
            .skip(self.candles.len() - period)
            .map(|c| c.return_pct())
            .collect();

        if returns.is_empty() {
            return;
        }

        // Calculate mean
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

        // Calculate variance
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        // Standard deviation
        self.current_volatility = variance.sqrt();
    }

    /// Calculate trailing distance based on configured method
    #[inline]
    pub fn calculate_distance(&mut self, current_price: f64) -> f64 {
        let distance = match self.config.method {
            TrailingMethod::FixedPercentage => {
                current_price * self.config.fixed_pct
            }
            TrailingMethod::AtrBased => {
                self.current_atr * self.config.atr_multiplier
            }
            TrailingMethod::VolatilityBased => {
                current_price * (self.current_volatility / 100.0) * self.config.vol_multiplier
            }
            TrailingMethod::StepTrailing => {
                // Step trailing handled separately
                self.calculate_step_distance(current_price)
            }
            TrailingMethod::Hybrid => {
                // Combine ATR and volatility
                let atr_dist = self.current_atr * self.config.atr_multiplier;
                let vol_dist = current_price * (self.current_volatility / 100.0) * self.config.vol_multiplier;
                atr_dist.max(vol_dist)
            }
        };

        // Apply constraints
        let constrained = self.apply_constraints(distance, current_price);
        
        // Round to tick size
        self.last_distance = self.round_to_tick(constrained);
        self.last_distance
    }

    /// Calculate step-trailing distance
    fn calculate_step_distance(&mut self, current_price: f64) -> f64 {
        if self.last_trigger_price == 0.0 {
            self.last_trigger_price = current_price;
            return current_price * self.config.step_size_pct;
        }

        let price_move = (current_price - self.last_trigger_price).abs();
        let step_threshold = self.last_trigger_price * self.config.step_size_pct;

        if price_move >= step_threshold {
            // Move to next step
            let steps_moved = (price_move / step_threshold) as u64;
            self.step_accumulator += steps_moved as f64 * self.config.step_size_pct;
            self.last_trigger_price = current_price;
        }

        current_price * (self.config.step_size_pct + self.step_accumulator * self.config.step_size_pct)
    }

    /// Apply min/max constraints to distance
    fn apply_constraints(&self, distance: f64, price: f64) -> f64 {
        let min_dist = price * self.config.min_distance_pct;
        let max_dist = price * self.config.max_distance_pct;

        distance.clamp(min_dist, max_dist)
    }

    /// Round distance to tick size
    fn round_to_tick(&self, distance: f64) -> f64 {
        let tick = self.config.min_tick_size;
        (distance / tick).round() * tick
    }

    /// Calculate trailing stop price for a long position
    #[inline]
    pub fn calculate_long_stop(&mut self, entry_price: f64, current_price: f64) -> f64 {
        let distance = self.calculate_distance(current_price);
        
        // Trailing stop only moves up for long positions
        let stop_price = current_price - distance;
        
        // Stop should never be below entry for a profitable trade protection
        // But we allow it initially for normal stop behavior
        stop_price.max(entry_price - distance)
    }

    /// Calculate trailing stop price for a short position
    #[inline]
    pub fn calculate_short_stop(&mut self, entry_price: f64, current_price: f64) -> f64 {
        let distance = self.calculate_distance(current_price);
        
        // Trailing stop only moves down for short positions
        let stop_price = current_price + distance;
        
        stop_price.min(entry_price + distance)
    }

    /// Get current ATR value
    #[inline]
    pub fn get_atr(&self) -> f64 {
        self.current_atr
    }

    /// Get current volatility
    #[inline]
    pub fn get_volatility(&self) -> f64 {
        self.current_volatility
    }

    /// Check if enough data is available for reliable calculation
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.candles.len() >= self.config.atr_period.min(self.config.vol_period)
    }

    /// Reset accumulator for step-trailing (call when position is closed)
    pub fn reset(&mut self) {
        self.step_accumulator = 0.0;
        self.last_trigger_price = 0.0;
        self.last_distance = 0.0;
    }
}

/// Active trailing stop manager for a position
pub struct ActiveTrailingStop {
    /// Position side: true for long, false for short
    is_long: bool,
    /// Entry price
    entry_price: f64,
    /// Quantity
    quantity: f64,
    /// Current stop price
    stop_price: AtomicU64, // Stored as fixed-point for atomicity
    /// Highest price seen (for long) or lowest (for short)
    extreme_price: AtomicU64,
    /// Whether stop has been triggered
    triggered: AtomicBool,
    /// Calculator reference
    calculator: Arc<std::sync::Mutex<TrailingStopCalculator>>,
    /// Symbol
    symbol: String,
    /// Created timestamp
    created_at: u64,
}

impl ActiveTrailingStop {
    /// Create a new active trailing stop
    pub fn new(
        is_long: bool,
        entry_price: f64,
        quantity: f64,
        calculator: Arc<std::sync::Mutex<TrailingStopCalculator>>,
        symbol: String,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let stop_fp = (entry_price * 0.95 * 1e8) as u64; // Initial 5% stop

        ActiveTrailingStop {
            is_long,
            entry_price,
            quantity,
            stop_price: AtomicU64::new(stop_fp),
            extreme_price: AtomicU64::new((entry_price * 1e8) as u64),
            triggered: AtomicBool::new(false),
            calculator,
            symbol,
            created_at: now,
        }
    }

    /// Update stop price based on new market price
    pub fn update(&self, current_price: f64) -> Option<f64> {
        if self.triggered.load(Ordering::Acquire) {
            return None;
        }

        let mut calc = self.calculator.lock().unwrap();
        let new_stop = if self.is_long {
            calc.calculate_long_stop(self.entry_price, current_price)
        } else {
            calc.calculate_short_stop(self.entry_price, current_price)
        };

        // Update extreme price
        let current_fp = (current_price * 1e8) as u64;
        let extreme_fp = self.extreme_price.load(Ordering::Acquire);

        if self.is_long {
            if current_fp > extreme_fp {
                self.extreme_price.store(current_fp, Ordering::Release);
            }
        } else {
            if current_fp < extreme_fp || extreme_fp == 0 {
                self.extreme_price.store(current_fp, Ordering::Release);
            }
        }

        // Check if stop moved
        let old_stop_fp = self.stop_price.load(Ordering::Acquire);
        let new_stop_fp = (new_stop * 1e8) as u64;

        let should_update = if self.is_long {
            new_stop_fp > old_stop_fp
        } else {
            new_stop_fp < old_stop_fp && new_stop_fp > 0
        };

        if should_update {
            self.stop_price.store(new_stop_fp, Ordering::Release);
            Some(new_stop)
        } else {
            None
        }
    }

    /// Check if stop is triggered
    pub fn check_trigger(&self, current_price: f64) -> bool {
        let stop_fp = self.stop_price.load(Ordering::Acquire);
        let stop_price = stop_fp as f64 / 1e8;

        let triggered = if self.is_long {
            current_price <= stop_price
        } else {
            current_price >= stop_price
        };

        if triggered {
            self.triggered.store(true, Ordering::Release);
        }

        triggered
    }

    /// Get current stop price
    #[inline]
    pub fn get_stop_price(&self) -> f64 {
        self.stop_price.load(Ordering::Acquire) as f64 / 1e8
    }

    /// Get unrealized PnL based on current stop
    pub fn get_unrealized_pnl(&self, current_price: f64) -> f64 {
        let stop_price = self.get_stop_price();
        
        if self.is_long {
            (stop_price - self.entry_price) * self.quantity
        } else {
            (self.entry_price - stop_price) * self.quantity
        }
    }

    /// Cancel the trailing stop
    pub fn cancel(&self) {
        self.triggered.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atr_calculation() {
        let config = TrailingConfig {
            method: TrailingMethod::AtrBased,
            atr_period: 5,
            ..Default::default()
        };

        let mut calc = TrailingStopCalculator::new("BTCUSDT".to_string(), config);

        // Add sample candles
        for i in 0..10 {
            let candle = Candle {
                timestamp: i as u64 * 60000,
                open: 50000.0 + i as f64 * 100.0,
                high: 50000.0 + i as f64 * 100.0 + 50.0,
                low: 50000.0 + i as f64 * 100.0 - 50.0,
                close: 50000.0 + i as f64 * 100.0 + 25.0,
                volume: 1.0,
            };
            calc.add_candle(candle);
        }

        assert!(calc.is_ready());
        assert!(calc.get_atr() > 0.0);
    }

    #[test]
    fn test_trailing_stop_long() {
        let config = TrailingConfig {
            method: TrailingMethod::FixedPercentage,
            fixed_pct: 0.02,
            ..Default::default()
        };

        let mut calc = TrailingStopCalculator::new("ETHUSDT".to_string(), config);

        let entry = 3000.0;
        let current = 3100.0;

        let stop = calc.calculate_long_stop(entry, current);
        
        // Stop should be 2% below current price
        assert!((stop - (current * 0.98)).abs() < 1.0);
        assert!(stop < current);
    }

    #[test]
    fn test_whipsaw_prevention() {
        let config = TrailingConfig {
            method: TrailingMethod::AtrBased,
            min_distance_pct: 0.005, // 0.5% minimum
            atr_multiplier: 2.0,
            atr_period: 14,
            ..Default::default()
        };

        let mut calc = TrailingStopCalculator::new("SOLUSDT".to_string(), config);

        // With very small ATR, minimum distance should apply
        for _ in 0..20 {
            calc.add_candle(Candle {
                timestamp: 0,
                open: 100.0,
                high: 100.01,
                low: 99.99,
                close: 100.0,
                volume: 1.0,
            });
        }

        let distance = calc.calculate_distance(100.0);
        let min_expected = 100.0 * 0.005;

        assert!(distance >= min_expected, "Distance {} should be >= {}", distance, min_expected);
    }
}
