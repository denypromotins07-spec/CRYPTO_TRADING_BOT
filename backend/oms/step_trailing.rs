//! ZAID PERSONAL CRYPTO TRADING BOT - Stage 32
//! Advanced Order Management System (OMS) - Chapter 2
//! File: backend/oms/step_trailing.rs
//!
//! Implements step-trailing that only moves stops at fixed intervals.
//! Prevents whipsaw by requiring price to move a full step before trailing.
//! Optimized for AMD Ryzen AI 5, strictly respecting 8GB RAM limit.
//! Targets 8k-20k INR/hour in a 4hr trading window.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for step-trailing stop
#[derive(Debug, Clone)]
pub struct StepTrailingConfig {
    /// Initial stop distance as percentage of entry price
    pub initial_distance_pct: f64,
    /// Step size: price must move this much before stop trails
    pub step_size_pct: f64,
    /// Minimum number of steps before first trail (prevents premature trailing)
    pub min_steps_before_trail: u32,
    /// Maximum steps allowed (caps the trailing distance)
    pub max_steps: u32,
    /// Minimum tick size for the symbol
    pub min_tick_size: f64,
    /// Whether to break even after first step
    pub break_even_after_first_step: bool,
}

impl Default for StepTrailingConfig {
    fn default() -> Self {
        StepTrailingConfig {
            initial_distance_pct: 0.02, // 2% initial stop
            step_size_pct: 0.01,        // 1% steps
            min_steps_before_trail: 1,
            max_steps: 50,              // Cap at 50% trailing
            min_tick_size: 0.01,
            break_even_after_first_step: true,
        }
    }
}

/// State of a step-trailing stop
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepTrailingState {
    /// Waiting for first step to be reached
    WaitingFirstStep,
    /// Actively trailing with N steps accumulated
    Trailing { steps: u32 },
    /// Stop has been triggered
    Triggered,
    /// Cancelled manually
    Cancelled,
}

/// Step-trailing stop manager for a single position
pub struct StepTrailingStop {
    /// Configuration
    config: StepTrailingConfig,
    /// Position side: true for long, false for short
    is_long: bool,
    /// Entry price
    entry_price: f64,
    /// Quantity
    quantity: f64,
    /// Current stop price (stored as fixed-point for atomicity)
    stop_price_fp: AtomicU64,
    /// Extreme price reached (highest for long, lowest for short)
    extreme_price_fp: AtomicU64,
    /// Number of steps accumulated
    steps_accumulated: AtomicU64,
    /// Last price at which we checked for a step
    last_check_price_fp: AtomicU64,
    /// Current state
    state: AtomicU64, // Encoded StepTrailingState
    /// Symbol
    symbol: String,
    /// Created timestamp
    created_at: u64,
    /// Last update timestamp
    updated_at: AtomicU64,
}

impl StepTrailingStop {
    /// Create a new step-trailing stop
    pub fn new(
        is_long: bool,
        entry_price: f64,
        quantity: f64,
        config: StepTrailingConfig,
        symbol: String,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Calculate initial stop price
        let initial_stop = if is_long {
            entry_price * (1.0 - config.initial_distance_pct)
        } else {
            entry_price * (1.0 + config.initial_distance_pct)
        };

        let initial_stop_fp = (initial_stop / config.min_tick_size).round() as u64;
        let entry_fp = (entry_price / config.min_tick_size).round() as u64;

        StepTrailingStop {
            config,
            is_long,
            entry_price,
            quantity,
            stop_price_fp: AtomicU64::new(initial_stop_fp),
            extreme_price_fp: AtomicU64::new(entry_fp),
            steps_accumulated: AtomicU64::new(0),
            last_check_price_fp: AtomicU64::new(entry_fp),
            state: AtomicU64::new(StepTrailingState::WaitingFirstStep as u64),
            symbol,
            created_at: now,
            updated_at: AtomicU64::new(now),
        }
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> StepTrailingState {
        unsafe { std::mem::transmute(self.state.load(Ordering::Acquire)) }
    }

    /// Update stop based on new market price
    /// Returns Some(new_stop_price) if stop moved, None otherwise
    pub fn update(&self, current_price: f64) -> Option<f64> {
        let state = self.get_state();
        if matches!(state, StepTrailingState::Triggered | StepTrailingState::Cancelled) {
            return None;
        }

        let tick = self.config.min_tick_size;
        let current_fp = (current_price / tick).round() as u64;
        
        // Update extreme price
        self.update_extreme_price(current_fp);

        // Check if we've moved enough for a new step
        let last_check_fp = self.last_check_price_fp.load(Ordering::Acquire);
        let step_threshold_fp = 
            ((self.entry_price * self.config.step_size_pct) / tick).round() as u64;

        let price_move_fp = if self.is_long {
            current_fp.saturating_sub(last_check_fp)
        } else {
            last_check_fp.saturating_sub(current_fp)
        };

        // Check if we've moved a full step
        if price_move_fp >= step_threshold_fp {
            // Calculate how many full steps we've moved
            let new_steps = price_move_fp / step_threshold_fp;
            
            if new_steps > 0 {
                // Update last check price
                let new_check_fp = if self.is_long {
                    last_check_fp + new_steps * step_threshold_fp
                } else {
                    last_check_fp.saturating_sub(new_steps * step_threshold_fp)
                };
                
                self.last_check_price_fp.store(new_check_fp, Ordering::Release);

                // Accumulate steps
                let old_steps = self.steps_accumulated.fetch_add(
                    new_steps,
                    Ordering::AcqRel,
                );

                let total_steps = old_steps + new_steps;
                
                // Cap at max steps
                let capped_steps = total_steps.min(self.config.max_steps as u64);
                if total_steps != capped_steps {
                    // Adjust for cap (simplified - would need more complex logic)
                    self.steps_accumulated.store(capped_steps, Ordering::Release);
                }

                // Calculate new stop price
                let new_stop_fp = self.calculate_stop_from_steps(capped_steps);
                
                // Only update if stop improves (moves in favorable direction)
                let should_update = if self.is_long {
                    new_stop_fp > self.stop_price_fp.load(Ordering::Acquire)
                } else {
                    let current_stop = self.stop_price_fp.load(Ordering::Acquire);
                    new_stop_fp < current_stop && new_stop_fp > 0
                };

                if should_update {
                    self.stop_price_fp.store(new_stop_fp, Ordering::Release);
                    self.updated_at.store(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64,
                        Ordering::Release,
                    );

                    // Update state
                    let final_steps = self.steps_accumulated.load(Ordering::Acquire) as u32;
                    self.state.store(
                        StepTrailingState::Trailing { steps: final_steps } as u64,
                        Ordering::Release,
                    );

                    return Some(new_stop_fp as f64 * tick);
                }
            }
        }

        None
    }

    /// Update extreme price atomically
    fn update_extreme_price(&self, current_fp: u64) {
        let extreme_fp = self.extreme_price_fp.load(Ordering::Acquire);

        let should_update = if self.is_long {
            current_fp > extreme_fp
        } else {
            current_fp < extreme_fp || extreme_fp == 0
        };

        if should_update {
            self.extreme_price_fp.store(current_fp, Ordering::Release);
        }
    }

    /// Calculate stop price from accumulated steps
    fn calculate_stop_from_steps(&self, steps: u64) -> u64 {
        let tick = self.config.min_tick_size;
        let entry_fp = (self.entry_price / tick).round() as u64;
        let step_fp = ((self.entry_price * self.config.step_size_pct) / tick).round() as u64;

        if self.is_long {
            // For long: stop moves up with each step
            let initial_distance_fp = 
                ((self.entry_price * self.config.initial_distance_pct) / tick).round() as u64;
            
            // After N steps, stop is N*step_size below extreme
            // But we track from entry: entry - initial + steps*step
            if steps == 0 {
                entry_fp.saturating_sub(initial_distance_fp)
            } else if self.config.break_even_after_first_step && steps >= 1 {
                // First step brings stop to break-even
                let breakeven_fp = entry_fp;
                let additional_steps = steps.saturating_sub(1);
                breakeven_fp + additional_steps * step_fp
            } else {
                entry_fp.saturating_sub(initial_distance_fp) + steps * step_fp
            }
        } else {
            // For short: stop moves down with each step
            let initial_distance_fp = 
                ((self.entry_price * self.config.initial_distance_pct) / tick).round() as u64;
            
            if steps == 0 {
                entry_fp + initial_distance_fp
            } else if self.config.break_even_after_first_step && steps >= 1 {
                let breakeven_fp = entry_fp;
                let additional_steps = steps.saturating_sub(1);
                breakeven_fp.saturating_sub(additional_steps * step_fp)
            } else {
                entry_fp + initial_distance_fp - steps * step_fp
            }
        }
    }

    /// Check if stop is triggered by current price
    pub fn check_trigger(&self, current_price: f64) -> bool {
        let stop_fp = self.stop_price_fp.load(Ordering::Acquire);
        let tick = self.config.min_tick_size;
        let stop_price = stop_fp as f64 * tick;
        let current_fp = (current_price / tick).round() as u64;

        let triggered = if self.is_long {
            current_fp <= stop_fp
        } else {
            current_fp >= stop_fp && stop_fp > 0
        };

        if triggered {
            self.state.store(StepTrailingState::Triggered as u64, Ordering::Release);
        }

        triggered
    }

    /// Get current stop price
    #[inline]
    pub fn get_stop_price(&self) -> f64 {
        let fp = self.stop_price_fp.load(Ordering::Acquire);
        fp as f64 * self.config.min_tick_size
    }

    /// Get number of accumulated steps
    #[inline]
    pub fn get_steps(&self) -> u32 {
        self.steps_accumulated.load(Ordering::Acquire) as u32
    }

    /// Get unrealized PnL based on current stop
    pub fn get_locked_profit(&self) -> f64 {
        let stop_price = self.get_stop_price();
        
        if self.is_long {
            (stop_price - self.entry_price) * self.quantity
        } else {
            (self.entry_price - stop_price) * self.quantity
        }
    }

    /// Get maximum adverse excursion (MAE) in ticks
    pub fn get_mae_ticks(&self) -> i64 {
        let entry_fp = self.extreme_price_fp.load(Ordering::Acquire);
        // MAE calculation would require tracking worst price seen
        // This is a simplified version
        0
    }

    /// Manually cancel the trailing stop
    pub fn cancel(&self) {
        self.state.store(StepTrailingState::Cancelled as u64, Ordering::Release);
    }

    /// Reset the trailing stop (for re-entry scenarios)
    pub fn reset(&self) {
        let initial_stop = if self.is_long {
            self.entry_price * (1.0 - self.config.initial_distance_pct)
        } else {
            self.entry_price * (1.0 + self.config.initial_distance_pct)
        };

        let tick = self.config.min_tick_size;
        let initial_stop_fp = (initial_stop / tick).round() as u64;
        let entry_fp = (self.entry_price / tick).round() as u64;

        self.stop_price_fp.store(initial_stop_fp, Ordering::Release);
        self.extreme_price_fp.store(entry_fp, Ordering::Release);
        self.steps_accumulated.store(0, Ordering::Release);
        self.last_check_price_fp.store(entry_fp, Ordering::Release);
        self.state.store(StepTrailingState::WaitingFirstStep as u64, Ordering::Release);
    }
}

/// Manager for multiple step-trailing stops
pub struct StepTrailingManager {
    /// Active trailing stops by position ID
    stops: dashmap::DashMap<String, StepTrailingStop>,
    /// Index by symbol
    symbol_index: dashmap::DashMap<String, Vec<String>>,
}

impl StepTrailingManager {
    pub fn new() -> Self {
        StepTrailingManager {
            stops: dashmap::DashMap::new(),
            symbol_index: dashmap::DashMap::new(),
        }
    }

    /// Create a new step-trailing stop for a position
    pub fn create_stop(
        &self,
        position_id: String,
        is_long: bool,
        entry_price: f64,
        quantity: f64,
        config: StepTrailingConfig,
        symbol: String,
    ) -> Result<(), &'static str> {
        if self.stops.contains_key(&position_id) {
            return Err("Position already has a trailing stop");
        }

        let stop = StepTrailingStop::new(
            is_long,
            entry_price,
            quantity,
            config,
            symbol.clone(),
        );

        self.stops.insert(position_id.clone(), stop);

        self.symbol_index
            .entry(symbol)
            .or_insert_with(Vec::new)
            .push(position_id);

        Ok(())
    }

    /// Update all stops for a symbol
    pub fn update_symbol(&self, symbol: &str, current_price: f64) -> Vec<(String, f64)> {
        let mut updates = Vec::new();

        if let Some(position_ids) = self.symbol_index.get(symbol) {
            for position_id in position_ids.iter() {
                if let Some(stop) = self.stops.get(position_id) {
                    if let Some(new_stop) = stop.update(current_price) {
                        updates.push((position_id.clone(), new_stop));
                    }
                }
            }
        }

        updates
    }

    /// Check triggers for all stops of a symbol
    pub fn check_triggers(&self, symbol: &str, current_price: f64) -> Vec<String> {
        let mut triggered = Vec::new();

        if let Some(position_ids) = self.symbol_index.get(symbol) {
            for position_id in position_ids.iter() {
                if let Some(stop) = self.stops.get(position_id) {
                    if stop.check_trigger(current_price) {
                        triggered.push(position_id.clone());
                    }
                }
            }
        }

        triggered
    }

    /// Get stop for a position
    pub fn get_stop(&self, position_id: &str) -> Option<StepTrailingStop> {
        self.stops.get(position_id).map(|r| r.value().clone())
    }

    /// Remove a completed/cancelled stop
    pub fn remove_stop(&self, position_id: &str) -> bool {
        if let Some(stop) = self.stops.remove(position_id) {
            let symbol = stop.1.symbol;
            if let Some(mut positions) = self.symbol_index.get_mut(&symbol) {
                positions.retain(|p| p != position_id);
            }
            true
        } else {
            false
        }
    }

    /// Get total locked profit across all stops
    pub fn total_locked_profit(&self) -> f64 {
        self.stops.iter()
            .map(|r| r.value().get_locked_profit())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_trailing_long() {
        let config = StepTrailingConfig {
            initial_distance_pct: 0.02,
            step_size_pct: 0.01,
            break_even_after_first_step: true,
            min_tick_size: 0.01,
            ..Default::default()
        };

        let stop = StepTrailingStop::new(
            true, // long
            100.0, // entry
            1.0,   // quantity
            config,
            "TEST".to_string(),
        );

        // Initial stop should be 2% below entry
        assert!((stop.get_stop_price() - 98.0).abs() < 0.02);

        // Price moves up 1% - should trigger first step to break-even
        stop.update(101.0);
        
        // After 1% move with break_even enabled, stop should be at entry
        assert!((stop.get_stop_price() - 100.0).abs() < 0.02);

        // Price moves up another 1% - stop should trail 1% below
        stop.update(102.0);
        
        assert!(stop.get_stop_price() >= 101.0);
        assert_eq!(stop.get_steps(), 2);
    }

    #[test]
    fn test_no_whipsaw() {
        let config = StepTrailingConfig {
            step_size_pct: 0.01, // 1% step
            min_tick_size: 0.01,
            ..Default::default()
        };

        let stop = StepTrailingStop::new(
            true,
            100.0,
            1.0,
            config,
            "TEST".to_string(),
        );

        // Small moves (< 1%) should NOT trigger trailing
        stop.update(100.5); // 0.5% move
        assert_eq!(stop.get_steps(), 0);

        stop.update(100.8); // Another 0.3%
        assert_eq!(stop.get_steps(), 0);

        // Only when we hit full 1% should it trail
        stop.update(101.0); // Now at 1%
        assert!(stop.get_steps() >= 1);
    }

    #[test]
    fn test_max_steps_cap() {
        let config = StepTrailingConfig {
            step_size_pct: 0.01,
            max_steps: 5,
            min_tick_size: 0.01,
            ..Default::default()
        };

        let stop = StepTrailingStop::new(
            true,
            100.0,
            1.0,
            config,
            "TEST".to_string(),
        );

        // Simulate large price move (10% = 10 steps worth)
        stop.update(110.0);

        // Should be capped at max_steps
        assert!(stop.get_steps() <= 5);
    }
}
