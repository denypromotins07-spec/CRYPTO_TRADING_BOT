//! Fill Probability Estimator based on Queue Decay
//!
//! This module estimates the probability of limit order fills based on
//! queue position, decay rates, and historical cancellation patterns.
//! It uses zero-cost abstractions and is optimized for the 8GB RAM constraint.
//!
//! Designed for the ZAID PERSONAL CRYPTO TRADING BOT.
//! Implements advanced queue decay analysis for accurate fill prediction.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;

/// Maximum history size for decay calculations
const MAX_HISTORY_SIZE: usize = 1000;

/// Minimum observations needed for reliable probability
const MIN_OBSERVATIONS: usize = 30;

/// Order ID type
pub type OrderId = u64;

/// Price tick type
pub type PriceTick = i64;

/// Volume type
pub type Volume = u64;

/// Fill probability estimate with confidence interval
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FillProbability {
    /// Point estimate of fill probability [0, 1]
    pub probability: f64,
    /// Lower bound of 95% confidence interval
    pub lower_bound: f64,
    /// Upper bound of 95% confidence interval
    pub upper_bound: f64,
    /// Number of observations used
    pub observations: usize,
    /// Estimated time to fill in milliseconds
    pub expected_time_ms: f64,
    /// Confidence score [0, 1]
    pub confidence: f64,
}

impl FillProbability {
    #[inline]
    pub const fn new(
        probability: f64,
        lower_bound: f64,
        upper_bound: f64,
        observations: usize,
        expected_time_ms: f64,
        confidence: f64,
    ) -> Self {
        Self {
            probability,
            lower_bound,
            upper_bound,
            observations,
            expected_time_ms,
            confidence,
        }
    }

    /// Create a zero-probability estimate
    #[inline]
    pub const fn zero() -> Self {
        Self {
            probability: 0.0,
            lower_bound: 0.0,
            upper_bound: 0.0,
            observations: 0,
            expected_time_ms: f64::INFINITY,
            confidence: 0.0,
        }
    }

    /// Check if probability is reliable (enough observations)
    #[inline]
    pub const fn is_reliable(&self) -> bool {
        self.observations >= MIN_OBSERVATIONS && self.confidence > 0.5
    }
}

/// Queue decay observation
#[derive(Debug, Clone, Copy)]
pub struct DecayObservation {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Queue position at observation
    pub position: u32,
    /// Volume ahead in queue
    pub volume_ahead: Volume,
    /// Whether this observation resulted in a fill
    pub was_filled: bool,
    /// Time to fill in ms (if filled)
    pub time_to_fill_ms: Option<f64>,
    /// Decay rate at observation
    pub decay_rate: f64,
}

/// Side of the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// Historical decay data for a price level
pub struct LevelDecayHistory {
    /// Circular buffer of observations
    observations: VecDeque<DecayObservation>,
    /// Running sum of decay rates
    decay_sum: f64,
    /// Count of fills
    fill_count: usize,
    /// Total time to fill for filled orders
    total_fill_time_ms: f64,
}

impl LevelDecayHistory {
    #[inline]
    pub fn new() -> Self {
        Self {
            observations: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            decay_sum: 0.0,
            fill_count: 0,
            total_fill_time_ms: 0.0,
        }
    }

    #[inline]
    pub fn add(&mut self, obs: DecayObservation) {
        if self.observations.len() >= MAX_HISTORY_SIZE {
            if let Some(old) = self.observations.pop_front() {
                self.decay_sum -= old.decay_rate;
            }
        }
        self.decay_sum += obs.decay_rate;
        
        if obs.was_filled {
            self.fill_count += 1;
            if let Some(ttf) = obs.time_to_fill_ms {
                self.total_fill_time_ms += ttf;
            }
        }
        
        self.observations.push_back(obs);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    #[inline]
    pub fn avg_decay_rate(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        self.decay_sum / self.observations.len() as f64
    }

    #[inline]
    pub fn fill_rate(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        self.fill_count as f64 / self.observations.len() as f64
    }

    #[inline]
    pub fn avg_time_to_fill(&self) -> f64 {
        if self.fill_count == 0 {
            return f64::INFINITY;
        }
        self.total_fill_time_ms / self.fill_count as f64
    }

    #[inline]
    pub fn recent_observations(&self, count: usize) -> &[DecayObservation] {
        let len = self.observations.len();
        let start = len.saturating_sub(count);
        &self.observations.as_slices().0[start..]
    }
}

/// Fill probability estimator
pub struct FillProbabilityEstimator {
    /// Symbol being tracked
    symbol: String,
    /// Decay history per price level and side
    histories: RwLock<HashMap<(Side, PriceTick), LevelDecayHistory>>,
    /// Current queue positions for active orders
    active_positions: RwLock<HashMap<OrderId, ActiveOrderState>>,
    /// Global decay rate multiplier based on market conditions
    market_activity_multiplier: RwLock<f64>,
    /// Last update timestamp
    last_update: RwLock<Instant>,
}

/// State of an active order
#[derive(Debug, Clone)]
pub struct ActiveOrderState {
    pub order_id: OrderId,
    pub side: Side,
    pub price: PriceTick,
    pub quantity: Volume,
    pub initial_position: u32,
    pub current_position: u32,
    pub volume_ahead: Volume,
    pub placed_at_us: u64,
    pub last_updated_us: u64,
}

impl ActiveOrderState {
    #[inline]
    pub fn age_ms(&self) -> f64 {
        let now_us = get_timestamp_us();
        ((now_us - self.placed_at_us) / 1000) as f64
    }

    #[inline]
    pub fn position_improvement(&self) -> i32 {
        self.initial_position as i32 - self.current_position as i32
    }
}

/// Get current timestamp in microseconds
#[inline]
pub fn get_timestamp_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

impl FillProbabilityEstimator {
    /// Create a new estimator for a symbol
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            histories: RwLock::new(HashMap::with_capacity(500)),
            active_positions: RwLock::new(HashMap::with_capacity(100)),
            market_activity_multiplier: RwLock::new(1.0),
            last_update: RwLock::new(Instant::now()),
        }
    }

    /// Record a queue decay observation
    pub fn record_observation(
        &self,
        side: Side,
        price: PriceTick,
        position: u32,
        volume_ahead: Volume,
        decay_rate: f64,
        was_filled: bool,
        time_to_fill_ms: Option<f64>,
    ) {
        let obs = DecayObservation {
            timestamp_us: get_timestamp_us(),
            position,
            volume_ahead,
            was_filled,
            time_to_fill_ms,
            decay_rate,
        };

        let mut histories = self.histories.write();
        histories
            .entry((side, price))
            .or_insert_with(LevelDecayHistory::new)
            .add(obs);

        *self.last_update.write() = Instant::now();
    }

    /// Register a new active order
    pub fn register_order(
        &self,
        order_id: OrderId,
        side: Side,
        price: PriceTick,
        quantity: Volume,
        initial_position: u32,
        volume_ahead: Volume,
    ) {
        let now = get_timestamp_us();
        let state = ActiveOrderState {
            order_id,
            side,
            price,
            quantity,
            initial_position,
            current_position: initial_position,
            volume_ahead,
            placed_at_us: now,
            last_updated_us: now,
        };

        self.active_positions.write().insert(order_id, state);
    }

    /// Update position of an active order
    pub fn update_order_position(&self, order_id: OrderId, new_position: u32, new_volume_ahead: Volume) -> bool {
        let mut positions = self.active_positions.write();
        if let Some(state) = positions.get_mut(&order_id) {
            state.current_position = new_position;
            state.volume_ahead = new_volume_ahead;
            state.last_updated_us = get_timestamp_us();
            true
        } else {
            false
        }
    }

    /// Remove a completed or cancelled order
    pub fn complete_order(&self, order_id: OrderId, was_filled: bool) -> Option<ActiveOrderState> {
        let mut positions = self.active_positions.write();
        if let Some(state) = positions.remove(&order_id) {
            // Record the final observation
            let time_to_fill = if was_filled {
                Some(state.age_ms())
            } else {
                None
            };

            drop(positions);
            self.record_observation(
                state.side,
                state.price,
                state.current_position,
                state.volume_ahead,
                self.calculate_level_decay_rate(state.side, state.price),
                was_filled,
                time_to_fill,
            );
            Some(state)
        } else {
            None
        }
    }

    /// Calculate decay rate for a specific level
    pub fn calculate_level_decay_rate(&self, side: Side, price: PriceTick) -> f64 {
        let histories = self.histories.read();
        if let Some(history) = histories.get(&(side, price)) {
            history.avg_decay_rate() * *self.market_activity_multiplier.read()
        } else {
            0.0
        }
    }

    /// Estimate fill probability for a given queue position
    pub fn estimate_probability(
        &self,
        side: Side,
        price: PriceTick,
        current_position: u32,
        volume_ahead: Volume,
    ) -> FillProbability {
        let histories = self.histories.read();
        
        if let Some(history) = histories.get(&(side, price)) {
            if history.len() < MIN_OBSERVATIONS {
                // Not enough data, return low-confidence estimate
                let base_prob = history.fill_rate();
                return FillProbability::new(
                    base_prob,
                    0.0,
                    base_prob * 2.0,
                    history.len(),
                    history.avg_time_to_fill(),
                    history.len() as f64 / MIN_OBSERVATIONS as f64,
                );
            }

            // Calculate probability based on multiple factors
            let base_fill_rate = history.fill_rate();
            let avg_decay = history.avg_decay_rate();
            
            // Position factor: closer to front = higher probability
            let position_factor = 1.0 / (1.0 + current_position as f64);
            
            // Volume ahead factor: less volume = higher probability
            let volume_factor = 1.0 / (1.0 + (volume_ahead as f64 / 1000.0));
            
            // Decay factor: higher decay = faster queue movement = higher probability
            let decay_factor = avg_decay.min(1.0);
            
            // Combine factors with weights
            let probability = base_fill_rate * 0.4 
                + position_factor * 0.3 
                + volume_factor * 0.2 
                + decay_factor * 0.1;
            
            let probability = probability.min(1.0).max(0.0);
            
            // Calculate confidence interval using Wilson score
            let n = history.len() as f64;
            let p = probability;
            let z = 1.96; // 95% CI
            
            let denominator = 1.0 + z * z / n;
            let center = (p + z * z / (2.0 * n)) / denominator;
            let margin = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt() / denominator;
            
            let lower = (center - margin).max(0.0);
            let upper = (center + margin).min(1.0);
            
            // Expected time to fill based on position and decay
            let expected_time = if avg_decay > 0.0 {
                (current_position as f64 / avg_decay) * *self.market_activity_multiplier.read()
            } else {
                f64::INFINITY
            };
            
            // Confidence based on sample size and variance
            let confidence = (n / MAX_HISTORY_SIZE as f64).min(1.0) * (1.0 - (upper - lower));
            
            FillProbability::new(
                probability,
                lower,
                upper,
                history.len(),
                expected_time,
                confidence,
            )
        } else {
            // No history, return zero estimate
            FillProbability::zero()
        }
    }

    /// Get fill probability for an active order
    pub fn get_order_probability(&self, order_id: OrderId) -> Option<FillProbability> {
        let positions = self.active_positions.read();
        if let Some(state) = positions.get(&order_id) {
            drop(positions);
            Some(self.estimate_probability(
                state.side,
                state.price,
                state.current_position,
                state.volume_ahead,
            ))
        } else {
            None
        }
    }

    /// Update market activity multiplier based on overall market conditions
    pub fn set_market_activity(&self, multiplier: f64) {
        *self.market_activity_multiplier.write() = multiplier.max(0.1).min(10.0);
    }

    /// Get all active orders sorted by fill probability
    pub fn get_active_orders_by_probability(&self) -> Vec<(OrderId, FillProbability)> {
        let positions = self.active_positions.read();
        let mut results: Vec<(OrderId, FillProbability)> = positions
            .iter()
            .filter_map(|(&id, state)| {
                Some((
                    id,
                    self.estimate_probability(
                        state.side,
                        state.price,
                        state.current_position,
                        state.volume_ahead,
                    ),
                ))
            })
            .collect();
        
        results.sort_by(|a, b| b.1.probability.partial_cmp(&a.1.probability).unwrap());
        results
    }

    /// Get symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get number of historical observations for a level
    pub fn get_history_size(&self, side: Side, price: PriceTick) -> usize {
        let histories = self.histories.read();
        histories
            .get(&(side, price))
            .map(|h| h.len())
            .unwrap_or(0)
    }

    /// Clear all data
    pub fn reset(&self) {
        self.histories.write().clear();
        self.active_positions.write().clear();
        *self.market_activity_multiplier.write() = 1.0;
        *self.last_update.write() = Instant::now();
    }
}

/// Builder pattern for FillProbabilityEstimator
pub struct FillProbabilityEstimatorBuilder {
    symbol: String,
    initial_activity_multiplier: f64,
}

impl FillProbabilityEstimatorBuilder {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            initial_activity_multiplier: 1.0,
        }
    }

    pub fn with_activity_multiplier(mut self, multiplier: f64) -> Self {
        self.initial_activity_multiplier = multiplier;
        self
    }

    pub fn build(self) -> FillProbabilityEstimator {
        let estimator = FillProbabilityEstimator::new(&self.symbol);
        estimator.set_market_activity(self.initial_activity_multiplier);
        estimator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_probability_estimation() {
        let estimator = FillProbabilityEstimator::new("BTCUSDT");
        
        // Record some observations
        for i in 0..50 {
            estimator.record_observation(
                Side::Bid,
                50000,
                i % 10,
                100 + i,
                0.1 + (i as f64 * 0.01),
                i % 3 == 0,
                if i % 3 == 0 { Some(100.0 + i as f64) } else { None },
            );
        }
        
        let prob = estimator.estimate_probability(Side::Bid, 50000, 5, 500);
        assert!(prob.probability >= 0.0 && prob.probability <= 1.0);
        assert!(prob.observations > 0);
    }

    #[test]
    fn test_order_lifecycle() {
        let estimator = FillProbabilityEstimator::new("ETHUSDT");
        
        estimator.register_order(1, Side::Ask, 3000, 100, 10, 500);
        assert!(estimator.get_order_probability(1).is_some());
        
        estimator.update_order_position(1, 5, 250);
        
        let completed = estimator.complete_order(1, true);
        assert!(completed.is_some());
        assert!(estimator.get_order_probability(1).is_none());
    }

    #[test]
    fn test_fill_probability_zero_when_no_data() {
        let estimator = FillProbabilityEstimator::new("SOLUSDT");
        
        let prob = estimator.estimate_probability(Side::Bid, 100, 0, 0);
        assert_eq!(prob.probability, 0.0);
        assert_eq!(prob.observations, 0);
    }
}
