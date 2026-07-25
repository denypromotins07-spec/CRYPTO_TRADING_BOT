// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// Fill Probability Estimation Module
// Estimates fill chances based on queue decay and position tracking
// Zero-cost abstractions for memory-efficient operation on 8GB RAM systems

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// Queue decay metrics for a price level
#[derive(Debug, Clone)]
pub struct QueueDecayMetrics {
    pub price: i64,
    pub side: Side,
    pub total_volume: u64,
    pub cancelled_volume: u64,
    pub filled_volume: u64,
    pub avg_order_age_ms: f64,
    pub decay_rate_per_ms: f64,
    pub last_update: Instant,
}

impl QueueDecayMetrics {
    pub fn new(price: i64, side: Side) -> Self {
        Self {
            price,
            side,
            total_volume: 0,
            cancelled_volume: 0,
            filled_volume: 0,
            avg_order_age_ms: 0.0,
            decay_rate_per_ms: 0.0,
            last_update: Instant::now(),
        }
    }

    /// Update metrics with new cancellation data
    pub fn record_cancellation(&mut self, volume: u64, age_ms: f64) {
        self.cancelled_volume += volume;
        self.update_decay_rate(age_ms);
        self.last_update = Instant::now();
    }

    /// Update metrics with new fill data
    pub fn record_fill(&mut self, volume: u64, age_ms: f64) {
        self.filled_volume += volume;
        self.update_decay_rate(age_ms);
        self.last_update = Instant::now();
    }

    /// Add volume to the level
    pub fn add_volume(&mut self, volume: u64) {
        self.total_volume += volume;
        self.last_update = Instant::now();
    }

    /// Remove volume from the level
    pub fn remove_volume(&mut self, volume: u64) {
        self.total_volume = self.total_volume.saturating_sub(volume);
        self.last_update = Instant::now();
    }

    fn update_decay_rate(&mut self, age_ms: f64) {
        // Exponential moving average for decay rate
        let new_decay = if self.total_volume > 0 {
            1.0 / (age_ms + 1.0)
        } else {
            0.0
        };
        
        // Smooth the decay rate
        self.decay_rate_per_ms = 0.7 * self.decay_rate_per_ms + 0.3 * new_decay;
        
        // Update average order age
        self.avg_order_age_ms = 0.8 * self.avg_order_age_ms + 0.2 * age_ms;
    }

    /// Get estimated time to fill in milliseconds
    pub fn estimated_fill_time_ms(&self, position_ahead: u64) -> f64 {
        if self.decay_rate_per_ms <= 0.0 || self.total_volume == 0 {
            return f64::INFINITY;
        }

        // Estimate based on decay rate and position
        let effective_rate = self.decay_rate_per_ms * self.total_volume as f64;
        if effective_rate <= 0.0 {
            return f64::INFINITY;
        }

        position_ahead as f64 / effective_rate
    }
}

/// Fill probability estimation result
#[derive(Debug, Clone)]
pub struct FillProbabilityResult {
    pub probability: f64,           // 0.0 to 1.0
    pub expected_fill_time_ms: f64,
    pub confidence: f64,            // 0.0 to 1.0
    pub adverse_selection_risk: f64, // 0.0 to 1.0
    pub recommended_action: Action,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    PlaceOrder,
    Wait,
    Cancel,
    AdjustPrice,
}

/// Fill Probability Engine using queue dynamics
pub struct FillProbabilityEngine {
    // Queue decay metrics per price level
    decay_metrics: HashMap<(i64, Side), QueueDecayMetrics>,
    
    // Historical fill data for calibration
    fill_history: VecDeque<FillRecord>,
    max_history_size: usize,
    
    // Global decay parameters
    base_decay_rate: f64,
    volatility_adjustment: f64,
    
    // Last calculation timestamp
    last_calculation: Instant,
    
    // Memory-efficient pre-allocated buffers
    temp_buffer: Vec<f64>,
}

#[derive(Debug, Clone)]
struct FillRecord {
    timestamp: Instant,
    price: i64,
    side: Side,
    fill_time_ms: f64,
    was_adverse: bool,
}

impl FillProbabilityEngine {
    pub fn new(max_history: usize) -> Self {
        Self {
            decay_metrics: HashMap::with_capacity(100),
            fill_history: VecDeque::with_capacity(max_history),
            max_history_size: max_history,
            base_decay_rate: 0.001, // Base decay rate per ms
            volatility_adjustment: 0.0,
            last_calculation: Instant::now(),
            temp_buffer: Vec::with_capacity(50),
        }
    }

    /// Record a new order at a price level
    pub fn record_order_placement(&mut self, price: i64, side: Side, volume: u64) {
        let key = (price, side);
        self.decay_metrics
            .entry(key)
            .or_insert_with(|| QueueDecayMetrics::new(price, side))
            .add_volume(volume);
        self.last_calculation = Instant::now();
    }

    /// Record a cancellation at a price level
    pub fn record_cancellation(&mut self, price: i64, side: Side, volume: u64, age_ms: f64) {
        if let Some(metrics) = self.decay_metrics.get_mut(&(price, side)) {
            metrics.record_cancellation(volume, age_ms);
            metrics.remove_volume(volume);
        }
        self.last_calculation = Instant::now();
    }

    /// Record a fill at a price level
    pub fn record_fill(&mut self, price: i64, side: Side, volume: u64, age_ms: f64, was_adverse: bool) {
        if let Some(metrics) = self.decay_metrics.get_mut(&(price, side)) {
            metrics.record_fill(volume, age_ms);
            metrics.remove_volume(volume);
        }

        // Record in history for calibration
        let record = FillRecord {
            timestamp: Instant::now(),
            price,
            side,
            fill_time_ms: age_ms,
            was_adverse,
        };

        self.fill_history.push_back(record);
        if self.fill_history.len() > self.max_history_size {
            self.fill_history.pop_front();
        }

        self.last_calculation = Instant::now();
    }

    /// Calculate fill probability for an order at a specific position
    pub fn calculate_fill_probability(
        &self,
        price: i64,
        side: Side,
        position_ahead: u64,
        our_volume: u64,
        time_horizon_ms: f64,
    ) -> FillProbabilityResult {
        let key = (price, side);
        
        // Get decay metrics for this level
        let metrics = match self.decay_metrics.get(&key) {
            Some(m) => m,
            None => return self.default_result(),
        };

        // Calculate base fill probability from queue position
        let total_volume_ahead = position_ahead as f64;
        let level_volume = metrics.total_volume as f64;
        
        // Position-based probability
        let position_prob = if level_volume > 0.0 {
            1.0 - (total_volume_ahead / (level_volume + our_volume as f64))
        } else {
            0.0
        };

        // Time-based probability using decay rate
        let estimated_fill_time = metrics.estimated_fill_time_ms(position_ahead);
        let time_prob = if estimated_fill_time.is_finite() && estimated_fill_time > 0.0 {
            1.0 - (-time_horizon_ms / estimated_fill_time).exp()
        } else {
            0.0
        };

        // Historical calibration factor
        let historical_factor = self.calculate_historical_factor(price, side);

        // Combine probabilities with weights
        let base_probability = 0.4 * position_prob + 0.4 * time_prob + 0.2 * historical_factor;

        // Apply volatility adjustment
        let adjusted_probability = base_probability * (1.0 + self.volatility_adjustment);

        // Calculate adverse selection risk
        let adverse_risk = self.calculate_adverse_selection_risk(price, side);

        // Determine confidence based on data availability
        let confidence = self.calculate_confidence(&key);

        // Determine recommended action
        let action = self.determine_action(adjusted_probability, adverse_risk, estimated_fill_time, time_horizon_ms);

        FillProbabilityResult {
            probability: adjusted_probability.clamp(0.0, 1.0),
            expected_fill_time_ms: estimated_fill_time,
            confidence,
            adverse_selection_risk: adverse_risk,
            recommended_action: action,
        }
    }

    /// Calculate fill probability for market order (immediate execution)
    pub fn calculate_market_order_fill_probability(
        &self,
        price: i64,
        side: Side,
        order_volume: u64,
    ) -> FillProbabilityResult {
        let key = (price, side);
        
        let metrics = match self.decay_metrics.get(&key) {
            Some(m) => m,
            None => return self.default_result(),
        };

        // For market orders, check if there's sufficient liquidity
        let available_volume = metrics.total_volume as f64;
        let fill_fraction = (order_volume as f64 / available_volume).min(1.0);

        // Market orders have high adverse selection risk
        let adverse_risk = 0.3 + (1.0 - fill_fraction) * 0.5;

        FillProbabilityResult {
            probability: fill_fraction,
            expected_fill_time_ms: 0.0, // Immediate
            confidence: 0.9, // High confidence for market orders
            adverse_selection_risk: adverse_risk.min(1.0),
            recommended_action: Action::PlaceOrder,
        }
    }

    /// Update volatility adjustment based on market conditions
    pub fn update_volatility_adjustment(&mut self, volatility: f64) {
        // Higher volatility increases decay rates
        self.volatility_adjustment = (volatility - 0.02).clamp(-0.5, 0.5);
        self.last_calculation = Instant::now();
    }

    /// Get queue decay rate for a price level
    pub fn get_decay_rate(&self, price: i64, side: Side) -> f64 {
        self.decay_metrics
            .get(&(price, side))
            .map(|m| m.decay_rate_per_ms)
            .unwrap_or(self.base_decay_rate)
    }

    /// Clear all stored data
    pub fn clear(&mut self) {
        self.decay_metrics.clear();
        self.fill_history.clear();
        self.temp_buffer.clear();
        self.last_calculation = Instant::now();
    }

    fn default_result(&self) -> FillProbabilityResult {
        FillProbabilityResult {
            probability: 0.0,
            expected_fill_time_ms: f64::INFINITY,
            confidence: 0.0,
            adverse_selection_risk: 0.5,
            recommended_action: Action::Wait,
        }
    }

    fn calculate_historical_factor(&self, price: i64, side: Side) -> f64 {
        // Filter relevant historical records
        self.temp_buffer.clear();
        
        for record in &self.fill_history {
            if record.price == price && record.side == side {
                self.temp_buffer.push(record.fill_time_ms);
            }
        }

        if self.temp_buffer.is_empty() {
            return 0.5; // Neutral prior
        }

        // Calculate average fill time from history
        let avg_fill_time: f64 = self.temp_buffer.iter().sum::<f64>() / self.temp_buffer.len() as f64;
        
        // Convert to probability (shorter fill time = higher probability)
        let normalized = 1.0 / (1.0 + avg_fill_time / 1000.0);
        normalized.clamp(0.0, 1.0)
    }

    fn calculate_adverse_selection_risk(&self, price: i64, side: Side) -> f64 {
        // Count adverse fills in recent history
        let mut adverse_count = 0;
        let mut total_count = 0;

        for record in self.fill_history.iter().rev().take(50) {
            if record.price == price && record.side == side {
                total_count += 1;
                if record.was_adverse {
                    adverse_count += 1;
                }
            }
        }

        if total_count == 0 {
            return 0.3; // Default moderate risk
        }

        adverse_count as f64 / total_count as f64
    }

    fn calculate_confidence(&self, key: &(i64, Side)) -> f64 {
        let mut confidence = 0.0;

        // Data sufficiency (based on history size)
        let history_factor = (self.fill_history.len() as f64 / self.max_history_size as f64).min(1.0);
        confidence += 0.4 * history_factor;

        // Level-specific data
        if let Some(metrics) = self.decay_metrics.get(key) {
            let level_factor = if metrics.total_volume > 0 { 1.0 } else { 0.0 };
            confidence += 0.3 * level_factor;
        }

        // Recent activity
        let recent_activity = self.fill_history.iter().filter(|r| r.timestamp.elapsed() < Duration::from_secs(60)).count();
        let activity_factor = (recent_activity as f64 / 100.0).min(1.0);
        confidence += 0.3 * activity_factor;

        confidence.clamp(0.0, 1.0)
    }

    fn determine_action(
        &self,
        probability: f64,
        adverse_risk: f64,
        expected_time_ms: f64,
        horizon_ms: f64,
    ) -> Action {
        // High adverse selection risk -> cancel or adjust
        if adverse_risk > 0.7 {
            return Action::Cancel;
        }

        // Very low probability -> wait or adjust price
        if probability < 0.2 {
            if expected_time_ms > horizon_ms * 2.0 {
                Action::AdjustPrice
            } else {
                Action::Wait
            }
        }
        // High probability and acceptable risk -> place order
        else if probability > 0.7 && adverse_risk < 0.4 {
            Action::PlaceOrder
        }
        // Moderate case -> wait
        else {
            Action::Wait
        }
    }
}

impl Default for FillProbabilityEngine {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_probability_basic() {
        let mut engine = FillProbabilityEngine::new(100);
        
        // Record some order placements
        engine.record_order_placement(50000, Side::Bid, 100);
        engine.record_order_placement(50000, Side::Bid, 200);
        
        // Simulate a fill
        engine.record_fill(50000, Side::Bid, 50, 100.0, false);
        
        // Calculate probability for new order
        let result = engine.calculate_fill_probability(
            50000,
            Side::Bid,
            100, // position ahead
            100, // our volume
            1000.0, // 1 second horizon
        );
        
        assert!(result.probability >= 0.0 && result.probability <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_queue_decay_metrics() {
        let mut metrics = QueueDecayMetrics::new(50000, Side::Bid);
        metrics.add_volume(100);
        metrics.record_cancellation(20, 50.0);
        
        assert_eq!(metrics.cancelled_volume, 20);
        assert!(metrics.decay_rate_per_ms > 0.0);
    }
}
