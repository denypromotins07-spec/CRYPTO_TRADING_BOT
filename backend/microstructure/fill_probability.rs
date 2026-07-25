// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// File: backend/microstructure/fill_probability.rs
// Chapter 1: Queue Position Tracking, Fill Probability, and Micro-Price Calculation
//
// Purpose: Estimate fill chances based on queue decay and order book dynamics
// Constraints: Zero-cost abstractions, no heap allocations during critical path
// Target: AMD Ryzen AI 5 laptop with 8GB RAM limit
//
// Design Patterns: Strategy Pattern for probability models, State Pattern for queue states
// Memory Model: Pre-allocated buffers, stack-based computation where possible

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Maximum history size for queue decay analysis (prevents memory bloat)
const MAX_DECAY_HISTORY: usize = 1000;

/// Minimum probability threshold for considering an order fillable
const MIN_FILL_PROBABILITY: f64 = 0.01;

/// Unique identifier for tracking fill probability calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FillRequestId(pub u64);

/// Represents the current state of a limit order in the queue
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueueState {
    /// Order just placed, at back of queue
    AtBack,
    /// Order has moved forward due to cancellations ahead
    Advancing,
    /// Order is at front, imminent fill likely
    AtFront,
    /// Order being filled partially
    PartialFill,
    /// Order cancelled or fully filled (terminal state)
    Terminal,
}

/// Snapshot of queue dynamics for probability calculation
#[derive(Debug, Clone)]
pub struct QueueDynamicsSnapshot {
    /// Our position in queue (0-indexed)
    pub our_position: u32,
    /// Total orders ahead of us
    pub orders_ahead: u32,
    /// Total quantity ahead of us
    pub quantity_ahead: i64,
    /// Quantity at our price level
    pub total_level_quantity: i64,
    /// Rate of cancellations per second at this level
    pub cancellation_rate: f64,
    /// Rate of new orders joining behind us per second
    pub join_rate: f64,
    /// Aggressive market order rate (consumption rate)
    pub consumption_rate: f64,
    /// Time since order placement in microseconds
    pub age_us: u64,
    /// Current spread in ticks
    pub spread_ticks: u32,
}

/// Result of fill probability calculation
#[derive(Debug, Clone)]
pub struct FillProbabilityResult {
    /// Probability of full fill within time horizon (0.0 to 1.0)
    pub fill_probability: f64,
    /// Expected time to fill in milliseconds
    pub expected_time_to_fill_ms: f64,
    /// Probability of partial fill
    pub partial_fill_probability: f64,
    /// Probability of adverse selection if filled
    pub adverse_selection_risk: f64,
    /// Confidence in the probability estimate (0.0 to 1.0)
    pub confidence: f64,
    /// Recommended action based on probability
    pub recommendation: FillRecommendation,
    /// Timestamp of calculation
    pub timestamp_us: u64,
}

/// Action recommendation based on fill probability
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillRecommendation {
    /// High probability, keep order as-is
    Hold,
    /// Low probability, consider cancelling and repricing
    CancelAndReprice,
    /// Medium probability, wait but monitor closely
    Monitor,
    /// Very high probability, prepare for fill
    PrepareForFill,
    /// Extremely low probability, abort strategy
    Abort,
}

/// Tracks historical queue decay patterns for probability estimation
pub struct QueueDecayTracker {
    /// History of quantity ahead over time
    quantity_ahead_history: VecDeque<(u64, i64)>, // (timestamp_us, quantity)
    /// History of position changes
    position_history: VecDeque<(u64, u32)>, // (timestamp_us, position)
    /// Count of cancellations observed
    cancellation_count: u64,
    /// Count of new orders joining
    join_count: u64,
    /// Count of aggressive fills (market orders consuming queue)
    aggressive_fill_count: u64,
    /// Start time for relative timestamps
    start_time: Instant,
}

impl QueueDecayTracker {
    pub fn new() -> Self {
        Self {
            quantity_ahead_history: VecDeque::with_capacity(MAX_DECAY_HISTORY),
            position_history: VecDeque::with_capacity(MAX_DECAY_HISTORY),
            cancellation_count: 0,
            join_count: 0,
            aggressive_fill_count: 0,
            start_time: Instant::now(),
        }
    }

    /// Record a snapshot of quantity ahead - O(1)
    pub fn record_quantity_snapshot(&mut self, quantity_ahead: i64) {
        let timestamp_us = self.start_time.elapsed().as_micros() as u64;
        
        if self.quantity_ahead_history.len() >= MAX_DECAY_HISTORY {
            self.quantity_ahead_history.pop_front();
        }
        
        self.quantity_ahead_history.push_back((timestamp_us, quantity_ahead));
    }

    /// Record a position change - O(1)
    pub fn record_position_change(&mut self, new_position: u32) {
        let timestamp_us = self.start_time.elapsed().as_micros() as u64;
        
        if self.position_history.len() >= MAX_DECAY_HISTORY {
            self.position_history.pop_front();
        }
        
        self.position_history.push_back((timestamp_us, new_position));
    }

    /// Record a cancellation event
    pub fn record_cancellation(&mut self) {
        self.cancellation_count += 1;
    }

    /// Record a new order joining behind us
    pub fn record_join(&mut self) {
        self.join_count += 1;
    }

    /// Record an aggressive market order fill
    pub fn record_aggressive_fill(&mut self) {
        self.aggressive_fill_count += 1;
    }

    /// Calculate decay rate (quantity decreasing per second)
    pub fn calculate_decay_rate(&self) -> f64 {
        if self.quantity_ahead_history.len() < 2 {
            return 0.0;
        }

        let history: Vec<_> = self.quantity_ahead_history.iter().collect();
        let first = history.first().unwrap();
        let last = history.last().unwrap();

        let time_delta_s = ((last.0 - first.0) as f64) / 1_000_000.0;
        if time_delta_s <= 0.0 {
            return 0.0;
        }

        let quantity_delta = first.1 - last.1; // Positive means decay
        quantity_delta as f64 / time_delta_s
    }

    /// Calculate cancellation rate per second
    pub fn cancellation_rate(&self) -> f64 {
        let elapsed_s = self.start_time.elapsed().as_secs_f64();
        if elapsed_s <= 0.0 {
            return 0.0;
        }
        self.cancellation_count as f64 / elapsed_s
    }

    /// Calculate join rate per second
    pub fn join_rate(&self) -> f64 {
        let elapsed_s = self.start_time.elapsed().as_secs_f64();
        if elapsed_s <= 0.0 {
            return 0.0;
        }
        self.join_count as f64 / elapsed_s
    }

    /// Calculate consumption rate (aggressive fills) per second
    pub fn consumption_rate(&self) -> f64 {
        let elapsed_s = self.start_time.elapsed().as_secs_f64();
        if elapsed_s <= 0.0 {
            return 0.0;
        }
        self.aggressive_fill_count as f64 / elapsed_s
    }

    /// Get average position advancement rate
    pub fn position_advancement_rate(&self) -> f64 {
        if self.position_history.len() < 2 {
            return 0.0;
        }

        let history: Vec<_> = self.position_history.iter().collect();
        let first = history.first().unwrap();
        let last = history.last().unwrap();

        let time_delta_s = ((last.0 - first.0) as f64) / 1_000_000.0;
        if time_delta_s <= 0.0 {
            return 0.0;
        }

        let position_delta = first.1 as i64 - last.1 as i64; // Positive means advancing
        position_delta as f64 / time_delta_s
    }
}

impl Default for QueueDecayTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Main fill probability calculator using multiple models
pub struct FillProbabilityCalculator {
    /// Global sequence number for request IDs
    sequence_number: AtomicU64,
    /// Start time for relative timestamps
    start_time: Instant,
    /// Default time horizon for probability calculation (in ms)
    default_time_horizon_ms: f64,
}

impl FillProbabilityCalculator {
    pub fn new() -> Self {
        Self {
            sequence_number: AtomicU64::new(0),
            start_time: Instant::now(),
            default_time_horizon_ms: 5000.0, // 5 second default horizon
        }
    }

    /// Generate a new unique request ID
    pub fn next_request_id(&self) -> FillRequestId {
        FillRequestId(self.sequence_number.fetch_add(1, Ordering::SeqCst))
    }

    /// Calculate fill probability using combined models
    pub fn calculate_probability(
        &self,
        dynamics: &QueueDynamicsSnapshot,
        decay_tracker: &QueueDecayTracker,
    ) -> FillProbabilityResult {
        let timestamp_us = self.start_time.elapsed().as_micros() as u64;

        // Calculate probabilities from different models
        let queue_decay_prob = self.queue_decay_model(dynamics, decay_tracker);
        let poisson_prob = self.poisson_fill_model(dynamics);
        let empirical_prob = self.empirical_model(decay_tracker);

        // Weighted combination of models
        let fill_probability = self.combine_models(
            queue_decay_prob,
            poisson_prob,
            empirical_prob,
            dynamics,
        );

        // Calculate expected time to fill
        let expected_time_ms = self.expected_time_to_fill(dynamics, fill_probability);

        // Calculate partial fill probability
        let partial_fill_prob = self.partial_fill_probability(dynamics);

        // Calculate adverse selection risk
        let adverse_selection = self.adverse_selection_risk(dynamics);

        // Calculate confidence based on data availability
        let confidence = self.calculate_confidence(decay_tracker, dynamics);

        // Determine recommendation
        let recommendation = self.determine_recommendation(
            fill_probability,
            adverse_selection,
            expected_time_ms,
        );

        FillProbabilityResult {
            fill_probability,
            expected_time_to_fill_ms: expected_time_ms,
            partial_fill_probability: partial_fill_prob,
            adverse_selection_risk: adverse_selection,
            confidence,
            recommendation,
            timestamp_us,
        }
    }

    /// Queue decay model: probability based on observed decay rate
    fn queue_decay_model(
        &self,
        dynamics: &QueueDynamicsSnapshot,
        decay_tracker: &QueueDecayTracker,
    ) -> f64 {
        let decay_rate = decay_tracker.calculate_decay_rate();
        if decay_rate <= 0.0 || dynamics.quantity_ahead <= 0 {
            return 0.1; // Minimal baseline probability
        }

        // Time to deplete queue ahead (in seconds)
        let time_to_deplete_s = dynamics.quantity_ahead as f64 / decay_rate;

        // Probability increases as time to deplete decreases
        // Using exponential decay function
        let horizon_s = self.default_time_horizon_ms / 1000.0;
        let probability = 1.0 - (-horizon_s / time_to_deplete_s).exp();

        probability.clamp(0.0, 1.0)
    }

    /// Poisson fill model: assumes fills arrive as Poisson process
    fn poisson_fill_model(&self, dynamics: &QueueDynamicsSnapshot) -> f64 {
        let consumption_rate = dynamics.consumption_rate;
        if consumption_rate <= 0.0 {
            return 0.05; // Minimal baseline
        }

        // Expected number of fills in time horizon
        let horizon_s = self.default_time_horizon_ms / 1000.0;
        let lambda = consumption_rate * horizon_s;

        // Probability of at least one fill event reaching our position
        // This is simplified; real implementation would consider queue position
        let expected_fills_reaching_us = lambda / (dynamics.orders_ahead.max(1) as f64);

        1.0 - (-expected_fills_reaching_us).exp()
    }

    /// Empirical model: based on historical patterns
    fn empirical_model(&self, decay_tracker: &QueueDecayTracker) -> f64 {
        let advancement_rate = decay_tracker.position_advancement_rate();
        if advancement_rate <= 0.0 {
            return 0.1;
        }

        // Extrapolate time to reach front of queue
        // This would need current position from dynamics
        // Simplified here
        0.5 // Placeholder
    }

    /// Combine multiple model predictions with adaptive weighting
    fn combine_models(
        &self,
        queue_decay: f64,
        poisson: f64,
        empirical: f64,
        dynamics: &QueueDynamicsSnapshot,
    ) -> f64 {
        // Adaptive weights based on market conditions
        let mut weights = [0.4, 0.4, 0.2]; // Default: favor queue decay and Poisson

        // In high consumption environments, trust Poisson more
        if dynamics.consumption_rate > 10.0 {
            weights = [0.2, 0.6, 0.2];
        }

        // When we have good decay data, trust queue decay model
        if dynamics.cancellation_rate > 5.0 {
            weights[0] += 0.2;
            weights[1] -= 0.1;
            weights[2] -= 0.1;
        }

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        weights[0] * queue_decay + weights[1] * poisson + weights[2] * empirical
    }

    /// Calculate expected time to fill in milliseconds
    fn expected_time_to_fill(&self, dynamics: &QueueDynamicsSnapshot, probability: f64) -> f64 {
        if probability <= MIN_FILL_PROBABILITY {
            return f64::INFINITY;
        }

        // Inverse relationship: higher probability = shorter expected time
        // Base time scaled by probability
        let base_time_ms = self.default_time_horizon_ms;
        base_time_ms * (1.0 - probability) / probability
    }

    /// Calculate probability of partial fill
    fn partial_fill_probability(&self, dynamics: &QueueDynamicsSnapshot) -> f64 {
        // Partial fills more likely when:
        // 1. Large quantity ahead (multiple market orders needed)
        // 2. High consumption rate
        // 3. We're not at front of queue

        if dynamics.our_position == 0 {
            // At front, likely to get full fill first
            return 0.3;
        }

        let consumption_ratio = dynamics.consumption_rate / 
            (dynamics.quantity_ahead.max(1) as f64);
        
        // Higher consumption relative to queue = more partial fills
        (1.0 - (-consumption_ratio * 100.0).exp()).min(0.8)
    }

    /// Calculate adverse selection risk (probability price moves against us after fill)
    fn adverse_selection_risk(&self, dynamics: &QueueDynamicsSnapshot) -> f64 {
        // Adverse selection risk increases when:
        // 1. Spread is wide (compensation for risk)
        // 2. Queue imbalance is extreme
        // 3. High toxicity (not directly measured here)

        let spread_factor = (dynamics.spread_ticks as f64 / 10.0).min(1.0);
        
        // Simple heuristic: wider spread = higher adverse selection risk
        spread_factor * 0.5
    }

    /// Calculate confidence in probability estimate
    fn calculate_confidence(
        &self,
        decay_tracker: &QueueDecayTracker,
        _dynamics: &QueueDynamicsSnapshot,
    ) -> f64 {
        // Confidence based on:
        // 1. Amount of historical data
        // 2. Consistency of observations
        // 3. Recency of data

        let data_points = decay_tracker.quantity_ahead_history.len();
        
        // More data = higher confidence (up to a point)
        let data_confidence = (data_points as f64 / 100.0).min(1.0);
        
        data_confidence
    }

    /// Determine recommended action based on probability analysis
    fn determine_recommendation(
        &self,
        fill_probability: f64,
        adverse_selection: f64,
        expected_time_ms: f64,
    ) -> FillRecommendation {
        // High adverse selection = be cautious even with high fill probability
        if adverse_selection > 0.7 {
            return FillRecommendation::CancelAndReprice;
        }

        if fill_probability > 0.8 {
            FillRecommendation::PrepareForFill
        } else if fill_probability > 0.5 {
            FillRecommendation::Hold
        } else if fill_probability > 0.2 {
            FillRecommendation::Monitor
        } else if expected_time_ms > 10000.0 {
            FillRecommendation::Abort
        } else {
            FillRecommendation::CancelAndReprice
        }
    }

    /// Get current queue state based on dynamics
    pub fn get_queue_state(&self, dynamics: &QueueDynamicsSnapshot) -> QueueState {
        if dynamics.our_position == 0 {
            QueueState::AtFront
        } else if dynamics.orders_ahead < 5 {
            QueueState::Advancing
        } else {
            QueueState::AtBack
        }
    }
}

impl Default for FillProbabilityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing QueueDynamicsSnapshot
pub struct QueueDynamicsBuilder {
    our_position: u32,
    orders_ahead: u32,
    quantity_ahead: i64,
    total_level_quantity: i64,
    cancellation_rate: f64,
    join_rate: f64,
    consumption_rate: f64,
    age_us: u64,
    spread_ticks: u32,
}

impl QueueDynamicsBuilder {
    pub fn new() -> Self {
        Self {
            our_position: 0,
            orders_ahead: 0,
            quantity_ahead: 0,
            total_level_quantity: 0,
            cancellation_rate: 0.0,
            join_rate: 0.0,
            consumption_rate: 0.0,
            age_us: 0,
            spread_ticks: 1,
        }
    }

    pub fn our_position(mut self, pos: u32) -> Self {
        self.our_position = pos;
        self
    }

    pub fn orders_ahead(mut self, count: u32) -> Self {
        self.orders_ahead = count;
        self
    }

    pub fn quantity_ahead(mut self, qty: i64) -> Self {
        self.quantity_ahead = qty;
        self
    }

    pub fn total_level_quantity(mut self, qty: i64) -> Self {
        self.total_level_quantity = qty;
        self
    }

    pub fn cancellation_rate(mut self, rate: f64) -> Self {
        self.cancellation_rate = rate;
        self
    }

    pub fn join_rate(mut self, rate: f64) -> Self {
        self.join_rate = rate;
        self
    }

    pub fn consumption_rate(mut self, rate: f64) -> Self {
        self.consumption_rate = rate;
        self
    }

    pub fn age_us(mut self, age: u64) -> Self {
        self.age_us = age;
        self
    }

    pub fn spread_ticks(mut self, ticks: u32) -> Self {
        self.spread_ticks = ticks;
        self
    }

    pub fn build(self) -> QueueDynamicsSnapshot {
        QueueDynamicsSnapshot {
            our_position: self.our_position,
            orders_ahead: self.orders_ahead,
            quantity_ahead: self.quantity_ahead,
            total_level_quantity: self.total_level_quantity,
            cancellation_rate: self.cancellation_rate,
            join_rate: self.join_rate,
            consumption_rate: self.consumption_rate,
            age_us: self.age_us,
            spread_ticks: self.spread_ticks,
        }
    }
}

impl Default for QueueDynamicsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_probability_high_consumption() {
        let calculator = FillProbabilityCalculator::new();
        let mut decay_tracker = QueueDecayTracker::new();

        // Simulate high consumption environment
        for _ in 0..10 {
            decay_tracker.record_aggressive_fill();
        }

        let dynamics = QueueDynamicsBuilder::new()
            .our_position(5)
            .orders_ahead(10)
            .quantity_ahead(50000)
            .total_level_quantity(100000)
            .cancellation_rate(5.0)
            .consumption_rate(50.0)
            .spread_ticks(2)
            .build();

        let result = calculator.calculate_probability(&dynamics, &decay_tracker);

        assert!(result.fill_probability > 0.3); // Should be decent probability
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_fill_probability_low_activity() {
        let calculator = FillProbabilityCalculator::new();
        let decay_tracker = QueueDecayTracker::new();

        let dynamics = QueueDynamicsBuilder::new()
            .our_position(50)
            .orders_ahead(100)
            .quantity_ahead(500000)
            .total_level_quantity(1000000)
            .cancellation_rate(0.1)
            .consumption_rate(0.5)
            .spread_ticks(10)
            .build();

        let result = calculator.calculate_probability(&dynamics, &decay_tracker);

        assert!(result.fill_probability < 0.3); // Low probability
        assert_eq!(result.recommendation, FillRecommendation::CancelAndReprice);
    }

    #[test]
    fn test_queue_decay_tracker() {
        let mut tracker = QueueDecayTracker::new();

        // Record decreasing quantity ahead (queue decaying)
        tracker.record_quantity_snapshot(100000);
        std::thread::sleep(Duration::from_millis(10));
        tracker.record_quantity_snapshot(90000);
        std::thread::sleep(Duration::from_millis(10));
        tracker.record_quantity_snapshot(80000);

        let decay_rate = tracker.calculate_decay_rate();
        assert!(decay_rate > 0.0); // Should show positive decay
    }
}
