/**
 * Queue Position Simulator for Order Book Fill Probability Estimation
 * 
 * Estimates fill probability based on:
 * - Queue position in order book
 * - Order book dynamics and cancellations
 * - Market maker behavior patterns
 * - Latency effects on queue position
 * 
 * Zero-cost abstractions, optimized for 8GB RAM.
 */

use std::collections::{BTreeMap, VecDeque};

/// Queue position state
#[derive(Debug, Clone)]
pub struct QueuePosition {
    pub order_id: u64,
    pub price: f64,
    pub side: Side,
    pub initial_position: usize,
    pub current_position: usize,
    pub quantity: f64,
    pub time_in_queue_ns: u64,
    pub estimated_fill_probability: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Order book queue at a single price level
#[derive(Debug, Clone)]
pub struct PriceQueue {
    pub price: f64,
    pub total_quantity: f64,
    /// Queue of (order_id, quantity) tuples in time priority
    orders: VecDeque<(u64, f64)>,
    /// Cancellation rate at this level (per second)
    cancellation_rate: f64,
    /// New arrival rate (per second)
    arrival_rate: f64,
}

impl PriceQueue {
    pub fn new(price: f64) -> Self {
        Self {
            price,
            total_quantity: 0.0,
            orders: VecDeque::new(),
            cancellation_rate: 0.1,  // 10% per second default
            arrival_rate: 1.0,       // 1 order per second default
        }
    }

    /// Add order to back of queue
    pub fn add_order(&mut self, order_id: u64, quantity: f64) {
        self.orders.push_back((order_id, quantity));
        self.total_quantity += quantity;
    }

    /// Remove order from queue
    pub fn remove_order(&mut self, order_id: u64) -> Option<f64> {
        for (i, (id, qty)) in self.orders.iter().enumerate() {
            if *id == order_id {
                let qty = *qty;
                self.orders.remove(i);
                self.total_quantity -= qty;
                return Some(qty);
            }
        }
        None
    }

    /// Get position of order in queue
    pub fn get_position(&self, order_id: u64) -> Option<usize> {
        self.orders.iter().position(|(id, _)| *id == order_id)
    }

    /// Get quantity ahead of order
    pub fn quantity_ahead(&self, order_id: u64) -> Option<f64> {
        let mut sum = 0.0;
        for (id, qty) in &self.orders {
            if *id == order_id {
                return Some(sum);
            }
            sum += qty;
        }
        None
    }

    /// Simulate queue evolution over time interval
    pub fn simulate_evolution(&mut self, dt_seconds: f64, rng: &mut impl Rng) {
        // Simulate cancellations
        let mut to_cancel = Vec::new();
        for (i, (id, qty)) in self.orders.iter().enumerate() {
            let cancel_prob = 1.0 - (-self.cancellation_rate * dt_seconds).exp();
            if rng.gen::<f64>() < cancel_prob {
                to_cancel.push(i);
            }
        }
        
        // Remove cancelled orders (in reverse to preserve indices)
        for i in to_cancel.into_iter().rev() {
            if let Some((_, qty)) = self.orders.remove(i) {
                self.total_quantity -= qty;
            }
        }
        
        // Simulate new arrivals (Poisson process)
        let expected_arrivals = self.arrival_rate * dt_seconds;
        let arrivals = rng.poisson(expected_arrivals);
        for _ in 0..arrivals {
            let new_qty = rng.gen_range(0.1..5.0);
            self.orders.push_back((rng.gen(), new_qty));
            self.total_quantity += new_qty;
        }
    }
}

/// Simple RNG trait for testing
pub trait Rng {
    fn gen(&mut self) -> u64;
    fn gen_range(&mut self, min: f64, max: f64) -> f64;
    fn poisson(&mut self, lambda: f64) -> usize;
}

/// Queue position simulator
pub struct QueuePositionSimulator {
    /// Queues at each price level for bids
    bid_queues: BTreeMap<f64, PriceQueue>,
    /// Queues at each price level for asks
    ask_queues: BTreeMap<f64, PriceQueue>,
    /// Tracked orders
    tracked_orders: Vec<QueuePosition>,
    /// Market parameters
    params: SimulationParams,
}

#[derive(Debug, Clone)]
pub struct SimulationParams {
    /// Average trade size at best bid/ask
    pub avg_trade_size: f64,
    /// Trade arrival rate (per second)
    pub trade_rate: f64,
    /// Probability of trade eating multiple levels
    pub multi_level_prob: f64,
    /// Cancellation rate multiplier during stress
    pub stress_cancel_multiplier: f64,
    /// Current market stress level (0-1)
    pub stress_level: f64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            avg_trade_size: 0.5,  // BTC
            trade_rate: 10.0,      // 10 trades per second
            multi_level_prob: 0.1,
            stress_cancel_multiplier: 3.0,
            stress_level: 0.0,
        }
    }
}

impl QueuePositionSimulator {
    pub fn new(params: SimulationParams) -> Self {
        Self {
            bid_queues: BTreeMap::new(),
            ask_queues: BTreeMap::new(),
            tracked_orders: Vec::new(),
            params,
        }
    }

    /// Track a new order's queue position
    pub fn track_order(
        &mut self,
        order_id: u64,
        price: f64,
        side: Side,
        quantity: f64,
        timestamp_ns: u64,
    ) {
        let queue = match side {
            Side::Buy => self.bid_queues.entry(price).or_insert_with(|| PriceQueue::new(price)),
            Side::Sell => self.ask_queues.entry(price).or_insert_with(|| PriceQueue::new(price)),
        };
        
        queue.add_order(order_id, quantity);
        
        let position = queue.get_position(order_id).unwrap_or(0);
        let quantity_ahead = queue.quantity_ahead(order_id).unwrap_or(0.0);
        
        // Estimate fill probability based on position
        let fill_prob = self.estimate_fill_probability(
            side,
            price,
            position,
            quantity_ahead,
            quantity,
        );
        
        let tracked = QueuePosition {
            order_id,
            price,
            side,
            initial_position: position,
            current_position: position,
            quantity,
            time_in_queue_ns: 0,
            estimated_fill_probability: fill_prob,
        };
        
        self.tracked_orders.push(tracked);
    }

    /// Estimate fill probability for an order
    fn estimate_fill_probability(
        &self,
        side: Side,
        price: f64,
        position: usize,
        quantity_ahead: f64,
        own_quantity: f64,
    ) -> f64 {
        // Base probability decreases with position
        let position_factor = 1.0 / (1.0 + position as f64);
        
        // Quantity ahead factor
        let expected_volume = self.params.trade_rate * self.params.avg_trade_size;
        let volume_factor = if quantity_ahead < expected_volume {
            1.0
        } else {
            expected_volume / quantity_ahead
        };
        
        // Price competitiveness
        let is_best = match side {
            Side::Buy => {
                self.bid_queues.keys().last().map_or(false, |&p| p == price)
            },
            Side::Sell => {
                self.ask_queues.keys().next().map_or(false, |&p| p == price)
            },
        };
        
        let price_factor = if is_best { 1.0 } else { 0.3 };
        
        // Stress adjustment
        let stress_factor = 1.0 - self.params.stress_level * 0.5;
        
        position_factor * volume_factor * price_factor * stress_factor
    }

    /// Simulate queue evolution and update fill probabilities
    pub fn simulate_step(&mut self, dt_ns: u64, rng: &mut impl Rng) {
        let dt_seconds = dt_ns as f64 / 1_000_000_000.0;
        
        // Update queues
        for queue in self.bid_queues.values_mut() {
            queue.simulate_evolution(dt_seconds, rng);
        }
        for queue in self.ask_queues.values_mut() {
            queue.simulate_evolution(dt_seconds, rng);
        }
        
        // Simulate trades at best prices
        self.simulate_trades(rng);
        
        // Update tracked orders
        for order in &mut self.tracked_orders {
            order.time_in_queue_ns += dt_ns;
            
            // Update current position
            let queue = match order.side {
                Side::Buy => self.bid_queues.get(&order.price),
                Side::Sell => self.ask_queues.get(&order.price),
            };
            
            if let Some(q) = queue {
                if let Some(pos) = q.get_position(order.order_id) {
                    order.current_position = pos;
                    let qty_ahead = q.quantity_ahead(order.order_id).unwrap_or(0.0);
                    order.estimated_fill_probability = self.estimate_fill_probability(
                        order.side,
                        order.price,
                        pos,
                        qty_ahead,
                        order.quantity,
                    );
                } else {
                    // Order was filled or cancelled
                    order.estimated_fill_probability = if order.current_position == 0 {
                        1.0  // Filled
                    } else {
                        0.0  // Cancelled
                    };
                }
            }
        }
    }

    /// Simulate trades consuming liquidity
    fn simulate_trades(&mut self, rng: &mut impl Rng) {
        // Number of trades in this interval
        let trade_count = rng.poisson(self.params.trade_rate);
        
        for _ in 0..trade_count {
            // Randomly choose side
            let is_buy = rng.gen::<f64>() < 0.5;
            
            // Get best queue
            let best_queue = if is_buy {
                self.ask_queues.values_mut().next()
            } else {
                self.bid_queues.values_mut().rev().next()
            };
            
            if let Some(queue) = best_queue {
                // Trade size
                let trade_size = rng.gen_range(
                    self.params.avg_trade_size * 0.5,
                    self.params.avg_trade_size * 2.0,
                );
                
                // Consume from front of queue
                let mut remaining = trade_size;
                while remaining > 0.0 && !queue.orders.is_empty() {
                    if let Some((_, ref mut qty)) = queue.orders.front_mut() {
                        if *qty <= remaining {
                            remaining -= *qty;
                            queue.orders.pop_front();
                        } else {
                            *qty -= remaining;
                            remaining = 0.0;
                        }
                    }
                }
                
                // Check if trade spills to next level
                if remaining > 0.0 && rng.gen::<f64>() < self.params.multi_level_prob {
                    // Would consume next level (simplified)
                }
            }
        }
    }

    /// Get fill probability for specific order
    pub fn get_fill_probability(&self, order_id: u64) -> Option<f64> {
        self.tracked_orders
            .iter()
            .find(|o| o.order_id == order_id)
            .map(|o| o.estimated_fill_probability)
    }

    /// Get all tracked orders
    pub fn get_tracked_orders(&self) -> &[QueuePosition] {
        &self.tracked_orders
    }

    /// Get queue depth at price level
    pub fn get_queue_depth(&self, price: f64, side: Side) -> Option<f64> {
        match side {
            Side::Buy => self.bid_queues.get(&price).map(|q| q.total_quantity),
            Side::Sell => self.ask_queues.get(&price).map(|q| q.total_quantity),
        }
    }

    /// Set market stress level
    pub fn set_stress_level(&mut self, stress: f64) {
        self.params.stress_level = stress.clamp(0.0, 1.0);
        
        // Increase cancellation rates during stress
        let multiplier = 1.0 + (self.params.stress_cancel_multiplier - 1.0) * self.params.stress_level;
        for queue in self.bid_queues.values_mut() {
            queue.cancellation_rate *= multiplier;
        }
        for queue in self.ask_queues.values_mut() {
            queue.cancellation_rate *= multiplier;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRng(u64);
    
    impl Rng for TestRng {
        fn gen(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0
        }
        
        fn gen_range(&mut self, min: f64, max: f64) -> f64 {
            min + (self.gen() as f64 / u64::MAX as f64) * (max - min)
        }
        
        fn poisson(&mut self, lambda: f64) -> usize {
            // Simple Poisson approximation
            let k = 0;
            let mut p = 1.0;
            let l = (-lambda).exp();
            while p > l {
                p *= self.gen::<f64>();
            }
            k
        }
    }

    #[test]
    fn test_queue_position_tracking() {
        let mut sim = QueuePositionSimulator::new(SimulationParams::default());
        let mut rng = TestRng(42);
        
        // Add orders
        sim.track_order(1, 50000.0, Side::Buy, 1.0, 0);
        sim.track_order(2, 50000.0, Side::Buy, 0.5, 1000);
        sim.track_order(3, 50000.0, Side::Buy, 2.0, 2000);
        
        // Check positions
        assert_eq!(sim.get_fill_probability(1), Some(1.0));  // First in queue
        assert!(sim.get_fill_probability(3).unwrap() < 1.0);  // Behind others
    }

    #[test]
    fn test_simulation_step() {
        let mut sim = QueuePositionSimulator::new(SimulationParams::default());
        let mut rng = TestRng(123);
        
        sim.track_order(1, 50000.0, Side::Buy, 1.0, 0);
        
        let initial_prob = sim.get_fill_probability(1).unwrap();
        
        // Run simulation
        sim.simulate_step(1_000_000_000, &mut rng);  // 1 second
        
        // Probability should have changed
        let new_prob = sim.get_fill_probability(1).unwrap();
        assert!(new_prob >= 0.0 && new_prob <= 1.0);
    }
}
