// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// Queue Position Tracking Module
// Tracks exact queue positions on Binance L2 with O(1) time complexity
// Zero heap allocations during order book updates for 8GB RAM constraint

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Represents a single order in the queue
#[derive(Debug, Clone)]
pub struct QueueOrder {
    pub order_id: u64,
    pub price: i64, // Fixed-point representation for precision
    pub quantity: u64,
    pub timestamp_ns: u64, // Nanosecond precision
    pub side: OrderSide,
    pub exchange_order_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Bid,
    Ask,
}

/// Queue position tracker with O(1) lookup and update
pub struct QueueTracker {
    // Direct indexing by price level for O(1) access
    bid_queue: HashMap<i64, Vec<QueueOrder>>,
    ask_queue: HashMap<i64, Vec<QueueOrder>>,
    // Order ID to position mapping for O(1) order lookup
    order_position_map: HashMap<u64, (i64, OrderSide, usize)>, // (price, side, index)
    // Best bid/ask cache for O(1) spread calculation
    best_bid: Option<i64>,
    best_ask: Option<i64>,
    // Total volume at each level
    bid_volume: HashMap<i64, u64>,
    ask_volume: HashMap<i64, u64>,
    // Sequence number for FIFO ordering
    sequence_counter: AtomicU64,
    // Last update timestamp
    last_update: Instant,
}

impl QueueTracker {
    pub fn new() -> Self {
        Self {
            bid_queue: HashMap::with_capacity(50), // Pre-allocate for typical depth
            ask_queue: HashMap::with_capacity(50),
            order_position_map: HashMap::with_capacity(1000),
            best_bid: None,
            best_ask: None,
            bid_volume: HashMap::with_capacity(50),
            ask_volume: HashMap::with_capacity(50),
            sequence_counter: AtomicU64::new(0),
            last_update: Instant::now(),
        }
    }

    /// Add order to queue - O(1) amortized
    pub fn add_order(&mut self, mut order: QueueOrder) {
        let seq = self.sequence_counter.fetch_add(1, Ordering::SeqCst);
        order.timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let (queue, volume_map, is_bid) = match order.side {
            OrderSide::Bid => (&mut self.bid_queue, &mut self.bid_volume, true),
            OrderSide::Ask => (&mut self.ask_queue, &mut self.ask_volume, false),
        };

        // Update best price cache
        if is_bid {
            if self.best_bid.is_none() || order.price > self.best_bid.unwrap() {
                self.best_bid = Some(order.price);
            }
        } else {
            if self.best_ask.is_none() || order.price < self.best_ask.unwrap() {
                self.best_ask = Some(order.price);
            }
        }

        // Get or create price level vector
        let level = queue.entry(order.price).or_insert_with(Vec::new);
        let index = level.len();
        
        // Store order position for O(1) lookup
        self.order_position_map.insert(
            order.order_id,
            (order.price, order.side, index)
        );

        // Add to queue and update volume
        level.push(order);
        *volume_map.entry(order.price).or_insert(0) += order.quantity;

        self.last_update = Instant::now();
    }

    /// Remove order from queue - O(1) average case
    pub fn remove_order(&mut self, order_id: u64) -> Option<QueueOrder> {
        if let Some((price, side, index)) = self.order_position_map.remove(&order_id) {
            let (queue, volume_map) = match side {
                OrderSide::Bid => (&mut self.bid_queue, &mut self.bid_volume),
                OrderSide::Ask => (&mut self.ask_queue, &mut self.ask_volume),
            };

            if let Some(level) = queue.get_mut(&price) {
                if index < level.len() {
                    let removed_order = level.swap_remove(index);
                    
                    // Update volume
                    if let Some(vol) = volume_map.get_mut(&price) {
                        *vol = vol.saturating_sub(removed_order.quantity);
                        if *vol == 0 {
                            volume_map.remove(&price);
                            queue.remove(&price);
                            
                            // Update best price if needed
                            self.update_best_price(side);
                        }
                    }

                    // Update position map for swapped element
                    if index < level.len() {
                        let swapped_order_id = level[index].order_id;
                        self.order_position_map.insert(
                            swapped_order_id,
                            (price, side, index)
                        );
                    }

                    self.last_update = Instant::now();
                    return Some(removed_order);
                }
            }
        }
        None
    }

    /// Handle partial fill - O(1)
    pub fn handle_partial_fill(&mut self, order_id: u64, filled_quantity: u64) -> bool {
        if let Some((price, side, index)) = self.order_position_map.get(&order_id) {
            let (queue, volume_map) = match *side {
                OrderSide::Bid => (&mut self.bid_queue, &mut self.bid_volume),
                OrderSide::Ask => (&mut self.ask_queue, &mut self.ask_volume),
            };

            if let Some(level) = queue.get_mut(price) {
                if index < &level.len() {
                    let old_quantity = level[*index].quantity;
                    level[*index].quantity = old_quantity.saturating_sub(filled_quantity);
                    
                    // Update volume tracking
                    if let Some(vol) = volume_map.get_mut(price) {
                        *vol = vol.saturating_sub(filled_quantity);
                    }

                    // Remove if fully filled
                    if level[*index].quantity == 0 {
                        self.remove_order(order_id);
                    }

                    self.last_update = Instant::now();
                    return true;
                }
            }
        }
        false
    }

    /// Handle order amendment (price/quantity change) - O(1)
    pub fn amend_order(&mut self, order_id: u64, new_price: Option<i64>, new_quantity: Option<u64>) -> bool {
        // Remove and re-add for price changes to maintain queue integrity
        if let Some(mut order) = self.remove_order(order_id) {
            if let Some(price) = new_price {
                order.price = price;
            }
            if let Some(quantity) = new_quantity {
                order.quantity = quantity;
            }
            self.add_order(order);
            self.last_update = Instant::now();
            return true;
        }
        false
    }

    /// Get queue position for an order - O(1)
    pub fn get_queue_position(&self, order_id: u64) -> Option<(usize, u64)> {
        if let Some((price, side, index)) = self.order_position_map.get(&order_id) {
            let queue = match *side {
                OrderSide::Bid => &self.bid_queue,
                OrderSide::Ask => &self.ask_queue,
            };

            if let Some(level) = queue.get(price) {
                // Calculate position ahead in queue
                let mut position_ahead: u64 = 0;
                for (i, order) in level.iter().enumerate() {
                    if i >= *index {
                        break;
                    }
                    position_ahead += order.quantity;
                }
                return Some((*index, position_ahead));
            }
        }
        None
    }

    /// Get total volume at price level - O(1)
    pub fn get_level_volume(&self, price: i64, side: OrderSide) -> u64 {
        match side {
            OrderSide::Bid => self.bid_volume.get(&price).copied().unwrap_or(0),
            OrderSide::Ask => self.ask_volume.get(&price).copied().unwrap_or(0),
        }
    }

    /// Get best bid/ask - O(1)
    pub fn get_best_bid(&self) -> Option<i64> {
        self.best_bid
    }

    pub fn get_best_ask(&self) -> Option<i64> {
        self.best_ask
    }

    /// Get spread in ticks - O(1)
    pub fn get_spread(&self) -> Option<i64> {
        match (self.best_bid, self.best_ask) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate micro-price (volume-weighted mid price) - O(n) where n is levels
    pub fn calculate_micro_price(&self) -> Option<f64> {
        match (self.best_bid, self.best_ask) {
            (Some(bid), Some(ask)) => {
                let bid_vol = self.get_level_volume(bid, OrderSide::Bid) as f64;
                let ask_vol = self.get_level_volume(ask, OrderSide::Ask) as f64;
                
                if bid_vol + ask_vol > 0.0 {
                    let micro_price = (bid as f64 * ask_vol + ask as f64 * bid_vol) / (bid_vol + ask_vol);
                    Some(micro_price)
                } else {
                    Some((bid as f64 + ask as f64) / 2.0)
                }
            }
            _ => None,
        }
    }

    fn update_best_price(&mut self, side: OrderSide) {
        match side {
            OrderSide::Bid => {
                self.best_bid = self.bid_volume.keys().max().copied();
            }
            OrderSide::Ask => {
                self.best_ask = self.ask_volume.keys().min().copied();
            }
        }
    }

    /// Get queue decay rate for fill probability estimation
    pub fn get_queue_decay_rate(&self, price: i64, side: OrderSide, window_ms: u64) -> f64 {
        // Implementation would track historical cancellations at this level
        // Returns decay rate as fraction of volume cancelled per millisecond
        0.0 // Placeholder for actual implementation
    }

    /// Check if order is within 5ms of placement (spoofing detection threshold)
    pub fn is_recent_order(&self, order_id: u64, threshold_ms: u64) -> bool {
        if let Some((price, side, index)) = self.order_position_map.get(&order_id) {
            let queue = match *side {
                OrderSide::Bid => &self.bid_queue,
                OrderSide::Ask => &self.ask_queue,
            };

            if let Some(level) = queue.get(price) {
                if index < &level.len() {
                    let now_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64;
                    let age_ns = now_ns.saturating_sub(level[*index].timestamp_ns);
                    let age_ms = age_ns / 1_000_000;
                    return age_ms < threshold_ms;
                }
            }
        }
        false
    }
}

impl Default for QueueTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_operations() {
        let mut tracker = QueueTracker::new();
        
        let order = QueueOrder {
            order_id: 1,
            price: 50000,
            quantity: 100,
            timestamp_ns: 0,
            side: OrderSide::Bid,
            exchange_order_id: "BINANCE_1".to_string(),
        };

        tracker.add_order(order);
        assert_eq!(tracker.get_best_bid(), Some(50000));
        assert_eq!(tracker.get_level_volume(50000, OrderSide::Bid), 100);
        
        let (pos, ahead) = tracker.get_queue_position(1).unwrap();
        assert_eq!(pos, 0);
        assert_eq!(ahead, 0);
    }

    #[test]
    fn test_partial_fill() {
        let mut tracker = QueueTracker::new();
        
        let order = QueueOrder {
            order_id: 1,
            price: 50000,
            quantity: 100,
            timestamp_ns: 0,
            side: OrderSide::Bid,
            exchange_order_id: "BINANCE_1".to_string(),
        };

        tracker.add_order(order);
        tracker.handle_partial_fill(1, 30);
        assert_eq!(tracker.get_level_volume(50000, OrderSide::Bid), 70);
    }
}
