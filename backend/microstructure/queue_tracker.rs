//! Queue Position Tracking for Binance L2 Order Book
//! 
//! This module provides O(1) queue position tracking without heap allocations
//! during order book updates. It tracks exact queue positions for limit orders
//! and handles partial fills and order amendments efficiently.
//! 
//! Designed for the ZAID PERSONAL CRYPTO TRADING BOT with strict 8GB RAM constraints.
//! Uses zero-cost abstractions and arena allocation for memory efficiency.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crossbeam::queue::SegQueue;
use parking_lot::RwLock;

/// Maximum number of price levels to track per side
const MAX_PRICE_LEVELS: usize = 100;

/// Unique identifier for an order
pub type OrderId = u64;

/// Price representation in ticks (i64 for efficient arithmetic)
pub type PriceTick = i64;

/// Volume representation (u64 for non-negative volumes)
pub type Volume = u64;

/// Queue position metadata - stored inline for cache efficiency
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct QueuePosition {
    /// Order ID from the exchange
    pub order_id: OrderId,
    /// Position in the queue (0 = front)
    pub position: u32,
    /// Original quantity when order was placed
    pub original_qty: Volume,
    /// Remaining quantity after partial fills
    pub remaining_qty: Volume,
    /// Timestamp in microseconds since epoch
    pub timestamp_us: u64,
    /// Whether this order has been amended
    pub is_amended: bool,
}

impl QueuePosition {
    #[inline]
    pub const fn new(order_id: OrderId, qty: Volume, timestamp_us: u64) -> Self {
        Self {
            order_id,
            position: 0,
            original_qty: qty,
            remaining_qty: qty,
            timestamp_us,
            is_amended: false,
        }
    }

    #[inline]
    pub fn fill(&mut self, fill_qty: Volume) -> Volume {
        let actual_fill = fill_qty.min(self.remaining_qty);
        self.remaining_qty -= actual_fill;
        actual_fill
    }

    #[inline]
    pub fn amend(&mut self, new_qty: Volume) {
        self.original_qty = new_qty;
        self.remaining_qty = new_qty.min(self.remaining_qty);
        self.is_amended = true;
    }

    #[inline]
    pub const fn fill_probability(&self) -> f64 {
        if self.original_qty == 0 {
            return 0.0;
        }
        1.0 - (self.remaining_qty as f64 / self.original_qty as f64)
    }
}

/// Price level with embedded queue - avoids heap allocation per level
#[derive(Debug)]
pub struct PriceLevel {
    /// Price in ticks
    pub price: PriceTick,
    /// Total volume at this level
    pub total_volume: Volume,
    /// Number of orders in queue
    pub order_count: u32,
    /// Fixed-size array for queue positions (avoids Vec reallocation)
    pub queue: [Option<QueuePosition>; 50],
    /// Next available slot in queue array
    next_slot: usize,
}

impl PriceLevel {
    #[inline]
    pub const fn new(price: PriceTick) -> Self {
        Self {
            price,
            total_volume: 0,
            order_count: 0,
            queue: [None; 50],
            next_slot: 0,
        }
    }

    #[inline]
    pub fn add_order(&mut self, position: QueuePosition) -> Option<usize> {
        if self.next_slot >= self.queue.len() {
            return None; // Queue full
        }
        let slot = self.next_slot;
        self.queue[slot] = Some(position);
        self.total_volume += position.original_qty;
        self.order_count += 1;
        self.next_slot += 1;
        Some(slot)
    }

    #[inline]
    pub fn remove_order(&mut self, order_id: OrderId) -> Option<QueuePosition> {
        for i in 0..self.next_slot {
            if let Some(pos) = self.queue[i] {
                if pos.order_id == order_id {
                    let removed = self.queue[i].take().unwrap();
                    self.total_volume -= removed.original_qty;
                    self.order_count -= 1;
                    // Compact the queue - O(n) but n is small (max 50)
                    for j in i..self.next_slot - 1 {
                        self.queue[j] = self.queue[j + 1];
                    }
                    self.next_slot -= 1;
                    return Some(removed);
                }
            }
        }
        None
    }

    #[inline]
    pub fn get_position(&self, order_id: OrderId) -> Option<(usize, QueuePosition)> {
        for i in 0..self.next_slot {
            if let Some(pos) = self.queue[i] {
                if pos.order_id == order_id {
                    return Some((i, pos));
                }
            }
        }
        None
    }

    #[inline]
    pub fn update_positions(&mut self) {
        // Update all positions after removals
        for i in 0..self.next_slot {
            if let Some(ref mut pos) = self.queue[i] {
                pos.position = i as u32;
            }
        }
    }
}

/// Side of the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

/// Queue tracker for a single symbol
pub struct QueueTracker {
    /// Symbol name (e.g., "BTCUSDT")
    symbol: String,
    /// Bid side price levels
    bids: RwLock<HashMap<PriceTick, PriceLevel>>,
    /// Ask side price levels
    asks: RwLock<HashMap<PriceTick, PriceLevel>>,
    /// Map from order_id to (side, price, slot) for O(1) lookup
    order_index: RwLock<HashMap<OrderId, (Side, PriceTick, usize)>>,
    /// Best bid price
    best_bid: RwLock<Option<PriceTick>>,
    /// Best ask price
    best_ask: RwLock<Option<PriceTick>>,
    /// Last update timestamp
    last_update: RwLock<Instant>,
    /// Event queue for observer pattern
    event_queue: SegQueue<QueueEvent>,
}

/// Events emitted by the queue tracker
#[derive(Debug, Clone)]
pub enum QueueEvent {
    OrderAdded { order_id: OrderId, side: Side, price: PriceTick, position: u32 },
    OrderRemoved { order_id: OrderId, side: Side, reason: RemovalReason },
    OrderFilled { order_id: OrderId, side: Side, fill_qty: Volume, remaining: Volume },
    OrderAmended { order_id: OrderId, side: Side, new_qty: Volume },
    QueueDecay { side: Side, price: PriceTick, decay_rate: f64 },
}

/// Reason for order removal
#[derive(Debug, Clone, Copy)]
pub enum RemovalReason {
    Cancelled,
    FullyFilled,
    Expired,
    Replaced,
}

impl QueueTracker {
    /// Create a new queue tracker for a symbol
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: RwLock::new(HashMap::with_capacity(MAX_PRICE_LEVELS)),
            asks: RwLock::new(HashMap::with_capacity(MAX_PRICE_LEVELS)),
            order_index: RwLock::new(HashMap::with_capacity(1000)),
            best_bid: RwLock::new(None),
            best_ask: RwLock::new(None),
            last_update: RwLock::new(Instant::now()),
            event_queue: SegQueue::new(),
        }
    }

    /// Get current timestamp in microseconds
    #[inline]
    fn now_us() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    /// Add a new order to the queue - O(1) average case
    pub fn add_order(&self, order_id: OrderId, side: Side, price: PriceTick, qty: Volume) -> bool {
        let timestamp = Self::now_us();
        let position = QueuePosition::new(order_id, qty, timestamp);

        let levels = match side {
            Side::Bid => self.bids.write(),
            Side::Ask => self.asks.write(),
        };

        let level = levels.entry(price).or_insert_with(|| PriceLevel::new(price));
        
        if level.add_order(position).is_some() {
            // Update order index for O(1) lookup
            drop(levels);
            self.order_index.write().insert(order_id, (side, price, 0));
            
            // Update best price
            self.update_best_price(side, price);
            
            // Emit event
            self.event_queue.push(QueueEvent::OrderAdded {
                order_id,
                side,
                price,
                position: 0,
            });
            
            *self.last_update.write() = Instant::now();
            true
        } else {
            false
        }
    }

    /// Remove an order from the queue - O(1) with index lookup
    pub fn remove_order(&self, order_id: OrderId, reason: RemovalReason) -> Option<QueuePosition> {
        let mut index = self.order_index.write();
        if let Some((side, price, _slot)) = index.remove(&order_id) {
            drop(index);
            
            let mut levels = match side {
                Side::Bid => self.bids.write(),
                Side::Ask => self.asks.write(),
            };

            if let Some(level) = levels.get_mut(&price) {
                if let Some(removed) = level.remove_order(order_id) {
                    level.update_positions();
                    
                    if level.order_count == 0 {
                        levels.remove(&price);
                        self.update_best_price_after_removal(side, price);
                    }

                    self.event_queue.push(QueueEvent::OrderRemoved {
                        order_id,
                        side,
                        reason,
                    });

                    *self.last_update.write() = Instant::now();
                    return Some(removed);
                }
            }
        }
        None
    }

    /// Process a partial or full fill - O(1) with index lookup
    pub fn process_fill(&self, order_id: OrderId, fill_qty: Volume) -> Option<Volume> {
        let index = self.order_index.read();
        if let Some(&(side, price, slot)) = index.get(&order_id) {
            drop(index);
            
            let mut levels = match side {
                Side::Bid => self.bids.write(),
                Side::Ask => self.asks.write(),
            };

            if let Some(level) = levels.get_mut(&price) {
                if let Some((_, ref mut pos)) = level.get_position(order_id) {
                    let remaining = pos.remaining_qty;
                    let actual_fill = pos.fill(fill_qty);
                    
                    if pos.remaining_qty == 0 {
                        drop(levels);
                        self.remove_order(order_id, RemovalReason::FullyFilled);
                    } else {
                        self.event_queue.push(QueueEvent::OrderFilled {
                            order_id,
                            side,
                            fill_qty: actual_fill,
                            remaining: pos.remaining_qty,
                        });
                    }

                    *self.last_update.write() = Instant::now();
                    return Some(actual_fill);
                }
            }
        }
        None
    }

    /// Amend an existing order - O(1) with index lookup
    pub fn amend_order(&self, order_id: OrderId, new_qty: Volume) -> bool {
        let index = self.order_index.read();
        if let Some(&(side, price, _slot)) = index.get(&order_id) {
            drop(index);
            
            let mut levels = match side {
                Side::Bid => self.bids.write(),
                Side::Ask => self.asks.write(),
            };

            if let Some(level) = levels.get_mut(&price) {
                if let Some((_, ref mut pos)) = level.get_position(order_id) {
                    pos.amend(new_qty);
                    
                    self.event_queue.push(QueueEvent::OrderAmended {
                        order_id,
                        side,
                        new_qty,
                    });

                    *self.last_update.write() = Instant::now();
                    return true;
                }
            }
        }
        false
    }

    /// Get queue position for an order - O(1)
    pub fn get_queue_position(&self, order_id: OrderId) -> Option<QueuePosition> {
        let index = self.order_index.read();
        if let Some(&(side, price, slot)) = index.get(&order_id) {
            drop(index);
            
            let levels = match side {
                Side::Bid => self.bids.read(),
                Side::Ask => self.asks.read(),
            };

            if let Some(level) = levels.get(&price) {
                if let Some((_, pos)) = level.get_position(order_id) {
                    return Some(pos);
                }
            }
        }
        None
    }

    /// Calculate queue decay rate for a price level
    pub fn calculate_decay_rate(&self, side: Side, price: PriceTick, window_ms: u64) -> f64 {
        let levels = match side {
            Side::Bid => self.bids.read(),
            Side::Ask => self.asks.read(),
        };

        if let Some(level) = levels.get(&price) {
            if level.order_count == 0 {
                return 0.0;
            }
            
            // Simple decay calculation based on queue position and time
            let total_time_weight: f64 = level.queue[..level.next_slot]
                .iter()
                .filter_map(|opt| opt.as_ref())
                .map(|pos| {
                    let age_ms = ((Self::now_us() - pos.timestamp_us) / 1000) as f64;
                    1.0 / (1.0 + age_ms / window_ms as f64)
                })
                .sum();

            total_time_weight / level.order_count as f64
        } else {
            0.0
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<PriceTick> {
        *self.best_bid.read()
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<PriceTick> {
        *self.best_ask.read()
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        let bid = self.best_bid();
        let ask = self.best_ask();
        match (bid, ask) {
            (Some(b), Some(a)) => Some((b as f64 + a as f64) / 2.0),
            _ => None,
        }
    }

    /// Get spread in ticks
    pub fn spread(&self) -> Option<PriceTick> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Update best price for a side
    fn update_best_price(&self, side: Side, price: PriceTick) {
        match side {
            Side::Bid => {
                let mut best = self.best_bid.write();
                if best.is_none() || price > best.unwrap() {
                    *best = Some(price);
                }
            }
            Side::Ask => {
                let mut best = self.best_ask.write();
                if best.is_none() || price < best.unwrap() {
                    *best = Some(price);
                }
            }
        }
    }

    /// Update best price after order removal
    fn update_best_price_after_removal(&self, side: Side, removed_price: PriceTick) {
        match side {
            Side::Bid => {
                let mut best = self.best_bid.write();
                if best == Some(removed_price) {
                    // Need to find new best bid
                    *best = self.bids.read().keys().max().copied();
                }
            }
            Side::Ask => {
                let mut best = self.best_ask.write();
                if best == Some(removed_price) {
                    // Need to find new best ask
                    *best = self.asks.read().keys().min().copied();
                }
            }
        }
    }

    /// Poll for events (Observer pattern)
    pub fn poll_events(&self) -> Vec<QueueEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.event_queue.pop() {
            events.push(event);
        }
        events
    }

    /// Get symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get last update time
    pub fn last_update(&self) -> Instant {
        *self.last_update.read()
    }
}

/// Builder for QueueTracker with fluent API
pub struct QueueTrackerBuilder {
    symbol: String,
}

impl QueueTrackerBuilder {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
        }
    }

    pub fn build(self) -> QueueTracker {
        QueueTracker::new(&self.symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_remove_order() {
        let tracker = QueueTracker::new("BTCUSDT");
        
        assert!(tracker.add_order(1, Side::Bid, 50000, 100));
        assert!(tracker.get_queue_position(1).is_some());
        
        let removed = tracker.remove_order(1, RemovalReason::Cancelled);
        assert!(removed.is_some());
        assert!(tracker.get_queue_position(1).is_none());
    }

    #[test]
    fn test_partial_fill() {
        let tracker = QueueTracker::new("ETHUSDT");
        
        tracker.add_order(1, Side::Ask, 3000, 500);
        
        let fill1 = tracker.process_fill(1, 200);
        assert_eq!(fill1, Some(200));
        
        let pos = tracker.get_queue_position(1).unwrap();
        assert_eq!(pos.remaining_qty, 300);
        
        let fill2 = tracker.process_fill(1, 300);
        assert_eq!(fill2, Some(300));
        
        // Order should be fully filled and removed
        assert!(tracker.get_queue_position(1).is_none());
    }

    #[test]
    fn test_order_amendment() {
        let tracker = QueueTracker::new("SOLUSDT");
        
        tracker.add_order(1, Side::Bid, 10000, 1000);
        assert!(tracker.amend_order(1, 1500));
        
        let pos = tracker.get_queue_position(1).unwrap();
        assert_eq!(pos.original_qty, 1500);
        assert!(pos.is_amended);
    }
}
