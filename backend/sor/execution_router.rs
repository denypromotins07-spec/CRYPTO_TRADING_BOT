//! backend/sor/execution_router.rs
//! 
//! ZAID PERSONAL CRYPTO TRADING BOT - Stage 33: Multi-Venue SOR
//! Chapter 2: Execution Router
//! 
//! Splits large parent orders across venues to minimize market impact.
//! Implements TWAP, VWAP, and Iceberg execution algorithms.
//! Uses zero-cost abstractions for maximum throughput on AMD Ryzen AI 5.
//! Strictly respects 8GB RAM limit.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::VecDeque;

/// Execution algorithm type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExecAlgo {
    Twap,      // Time-weighted average price
    Vwap,      // Volume-weighted average price
    Iceberg,   // Hide true order size
    Sniper,    // Immediate execution on liquidity detection
}

/// Child order resulting from parent order split
#[derive(Clone, Debug)]
pub struct ChildOrder {
    pub order_id: u64,
    pub parent_order_id: u64,
    pub venue_id: u8,
    pub side: Side,
    pub price_micro: u64,
    pub quantity_micro: u64,
    pub algo: ExecAlgo,
    pub sequence: usize,
    pub total_sequences: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Parent order to be executed
#[derive(Clone, Debug)]
pub struct ParentOrder {
    pub order_id: u64,
    pub symbol_hash: u64,
    pub side: Side,
    pub total_quantity_micro: u64,
    pub executed_quantity_micro: u64,
    pub algo: ExecAlgo,
    pub time_horizon_ms: u64,
    pub max_participation_rate: f64, // Max % of venue volume
    pub venues: Vec<u8>,
    pub created_at_ns: u64,
}

impl ParentOrder {
    pub fn remaining(&self) -> u64 {
        self.total_quantity_micro.saturating_sub(self.executed_quantity_micro)
    }
    
    pub fn execution_pct(&self) -> f64 {
        if self.total_quantity_micro == 0 {
            return 0.0;
        }
        self.executed_quantity_micro as f64 / self.total_quantity_micro as f64
    }
}

/// Execution Router - splits parent orders across venues
pub struct ExecutionRouter {
    /// Pending parent orders
    parent_orders: RwLock<Vec<ParentOrder>>,
    /// Generated child orders queue
    child_queue: RwLock<VecDeque<ChildOrder>>,
    /// Order ID counter
    order_counter: AtomicU64,
    /// Running flag
    running: AtomicBool,
    /// Total orders routed
    orders_routed: AtomicU64,
    /// Default venues for routing
    default_venues: Vec<u8>,
}

impl ExecutionRouter {
    pub fn new(default_venues: Vec<u8>) -> Self {
        Self {
            parent_orders: RwLock::new(Vec::with_capacity(100)),
            child_queue: RwLock::new(VecDeque::with_capacity(1000)),
            order_counter: AtomicU64::new(1),
            running: AtomicBool::new(true),
            orders_routed: AtomicU64::new(0),
            default_venues,
        }
    }
    
    /// Generate new unique order ID
    #[inline]
    fn next_order_id(&self) -> u64 {
        self.order_counter.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Submit a parent order for execution
    pub fn submit_parent_order(&self, order: ParentOrder) -> u64 {
        let order_id = order.order_id;
        let mut orders = self.parent_orders.write();
        orders.push(order);
        order_id
    }
    
    /// Split parent order into child orders based on algorithm
    pub fn split_order(&self, parent: &ParentOrder, current_prices: &[(u8, u64)]) -> Vec<ChildOrder> {
        let remaining = parent.remaining();
        if remaining == 0 || parent.venues.is_empty() {
            return Vec::new();
        }
        
        match parent.algo {
            ExecAlgo::Twap => self.split_twap(parent, remaining),
            ExecAlgo::Vwap => self.split_vwap(parent, remaining, current_prices),
            ExecAlgo::Iceberg => self.split_iceberg(parent, remaining),
            ExecAlgo::Sniper => self.split_sniper(parent, remaining),
        }
    }
    
    /// TWAP: Split evenly across time intervals
    fn split_twap(&self, parent: &ParentOrder, remaining: u64) -> Vec<ChildOrder> {
        let num_slices = parent.time_horizon_ms / 1000; // 1 slice per second
        let num_slices = num_slices.max(1).min(100) as usize;
        
        let qty_per_slice = remaining / num_slices as u64;
        let venues = if parent.venues.is_empty() {
            &self.default_venues
        } else {
            &parent.venues
        };
        
        let mut children = Vec::with_capacity(num_slices);
        
        for i in 0..num_slices {
            let venue_idx = i % venues.len();
            children.push(ChildOrder {
                order_id: self.next_order_id(),
                parent_order_id: parent.order_id,
                venue_id: venues[venue_idx],
                side: parent.side,
                price_micro: 0, // Market order for TWAP
                quantity_micro: qty_per_slice,
                algo: ExecAlgo::Twap,
                sequence: i,
                total_sequences: num_slices,
            });
        }
        
        children
    }
    
    /// VWAP: Split based on historical volume distribution
    fn split_vwap(
        &self,
        parent: &ParentOrder,
        remaining: u64,
        current_prices: &[(u8, u64)],
    ) -> Vec<ChildOrder> {
        // Simplified VWAP: distribute based on venue liquidity
        let venues = if parent.venues.is_empty() {
            &self.default_venues
        } else {
            &parent.venues
        };
        
        if venues.is_empty() {
            return Vec::new();
        }
        
        // Equal distribution for now (would use volume profile in production)
        let qty_per_venue = remaining / venues.len() as u64;
        
        let mut children = Vec::with_capacity(venues.len());
        
        for (i, &venue_id) in venues.iter().enumerate() {
            // Find price for this venue
            let price = current_prices
                .iter()
                .find(|(vid, _)| *vid == venue_id)
                .map(|(_, p)| *p)
                .unwrap_or(0);
            
            children.push(ChildOrder {
                order_id: self.next_order_id(),
                parent_order_id: parent.order_id,
                venue_id,
                side: parent.side,
                price_micro: price,
                quantity_micro: qty_per_venue,
                algo: ExecAlgo::Vwap,
                sequence: i,
                total_sequences: venues.len(),
            });
        }
        
        children
    }
    
    /// Iceberg: Show only small portion of total order
    fn split_iceberg(&self, parent: &ParentOrder, remaining: u64) -> Vec<ChildOrder> {
        // Iceberg shows 10% of order at a time, min 0.01 units
        let display_qty = (remaining / 10).max(10_000); // 0.01 in micro-units
        
        let venues = if parent.venues.is_empty() {
            &self.default_venues
        } else {
            &parent.venues
        };
        
        let mut children = Vec::new();
        
        // Create first visible slice
        if let Some(&venue_id) = venues.first() {
            children.push(ChildOrder {
                order_id: self.next_order_id(),
                parent_order_id: parent.order_id,
                venue_id,
                side: parent.side,
                price_micro: 0, // Will be set by caller
                quantity_micro: display_qty,
                algo: ExecAlgo::Iceberg,
                sequence: 0,
                total_sequences: ((remaining + display_qty - 1) / display_qty) as usize,
            });
        }
        
        children
    }
    
    /// Sniper: Create single aggressive order
    fn split_sniper(&self, parent: &ParentOrder, remaining: u64) -> Vec<ChildOrder> {
        let venues = if parent.venues.is_empty() {
            &self.default_venues
        } else {
            &parent.venues
        };
        
        let mut children = Vec::with_capacity(venues.len());
        
        // Send to all venues simultaneously for fastest fill
        for (i, &venue_id) in venues.iter().enumerate() {
            children.push(ChildOrder {
                order_id: self.next_order_id(),
                parent_order_id: parent.order_id,
                venue_id,
                side: parent.side,
                price_micro: u64::MAX, // Aggressive market order
                quantity_micro: remaining / venues.len() as u64,
                algo: ExecAlgo::Sniper,
                sequence: i,
                total_sequences: venues.len(),
            });
        }
        
        children
    }
    
    /// Queue child orders for execution
    pub fn queue_children(&self, children: Vec<ChildOrder>) {
        let mut queue = self.child_queue.write();
        for child in children {
            queue.push_back(child);
            self.orders_routed.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Pop next child order to execute
    pub fn pop_next_child(&self) -> Option<ChildOrder> {
        let mut queue = self.child_queue.write();
        queue.pop_front()
    }
    
    /// Update parent order with execution
    pub fn update_execution(&self, parent_order_id: u64, executed_qty_micro: u64) {
        let mut orders = self.parent_orders.write();
        for order in orders.iter_mut() {
            if order.order_id == parent_order_id {
                order.executed_quantity_micro = order
                    .executed_quantity_micro
                    .saturating_add(executed_qty_micro);
                break;
            }
        }
    }
    
    /// Get pending parent orders count
    pub fn pending_count(&self) -> usize {
        self.parent_orders.read().len()
    }
    
    /// Get queued child orders count
    pub fn queued_count(&self) -> usize {
        self.child_queue.read().len()
    }
    
    /// Total orders routed
    pub fn total_routed(&self) -> u64 {
        self.orders_routed.load(Ordering::Relaxed)
    }
    
    /// Stop routing
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
    
    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_twap_split() {
        let router = ExecutionRouter::new(vec![1, 2, 3]);
        
        let parent = ParentOrder {
            order_id: 1,
            symbol_hash: 12345,
            side: Side::Buy,
            total_quantity_micro: 1_000_000, // 1.0 BTC
            executed_quantity_micro: 0,
            algo: ExecAlgo::Twap,
            time_horizon_ms: 10_000, // 10 seconds
            max_participation_rate: 0.1,
            venues: vec![1, 2],
            created_at_ns: 0,
        };
        
        let children = router.split_order(&parent, &[]);
        assert!(!children.is_empty());
        assert_eq!(children[0].algo, ExecAlgo::Twap);
    }
    
    #[test]
    fn test_iceberg_split() {
        let router = ExecutionRouter::new(vec![1]);
        
        let parent = ParentOrder {
            order_id: 2,
            symbol_hash: 12345,
            side: Side::Sell,
            total_quantity_micro: 10_000_000, // 10 BTC
            executed_quantity_micro: 0,
            algo: ExecAlgo::Iceberg,
            time_horizon_ms: 60_000,
            max_participation_rate: 0.05,
            venues: vec![1],
            created_at_ns: 0,
        };
        
        let children = router.split_order(&parent, &[]);
        assert!(!children.is_empty());
        assert_eq!(children[0].algo, ExecAlgo::Iceberg);
        // Display qty should be ~10% of total
        assert!(children[0].quantity_micro < parent.total_quantity_micro / 5);
    }
}
