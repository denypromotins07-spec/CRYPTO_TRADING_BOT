//! ZAID PERSONAL CRYPTO TRADING BOT - Stage 32
//! Advanced Order Management System (OMS) - Chapter 1
//! File: backend/oms/synthetic_router.rs
//!
//! Routes synthetic orders to Binance as atomic micro-children.
//! Ensures parent synthetic orders are split into optimal child orders
//! for minimal market impact and fee optimization.
//! Optimized for AMD Ryzen AI 5, strictly respecting 8GB RAM limit.
//! Targets 8k-20k INR/hour in a 4hr trading window.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::collections::VecDeque;

/// Types of synthetic orders supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticOrderType {
    /// Large order split into multiple smaller children
    Iceberg,
    /// Order executed over time using TWAP algorithm
    Twap,
    /// Order executed based on volume participation
    Vwap,
    /// Sniper order for immediate execution at specific price
    Sniper,
    /// Bracket order with TP/SL
    Bracket,
}

/// Status of a synthetic order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticStatus {
    Pending,
    Active,
    PartiallyFilled,
    FullyFilled,
    Cancelled,
    Failed,
}

/// Configuration for splitting synthetic orders
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Minimum child order size
    pub min_child_size: f64,
    /// Maximum child order size
    pub max_child_size: f64,
    /// Target number of children (if possible)
    pub target_children: usize,
    /// Maximum deviation from VWAP for execution
    pub vwap_tolerance_pct: f64,
    /// Time horizon for execution in milliseconds
    pub time_horizon_ms: u64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        SplitConfig {
            min_child_size: 0.001, // Minimum BTC equivalent
            max_child_size: 0.1,   // Maximum per child to reduce impact
            target_children: 10,
            vwap_tolerance_pct: 0.05, // 0.05% tolerance
            time_horizon_ms: 60000,   // 1 minute default
        }
    }
}

/// Represents a child order of a synthetic parent
#[derive(Debug, Clone)]
pub struct ChildOrder {
    /// Unique child order ID
    pub child_id: u64,
    /// Parent synthetic order ID
    pub parent_id: u64,
    /// Child sequence number
    pub sequence: usize,
    /// Symbol to trade
    pub symbol: String,
    /// Side (BUY/SELL)
    pub side: String,
    /// Order type for child (LIMIT/MARKET)
    pub order_type: String,
    /// Price (0 for market orders)
    pub price: f64,
    /// Quantity for this child
    pub quantity: f64,
    /// Filled quantity
    pub filled_quantity: f64,
    /// Current status
    pub status: SyntheticStatus,
    /// Exchange order ID (once submitted)
    pub exchange_order_id: Option<u64>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
}

impl ChildOrder {
    pub fn new(
        parent_id: u64,
        sequence: usize,
        symbol: String,
        side: String,
        order_type: String,
        price: f64,
        quantity: f64,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        ChildOrder {
            child_id: now.wrapping_add(sequence as u64),
            parent_id,
            sequence,
            symbol,
            side,
            order_type,
            price,
            quantity,
            filled_quantity: 0.0,
            status: SyntheticStatus::Pending,
            exchange_order_id: None,
            created_at: now,
            updated_at: now,
        }
    }

    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status,
            SyntheticStatus::FullyFilled | SyntheticStatus::Cancelled | SyntheticStatus::Failed
        )
    }

    #[inline]
    pub fn remaining_quantity(&self) -> f64 {
        (self.quantity - self.filled_quantity).max(0.0)
    }
}

/// Synthetic order that will be split into atomic micro-children
pub struct SyntheticOrder {
    /// Unique synthetic order ID
    pub order_id: u64,
    /// Type of synthetic order
    pub order_type: SyntheticOrderType,
    /// Current status
    status: AtomicU64,
    /// Symbol to trade
    pub symbol: String,
    /// Side (BUY/SELL)
    pub side: String,
    /// Total quantity to execute
    pub total_quantity: f64,
    /// Executed quantity across all children
    executed_quantity: AtomicU64, // Stored as fixed-point for atomicity
    /// Split configuration
    pub config: SplitConfig,
    /// Child orders queue
    children: dashmap::DashMap<u64, ChildOrder>,
    /// Next child sequence number
    next_sequence: AtomicU64,
    /// Flag indicating if all children have been generated
    all_children_generated: AtomicBool,
    /// Execution start time
    started_at: Option<u64>,
    /// Execution end time
    completed_at: Option<u64>,
}

impl SyntheticOrder {
    /// Create a new synthetic order
    pub fn new(
        order_type: SyntheticOrderType,
        symbol: String,
        side: String,
        total_quantity: f64,
        config: SplitConfig,
    ) -> Self {
        let order_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        SyntheticOrder {
            order_id,
            order_type,
            status: AtomicU64::new(SyntheticStatus::Pending as u64),
            symbol,
            side,
            total_quantity,
            executed_quantity: AtomicU64::new(0),
            config,
            children: dashmap::DashMap::new(),
            next_sequence: AtomicU64::new(0),
            all_children_generated: AtomicBool::new(false),
            started_at: None,
            completed_at: None,
        }
    }

    #[inline]
    pub fn get_status(&self) -> SyntheticStatus {
        unsafe { std::mem::transmute(self.status.load(Ordering::Acquire)) }
    }

    fn set_status(&self, status: SyntheticStatus) {
        self.status.store(status as u64, Ordering::Release);
    }

    /// Generate child orders based on the synthetic order type
    pub fn generate_children(&self) -> Vec<ChildOrder> {
        let mut children = Vec::new();
        
        match self.order_type {
            SyntheticOrderType::Iceberg => {
                self.generate_iceberg_children(&mut children);
            }
            SyntheticOrderType::Twap => {
                self.generate_twap_children(&mut children);
            }
            SyntheticOrderType::Vwap => {
                self.generate_vwap_children(&mut children);
            }
            SyntheticOrderType::Sniper => {
                // Sniper orders are single child, no splitting
                self.generate_sniper_child(&mut children);
            }
            SyntheticOrderType::Bracket => {
                // Bracket orders handled separately
                return children;
            }
        }

        // Register children
        for child in &children {
            self.children.insert(child.child_id, child.clone());
        }

        self.all_children_generated.store(true, Ordering::Release);
        children
    }

    /// Generate iceberg order children
    fn generate_iceberg_children(&self, children: &mut Vec<ChildOrder>) {
        let mut remaining = self.total_quantity;
        let mut seq = 0;

        while remaining > self.config.min_child_size {
            let child_qty = remaining.min(self.config.max_child_size);
            
            let child = ChildOrder::new(
                self.order_id,
                seq,
                self.symbol.clone(),
                self.side.clone(),
                "LIMIT".to_string(),
                0.0, // Price determined at execution time
                child_qty,
            );

            children.push(child);
            remaining -= child_qty;
            seq += 1;

            // Safety limit on number of children
            if seq >= 100 {
                break;
            }
        }
    }

    /// Generate TWAP order children
    fn generate_twap_children(&self, children: &mut Vec<ChildOrder>) {
        let num_children = self.config.target_children.min(
            (self.total_quantity / self.config.min_child_size) as usize
        );

        let qty_per_child = self.total_quantity / num_children as f64;

        for seq in 0..num_children {
            let child = ChildOrder::new(
                self.order_id,
                seq,
                self.symbol.clone(),
                self.side.clone(),
                "LIMIT".to_string(),
                0.0,
                qty_per_child,
            );
            children.push(child);
        }
    }

    /// Generate VWAP order children
    fn generate_vwap_children(&self, children: &mut Vec<ChildOrder>) {
        // VWAP children are generated dynamically based on volume profile
        // For now, use similar approach to TWAP with volume weighting
        self.generate_twap_children(children);
    }

    /// Generate sniper child (single order)
    fn generate_sniper_child(&self, children: &mut Vec<ChildOrder>) {
        let child = ChildOrder::new(
            self.order_id,
            0,
            self.symbol.clone(),
            self.side.clone(),
            "MARKET".to_string(),
            0.0,
            self.total_quantity,
        );
        children.push(child);
    }

    /// Update child order fill status
    pub fn update_child_fill(&self, child_id: u64, filled_qty: f64) -> bool {
        if let Some(mut child) = self.children.get_mut(&child_id) {
            child.filled_quantity = filled_qty;
            child.updated_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // Update status
            if child.filled_quantity >= child.quantity {
                child.status = SyntheticStatus::FullyFilled;
            } else if child.filled_quantity > 0.0 {
                child.status = SyntheticStatus::PartiallyFilled;
            }

            // Update total executed quantity atomically
            let total_executed = self.executed_quantity.fetch_add(
                (filled_qty * 1e8) as u64, // Fixed-point conversion
                Ordering::AcqRel,
            );

            // Check if parent is complete
            let total_now = ((total_executed + (filled_qty * 1e8) as u64) as f64) / 1e8;
            if total_now >= self.total_quantity * 0.999 {
                self.set_status(SyntheticStatus::FullyFilled);
                self.completed_at = Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                );
            } else if total_now > 0.0 {
                self.set_status(SyntheticStatus::PartiallyFilled);
            }

            true
        } else {
            false
        }
    }

    /// Get next pending child for execution
    pub fn get_next_pending_child(&self) -> Option<ChildOrder> {
        for entry in self.children.iter() {
            if entry.value().status == SyntheticStatus::Pending {
                return Some(entry.value().clone());
            }
        }
        None
    }

    /// Get all children
    pub fn get_all_children(&self) -> Vec<ChildOrder> {
        self.children.iter().map(|e| e.value().clone()).collect()
    }

    /// Calculate execution progress percentage
    pub fn execution_progress(&self) -> f64 {
        let executed = self.executed_quantity.load(Ordering::Acquire) as f64 / 1e8;
        (executed / self.total_quantity * 100.0).min(100.0)
    }

    /// Start execution timer
    pub fn start_execution(&mut self) {
        self.started_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        );
        self.set_status(SyntheticStatus::Active);
    }

    /// Check if execution has exceeded time horizon
    pub fn is_timeout(&self) -> bool {
        if let Some(started) = self.started_at {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            now - started > self.config.time_horizon_ms
        } else {
            false
        }
    }
}

/// Router that manages synthetic orders and routes children to exchange
pub struct SyntheticRouter {
    /// Active synthetic orders
    orders: dashmap::DashMap<u64, Arc<SyntheticOrder>>,
    /// Queue for pending child orders awaiting routing
    pending_queue: dashmap::DashMap<u64, VecDeque<ChildOrder>>,
    /// Maximum concurrent synthetic orders
    max_orders: usize,
    /// Flag to pause routing
    paused: AtomicBool,
}

impl SyntheticRouter {
    pub fn new(max_orders: usize) -> Self {
        SyntheticRouter {
            orders: dashmap::DashMap::new(),
            pending_queue: dashmap::DashMap::new(),
            max_orders,
            paused: AtomicBool::new(false),
        }
    }

    /// Submit a new synthetic order
    pub fn submit_synthetic(&self, order: SyntheticOrder) -> Result<u64, &'static str> {
        if self.orders.len() >= self.max_orders {
            return Err("Maximum synthetic orders reached");
        }

        let order_arc = Arc::new(order);
        let order_id = order_arc.order_id;

        // Generate children immediately
        let children = order_arc.generate_children();

        // Add to pending queue
        let mut queue = VecDeque::new();
        for child in children {
            queue.push_back(child);
        }
        self.pending_queue.insert(order_id, queue);

        self.orders.insert(order_id, order_arc);

        Ok(order_id)
    }

    /// Get next child to route for a synthetic order
    pub fn get_next_child_to_route(&self, order_id: u64) -> Option<ChildOrder> {
        if self.paused.load(Ordering::Acquire) {
            return None;
        }

        if let Some(queue) = self.pending_queue.get(&order_id) {
            // Note: This is a simplified implementation
            // In production, would need mutable access to pop from queue
            queue.front().cloned()
        } else {
            None
        }
    }

    /// Mark child as routed (submitted to exchange)
    pub fn mark_child_routed(&self, order_id: u64, child_id: u64, exchange_id: u64) -> bool {
        if let Some(mut queue) = self.pending_queue.get_mut(&order_id) {
            if let Some(front) = queue.front() {
                if front.child_id == child_id {
                    queue.pop_front();
                    
                    // Update child with exchange ID
                    if let Some(order) = self.orders.get(&order_id) {
                        if let Some(mut child) = order.children.get_mut(&child_id) {
                            child.exchange_order_id = Some(exchange_id);
                            child.status = SyntheticStatus::Active;
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Pause all routing (emergency stop)
    pub fn pause_routing(&self) {
        self.paused.store(true, Ordering::Release);
    }

    /// Resume routing
    pub fn resume_routing(&self) {
        self.paused.store(false, Ordering::Release);
    }

    /// Cancel a synthetic order and all pending children
    pub fn cancel_synthetic(&self, order_id: u64) -> bool {
        if let Some(order) = self.orders.get(&order_id) {
            // Clear pending queue
            self.pending_queue.remove(&order_id);

            // Mark all non-completed children as cancelled
            for mut child in order.children.iter_mut() {
                if !child.is_complete() {
                    child.status = SyntheticStatus::Cancelled;
                }
            }

            order.set_status(SyntheticStatus::Cancelled);
            true
        } else {
            false
        }
    }

    /// Get synthetic order by ID
    pub fn get_synthetic(&self, order_id: u64) -> Option<Arc<SyntheticOrder>> {
        self.orders.get(&order_id).map(|r| r.value().clone())
    }

    /// Cleanup completed synthetic orders
    pub fn cleanup_completed(&self, max_age_ms: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.orders.retain(|order_id, order| {
            if let Some(completed_at) = order.completed_at {
                if now - completed_at > max_age_ms {
                    self.pending_queue.remove(order_id);
                    return false;
                }
            }
            true
        });
    }

    /// Get total pending quantity across all synthetics
    pub fn total_pending_quantity(&self, symbol: &str) -> f64 {
        let mut total = 0.0;
        for order in self.orders.iter() {
            if order.value().symbol == symbol && order.value().get_status() == SyntheticStatus::Active {
                for child in order.value().children.iter() {
                    if child.value().status == SyntheticStatus::Pending {
                        total += child.value().quantity;
                    }
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_order_creation() {
        let config = SplitConfig::default();
        let order = SyntheticOrder::new(
            SyntheticOrderType::Iceberg,
            "BTCUSDT".to_string(),
            "BUY".to_string(),
            1.0,
            config,
        );

        assert_eq!(order.order_type, SyntheticOrderType::Iceberg);
        assert_eq!(order.total_quantity, 1.0);
        assert_eq!(order.get_status(), SyntheticStatus::Pending);
    }

    #[test]
    fn test_iceberg_child_generation() {
        let config = SplitConfig {
            min_child_size: 0.1,
            max_child_size: 0.3,
            target_children: 5,
            ..Default::default()
        };

        let order = SyntheticOrder::new(
            SyntheticOrderType::Iceberg,
            "ETHUSDT".to_string(),
            "SELL".to_string(),
            1.0,
            config,
        );

        let children = order.generate_children();
        assert!(!children.is_empty());
        assert!(children.len() <= 10);

        // Verify total quantity matches
        let total: f64 = children.iter().map(|c| c.quantity).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_synthetic_router() {
        let router = SyntheticRouter::new(100);
        
        let config = SplitConfig::default();
        let order = SyntheticOrder::new(
            SyntheticOrderType::Sniper,
            "SOLUSDT".to_string(),
            "BUY".to_string(),
            10.0,
            config,
        );

        let order_id = router.submit_synthetic(order).unwrap();
        assert!(order_id > 0);

        let synthetic = router.get_synthetic(order_id).unwrap();
        assert_eq!(synthetic.symbol, "SOLUSDT");
    }
}
