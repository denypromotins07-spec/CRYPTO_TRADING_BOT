//! backend/venues/sor_analytics.rs
//! 
//! ZAID PERSONAL CRYPTO TRADING BOT - Stage 33: Multi-Venue SOR
//! Chapter 4: SOR Analytics and TCA
//! 
//! Measures fill rates and slippage per venue for Transaction Cost Analysis (TCA).
//! Tracks execution quality metrics for smart order routing optimization.
//! Uses zero-cost abstractions for maximum throughput on AMD Ryzen AI 5.
//! Strictly respects 8GB RAM limit.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};

/// Execution quality metrics for a venue
#[derive(Clone, Debug)]
pub struct VenueExecutionMetrics {
    pub venue_id: u8,
    pub total_orders: u64,
    pub filled_orders: u64,
    pub partially_filled: u64,
    pub cancelled_orders: u64,
    pub rejected_orders: u64,
    pub total_volume_micro: u64,
    pub total_slippage_bps: f64,
    pub avg_fill_time_us: f64,
    pub p95_fill_time_us: f64,
    pub p99_fill_time_us: f64,
    pub maker_fill_pct: f64,
    pub taker_fill_pct: f64,
}

impl VenueExecutionMetrics {
    pub fn new(venue_id: u8) -> Self {
        Self {
            venue_id,
            total_orders: 0,
            filled_orders: 0,
            partially_filled: 0,
            cancelled_orders: 0,
            rejected_orders: 0,
            total_volume_micro: 0,
            total_slippage_bps: 0.0,
            avg_fill_time_us: 0.0,
            p95_fill_time_us: 0.0,
            p99_fill_time_us: 0.0,
            maker_fill_pct: 0.0,
            taker_fill_pct: 0.0,
        }
    }
    
    /// Calculate fill rate
    pub fn fill_rate(&self) -> f64 {
        if self.total_orders == 0 {
            return 0.0;
        }
        (self.filled_orders as f64) / (self.total_orders as f64) * 100.0
    }
    
    /// Calculate average slippage in bps
    pub fn avg_slippage_bps(&self) -> f64 {
        if self.filled_orders == 0 {
            return 0.0;
        }
        self.total_slippage_bps / (self.filled_orders as f64)
    }
    
    /// Calculate rejection rate
    pub fn rejection_rate(&self) -> f64 {
        if self.total_orders == 0 {
            return 0.0;
        }
        (self.rejected_orders as f64) / (self.total_orders as f64) * 100.0
    }
}

/// Single order execution record
#[derive(Clone, Debug)]
pub struct ExecutionRecord {
    pub order_id: u64,
    pub venue_id: u8,
    pub symbol_hash: u64,
    pub side: Side,
    pub requested_qty_micro: u64,
    pub filled_qty_micro: u64,
    pub requested_price_micro: u64,
    pub avg_fill_price_micro: u64,
    pub fill_time_us: u64,
    pub slippage_bps: f64,
    is_maker: bool,
    status: OrderStatus,
    timestamp_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// SOR Analytics Engine - tracks execution quality
pub struct SorAnalytics {
    /// Per-venue metrics
    venue_metrics: RwLock<HashMap<u8, VenueExecutionMetrics>>,
    /// Recent execution records (circular buffer)
    execution_history: RwLock<VecDeque<ExecutionRecord>>,
    /// Max history size (memory control)
    max_history: usize,
    /// Total orders tracked
    total_orders: AtomicU64,
    /// Total volume tracked
    total_volume_micro: AtomicU64,
    /// Routing decisions count
    routing_decisions: AtomicU64,
}

impl SorAnalytics {
    pub fn new(max_history: usize) -> Self {
        Self {
            venue_metrics: RwLock::new(HashMap::with_capacity(8)),
            execution_history: RwLock::new(VecDeque::with_capacity(max_history)),
            max_history,
            total_orders: AtomicU64::new(0),
            total_volume_micro: AtomicU64::new(0),
            routing_decisions: AtomicU64::new(0),
        }
    }
    
    /// Register a venue for tracking
    pub fn register_venue(&self, venue_id: u8) {
        let mut metrics = self.venue_metrics.write();
        metrics.entry(venue_id).or_insert_with(|| VenueExecutionMetrics::new(venue_id));
    }
    
    /// Record an order submission
    pub fn record_order_submission(
        &self,
        venue_id: u8,
        order_id: u64,
        symbol_hash: u64,
        side: Side,
        qty_micro: u64,
        price_micro: u64,
    ) {
        self.total_orders.fetch_add(1, Ordering::Relaxed);
        self.total_volume_micro.fetch_add(qty_micro, Ordering::Relaxed);
        
        // Update venue metrics
        let mut metrics = self.venue_metrics.write();
        if let Some(m) = metrics.get_mut(&venue_id) {
            m.total_orders += 1;
            m.total_volume_micro += qty_micro;
        }
    }
    
    /// Record an order fill
    pub fn record_fill(
        &self,
        venue_id: u8,
        order_id: u64,
        symbol_hash: u64,
        side: Side,
        requested_qty_micro: u64,
        filled_qty_micro: u64,
        requested_price_micro: u64,
        avg_fill_price_micro: u64,
        fill_time_us: u64,
        is_maker: bool,
    ) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Calculate slippage
        let slippage_bps = self.calculate_slippage(
            side,
            requested_price_micro,
            avg_fill_price_micro,
        );
        
        let status = if filled_qty_micro >= requested_qty_micro {
            OrderStatus::Filled
        } else {
            OrderStatus::PartiallyFilled
        };
        
        let record = ExecutionRecord {
            order_id,
            venue_id,
            symbol_hash,
            side,
            requested_qty_micro,
            filled_qty_micro,
            requested_price_micro,
            avg_fill_price_micro,
            fill_time_us,
            slippage_bps,
            is_maker,
            status,
            timestamp_ns: now_ns,
        };
        
        // Add to history
        {
            let mut history = self.execution_history.write();
            history.push_back(record);
            while history.len() > self.max_history {
                history.pop_front();
            }
        }
        
        // Update venue metrics
        let mut metrics = self.venue_metrics.write();
        if let Some(m) = metrics.get_mut(&venue_id) {
            match status {
                OrderStatus::Filled => m.filled_orders += 1,
                OrderStatus::PartiallyFilled => m.partially_filled += 1,
                _ => {}
            }
            m.total_slippage_bps += slippage_bps;
            
            // Recalculate averages and percentiles
            self.recalculate_fill_stats(&mut m);
        }
    }
    
    /// Record order cancellation
    pub fn record_cancel(&self, venue_id: u8) {
        let mut metrics = self.venue_metrics.write();
        if let Some(m) = metrics.get_mut(&venue_id) {
            m.cancelled_orders += 1;
        }
    }
    
    /// Record order rejection
    pub fn record_reject(&self, venue_id: u8) {
        let mut metrics = self.venue_metrics.write();
        if let Some(m) = metrics.get_mut(&venue_id) {
            m.rejected_orders += 1;
        }
    }
    
    /// Calculate slippage in basis points
    fn calculate_slippage(
        &self,
        side: Side,
        requested_price: u64,
        fill_price: u64,
    ) -> f64 {
        if requested_price == 0 || fill_price == 0 {
            return 0.0;
        }
        
        match side {
            Side::Buy => {
                // For buys, slippage is positive if fill price > requested price
                if fill_price > requested_price {
                    ((fill_price - requested_price) as f64) / (requested_price as f64) * 10_000.0
                } else {
                    0.0 // Negative slippage (better fill) = 0 for reporting
                }
            }
            Side::Sell => {
                // For sells, slippage is positive if fill price < requested price
                if fill_price < requested_price {
                    ((requested_price - fill_price) as f64) / (requested_price as f64) * 10_000.0
                } else {
                    0.0
                }
            }
        }
    }
    
    /// Recalculate fill statistics for a venue
    fn recalculate_fill_stats(&self, metrics: &mut VenueExecutionMetrics) {
        let history = self.execution_history.read();
        
        // Filter to this venue
        let venue_records: Vec<&ExecutionRecord> = history
            .iter()
            .filter(|r| r.venue_id == metrics.venue_id)
            .collect();
        
        if venue_records.is_empty() {
            return;
        }
        
        // Calculate average fill time
        let total_fill_time: u64 = venue_records.iter().map(|r| r.fill_time_us).sum();
        metrics.avg_fill_time_us = total_fill_time as f64 / venue_records.len() as f64;
        
        // Calculate percentiles
        let mut fill_times: Vec<u64> = venue_records.iter().map(|r| r.fill_time_us).collect();
        fill_times.sort_unstable();
        
        metrics.p95_fill_time_us = percentile(&fill_times, 95);
        metrics.p99_fill_time_us = percentile(&fill_times, 99);
        
        // Calculate maker/taker ratio
        let maker_count = venue_records.iter().filter(|r| r.is_maker).count();
        metrics.maker_fill_pct = (maker_count as f64) / (venue_records.len() as f64) * 100.0;
        metrics.taker_fill_pct = 100.0 - metrics.maker_fill_pct;
    }
    
    /// Get metrics for a venue
    pub fn get_venue_metrics(&self, venue_id: u8) -> Option<VenueExecutionMetrics> {
        let metrics = self.venue_metrics.read();
        metrics.get(&venue_id).cloned()
    }
    
    /// Get all venue metrics
    pub fn get_all_metrics(&self) -> HashMap<u8, VenueExecutionMetrics> {
        let metrics = self.venue_metrics.read();
        metrics.clone()
    }
    
    /// Find best venue by fill rate
    pub fn best_venue_by_fill_rate(&self) -> Option<u8> {
        let metrics = self.venue_metrics.read();
        metrics.iter()
            .max_by(|a, b| a.1.fill_rate().partial_cmp(&b.1.fill_rate()).unwrap())
            .map(|(&id, _)| id)
    }
    
    /// Find best venue by lowest slippage
    pub fn best_venue_by_slippage(&self) -> Option<u8> {
        let metrics = self.venue_metrics.read();
        metrics.iter()
            .min_by(|a, b| a.1.avg_slippage_bps().partial_cmp(&b.1.avg_slippage_bps()).unwrap())
            .map(|(&id, _)| id)
    }
    
    /// Find best venue by fastest fill time
    pub fn best_venue_by_speed(&self) -> Option<u8> {
        let metrics = self.venue_metrics.read();
        metrics.iter()
            .min_by(|a, b| a.1.avg_fill_time_us.partial_cmp(&b.1.avg_fill_time_us).unwrap())
            .map(|(&id, _)| id)
    }
    
    /// Record a routing decision (for analysis)
    pub fn record_routing_decision(&self) {
        self.routing_decisions.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get total orders tracked
    pub fn total_orders(&self) -> u64 {
        self.total_orders.load(Ordering::Relaxed)
    }
    
    /// Get total volume tracked
    pub fn total_volume(&self) -> u64 {
        self.total_volume_micro.load(Ordering::Relaxed)
    }
    
    /// Get routing decisions count
    pub fn routing_decisions(&self) -> u64 {
        self.routing_decisions.load(Ordering::Relaxed)
    }
}

impl Default for SorAnalytics {
    fn default() -> Self {
        Self::new(10000)
    }
}

/// Calculate percentile of a sorted slice
fn percentile(values: &[u64], pct: u64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let idx = ((pct as f64 / 100.0) * (values.len() - 1) as f64) as usize;
    values[idx.min(values.len() - 1)] as f64
}

// Zero-cost abstraction: compile-time venue IDs
pub mod venue_ids {
    pub const BINANCE: u8 = 1;
    pub const COINBASE: u8 = 2;
    pub const KRAKEN: u8 = 3;
    pub const BYBIT: u8 = 4;
    pub const OKX: u8 = 5;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_execution_tracking() {
        let analytics = SorAnalytics::new(100);
        analytics.register_venue(venue_ids::BINANCE);
        
        // Record order submission
        analytics.record_order_submission(
            venue_ids::BINANCE,
            1,
            12345,
            Side::Buy,
            1_000_000,
            50_000_000_000,
        );
        
        // Record fill with slight slippage
        analytics.record_fill(
            venue_ids::BINANCE,
            1,
            12345,
            Side::Buy,
            1_000_000,
            1_000_000,
            50_000_000_000,
            50_010_000_000, // 10 ticks worse
            5000,          // 5ms fill time
            true,          // Maker
        );
        
        let metrics = analytics.get_venue_metrics(venue_ids::BINANCE).unwrap();
        
        assert_eq!(metrics.total_orders, 1);
        assert_eq!(metrics.filled_orders, 1);
        assert!(metrics.fill_rate() > 99.0);
        assert!(metrics.avg_slippage_bps() > 0.0);
    }
    
    #[test]
    fn test_best_venue_selection() {
        let analytics = SorAnalytics::new(100);
        analytics.register_venue(venue_ids::BINANCE);
        analytics.register_venue(venue_ids::COINBASE);
        
        // Simulate better fills on Binance
        for i in 0..10 {
            analytics.record_fill(
                venue_ids::BINANCE,
                i,
                12345,
                Side::Buy,
                1_000_000,
                1_000_000,
                50_000_000_000,
                50_000_000_000, // No slippage
                1000,           // Fast
                true,
            );
        }
        
        // Simulate worse fills on Coinbase
        for i in 0..5 {
            analytics.record_fill(
                venue_ids::COINBASE,
                i + 100,
                12345,
                Side::Buy,
                1_000_000,
                1_000_000,
                50_000_000_000,
                50_050_000_000, // More slippage
                5000,           // Slower
                false,
            );
        }
        
        let best_by_slippage = analytics.best_venue_by_slippage();
        assert_eq!(best_by_slippage, Some(venue_ids::BINANCE));
        
        let best_by_speed = analytics.best_venue_by_speed();
        assert_eq!(best_by_speed, Some(venue_ids::BINANCE));
    }
}
