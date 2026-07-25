//! Atomic Counters for PnL and Position Tracking
//! 
//! This module implements lock-free atomic counters using Rust's atomic operations
//! for high-performance PnL and position tracking without thread contention.
//! Designed for the ZAID Personal Crypto Trading Bot to achieve 8k-20k INR/hour targets.
//!
//! Key Features:
//! - Lock-free atomic operations for maximum throughput
//! - Memory ordering guarantees for correct visibility
//! - Support for decimal precision in financial calculations
//! - Thread-safe accumulation without mutex overhead
//! - Compatible with 8GB RAM constraint
//!
//! Domain Integration: Quantitative Finance Domains 97-108 (Atomic Operations, Financial Math)

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};

/// Scale factor for fixed-point arithmetic (nanounits for precision)
const SCALE_FACTOR: i64 = 1_000_000_000; // 9 decimal places

/// Convert f64 to scaled i64 for atomic operations
#[inline]
fn to_scaled(value: f64) -> i64 {
    (value * SCALE_FACTOR as f64) as i64
}

/// Convert scaled i64 back to f64
#[inline]
fn from_scaled(scaled: i64) -> f64 {
    scaled as f64 / SCALE_FACTOR as f64
}

/// Symbol identifiers for crypto assets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SymbolId {
    BTC = 0,
    SOL = 1,
    ETH = 2,
    USDT = 3,
}

impl SymbolId {
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(SymbolId::BTC),
            1 => Some(SymbolId::SOL),
            2 => Some(SymbolId::ETH),
            3 => Some(SymbolId::USDT),
            _ => None,
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            SymbolId::BTC => "BTC",
            SymbolId::SOL => "SOL",
            SymbolId::ETH => "ETH",
            SymbolId::USDT => "USDT",
        }
    }
}

/// Atomic counter for PnL tracking with nanounit precision
pub struct AtomicPnLCounter {
    /// Realized PnL in scaled units
    realized_pnl: AtomicI64,
    /// Unrealized PnL in scaled units
    unrealized_pnl: AtomicI64,
    /// Total PnL (realized + unrealized)
    total_pnl: AtomicI64,
    /// Trade count
    trade_count: AtomicU64,
    /// Win count (profitable trades)
    win_count: AtomicU64,
    /// Loss count (losing trades)
    loss_count: AtomicU64,
    /// Maximum drawdown in scaled units
    max_drawdown: AtomicI64,
    /// Peak PnL for drawdown calculation
    peak_pnl: AtomicI64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Active flag
    is_active: AtomicBool,
}

impl AtomicPnLCounter {
    /// Create a new PnL counter initialized to zero
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        Self {
            realized_pnl: AtomicI64::new(0),
            unrealized_pnl: AtomicI64::new(0),
            total_pnl: AtomicI64::new(0),
            trade_count: AtomicU64::new(0),
            win_count: AtomicU64::new(0),
            loss_count: AtomicU64::new(0),
            max_drawdown: AtomicI64::new(0),
            peak_pnl: AtomicI64::new(0),
            last_update_ns: AtomicU64::new(now),
            is_active: AtomicBool::new(true),
        }
    }
    
    /// Add realized PnL from a completed trade
    pub fn add_realized_pnl(&self, pnl: f64) -> f64 {
        let scaled_pnl = to_scaled(pnl);
        
        // Atomically add to realized PnL
        let prev_realized = self.realized_pnl.fetch_add(scaled_pnl, Ordering::SeqCst);
        let new_realized = prev_realized + scaled_pnl;
        
        // Update total
        let unrealized = self.unrealized_pnl.load(Ordering::Relaxed);
        self.total_pnl.store(new_realized + unrealized, Ordering::Release);
        
        // Update trade statistics
        self.trade_count.fetch_add(1, Ordering::Relaxed);
        
        if pnl > 0.0 {
            self.win_count.fetch_add(1, Ordering::Relaxed);
        } else if pnl < 0.0 {
            self.loss_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update peak and drawdown
        self.update_peak_and_drawdown(new_realized + unrealized);
        
        self.update_timestamp();
        
        from_scaled(new_realized)
    }
    
    /// Update unrealized PnL (mark-to-market)
    pub fn update_unrealized_pnl(&self, pnl: f64) -> f64 {
        let scaled_pnl = to_scaled(pnl);
        
        self.unrealized_pnl.store(scaled_pnl, Ordering::Release);
        
        // Update total
        let realized = self.realized_pnl.load(Ordering::Relaxed);
        self.total_pnl.store(realized + scaled_pnl, Ordering::Release);
        
        // Update peak and drawdown
        self.update_peak_and_drawdown(realized + scaled_pnl);
        
        self.update_timestamp();
        
        pnl
    }
    
    /// Update peak PnL and maximum drawdown
    #[inline]
    fn update_peak_and_drawdown(&self, current_pnl: i64) {
        // Update peak (CAS loop for correctness)
        let mut peak = self.peak_pnl.load(Ordering::Acquire);
        while current_pnl > peak {
            match self.peak_pnl.compare_exchange_weak(
                peak,
                current_pnl,
                Ordering::SeqCst,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(prev_peak) => peak = prev_peak,
            }
        }
        
        // Calculate and update drawdown
        let drawdown = peak - current_pnl;
        if drawdown > 0 {
            let mut max_dd = self.max_drawdown.load(Ordering::Acquire);
            while drawdown > max_dd {
                match self.max_drawdown.compare_exchange_weak(
                    max_dd,
                    drawdown,
                    Ordering::SeqCst,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(prev_max) => max_dd = prev_max,
                }
            }
        }
    }
    
    /// Get current realized PnL
    pub fn get_realized_pnl(&self) -> f64 {
        from_scaled(self.realized_pnl.load(Ordering::Acquire))
    }
    
    /// Get current unrealized PnL
    pub fn get_unrealized_pnl(&self) -> f64 {
        from_scaled(self.unrealized_pnl.load(Ordering::Acquire))
    }
    
    /// Get total PnL
    pub fn get_total_pnl(&self) -> f64 {
        from_scaled(self.total_pnl.load(Ordering::Acquire))
    }
    
    /// Get trade count
    pub fn get_trade_count(&self) -> u64 {
        self.trade_count.load(Ordering::Relaxed)
    }
    
    /// Get win rate
    pub fn get_win_rate(&self) -> f64 {
        let wins = self.win_count.load(Ordering::Relaxed);
        let total = self.trade_count.load(Ordering::Relaxed);
        
        if total == 0 {
            return 0.0;
        }
        
        wins as f64 / total as f64
    }
    
    /// Get maximum drawdown
    pub fn get_max_drawdown(&self) -> f64 {
        from_scaled(self.max_drawdown.load(Ordering::Acquire))
    }
    
    /// Get PnL snapshot
    pub fn get_snapshot(&self) -> PnLSnapshot {
        PnLSnapshot {
            realized_pnl: self.get_realized_pnl(),
            unrealized_pnl: self.get_unrealized_pnl(),
            total_pnl: self.get_total_pnl(),
            trade_count: self.get_trade_count(),
            win_rate: self.get_win_rate(),
            max_drawdown: self.get_max_drawdown(),
            timestamp_ns: self.last_update_ns.load(Ordering::Acquire),
        }
    }
    
    /// Update timestamp
    #[inline]
    fn update_timestamp(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_update_ns.store(now, Ordering::Release);
    }
    
    /// Reset all counters
    pub fn reset(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        self.realized_pnl.store(0, Ordering::Release);
        self.unrealized_pnl.store(0, Ordering::Release);
        self.total_pnl.store(0, Ordering::Release);
        self.trade_count.store(0, Ordering::Release);
        self.win_count.store(0, Ordering::Release);
        self.loss_count.store(0, Ordering::Release);
        self.max_drawdown.store(0, Ordering::Release);
        self.peak_pnl.store(0, Ordering::Release);
        self.last_update_ns.store(now, Ordering::Release);
    }
    
    /// Deactivate the counter
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
    }
    
    /// Check if counter is active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }
}

impl Default for AtomicPnLCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of PnL state at a point in time
#[derive(Debug, Clone)]
pub struct PnLSnapshot {
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub trade_count: u64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub timestamp_ns: u64,
}

/// Atomic position tracker for a single symbol
pub struct AtomicPosition {
    /// Symbol identifier
    symbol_id: SymbolId,
    /// Current position size (positive = long, negative = short)
    size: AtomicI64,
    /// Average entry price (scaled)
    avg_entry_price: AtomicI64,
    /// Total quantity bought
    total_bought: AtomicU64,
    /// Total quantity sold
    total_sold: AtomicU64,
    /// Position open timestamp
    open_timestamp_ns: AtomicU64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Active flag
    is_open: AtomicBool,
}

impl AtomicPosition {
    /// Create a new position tracker
    pub fn new(symbol_id: SymbolId) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        Self {
            symbol_id,
            size: AtomicI64::new(0),
            avg_entry_price: AtomicI64::new(0),
            total_bought: AtomicU64::new(0),
            total_sold: AtomicU64::new(0),
            open_timestamp_ns: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(now),
            is_open: AtomicBool::new(false),
        }
    }
    
    /// Add to position (buy)
    pub fn add_to_position(&self, quantity: f64, price: f64) -> f64 {
        let scaled_qty = to_scaled(quantity);
        let scaled_price = to_scaled(price);
        
        // Update total bought
        self.total_bought.fetch_add(scaled_qty as u64, Ordering::Relaxed);
        
        // Update average entry price
        let current_size = self.size.load(Ordering::Acquire);
        let current_avg = self.avg_entry_price.load(Ordering::Acquire);
        
        let new_size = current_size + scaled_qty;
        let new_avg = if new_size != 0 {
            ((current_size * current_avg) + (scaled_qty * scaled_price)) / new_size
        } else {
            0
        };
        
        self.size.store(new_size, Ordering::Release);
        self.avg_entry_price.store(new_avg, Ordering::Release);
        
        // Mark position as open
        if !self.is_open.load(Ordering::Relaxed) {
            self.is_open.store(true, Ordering::Release);
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            self.open_timestamp_ns.store(now, Ordering::Release);
        }
        
        self.update_timestamp();
        
        from_scaled(new_size)
    }
    
    /// Reduce position (sell)
    pub fn reduce_position(&self, quantity: f64) -> f64 {
        let scaled_qty = to_scaled(quantity);
        
        // Update total sold
        self.total_sold.fetch_add(scaled_qty as u64, Ordering::Relaxed);
        
        // Reduce size
        let current_size = self.size.load(Ordering::Acquire);
        let new_size = current_size - scaled_qty;
        
        self.size.store(new_size, Ordering::Release);
        
        // Close position if flat
        if new_size == 0 {
            self.is_open.store(false, Ordering::Release);
            self.avg_entry_price.store(0, Ordering::Release);
        }
        
        self.update_timestamp();
        
        from_scaled(new_size)
    }
    
    /// Get current position size
    pub fn get_size(&self) -> f64 {
        from_scaled(self.size.load(Ordering::Acquire))
    }
    
    /// Get average entry price
    pub fn get_avg_entry_price(&self) -> f64 {
        from_scaled(self.avg_entry_price.load(Ordering::Acquire))
    }
    
    /// Get total bought quantity
    pub fn get_total_bought(&self) -> f64 {
        from_scaled(self.total_bought.load(Ordering::Relaxed) as i64)
    }
    
    /// Get total sold quantity
    pub fn get_total_sold(&self) -> f64 {
        from_scaled(self.total_sold.load(Ordering::Relaxed) as i64)
    }
    
    /// Check if position is open
    pub fn is_position_open(&self) -> bool {
        self.is_open.load(Ordering::Acquire)
    }
    
    /// Get position snapshot
    pub fn get_snapshot(&self) -> PositionSnapshot {
        PositionSnapshot {
            symbol_id: self.symbol_id,
            size: self.get_size(),
            avg_entry_price: self.get_avg_entry_price(),
            total_bought: self.get_total_bought(),
            total_sold: self.get_total_sold(),
            is_open: self.is_position_open(),
            timestamp_ns: self.last_update_ns.load(Ordering::Acquire),
        }
    }
    
    /// Update timestamp
    #[inline]
    fn update_timestamp(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_update_ns.store(now, Ordering::Release);
    }
    
    /// Close position forcibly
    pub fn close(&self) {
        self.size.store(0, Ordering::Release);
        self.avg_entry_price.store(0, Ordering::Release);
        self.is_open.store(false, Ordering::Release);
        self.update_timestamp();
    }
}

/// Snapshot of position state at a point in time
#[derive(Debug, Clone)]
pub struct PositionSnapshot {
    pub symbol_id: SymbolId,
    pub size: f64,
    pub avg_entry_price: f64,
    pub total_bought: f64,
    pub total_sold: f64,
    pub is_open: bool,
    pub timestamp_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_pnl_counter() {
        let counter = AtomicPnLCounter::new();
        
        // Add some trades
        counter.add_realized_pnl(100.0);
        counter.add_realized_pnl(-50.0);
        counter.add_realized_pnl(75.0);
        
        assert!((counter.get_realized_pnl() - 125.0).abs() < 0.001);
        assert_eq!(counter.get_trade_count(), 3);
        assert!((counter.get_win_rate() - 0.666).abs() < 0.01);
    }
    
    #[test]
    fn test_position_tracking() {
        let position = AtomicPosition::new(SymbolId::BTC);
        
        // Buy 1 BTC at 50000
        position.add_to_position(1.0, 50000.0);
        
        assert!((position.get_size() - 1.0).abs() < 0.001);
        assert!((position.get_avg_entry_price() - 50000.0).abs() < 0.01);
        assert!(position.is_position_open());
        
        // Sell 0.5 BTC
        position.reduce_position(0.5);
        
        assert!((position.get_size() - 0.5).abs() < 0.001);
        
        // Sell remaining
        position.reduce_position(0.5);
        
        assert!(position.get_size().abs() < 0.001);
        assert!(!position.is_position_open());
    }
    
    #[test]
    fn test_concurrent_pnl_updates() {
        let counter = Arc::new(AtomicPnLCounter::new());
        let mut handles = vec![];
        
        // Spawn threads to add PnL concurrently
        for _ in 0..10 {
            let counter_clone = counter.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    counter_clone.add_realized_pnl(1.0);
                }
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Should have 1000 total trades
        assert_eq!(counter.get_trade_count(), 1000);
        assert!((counter.get_realized_pnl() - 1000.0).abs() < 0.01);
    }
}
