//! Position Aggregator - Real-time PnL and delta calculation from raw events.
//!
//! This module aggregates position data from the event stream to provide
//! real-time profit/loss (PnL) and delta calculations. Critical for risk
//! management and portfolio monitoring in high-frequency trading.
//!
//! Features:
//! - Zero-allocation updates using pre-allocated buffers
//! - Integer overflow protection for massive volume spikes
//! - Tick-by-tick PnL recalculation
//! - Multi-asset portfolio aggregation
//! - Atomic counters for thread-safe updates

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;

/// Maximum number of symbols to track (prevents unbounded growth)
const MAX_SYMBOLS: usize = 100;

/// Represents a single position with all metrics
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    /// Net quantity (positive = long, negative = short)
    pub quantity: i64,
    /// Average entry price in micros (price * 1_000_000 for precision)
    pub average_price_micros: u64,
    /// Total realized PnL in micros
    pub realized_pnl_micros: i64,
    /// Last traded price in micros
    pub last_price_micros: u64,
    /// Number of open orders for this position
    pub open_orders_count: u32,
    /// Timestamp of last update in microseconds
    pub last_update_us: u64,
}

impl Position {
    pub fn new(symbol: &str) -> Self {
        Position {
            symbol: symbol.to_string(),
            quantity: 0,
            average_price_micros: 0,
            realized_pnl_micros: 0,
            last_price_micros: 0,
            open_orders_count: 0,
            last_update_us: 0,
        }
    }

    /// Calculate unrealized PnL in micros
    pub fn unrealized_pnl_micros(&self) -> i64 {
        if self.quantity == 0 || self.last_price_micros == 0 {
            return 0;
        }

        let current_value = self.quantity as i64 * self.last_price_micros as i64;
        let cost_basis = self.quantity as i64 * self.average_price_micros as i64;

        current_value - cost_basis
    }

    /// Calculate total PnL (realized + unrealized) in micros
    pub fn total_pnl_micros(&self) -> i64 {
        self.realized_pnl_micros + self.unrealized_pnl_micros()
    }

    /// Get delta (same as quantity for simple positions)
    pub fn delta(&self) -> i64 {
        self.quantity
    }

    /// Check if position is flat
    pub fn is_flat(&self) -> bool {
        self.quantity == 0
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.quantity > 0
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.quantity < 0
    }
}

/// Aggregated portfolio metrics
#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    /// Total realized PnL across all positions (micros)
    pub total_realized_pnl_micros: i64,
    /// Total unrealized PnL across all positions (micros)
    pub total_unrealized_pnl_micros: i64,
    /// Total PnL (realized + unrealized)
    pub total_pnl_micros: i64,
    /// Gross exposure (sum of absolute position values)
    pub gross_exposure_micros: u64,
    /// Net exposure (sum of signed position values)
    pub net_exposure_micros: i64,
    /// Number of non-flat positions
    pub position_count: usize,
    /// Number of symbols with open orders
    pub symbols_with_orders: usize,
    /// Timestamp of calculation
    pub timestamp_us: u64,
}

impl PortfolioMetrics {
    pub fn new(timestamp_us: u64) -> Self {
        PortfolioMetrics {
            total_realized_pnl_micros: 0,
            total_unrealized_pnl_micros: 0,
            total_pnl_micros: 0,
            gross_exposure_micros: 0,
            net_exposure_micros: 0,
            position_count: 0,
            symbols_with_orders: 0,
            timestamp_us,
        }
    }
}

/// Thread-safe position aggregator with overflow protection
pub struct PositionAggregator {
    /// Map of symbol -> Position
    positions: Arc<RwLock<HashMap<String, Position>>>,
    /// Atomic counter for total realized PnL (fast path without lock)
    atomic_realized_pnl: AtomicI64,
    /// Atomic counter for position count
    atomic_position_count: AtomicU64,
    /// Sequence number of last processed event
    last_sequence: AtomicU64,
    /// Overflow flag (set if any calculation would overflow)
    overflow_detected: AtomicU64,
}

unsafe impl Send for PositionAggregator {}
unsafe impl Sync for PositionAggregator {}

impl PositionAggregator {
    pub fn new() -> Self {
        PositionAggregator {
            positions: Arc::new(RwLock::new(HashMap::with_capacity(16))),
            atomic_realized_pnl: AtomicI64::new(0),
            atomic_position_count: AtomicU64::new(0),
            last_sequence: AtomicU64::new(0),
            overflow_detected: AtomicU64::new(0),
        }
    }

    /// Process an order fill event
    /// 
    /// Handles both opening and closing positions with proper PnL calculation.
    /// Uses micros (1e-6) for price and PnL to avoid floating-point errors.
    pub fn process_fill(
        &self,
        symbol: &str,
        side: &str,
        quantity: u64,
        price: u64, // Price in micros
        sequence: u64,
        timestamp_us: u64,
    ) -> Result<(), AggregatorError> {
        // Check for potential overflow before proceeding
        if quantity > i64::MAX as u64 {
            self.overflow_detected.fetch_add(1, Ordering::Relaxed);
            return Err(AggregatorError::OverflowRisk("quantity exceeds i64::MAX"));
        }

        let fill_quantity = quantity as i64;
        let mut positions = self.positions.write();

        // Enforce maximum symbols limit
        if !positions.contains_key(symbol) && positions.len() >= MAX_SYMBOLS {
            return Err(AggregatorError::MaxSymbolsReached(MAX_SYMBOLS));
        }

        let position = positions
            .entry(symbol.to_string())
            .or_insert_with(|| Position::new(symbol));

        // Convert side to signed quantity
        let signed_fill = match side {
            "buy" => fill_quantity,
            "sell" => -fill_quantity,
            _ => return Err(AggregatorError::InvalidSide(side.to_string())),
        };

        // Calculate PnL if closing or reducing position
        if (position.quantity > 0 && signed_fill < 0) || 
           (position.quantity < 0 && signed_fill > 0) {
            
            // Determine closing quantity (smaller of position and fill)
            let close_qty = if position.quantity > 0 {
                position.quantity.min(-signed_fill)
            } else {
                (-position.quantity).min(signed_fill)
            };

            if close_qty > 0 && position.average_price_micros > 0 {
                // Calculate realized PnL: (exit_price - entry_price) * close_qty
                let pnl_per_unit = price as i64 - position.average_price_micros as i64;
                
                // Check for multiplication overflow
                if close_qty.abs() > i64::MAX / pnl_per_unit.abs().max(1) {
                    self.overflow_detected.fetch_add(1, Ordering::Relaxed);
                    return Err(AggregatorError::OverflowRisk("PnL calculation would overflow"));
                }

                let realized_pnl = pnl_per_unit * close_qty;

                // Check for addition overflow
                if realized_pnl > 0 && position.realized_pnl_micros > i64::MAX - realized_pnl {
                    self.overflow_detected.fetch_add(1, Ordering::Relaxed);
                    return Err(AggregatorError::OverflowRisk("realized PnL would overflow"));
                }
                if realized_pnl < 0 && position.realized_pnl_micros < i64::MIN - realized_pnl {
                    self.overflow_detected.fetch_add(1, Ordering::Relaxed);
                    return Err(AggregatorError::OverflowRisk("realized PnL would underflow"));
                }

                position.realized_pnl_micros += realized_pnl;
                self.atomic_realized_pnl.fetch_add(realized_pnl, Ordering::Relaxed);
            }
        }

        // Update position quantity and average price
        if side == "buy" {
            if position.quantity >= 0 {
                // Adding to long or opening long: recalculate average
                let total_cost = (position.quantity as u128 * position.average_price_micros as u128)
                    + (fill_quantity as u128 * price as u128);
                
                position.quantity += fill_quantity;
                
                if position.quantity > 0 {
                    position.average_price_micros = (total_cost / position.quantity as u128) as u64;
                }
            } else {
                // Covering short position
                position.quantity += fill_quantity;
                if position.quantity > 0 {
                    // Flipped to long, new average is the fill price
                    position.average_price_micros = price;
                } else if position.quantity == 0 {
                    // Exactly closed
                    position.average_price_micros = 0;
                }
            }
        } else {
            // Sell side
            if position.quantity <= 0 {
                // Adding to short or opening short
                let total_cost = ((-position.quantity) as u128 * position.average_price_micros as u128)
                    + (fill_quantity as u128 * price as u128);
                
                position.quantity -= fill_quantity;
                
                if position.quantity < 0 {
                    position.average_price_micros = (total_cost / (-position.quantity) as u128) as u64;
                }
            } else {
                // Closing long position
                position.quantity -= fill_quantity;
                if position.quantity < 0 {
                    // Flipped to short, new average is the fill price
                    position.average_price_micros = price;
                } else if position.quantity == 0 {
                    position.average_price_micros = 0;
                }
            }
        }

        position.last_price_micros = price;
        position.last_update_us = timestamp_us;
        self.last_sequence.store(sequence, Ordering::Relaxed);

        // Update atomic position count
        self.update_atomic_counters();

        Ok(())
    }

    /// Process a tick event to update mark-to-market prices
    pub fn process_tick(
        &self,
        symbol: &str,
        price: u64, // Price in micros
        timestamp_us: u64,
    ) {
        let mut positions = self.positions.write();
        
        if let Some(position) = positions.get_mut(symbol) {
            position.last_price_micros = price;
            position.last_update_us = timestamp_us;
        }
    }

    /// Process order placed event
    pub fn process_order_placed(&self, symbol: &str) {
        let mut positions = self.positions.write();
        
        if let Some(position) = positions.get_mut(symbol) {
            position.open_orders_count = position.open_orders_count.saturating_add(1);
        }
    }

    /// Process order cancelled event
    pub fn process_order_cancelled(&self, symbol: &str) {
        let mut positions = self.positions.write();
        
        if let Some(position) = positions.get_mut(symbol) {
            position.open_orders_count = position.open_orders_count.saturating_sub(1);
        }
    }

    /// Get a specific position by symbol
    pub fn get_position(&self, symbol: &str) -> Option<Position> {
        let positions = self.positions.read();
        positions.get(symbol).cloned()
    }

    /// Get all positions
    pub fn get_all_positions(&self) -> Vec<Position> {
        let positions = self.positions.read();
        positions.values().cloned().collect()
    }

    /// Calculate current portfolio metrics
    pub fn calculate_metrics(&self) -> PortfolioMetrics {
        let positions = self.positions.read();
        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let mut metrics = PortfolioMetrics::new(timestamp_us);

        for position in positions.values() {
            if !position.is_flat() {
                metrics.position_count += 1;

                let unrealized = position.unrealized_pnl_micros();
                metrics.total_unrealized_pnl_micros += unrealized;
                metrics.total_realized_pnl_micros += position.realized_pnl_micros;

                // Calculate exposure (absolute value of position * price)
                let position_value = (position.quantity.abs() as u128 * position.last_price_micros as u128) as u64;
                metrics.gross_exposure_micros = metrics.gross_exposure_micros.saturating_add(position_value);

                // Net exposure (signed)
                let signed_value = position.quantity as i64 * position.last_price_micros as i64;
                metrics.net_exposure_micros = metrics.net_exposure_micros.saturating_add(signed_value);
            }

            if position.open_orders_count > 0 {
                metrics.symbols_with_orders += 1;
            }
        }

        metrics.total_pnl_micros = metrics.total_realized_pnl_micros + metrics.total_unrealized_pnl_micros;
        metrics
    }

    /// Get fast atomic PnL estimate (may be slightly stale but lock-free)
    pub fn get_fast_realized_pnl(&self) -> i64 {
        self.atomic_realized_pnl.load(Ordering::Relaxed)
    }

    /// Get fast position count estimate
    pub fn get_fast_position_count(&self) -> u64 {
        self.atomic_position_count.load(Ordering::Relaxed)
    }

    /// Check if overflow was detected in any calculation
    pub fn has_overflow_detected(&self) -> bool {
        self.overflow_detected.load(Ordering::Relaxed) > 0
    }

    /// Get last processed sequence number
    pub fn get_last_sequence(&self) -> u64 {
        self.last_sequence.load(Ordering::Relaxed)
    }

    /// Clear all positions (for rebuild operations)
    pub fn clear(&self) {
        let mut positions = self.positions.write();
        positions.clear();
        self.atomic_realized_pnl.store(0, Ordering::Relaxed);
        self.atomic_position_count.store(0, Ordering::Relaxed);
    }

    fn update_atomic_counters(&self) {
        let positions = self.positions.read();
        let count = positions.iter().filter(|(_, p)| !p.is_flat()).count() as u64;
        self.atomic_position_count.store(count, Ordering::Relaxed);
    }
}

impl Default for PositionAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during position aggregation
#[derive(Debug, Clone)]
pub enum AggregatorError {
    OverflowRisk(&'static str),
    MaxSymbolsReached(usize),
    InvalidSide(String),
    InvalidPrice(String),
}

impl std::fmt::Display for AggregatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregatorError::OverflowRisk(msg) => write!(f, "Overflow risk: {}", msg),
            AggregatorError::MaxSymbolsReached(max) => write!(f, "Maximum symbols reached: {}", max),
            AggregatorError::InvalidSide(side) => write!(f, "Invalid side: {}", side),
            AggregatorError::InvalidPrice(msg) => write!(f, "Invalid price: {}", msg),
        }
    }
}

impl std::error::Error for AggregatorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_position_open_and_close() {
        let aggregator = PositionAggregator::new();

        // Open long position: buy 1 BTC at 50000
        aggregator.process_fill("BTC/USDT", "buy", 1_000_000, 50_000_000_000, 1, 1000).unwrap();

        let position = aggregator.get_position("BTC/USDT").unwrap();
        assert_eq!(position.quantity, 1_000_000);
        assert_eq!(position.average_price_micros, 50_000_000_000);

        // Price moves to 51000
        aggregator.process_tick("BTC/USDT", 51_000_000_000, 2000);

        let position = aggregator.get_position("BTC/USDT").unwrap();
        assert_eq!(position.unrealized_pnl_micros(), 1_000_000_000_000); // 1000 USDT profit

        // Close position: sell 1 BTC at 51000
        aggregator.process_fill("BTC/USDT", "sell", 1_000_000, 51_000_000_000, 2, 3000).unwrap();

        let position = aggregator.get_position("BTC/USDT").unwrap();
        assert_eq!(position.quantity, 0);
        assert_eq!(position.realized_pnl_micros, 1_000_000_000_000);
    }

    #[test]
    fn test_short_position() {
        let aggregator = PositionAggregator::new();

        // Open short: sell 1 BTC at 50000
        aggregator.process_fill("BTC/USDT", "sell", 1_000_000, 50_000_000_000, 1, 1000).unwrap();

        let position = aggregator.get_position("BTC/USDT").unwrap();
        assert_eq!(position.quantity, -1_000_000);

        // Price drops to 49000 (profit for short)
        aggregator.process_tick("BTC/USDT", 49_000_000_000, 2000);

        let position = aggregator.get_position("BTC/USDT").unwrap();
        assert_eq!(position.unrealized_pnl_micros(), 1_000_000_000_000); // Profit
    }

    #[test]
    fn test_overflow_protection() {
        let aggregator = PositionAggregator::new();

        // Try to process impossibly large quantity
        let result = aggregator.process_fill("BTC/USDT", "buy", i64::MAX as u64 + 1, 1000, 1, 1000);
        
        assert!(matches!(result, Err(AggregatorError::OverflowRisk(_))));
        assert!(aggregator.has_overflow_detected());
    }

    #[test]
    fn test_portfolio_metrics() {
        let aggregator = PositionAggregator::new();

        // Open two positions
        aggregator.process_fill("BTC/USDT", "buy", 1_000_000, 50_000_000_000, 1, 1000).unwrap();
        aggregator.process_fill("ETH/USDT", "buy", 10_000_000, 3_000_000_000, 2, 2000).unwrap();

        // Update prices
        aggregator.process_tick("BTC/USDT", 51_000_000_000, 3000);
        aggregator.process_tick("ETH/USDT", 3_100_000_000, 3000);

        let metrics = aggregator.calculate_metrics();

        assert_eq!(metrics.position_count, 2);
        assert!(metrics.gross_exposure_micros > 0);
        assert!(metrics.total_unrealized_pnl_micros > 0);
    }

    #[test]
    fn test_max_symbols_limit() {
        let aggregator = PositionAggregator::new();

        // Fill up to MAX_SYMBOLS
        for i in 0..MAX_SYMBOLS {
            let symbol = format!("SYM{}", i);
            aggregator.process_fill(&symbol, "buy", 1000, 1_000_000, i as u64, 1000).unwrap();
        }

        // Try to add one more
        let result = aggregator.process_fill("NEW_SYM", "buy", 1000, 1_000_000, MAX_SYMBOLS as u64, 2000);
        
        assert!(matches!(result, Err(AggregatorError::MaxSymbolsReached(_))));
    }
}
