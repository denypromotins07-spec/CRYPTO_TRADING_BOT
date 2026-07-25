//! Real-Time Transaction Cost Analysis (TCA)
//! Calculates basis points of slippage per micro-order.
//! Provides granular cost attribution for execution quality assessment.
//! 
//! Stage 13: Advanced Execution Algorithms
//! Target: Minimize market impact to secure 8k-20k INR/hour

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Micro-order execution record
#[derive(Debug, Clone)]
pub struct MicroOrder {
    pub order_id: u64,
    pub parent_order_id: u64,
    pub timestamp_ns: u64,
    pub side: OrderSide,
    pub requested_qty: u64,
    pub filled_qty: u64,
    pub request_price_bps: i64,
    pub fill_price_bps: i64,
    pub venue: Venue,
    pub order_type: OrderType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Venue {
    Spot,
    Futures,
    Margin,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Limit,
    Market,
    PostOnly,
    IOC,
    FOK,
}

/// Slippage breakdown for a micro-order
#[derive(Debug, Clone)]
pub struct SlippageBreakdown {
    pub total_slippage_bps: i64,
    pub spread_cost_bps: i64,
    pub market_impact_bps: i64,
    pub timing_cost_bps: i64,
    pub fee_cost_bps: i64,
    pub opportunity_cost_bps: i64,
}

/// Real-time TCA calculator
pub struct RealtimeTca {
    micro_orders: HashMap<u64, MicroOrder>,
    parent_order_stats: HashMap<u64, ParentOrderStats>,
    running_totals: TcaRunningTotals,
    config: TcaConfig,
    is_active: AtomicBool,
}

#[derive(Debug, Clone)]
struct ParentOrderStats {
    total_requested: u64,
    total_filled: u64,
    total_value_bps: i64,
    arrival_price_bps: i64,
    start_time_ns: u64,
    micro_order_count: u64,
}

#[derive(Debug, Clone)]
struct TcaRunningTotals {
    total_slippage_bps: AtomicI64,
    total_market_impact_bps: AtomicI64,
    total_spread_cost_bps: AtomicI64,
    total_timing_cost_bps: AtomicI64,
    total_fees_bps: AtomicI64,
    order_count: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub struct TcaConfig {
    /// Fee in basis points for each venue
    pub spot_fee_bps: i64,
    pub futures_fee_bps: i64,
    pub margin_fee_bps: i64,
    /// Spread cost estimation factor
    pub spread_factor: f64,
    /// Market impact model coefficient
    pub impact_coefficient: f64,
}

impl Default for TcaConfig {
    fn default() -> Self {
        Self {
            spot_fee_bps: 10,
            futures_fee_bps: 4,
            margin_fee_bps: 10,
            spread_factor: 0.5,
            impact_coefficient: 0.1,
        }
    }
}

impl RealtimeTca {
    pub fn new(config: TcaConfig) -> Self {
        Self {
            micro_orders: HashMap::with_capacity(1000),
            parent_order_stats: HashMap::with_capacity(100),
            running_totals: TcaRunningTotals {
                total_slippage_bps: AtomicI64::new(0),
                total_market_impact_bps: AtomicI64::new(0),
                total_spread_cost_bps: AtomicI64::new(0),
                total_timing_cost_bps: AtomicI64::new(0),
                total_fees_bps: AtomicI64::new(0),
                order_count: AtomicU64::new(0),
            },
            config,
            is_active: AtomicBool::new(true),
        }
    }

    /// Record a micro-order execution
    pub fn record_micro_order(&mut self, order: MicroOrder) -> SlippageBreakdown {
        if !self.is_active.load(Ordering::Relaxed) {
            return SlippageBreakdown {
                total_slippage_bps: 0,
                spread_cost_bps: 0,
                market_impact_bps: 0,
                timing_cost_bps: 0,
                fee_cost_bps: 0,
                opportunity_cost_bps: 0,
            };
        }

        // Calculate slippage
        let breakdown = self.calculate_slippage(&order);

        // Store micro-order
        self.micro_orders.insert(order.order_id, order.clone());

        // Update parent order stats
        self.update_parent_stats(&order, &breakdown);

        // Update running totals
        self.running_totals.total_slippage_bps.fetch_add(breakdown.total_slippage_bps, Ordering::Relaxed);
        self.running_totals.total_market_impact_bps.fetch_add(breakdown.market_impact_bps, Ordering::Relaxed);
        self.running_totals.total_spread_cost_bps.fetch_add(breakdown.spread_cost_bps, Ordering::Relaxed);
        self.running_totals.total_timing_cost_bps.fetch_add(breakdown.timing_cost_bps, Ordering::Relaxed);
        self.running_totals.total_fees_bps.fetch_add(breakdown.fee_cost_bps, Ordering::Relaxed);
        self.running_totals.order_count.fetch_add(1, Ordering::Relaxed);

        // Prune old orders if needed
        if self.micro_orders.len() > 10000 {
            self.prune_old_orders();
        }

        breakdown
    }

    /// Calculate slippage breakdown for a micro-order
    fn calculate_slippage(&self, order: &MicroOrder) -> SlippageBreakdown {
        // Calculate raw slippage
        let raw_slippage_bps = match order.side {
            OrderSide::Buy => order.fill_price_bps - order.request_price_bps,
            OrderSide::Sell => order.request_price_bps - order.fill_price_bps,
        };

        // Get fee based on venue
        let fee_bps = match order.venue {
            Venue::Spot => self.config.spot_fee_bps,
            Venue::Futures => self.config.futures_fee_bps,
            Venue::Margin => self.config.margin_fee_bps,
        };

        // Estimate spread cost (simplified - would use real spread data)
        let spread_estimate_bps = (order.request_price_bps as f64 * 0.0001 * self.config.spread_factor) as i64;

        // Estimate market impact using square-root model
        // Impact = coefficient * sqrt(qty / avg_daily_volume)
        // Simplified version uses qty directly
        let impact_estimate_bps = (self.config.impact_coefficient * (order.filled_qty as f64).sqrt()) as i64;

        // Timing cost (difference between arrival and request price)
        // Would need arrival price tracking - simplified here
        let timing_cost_bps = raw_slippage_bps - spread_estimate_bps - impact_estimate_bps;

        // Opportunity cost (unfilled portion)
        let unfilled_ratio = if order.requested_qty > 0 {
            (order.requested_qty - order.filled_qty) as f64 / order.requested_qty as f64
        } else {
            0.0
        };
        let opportunity_cost_bps = (unfilled_ratio * raw_slippage_bps as f64) as i64;

        SlippageBreakdown {
            total_slippage_bps: raw_slippage_bps + fee_bps,
            spread_cost_bps: spread_estimate_bps,
            market_impact_bps: impact_estimate_bps,
            timing_cost_bps: timing_cost_bps.max(0),
            fee_cost_bps: fee_bps,
            opportunity_cost_bps,
        }
    }

    /// Update parent order statistics
    fn update_parent_stats(&mut self, order: &MicroOrder, breakdown: &SlippageBreakdown) {
        let stats = self.parent_order_stats.entry(order.parent_order_id)
            .or_insert_with(|| ParentOrderStats {
                total_requested: 0,
                total_filled: 0,
                total_value_bps: 0,
                arrival_price_bps: order.request_price_bps,
                start_time_ns: order.timestamp_ns,
                micro_order_count: 0,
            });

        stats.total_requested += order.requested_qty;
        stats.total_filled += order.filled_qty;
        stats.total_value_bps += order.fill_price_bps * order.filled_qty as i64;
        stats.micro_order_count += 1;
    }

    /// Get aggregate TCA metrics for a parent order
    pub fn get_parent_order_tca(&self, parent_order_id: u64) -> Option<ParentOrderTca> {
        let stats = self.parent_order_stats.get(&parent_order_id)?;

        if stats.total_filled == 0 {
            return None;
        }

        let avg_fill_price_bps = stats.total_value_bps / stats.total_filled as i64;
        let slippage_vs_arrival = avg_fill_price_bps - stats.arrival_price_bps;

        let fill_rate = stats.total_filled as f64 / stats.total_requested as f64;

        Some(ParentOrderTca {
            parent_order_id,
            total_requested: stats.total_requested,
            total_filled: stats.total_filled,
            fill_rate,
            avg_fill_price_bps,
            arrival_price_bps: stats.arrival_price_bps,
            total_slippage_bps: slippage_vs_arrival,
            micro_order_count: stats.micro_order_count,
        })
    }

    /// Get running aggregate metrics
    pub fn get_running_totals(&self) -> RunningTcaTotals {
        RunningTcaTotals {
            total_slippage_bps: self.running_totals.total_slippage_bps.load(Ordering::Relaxed),
            total_market_impact_bps: self.running_totals.total_market_impact_bps.load(Ordering::Relaxed),
            total_spread_cost_bps: self.running_totals.total_spread_cost_bps.load(Ordering::Relaxed),
            total_timing_cost_bps: self.running_totals.total_timing_cost_bps.load(Ordering::Relaxed),
            total_fees_bps: self.running_totals.total_fees_bps.load(Ordering::Relaxed),
            order_count: self.running_totals.order_count.load(Ordering::Relaxed),
        }
    }

    /// Prune old micro-orders to prevent memory growth
    fn prune_old_orders(&mut self) {
        // Keep only last 5000 orders
        let keys_to_remove: Vec<_> = self.micro_orders.keys().take(self.micro_orders.len() - 5000).cloned().collect();
        for key in keys_to_remove {
            self.micro_orders.remove(&key);
        }
    }

    /// Reset TCA state
    pub fn reset(&mut self) {
        self.micro_orders.clear();
        self.parent_order_stats.clear();
        self.running_totals.total_slippage_bps.store(0, Ordering::Relaxed);
        self.running_totals.total_market_impact_bps.store(0, Ordering::Relaxed);
        self.running_totals.total_spread_cost_bps.store(0, Ordering::Relaxed);
        self.running_totals.total_timing_cost_bps.store(0, Ordering::Relaxed);
        self.running_totals.total_fees_bps.store(0, Ordering::Relaxed);
        self.running_totals.order_count.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct ParentOrderTca {
    pub parent_order_id: u64,
    pub total_requested: u64,
    pub total_filled: u64,
    pub fill_rate: f64,
    pub avg_fill_price_bps: i64,
    pub arrival_price_bps: i64,
    pub total_slippage_bps: i64,
    pub micro_order_count: u64,
}

#[derive(Debug)]
pub struct RunningTcaTotals {
    pub total_slippage_bps: i64,
    pub total_market_impact_bps: i64,
    pub total_spread_cost_bps: i64,
    pub total_timing_cost_bps: i64,
    pub total_fees_bps: i64,
    pub order_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_calculation() {
        let config = TcaConfig::default();
        let mut tca = RealtimeTca::new(config);

        let order = MicroOrder {
            order_id: 1,
            parent_order_id: 100,
            timestamp_ns: 1000000,
            side: OrderSide::Buy,
            requested_qty: 1000,
            filled_qty: 1000,
            request_price_bps: 500000,
            fill_price_bps: 500050,
            venue: Venue::Spot,
            order_type: OrderType::Limit,
        };

        let breakdown = tca.record_micro_order(order);

        assert!(breakdown.total_slippage_bps > 0);
        assert!(breakdown.fee_cost_bps > 0);
    }

    #[test]
    fn test_parent_order_aggregation() {
        let config = TcaConfig::default();
        let mut tca = RealtimeTca::new(config);

        // Record multiple micro-orders for same parent
        for i in 0..5 {
            let order = MicroOrder {
                order_id: i + 1,
                parent_order_id: 100,
                timestamp_ns: 1000000 + i * 100000,
                side: OrderSide::Buy,
                requested_qty: 200,
                filled_qty: 200,
                request_price_bps: 500000,
                fill_price_bps: 500000 + (i as i64 * 10),
                venue: Venue::Spot,
                order_type: OrderType::Limit,
            };
            tca.record_micro_order(order);
        }

        let parent_tca = tca.get_parent_order_tca(100).unwrap();
        assert_eq!(parent_tca.total_filled, 1000);
        assert_eq!(parent_tca.micro_order_count, 5);
        assert!(parent_tca.fill_rate > 0.99);
    }
}
