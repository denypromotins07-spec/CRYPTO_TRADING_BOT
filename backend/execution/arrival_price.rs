//! Arrival Price Execution Algorithm
//! Calculates real-time Implementation Shortfall (IS) and slippage costs.
//! Optimized for zero-cost abstractions and microsecond-level execution.
//! 
//! Stage 13: Advanced Execution Algorithms
//! Target: Minimize market impact to secure 8k-20k INR/hour

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use crossbeam_channel::Sender;

/// Represents the arrival price snapshot at order initiation
#[derive(Debug, Clone, Copy)]
pub struct ArrivalPrice {
    pub timestamp_ns: u64,
    pub price_bps: i64, // Price in basis points * 10000 for precision
    pub volume: u64,
    pub side: OrderSide,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Real-time Implementation Shortfall calculator
pub struct ImplementationShortfallCalculator {
    arrival_price: Option<ArrivalPrice>,
    total_filled_value: u64,
    total_filled_cost: u64,
    start_time: Instant,
    max_slippage_bps: AtomicU64,
}

impl ImplementationShortfallCalculator {
    pub fn new(max_slippage_bps: u64) -> Self {
        Self {
            arrival_price: None,
            total_filled_value: 0,
            total_filled_cost: 0,
            start_time: Instant::now(),
            max_slippage_bps: AtomicU64::new(max_slippage_bps),
        }
    }

    /// Set the arrival price benchmark
    #[inline]
    pub fn set_arrival_price(&mut self, price: ArrivalPrice) {
        self.arrival_price = Some(price);
        self.start_time = Instant::now();
    }

    /// Record a fill event
    #[inline]
    pub fn record_fill(&mut self, price_bps: i64, volume: u64) {
        self.total_filled_value += volume;
        self.total_filled_cost += price_bps * volume as i64;
    }

    /// Calculate current IS in basis points
    #[inline]
    pub fn calculate_is_bps(&self) -> Option<i64> {
        let arrival = self.arrival_price?;
        
        if self.total_filled_value == 0 {
            return Some(0);
        }

        let avg_execution_price = self.total_filled_cost / self.total_filled_value as i64;
        let arrival_price_bps = arrival.price_bps;

        let is_bps = match arrival.side {
            OrderSide::Buy => avg_execution_price - arrival_price_bps,
            OrderSide::Sell => arrival_price_bps - avg_execution_price,
        };

        Some(is_bps)
    }

    /// Check if slippage exceeds threshold
    #[inline]
    pub fn is_slippage_exceeded(&self) -> bool {
        if let Some(is_bps) = self.calculate_is_bps() {
            is_bps.abs() as u64 > self.max_slippage_bps.load(Ordering::Relaxed)
        } else {
            false
        }
    }

    /// Get execution time elapsed in microseconds
    #[inline]
    pub fn elapsed_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }

    /// Reset calculator for new order
    #[inline]
    pub fn reset(&mut self) {
        self.arrival_price = None;
        self.total_filled_value = 0;
        self.total_filled_cost = 0;
        self.start_time = Instant::now();
    }
}

/// Dynamic aggression adjuster based on market movement
pub struct AggressionController {
    base_aggression: f64,
    current_aggression: f64,
    volatility_multiplier: f64,
    market_move_threshold_bps: i64,
}

impl AggressionController {
    pub fn new(base_aggression: f64, volatility_mult: f64, threshold_bps: i64) -> Self {
        Self {
            base_aggression,
            current_aggression: base_aggression,
            volatility_multiplier: volatility_mult,
            market_move_threshold_bps: threshold_bps,
        }
    }

    /// Adjust aggression based on market move against position
    #[inline]
    pub fn adjust_aggression(&mut self, market_move_bps: i64, volatility: f64) {
        let vol_adjustment = volatility * self.volatility_multiplier;
        
        if market_move_bps.abs() > self.market_move_threshold_bps {
            // Accelerate execution if market moves against us
            self.current_aggression = (self.base_aggression * (1.0 + vol_adjustment))
                .min(1.0)
                .max(0.1);
        } else {
            self.current_aggression = self.base_aggression;
        }
    }

    #[inline]
    pub fn get_current_aggression(&self) -> f64 {
        self.current_aggression
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implementation_shortfall_buy() {
        let mut calc = ImplementationShortfallCalculator::new(50);
        let arrival = ArrivalPrice {
            timestamp_ns: 0,
            price_bps: 500000, // $50,000.00
            volume: 100,
            side: OrderSide::Buy,
        };
        calc.set_arrival_price(arrival);
        
        // Fill at slightly worse price
        calc.record_fill(500050, 50); // $50,005.00
        calc.record_fill(500100, 50); // $50,010.00
        
        let is_bps = calc.calculate_is_bps().unwrap();
        assert_eq!(is_bps, 75); // Average execution was 75 bps worse
    }

    #[test]
    fn test_aggression_acceleration() {
        let mut controller = AggressionController::new(0.5, 0.1, 10);
        
        // Market moves against us by 20 bps with high volatility
        controller.adjust_aggression(20, 0.5);
        
        assert!(controller.get_current_aggression() > 0.5);
    }
}
