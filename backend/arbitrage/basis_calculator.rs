//! Basis Calculator - Real-time Spot vs Perpetual Price Spreads
//! 
//! This module provides O(1) time complexity calculations for spot-futures basis spreads.
//! Optimized for front-running slower market makers by pre-computing spread thresholds.
//! Uses zero-cost abstractions and avoids heap allocations during critical trading paths.
//!
//! Chapter 1: Spot-Futures Basis Trading and Cash-and-Carry Arbitrage Logic

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Represents a price tick with nanosecond precision
#[derive(Clone, Copy, Debug)]
pub struct PriceTick {
    /// Price in micro-units (e.g., micro-BTC for precision)
    pub price_micro: u64,
    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: u64,
    /// Exchange identifier
    pub exchange_id: u8,
}

/// Basis spread calculation result
#[derive(Clone, Copy, Debug)]
pub struct BasisSpread {
    /// Spread in basis points (bps)
    pub spread_bps: i64,
    /// Annualized basis rate
    pub annualized_rate: f64,
    /// Days to expiry (for perpetual, this is funding interval)
    pub days_to_expiry: f64,
    /// Timestamp of calculation
    pub calc_timestamp_ns: u64,
}

/// Pre-computed thresholds for fast arbitrage decisions
pub struct BasisThresholds {
    /// Minimum spread to trigger arb (in bps)
    pub min_trigger_bps: i64,
    /// Maximum spread before risk alert (in bps)
    pub max_alert_bps: i64,
    /// Transaction cost buffer (in bps)
    pub tx_cost_bps: i64,
}

impl BasisThresholds {
    pub const fn new(min_trigger: i64, max_alert: i64, tx_cost: i64) -> Self {
        Self {
            min_trigger_bps: min_trigger,
            max_alert_bps: max_alert,
            tx_cost_bps: tx_cost,
        }
    }

    /// Check if spread is profitable after costs - O(1) operation
    #[inline]
    pub const fn is_profitable(&self, spread_bps: i64) -> bool {
        spread_bps > (self.min_trigger_bps + self.tx_cost_bps)
    }
}

/// High-performance basis calculator with zero heap allocations
pub struct BasisCalculator {
    /// Cached spot price (micro-units)
    spot_price_micro: AtomicU64,
    /// Cached perpetual price (micro-units)
    perp_price_micro: AtomicU64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Pre-computed thresholds
    thresholds: BasisThresholds,
    /// Funding rate interval in hours (typically 8)
    funding_interval_hours: u32,
}

impl BasisCalculator {
    /// Create a new basis calculator with given thresholds
    pub const fn new(thresholds: BasisThresholds, funding_interval: u32) -> Self {
        Self {
            spot_price_micro: AtomicU64::new(0),
            perp_price_micro: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            thresholds,
            funding_interval_hours: funding_interval,
        }
    }

    /// Update spot price atomically - thread-safe
    #[inline]
    pub fn update_spot(&self, price_micro: u64) {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        self.spot_price_micro.store(price_micro, Ordering::SeqCst);
        self.last_update_ns.store(now_ns, Ordering::SeqCst);
    }

    /// Update perpetual price atomically - thread-safe
    #[inline]
    pub fn update_perp(&self, price_micro: u64) {
        self.perp_price_micro.store(price_micro, Ordering::SeqCst);
    }

    /// Calculate basis spread in O(1) time - no allocations
    /// Returns None if prices are not yet initialized
    #[inline]
    pub fn calculate_spread(&self) -> Option<BasisSpread> {
        let spot = self.spot_price_micro.load(Ordering::Acquire);
        let perp = self.perp_price_micro.load(Ordering::Acquire);

        if spot == 0 || perp == 0 {
            return None;
        }

        // Calculate spread in basis points: ((perp - spot) / spot) * 10000
        // Using integer arithmetic to avoid float precision issues
        let spread_bps = if perp >= spot {
            ((perp - spot) * 10000 / spot) as i64
        } else {
            -((spot - perp) * 10000 / spot) as i64
        };

        // Annualized rate: spread * (365 / days_to_funding)
        // For perpetual swaps, we use funding interval as the "expiry"
        let days_fraction = self.funding_interval_hours as f64 / 24.0;
        let annualized_rate = (spread_bps as f64 / 10000.0) * (365.0 / days_fraction);

        let now_ns = self.last_update_ns.load(Ordering::Acquire);

        Some(BasisSpread {
            spread_bps,
            annualized_rate,
            days_to_expiry: days_fraction,
            calc_timestamp_ns: now_ns,
        })
    }

    /// Fast check if arbitrage opportunity exists - O(1)
    #[inline]
    pub fn is_arb_opportunity(&self) -> bool {
        if let Some(spread) = self.calculate_spread() {
            self.thresholds.is_profitable(spread.spread_bps)
        } else {
            false
        }
    }

    /// Get current spread in bps without full calculation
    #[inline]
    pub fn get_spread_bps_quick(&self) -> Option<i64> {
        let spot = self.spot_price_micro.load(Ordering::Acquire);
        let perp = self.perp_price_micro.load(Ordering::Acquire);

        if spot == 0 || perp == 0 {
            return None;
        }

        if perp >= spot {
            Some(((perp - spot) * 10000 / spot) as i64)
        } else {
            Some(-((spot - perp) * 10000 / spot) as i64)
        }
    }

    /// Reset prices (used during market halt or error recovery)
    #[inline]
    pub fn reset(&self) {
        self.spot_price_micro.store(0, Ordering::Release);
        self.perp_price_micro.store(0, Ordering::Release);
    }
}

/// Cache-line aligned structure for multi-threaded access
#[repr(align(64))]
pub struct AlignedBasisCalculator {
    inner: BasisCalculator,
    _padding: [u8; 64],
}

impl AlignedBasisCalculator {
    pub const fn new(thresholds: BasisThresholds, funding_interval: u32) -> Self {
        Self {
            inner: BasisCalculator::new(thresholds, funding_interval),
            _padding: [0; 64],
        }
    }

    #[inline]
    pub fn update_spot(&self, price_micro: u64) {
        self.inner.update_spot(price_micro);
    }

    #[inline]
    pub fn update_perp(&self, price_micro: u64) {
        self.inner.update_perp(price_micro);
    }

    #[inline]
    pub fn calculate_spread(&self) -> Option<BasisSpread> {
        self.inner.calculate_spread()
    }

    #[inline]
    pub fn is_arb_opportunity(&self) -> bool {
        self.inner.is_arb_opportunity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_calculation() {
        let thresholds = BasisThresholds::new(10, 100, 5);
        let calc = BasisCalculator::new(thresholds, 8);

        // Spot: 50000 USDT, Perp: 50050 USDT (10 bps spread)
        calc.update_spot(50_000_000_000); // 50000 * 1_000_000
        calc.update_perp(50_050_000_000); // 50050 * 1_000_000

        let spread = calc.calculate_spread().unwrap();
        assert!(spread.spread_bps >= 9); // Allow for integer rounding
        assert!(spread.spread_bps <= 11);
    }

    #[test]
    fn test_arb_threshold() {
        let thresholds = BasisThresholds::new(10, 100, 5);
        let calc = BasisCalculator::new(thresholds, 8);

        // Not profitable: spread = 10 bps, threshold = 15 bps (10 + 5)
        calc.update_spot(50_000_000_000);
        calc.update_perp(50_050_000_000);
        assert!(!calc.is_arb_opportunity());

        // Profitable: spread = 20 bps, threshold = 15 bps
        calc.update_perp(50_100_000_000);
        assert!(calc.is_arb_opportunity());
    }
}
