//! Maker-Taker Router
//! Optimizes fee tiers by routing between maker and taker orders.
//! Zero-cost abstractions for real-time fee optimization.
//!
//! This module implements intelligent order routing that:
//! - Tracks VIP tier progress on exchanges (e.g., Binance)
//! - Calculates optimal maker vs taker execution
//! - Considers rebates vs fees in execution decisions
//! - Minimizes total trading costs
//!
//! Fee Structure (Binance example):
//! - Regular: Maker 0.1%, Taker 0.1%
//! - VIP 1: Maker 0.09%, Taker 0.1%
//! - With BNB: Additional 25% discount

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Fee tier configuration
#[derive(Debug, Clone, Copy)]
pub struct FeeTier {
    /// Maker fee rate (negative = rebate)
    pub maker_fee_bps: i64,
    /// Taker fee rate
    pub taker_fee_bps: i64,
    /// Minimum 30-day volume for this tier (in USD, scaled by 1e6)
    pub min_volume_usd: u64,
    /// Minimum BNB holdings for discount
    pub min_bnb_holdings: f64,
}

impl FeeTier {
    /// Calculate maker fee for a given notional value
    #[inline]
    pub fn maker_fee(&self, notional_usd: f64) -> f64 {
        notional_usd * (self.maker_fee_bps as f64 / 10000.0)
    }

    /// Calculate taker fee for a given notional value
    #[inline]
    pub fn taker_fee(&self, notional_usd: f64) -> f64 {
        notional_usd * (self.taker_fee_bps as f64 / 10000.0)
    }

    /// Check if BNB discount applies
    #[inline]
    pub fn has_bnb_discount(&self, bnb_holdings: f64) -> bool {
        bnb_holdings >= self.min_bnb_holdings
    }

    /// Apply BNB discount to fees (25% off)
    #[inline]
    pub fn apply_bnb_discount(&self, fee: f64) -> f64 {
        fee * 0.75
    }
}

/// Default fee tiers (Binance-style)
impl Default for FeeTier {
    fn default() -> Self {
        Self {
            maker_fee_bps: 10, // 0.1%
            taker_fee_bps: 10, // 0.1%
            min_volume_usd: 0,
            min_bnb_holdings: 0.0,
        }
    }
}

/// VIP tier definitions
pub const VIP_TIERS: [FeeTier; 10] = [
    FeeTier { maker_fee_bps: 10, taker_fee_bps: 10, min_volume_usd: 0, min_bnb_holdings: 0.0 },      // Regular
    FeeTier { maker_fee_bps: 9, taker_fee_bps: 10, min_volume_usd: 1_000_000_000, min_bnb_holdings: 0.0 }, // VIP 1
    FeeTier { maker_fee_bps: 8, taker_fee_bps: 10, min_volume_usd: 5_000_000_000, min_bnb_holdings: 0.0 }, // VIP 2
    FeeTier { maker_fee_bps: 7, taker_fee_bps: 9, min_volume_usd: 25_000_000_000, min_bnb_holdings: 0.0 }, // VIP 3
    FeeTier { maker_fee_bps: 6, taker_fee_bps: 8, min_volume_usd: 50_000_000_000, min_bnb_holdings: 0.0 }, // VIP 4
    FeeTier { maker_fee_bps: 5, taker_fee_bps: 7, min_volume_usd: 100_000_000_000, min_bnb_holdings: 0.0 }, // VIP 5
    FeeTier { maker_fee_bps: 4, taker_fee_bps: 6, min_volume_usd: 250_000_000_000, min_bnb_holdings: 0.0 }, // VIP 6
    FeeTier { maker_fee_bps: 3, taker_fee_bps: 5, min_volume_usd: 500_000_000_000, min_bnb_holdings: 0.0 }, // VIP 7
    FeeTier { maker_fee_bps: 2, taker_fee_bps: 4, min_volume_usd: 1_000_000_000_000, min_bnb_holdings: 0.0 }, // VIP 8
    FeeTier { maker_fee_bps: 0, taker_fee_bps: 4, min_volume_usd: 5_000_000_000_000, min_bnb_holdings: 0.0 }, // VIP 9
];

/// Order execution type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionType {
    Maker,   // Limit order providing liquidity
    Taker,   // Market order taking liquidity
    PostOnly, // Limit order that must not cross
}

/// Routing decision result
#[derive(Debug, Clone, Copy)]
pub struct RoutingDecision {
    pub execution_type: ExecutionType,
    pub expected_fee_bps: i64,
    pub expected_cost_usd: f64,
    pub is_optimal: bool,
    pub reason: &'static str,
}

/// Main maker-taker router
pub struct MakerTakerRouter {
    /// Current 30-day volume (scaled by 1e6 for precision)
    volume_30d_usd: AtomicU64,
    /// Current BNB holdings
    bnb_holdings: AtomicU64, // Scaled by 1e8
    /// Current fee tier index
    current_tier: AtomicU64,
    /// Enable BNB discount
    use_bnb_discount: AtomicBool,
    /// Total fees paid (scaled by 1e6)
    total_fees_paid: AtomicU64,
    /// Total rebates earned (scaled by 1e6)
    total_rebates_earned: AtomicU64,
    /// Validity flag
    is_valid: AtomicBool,
}

impl MakerTakerRouter {
    /// Create a new router with default settings
    pub fn new() -> Self {
        Self {
            volume_30d_usd: AtomicU64::new(0),
            bnb_holdings: AtomicU64::new(0),
            current_tier: AtomicU64::new(0),
            use_bnb_discount: AtomicBool::new(false),
            total_fees_paid: AtomicU64::new(0),
            total_rebates_earned: AtomicU64::new(0),
            is_valid: AtomicBool::new(true),
        }
    }

    /// Update 30-day volume and recalculate tier
    #[inline]
    pub fn update_volume(&self, volume_usd: u64) {
        self.volume_30d_usd.store(volume_usd, Ordering::Relaxed);
        self.recalculate_tier();
    }

    /// Add to 30-day volume
    #[inline]
    pub fn add_volume(&self, volume_usd: u64) {
        self.volume_30d_usd.fetch_add(volume_usd, Ordering::Relaxed);
        self.recalculate_tier();
    }

    /// Update BNB holdings
    #[inline]
    pub fn update_bnb_holdings(&self, bnb_amount: f64) {
        self.bnb_holdings.store((bnb_amount * 1e8) as u64, Ordering::Relaxed);
    }

    /// Enable/disable BNB discount
    #[inline]
    pub fn set_bnb_discount(&self, enabled: bool) {
        self.use_bnb_discount.store(enabled, Ordering::Relaxed);
    }

    /// Recalculate current fee tier based on volume
    #[inline]
    fn recalculate_tier(&self) {
        let volume = self.volume_30d_usd.load(Ordering::Relaxed);
        
        let mut tier_idx = 0;
        for (idx, tier) in VIP_TIERS.iter().enumerate() {
            if volume >= tier.min_volume_usd {
                tier_idx = idx as u64;
            } else {
                break;
            }
        }
        
        self.current_tier.store(tier_idx, Ordering::Relaxed);
    }

    /// Get current fee tier
    #[inline]
    pub fn get_current_tier(&self) -> &FeeTier {
        let tier_idx = self.current_tier.load(Ordering::Relaxed) as usize;
        &VIP_TIERS[tier_idx.min(VIP_TIERS.len() - 1)]
    }

    /// Get current tier index
    #[inline]
    pub fn get_tier_index(&self) -> usize {
        self.current_tier.load(Ordering::Relaxed) as usize
    }

    /// Calculate optimal execution type for given scenario
    #[inline]
    pub fn calculate_optimal_execution(
        &self,
        notional_usd: f64,
        urgency_score: f64, // 0.0 = no urgency, 1.0 = maximum urgency
        spread_bps: f64,
    ) -> RoutingDecision {
        let tier = self.get_current_tier();
        let use_bnb = self.use_bnb_discount.load(Ordering::Relaxed);
        
        // Calculate maker cost (fee or rebate)
        let mut maker_fee = tier.maker_fee(notional_usd);
        if use_bnb && tier.has_bnb_discount(self.bnb_holdings.load(Ordering::Relaxed) as f64 / 1e8) {
            maker_fee = tier.apply_bnb_discount(maker_fee);
        }
        
        // Calculate taker cost (fee + spread impact)
        let mut taker_fee = tier.taker_fee(notional_usd);
        if use_bnb && tier.has_bnb_discount(self.bnb_holdings.load(Ordering::Relaxed) as f64 / 1e8) {
            taker_fee = tier.apply_bnb_discount(taker_fee);
        }
        
        // Taker also pays half-spread as implicit cost
        let spread_cost = notional_usd * (spread_bps / 10000.0) * 0.5;
        let total_taker_cost = taker_fee + spread_cost;
        
        // Decision logic
        let (execution_type, expected_fee, cost, is_optimal, reason) = if urgency_score > 0.8 {
            // High urgency: use taker despite higher cost
            (
                ExecutionType::Taker,
                tier.taker_fee_bps,
                total_taker_cost,
                false,
                "High urgency execution",
            )
        } else if maker_fee < 0.0 {
            // Maker rebate available
            (
                ExecutionType::Maker,
                tier.maker_fee_bps,
                maker_fee,
                true,
                "Maker rebate available",
            )
        } else if maker_fee < total_taker_cost {
            // Maker is cheaper
            (
                ExecutionType::Maker,
                tier.maker_fee_bps,
                maker_fee,
                true,
                "Lower cost as maker",
            )
        } else {
            // Taker might be worth it for speed
            (
                ExecutionType::Taker,
                tier.taker_fee_bps,
                total_taker_cost,
                urgency_score < 0.3,
                if urgency_score < 0.3 { "Cost optimized" } else { "Speed prioritized" },
            )
        };

        RoutingDecision {
            execution_type,
            expected_fee_bps: expected_fee,
            expected_cost_usd: cost.abs(),
            is_optimal,
            reason,
        }
    }

    /// Record a executed trade and update statistics
    #[inline]
    pub fn record_execution(&self, notional_usd: f64, execution_type: ExecutionType) {
        let tier = self.get_current_tier();
        let use_bnb = self.use_bnb_discount.load(Ordering::Relaxed);
        
        let fee = match execution_type {
            ExecutionType::Maker => {
                let mut f = tier.maker_fee(notional_usd);
                if use_bnb && tier.has_bnb_discount(self.bnb_holdings.load(Ordering::Relaxed) as f64 / 1e8) {
                    f = tier.apply_bnb_discount(f);
                }
                f
            }
            ExecutionType::Taker => {
                let mut f = tier.taker_fee(notional_usd);
                if use_bnb && tier.has_bnb_discount(self.bnb_holdings.load(Ordering::Relaxed) as f64 / 1e8) {
                    f = tier.apply_bnb_discount(f);
                }
                f
            }
        };

        if fee > 0.0 {
            self.total_fees_paid.fetch_add((fee * 1e6) as u64, Ordering::Relaxed);
        } else {
            self.total_rebates_earned.fetch_add((fee.abs() * 1e6) as u64, Ordering::Relaxed);
        }
        
        // Add to volume
        self.add_volume((notional_usd * 1e6) as u64);
    }

    /// Get fee savings estimate vs base tier
    #[inline]
    pub fn get_fee_savings(&self) -> FeeSavings {
        let current_tier = self.get_current_tier();
        let base_tier = &VIP_TIERS[0];
        
        let maker_savings = (base_tier.maker_fee_bps - current_tier.maker_fee_bps) as f64 / 10000.0;
        let taker_savings = (base_tier.taker_fee_bps - current_tier.taker_fee_bps) as f64 / 10000.0;
        
        FeeSavings {
            maker_savings_bps: maker_savings * 10000.0,
            taker_savings_bps: taker_savings * 10000.0,
            current_tier_index: self.get_tier_index(),
            volume_to_next_tier: self.get_volume_to_next_tier(),
        }
    }

    /// Get volume needed to reach next tier
    #[inline]
    pub fn get_volume_to_next_tier(&self) -> u64 {
        let current_idx = self.get_tier_index();
        if current_idx >= VIP_TIERS.len() - 1 {
            return 0;
        }
        
        let next_tier = &VIP_TIERS[current_idx + 1];
        let current_volume = self.volume_30d_usd.load(Ordering::Relaxed);
        
        next_tier.min_volume_usd.saturating_sub(current_volume)
    }

    /// Get comprehensive statistics
    #[inline]
    pub fn get_stats(&self) -> RouterStats {
        RouterStats {
            current_tier: self.get_tier_index(),
            volume_30d_usd: self.volume_30d_usd.load(Ordering::Relaxed) as f64 / 1e6,
            bnb_holdings: self.bnb_holdings.load(Ordering::Relaxed) as f64 / 1e8,
            bnb_discount_enabled: self.use_bnb_discount.load(Ordering::Relaxed),
            total_fees_paid: self.total_fees_paid.load(Ordering::Relaxed) as f64 / 1e6,
            total_rebates_earned: self.total_rebates_earned.load(Ordering::Relaxed) as f64 / 1e6,
            net_fees: (self.total_fees_paid.load(Ordering::Relaxed) as i64 
                      - self.total_rebates_earned.load(Ordering::Relaxed) as i64) as f64 / 1e6,
            volume_to_next_tier: self.get_volume_to_next_tier() as f64 / 1e6,
            is_valid: self.is_valid.load(Ordering::Relaxed),
        }
    }

    /// Invalidate router
    #[inline]
    pub fn invalidate(&self) {
        self.is_valid.store(false, Ordering::Relaxed);
    }
}

impl Default for MakerTakerRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Fee savings information
#[derive(Debug, Clone, Copy)]
pub struct FeeSavings {
    pub maker_savings_bps: f64,
    pub taker_savings_bps: f64,
    pub current_tier_index: usize,
    pub volume_to_next_tier: f64,
}

/// Router statistics
#[derive(Debug, Clone, Copy)]
pub struct RouterStats {
    pub current_tier: usize,
    pub volume_30d_usd: f64,
    pub bnb_holdings: f64,
    pub bnb_discount_enabled: bool,
    pub total_fees_paid: f64,
    pub total_rebates_earned: f64,
    pub net_fees: f64,
    pub volume_to_next_tier: f64,
    pub is_valid: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_router() {
        let router = MakerTakerRouter::new();
        
        assert_eq!(router.get_tier_index(), 0);
        assert!(!router.use_bnb_discount.load(Ordering::Relaxed));
        
        let stats = router.get_stats();
        assert_eq!(stats.current_tier, 0);
        assert_eq!(stats.volume_30d_usd, 0.0);
    }

    #[test]
    fn test_tier_progression() {
        let router = MakerTakerRouter::new();
        
        // Add volume to reach VIP 3
        router.update_volume(30_000_000_000);
        
        assert_eq!(router.get_tier_index(), 3);
        
        let tier = router.get_current_tier();
        assert_eq!(tier.maker_fee_bps, 7);
        assert_eq!(tier.taker_fee_bps, 9);
    }

    #[test]
    fn test_execution_decision_low_urgency() {
        let router = MakerTakerRouter::new();
        
        let decision = router.calculate_optimal_execution(
            10000.0, // $10k notional
            0.1,     // Low urgency
            5.0,     // 5 bps spread
        );
        
        // Should prefer maker for low urgency
        assert_eq!(decision.execution_type, ExecutionType::Maker);
        assert!(decision.is_optimal);
    }

    #[test]
    fn test_execution_decision_high_urgency() {
        let router = MakerTakerRouter::new();
        
        let decision = router.calculate_optimal_execution(
            10000.0, // $10k notional
            0.9,     // High urgency
            5.0,     // 5 bps spread
        );
        
        // Should use taker for high urgency
        assert_eq!(decision.execution_type, ExecutionType::Taker);
        assert!(!decision.is_optimal); // Not cost-optimal but speed-optimal
    }

    #[test]
    fn test_fee_recording() {
        let router = MakerTakerRouter::new();
        
        router.record_execution(10000.0, ExecutionType::Taker);
        
        let stats = router.get_stats();
        assert!(stats.total_fees_paid > 0.0);
        assert!(stats.volume_30d_usd > 0.0);
    }

    #[test]
    fn test_bnb_discount() {
        let router = MakerTakerRouter::new();
        router.set_bnb_discount(true);
        router.update_bnb_holdings(100.0); // 100 BNB
        
        router.record_execution(10000.0, ExecutionType::Taker);
        
        let stats = router.get_stats();
        // Fees should be lower with BNB discount
        assert!(stats.total_fees_paid < 10.0); // Less than $10 (0.1% of $10k)
    }

    #[test]
    fn test_volume_to_next_tier() {
        let router = MakerTakerRouter::new();
        router.update_volume(30_000_000_000); // VIP 3
        
        let volume_needed = router.get_volume_to_next_tier();
        // Need 20B more to reach VIP 4 (50B threshold)
        assert_eq!(volume_needed, 20_000_000_000);
    }
}
