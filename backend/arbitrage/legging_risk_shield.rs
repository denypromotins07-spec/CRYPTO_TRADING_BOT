//! Legging Risk Shield - Atomic Order Cancellation to Prevent Naked Exposure
//!
//! This module provides atomic-like order execution for multi-leg arbitrage trades.
//! If one leg fails to fill instantly, immediately cancels all other legs to prevent
//! naked directional exposure. Uses strict timeouts and monitoring.
//!
//! Chapter 4: Arbitrage Risk Management, Legging Risk, and SOUL.md Arb Logging

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::HashMap;

/// Maximum time allowed for leg execution (milliseconds)
pub const MAX_LEG_EXECUTION_MS: u64 = 500;

/// Maximum acceptable leg fill delay (milliseconds)
pub const MAX_FILL_DELAY_MS: u64 = 200;

/// Status of a trade leg
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LegStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Failed,
    Expired,
}

/// Represents a single leg of an arbitrage trade
#[derive(Clone, Debug)]
pub struct TradeLeg {
    /// Unique leg identifier
    pub leg_id: u64,
    /// Parent arb trade ID
    pub arb_id: u64,
    /// Symbol being traded
    pub symbol: &'static str,
    /// Side: true for buy, false for sell
    pub is_buy: bool,
    /// Quantity
    pub quantity: f64,
    /// Limit price (None for market orders)
    pub limit_price: Option<f64>,
    /// Current status
    pub status: LegStatus,
    /// Fill price (if filled)
    pub fill_price: Option<f64>,
    /// Fill quantity (may be partial)
    pub fill_quantity: f64,
    /// Submission timestamp (nanoseconds)
    pub submitted_at_ns: u64,
    /// Fill timestamp (nanoseconds)
    pub filled_at_ns: Option<u64>,
    /// Error message (if failed)
    pub error_message: Option<String>,
}

impl TradeLeg {
    /// Create a new pending leg
    pub fn new(leg_id: u64, arb_id: u64, symbol: &'static str, is_buy: bool, quantity: f64) -> Self {
        Self {
            leg_id,
            arb_id,
            symbol,
            is_buy,
            quantity,
            limit_price: None,
            status: LegStatus::Pending,
            fill_price: None,
            fill_quantity: 0.0,
            submitted_at_ns: 0,
            filled_at_ns: None,
            error_message: None,
        }
    }

    /// Check if leg is completely filled
    #[inline]
    pub fn is_filled(&self) -> bool {
        self.status == LegStatus::Filled && (self.fill_quantity - self.quantity).abs() < 1e-8
    }

    /// Check if leg has any fill
    #[inline]
    pub fn has_any_fill(&self) -> bool {
        self.fill_quantity > 1e-8
    }

    /// Get elapsed time since submission in milliseconds
    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        if self.submitted_at_ns == 0 {
            return 0;
        }
        
        (now_ns - self.submitted_at_ns) / 1_000_000
    }

    /// Check if leg has exceeded timeout
    #[inline]
    pub fn is_timed_out(&self) -> bool {
        self.elapsed_ms() > MAX_LEG_EXECUTION_MS
    }
}

/// Multi-leg arbitrage trade with atomic execution guarantees
#[derive(Debug)]
pub struct ArbTrade {
    /// Unique arb trade ID
    pub arb_id: u64,
    /// Legs of the arb trade
    pub legs: Vec<TradeLeg>,
    /// Creation timestamp
    pub created_at: Instant,
    /// Whether all legs are successfully filled
    pub is_complete: bool,
    /// Whether trade was aborted due to leg failure
    pub is_aborted: bool,
    /// Total slippage in basis points
    pub total_slippage_bps: f64,
}

impl ArbTrade {
    /// Create a new arb trade with given legs
    pub fn new(arb_id: u64, legs: Vec<TradeLeg>) -> Self {
        Self {
            arb_id,
            legs,
            created_at: Instant::now(),
            is_complete: false,
            is_aborted: false,
            total_slippage_bps: 0.0,
        }
    }

    /// Check if all legs are filled
    #[inline]
    pub fn all_legs_filled(&self) -> bool {
        self.legs.iter().all(|leg| leg.is_filled())
    }

    /// Check if any leg has failed
    #[inline]
    pub fn any_leg_failed(&self) -> bool {
        self.legs.iter().any(|leg| {
            matches!(leg.status, LegStatus::Failed | LegStatus::Expired)
        })
    }

    /// Check if any leg is timed out
    #[inline]
    pub fn any_leg_timed_out(&self) -> bool {
        self.legs.iter().any(|leg| leg.is_timed_out())
    }

    /// Get count of filled legs
    #[inline]
    pub fn filled_leg_count(&self) -> usize {
        self.legs.iter().filter(|leg| leg.is_filled()).count()
    }

    /// Get count of legs with any fill
    #[inline]
    pub fn partially_filled_count(&self) -> usize {
        self.legs.iter().filter(|leg| leg.has_any_fill()).count()
    }
}

/// High-performance legging risk shield
pub struct LeggingRiskShield {
    /// Active arb trades
    active_trades: HashMap<u64, ArbTrade>,
    /// Next leg ID counter
    next_leg_id: AtomicU64,
    /// Next arb ID counter
    next_arb_id: AtomicU64,
    /// Whether shield is enabled
    enabled: AtomicBool,
    /// Statistics
    total_arbs_initiated: AtomicU64,
    successful_completions: AtomicU64,
    aborted_trades: AtomicU64,
    leg_failures: AtomicU64,
    cancellations_executed: AtomicU64,
    /// Maximum concurrent arb trades
    max_concurrent_arbs: usize,
}

impl LeggingRiskShield {
    /// Create a new legging risk shield
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            active_trades: HashMap::with_capacity(max_concurrent),
            next_leg_id: AtomicU64::new(1),
            next_arb_id: AtomicU64::new(1),
            enabled: AtomicBool::new(true),
            total_arbs_initiated: AtomicU64::new(0),
            successful_completions: AtomicU64::new(0),
            aborted_trades: AtomicU64::new(0),
            leg_failures: AtomicU64::new(0),
            cancellations_executed: AtomicU64::new(0),
            max_concurrent_arbs: max_concurrent,
        }
    }

    /// Generate unique leg ID
    #[inline]
    pub fn generate_leg_id(&self) -> u64 {
        self.next_leg_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Generate unique arb ID
    #[inline]
    pub fn generate_arb_id(&self) -> u64 {
        self.next_arb_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Create a new arb trade with multiple legs
    #[inline]
    pub fn create_arb_trade(&mut self, legs: Vec<TradeLeg>) -> Option<u64> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }

        if self.active_trades.len() >= self.max_concurrent_arbs {
            return None; // At capacity
        }

        let arb_id = self.generate_arb_id();
        
        // Set arb_id on all legs
        let mut legs_with_id = legs;
        for leg in &mut legs_with_id {
            leg.arb_id = arb_id;
        }

        let trade = ArbTrade::new(arb_id, legs_with_id);
        self.active_trades.insert(arb_id, trade);
        self.total_arbs_initiated.fetch_add(1, Ordering::SeqCst);

        Some(arb_id)
    }

    /// Update leg status - call this when receiving fill/cancel responses
    #[inline]
    pub fn update_leg_status(
        &mut self,
        arb_id: u64,
        leg_id: u64,
        status: LegStatus,
        fill_price: Option<f64>,
        fill_quantity: f64,
    ) -> Option<LegAction> {
        let trade = self.active_trades.get_mut(&arb_id)?;

        let leg = trade.legs.iter_mut().find(|l| l.leg_id == leg_id)?;
        leg.status = status;
        leg.fill_price = fill_price;
        leg.fill_quantity = fill_quantity;

        if status == LegStatus::Filled {
            leg.filled_at_ns = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64
            );
        }

        // Check if we need to abort
        if self.should_abort_trade(trade) {
            Some(LegAction::AbortTrade)
        } else if trade.all_legs_filled() {
            trade.is_complete = true;
            Some(LegAction::CompleteTrade)
        } else {
            Some(LegAction::Continue)
        }
    }

    /// Check if a trade should be aborted
    #[inline]
    fn should_abort_trade(&self, trade: &ArbTrade) -> bool {
        // Abort conditions:
        // 1. Any leg has failed
        // 2. Any leg is timed out
        // 3. Partial fill with no progress for too long

        if trade.any_leg_failed() {
            return true;
        }

        if trade.any_leg_timed_out() {
            return true;
        }

        // Check for dangerous partial fills
        // If some legs are filled but others aren't making progress
        let filled_count = trade.filled_leg_count();
        let partial_count = trade.partially_filled_count();

        if filled_count > 0 && filled_count < trade.legs.len() {
            // Some legs filled, check if remaining legs are stuck
            for leg in &trade.legs {
                if !leg.is_filled() && leg.is_timed_out() {
                    return true;
                }
            }
        }

        false
    }

    /// Abort a trade and cancel all unfilled legs
    #[inline]
    pub fn abort_trade(&mut self, arb_id: u64) -> Vec<CancelOrder> {
        let mut cancel_orders = Vec::new();

        if let Some(trade) = self.active_trades.get_mut(&arb_id) {
            trade.is_aborted = true;

            for leg in &trade.legs {
                if !leg.is_filled() && leg.status != LegStatus::Cancelled {
                    cancel_orders.push(CancelOrder {
                        leg_id: leg.leg_id,
                        symbol: leg.symbol,
                        reason: "arb_aborted",
                    });
                    self.cancellations_executed.fetch_add(1, Ordering::SeqCst);
                }
            }

            self.aborted_trades.fetch_add(1, Ordering::SeqCst);
            
            // Count leg failures
            for leg in &trade.legs {
                if matches!(leg.status, LegStatus::Failed | LegStatus::Expired) {
                    self.leg_failures.fetch_add(1, Ordering::SeqCst);
                }
            }
        }

        cancel_orders
    }

    /// Complete a trade and remove from active
    #[inline]
    pub fn complete_trade(&mut self, arb_id: u64) -> Option<ArbTrade> {
        if let Some(trade) = self.active_trades.remove(&arb_id) {
            self.successful_completions.fetch_add(1, Ordering::SeqCst);
            Some(trade)
        } else {
            None
        }
    }

    /// Monitor all active trades and return required actions
    #[inline]
    pub fn monitor_active_trades(&mut self) -> Vec<(u64, LegAction)> {
        let mut actions = Vec::new();
        let arb_ids: Vec<u64> = self.active_trades.keys().copied().collect();

        for arb_id in arb_ids {
            if let Some(trade) = self.active_trades.get(&arb_id) {
                if self.should_abort_trade(trade) {
                    actions.push((arb_id, LegAction::AbortTrade));
                } else if trade.all_legs_filled() {
                    actions.push((arb_id, LegAction::CompleteTrade));
                }
            }
        }

        actions
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> RiskShieldStats {
        let total = self.total_arbs_initiated.load(Ordering::Acquire);
        let successful = self.successful_completions.load(Ordering::Acquire);
        let aborted = self.aborted_trades.load(Ordering::Acquire);

        RiskShieldStats {
            total_arbs: total,
            successful_completions: successful,
            aborted_trades: aborted,
            leg_failures: self.leg_failures.load(Ordering::Acquire),
            cancellations: self.cancellations_executed.load(Ordering::Acquire),
            success_rate: if total > 0 { successful as f64 / total as f64 } else { 0.0 },
            active_count: self.active_trades.len(),
        }
    }

    /// Enable/disable the shield
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    /// Check if shield is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }
}

/// Action to take for a leg/trade
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LegAction {
    Continue,
    AbortTrade,
    CompleteTrade,
}

/// Cancel order instruction
#[derive(Clone, Debug)]
pub struct CancelOrder {
    pub leg_id: u64,
    pub symbol: &'static str,
    pub reason: &'static str,
}

/// Statistics for the risk shield
#[derive(Clone, Copy, Debug)]
pub struct RiskShieldStats {
    pub total_arbs: u64,
    pub successful_completions: u64,
    pub aborted_trades: u64,
    pub leg_failures: u64,
    pub cancellations: u64,
    pub success_rate: f64,
    pub active_count: usize,
}

/// Flash crash detector for emergency liquidation
pub struct FlashCrashDetector {
    /// Price history per symbol (circular buffer)
    price_history: HashMap<&'static str, VecDeque<f64>>,
    /// Maximum history size
    max_history: usize,
    /// Crash threshold (percentage drop)
    crash_threshold_pct: f64,
    /// Time window for crash detection (milliseconds)
    detection_window_ms: u64,
    /// Last crash alert timestamp
    last_alert_ns: AtomicU64,
    /// Cooldown between alerts (milliseconds)
    alert_cooldown_ms: u64,
}

use std::collections::VecDeque;

impl FlashCrashDetector {
    /// Create a new flash crash detector
    pub fn new(crash_threshold_pct: f64, detection_window_ms: u64) -> Self {
        Self {
            price_history: HashMap::new(),
            max_history: 100,
            crash_threshold_pct,
            detection_window_ms,
            last_alert_ns: AtomicU64::new(0),
            alert_cooldown_ms: 5000, // 5 second cooldown
        }
    }

    /// Add price observation
    #[inline]
    pub fn add_price(&mut self, symbol: &'static str, price: f64) {
        let history = self.price_history.entry(symbol).or_insert_with(|| {
            VecDeque::with_capacity(self.max_history)
        });

        history.push_back(price);

        if history.len() > self.max_history {
            history.pop_front();
        }
    }

    /// Check for flash crash condition
    #[inline]
    pub fn detect_crash(&self, symbol: &str) -> Option<FlashCrashAlert> {
        let history = match self.price_history.get(symbol) {
            Some(h) if h.len() >= 10 => h,
            _ => return None,
        };

        // Compare current price to rolling maximum
        let current_price = history.back().copied()?;
        let max_price = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let pct_drop = (max_price - current_price) / max_price * 100.0;

        if pct_drop >= self.crash_threshold_pct {
            // Check cooldown
            let now_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            
            let last_alert = self.last_alert_ns.load(Ordering::Acquire);
            let elapsed_ms = (now_ns - last_alert) / 1_000_000;

            if elapsed_ms >= self.alert_cooldown_ms {
                self.last_alert_ns.store(now_ns, Ordering::SeqCst);
                
                return Some(FlashCrashAlert {
                    symbol,
                    pct_drop,
                    max_price,
                    current_price,
                    timestamp_ns: now_ns,
                });
            }
        }

        None
    }

    /// Emergency liquidation signal
    #[inline]
    pub fn should_liquidate_all(&self) -> bool {
        // Check multiple symbols for simultaneous crashes
        let crash_count = self.price_history.keys()
            .filter(|&&sym| self.detect_crash(sym).is_some())
            .count();

        // If 2+ symbols crashing simultaneously, liquidate all
        crash_count >= 2
    }
}

/// Flash crash alert
#[derive(Clone, Copy, Debug)]
pub struct FlashCrashAlert {
    pub symbol: &'static str,
    pub pct_drop: f64,
    pub max_price: f64,
    pub current_price: f64,
    pub timestamp_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_shield_basic() {
        let mut shield = LeggingRiskShield::new(10);

        // Create a 2-leg arb trade
        let leg1 = TradeLeg::new(1, 0, "BTC", true, 0.1);
        let leg2 = TradeLeg::new(2, 0, "ETH", false, 1.0);

        let arb_id = shield.create_arb_trade(vec![leg1, leg2]).unwrap();

        // Update first leg as filled
        let action = shield.update_leg_status(arb_id, 1, LegStatus::Filled, Some(50000.0), 0.1);
        assert_eq!(action, Some(LegAction::Continue));

        // Update second leg as filled
        let action = shield.update_leg_status(arb_id, 2, LegStatus::Filled, Some(3000.0), 1.0);
        assert_eq!(action, Some(LegAction::CompleteTrade));

        // Complete the trade
        let trade = shield.complete_trade(arb_id);
        assert!(trade.is_some());
        assert!(trade.unwrap().is_complete);
    }

    #[test]
    fn test_risk_shield_abort_on_failure() {
        let mut shield = LeggingRiskShield::new(10);

        let leg1 = TradeLeg::new(1, 0, "BTC", true, 0.1);
        let leg2 = TradeLeg::new(2, 0, "ETH", false, 1.0);

        let arb_id = shield.create_arb_trade(vec![leg1, leg2]).unwrap();

        // First leg fills
        shield.update_leg_status(arb_id, 1, LegStatus::Filled, Some(50000.0), 0.1);

        // Second leg fails
        let action = shield.update_leg_status(arb_id, 2, LegStatus::Failed, None, 0.0);
        assert_eq!(action, Some(LegAction::AbortTrade));

        // Get cancellation orders
        let cancels = shield.abort_trade(arb_id);
        assert!(!cancels.is_empty());

        let stats = shield.get_stats();
        assert_eq!(stats.aborted_trades, 1);
    }
}
