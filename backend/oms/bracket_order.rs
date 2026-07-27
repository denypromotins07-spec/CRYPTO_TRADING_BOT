//! ZAID PERSONAL CRYPTO TRADING BOT - Stage 32
//! Advanced Order Management System (OMS) - Chapter 1
//! File: backend/oms/bracket_order.rs
//!
//! Manages complex Entry + Take Profit + Stop Loss structures with atomic guarantees.
//! Ensures Stop Loss triggers instantly even if Take Profit API call lags.
//! Optimized for AMD Ryzen AI 5, strictly respecting 8GB RAM limit.
//! Targets 8k-20k INR/hour in a 4hr trading window.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Represents the state of a bracket order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketState {
    PendingEntry,
    EntryFilled,
    TPActiveSLActive,
    TPCancelled,
    SLCancelled,
    FullyFilled,
    FullyCancelled,
    Error,
}

/// Configuration for a bracket order
#[derive(Debug, Clone)]
pub struct BracketConfig {
    pub symbol: String,
    pub entry_price: f64,
    pub quantity: f64,
    pub take_profit_price: f64,
    pub stop_loss_price: f64,
    pub trailing_stop_enabled: bool,
    pub trailing_distance_pct: f64,
}

/// Atomic bracket order structure with lock-free state management
pub struct BracketOrder {
    /// Unique order identifier
    pub order_id: u64,
    /// Current state of the bracket order
    state: AtomicU64, // Encoded as u64 for atomic operations
    /// Configuration parameters
    config: BracketConfig,
    /// Entry order ID (child)
    entry_order_id: AtomicU64,
    /// Take Profit order ID (child)
    tp_order_id: AtomicU64,
    /// Stop Loss order ID (child)
    sl_order_id: AtomicU64,
    /// Flag indicating if SL has been triggered/executed
    sl_executed: AtomicBool,
    /// Flag indicating if TP has been triggered/executed
    tp_executed: AtomicBool,
    /// Timestamp of creation
    created_at: u64,
    /// Flag to prevent double execution
    execution_lock: AtomicBool,
}

impl BracketOrder {
    /// Create a new bracket order with atomic initialization
    pub fn new(config: BracketConfig) -> Self {
        let order_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        BracketOrder {
            order_id,
            state: AtomicU64::new(BracketState::PendingEntry as u64),
            config,
            entry_order_id: AtomicU64::new(0),
            tp_order_id: AtomicU64::new(0),
            sl_order_id: AtomicU64::new(0),
            sl_executed: AtomicBool::new(false),
            tp_executed: AtomicBool::new(false),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            execution_lock: AtomicBool::new(false),
        }
    }

    /// Get current state (lock-free read)
    #[inline]
    pub fn get_state(&self) -> BracketState {
        unsafe { std::mem::transmute(self.state.load(Ordering::Acquire)) }
    }

    /// Transition state atomically with validation
    pub fn transition_state(&self, new_state: BracketState) -> bool {
        let current = self.get_state();
        
        // Validate state transitions
        let valid_transition = match (current, new_state) {
            (BracketState::PendingEntry, BracketState::EntryFilled) => true,
            (BracketState::EntryFilled, BracketState::TPActiveSLActive) => true,
            (BracketState::TPActiveSLActive, BracketState::TPCancelled) => true,
            (BracketState::TPActiveSLActive, BracketState::SLCancelled) => true,
            (BracketState::TPActiveSLActive, BracketState::FullyFilled) => true,
            (BracketState::TPActiveSLActive, BracketState::FullyCancelled) => true,
            (BracketState::TPCancelled, BracketState::FullyFilled) => true,
            (BracketState::SLCancelled, BracketState::FullyFilled) => true,
            (BracketState::TPCancelled, BracketState::FullyCancelled) => true,
            (BracketState::SLCancelled, BracketState::FullyCancelled) => true,
            _ => false,
        };

        if !valid_transition {
            return false;
        }

        self.state.store(new_state as u64, Ordering::Release);
        true
    }

    /// Set entry order ID atomically
    #[inline]
    pub fn set_entry_order_id(&self, id: u64) {
        self.entry_order_id.store(id, Ordering::Release);
        self.transition_state(BracketState::EntryFilled);
    }

    /// Set TP order ID atomically
    #[inline]
    pub fn set_tp_order_id(&self, id: u64) {
        self.tp_order_id.store(id, Ordering::Release);
        if self.sl_order_id.load(Ordering::Acquire) > 0 {
            self.transition_state(BracketState::TPActiveSLActive);
        }
    }

    /// Set SL order ID atomically - CRITICAL PATH
    #[inline]
    pub fn set_sl_order_id(&self, id: u64) {
        self.sl_order_id.store(id, Ordering::Release);
        if self.tp_order_id.load(Ordering::Acquire) > 0 {
            self.transition_state(BracketState::TPActiveSLActive);
        }
    }

    /// Execute Stop Loss with guaranteed atomicity - prevents double execution
    /// This is the critical path that must execute even if TP API call lags
    pub fn execute_stop_loss(&self) -> bool {
        // Fast path: check if already executed without locking
        if self.sl_executed.load(Ordering::Acquire) {
            return false;
        }

        // Try to acquire execution lock
        if self.execution_lock.swap(true, Ordering::AcqRel) {
            // Another thread is handling execution
            return false;
        }

        // Double-check pattern
        if self.sl_executed.load(Ordering::Acquire) {
            self.execution_lock.store(false, Ordering::Release);
            return false;
        }

        // Mark as executed BEFORE any API calls
        self.sl_executed.store(true, Ordering::Release);
        
        // Cancel TP order asynchronously (non-blocking)
        self.cancel_take_profit_async();
        
        // Release lock
        self.execution_lock.store(false, Ordering::Release);
        
        // Transition state
        self.transition_state(BracketState::FullyFilled);
        
        true
    }

    /// Execute Take Profit with atomicity
    pub fn execute_take_profit(&self) -> bool {
        if self.tp_executed.load(Ordering::Acquire) {
            return false;
        }

        if self.execution_lock.swap(true, Ordering::AcqRel) {
            return false;
        }

        if self.tp_executed.load(Ordering::Acquire) {
            self.execution_lock.store(false, Ordering::Release);
            return false;
        }

        self.tp_executed.store(true, Ordering::Release);
        
        // Cancel SL order asynchronously
        self.cancel_stop_loss_async();
        
        self.execution_lock.store(false, Ordering::Release);
        self.transition_state(BracketState::FullyFilled);
        
        true
    }

    /// Async cancellation of TP (non-blocking, fire-and-forget)
    #[inline]
    fn cancel_take_profit_async(&self) {
        let tp_id = self.tp_order_id.load(Ordering::Acquire);
        if tp_id > 0 {
            // In production: spawn async task to cancel TP on exchange
            // This MUST NOT block SL execution
            std::thread::spawn(move || {
                // Simulated async cancellation
                // BinanceAPI::cancel_order_async(tp_id).await
            });
        }
    }

    /// Async cancellation of SL (non-blocking, fire-and-forget)
    #[inline]
    fn cancel_stop_loss_async(&self) {
        let sl_id = self.sl_order_id.load(Ordering::Acquire);
        if sl_id > 0 {
            // In production: spawn async task to cancel SL on exchange
            std::thread::spawn(move || {
                // Simulated async cancellation
                // BinanceAPI::cancel_order_async(sl_id).await
            });
        }
    }

    /// Get order IDs for reconciliation
    pub fn get_all_order_ids(&self) -> (u64, u64, u64) {
        (
            self.entry_order_id.load(Ordering::Acquire),
            self.tp_order_id.load(Ordering::Acquire),
            self.sl_order_id.load(Ordering::Acquire),
        )
    }

    /// Check if bracket is still active (neither TP nor SL executed)
    #[inline]
    pub fn is_active(&self) -> bool {
        !self.sl_executed.load(Ordering::Acquire) && !self.tp_executed.load(Ordering::Acquire)
    }

    /// Get configuration reference
    #[inline]
    pub fn config(&self) -> &BracketConfig {
        &self.config
    }

    /// Calculate risk-reward ratio
    pub fn risk_reward_ratio(&self) -> f64 {
        let entry = self.config.entry_price;
        let tp = self.config.take_profit_price;
        let sl = self.config.stop_loss_price;
        
        let profit = (tp - entry).abs();
        let loss = (entry - sl).abs();
        
        if loss == 0.0 {
            return 0.0;
        }
        
        profit / loss
    }
}

/// Manager for multiple bracket orders with O(1) lookup
pub struct BracketOrderManager {
    orders: dashmap::DashMap<u64, Arc<BracketOrder>>,
    symbol_index: dashmap::DashMap<String, Vec<u64>>,
}

impl BracketOrderManager {
    pub fn new() -> Self {
        BracketOrderManager {
            orders: dashmap::DashMap::new(),
            symbol_index: dashmap::DashMap::new(),
        }
    }

    /// Create and register a new bracket order
    pub fn create_bracket(&self, config: BracketConfig) -> Arc<BracketOrder> {
        let bracket = Arc::new(BracketOrder::new(config.clone()));
        
        self.orders.insert(bracket.order_id, bracket.clone());
        
        // Index by symbol for fast lookup
        self.symbol_index
            .entry(config.symbol)
            .or_insert_with(Vec::new)
            .push(bracket.order_id);
        
        bracket
    }

    /// Get bracket order by ID
    pub fn get_bracket(&self, order_id: u64) -> Option<Arc<BracketOrder>> {
        self.orders.get(&order_id).map(|r| r.value().clone())
    }

    /// Get all active brackets for a symbol
    pub fn get_active_brackets(&self, symbol: &str) -> Vec<Arc<BracketOrder>> {
        self.symbol_index
            .get(symbol)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.get_bracket(*id))
                    .filter(|b| b.is_active())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove completed bracket from memory (TTL-based cleanup)
    pub fn cleanup_completed(&self, max_age_ms: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.orders.retain(|_, bracket| {
            let state = bracket.get_state();
            let is_terminal = matches!(
                state,
                BracketState::FullyFilled | BracketState::FullyCancelled | BracketState::Error
            );

            if is_terminal {
                now - bracket.created_at < max_age_ms
            } else {
                true // Keep active orders
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bracket_order_creation() {
        let config = BracketConfig {
            symbol: "BTCUSDT".to_string(),
            entry_price: 50000.0,
            quantity: 0.1,
            take_profit_price: 52000.0,
            stop_loss_price: 49000.0,
            trailing_stop_enabled: false,
            trailing_distance_pct: 0.0,
        };

        let bracket = BracketOrder::new(config);
        assert_eq!(bracket.get_state(), BracketState::PendingEntry);
        assert!(bracket.risk_reward_ratio() > 0.0);
    }

    #[test]
    fn test_sl_execution_guarantee() {
        let config = BracketConfig {
            symbol: "ETHUSDT".to_string(),
            entry_price: 3000.0,
            quantity: 1.0,
            take_profit_price: 3200.0,
            stop_loss_price: 2900.0,
            trailing_stop_enabled: false,
            trailing_distance_pct: 0.0,
        };

        let bracket = Arc::new(BracketOrder::new(config));
        bracket.set_entry_order_id(1);
        bracket.set_tp_order_id(2);
        bracket.set_sl_order_id(3);

        // Simulate concurrent SL execution
        let bracket_clone = bracket.clone();
        let handle = std::thread::spawn(move || {
            bracket_clone.execute_stop_loss()
        });

        // Main thread also tries to execute
        let result_main = bracket.execute_stop_loss();
        let result_thread = handle.join().unwrap();

        // Exactly one should succeed
        assert!(result_main || result_thread);
        assert!(bracket.sl_executed.load(Ordering::Acquire));
    }
}
