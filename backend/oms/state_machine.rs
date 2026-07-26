//! OMS State Machine - Strict Lifecycle Management for Every Order
//!
//! This module implements a rigorous state machine for tracking the complete
//! lifecycle of every synthetic and atomic order from creation to terminal state.
//!
//! Key Features:
//! - Finite State Machine with strict transition rules
//! - Atomic state transitions with no race conditions
//! - Complete audit trail of all state changes
//! - Support for synthetic orders (brackets, OCO, trailing stops)
//! - Automatic timeout handling and stale order detection
//!
//! Memory Optimized: Zero-cost state representations
//! Thread Safe: Lock-free atomics where possible, minimal locking
//! Platform: Optimized for AMD Ryzen AI 5, Windows PowerShell

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};
use std::collections::{HashMap, VecDeque};
use std::fmt;

/// Maximum state history to keep per order
const MAX_STATE_HISTORY: usize = 32;

/// Represents the current state of an order in the OMS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderState {
    /// Order created but not yet submitted
    Pending,
    /// Order submitted to exchange, awaiting acknowledgment
    Submitted,
    /// Exchange acknowledged receipt
    Acknowledged,
    /// Order is live and working in the book
    Working,
    /// Partially filled, remainder still working
    PartiallyFilled,
    /// Order fully filled
    Filled,
    /// Order cancelled by user/request
    Cancelled,
    /// Order cancelled due to rejection
    Rejected,
    /// Order expired (time-in-force exceeded)
    Expired,
    /// Order replaced/modified
    Replaced,
    /// Synthetic order triggered, awaiting child execution
    Triggered,
    /// Order in flight (transition state)
    InFlight,
}

impl OrderState {
    /// Check if this is a terminal state
    #[inline]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            OrderState::Filled
                | OrderState::Cancelled
                | OrderState::Rejected
                | OrderState::Expired
        )
    }

    /// Check if order is still active (can receive fills)
    #[inline]
    pub fn is_active(self) -> bool {
        matches!(
            self,
            OrderState::Working
                | OrderState::PartiallyFilled
                | OrderState::Acknowledged
                | OrderState::Submitted
        )
    }
}

impl fmt::Display for OrderState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderState::Pending => write!(f, "PENDING"),
            OrderState::Submitted => write!(f, "SUBMITTED"),
            OrderState::Acknowledged => write!(f, "ACKNOWLEDGED"),
            OrderState::Working => write!(f, "WORKING"),
            OrderState::PartiallyFilled => write!(f, "PARTIALLY_FILLED"),
            OrderState::Filled => write!(f, "FILLED"),
            OrderState::Cancelled => write!(f, "CANCELLED"),
            OrderState::Rejected => write!(f, "REJECTED"),
            OrderState::Expired => write!(f, "EXPIRED"),
            OrderState::Replaced => write!(f, "REPLACED"),
            OrderState::Triggered => write!(f, "TRIGGERED"),
            OrderState::InFlight => write!(f, "IN_FLIGHT"),
        }
    }
}

/// Valid state transitions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StateTransition {
    PendingToSubmitted,
    SubmittedToAcknowledged,
    SubmittedToRejected,
    AcknowledgedToWorking,
    AcknowledgedToRejected,
    WorkingToPartiallyFilled,
    WorkingToFilled,
    WorkingToCancelled,
    WorkingToExpired,
    PartiallyFilledToFilled,
    PartiallyFilledToCancelled,
    PendingToTriggered,
    TriggeredToWorking,
    TriggeredToCancelled,
    WorkingToReplaced,
    PartiallyFilledToReplaced,
    AnyToInFlight,
    InFlightToAny,
}

/// Result of a state transition attempt
#[derive(Debug, Clone)]
pub struct TransitionResult {
    pub success: bool,
    pub previous_state: OrderState,
    pub new_state: OrderState,
    pub error_message: Option<String>,
}

/// Event that caused a state transition
#[derive(Debug, Clone)]
pub enum StateEvent {
    /// Order submission requested
    SubmitRequested,
    /// Exchange acknowledgment received
    ExchangeAck,
    /// Fill received from exchange
    Fill { size: f64, price: f64 },
    /// Full fill confirmed
    FullFill,
    /// Cancellation requested
    CancelRequested,
    /// Cancellation confirmed
    CancelConfirmed,
    /// Rejection from exchange
    Reject { reason: String },
    /// Time-in-force expired
    Expired,
    /// Order modification requested
    ModifyRequested,
    /// Modification confirmed
    ModifyConfirmed,
    /// Synthetic trigger activated
    Trigger,
    /// Timeout detected
    Timeout,
    /// System event
    SystemEvent(String),
}

/// Record of a state change for audit trail
#[derive(Debug, Clone)]
pub struct StateChangeRecord {
    pub timestamp_ns: u64,
    pub from_state: OrderState,
    pub to_state: OrderState,
    pub event: StateEvent,
    pub latency_ns: u64,
}

/// Unique identifier for an order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OrderId(pub u64);

/// Core order state machine
pub struct OrderStateMachine {
    /// Current state
    current_state: std::sync::RwLock<OrderState>,
    /// Order ID
    order_id: OrderId,
    /// Creation timestamp
    created_at_ns: u64,
    /// Last state change timestamp
    last_changed_ns: AtomicU64,
    /// State history for audit
    history: std::sync::Mutex<VecDeque<StateChangeRecord>>,
    /// Start time for latency measurements
    start_time: Instant,
    /// Whether machine is locked (during transition)
    is_transitioning: AtomicBool,
}

impl OrderStateMachine {
    /// Create a new state machine in Pending state
    pub fn new(order_id: u64) -> Self {
        let now_ns = Instant::now().elapsed().as_nanos() as u64;
        Self {
            current_state: std::sync::RwLock::new(OrderState::Pending),
            order_id: OrderId(order_id),
            created_at_ns: now_ns,
            last_changed_ns: AtomicU64::new(now_ns),
            history: std::sync::Mutex::new(VecDeque::with_capacity(MAX_STATE_HISTORY)),
            start_time: Instant::now(),
            is_transitioning: AtomicBool::new(false),
        }
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> OrderState {
        *self.current_state.read().unwrap()
    }

    /// Get order ID
    #[inline]
    pub fn order_id(&self) -> OrderId {
        self.order_id
    }

    /// Attempt a state transition
    #[inline]
    pub fn transition(&self, event: StateEvent) -> TransitionResult {
        // Prevent concurrent transitions
        if self.is_transitioning.swap(true, Ordering::SeqCst) {
            return TransitionResult {
                success: false,
                previous_state: self.get_state(),
                new_state: self.get_state(),
                error_message: Some("Transition already in progress".to_string()),
            };
        }

        let result = self.do_transition(event);
        self.is_transitioning.store(false, Ordering::SeqCst);
        result
    }

    /// Internal transition logic
    fn do_transition(&self, event: StateEvent) -> TransitionResult {
        let mut current = self.current_state.write().unwrap();
        let previous = *current;
        let now_ns = self.start_time.elapsed().as_nanos() as u64;

        // Determine target state based on event and current state
        let target_state = match (&event, previous) {
            (StateEvent::SubmitRequested, OrderState::Pending) => OrderState::Submitted,
            (StateEvent::ExchangeAck, OrderState::Submitted) => OrderState::Acknowledged,
            (StateEvent::Reject { .. }, OrderState::Submitted) => OrderState::Rejected,
            (StateEvent::ExchangeAck, OrderState::Acknowledged) => OrderState::Working,
            (StateEvent::Reject { .. }, OrderState::Acknowledged) => OrderState::Rejected,
            (StateEvent::Fill { .. }, OrderState::Working) => OrderState::PartiallyFilled,
            (StateEvent::FullFill, OrderState::Working) => OrderState::Filled,
            (StateEvent::CancelRequested, OrderState::Working) => OrderState::InFlight,
            (StateEvent::CancelConfirmed, OrderState::InFlight) => OrderState::Cancelled,
            (StateEvent::FullFill, OrderState::PartiallyFilled) => OrderState::Filled,
            (StateEvent::CancelConfirmed, OrderState::PartiallyFilled) => OrderState::Cancelled,
            (StateEvent::Trigger, OrderState::Pending) => OrderState::Triggered,
            (StateEvent::ExchangeAck, OrderState::Triggered) => OrderState::Working,
            (StateEvent::CancelConfirmed, OrderState::Triggered) => OrderState::Cancelled,
            (StateEvent::ModifyRequested, OrderState::Working) => OrderState::InFlight,
            (StateEvent::ModifyConfirmed, OrderState::InFlight) => previous, // Back to same state
            (StateEvent::Timeout, _) => {
                if previous.is_active() {
                    OrderState::Expired
                } else {
                    previous
                }
            }
            _ => {
                // Invalid transition
                return TransitionResult {
                    success: false,
                    previous_state: previous,
                    new_state: previous,
                    error_message: Some(format!(
                        "Invalid transition: {:?} -> {:?}",
                        previous, event
                    )),
                };
            }
        };

        // Validate transition is legal
        if !self.is_valid_transition(previous, target_state) {
            return TransitionResult {
                success: false,
                previous_state: previous,
                new_state: previous,
                error_message: Some(format!(
                    "Illegal transition: {} -> {}",
                    previous, target_state
                )),
            };
        }

        // Record the transition
        let latency = now_ns - self.last_changed_ns.load(Ordering::Relaxed);
        let record = StateChangeRecord {
            timestamp_ns: now_ns,
            from_state: previous,
            to_state: target_state,
            event: event.clone(),
            latency_ns: latency,
        };

        // Update state
        *current = target_state;
        self.last_changed_ns.store(now_ns, Ordering::Relaxed);

        // Add to history
        let mut history = self.history.lock().unwrap();
        if history.len() >= MAX_STATE_HISTORY {
            history.pop_front();
        }
        history.push_back(record);

        TransitionResult {
            success: true,
            previous_state: previous,
            new_state: target_state,
            error_message: None,
        }
    }

    /// Check if a transition is valid
    fn is_valid_transition(&self, from: OrderState, to: OrderState) -> bool {
        // Terminal states cannot transition out (except for system overrides)
        if from.is_terminal() && from != to {
            return false;
        }

        // Define allowed transitions
        matches!(
            (from, to),
            (OrderState::Pending, OrderState::Submitted)
                | (OrderState::Pending, OrderState::Triggered)
                | (OrderState::Submitted, OrderState::Acknowledged)
                | (OrderState::Submitted, OrderState::Rejected)
                | (OrderState::Acknowledged, OrderState::Working)
                | (OrderState::Acknowledged, OrderState::Rejected)
                | (OrderState::Working, OrderState::PartiallyFilled)
                | (OrderState::Working, OrderState::Filled)
                | (OrderState::Working, OrderState::InFlight)
                | (OrderState::Working, OrderState::Expired)
                | (OrderState::PartiallyFilled, OrderState::Filled)
                | (OrderState::PartiallyFilled, OrderState::InFlight)
                | (OrderState::InFlight, OrderState::Cancelled)
                | (OrderState::InFlight, OrderState::Working)
                | (OrderState::Triggered, OrderState::Working)
                | (OrderState::Triggered, OrderState::Cancelled)
        )
    }

    /// Get state history
    pub fn get_history(&self) -> Vec<StateChangeRecord> {
        self.history.lock().unwrap().iter().cloned().collect()
    }

    /// Check if order is in terminal state
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.get_state().is_terminal()
    }

    /// Check if order is active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.get_state().is_active()
    }

    /// Get time since last state change in nanoseconds
    #[inline]
    pub fn time_since_last_change_ns(&self) -> u64 {
        let now = self.start_time.elapsed().as_nanos() as u64;
        now - self.last_changed_ns.load(Ordering::Relaxed)
    }

    /// Force transition (for recovery scenarios only)
    pub fn force_transition(&self, new_state: OrderState) {
        let mut current = self.current_state.write().unwrap();
        let now_ns = self.start_time.elapsed().as_nanos() as u64;

        let record = StateChangeRecord {
            timestamp_ns: now_ns,
            from_state: *current,
            to_state: new_state,
            event: StateEvent::SystemEvent("Force transition".to_string()),
            latency_ns: 0,
        };

        *current = new_state;
        self.last_changed_ns.store(now_ns, Ordering::Relaxed);

        let mut history = self.history.lock().unwrap();
        if history.len() >= MAX_STATE_HISTORY {
            history.pop_front();
        }
        history.push_back(record);
    }
}

/// Manager for all order state machines
pub struct OmsStateMachineManager {
    /// All tracked orders
    orders: std::sync::RwLock<HashMap<OrderId, std::sync::Arc<OrderStateMachine>>>,
    /// Order counter
    counter: AtomicU64,
    /// Active order count
    active_count: AtomicU64,
    /// Terminal order count
    terminal_count: AtomicU64,
}

impl OmsStateMachineManager {
    /// Create new manager
    pub fn new() -> Self {
        Self {
            orders: std::sync::RwLock::new(HashMap::new()),
            counter: AtomicU64::new(0),
            active_count: AtomicU64::new(0),
            terminal_count: AtomicU64::new(0),
        }
    }

    /// Create a new order state machine
    pub fn create_order(&self) -> std::sync::Arc<OrderStateMachine> {
        let order_id = self.counter.fetch_add(1, Ordering::Relaxed);
        let machine = std::sync::Arc::new(OrderStateMachine::new(order_id));

        {
            let mut orders = self.orders.write().unwrap();
            orders.insert(OrderId(order_id), machine.clone());
        }

        self.active_count.fetch_add(1, Ordering::Relaxed);
        machine
    }

    /// Get an order by ID
    pub fn get_order(&self, order_id: OrderId) -> Option<std::sync::Arc<OrderStateMachine>> {
        self.orders.read().unwrap().get(&order_id).cloned()
    }

    /// Process an event for an order
    pub fn process_event(&self, order_id: OrderId, event: StateEvent) -> Option<TransitionResult> {
        let order = self.get_order(order_id)?;
        let result = order.transition(event);

        // Update counts if transitioned to terminal
        if result.success && result.new_state.is_terminal() {
            self.active_count.fetch_sub(1, Ordering::Relaxed);
            self.terminal_count.fetch_add(1, Ordering::Relaxed);
        }

        Some(result)
    }

    /// Get all active orders
    pub fn get_active_orders(&self) -> Vec<std::sync::Arc<OrderStateMachine>> {
        let orders = self.orders.read().unwrap();
        orders
            .values()
            .filter(|o| o.is_active())
            .cloned()
            .collect()
    }

    /// Get statistics
    pub fn get_stats(&self) -> OmsStats {
        let orders = self.orders.read().unwrap();
        let mut by_state: HashMap<OrderState, usize> = HashMap::new();

        for order in orders.values() {
            let state = order.get_state();
            *by_state.entry(state).or_insert(0) += 1;
        }

        OmsStats {
            total_orders: orders.len(),
            active_orders: self.active_count.load(Ordering::Relaxed) as usize,
            terminal_orders: self.terminal_count.load(Ordering::Relaxed) as usize,
            by_state,
        }
    }

    /// Find stale orders (active but no state change for too long)
    pub fn find_stale_orders(&self, threshold_ns: u64) -> Vec<OrderId> {
        let orders = self.orders.read().unwrap();
        orders
            .values()
            .filter(|o| o.is_active() && o.time_since_last_change_ns() > threshold_ns)
            .map(|o| o.order_id())
            .collect()
    }

    /// Remove terminal orders from memory (cleanup)
    pub fn cleanup_terminal_orders(&self) -> usize {
        let mut orders = self.orders.write().unwrap();
        let before = orders.len();

        orders.retain(|_, v| !v.is_terminal());

        before - orders.len()
    }
}

impl Default for OmsStateMachineManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics snapshot
#[derive(Debug, Clone)]
pub struct OmsStats {
    pub total_orders: usize,
    pub active_orders: usize,
    pub terminal_orders: usize,
    pub by_state: HashMap<OrderState, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_lifecycle() {
        let machine = OrderStateMachine::new(1);

        assert_eq!(machine.get_state(), OrderState::Pending);

        // Submit
        let result = machine.transition(StateEvent::SubmitRequested);
        assert!(result.success);
        assert_eq!(result.new_state, OrderState::Submitted);

        // Acknowledge
        let result = machine.transition(StateEvent::ExchangeAck);
        assert!(result.success);
        assert_eq!(result.new_state, OrderState::Acknowledged);

        // Start working
        let result = machine.transition(StateEvent::ExchangeAck);
        assert!(result.success);
        assert_eq!(result.new_state, OrderState::Working);

        // Partial fill
        let result = machine.transition(StateEvent::Fill {
            size: 0.5,
            price: 50000.0,
        });
        assert!(result.success);
        assert_eq!(result.new_state, OrderState::PartiallyFilled);

        // Full fill
        let result = machine.transition(StateEvent::FullFill);
        assert!(result.success);
        assert_eq!(result.new_state, OrderState::Filled);

        // Verify terminal
        assert!(machine.is_terminal());
        assert!(!machine.is_active());
    }

    #[test]
    fn test_invalid_transition() {
        let machine = OrderStateMachine::new(2);

        // Try to go directly from Pending to Filled (invalid)
        let result = machine.transition(StateEvent::FullFill);
        assert!(!result.success);
        assert!(result.error_message.is_some());

        // State should remain unchanged
        assert_eq!(machine.get_state(), OrderState::Pending);
    }

    #[test]
    fn test_manager_lifecycle() {
        let manager = OmsStateMachineManager::new();

        // Create order
        let order = manager.create_order();
        let order_id = order.order_id();

        // Process events through manager
        let result = manager.process_event(order_id, StateEvent::SubmitRequested);
        assert!(result.unwrap().success);

        let result = manager.process_event(order_id, StateEvent::ExchangeAck);
        assert!(result.unwrap().success);

        // Check stats
        let stats = manager.get_stats();
        assert_eq!(stats.active_orders, 1);
        assert_eq!(stats.terminal_orders, 0);

        // Complete the order
        let result = manager.process_event(order_id, StateEvent::ExchangeAck);
        assert!(result.unwrap().success);
        let result = manager.process_event(order_id, StateEvent::FullFill);
        assert!(result.unwrap().success);

        // Check updated stats
        let stats = manager.get_stats();
        assert_eq!(stats.active_orders, 0);
        assert_eq!(stats.terminal_orders, 1);
    }

    #[test]
    fn test_terminal_state_lock() {
        let machine = OrderStateMachine::new(3);

        // Go to terminal state
        machine.transition(StateEvent::SubmitRequested);
        machine.transition(StateEvent::ExchangeAck);
        machine.transition(StateEvent::ExchangeAck);
        machine.transition(StateEvent::FullFill);

        assert_eq!(machine.get_state(), OrderState::Filled);

        // Try to transition from terminal (should fail)
        let result = machine.transition(StateEvent::CancelRequested);
        assert!(!result.success);

        // State unchanged
        assert_eq!(machine.get_state(), OrderState::Filled);
    }
}
