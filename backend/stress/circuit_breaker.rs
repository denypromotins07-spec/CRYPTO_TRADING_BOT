//! ZAID PERSONAL CRYPTO TRADING BOT - Stress Testing & Chaos Engineering
//! Chapter 3: Circuit Breaker
//! 
//! This module implements a hard kill switch for exchange disconnects and
//! extreme market conditions. It provides sub-100μs response time to protect
//! capital during flash crashes, network failures, or system anomalies.
//! 
//! Memory Budget: <10MB for circuit breaker state
//! Target Response: <100μs trigger time
//! Safety: Multiple redundancy layers, fail-safe defaults
//! Integration: Connected to all risk management modules

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use std::collections::HashMap;

/// Maximum number of trip events to retain
const MAX_TRIP_HISTORY: usize = 100;

/// Default cooldown period after circuit breaker trip
const DEFAULT_COOLDOWN_MS: u64 = 60000; // 60 seconds

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,      // Normal operation
    Open,        // Tripped, blocking trades
    HalfOpen,    // Testing if safe to resume
}

/// Trip reasons for circuit breaker
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TripReason {
    FlashCrash,           // >10% price drop
    ExchangeDisconnect,   // WS connection lost
    LatencySpike,         // Response time exceeded threshold
    SpreadWidening,       // Bid-ask spread too wide
    RiskLimitBreached,    // VaR/CVaR exceeded
    ManualTrigger,        // Human operator intervention
    SystemAnomaly,        // Internal error detected
    NetworkPacketLoss,    // Excessive packet loss
    OrderBookFrozen,      // No book updates received
    CorrelationBreakdown, // Asset correlations broke down
}

/// Configuration for circuit breaker
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub cooldown_ms: u64,
    pub half_open_test_duration_ms: u64,
    pub max_trips_before_lockout: usize,
    pub auto_reset_enabled: bool,
    pub require_manual_reset_after: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            cooldown_ms: DEFAULT_COOLDOWN_MS,
            half_open_test_duration_ms: 5000,
            max_trips_before_lockout: 3,
            auto_reset_enabled: true,
            require_manual_reset_after: 5,
        }
    }
}

/// Trip event record
#[derive(Debug, Clone)]
pub struct TripEvent {
    pub reason: TripReason,
    pub timestamp_ns: u64,
    pub asset: Option<String>,
    pub severity: u8,  // 1-5 scale
    pub details: String,
}

/// Circuit breaker result
#[derive(Debug, Clone)]
pub struct CircuitBreakerResult {
    pub is_safe_to_trade: bool,
    pub current_state: CircuitState,
    pub time_until_reset_ms: Option<u64>,
    pub last_trip_reason: Option<TripReason>,
    pub trip_count_24h: usize,
}

/// Ultra-fast circuit breaker with atomic operations
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    /// Current state (atomic for lock-free reads)
    state: AtomicUsize,  // Encodes CircuitState
    /// Trip timestamp in nanoseconds
    trip_timestamp_ns: AtomicU64,
    /// Trip counter
    trip_count: AtomicUsize,
    /// Trip history
    trip_history: Arc<RwLock<Vec<TripEvent>>>,
    /// Per-asset circuit breakers
    asset_breakers: Arc<RwLock<HashMap<String, Arc<CircuitBreaker>>>>,
    /// Last reason for trip
    last_trip_reason: Arc<RwLock<Option<TripReason>>>,
    /// Global lockout flag
    is_locked_out: AtomicBool,
    /// Creation timestamp
    created_at_ns: u64,
}

unsafe impl Send for CircuitBreaker {}
unsafe impl Sync for CircuitBreaker {}

impl CircuitBreaker {
    /// Create new circuit breaker with default config
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }
    
    /// Create circuit breaker with custom config
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            config,
            state: AtomicUsize::new(CircuitState::Closed as usize),
            trip_timestamp_ns: AtomicU64::new(0),
            trip_count: AtomicUsize::new(0),
            trip_history: Arc::new(RwLock::new(Vec::with_capacity(MAX_TRIP_HISTORY))),
            asset_breakers: Arc::new(RwLock::new(HashMap::new())),
            last_trip_reason: Arc::new(RwLock::new(None)),
            is_locked_out: AtomicBool::new(false),
            created_at_ns: now_ns,
        }
    }
    
    /// Get current state (lock-free read)
    #[inline]
    pub fn get_state(&self) -> CircuitState {
        match self.state.load(Ordering::Acquire) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }
    
    /// Check if safe to trade (ultra-fast path)
    #[inline]
    pub fn is_safe_to_trade(&self) -> bool {
        if self.is_locked_out.load(Ordering::Acquire) {
            return false;
        }
        
        match self.get_state() {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if cooldown has elapsed
                let elapsed = self.elapsed_since_trip_ms();
                elapsed >= self.config.cooldown_ms
            }
            CircuitState::HalfOpen => true,  // Allow limited trading for testing
        }
    }
    
    /// Get full circuit breaker status
    pub fn get_status(&self) -> CircuitBreakerResult {
        let current_state = self.get_state();
        let time_until_reset = if current_state == CircuitState::Open {
            let elapsed = self.elapsed_since_trip_ms();
            if elapsed < self.config.cooldown_ms {
                Some(self.config.cooldown_ms - elapsed)
            } else {
                None
            }
        } else {
            None
        };
        
        let last_reason = self.last_trip_reason
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        
        let trip_count = self.trip_count_24h();
        
        CircuitBreakerResult {
            is_safe_to_trade: self.is_safe_to_trade(),
            current_state,
            time_until_reset_ms: time_until_reset,
            last_trip_reason: last_reason,
            trip_count_24h: trip_count,
        }
    }
    
    /// Trip the circuit breaker (CRITICAL PATH - must be fast)
    #[inline]
    pub fn trip(&self, reason: TripReason, asset: Option<&str>, severity: u8) -> bool {
        // Don't trip if already open
        if self.get_state() == CircuitState::Open {
            return false;
        }
        
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Update state atomically
        self.state.store(CircuitState::Open as usize, Ordering::Release);
        self.trip_timestamp_ns.store(now_ns, Ordering::Release);
        
        // Increment trip counter
        let new_count = self.trip_count.fetch_add(1, Ordering::AcqRel) + 1;
        
        // Check for lockout
        if new_count >= self.config.max_trips_before_lockout {
            self.is_locked_out.store(true, Ordering::Release);
        }
        
        // Record trip reason
        {
            let mut last_reason = self.last_trip_reason
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *last_reason = Some(reason.clone());
        }
        
        // Record trip event
        let event = TripEvent {
            reason,
            timestamp_ns: now_ns,
            asset: asset.map(String::from),
            severity,
            details: format!("Circuit breaker triggered"),
        };
        
        {
            let mut history = self.trip_history
                .write()
                .unwrap_or_else(|e| e.into_inner());
            
            if history.len() >= MAX_TRIP_HISTORY {
                history.remove(0);
            }
            history.push(event);
        }
        
        true
    }
    
    /// Reset circuit breaker to closed state
    pub fn reset(&self) -> bool {
        if self.is_locked_out.load(Ordering::Acquire) {
            return false;  // Must manually unlock first
        }
        
        self.state.store(CircuitState::Closed as usize, Ordering::Release);
        self.trip_timestamp_ns.store(0, Ordering::Release);
        
        true
    }
    
    /// Move to half-open state for testing
    pub fn transition_to_half_open(&self) -> bool {
        if self.get_state() != CircuitState::Open {
            return false;
        }
        
        let elapsed = self.elapsed_since_trip_ms();
        if elapsed < self.config.cooldown_ms {
            return false;  // Too soon
        }
        
        self.state.store(CircuitState::HalfOpen as usize, Ordering::Release);
        true
    }
    
    /// Manually lock out the circuit breaker
    pub fn manual_lockout(&self) {
        self.is_locked_out.store(true, Ordering::Release);
        self.trip(TripReason::ManualTrigger, None, 5);
    }
    
    /// Release manual lockout
    pub fn release_lockout(&self) {
        self.is_locked_out.store(false, Ordering::Release);
        self.reset();
    }
    
    /// Get elapsed time since trip in milliseconds
    #[inline]
    fn elapsed_since_trip_ms(&self) -> u64 {
        let trip_ts = self.trip_timestamp_ns.load(Ordering::Acquire);
        if trip_ts == 0 {
            return u64::MAX;  // Never tripped
        }
        
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        (now_ns - trip_ts) / 1_000_000
    }
    
    /// Get trip count in last 24 hours
    pub fn trip_count_24h(&self) -> usize {
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let twenty_four_hours_ns = 24 * 60 * 60 * 1_000_000_000u64;
        let cutoff = now_ns.saturating_sub(twenty_four_hours_ns);
        
        let history = self.trip_history
            .read()
            .unwrap_or_else(|e| e.into_inner());
        
        history.iter()
            .filter(|e| e.timestamp_ns >= cutoff)
            .count()
    }
    
    /// Get trip history
    pub fn get_trip_history(&self) -> Vec<TripEvent> {
        self.trip_history
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    
    /// Get or create asset-specific circuit breaker
    pub fn get_asset_breaker(&self, asset: &str) -> Arc<CircuitBreaker> {
        let mut breakers = self.asset_breakers
            .write()
            .unwrap_or_else(|e| e.into_inner());
        
        breakers.entry(asset.to_string())
            .or_insert_with(|| Arc::new(CircuitBreaker::with_config(self.config.clone())))
            .clone()
    }
    
    /// Trip all asset-specific breakers
    pub fn trip_all_assets(&self, reason: TripReason, severity: u8) {
        let breakers = self.asset_breakes
            .read()
            .unwrap_or_else(|e| e.into_inner());
        
        for (_, breaker) in breakers.iter() {
            breaker.trip(reason.clone(), None, severity);
        }
    }
    
    /// Check if any breaker is tripped
    pub fn any_breaker_tripped(&self) -> bool {
        if !self.is_safe_to_trade() {
            return true;
        }
        
        let breakers = self.asset_breakes
            .read()
            .unwrap_or_else(|e| e.into_inner());
        
        breakers.values().any(|b| !b.is_safe_to_trade())
    }
}

/// Global circuit breaker manager for the entire bot
pub struct CircuitBreakerManager {
    global_breaker: Arc<CircuitBreaker>,
    risk_limits: Arc<RwLock<RiskLimits>>,
}

/// Risk limits that can trigger circuit breaker
#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_daily_loss_pct: f64,
    pub max_position_size_usd: f64,
    pub max_portfolio_var: f64,
    pub max_drawdown_pct: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_daily_loss_pct: 0.05,  // 5%
            max_position_size_usd: 100000.0,
            max_portfolio_var: 0.03,  // 3%
            max_drawdown_pct: 0.10,  // 10%
        }
    }
}

impl CircuitBreakerManager {
    pub fn new() -> Self {
        Self {
            global_breaker: Arc::new(CircuitBreaker::new()),
            risk_limits: Arc::new(RwLock::new(RiskLimits::default())),
        }
    }
    
    /// Get global circuit breaker
    pub fn get_global_breaker(&self) -> Arc<CircuitBreaker> {
        self.global_breaker.clone()
    }
    
    /// Check and potentially trip based on risk limits
    pub fn check_risk_limits(&self, current_loss_pct: f64, current_var: f64) -> bool {
        let limits = self.risk_limits
            .read()
            .unwrap_or_else(|e| e.into_inner());
        
        if current_loss_pct >= limits.max_daily_loss_pct {
            self.global_breaker.trip(
                TripReason::RiskLimitBreached,
                None,
                5
            );
            return false;
        }
        
        if current_var >= limits.max_portfolio_var {
            self.global_breaker.trip(
                TripReason::RiskLimitBreached,
                None,
                4
            );
            return false;
        }
        
        true
    }
    
    /// Set risk limits
    pub fn set_risk_limits(&self, limits: RiskLimits) {
        let mut current = self.risk_limits
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *current = limits;
    }
    
    /// Emergency stop all trading
    pub fn emergency_stop(&self) {
        self.global_breaker.manual_lockout();
    }
    
    /// Resume trading after emergency stop
    pub fn resume_trading(&self) -> bool {
        if !self.global_breaker.is_safe_to_trade() {
            return false;
        }
        self.global_breaker.release_lockout();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_circuit_breaker_basic() {
        let cb = CircuitBreaker::new();
        
        // Initially closed
        assert_eq!(cb.get_state(), CircuitState::Closed);
        assert!(cb.is_safe_to_trade());
        
        // Trip the breaker
        cb.trip(TripReason::FlashCrash, Some("BTC"), 5);
        
        assert_eq!(cb.get_state(), CircuitState::Open);
        assert!(!cb.is_safe_to_trade());
    }
    
    #[test]
    fn test_cooldown_period() {
        let mut config = CircuitBreakerConfig::default();
        config.cooldown_ms = 100;  // 100ms for testing
        
        let cb = CircuitBreaker::with_config(config);
        cb.trip(TripReason::LatencySpike, None, 3);
        
        assert!(!cb.is_safe_to_trade());
        
        // Wait for cooldown
        thread::sleep(Duration::from_millis(150));
        
        // Should auto-transition to half-open
        cb.transition_to_half_open();
        assert_eq!(cb.get_state(), CircuitState::HalfOpen);
    }
    
    #[test]
    fn test_lockout_after_max_trips() {
        let mut config = CircuitBreakerConfig::default();
        config.max_trips_before_lockout = 3;
        
        let cb = CircuitBreaker::with_config(config);
        
        // Trip multiple times
        for _ in 0..3 {
            cb.trip(TripReason::SpreadWidening, None, 2);
            cb.reset();
        }
        
        assert!(!cb.is_safe_to_trade());
    }
}
