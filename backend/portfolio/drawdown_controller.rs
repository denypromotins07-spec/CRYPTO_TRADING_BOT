// Drawdown Controller: Hard stops and global circuit breakers in Rust.
// Enforces strict drawdown limits with instant position liquidation.
// Implements multiple layers of risk protection for extreme market conditions.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, atomic::{AtomicBool, AtomicF64, Ordering}};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use serde::{Deserialize, Serialize};

/// Configuration for drawdown control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownConfig {
    /// Maximum daily drawdown (e.g., 0.02 = 2%)
    pub max_daily_drawdown: f64,
    /// Maximum weekly drawdown
    pub max_weekly_drawdown: f64,
    /// Maximum monthly drawdown
    pub max_monthly_drawdown: f64,
    /// Maximum total drawdown from peak
    pub max_total_drawdown: f64,
    /// Circuit breaker trigger level
    pub circuit_breaker_level: f64,
    /// Cooldown period after circuit breaker (seconds)
    pub cooldown_seconds: u64,
    /// Enable gradual position reduction
    pub enable_gradual_reduction: bool,
    /// Reduction threshold (start reducing before hard stop)
    pub reduction_threshold: f64,
}

impl Default for DrawdownConfig {
    fn default() -> Self {
        Self {
            max_daily_drawdown: 0.02,      // 2% daily limit
            max_weekly_drawdown: 0.05,     // 5% weekly limit
            max_monthly_drawdown: 0.10,    // 10% monthly limit
            max_total_drawdown: 0.15,      // 15% total limit
            circuit_breaker_level: 0.015,  // 1.5% triggers circuit breaker
            cooldown_seconds: 3600,        // 1 hour cooldown
            enable_gradual_reduction: true,
            reduction_threshold: 0.01,     // Start reducing at 1% drawdown
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    Active,
    Tripped,
    Cooldown,
}

/// Drawdown tracking for different time periods
#[derive(Debug, Clone)]
pub struct DrawdownTracker {
    /// Peak portfolio value
    peak_value: AtomicF64,
    /// Current portfolio value
    current_value: AtomicF64,
    /// Daily PnL
    daily_pnl: AtomicF64,
    /// Weekly PnL
    weekly_pnl: AtomicF64,
    /// Monthly PnL
    monthly_pnl: AtomicF64,
    /// Last reset times
    daily_reset_ns: AtomicU64,
    weekly_reset_ns: AtomicU64,
    monthly_reset_ns: AtomicU64,
}

impl DrawdownTracker {
    pub fn new(initial_value: f64) -> Self {
        let now_ns = current_time_ns();
        
        Self {
            peak_value: AtomicF64::new(initial_value),
            current_value: AtomicF64::new(initial_value),
            daily_pnl: AtomicF64::new(0.0),
            weekly_pnl: AtomicF64::new(0.0),
            monthly_pnl: AtomicF64::new(0.0),
            daily_reset_ns: AtomicU64::new(now_ns),
            weekly_reset_ns: AtomicU64::new(now_ns),
            monthly_reset_ns: AtomicU64::new(now_ns),
        }
    }
    
    pub fn update_value(&self, new_value: f64) {
        self.current_value.store(new_value, Ordering::SeqCst);
        
        // Update peak if new value is higher
        let current_peak = self.peak_value.load(Ordering::SeqCst);
        if new_value > current_peak {
            self.peak_value.store(new_value, Ordering::SeqCst);
        }
        
        // Check for period resets
        self.check_period_resets();
    }
    
    fn check_period_resets(&self) {
        let now_ns = current_time_ns();
        
        // Daily reset (24 hours)
        let daily_elapsed = now_ns - self.daily_reset_ns.load(Ordering::SeqCst);
        if daily_elapsed > 24 * 3600 * 1_000_000_000 {
            self.daily_pnl.store(0.0, Ordering::SeqCst);
            self.daily_reset_ns.store(now_ns, Ordering::SeqCst);
        }
        
        // Weekly reset (7 days)
        let weekly_elapsed = now_ns - self.weekly_reset_ns.load(Ordering::SeqCst);
        if weekly_elapsed > 7 * 24 * 3600 * 1_000_000_000 {
            self.weekly_pnl.store(0.0, Ordering::SeqCst);
            self.weekly_reset_ns.store(now_ns, Ordering::SeqCst);
        }
        
        // Monthly reset (30 days)
        let monthly_elapsed = now_ns - self.monthly_reset_ns.load(Ordering::SeqCst);
        if monthly_elapsed > 30 * 24 * 3600 * 1_000_000_000 {
            self.monthly_pnl.store(0.0, Ordering::SeqCst);
            self.monthly_reset_ns.store(now_ns, Ordering::SeqCst);
        }
    }
    
    pub fn get_current_drawdown(&self) -> f64 {
        let peak = self.peak_value.load(Ordering::SeqCst);
        let current = self.current_value.load(Ordering::SeqCst);
        
        if peak <= 0.0 {
            return 0.0;
        }
        
        (peak - current) / peak
    }
    
    pub fn get_daily_drawdown(&self) -> f64 {
        let peak = self.peak_value.load(Ordering::SeqCst);
        let daily = self.daily_pnl.load(Ordering::SeqCst);
        
        if peak <= 0.0 {
            return 0.0;
        }
        
        (-daily).max(0.0) / peak
    }
}

/// Global drawdown controller with circuit breaker
pub struct DrawdownController {
    config: DrawdownConfig,
    tracker: Arc<DrawdownTracker>,
    /// Circuit breaker state
    circuit_breaker: AtomicBool,
    /// Circuit breaker trip time
    trip_time_ns: AtomicU64,
    /// Trading halted flag
    trading_halted: AtomicBool,
    /// Liquidation in progress
    liquidation_active: AtomicBool,
    /// Asset-specific drawdown limits
    asset_limits: Arc<RwLock<HashMap<String, f64>>>,
}

impl DrawdownController {
    pub fn new(config: DrawdownConfig, initial_value: f64) -> Self {
        Self {
            config,
            tracker: Arc::new(DrawdownTracker::new(initial_value)),
            circuit_breaker: AtomicBool::new(false),
            trip_time_ns: AtomicU64::new(0),
            trading_halted: AtomicBool::new(false),
            liquidation_active: AtomicBool::new(false),
            asset_limits: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Update portfolio value and check all drawdown limits
    pub fn update_portfolio_value(&self, value: f64) -> DrawdownStatus {
        self.tracker.update_value(value);
        
        // Check all drawdown limits
        let status = self.check_all_limits();
        
        // Update circuit breaker if needed
        if status.circuit_breaker_triggered {
            self.trigger_circuit_breaker();
        }
        
        status
    }
    
    /// Check all drawdown limits
    fn check_all_limits(&self) -> DrawdownStatus {
        let total_dd = self.tracker.get_current_drawdown();
        let daily_dd = self.tracker.get_daily_drawdown();
        
        let mut status = DrawdownStatus {
            total_drawdown: total_dd,
            daily_drawdown: daily_dd,
            weekly_drawdown: 0.0,
            monthly_drawdown: 0.0,
            limit_breached: false,
            breach_type: None,
            circuit_breaker_triggered: false,
            liquidation_required: false,
            gradual_reduction: false,
        };
        
        // Check circuit breaker first (most urgent)
        if total_dd >= self.config.circuit_breaker_level && !self.circuit_breaker.load(Ordering::SeqCst) {
            status.circuit_breaker_triggered = true;
        }
        
        // Check daily limit
        if daily_dd >= self.config.max_daily_drawdown {
            status.limit_breached = true;
            status.breach_type = Some(BreachType::Daily);
            status.liquidation_required = true;
            return status;
        }
        
        // Check total drawdown
        if total_dd >= self.config.max_total_drawdown {
            status.limit_breached = true;
            status.breach_type = Some(BreachType::Total);
            status.liquidation_required = true;
            return status;
        }
        
        // Check if gradual reduction should start
        if self.config.enable_gradual_reduction && 
           total_dd >= self.config.reduction_threshold &&
           total_dd < self.config.circuit_breaker_level {
            status.gradual_reduction = true;
        }
        
        status
    }
    
    /// Trigger circuit breaker
    fn trigger_circuit_breaker(&self) {
        self.circuit_breaker.store(true, Ordering::SeqCst);
        self.trip_time_ns.store(current_time_ns(), Ordering::SeqCst);
        self.trading_halted.store(true, Ordering::SeqCst);
        
        eprintln!("⚠️  CIRCUIT BREAKER TRIGGERED - Trading Halted");
    }
    
    /// Check if trading can resume after circuit breaker
    pub fn can_resume_trading(&self) -> bool {
        if !self.circuit_breaker.load(Ordering::SeqCst) {
            return true;
        }
        
        let trip_time = self.trip_time_ns.load(Ordering::SeqCst);
        let elapsed = current_time_ns() - trip_time;
        let cooldown_ns = self.config.cooldown_seconds * 1_000_000_000;
        
        if elapsed > cooldown_ns {
            // Cooldown period expired, check if drawdown has improved
            let current_dd = self.tracker.get_current_drawdown();
            if current_dd < self.config.circuit_breaker_level * 0.8 {
                self.circuit_breaker.store(false, Ordering::SeqCst);
                self.trading_halted.store(false, Ordering::SeqCst);
                return true;
            }
        }
        
        false
    }
    
    /// Initiate emergency liquidation
    pub fn initiate_liquidation(&self) -> LiquidationPlan {
        self.liquidation_active.store(true, Ordering::SeqCst);
        self.trading_halted.store(true, Ordering::SeqCst);
        
        LiquidationPlan {
            immediate: true,
            reduce_by_pct: 1.0,  // 100% liquidation
            reason: "Drawdown limit breached".to_string(),
        }
    }
    
    /// Get gradual reduction plan
    pub fn get_gradual_reduction_plan(&self, current_dd: f64) -> Option<LiquidationPlan> {
        if !self.config.enable_gradual_reduction {
            return None;
        }
        
        if current_dd < self.config.reduction_threshold {
            return None;
        }
        
        // Calculate reduction percentage based on drawdown severity
        let reduction_pct = ((current_dd - self.config.reduction_threshold) / 
                            (self.config.circuit_breaker_level - self.config.reduction_threshold))
            .min(0.5);  // Max 50% reduction at once
        
        Some(LiquidationPlan {
            immediate: false,
            reduce_by_pct: reduction_pct,
            reason: "Gradual risk reduction".to_string(),
        })
    }
    
    /// Set asset-specific drawdown limit
    pub fn set_asset_limit(&self, asset: &str, limit: f64) {
        if let Ok(mut limits) = self.asset_limits.write() {
            limits.insert(asset.to_string(), limit.clamp(0.0, 1.0));
        }
    }
    
    /// Check asset-specific drawdown
    pub fn check_asset_limit(&self, asset: &str, asset_dd: f64) -> bool {
        if let Ok(limits) = self.asset_limits.read() {
            if let Some(&limit) = limits.get(asset) {
                return asset_dd < limit;
            }
        }
        // Use global limit if no asset-specific limit
        asset_dd < self.config.max_total_drawdown
    }
    
    /// Is trading currently halted?
    pub fn is_trading_halted(&self) -> bool {
        self.trading_halted.load(Ordering::SeqCst)
    }
    
    /// Is liquidation in progress?
    pub fn is_liquidation_active(&self) -> bool {
        self.liquidation_active.load(Ordering::SeqCst)
    }
    
    /// Get current drawdown status
    pub fn get_status(&self) -> DrawdownStatus {
        self.check_all_limits()
    }
    
    /// Reset controller (for testing or manual override)
    pub fn reset(&self, new_value: f64) {
        self.tracker.update_value(new_value);
        self.circuit_breaker.store(false, Ordering::SeqCst);
        self.trading_halted.store(false, Ordering::SeqCst);
        self.liquidation_active.store(false, Ordering::SeqCst);
    }
}

/// Type of drawdown breach
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreachType {
    Daily,
    Weekly,
    Monthly,
    Total,
    AssetSpecific,
}

/// Current drawdown status
#[derive(Debug, Clone)]
pub struct DrawdownStatus {
    pub total_drawdown: f64,
    pub daily_drawdown: f64,
    pub weekly_drawdown: f64,
    pub monthly_drawdown: f64,
    pub limit_breached: bool,
    pub breach_type: Option<BreachType>,
    pub circuit_breaker_triggered: bool,
    pub liquidation_required: bool,
    pub gradual_reduction: bool,
}

/// Liquidation plan
#[derive(Debug, Clone)]
pub struct LiquidationPlan {
    pub immediate: bool,
    pub reduce_by_pct: f64,
    pub reason: String,
}

/// Helper function to get current time in nanoseconds
fn current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Builder for DrawdownController
pub struct DrawdownControllerBuilder {
    config: DrawdownConfig,
    initial_value: f64,
}

impl DrawdownControllerBuilder {
    pub fn new() -> Self {
        Self {
            config: DrawdownConfig::default(),
            initial_value: 100_000.0,
        }
    }
    
    pub fn max_daily_drawdown(mut self, pct: f64) -> Self {
        self.config.max_daily_drawdown = pct;
        self
    }
    
    pub fn max_total_drawdown(mut self, pct: f64) -> Self {
        self.config.max_total_drawdown = pct;
        self
    }
    
    pub fn circuit_breaker_level(mut self, pct: f64) -> Self {
        self.config.circuit_breaker_level = pct;
        self
    }
    
    pub fn initial_value(mut self, value: f64) -> Self {
        self.initial_value = value;
        self
    }
    
    pub fn build(self) -> DrawdownController {
        DrawdownController::new(self.config, self.initial_value)
    }
}

impl Default for DrawdownControllerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_drawdown_tracking() {
        let controller = DrawdownControllerBuilder::new()
            .max_daily_drawdown(0.02)
            .initial_value(100_000.0)
            .build();
        
        // Initial state - no drawdown
        let status = controller.update_portfolio_value(100_000.0);
        assert_eq!(status.total_drawdown, 0.0);
        assert!(!status.limit_breached);
        
        // Value increases - new peak
        controller.update_portfolio_value(110_000.0);
        assert_eq!(controller.tracker.get_current_drawdown(), 0.0);
        
        // Value decreases - drawdown
        controller.update_portfolio_value(99_000.0);
        let dd = controller.tracker.get_current_drawdown();
        assert!(dd > 0.09);  // ~9% drawdown from 110k peak
    }
    
    #[test]
    fn test_circuit_breaker() {
        let controller = DrawdownControllerBuilder::new()
            .max_daily_drawdown(0.02)
            .circuit_breaker_level(0.015)
            .initial_value(100_000.0)
            .build();
        
        // Trigger circuit breaker
        controller.update_portfolio_value(98_500.0);  // 1.5% drop
        
        assert!(controller.is_trading_halted());
        assert!(!controller.can_resume_trading());
    }
    
    #[test]
    fn test_liquidation_trigger() {
        let controller = DrawdownControllerBuilder::new()
            .max_daily_drawdown(0.02)
            .initial_value(100_000.0)
            .build();
        
        // Breach daily limit
        let status = controller.update_portfolio_value(97_900.0);  // 2.1% drop
        
        assert!(status.limit_breached);
        assert!(status.liquidation_required);
        
        let plan = controller.initiate_liquidation();
        assert!(plan.immediate);
        assert_eq!(plan.reduce_by_pct, 1.0);
    }
}

// FFI exports for Python integration
#[no_mangle]
pub extern "C" fn create_drawdown_controller(initial_value: f64) -> *mut DrawdownController {
    Box::into_raw(Box::new(DrawdownController::new(DrawdownConfig::default(), initial_value)))
}

#[no_mangle]
pub extern "C" fn update_and_check(
    controller: *mut DrawdownController,
    value: f64,
) -> DrawdownStatus {
    unsafe {
        let controller = &mut *controller;
        controller.update_portfolio_value(value)
    }
}

#[no_mangle]
pub extern "C" fn is_trading_halted(controller: *mut DrawdownController) -> bool {
    unsafe {
        let controller = &*controller;
        controller.is_trading_halted()
    }
}
