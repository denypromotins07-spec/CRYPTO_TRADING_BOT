//! Propagator Model for Market Order Impact Propagation
//!
//! This module simulates how a single market order propagates through
//! the limit order book, affecting prices across multiple levels. The
//! propagator model captures both immediate impact and delayed price
//! effects as liquidity replenishes.
//!
//! Key Features:
//! - Multi-level order book impact simulation
//! - Decay kernel for impact relaxation
//! - Permanent vs temporary impact separation
//! - Zero-cost abstractions with strict borrowing checks
//!
//! Mathematical Foundation:
//! The propagator model describes impact decay as:
//!   G(t) = G_∞ + (G_0 - G_∞) * exp(-t/τ)
//! where:
//!   G_0 = immediate impact coefficient
//!   G_∞ = permanent impact coefficient
//!   τ = relaxation time constant
//!
//! Total impact at time t: I(t) = ∫₀ᵗ G(t-s) dQ(s)

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for propagator model
#[derive(Debug, Clone)]
pub struct PropagatorConfig {
    /// Immediate impact coefficient (G_0)
    pub g0: f64,
    /// Permanent impact coefficient (G_∞)
    pub g_infinity: f64,
    /// Relaxation time constant (τ) in seconds
    pub tau: f64,
    /// Number of order book levels to track
    pub n_levels: usize,
    /// Maximum history duration
    pub max_history: Duration,
}

impl Default for PropagatorConfig {
    fn default() -> Self {
        Self {
            g0: 1.0,           // Full immediate impact
            g_infinity: 0.3,   // 30% permanent
            tau: 10.0,         // 10 second relaxation
            n_levels: 10,
            max_history: Duration::from_secs(60),
        }
    }
}

impl PropagatorConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), PropagatorError> {
        if self.g0 < 0.0 {
            return Err(PropagatorError::NegativeImpact);
        }
        if self.g_infinity < 0.0 || self.g_infinity > self.g0 {
            return Err(PropagatorError::InvalidPermanentImpact);
        }
        if self.tau <= 0.0 {
            return Err(PropagatorError::NonPositiveTau);
        }
        Ok(())
    }
}

/// Errors for propagator model
#[derive(Debug, Clone, PartialEq)]
pub enum PropagatorError {
    NegativeImpact,
    InvalidPermanentImpact,
    NonPositiveTau,
    NumericalOverflow,
}

/// Single order book level state
#[derive(Debug, Clone)]
pub struct LevelState {
    /// Price at this level
    pub price: f64,
    /// Volume available at this level
    pub volume: f64,
    /// Cumulative impact at this level
    pub cumulative_impact: f64,
    /// Time since last update
    pub last_update: Instant,
}

/// Impact event record
#[derive(Debug, Clone)]
pub struct ImpactEvent {
    /// Timestamp of the event
    pub timestamp: Instant,
    /// Order size (signed: positive=buy, negative=sell)
    pub size: f64,
    /// Initial impact caused
    pub initial_impact: f64,
    /// Side of the order (1=buy, -1=sell)
    pub side: i8,
}

/// Propagator model for tracking impact evolution
pub struct PropagatorModel {
    config: PropagatorConfig,
    /// Current impact value
    current_impact: f64,
    /// History of impact events
    event_history: VecDeque<ImpactEvent>,
    /// Start time
    start_time: Instant,
    /// Last update time
    last_update: Instant,
    /// Order book level states
    levels: Vec<LevelState>,
}

impl PropagatorModel {
    /// Create new propagator model
    pub fn new(config: PropagatorConfig) -> Result<Self, PropagatorError> {
        config.validate()?;
        let now = Instant::now();
        
        Ok(Self {
            config,
            current_impact: 0.0,
            event_history: VecDeque::with_capacity(1000),
            start_time: now,
            last_update: now,
            levels: Vec::with_capacity(config.n_levels),
        })
    }
    
    /// Initialize order book levels
    pub fn initialize_levels(&mut self, base_price: f64, tick_size: f64) {
        let now = Instant::now();
        self.levels.clear();
        
        for i in 0..self.config.n_levels {
            let price_offset = (i + 1) as f64 * tick_size;
            self.levels.push(LevelState {
                price: base_price - price_offset, // Bid side example
                volume: 100.0 * (i + 1) as f64,   // Increasing depth
                cumulative_impact: 0.0,
                last_update: now,
            });
        }
    }
    
    /// Apply a market order and calculate propagated impact
    ///
    /// Returns the immediate price impact in price units
    pub fn apply_order(
        &mut self, 
        size: f64, 
        mid_price: f64,
    ) -> f64 {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        
        // Decay previous impact
        self.current_impact *= (-dt / self.config.tau).exp();
        
        // Calculate volume-normalized order size
        let avg_volume: f64 = self.levels.iter()
            .map(|l| l.volume)
            .sum::<f64>() / self.levels.len() as f64;
        
        let normalized_size = if avg_volume > 0.0 {
            size.abs() / avg_volume
        } else {
            size.abs()
        };
        
        // Calculate immediate impact using propagator kernel
        // G(t=0) = g0 * sign(Q) * |Q|^δ (typically δ ≈ 0.5)
        let delta = 0.5; // Square root impact
        let immediate_impact = self.config.g0 
            * size.signum() 
            * normalized_size.powf(delta);
        
        // Update current impact
        self.current_impact += immediate_impact;
        
        // Record event
        self.event_history.push_back(ImpactEvent {
            timestamp: now,
            size,
            initial_impact: immediate_impact,
            side: if size >= 0.0 { 1 } else { -1 },
        });
        
        // Prune old events
        self.prune_events();
        
        // Update level states
        self.update_levels(immediate_impact, mid_price);
        
        self.last_update = now;
        
        // Return impact in price units
        self.current_impact * mid_price * 0.0001 // Scale to basis points
    }
    
    /// Get current predicted impact
    #[inline]
    pub fn current_impact(&self) -> f64 {
        self.current_impact
    }
    
    /// Get permanent impact component
    pub fn permanent_impact(&self) -> f64 {
        self.current_impact * (self.config.g_infinity / self.config.g0)
    }
    
    /// Get temporary impact component
    pub fn temporary_impact(&self) -> f64 {
        self.current_impact - self.permanent_impact()
    }
    
    /// Predict impact at future time
    pub fn predict_impact_at(&self, delay_s: f64) -> f64 {
        self.current_impact * (-delay_s / self.config.tau).exp()
    }
    
    /// Calculate total impact from sequence of orders
    pub fn calculate_sequence_impact(&self, orders: &[(f64, f64)]) -> f64 {
        // orders: Vec of (size, time_offset_seconds)
        let mut total_impact = 0.0;
        
        for (i, &(size, t_i)) in orders.iter().enumerate() {
            // Impact from this order
            let normalized_size = size.abs().sqrt();
            let direct_impact = self.config.g0 * size.signum() * normalized_size;
            
            // Add decayed impact from all previous orders
            for j in 0..i {
                let (_, t_j) = orders[j];
                let dt = t_i - t_j;
                if dt >= 0.0 {
                    let decay = (-dt / self.config.tau).exp();
                    total_impact += direct_impact * decay;
                }
            }
            
            total_impact += direct_impact;
        }
        
        total_impact
    }
    
    /// Update order book level states after impact
    fn update_levels(&mut self, impact: f64, mid_price: f64) {
        let now = Instant::now();
        
        for (i, level) in self.levels.iter_mut().enumerate() {
            // Impact decays with distance from top of book
            let distance_factor = 1.0 / (i + 1) as f64;
            level.cumulative_impact += impact * distance_factor;
            level.last_update = now;
            
            // Adjust volume based on impact (liquidity withdrawal)
            let withdrawal = impact.abs() * level.volume * 0.01;
            level.volume = (level.volume - withdrawal).max(0.0);
        }
    }
    
    /// Prune events outside history window
    fn prune_events(&mut self) {
        let cutoff = self.last_update - self.config.max_history;
        
        while let Some(front) = self.event_history.front() {
            if front.timestamp < cutoff {
                self.event_history.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Get impact decay curve parameters
    pub fn decay_parameters(&self) -> (f64, f64, f64) {
        (self.config.g0, self.config.g_infinity, self.config.tau)
    }
    
    /// Reset the model
    pub fn reset(&mut self) {
        self.current_impact = 0.0;
        self.event_history.clear();
        self.start_time = Instant::now();
        self.last_update = self.start_time;
    }
}

/// Multi-asset propagator for cross-asset impact spillover
pub struct CrossAssetPropagator {
    /// Primary asset propagator
    primary: PropagatorModel,
    /// Spillover coefficients to other assets
    spillover: std::collections::HashMap<String, f64>,
    /// Cached spillover impacts
    spillover_impacts: std::collections::HashMap<String, f64>,
}

impl CrossAssetPropagator {
    /// Create new cross-asset propagator
    pub fn new(primary_config: PropagatorConfig) -> Result<Self, PropagatorError> {
        let primary = PropagatorModel::new(primary_config)?;
        
        Ok(Self {
            primary,
            spillover: std::collections::HashMap::new(),
            spillover_impacts: std::collections::HashMap::new(),
        })
    }
    
    /// Set spillover coefficient to another asset
    pub fn set_spillover(&mut self, asset: String, coefficient: f64) {
        self.spillover.insert(asset, coefficient.clamp(0.0, 1.0));
    }
    
    /// Apply order and get impacts on all assets
    pub fn apply_order_multi(
        &mut self,
        size: f64,
        mid_price: f64,
    ) -> std::collections::HashMap<String, f64> {
        // Apply to primary
        let primary_impact = self.primary.apply_order(size, mid_price);
        
        let mut impacts = std::collections::HashMap::new();
        impacts.insert("PRIMARY".to_string(), primary_impact);
        
        // Calculate spillover
        for (asset, &coeff) in &self.spillover {
            let spillover = primary_impact * coeff;
            self.spillover_impacts.insert(asset.clone(), spillover);
            impacts.insert(asset.clone(), spillover);
        }
        
        impacts
    }
    
    /// Get spillover impact for specific asset
    pub fn get_spillover(&self, asset: &str) -> f64 {
        self.spillover_impacts.get(asset).copied().unwrap_or(0.0)
    }
}

/// Optimal execution using propagator model
pub struct PropagatorExecutor {
    model: PropagatorModel,
    /// Target quantity to execute
    target_qty: f64,
    /// Executed quantity
    executed_qty: f64,
    /// Accumulated cost
    accumulated_cost: f64,
}

impl PropagatorExecutor {
    /// Create new executor
    pub fn new(config: PropagatorConfig) -> Result<Self, PropagatorError> {
        let model = PropagatorModel::new(config)?;
        Ok(Self {
            model,
            target_qty: 0.0,
            executed_qty: 0.0,
            accumulated_cost: 0.0,
        })
    }
    
    /// Set execution target
    pub fn set_target(&mut self, qty: f64) {
        self.target_qty = qty;
        self.executed_qty = 0.0;
        self.accumulated_cost = 0.0;
    }
    
    /// Calculate optimal child order size considering impact decay
    pub fn optimal_child_order(
        &self,
        remaining_time_s: f64,
        current_impact: f64,
    ) -> f64 {
        // If current impact is high, wait for decay
        let impact_threshold = 0.5;
        
        if current_impact.abs() > impact_threshold {
            // Reduce size to let impact decay
            let decay_factor = (-remaining_time_s / self.model.decay_parameters().2).exp();
            return self.target_qty * 0.1 * decay_factor;
        }
        
        // Normal execution
        let remaining = self.target_qty - self.executed_qty;
        let intervals = (remaining_time_s / 5.0).ceil().max(1.0);
        remaining / intervals
    }
    
    /// Record execution
    pub fn record_execution(&mut self, qty: f64, impact_cost: f64) {
        self.executed_qty += qty;
        self.accumulated_cost += impact_cost;
    }
    
    /// Get execution progress
    pub fn progress(&self) -> f64 {
        if self.target_qty == 0.0 {
            return 0.0;
        }
        self.executed_qty / self.target_qty
    }
    
    /// Get average cost per unit
    pub fn avg_cost(&self) -> f64 {
        if self.executed_qty == 0.0 {
            return 0.0;
        }
        self.accumulated_cost / self.executed_qty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_propagator_creation() {
        let config = PropagatorConfig::default();
        let model = PropagatorModel::new(config);
        assert!(model.is_ok());
    }
    
    #[test]
    fn test_impact_decay() {
        let config = PropagatorConfig {
            g0: 1.0,
            g_infinity: 0.3,
            tau: 5.0,
            ..Default::default()
        };
        let mut model = PropagatorModel::new(config).unwrap();
        model.initialize_levels(50000.0, 1.0);
        
        // Apply an order
        let impact = model.apply_order(100.0, 50000.0);
        assert!(impact > 0.0);
        
        let initial = model.current_impact();
        
        // Wait (simulate by checking prediction)
        let predicted = model.predict_impact_at(5.0);
        
        // Should decay by ~37% after one tau
        assert!(predicted < initial);
        assert!((predicted - initial * (-1.0_f64).exp()).abs() < 0.01);
    }
    
    #[test]
    fn test_permanent_temporary_split() {
        let config = PropagatorConfig {
            g0: 1.0,
            g_infinity: 0.3,
            ..Default::default()
        };
        let mut model = PropagatorModel::new(config).unwrap();
        model.initialize_levels(50000.0, 1.0);
        
        model.apply_order(100.0, 50000.0);
        
        let total = model.current_impact();
        let perm = model.permanent_impact();
        let temp = model.temporary_impact();
        
        assert!((total - (perm + temp)).abs() < 1e-10);
        assert!((perm / total - 0.3).abs() < 0.01);
    }
}
