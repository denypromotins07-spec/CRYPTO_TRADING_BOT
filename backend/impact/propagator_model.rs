//! Propagator Model for Market Order Impact Propagation
//! 
//! This module implements the propagator model which simulates how a single
//! market order propagates through the limit order book, affecting prices
//! at multiple levels and decaying over time.
//! 
//! Key Features:
//! - Models impact decay and recovery (resilience)
//! - Tracks propagation across price levels
//! - Zero-cost abstractions for HFT performance
//! - Integrates with Almgren-Chriss for comprehensive impact modeling

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for propagator model
#[derive(Debug, Clone)]
pub struct PropagatorConfig {
    /// Number of order book levels to track
    pub num_levels: usize,
    /// Decay rate of impact (per second)
    pub decay_rate: f64,
    /// Resilience factor (speed of book recovery)
    pub resilience: f64,
    /// Propagation coefficient (how much impact spreads to adjacent levels)
    pub propagation_coef: f64,
    /// Maximum impact duration (seconds)
    pub max_duration_secs: f64,
}

impl Default for PropagatorConfig {
    fn default() -> Self {
        Self {
            num_levels: 10,
            decay_rate: 0.5,
            resilience: 0.3,
            propagation_coef: 0.2,
            max_duration_secs: 60.0,
        }
    }
}

/// Impact state at a single price level
#[derive(Debug, Clone)]
pub struct LevelImpact {
    /// Current impact in basis points
    pub impact_bps: f64,
    /// Time since impact was applied
    pub age_secs: f64,
    /// Original impact magnitude
    pub initial_impact: f64,
}

/// Snapshot of propagator state
#[derive(Debug, Clone)]
pub struct PropagatorSnapshot {
    /// Impacts across all levels
    pub level_impacts: Vec<LevelImpact>,
    /// Total propagated impact
    pub total_impact_bps: f64,
    /// Time since last event
    pub elapsed_secs: f64,
    /// Number of active propagations
    pub active_propagations: usize,
}

/// Main propagator model for tracking impact propagation
pub struct ImpactPropagator {
    config: PropagatorConfig,
    /// Impact state for each level (bid and ask separately)
    bid_impacts: Vec<LevelImpact>,
    ask_impacts: Vec<LevelImpact>,
    /// Event history for decay calculation
    event_history: VecDeque<(Instant, f64, String)>, // (time, size, side)
    /// Last update time
    last_update: Instant,
    /// Thread safety
    _lock: Arc<Mutex<()>>,
}

impl ImpactPropagator {
    pub fn new(config: PropagatorConfig) -> Self {
        let empty_impact = LevelImpact {
            impact_bps: 0.0,
            age_secs: 0.0,
            initial_impact: 0.0,
        };
        
        Self {
            config,
            bid_impacts: vec![empty_impact.clone(); config.num_levels],
            ask_impacts: vec![empty_impact; config.num_levels],
            event_history: VecDeque::with_capacity(100),
            last_update: Instant::now(),
            _lock: Arc::new(Mutex::new(())),
        }
    }
    
    /// Apply a market order and propagate its impact
    pub fn apply_market_order(&mut self, size: f64, side: &str, price_level: usize) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        
        // First, decay existing impacts
        self.decay_all_impacts(elapsed);
        
        // Calculate initial impact based on size
        let base_impact = self.calculate_base_impact(size);
        
        // Apply impact to specified level and propagate
        if side.to_lowercase() == "buy" {
            self.apply_impact_to_side(&mut self.ask_impacts, base_impact, price_level, now);
        } else {
            self.apply_impact_to_side(&mut self.bid_impacts, base_impact, price_level, now);
        }
        
        // Record event
        self.event_history.push_back((now, size, side.to_string()));
        
        // Cleanup old events
        while self.event_history.len() > 50 {
            self.event_history.pop_front();
        }
        
        self.last_update = now;
    }
    
    /// Get current snapshot of propagator state
    pub fn get_snapshot(&mut self) -> PropagatorSnapshot {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        
        // Decay before snapshot
        self.decay_all_impacts(elapsed);
        self.last_update = now;
        
        let mut total_impact = 0.0;
        let mut active_count = 0;
        
        let all_impacts: Vec<LevelImpact> = self.bid_impacts
            .iter()
            .chain(self.ask_impacts.iter())
            .cloned()
            .map(|impact| {
                if impact.impact_bps > 0.01 {
                    active_count += 1;
                    total_impact += impact.impact_bps;
                }
                impact
            })
            .collect();
        
        PropagatorSnapshot {
            level_impacts: all_impacts,
            total_impact_bps: total_impact,
            elapsed_secs: elapsed,
            active_propagations: active_count,
        }
    }
    
    /// Calculate base impact from order size
    fn calculate_base_impact(&self, size: f64) -> f64 {
        // Square root law for base impact
        let base = 0.01 * size.sqrt();
        base.min(0.1) // Cap at 1000 bps (10%)
    }
    
    /// Apply impact to one side of the book with propagation
    fn apply_impact_to_side(
        &mut self,
        impacts: &mut [LevelImpact],
        base_impact: f64,
        center_level: usize,
        now: Instant,
    ) {
        let now_secs = now.elapsed().as_secs_f64();
        
        for (i, impact) in impacts.iter_mut().enumerate() {
            // Distance from center level
            let distance = (i as i32 - center_level as i32).abs() as f64;
            
            // Propagation decays with distance
            let propagation_factor = (-self.config.propagation_coef * distance).exp();
            
            // Additional impact at this level
            let additional_impact = base_impact * propagation_factor;
            
            // Update impact state
            impact.initial_impact += additional_impact;
            impact.impact_bps += additional_impact;
            impact.age_secs = 0.0; // Reset age for this level
        }
    }
    
    /// Decay all impacts based on elapsed time
    fn decay_all_impacts(&mut self, elapsed_secs: f64) {
        let decay_factor = (-self.config.decay_rate * elapsed_secs).exp();
        let resilience_recovery = self.config.resilience * elapsed_secs;
        
        for impacts in [&mut self.bid_impacts, &mut self.ask_impacts] {
            for impact in impacts.iter_mut() {
                // Exponential decay
                impact.impact_bps *= decay_factor;
                
                // Resilience recovery toward zero
                impact.impact_bps -= resilience_recovery * impact.initial_impact;
                impact.impact_bps = impact.impact_bps.max(0.0);
                
                // Update age
                impact.age_secs += elapsed_secs;
                
                // Clear very old/small impacts
                if impact.age_secs > self.config.max_duration_secs || impact.impact_bps < 0.001 {
                    impact.impact_bps = 0.0;
                    impact.initial_impact = 0.0;
                }
            }
        }
    }
    
    /// Estimate price recovery time after impact
    pub fn estimate_recovery_time(&self, impact_threshold_bps: f64) -> f64 {
        let max_current_impact = self.bid_impacts
            .iter()
            .chain(self.ask_impacts.iter())
            .map(|i| i.impact_bps)
            .fold(0.0_f64, f64::max);
        
        if max_current_impact <= impact_threshold_bps {
            return 0.0;
        }
        
        // Solve: max_impact * exp(-decay * t) = threshold
        // t = ln(threshold / max_impact) / (-decay)
        let ratio = impact_threshold_bps / max_current_impact;
        if ratio <= 0.0 {
            return f64::MAX;
        }
        
        (-ratio.ln()) / self.config.decay_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_impact_propagation() {
        let mut propagator = ImpactPropagator::new(PropagatorConfig::default());
        
        // Apply a buy order
        propagator.apply_market_order(10.0, "buy", 0);
        
        let snapshot = propagator.get_snapshot();
        assert!(snapshot.total_impact_bps > 0.0);
    }
    
    #[test]
    fn test_impact_decay() {
        let mut propagator = ImpactPropagator::new(PropagatorConfig {
            decay_rate: 1.0,
            ..Default::default()
        });
        
        propagator.apply_market_order(10.0, "buy", 0);
        let initial = propagator.get_snapshot();
        
        // Simulate time passing by updating
        std::thread::sleep(Duration::from_millis(100));
        let later = propagator.get_snapshot();
        
        assert!(later.total_impact_bps <= initial.total_impact_bps);
    }
}
