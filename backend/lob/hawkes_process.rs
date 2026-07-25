//! Hawkes Process Implementation for Order Arrival Modeling
//! 
//! This module implements a self-exciting Hawkes process to model the clustering
//! of order arrivals in cryptocurrency markets. It captures the phenomenon where
//! one order increases the probability of subsequent orders (momentum/herding).
//! 
//! Optimized for O(1) intensity updates using exponential decay kernels.
//! Zero-cost abstractions ensure minimal heap allocation during high-frequency updates.
//! 
//! # Features
//! - Multi-dimensional Hawkes processes for bid/ask events
//! - Exponential decay kernel for efficient O(1) updates
//! - Branching ratio estimation for market stability analysis
//! - Thread-safe design for concurrent order book processing

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for the Hawkes process parameters
#[derive(Debug, Clone)]
pub struct HawkesConfig {
    /// Base intensity (mu) - background arrival rate independent of past events
    pub base_intensity: f64,
    /// Excitation factor (alpha) - how much one event increases intensity
    pub excitation_factor: f64,
    /// Decay rate (beta) - how quickly the excitation decays over time
    pub decay_rate: f64,
    /// Maximum intensity cap to prevent numerical overflow
    pub max_intensity: f64,
    /// Time window for considering past events (in milliseconds)
    pub time_window_ms: u64,
}

impl Default for HawkesConfig {
    fn default() -> Self {
        Self {
            base_intensity: 1.0,
            excitation_factor: 0.5,
            decay_rate: 2.0,
            max_intensity: 1000.0,
            time_window_ms: 5000, // 5 seconds
        }
    }
}

/// Represents a single event in the Hawkes process
#[derive(Debug, Clone)]
pub struct HawkesEvent {
    /// Timestamp of the event
    pub timestamp: Instant,
    /// Event type (e.g., "bid_market", "ask_limit", "cancel")
    pub event_type: String,
    /// Magnitude of the event (e.g., order size)
    pub magnitude: f64,
    /// Price level associated with the event
    pub price: f64,
}

/// Single-dimensional Hawkes process with exponential kernel
/// 
/// Uses the recursive formulation: lambda(t) = mu + alpha * sum(exp(-beta * (t - t_i)))
/// This allows O(1) updates by maintaining a running sum of decaying excitations.
#[derive(Debug)]
pub struct HawkesProcess1D {
    config: HawkesConfig,
    /// Current intensity value (lambda)
    current_intensity: f64,
    /// Accumulated excitation term (sum of alpha * exp(-beta * elapsed))
    accumulated_excitation: f64,
    /// Last update timestamp for decay calculation
    last_update: Instant,
    /// Number of events processed
    event_count: usize,
    /// Cache for event history (limited by time_window_ms)
    event_history: Vec<HawkesEvent>,
}

impl HawkesProcess1D {
    /// Create a new Hawkes process with given configuration
    pub fn new(config: HawkesConfig) -> Self {
        Self {
            config,
            current_intensity: config.base_intensity,
            accumulated_excitation: 0.0,
            last_update: Instant::now(),
            event_count: 0,
            event_history: Vec::with_capacity(1000), // Pre-allocate to avoid reallocations
        }
    }

    /// Update intensity after a new event occurs
    /// 
    /// This is O(1) amortized due to exponential decay and periodic cleanup
    #[inline]
    pub fn record_event(&mut self, event: HawkesEvent) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        
        // Apply exponential decay to accumulated excitation
        self.accumulated_excitation *= (-self.config.decay_rate * elapsed).exp();
        
        // Add new excitation from this event
        let excitation = self.config.excitation_factor * event.magnitude;
        self.accumulated_excitation += excitation;
        
        // Update current intensity
        self.current_intensity = (self.config.base_intensity + self.accumulated_excitation)
            .min(self.config.max_intensity);
        
        // Record event and cleanup old events
        self.event_history.push(event);
        self.event_count += 1;
        self.last_update = now;
        
        // Periodic cleanup of events outside time window (amortized O(1))
        if self.event_count % 100 == 0 {
            self.cleanup_old_events(now);
        }
    }

    /// Get current intensity without updating
    #[inline]
    pub fn get_intensity(&mut self) -> f64 {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        
        // Apply decay since last update
        let decayed_excitation = self.accumulated_excitation * (-self.config.decay_rate * elapsed).exp();
        (self.config.base_intensity + decayed_excitation).min(self.config.max_intensity)
    }

    /// Calculate expected number of events in next time interval
    pub fn expected_events_in_interval(&mut self, duration_secs: f64) -> f64 {
        let current_lambda = self.get_intensity();
        // For exponential kernel: E[N(t, t+h)] = integral of lambda(s) ds
        // Simplified approximation for small h
        current_lambda * duration_secs
    }

    /// Cleanup events outside the time window
    fn cleanup_old_events(&mut self, now: Instant) {
        let cutoff = Duration::from_millis(self.config.time_window_ms);
        self.event_history.retain(|event| {
            now.duration_since(event.timestamp) < cutoff
        });
    }

    /// Reset the process state
    pub fn reset(&mut self) {
        self.current_intensity = self.config.base_intensity;
        self.accumulated_excitation = 0.0;
        self.last_update = Instant::now();
        self.event_count = 0;
        self.event_history.clear();
    }

    /// Get statistics about the process
    pub fn get_stats(&self) -> HawkesStats {
        HawkesStats {
            current_intensity: self.current_intensity,
            event_count: self.event_count,
            base_intensity: self.config.base_intensity,
            excitation_factor: self.config.excitation_factor,
            decay_rate: self.config.decay_rate,
        }
    }
}

/// Statistics snapshot for a Hawkes process
#[derive(Debug, Clone)]
pub struct HawkesStats {
    pub current_intensity: f64,
    pub event_count: usize,
    pub base_intensity: f64,
    pub excitation_factor: f64,
    pub decay_rate: f64,
}

/// Multi-dimensional Hawkes process for modeling cross-excitation between event types
/// 
/// Models interactions between different event types (e.g., how market buys trigger limit sells)
/// Uses a matrix of excitation factors: alpha_{ij} represents effect of event j on event i
#[derive(Debug)]
pub struct HawkesProcessMD {
    configs: HashMap<String, HawkesConfig>,
    processes: HashMap<String, HawkesProcess1D>,
    /// Cross-excitation matrix: alpha[target][source]
    cross_excitation: HashMap<String, HashMap<String, f64>>,
    /// Global decay rate for simplicity
    global_decay: f64,
    last_global_update: Instant,
}

impl HawkesProcessMD {
    /// Create a new multi-dimensional Hawkes process
    pub fn new(event_types: Vec<&str>) -> Self {
        let mut configs = HashMap::new();
        let mut processes = HashMap::new();
        let mut cross_excitation = HashMap::new();

        for event_type in &event_types {
            let config = HawkesConfig::default();
            configs.insert(event_type.to_string(), config.clone());
            processes.insert(event_type.to_string(), HawkesProcess1D::new(config));
            cross_excitation.insert(event_type.to_string(), HashMap::new());
        }

        // Initialize diagonal (self-excitation) and off-diagonal (cross-excitation)
        for source in &event_types {
            for target in &event_types {
                let alpha = if source == target { 0.5 } else { 0.1 }; // Lower cross-excitation
                cross_excitation
                    .get_mut(*target)
                    .unwrap()
                    .insert(source.to_string(), alpha);
            }
        }

        Self {
            configs,
            processes,
            cross_excitation,
            global_decay: 2.0,
            last_global_update: Instant::now(),
        }
    }

    /// Record an event and update all affected intensities
    pub fn record_event(&mut self, event_type: &str, magnitude: f64, price: f64) {
        let now = Instant::now();
        let event = HawkesEvent {
            timestamp: now,
            event_type: event_type.to_string(),
            magnitude,
            price,
        };

        // Update the process for this event type
        if let Some(process) = self.processes.get_mut(event_type) {
            process.record_event(event);
        }

        // Propagate excitation to other event types (cross-excitation)
        if let Some(targets) = self.cross_excitation.get(event_type) {
            for (target_type, &alpha) in targets {
                if target_type != event_type {
                    if let Some(target_process) = self.processes.get_mut(target_type) {
                        // Inject cross-excitation
                        let cross_event = HawkesEvent {
                            timestamp: now,
                            event_type: target_type.clone(),
                            magnitude: alpha * magnitude,
                            price,
                        };
                        // Only add excitation, don't count as real event
                        target_process.add_external_excitation(alpha * magnitude, now);
                    }
                }
            }
        }
    }

    /// Get current intensity for a specific event type
    pub fn get_intensity(&mut self, event_type: &str) -> Option<f64> {
        self.processes.get_mut(event_type).map(|p| p.get_intensity())
    }

    /// Get all current intensities
    pub fn get_all_intensities(&mut self) -> HashMap<String, f64> {
        self.processes
            .iter_mut()
            .map(|(k, v)| (k.clone(), v.get_intensity()))
            .collect()
    }

    /// Estimate branching ratio (criticality measure)
    /// 
    /// Branching ratio n = alpha / beta
    /// If n >= 1, the process is critical/super-critical (unstable)
    /// If n < 1, the process is sub-critical (stable)
    pub fn estimate_branching_ratio(&self, event_type: &str) -> Option<f64> {
        self.configs.get(event_type).map(|config| {
            config.excitation_factor / config.decay_rate
        })
    }

    /// Check if the market is in a critical state (branching ratio close to 1)
    pub fn is_critical(&self, event_type: &str, threshold: f64) -> bool {
        self.estimate_branching_ratio(event_type)
            .map(|ratio| ratio >= threshold)
            .unwrap_or(false)
    }
}

// Extension trait for external excitation injection
impl HawkesProcess1D {
    /// Add excitation from an external source (cross-excitation from other event types)
    pub fn add_external_excitation(&mut self, excitation: f64, now: Instant) {
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.accumulated_excitation *= (-self.config.decay_rate * elapsed).exp();
        self.accumulated_excitation += excitation;
        self.current_intensity = (self.config.base_intensity + self.accumulated_excitation)
            .min(self.config.max_intensity);
        self.last_update = now;
    }
}

/// Strategy pattern for different kernel functions
pub trait KernelFunction: Send + Sync {
    /// Compute kernel value at time t
    fn compute(&self, t: f64) -> f64;
    /// Compute integral of kernel from 0 to t
    fn integrate(&self, t: f64) -> f64;
}

/// Exponential decay kernel (default, optimized for O(1) updates)
#[derive(Debug, Clone)]
pub struct ExponentialKernel {
    pub alpha: f64,
    pub beta: f64,
}

impl KernelFunction for ExponentialKernel {
    #[inline]
    fn compute(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        self.alpha * (-self.beta * t).exp()
    }

    #[inline]
    fn integrate(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        (self.alpha / self.beta) * (1.0 - (-self.beta * t).exp())
    }
}

/// Power-law kernel for long-memory effects
#[derive(Debug, Clone)]
pub struct PowerLawKernel {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl KernelFunction for PowerLawKernel {
    #[inline]
    fn compute(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        self.alpha * (1.0 + self.beta * t).powf(-self.gamma)
    }

    fn integrate(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        // Numerical integration for power law
        let steps = 100;
        let dt = t / steps as f64;
        let mut sum = 0.0;
        for i in 0..steps {
            sum += self.compute((i as f64 + 0.5) * dt);
        }
        sum * dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hawkes_intensity_increase() {
        let config = HawkesConfig {
            base_intensity: 1.0,
            excitation_factor: 0.5,
            decay_rate: 2.0,
            ..Default::default()
        };
        let mut process = HawkesProcess1D::new(config);
        
        let initial_intensity = process.get_intensity();
        assert!((initial_intensity - 1.0).abs() < 0.01);

        // Record an event
        let event = HawkesEvent {
            timestamp: Instant::now(),
            event_type: "buy".to_string(),
            magnitude: 1.0,
            price: 50000.0,
        };
        process.record_event(event);

        // Intensity should increase
        let new_intensity = process.get_intensity();
        assert!(new_intensity > initial_intensity);
    }

    #[test]
    fn test_branching_ratio_stability() {
        let config = HawkesConfig {
            excitation_factor: 0.5,
            decay_rate: 2.0,
            ..Default::default()
        };
        
        let branching_ratio = config.excitation_factor / config.decay_rate;
        assert!(branching_ratio < 1.0, "Process should be sub-critical (stable)");
    }

    #[test]
    fn test_exponential_kernel() {
        let kernel = ExponentialKernel { alpha: 1.0, beta: 2.0 };
        
        // At t=0, kernel should equal alpha
        assert!((kernel.compute(0.0) - 1.0).abs() < 1e-10);
        
        // Kernel should decay
        assert!(kernel.compute(1.0) < kernel.compute(0.0));
    }
}
