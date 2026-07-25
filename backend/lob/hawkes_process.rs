//! Hawkes Process Implementation for Order Arrival Modeling
//! 
//! This module implements self-exciting point processes to model the clustering
//! of order arrivals in cryptocurrency markets. Hawkes processes capture the
//! phenomenon where one order arrival increases the probability of subsequent
//! arrivals (endogenous excitation).
//!
//! Key Features:
//! - O(1) intensity computation using exponential decay kernels
//! - Multi-dimensional modeling for bid/ask events
//! - Memory-efficient circular buffer for event history
//! - Zero-cost abstractions with strict borrowing checks
//!
//! Mathematical Foundation:
//! λ(t) = μ + ∑_{t_i < t} α * exp(-β * (t - t_i))
//! where:
//!   μ = baseline intensity (exogenous arrivals)
//!   α = excitation coefficient (impact of each event)
//!   β = decay rate (how quickly impact fades)
//!   t_i = timestamps of previous events

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// Configuration for Hawkes process parameters
#[derive(Debug, Clone)]
pub struct HawkesConfig {
    /// Baseline intensity (exogenous arrival rate)
    pub mu: f64,
    /// Excitation coefficient (must be < β for stationarity)
    pub alpha: f64,
    /// Decay rate (inverse of characteristic time)
    pub beta: f64,
    /// Maximum history window for memory efficiency
    pub max_history_duration: Duration,
    /// Maximum number of events to store
    pub max_events: usize,
}

impl Default for HawkesConfig {
    fn default() -> Self {
        Self {
            mu: 1.0,           // 1 event per second baseline
            alpha: 0.5,        // Moderate excitation
            beta: 2.0,         // Fast decay (0.5s characteristic time)
            max_history_duration: Duration::from_secs(60),
            max_events: 10000,
        }
    }
}

impl HawkesConfig {
    /// Validate parameters for stationarity and stability
    pub fn validate(&self) -> Result<(), HawkesError> {
        if self.mu <= 0.0 {
            return Err(HawkesError::InvalidBaseline);
        }
        if self.alpha <= 0.0 || self.alpha >= self.beta {
            return Err(HawkesError::InvalidExcitation);
        }
        if self.beta <= 0.0 {
            return Err(HawkesError::InvalidDecay);
        }
        Ok(())
    }
}

/// Errors specific to Hawkes process computation
#[derive(Debug, Clone, PartialEq)]
pub enum HawkesError {
    InvalidBaseline,
    InvalidExcitation,
    InvalidDecay,
    Overflow,
    NotStationary,
}

/// Single event in the Hawkes process
#[derive(Debug, Clone, Copy)]
pub struct HawkesEvent {
    /// Timestamp of the event (microseconds since epoch)
    pub timestamp_us: u64,
    /// Event type (0=bid market, 1=ask market, 2=bid limit, 3=ask limit, etc.)
    pub event_type: u8,
    /// Volume associated with the event
    pub volume: f64,
    /// Price level (if applicable)
    pub price: f64,
}

/// Multi-dimensional Hawkes process for order book events
/// 
/// Uses exponential decay kernel for O(1) intensity updates via recursive formula:
/// λ(t) = μ + (λ(t_prev) - μ) * exp(-β * Δt) + α * δ(t - t_i)
pub struct HawkesProcess {
    config: HawkesConfig,
    /// Current intensity value (cached for O(1) access)
    current_intensity: f64,
    /// Last update timestamp
    last_update_us: u64,
    /// Event history with automatic pruning
    events: VecDeque<HawkesEvent>,
    /// Start time for the process
    start_time: Instant,
    /// Branching ratio (alpha/beta) for stability monitoring
    branching_ratio: f64,
}

impl HawkesProcess {
    /// Create a new Hawkes process with given configuration
    pub fn new(config: HawkesConfig) -> Result<Self, HawkesError> {
        config.validate()?;
        let branching_ratio = config.alpha / config.beta;
        
        Ok(Self {
            config,
            current_intensity: config.mu,
            last_update_us: 0,
            events: VecDeque::with_capacity(config.max_events),
            start_time: Instant::now(),
            branching_ratio,
        })
    }
    
    /// Get current intensity in O(1) time
    #[inline]
    pub fn intensity(&self) -> f64 {
        self.current_intensity
    }
    
    /// Get branching ratio (should be < 1 for stationarity)
    #[inline]
    pub fn branching_ratio(&self) -> f64 {
        self.branching_ratio
    }
    
    /// Check if process is stationary (branching ratio < 1)
    #[inline]
    pub fn is_stationary(&self) -> bool {
        self.branching_ratio < 1.0
    }
    
    /// Record a new event and update intensity
    /// 
    /// Uses recursive formula for O(1) update:
    /// 1. Decay previous intensity: λ' = μ + (λ - μ) * exp(-β * Δt)
    /// 2. Add excitation: λ'' = λ' + α
    pub fn add_event(&mut self, event: HawkesEvent) -> Result<(), HawkesError> {
        let now_us = event.timestamp_us;
        
        // Initialize on first event
        if self.last_update_us == 0 {
            self.last_update_us = now_us;
            self.events.push_back(event);
            return Ok(());
        }
        
        // Calculate time delta
        let dt_us = now_us.saturating_sub(self.last_update_us);
        let dt_seconds = dt_us as f64 / 1_000_000.0;
        
        // Recursive intensity update with exponential decay
        // λ(t) = μ + (λ(t-Δt) - μ) * exp(-β * Δt)
        let decay_factor = (-self.config.beta * dt_seconds).exp();
        self.current_intensity = self.config.mu 
            + (self.current_intensity - self.config.mu) * decay_factor;
        
        // Add excitation from new event
        self.current_intensity += self.config.alpha;
        
        // Prevent overflow
        if !self.current_intensity.is_finite() || self.current_intensity < 0.0 {
            return Err(HawkesError::Overflow);
        }
        
        self.last_update_us = now_us;
        self.events.push_back(event);
        
        // Prune old events for memory efficiency
        self.prune_events(now_us);
        
        Ok(())
    }
    
    /// Get intensity at a specific future time (for prediction)
    pub fn intensity_at(&self, future_us: u64) -> f64 {
        if future_us <= self.last_update_us {
            return self.current_intensity;
        }
        
        let dt_us = future_us - self.last_update_us;
        let dt_seconds = dt_us as f64 / 1_000_000.0;
        let decay_factor = (-self.config.beta * dt_seconds).exp();
        
        self.config.mu + (self.current_intensity - self.config.mu) * decay_factor
    }
    
    /// Get expected number of events in next T seconds
    pub fn expected_events(&self, horizon_seconds: f64) -> f64 {
        // E[N(T)] = μ*T + (λ(0) - μ) * (1 - exp(-β*T)) / β
        let lambda_0 = self.current_intensity;
        let mu = self.config.mu;
        let beta = self.config.beta;
        
        mu * horizon_seconds 
            + (lambda_0 - mu) * (1.0 - (-beta * horizon_seconds).exp()) / beta
    }
    
    /// Prune events outside the history window
    fn prune_events(&mut self, current_us: u64) {
        let cutoff_us = current_us.saturating_sub(
            self.config.max_history_duration.as_micros() as u64
        );
        
        while let Some(front) = self.events.front() {
            if front.timestamp_us < cutoff_us || self.events.len() > self.config.max_events {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Get event count by type
    pub fn event_count_by_type(&self, event_type: u8) -> usize {
        self.events.iter().filter(|e| e.event_type == event_type).count()
    }
    
    /// Reset the process
    pub fn reset(&mut self) {
        self.current_intensity = self.config.mu;
        self.last_update_us = 0;
        self.events.clear();
        self.start_time = Instant::now();
    }
}

/// Multi-variate Hawkes process for cross-excitation between event types
pub struct MultivariateHawkes {
    /// Number of event types
    n_types: usize,
    /// Baseline intensities for each type
    mu: Vec<f64>,
    /// Excitation matrix [i,j] = impact of type j on type i
    alpha: Vec<Vec<f64>>,
    /// Decay rates for each type pair
    beta: Vec<Vec<f64>>,
    /// Current intensities
    intensities: Vec<f64>,
    /// Last update time
    last_update_us: u64,
    /// Event history
    events: VecDeque<HawkesEvent>,
}

impl MultivariateHawkes {
    /// Create new multivariate Hawkes process
    pub fn new(n_types: usize) -> Self {
        Self {
            n_types,
            mu: vec![1.0; n_types],
            alpha: vec![vec![0.0; n_types]; n_types],
            beta: vec![vec![2.0; n_types]; n_types],
            intensities: vec![1.0; n_types],
            last_update_us: 0,
            events: VecDeque::with_capacity(10000),
        }
    }
    
    /// Set baseline intensity for event type
    pub fn set_baseline(&mut self, type_idx: usize, mu: f64) {
        if type_idx < self.n_types {
            self.mu[type_idx] = mu;
        }
    }
    
    /// Set excitation coefficient
    pub fn set_excitation(&mut self, from_type: usize, to_type: usize, alpha: f64) {
        if from_type < self.n_types && to_type < self.n_types {
            self.alpha[to_type][from_type] = alpha;
        }
    }
    
    /// Add event and update all intensities
    pub fn add_event(&mut self, event: HawkesEvent) {
        let now_us = event.timestamp_us;
        let event_type = event.event_type as usize;
        
        if event_type >= self.n_types {
            return;
        }
        
        if self.last_update_us == 0 {
            self.last_update_us = now_us;
            self.events.push_back(event);
            return;
        }
        
        let dt_us = now_us.saturating_sub(self.last_update_us);
        let dt_seconds = dt_us as f64 / 1_000_000.0;
        
        // Update all intensities with decay and excitation
        for i in 0..self.n_types {
            let decay = (-self.beta[i][event_type] * dt_seconds).exp();
            self.intensities[i] = self.mu[i] 
                + (self.intensities[i] - self.mu[i]) * decay
                + self.alpha[i][event_type];
        }
        
        self.last_update_us = now_us;
        self.events.push_back(event);
    }
    
    /// Get intensity for specific event type
    pub fn intensity(&self, type_idx: usize) -> f64 {
        if type_idx < self.n_types {
            self.intensities[type_idx]
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hawkes_creation() {
        let config = HawkesConfig::default();
        let process = HawkesProcess::new(config);
        assert!(process.is_ok());
    }
    
    #[test]
    fn test_stationarity_check() {
        let mut config = HawkesConfig::default();
        config.alpha = 0.5;
        config.beta = 2.0;
        let process = HawkesProcess::new(config).unwrap();
        assert!(process.is_stationary());
        assert!((process.branching_ratio() - 0.25).abs() < 1e-10);
    }
    
    #[test]
    fn test_intensity_update() {
        let config = HawkesConfig {
            mu: 1.0,
            alpha: 0.5,
            beta: 2.0,
            ..Default::default()
        };
        let mut process = HawkesProcess::new(config).unwrap();
        
        let event = HawkesEvent {
            timestamp_us: 1_000_000,
            event_type: 0,
            volume: 1.0,
            price: 50000.0,
        };
        
        process.add_event(event).unwrap();
        assert!((process.intensity() - 1.5).abs() < 1e-10);
    }
}
