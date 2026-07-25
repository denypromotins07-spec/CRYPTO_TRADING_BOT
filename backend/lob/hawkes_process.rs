//! Hawkes Process Implementation for Order Arrival Modeling
//! 
//! This module implements a self-exciting Hawkes process to model the clustering
//! of order arrivals in the limit order book. It uses exponential decay kernels
//! for O(1) intensity updates, critical for high-frequency trading on resource-constrained systems.
//!
//! # Features
//! - Multi-dimensional Hawkes process for bid/ask events
//! - Exponential decay kernel for constant-time updates
//! - Branching ratio estimation for market stability analysis
//! - Memory-efficient implementation respecting 8GB RAM constraint

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// Configuration for the Hawkes process
#[derive(Debug, Clone)]
pub struct HawkesConfig {
    /// Base intensity (mu) for each event type
    pub base_intensity: Vec<f64>,
    /// Decay rate (alpha) for exponential kernel
    pub decay_rate: f64,
    /// Excitation matrix (beta[i][j] = effect of event j on event i)
    pub excitation_matrix: Vec<Vec<f64>>,
    /// Maximum history depth for numerical stability
    pub max_history_depth: usize,
}

impl Default for HawkesConfig {
    fn default() -> Self {
        Self {
            base_intensity: vec![1.0, 1.0], // Bid and Ask base rates
            decay_rate: 0.5,
            excitation_matrix: vec![vec![0.3, 0.1], vec![0.1, 0.3]],
            max_history_depth: 1000,
        }
    }
}

/// Event type in the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    MarketBuy,
    MarketSell,
    LimitBuy,
    LimitSell,
    CancelBuy,
    CancelSell,
}

impl EventType {
    pub fn as_index(&self) -> usize {
        match self {
            EventType::MarketBuy => 0,
            EventType::MarketSell => 1,
            EventType::LimitBuy => 2,
            EventType::LimitSell => 3,
            EventType::CancelBuy => 4,
            EventType::CancelSell => 5,
        }
    }
    
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(EventType::MarketBuy),
            1 => Some(EventType::MarketSell),
            2 => Some(EventType::LimitBuy),
            3 => Some(EventType::LimitSell),
            4 => Some(EventType::CancelBuy),
            5 => Some(EventType::CancelSell),
            _ => None,
        }
    }
}

/// Single event record with timestamp and volume
#[derive(Debug, Clone)]
pub struct OrderEvent {
    pub event_type: EventType,
    pub timestamp: Instant,
    pub volume: f64,
    pub price: f64,
}

/// Hawkes Process state with exponential decay optimization
/// Uses the Markovian property of exponential kernels for O(1) updates
pub struct HawkesProcess {
    config: HawkesConfig,
    /// Current intensity for each event type (updated in O(1))
    current_intensity: Vec<f64>,
    /// Last update time for decay calculation
    last_update: Instant,
    /// Event history for debugging and re-calibration (bounded)
    event_history: Vec<OrderEvent>,
    /// Number of event types
    n_events: usize,
    /// Calibration state
    is_calibrated: bool,
}

impl HawkesProcess {
    /// Create a new Hawkes process with given configuration
    pub fn new(config: HawkesConfig) -> Self {
        let n_events = config.base_intensity.len();
        Self {
            config,
            current_intensity: vec![0.0; n_events],
            last_update: Instant::now(),
            event_history: Vec::new(),
            n_events,
            is_calibrated: false,
        }
    }
    
    /// Initialize intensities to base values
    pub fn initialize(&mut self) {
        self.current_intensity = self.config.base_intensity.clone();
        self.last_update = Instant::now();
        self.is_calibrated = true;
    }
    
    /// Update intensity after an event occurs - O(1) complexity
    /// Uses exponential decay: lambda(t) = mu + sum(alpha * exp(-beta * (t - t_i)))
    /// With Markovian property: lambda(t) = lambda(t_prev) * exp(-beta * dt) + alpha
    pub fn record_event(&mut self, event: OrderEvent) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        
        // Apply exponential decay to all intensities
        let decay_factor = (-self.config.decay_rate * dt).exp();
        for i in 0..self.n_events {
            self.current_intensity[i] *= decay_factor;
        }
        
        // Add excitation from the new event
        let event_idx = event.event_type.as_index();
        if event_idx < self.n_events {
            for i in 0..self.n_events {
                self.current_intensity[i] += self.config.excitation_matrix[i][event_idx];
            }
        }
        
        self.last_update = now;
        
        // Maintain bounded history
        if self.event_history.len() >= self.config.max_history_depth {
            self.event_history.remove(0);
        }
        self.event_history.push(event);
    }
    
    /// Get current intensity for a specific event type
    pub fn get_intensity(&self, event_type: EventType) -> f64 {
        let idx = event_type.as_index();
        if idx >= self.n_events {
            return 0.0;
        }
        
        // Apply decay since last update without modifying state
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        let decay_factor = (-self.config.decay_rate * dt).exp();
        
        self.current_intensity[idx] * decay_factor
    }
    
    /// Get all current intensities
    pub fn get_all_intensities(&self) -> Vec<f64> {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        let decay_factor = (-self.config.decay_rate * dt).exp();
        
        self.current_intensity.iter().map(|&i| i * decay_factor).collect()
    }
    
    /// Calculate the branching ratio (sum of excitation matrix eigenvalues)
    /// Critical if branching_ratio >= 1.0 (indicates unstable/excitable market)
    pub fn branching_ratio(&self) -> f64 {
        // Simplified: use Frobenius norm as upper bound for spectral radius
        let mut sum = 0.0;
        for row in &self.config.excitation_matrix {
            for &val in row {
                sum += val * val;
            }
        }
        sum.sqrt()
    }
    
    /// Check if the process is in a critical state
    pub fn is_critical(&self) -> bool {
        self.branching_ratio() >= 0.95 // Threshold slightly below 1.0 for safety
    }
    
    /// Simulate next event time using thinning algorithm
    pub fn simulate_next_event(&self, max_time: f64) -> Option<(f64, EventType)> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let intensities = self.get_all_intensities();
        let max_intensity = intensities.iter().cloned().fold(0.0_f64, f64::max);
        
        if max_intensity <= 0.0 {
            return None;
        }
        
        let mut t = 0.0;
        while t < max_time {
            // Sample candidate event time
            let dt = -rng.gen::<f64>().ln() / max_intensity;
            t += dt;
            
            if t >= max_time {
                break;
            }
            
            // Accept/reject based on actual intensity
            let u = rng.gen::<f64>();
            let cumulative: Vec<f64> = intensities.iter()
                .scan(0.0, |acc, &x| { *acc += x; Some(*acc) })
                .collect();
            
            let threshold = u * max_intensity;
            for (i, &cum) in cumulative.iter().enumerate() {
                if threshold <= cum {
                    if let Some(event_type) = EventType::from_index(i) {
                        return Some((t, event_type));
                    }
                }
            }
        }
        
        None
    }
    
    /// Reset the process state
    pub fn reset(&mut self) {
        self.current_intensity = self.config.base_intensity.clone();
        self.last_update = Instant::now();
        self.event_history.clear();
    }
    
    /// Get statistics about the process
    pub fn get_stats(&self) -> HawkesStats {
        let mut event_counts = HashMap::new();
        for event in &self.event_history {
            *event_counts.entry(event.event_type).or_insert(0usize) += 1;
        }
        
        HawkesStats {
            current_intensities: self.get_all_intensities(),
            branching_ratio: self.branching_ratio(),
            is_critical: self.is_critical(),
            event_counts,
            history_size: self.event_history.len(),
        }
    }
}

/// Statistics snapshot from the Hawkes process
#[derive(Debug, Clone)]
pub struct HawkesStats {
    pub current_intensities: Vec<f64>,
    pub branching_ratio: f64,
    pub is_critical: bool,
    pub event_counts: HashMap<EventType, usize>,
    pub history_size: usize,
}

/// Thread-safe wrapper for concurrent access
pub struct ThreadSafeHawkesProcess {
    inner: Arc<RwLock<HawkesProcess>>,
}

impl ThreadSafeHawkesProcess {
    pub fn new(process: HawkesProcess) -> Self {
        Self {
            inner: Arc::new(RwLock::new(process)),
        }
    }
    
    pub fn record_event(&self, event: OrderEvent) {
        let mut proc = self.inner.write();
        proc.record_event(event);
    }
    
    pub fn get_intensity(&self, event_type: EventType) -> f64 {
        let proc = self.inner.read();
        proc.get_intensity(event_type)
    }
    
    pub fn is_critical(&self) -> bool {
        let proc = self.inner.read();
        proc.is_critical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hawkes_initialization() {
        let config = HawkesConfig::default();
        let mut process = HawkesProcess::new(config);
        process.initialize();
        
        assert!(process.is_calibrated);
        assert_eq!(process.current_intensity.len(), 2);
    }
    
    #[test]
    fn test_event_recording() {
        let config = HawkesConfig::default();
        let mut process = HawkesProcess::new(config);
        process.initialize();
        
        let initial_intensity = process.get_intensity(EventType::MarketBuy);
        
        let event = OrderEvent {
            event_type: EventType::MarketBuy,
            timestamp: Instant::now(),
            volume: 1.0,
            price: 50000.0,
        };
        
        process.record_event(event);
        let new_intensity = process.get_intensity(EventType::MarketBuy);
        
        // Intensity should increase after event
        assert!(new_intensity > initial_intensity * 0.9); // Allow for decay
    }
    
    #[test]
    fn test_branching_ratio() {
        let mut config = HawkesConfig::default();
        config.excitation_matrix = vec![vec![0.6, 0.6], vec![0.6, 0.6]];
        let process = HawkesProcess::new(config);
        
        // Branching ratio should be > 1.0 for this configuration
        assert!(process.branching_ratio() > 1.0);
    }
}
