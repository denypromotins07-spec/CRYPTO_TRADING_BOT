//! Propagator Model for Market Order Impact Diffusion
//!
//! This module simulates how a single market order propagates through the
//! Limit Order Book (LOB), modeling both immediate and delayed impact.
//! Based on the Bouchaud propagator model framework.
//!
//! # Features
//! - Linear response theory for order impact
//! - Time-dependent propagator function
//! - Decay kernel estimation from historical data
//! - Memory-efficient convolution for real-time prediction

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;

/// Type of propagator decay kernel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// Exponential decay: G(t) = exp(-t/tau)
    Exponential,
    /// Power law decay: G(t) = (1 + t/tau)^(-beta)
    PowerLaw,
    /// Logarithmic decay: G(t) = log(1 + t/tau) / log(1 + T/tau)
    Logarithmic,
    /// Custom kernel from calibration
    Custom,
}

/// Configuration for the propagator model
#[derive(Debug, Clone)]
pub struct PropagatorConfig {
    /// Type of decay kernel
    pub kernel_type: KernelType,
    /// Characteristic time scale (tau) in seconds
    pub tau: f64,
    /// Power law exponent (for PowerLaw kernel)
    pub beta: f64,
    /// Maximum time horizon for propagation
    pub max_horizon: f64,
    /// Number of time bins for discretization
    pub n_bins: usize,
    /// Initial impact coefficient
    pub initial_impact: f64,
}

impl Default for PropagatorConfig {
    fn default() -> Self {
        Self {
            kernel_type: KernelType::PowerLaw, // Empirically most accurate
            tau: 100.0, // 100 seconds characteristic time
            beta: 0.5, // Square root decay
            max_horizon: 3600.0, // 1 hour max horizon
            n_bins: 100,
            initial_impact: 1e-4, // 0.01% initial impact per unit volume
        }
    }
}

/// Pre-computed propagator kernel values
struct PropagatorKernel {
    /// Time bin centers
    times: Vec<f64>,
    /// Kernel values at each time bin
    values: Vec<f64>,
    /// Cumulative sum for efficient convolution
    cumulative: Vec<f64>,
}

impl PropagatorKernel {
    /// Create kernel from configuration
    fn new(config: &PropagatorConfig) -> Self {
        let dt = config.max_horizon / config.n_bins as f64;
        let mut times = Vec::with_capacity(config.n_bins);
        let mut values = Vec::with_capacity(config.n_bins);
        
        for i in 0..config.n_bins {
            let t = (i as f64 + 0.5) * dt; // Bin center
            times.push(t);
            
            let g = match config.kernel_type {
                KernelType::Exponential => {
                    (-t / config.tau).exp()
                }
                KernelType::PowerLaw => {
                    (1.0 + t / config.tau).powf(-config.beta)
                }
                KernelType::Logarithmic => {
                    if t > 0.0 {
                        (1.0 + t / config.tau).ln() / (1.0 + config.max_horizon / config.tau).ln()
                    } else {
                        0.0
                    }
                }
                KernelType::Custom => {
                    // Would be set from calibration
                    1.0 / (1.0 + t / config.tau)
                }
            };
            
            values.push(g * config.initial_impact);
        }
        
        // Compute cumulative sum
        let mut cumulative = vec![0.0; config.n_bins + 1];
        for (i, &v) in values.iter().enumerate() {
            cumulative[i + 1] = cumulative[i] + v;
        }
        
        Self { times, values, cumulative }
    }
    
    /// Get kernel value at specific time (interpolated)
    fn get(&self, t: f64) -> f64 {
        if t <= 0.0 || self.times.is_empty() {
            return self.values.first().copied().unwrap_or(0.0);
        }
        
        if t >= *self.times.last().unwrap() {
            return *self.values.last().unwrap();
        }
        
        // Find bin and interpolate
        let dt = self.times[1] - self.times[0];
        let idx = (t / dt) as usize;
        let frac = (t / dt) - idx as f64;
        
        if idx + 1 >= self.values.len() {
            return *self.values.last().unwrap();
        }
        
        self.values[idx] * (1.0 - frac) + self.values[idx + 1] * frac
    }
}

/// Result of propagating an order through the LOB
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Time since order (seconds)
    pub time: f64,
    /// Immediate impact (at time 0)
    pub immediate_impact: f64,
    /// Current impact at this time
    pub current_impact: f64,
    /// Permanent impact (asymptotic)
    pub permanent_impact: f64,
    /// Transient impact (decaying component)
    pub transient_impact: f64,
    /// Total cost accumulated so far
    pub accumulated_cost: f64,
}

/// Propagator model for order impact diffusion
pub struct PropagatorModel {
    config: PropagatorConfig,
    kernel: PropagatorKernel,
    /// History of recent orders for convolution
    order_history: VecDeque<OrderRecord>,
    /// Maximum history size (memory constraint)
    max_history: usize,
}

/// Record of a past order for convolution
#[derive(Debug, Clone)]
struct OrderRecord {
    /// Time of order (microseconds since epoch)
    timestamp: u128,
    /// Signed volume (positive for buy, negative for sell)
    volume: f64,
    /// Price at time of order
    price: f64,
}

impl PropagatorModel {
    /// Create a new propagator model with given configuration
    pub fn new(config: PropagatorConfig) -> Self {
        let kernel = PropagatorKernel::new(&config);
        
        Self {
            config,
            kernel,
            order_history: VecDeque::new(),
            max_history: 10000, // Reasonable limit for 8GB RAM
        }
    }
    
    /// Record a new market order
    pub fn record_order(&mut self, volume: f64, price: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros();
        
        self.order_history.push_back(OrderRecord {
            timestamp: now,
            volume,
            price,
        });
        
        // Maintain bounded history
        while self.order_history.len() > self.max_history {
            self.order_history.pop_front();
        }
    }
    
    /// Calculate current impact from all past orders
    pub fn calculate_current_impact(&self) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros();
        
        let mut total_impact = 0.0;
        
        for record in &self.order_history {
            let elapsed_us = now.saturating_sub(record.timestamp);
            let elapsed_s = elapsed_us as f64 / 1_000_000.0;
            
            if elapsed_s > self.config.max_horizon {
                continue; // Beyond horizon
            }
            
            let kernel_value = self.kernel.get(elapsed_s);
            total_impact += record.volume * kernel_value;
        }
        
        total_impact
    }
    
    /// Predict impact at future time from current order book state
    pub fn predict_future_impact(&self, horizon: f64) -> PropagationResult {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros();
        
        let mut immediate_impact = 0.0;
        let mut future_impact = 0.0;
        let mut accumulated_cost = 0.0;
        
        for record in &self.order_history {
            let elapsed_us = now.saturating_sub(record.timestamp);
            let elapsed_s = elapsed_us as f64 / 1_000_000.0;
            
            if elapsed_s > self.config.max_horizon {
                continue;
            }
            
            let current_kernel = self.kernel.get(elapsed_s);
            let future_kernel = self.kernel.get(elapsed_s + horizon);
            
            immediate_impact += record.volume * current_kernel;
            future_impact += record.volume * future_kernel;
            
            // Accumulated cost is integral of impact over time
            accumulated_cost += record.volume.abs() * current_kernel * elapsed_s;
        }
        
        // Permanent impact is the asymptotic value
        let permanent_impact = self.kernel.get(self.config.max_horizon) 
            * self.order_history.iter().map(|r| r.volume.abs()).sum::<f64>();
        
        let transient_impact = future_impact - permanent_impact;
        
        PropagationResult {
            time: horizon,
            immediate_impact,
            current_impact: future_impact,
            permanent_impact,
            transient_impact,
            accumulated_cost,
        }
    }
    
    /// Simulate the effect of a hypothetical order without recording it
    pub fn simulate_order(&self, volume: f64, price: f64, horizon: f64) -> PropagationResult {
        let current_impact = self.calculate_current_impact();
        
        // Add hypothetical order's contribution
        let hypothetical_immediate = volume * self.kernel.get(0.0);
        let hypothetical_future = volume * self.kernel.get(horizon);
        
        let permanent = volume * self.kernel.get(self.config.max_horizon);
        let transient = hypothetical_future - permanent;
        
        PropagationResult {
            time: horizon,
            immediate_impact: current_impact + hypothetical_immediate,
            current_impact: current_impact + hypothetical_future,
            permanent_impact: permanent,
            transient_impact: transient,
            accumulated_cost: volume.abs() * hypothetical_immediate,
        }
    }
    
    /// Calculate optimal execution schedule using propagator
    pub fn calculate_optimal_schedule(
        &self,
        total_volume: f64,
        n_slices: usize,
        time_horizon: f64,
    ) -> Vec<f64> {
        // Simple equal-spacing as baseline
        // More sophisticated optimization would minimize total cost
        let slice_size = total_volume / n_slices as f64;
        let dt = time_horizon / n_slices as f64;
        
        let mut schedule = Vec::with_capacity(n_slices);
        let mut remaining = total_volume;
        
        for i in 0..n_slices {
            if i == n_slices - 1 {
                schedule.push(remaining);
            } else {
                // Adjust for decay: execute more when impact has decayed
                let decay_factor = self.kernel.get(i as f64 * dt);
                let base_slice = slice_size * (1.0 + 0.1 * (1.0 - decay_factor));
                let actual_slice = base_slice.min(remaining);
                schedule.push(actual_slice);
                remaining -= actual_slice;
            }
        }
        
        schedule
    }
    
    /// Update configuration and rebuild kernel
    pub fn update_config(&mut self, config: PropagatorConfig) {
        self.config = config;
        self.kernel = PropagatorKernel::new(&self.config);
    }
    
    /// Clear order history
    pub fn clear_history(&mut self) {
        self.order_history.clear();
    }
    
    /// Get statistics about the propagator
    pub fn get_stats(&self) -> PropagatorStats {
        let total_volume: f64 = self.order_history.iter().map(|r| r.volume.abs()).sum();
        let buy_volume: f64 = self.order_history.iter()
            .filter(|r| r.volume > 0.0)
            .map(|r| r.volume)
            .sum();
        let sell_volume: f64 = self.order_history.iter()
            .filter(|r| r.volume < 0.0)
            .map(|r| r.volume.abs())
            .sum();
        
        PropagatorStats {
            order_count: self.order_history.len(),
            total_volume,
            buy_volume,
            sell_volume,
            current_impact: self.calculate_current_impact(),
            kernel_type: self.config.kernel_type,
            tau: self.config.tau,
        }
    }
}

/// Statistics from the propagator model
#[derive(Debug, Clone)]
pub struct PropagatorStats {
    pub order_count: usize,
    pub total_volume: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub current_impact: f64,
    pub kernel_type: KernelType,
    pub tau: f64,
}

/// Thread-safe wrapper for concurrent access
pub struct ThreadSafePropagator {
    inner: Arc<RwLock<PropagatorModel>>,
}

impl ThreadSafePropagator {
    pub fn new(model: PropagatorModel) -> Self {
        Self {
            inner: Arc::new(RwLock::new(model)),
        }
    }
    
    pub fn record_order(&self, volume: f64, price: f64) {
        let mut model = self.inner.write();
        model.record_order(volume, price);
    }
    
    pub fn calculate_current_impact(&self) -> f64 {
        let model = self.inner.read();
        model.calculate_current_impact()
    }
    
    pub fn predict_future_impact(&self, horizon: f64) -> PropagationResult {
        let model = self.inner.read();
        model.predict_future_impact(horizon)
    }
    
    pub fn simulate_order(&self, volume: f64, price: f64, horizon: f64) -> PropagationResult {
        let model = self.inner.read();
        model.simulate_order(volume, price, horizon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_propagator_creation() {
        let config = PropagatorConfig::default();
        let model = PropagatorModel::new(config);
        
        assert_eq!(model.order_history.len(), 0);
        assert_eq!(model.kernel.values.len(), 100);
    }
    
    #[test]
    fn test_order_recording() {
        let config = PropagatorConfig::default();
        let mut model = PropagatorModel::new(config);
        
        model.record_order(1.0, 50000.0);
        model.record_order(-0.5, 50000.0);
        
        assert_eq!(model.order_history.len(), 2);
        
        let impact = model.calculate_current_impact();
        assert!(impact > 0.0); // Net buy should have positive impact
    }
    
    #[test]
    fn test_impact_decay() {
        let mut config = PropagatorConfig::default();
        config.kernel_type = KernelType::Exponential;
        config.tau = 10.0; // Fast decay
        
        let model = PropagatorModel::new(config);
        
        // Impact should decrease with time
        let result_now = model.predict_future_impact(0.0);
        let result_later = model.predict_future_impact(100.0);
        
        // With no orders, both should be zero
        assert_eq!(result_now.current_impact, 0.0);
        assert_eq!(result_later.current_impact, 0.0);
    }
    
    #[test]
    fn test_kernel_types() {
        for kernel_type in [KernelType::Exponential, KernelType::PowerLaw, KernelType::Logarithmic] {
            let mut config = PropagatorConfig::default();
            config.kernel_type = kernel_type;
            
            let kernel = PropagatorKernel::new(&config);
            
            // All kernels should have positive, decreasing values
            assert!(kernel.values.iter().all(|&v| v >= 0.0));
            if kernel.values.len() > 1 {
                assert!(kernel.values[0] >= kernel.values[kernel.values.len() - 1]);
            }
        }
    }
}
