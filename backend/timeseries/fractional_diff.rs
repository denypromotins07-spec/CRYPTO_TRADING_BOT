//! Fractional Differencing Module for Long-Term Memory Preservation
//! 
//! This module implements fractional differencing to transform non-stationary crypto
//! price series into stationary series while preserving long-term memory (Hurst exponent).
//! Unlike integer differencing which destroys memory, fractional differencing with d ∈ (0,1)
//! maintains the predictive structure crucial for alpha generation.
//! 
//! Optimized for O(1) incremental updates using circular buffers and pre-computed binomial weights.
//! Zero-cost abstractions ensure no heap allocations during real-time processing.

use std::collections::VecDeque;
use std::sync::Arc;
use atomic_refcell::AtomicRefCell;

/// Maximum history window for fractional differencing (covers ~1 year of minute bars)
const MAX_HISTORY: usize = 525600;

/// Pre-computed binomial coefficients for fractional differencing
/// w_k = (-1)^k * Γ(d+1) / (Γ(k+1) * Γ(d-k+1))
#[derive(Clone)]
pub struct FractionalWeights {
    weights: Vec<f64>,
    d: f64,
    max_lag: usize,
}

impl FractionalWeights {
    /// Create new fractional weights with differencing parameter d
    /// d ∈ (0, 1): typical values 0.3-0.7 for crypto assets
    pub fn new(d: f64, max_lag: usize) -> Result<Self, &'static str> {
        if d <= 0.0 || d >= 1.0 {
            return Err("Differencing parameter d must be in (0, 1)");
        }
        if max_lag == 0 || max_lag > MAX_HISTORY {
            return Err("Max lag must be between 1 and MAX_HISTORY");
        }

        let mut weights = Vec::with_capacity(max_lag);
        
        // w_0 = 1
        weights.push(1.0);
        
        // Recursively compute weights: w_k = w_{k-1} * (k - 1 - d) / k
        for k in 1..max_lag {
            let prev = *weights.last().unwrap();
            let weight = prev * (k as f64 - 1.0 - d) / k as f64;
            weights.push(weight);
        }

        Ok(Self { weights, d, max_lag })
    }

    /// Get weight at index k
    #[inline]
    pub fn get(&self, k: usize) -> f64 {
        if k < self.weights.len() {
            self.weights[k]
        } else {
            0.0
        }
    }

    /// Get the differencing parameter
    pub fn d(&self) -> f64 {
        self.d
    }
}

/// Circular buffer for efficient sliding window operations
/// Pre-allocated to avoid heap allocations during runtime
struct CircularBuffer {
    data: Vec<f64>,
    head: usize,
    size: usize,
    capacity: usize,
}

impl CircularBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            head: 0,
            size: 0,
            capacity,
        }
    }

    #[inline]
    fn push(&mut self, value: f64) {
        let idx = (self.head + self.size) % self.capacity;
        if self.size < self.capacity {
            self.data[idx] = value;
            self.size += 1;
        } else {
            // Overwrite oldest value
            self.data[idx] = value;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    #[inline]
    fn get(&self, index: usize) -> Option<f64> {
        if index >= self.size {
            return None;
        }
        let idx = (self.head + self.size - 1 - index) % self.capacity;
        Some(self.data[idx])
    }

    fn len(&self) -> usize {
        self.size
    }

    fn is_full(&self) -> bool {
        self.size == self.capacity
    }
}

/// Main fractional differencing engine
/// Uses Observer pattern to notify downstream consumers of new differentiated values
pub struct FractionalDifferencer {
    weights: Arc<AtomicRefCell<FractionalWeights>>,
    buffer: CircularBuffer,
    last_diff_value: f64,
    observation_count: u64,
    subscribers: Vec<Box<dyn DifferencingObserver>>,
}

/// Observer trait for downstream consumers
pub trait DifferencingObserver: Send + Sync {
    fn on_new_differentiated_value(&self, timestamp: u64, original: f64, diffed: f64);
}

impl FractionalDifferencer {
    /// Create a new differencer with specified d parameter and history window
    pub fn new(d: f64, max_lag: usize) -> Result<Self, &'static str> {
        let weights = FractionalWeights::new(d, max_lag)?;
        let buffer = CircularBuffer::new(max_lag);
        
        Ok(Self {
            weights: Arc::new(AtomicRefCell::new(weights)),
            buffer,
            last_diff_value: 0.0,
            observation_count: 0,
            subscribers: Vec::new(),
        })
    }

    /// Subscribe to differentiated value updates
    pub fn subscribe(&mut self, observer: Box<dyn DifferencingObserver>) {
        self.subscribers.push(observer);
    }

    /// Process a new price tick and return the fractionally differenced value
    /// O(max_lag) computation, optimized with early termination for negligible weights
    #[inline]
    pub fn update(&mut self, price: f64) -> f64 {
        // Store log price for numerical stability
        let log_price = price.ln();
        self.buffer.push(log_price);

        self.observation_count += 1;

        // Need full window before computing reliable differenced value
        if self.buffer.len() < self.weights.borrow().max_lag.min(100) {
            return log_price; // Return raw log price until buffer fills
        }

        // Compute fractional difference: sum(w_k * log_price_{t-k})
        let weights_ref = self.weights.borrow();
        let mut diff_value = 0.0;
        
        // Early termination threshold (weights decay rapidly)
        let epsilon = 1e-10;
        
        for k in 0..self.buffer.len().min(weights_ref.max_lag) {
            if let Some(past_price) = self.buffer.get(k) {
                let weight = weights_ref.get(k);
                diff_value += weight * past_price;
                
                // Early termination if weight becomes negligible
                if weight.abs() < epsilon && k > 50 {
                    break;
                }
            }
        }

        self.last_diff_value = diff_value;

        // Notify observers
        let timestamp = self.observation_count;
        for subscriber in &self.subscribers {
            subscriber.on_new_differentiated_value(timestamp, price, diff_value);
        }

        diff_value
    }

    /// Update the differencing parameter dynamically (for regime adaptation)
    pub fn update_d(&mut self, new_d: f64) -> Result<(), &'static str> {
        let new_weights = FractionalWeights::new(new_d, self.weights.borrow().max_lag)?;
        *self.weights.borrow_mut() = new_weights;
        Ok(())
    }

    /// Get current differencing parameter
    pub fn current_d(&self) -> f64 {
        self.weights.borrow().d
    }

    /// Get the last computed differenced value
    pub fn last_value(&self) -> f64 {
        self.last_diff_value
    }

    /// Get observation count
    pub fn count(&self) -> u64 {
        self.observation_count
    }

    /// Reset the differencer (clear buffer, keep weights)
    pub fn reset(&mut self) {
        let capacity = self.buffer.capacity;
        self.buffer = CircularBuffer::new(capacity);
        self.last_diff_value = 0.0;
        self.observation_count = 0;
    }
}

/// Strategy pattern for adaptive differencing based on market regime
pub trait DifferencingStrategy: Send + Sync {
    /// Determine optimal d parameter for current market conditions
    fn compute_optimal_d(&self, hurst_exponent: f64, volatility: f64) -> f64;
}

/// Default strategy: higher d for more persistent (high Hurst) series
pub struct DefaultDifferencingStrategy;

impl DifferencingStrategy for DefaultDifferencingStrategy {
    fn compute_optimal_d(&self, hurst_exponent: f64, _volatility: f64) -> f64 {
        // Optimal d ≈ H - 0.5 for fractional Brownian motion
        // Clamp to valid range (0, 1)
        let d = hurst_exponent - 0.5;
        d.clamp(0.1, 0.9)
    }
}

/// Adaptive differencer that adjusts d based on market regime
pub struct AdaptiveFractionalDifferencer {
    base_differencer: FractionalDifferencer,
    strategy: Box<dyn DifferencingStrategy>,
    current_hurst: f64,
    current_volatility: f64,
    last_update_time: u64,
    update_interval: u64,
}

impl AdaptiveFractionalDifferencer {
    pub fn new(
        initial_d: f64,
        max_lag: usize,
        strategy: Box<dyn DifferencingStrategy>,
        update_interval: u64,
    ) -> Result<Self, &'static str> {
        let base_differencer = FractionalDifferencer::new(initial_d, max_lag)?;
        
        Ok(Self {
            base_differencer,
            strategy,
            current_hurst: 0.5, // Default to random walk
            current_volatility: 0.0,
            last_update_time: 0,
            update_interval,
        })
    }

    /// Update market regime estimates and potentially adjust d
    pub fn update_regime(&mut self, hurst: f64, volatility: f64, current_time: u64) {
        self.current_hurst = hurst;
        self.current_volatility = volatility;

        // Check if it's time to recompute optimal d
        if current_time - self.last_update_time >= self.update_interval {
            let new_d = self.strategy.compute_optimal_d(hurst, volatility);
            let current_d = self.base_differencer.current_d();
            
            // Only update if change is significant (avoid thrashing)
            if (new_d - current_d).abs() > 0.05 {
                if let Ok(_) = self.base_differencer.update_d(new_d) {
                    self.last_update_time = current_time;
                }
            }
        }
    }

    /// Process a new price tick
    pub fn update(&mut self, price: f64) -> f64 {
        self.base_differencer.update(price)
    }

    /// Subscribe to updates
    pub fn subscribe(&mut self, observer: Box<dyn DifferencingObserver>) {
        self.base_differencer.subscribe(observer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractional_weights_creation() {
        let weights = FractionalWeights::new(0.5, 100).unwrap();
        assert_eq!(weights.get(0), 1.0);
        assert!(weights.get(1) < 0.0); // Second weight should be negative for d=0.5
    }

    #[test]
    fn test_invalid_d_parameter() {
        assert!(FractionalWeights::new(0.0, 100).is_err());
        assert!(FractionalWeights::new(1.0, 100).is_err());
        assert!(FractionalWeights::new(-0.5, 100).is_err());
    }

    #[test]
    fn test_differencer_basic() {
        let mut differencer = FractionalDifferencer::new(0.5, 100).unwrap();
        
        // Feed some prices
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0];
        for price in prices {
            let _ = differencer.update(price);
        }
        
        assert!(differencer.count() == 5);
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(5);
        
        for i in 0..10 {
            buffer.push(i as f64);
        }
        
        // Buffer should contain last 5 elements: 5,6,7,8,9
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.get(0), Some(9.0)); // Most recent
        assert_eq!(buffer.get(4), Some(5.0)); // Oldest in buffer
    }
}
