//! Synaptic Plasticity: Hebbian Learning for Online Weight Adaptation
//!
//! This module implements Hebbian learning rules for continuous adaptation
//! of LTC network weights during online trading, enabling the model to
//! adapt to changing market regimes without full retraining.
//!
//! Key Features:
//! - Oja's rule for stable Hebbian learning
//! - Spike-timing-dependent plasticity (STDP) inspired updates
//! - Memory-efficient weight updates without full gradient storage
//! - Bounded weight growth for numerical stability
//!
//! Mathematical Foundation:
//! Standard Hebbian: dW/dt = eta * pre * post
//! Oja's Rule: dW/dt = eta * (pre * post - post^2 * W)
//! Where eta is the learning rate, pre and post are neuron activations.

use std::time::{Duration, Instant};

/// Configuration for synaptic plasticity
#[derive(Debug, Clone)]
pub struct PlasticityConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub max_weight: f64,
    pub min_weight: f64,
    pub oja_normalization: bool,
    pub eligibility_trace_decay: f64,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,      // Conservative learning rate for stability
            weight_decay: 0.0001,      // L2 regularization
            max_weight: 5.0,           // Upper bound for weights
            min_weight: -5.0,          // Lower bound for weights
            oja_normalization: true,   // Use Oja's rule for stability
            eligibility_trace_decay: 0.9, // Trace decay for delayed rewards
        }
    }
}

/// Synaptic plasticity manager for online weight adaptation
pub struct SynapticPlasticity {
    config: PlasticityConfig,
    /// Weight matrices (flattened)
    weights: Vec<f64>,
    /// Weight dimensions
    num_pre: usize,
    num_post: usize,
    /// Eligibility traces for delayed credit assignment
    eligibility_traces: Vec<f64>,
    /// Pre-synaptic activity history
    pre_activity: Vec<f64>,
    /// Post-synaptic activity history
    post_activity: Vec<f64>,
    /// Running average of post-synaptic activity (for Oja's rule)
    post_avg: Vec<f64>,
    /// Total weight updates performed
    update_count: u64,
}

impl SynapticPlasticity {
    /// Create a new synaptic plasticity manager
    pub fn new(num_pre: usize, num_post: usize, config: PlasticityConfig) -> Self {
        let total_weights = num_pre * num_post;
        
        Self {
            config,
            weights: vec![0.0; total_weights],
            num_pre,
            num_post,
            eligibility_traces: vec![0.0; total_weights],
            pre_activity: vec![0.0; num_pre],
            post_activity: vec![0.0; num_post],
            post_avg: vec![0.0; num_post],
            update_count: 0,
        }
    }
    
    /// Initialize weights with small random values
    pub fn initialize_weights(&mut self, seed: u64) {
        let mut rng = seed;
        let lcg_next = |x: u64| x.wrapping_mul(6364136223846793005).wrapping_add(1);
        
        for i in 0..self.weights.len() {
            rng = lcg_next(rng);
            // Xavier-like initialization scaled for Hebbian learning
            let scale = 1.0 / ((self.num_pre + self.num_post) as f64).sqrt();
            self.weights[i] = ((rng % 1000) as f64 / 1000.0 - 0.5) * 2.0 * scale;
        }
    }
    
    /// Record pre-synaptic activity
    #[inline]
    pub fn record_pre_activity(&mut self, activity: &[f64]) {
        if activity.len() != self.num_pre {
            return; // Dimension mismatch, skip
        }
        self.pre_activity.copy_from_slice(activity);
    }
    
    /// Record post-synaptic activity and apply Hebbian update
    #[inline]
    pub fn record_post_activity(&mut self, activity: &[f64]) {
        if activity.len() != self.num_post {
            return; // Dimension mismatch, skip
        }
        
        // Update running average for Oja's normalization
        for i in 0..self.num_post {
            self.post_avg[i] = 0.99 * self.post_avg[i] + 0.01 * activity[i].powi(2);
        }
        
        // Apply Hebbian learning rule
        self.apply_hebbian_update(activity);
        
        // Store current activity for next step
        self.post_activity.copy_from_slice(activity);
        
        self.update_count += 1;
    }
    
    /// Apply Hebbian weight update
    #[inline]
    fn apply_hebbian_update(&mut self, post_activity: &[f64]) {
        let eta = self.config.learning_rate;
        let decay = self.config.weight_decay;
        
        for j in 0..self.num_post {
            let post = post_activity[j];
            let post_sq = post * post;
            
            for i in 0..self.num_pre {
                let pre = self.pre_activity[i];
                let idx = j * self.num_pre + i;
                
                // Eligibility trace update
                self.eligibility_traces[idx] = 
                    self.config.eligibility_trace_decay * self.eligibility_traces[idx] 
                    + pre * post;
                
                // Hebbian update with Oja's normalization
                let mut delta = eta * self.eligibility_traces[idx];
                
                if self.config.oja_normalization {
                    // Oja's rule: subtract post^2 * W term
                    delta -= eta * post_sq * self.weights[idx];
                }
                
                // Add weight decay (L2 regularization)
                delta -= decay * self.weights[idx];
                
                // Apply update
                self.weights[idx] += delta;
                
                // Clip weights to prevent explosion
                self.weights[idx] = self.weights[idx].clamp(
                    self.config.min_weight,
                    self.config.max_weight,
                );
            }
        }
    }
    
    /// Apply reward-modulated Hebbian update (for reinforcement learning)
    #[inline]
    pub fn apply_reward_modulated_update(&mut self, reward: f64) {
        if reward.abs() < 1e-10 {
            return; // No reward signal, skip
        }
        
        let eta = self.config.learning_rate * reward.clamp(-1.0, 1.0);
        
        for idx in 0..self.weights.len() {
            // Reward scales the eligibility trace contribution
            let delta = eta * self.eligibility_traces[idx];
            
            self.weights[idx] += delta;
            self.weights[idx] = self.weights[idx].clamp(
                self.config.min_weight,
                self.config.max_weight,
            );
            
            // Decay eligibility trace after use
            self.eligibility_traces[idx] *= 0.5;
        }
    }
    
    /// Get weight at specific indices
    #[inline]
    pub fn get_weight(&self, pre_idx: usize, post_idx: usize) -> Option<f64> {
        if pre_idx >= self.num_pre || post_idx >= self.num_post {
            return None;
        }
        Some(self.weights[post_idx * self.num_pre + pre_idx])
    }
    
    /// Set weight at specific indices
    #[inline]
    pub fn set_weight(&mut self, pre_idx: usize, post_idx: usize, value: f64) -> bool {
        if pre_idx >= self.num_pre || post_idx >= self.num_post {
            return false;
        }
        let clamped = value.clamp(self.config.min_weight, self.config.max_weight);
        self.weights[post_idx * self.num_pre + pre_idx] = clamped;
        true
    }
    
    /// Get all weights as a slice
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }
    
    /// Get weight statistics for monitoring
    pub fn get_weight_statistics(&self) -> WeightStats {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        
        for &w in &self.weights {
            sum += w;
            sum_sq += w * w;
            min_val = min_val.min(w);
            max_val = max_val.max(w);
        }
        
        let n = self.weights.len() as f64;
        let mean = sum / n;
        let variance = sum_sq / n - mean * mean;
        let std_dev = variance.sqrt();
        
        WeightStats {
            mean,
            std_dev,
            min: min_val,
            max: max_val,
            l2_norm: sum_sq.sqrt(),
        }
    }
    
    /// Reset plasticity state (keep weights)
    pub fn reset_state(&mut self) {
        self.eligibility_traces.fill(0.0);
        self.pre_activity.fill(0.0);
        self.post_activity.fill(0.0);
        self.post_avg.fill(0.0);
    }
    
    /// Get total number of updates performed
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
    
    /// Estimate memory usage in bytes
    pub fn estimate_memory_usage(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let weight_size = self.weights.len() * 8; // f64
        base_size + 5 * weight_size // weights, traces, pre, post, post_avg
    }
}

/// Statistics about weight distribution
#[derive(Debug, Clone)]
pub struct WeightStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub l2_norm: f64,
}

/// STDP-inspired plasticity for spike-based learning
pub struct StdpPlasticity {
    config: PlasticityConfig,
    /// Pre-synaptic spike times
    pre_spikes: Vec<u64>,
    /// Post-synaptic spike times
    post_spikes: Vec<u64>,
    /// Current time step
    current_time: u64,
    /// Time window for STDP (microseconds)
    time_window: u64,
    /// Weights
    weights: Vec<f64>,
    num_pre: usize,
    num_post: usize,
}

impl StdpPlasticity {
    pub fn new(num_pre: usize, num_post: usize, config: PlasticityConfig) -> Self {
        Self {
            config,
            pre_spikes: vec![0; num_pre],
            post_spikes: vec![0; num_post],
            current_time: 0,
            time_window: 1000, // 1ms window
            weights: vec![0.0; num_pre * num_post],
            num_pre,
            num_post,
        }
    }
    
    /// Record a pre-synaptic spike
    #[inline]
    pub fn pre_spike(&mut self, neuron_idx: usize) {
        if neuron_idx < self.num_pre {
            self.pre_spikes[neuron_idx] = self.current_time;
        }
    }
    
    /// Record a post-synaptic spike and apply STDP update
    #[inline]
    pub fn post_spike(&mut self, neuron_idx: usize) {
        if neuron_idx >= self.num_post {
            return;
        }
        
        let post_time = self.current_time;
        
        // Apply STDP for all pre-synaptic neurons
        for i in 0..self.num_pre {
            let pre_time = self.pre_spikes[i];
            let delta_t = post_time as i64 - pre_time as i64;
            
            if delta_t.abs() as u64 > self.time_window {
                continue; // Outside STDP window
            }
            
            // STDP curve: potentiation for pre-before-post, depression otherwise
            let delta_w = if delta_t > 0 {
                // LTP: pre before post
                self.config.learning_rate * (-delta_t as f64 / 200.0).exp()
            } else {
                // LTD: post before pre
                -self.config.learning_rate * (delta_t as f64 / 200.0).exp()
            };
            
            let idx = neuron_idx * self.num_pre + i;
            self.weights[idx] += delta_w;
            self.weights[idx] = self.weights[idx].clamp(
                self.config.min_weight,
                self.config.max_weight,
            );
        }
        
        self.post_spikes[neuron_idx] = post_time;
    }
    
    /// Advance simulation time
    #[inline]
    pub fn advance_time(&mut self, delta: u64) {
        self.current_time += delta;
    }
    
    /// Get weights
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hebbian_initialization() {
        let config = PlasticityConfig::default();
        let mut plasticity = SynapticPlasticity::new(16, 32, config);
        plasticity.initialize_weights(42);
        
        let stats = plasticity.get_weight_statistics();
        assert!(stats.std_dev > 0.0);
        assert!(stats.min >= -5.0);
        assert!(stats.max <= 5.0);
    }

    #[test]
    fn test_hebbian_update() {
        let config = PlasticityConfig::default();
        let mut plasticity = SynapticPlasticity::new(4, 8, config);
        plasticity.initialize_weights(42);
        
        let initial_weights = plasticity.get_weights().to_vec();
        
        // Simulate activity
        for _ in 0..100 {
            let pre = vec![0.1, 0.2, 0.3, 0.4];
            let post = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8];
            
            plasticity.record_pre_activity(&pre);
            plasticity.record_post_activity(&post);
        }
        
        // Weights should have changed
        let final_weights = plasticity.get_weights();
        let mut changed = false;
        for (i, &w) in final_weights.iter().enumerate() {
            if (w - initial_weights[i]).abs() > 1e-10 {
                changed = true;
                break;
            }
        }
        assert!(changed, "Weights should change after Hebbian updates");
    }

    #[test]
    fn test_weight_bounding() {
        let mut config = PlasticityConfig::default();
        config.max_weight = 1.0;
        config.min_weight = -1.0;
        
        let mut plasticity = SynapticPlasticity::new(4, 8, config);
        plasticity.initialize_weights(42);
        
        // Force large updates
        for _ in 0..1000 {
            let pre = vec![1.0, 1.0, 1.0, 1.0];
            let post = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            
            plasticity.record_pre_activity(&pre);
            plasticity.record_post_activity(&post);
        }
        
        // All weights should be bounded
        for &w in plasticity.get_weights() {
            assert!(w >= -1.0 && w <= 1.0, "Weight {} out of bounds", w);
        }
    }
}
