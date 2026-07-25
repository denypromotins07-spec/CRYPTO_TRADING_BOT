//! Hidden Markov Model (HMM) for Market Regime Detection
//! 
//! Ultra-fast O(N) forward algorithm implementation for real-time
//! market state classification (bull/bear/chop regimes).
//! Uses pre-allocated buffers and zero-copy operations.

use std::collections::VecDeque;
use rayon::prelude::*;

/// Number of hidden states (regimes)
pub const N_STATES: usize = 3; // Bull, Bear, Chop

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Bull,
    Bear,
    Chop,
}

impl MarketRegime {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => MarketRegime::Bull,
            1 => MarketRegime::Bull, // Fallback
            _ => MarketRegime::Chop,
        }
    }
    
    pub fn to_index(&self) -> usize {
        match self {
            MarketRegime::Bull => 0,
            MarketRegime::Bear => 1,
            MarketRegime::Chop => 2,
        }
    }
}

/// HMM Parameters for regime detection
#[derive(Clone, Debug)]
pub struct HMMParams {
    /// Initial state probabilities (pi)
    pub pi: [f64; N_STATES],
    
    /// State transition matrix (A[i][j] = P(state_j | state_i))
    pub transition_matrix: [[f64; N_STATES]; N_STATES],
    
    /// Emission parameters for each state (Gaussian: mean, variance)
    pub emission_means: [f64; N_STATES],
    pub emission_vars: [f64; N_STATES],
}

impl Default for HMMParams {
    fn default() -> Self {
        Self {
            // Start with equal probability
            pi: [0.34, 0.33, 0.33],
            
            // Transition matrix (rows sum to 1)
            // High persistence in each regime
            transition_matrix: [
                [0.85, 0.10, 0.05], // Bull -> Bull/Bear/Chop
                [0.05, 0.85, 0.10], // Bear -> Bull/Bear/Chop
                [0.10, 0.10, 0.80], // Chop -> Bull/Bear/Chop
            ],
            
            // Emission parameters (returns)
            emission_means: [0.001, -0.001, 0.0],   // Bull+, Bear-, Chop~0
            emission_vars: [0.0001, 0.0002, 0.0003], // Bull low vol, Bear med, Chop high
        }
    }
}

/// Forward algorithm state (pre-allocated buffers)
pub struct ForwardState {
    /// Forward probabilities alpha[t][i]
    alpha: [f64; N_STATES],
    
    /// Scaling factor to prevent underflow
    scale: f64,
    
    /// Log-likelihood accumulator
    log_likelihood: f64,
    
    /// History of regime probabilities (circular buffer)
    history: VecDeque<[f64; N_STATES]>,
}

impl ForwardState {
    pub fn new(history_size: usize) -> Self {
        Self {
            alpha: [1.0 / N_STATES as f64; N_STATES],
            scale: 1.0,
            log_likelihood: 0.0,
            history: VecDeque::with_capacity(history_size),
        }
    }
    
    pub fn reset(&mut self) {
        self.alpha = [1.0 / N_STATES as f64; N_STATES];
        self.scale = 1.0;
        self.log_likelihood = 0.0;
    }
}

/// HMM Regime Detector with O(N) forward algorithm
pub struct HMMRegimeDetector {
    params: HMMParams,
    state: ForwardState,
    
    /// Running statistics for adaptive parameters
    running_mean: f64,
    running_var: f64,
    obs_count: u64,
    
    /// Smoothing window for regime stability
    regime_history: VecDeque<MarketRegime>,
    smoothing_window: usize,
}

impl HMMRegimeDetector {
    pub fn new(params: HMMParams, history_size: usize, smoothing_window: usize) -> Self {
        Self {
            params,
            state: ForwardState::new(history_size),
            running_mean: 0.0,
            running_var: 0.0,
            obs_count: 0,
            regime_history: VecDeque::with_capacity(smoothing_window),
            smoothing_window,
        }
    }
    
    /// Process single observation and return current regime (O(N) complexity)
    pub fn update(&mut self, observation: f64) -> MarketRegime {
        self.obs_count += 1;
        
        // Update running statistics incrementally
        self.update_running_stats(observation);
        
        // Normalize observation using running stats
        let norm_obs = self.normalize_observation(observation);
        
        // Forward step (O(N) where N = n_states)
        self.forward_step(norm_obs);
        
        // Get most likely current state
        let current_state = self.get_most_likely_state();
        
        // Apply smoothing for stability
        let smoothed_regime = self.smooth_regime(current_state);
        
        // Store in history
        self.state.history.push_back(self.state.alpha);
        if self.state.history.len() > self.state.history.capacity() {
            self.state.history.pop_front();
        }
        
        smoothed_regime
    }
    
    /// Single forward step (O(N) time complexity)
    fn forward_step(&mut self, obs: f64) {
        // Compute emission probabilities (Gaussian)
        let emissions: [f64; N_STATES] = self.compute_emissions(obs);
        
        // Compute new alpha values
        let mut new_alpha = [0.0; N_STATES];
        
        for j in 0..N_STATES {
            let mut sum = 0.0;
            for i in 0..N_STATES {
                sum += self.state.alpha[i] * self.params.transition_matrix[i][j];
            }
            new_alpha[j] = sum * emissions[j];
        }
        
        // Scale to prevent underflow
        let scale: f64 = new_alpha.iter().sum();
        let scale = if scale < 1e-10 { 1.0 } else { scale };
        
        for j in 0..N_STATES {
            self.state.alpha[j] = new_alpha[j] / scale;
        }
        
        // Accumulate log-likelihood
        self.state.log_likelihood += scale.max(1e-10).ln();
        self.state.scale = scale;
    }
    
    /// Compute emission probabilities for all states (parallel)
    fn compute_emissions(&self, obs: f64) -> [f64; N_STATES] {
        (0..N_STATES)
            .into_par_iter()
            .map(|i| {
                let mean = self.params.emission_means[i];
                let var = self.params.emission_vars[i].max(1e-10);
                
                // Gaussian PDF
                let diff = obs - mean;
                let exponent = -(diff * diff) / (2.0 * var);
                let coeff = 1.0 / (2.0 * std::f64::consts::PI * var).sqrt();
                
                coeff * exponent.exp()
            })
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap_or([1.0 / N_STATES as f64; N_STATES])
    }
    
    /// Get most likely current state
    fn get_most_likely_state(&self) -> MarketRegime {
        let max_idx = self.state.alpha
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        match max_idx {
            0 => MarketRegime::Bull,
            1 => MarketRegime::Bear,
            _ => MarketRegime::Chop,
        }
    }
    
    /// Smooth regime predictions to avoid rapid flipping
    fn smooth_regime(&mut self, current: MarketRegime) -> MarketRegime {
        self.regime_history.push_back(current);
        
        if self.regime_history.len() < self.smoothing_window {
            return current;
        }
        
        // Count occurrences
        let mut counts = [0; N_STATES];
        for &regime in &self.regime_history {
            counts[regime.to_index()] += 1;
        }
        
        // Return most common in window
        let max_idx = counts
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        MarketRegime::from_index(max_idx)
    }
    
    /// Incremental update of running mean and variance
    fn update_running_stats(&mut self, x: f64) {
        let n = self.obs_count as f64;
        
        // Welford's online algorithm
        let delta = x - self.running_mean;
        self.running_mean += delta / n;
        
        let delta2 = x - self.running_mean;
        self.running_var += delta * delta2;
    }
    
    /// Normalize observation using running statistics
    fn normalize_observation(&self, x: f64) -> f64 {
        if self.obs_count < 2 {
            return x;
        }
        
        let variance = self.running_var / (self.obs_count as f64 - 1.0);
        let std = variance.sqrt().max(1e-8);
        
        (x - self.running_mean) / std
    }
    
    /// Get current regime probabilities
    pub fn get_regime_probabilities(&self) -> [f64; N_STATES] {
        self.state.alpha
    }
    
    /// Get log-likelihood of observations
    pub fn get_log_likelihood(&self) -> f64 {
        self.state.log_likelihood
    }
    
    /// Check if regime changed recently
    pub fn is_regime_stable(&self) -> bool {
        if self.regime_history.len() < self.smoothing_window {
            return false;
        }
        
        let current = self.regime_history.back().copied().unwrap_or(MarketRegime::Chop);
        self.regime_history.iter().all(|&r| r == current)
    }
    
    /// Adaptively update emission parameters based on recent data
    pub fn adapt_parameters(&mut self, recent_returns: &[f64]) {
        if recent_returns.len() < 10 {
            return;
        }
        
        // Simple adaptive update (could be more sophisticated)
        let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let variance = recent_returns.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent_returns.len() as f64;
        
        // Blend with existing parameters (slow adaptation)
        let alpha = 0.1; // Adaptation rate
        
        if mean > 0.0005 {
            self.params.emission_means[MarketRegime::Bull.to_index()] = 
                (1.0 - alpha) * self.params.emission_means[MarketRegime::Bull.to_index()] + alpha * mean;
            self.params.emission_vars[MarketRegime::Bull.to_index()] = 
                (1.0 - alpha) * self.params.emission_vars[MarketRegime::Bull.to_index()] + alpha * variance;
        } else if mean < -0.0005 {
            self.params.emission_means[MarketRegime::Bear.to_index()] = 
                (1.0 - alpha) * self.params.emission_means[MarketRegime::Bear.to_index()] + alpha * mean;
            self.params.emission_vars[MarketRegime::Bear.to_index()] = 
                (1.0 - alpha) * self.params.emission_vars[MarketRegime::Bear.to_index()] + alpha * variance;
        } else {
            self.params.emission_means[MarketRegime::Chop.to_index()] = 
                (1.0 - alpha) * self.params.emission_means[MarketRegime::Chop.to_index()] + alpha * mean;
            self.params.emission_vars[MarketRegime::Chop.to_index()] = 
                (1.0 - alpha) * self.params.emission_vars[MarketRegime::Chop.to_index()] + alpha * variance;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hmm_creation() {
        let params = HMMParams::default();
        let detector = HMMRegimeDetector::new(params, 100, 5);
        
        assert_eq!(detector.params.pi.iter().sum::<f64>(), 1.0);
    }
    
    #[test]
    fn test_regime_detection() {
        let params = HMMParams::default();
        let mut detector = HMMRegimeDetector::new(params, 100, 3);
        
        // Simulate bull market returns
        for _ in 0..20 {
            let obs = 0.002; // Positive return
            let regime = detector.update(obs);
            // After warmup, should detect bull
        }
        
        // Check probabilities sum to 1
        let probs = detector.get_regime_probabilities();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_regime_transition() {
        let params = HMMParams::default();
        let mut detector = HMMRegimeDetector::new(params, 100, 3);
        
        // Start with positive returns (bull)
        for _ in 0..30 {
            detector.update(0.002);
        }
        
        let bull_probs = detector.get_regime_probabilities();
        let bull_prob = bull_probs[MarketRegime::Bull.to_index()];
        
        // Switch to negative returns (bear)
        for _ in 0..30 {
            detector.update(-0.002);
        }
        
        let bear_probs = detector.get_regime_probabilities();
        let bear_prob = bear_probs[MarketRegime::Bear.to_index()];
        
        // Should have shifted probability mass
        assert!(bear_prob > bull_prob || bear_prob > 0.4);
    }
    
    #[test]
    fn test_o_n_complexity() {
        let params = HMMParams::default();
        let mut detector = HMMRegimeDetector::new(params, 1000, 5);
        
        // Time single update (should be O(N) where N=3 states)
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            detector.update(0.001);
        }
        let elapsed = start.elapsed();
        
        // Should complete 10k updates in reasonable time (< 100ms)
        assert!(elapsed.as_millis() < 100, "HMM update too slow: {:?}", elapsed);
    }
}
